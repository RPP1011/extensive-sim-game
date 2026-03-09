//! Actor-critic RL episode generation using the ability transformer.
//!
//! Runs scenarios with the transformer making ALL hero decisions (not just abilities).
//! Records episodes as JSONL for PPO training in Python.
//!
//! Supports two weight formats:
//! - Actor-critic JSON (from `export_actor_critic.py`): full 14-action policy
//! - Legacy transformer JSON (from `export_weights.py`): urgency-based ability logits,
//!   uniform base action logits (bootstrap mode)

use std::io::Write;
use std::process::ExitCode;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::collect_toml_paths;

// ---------------------------------------------------------------------------
// Episode types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlEpisode {
    pub scenario: String,
    pub outcome: String,
    pub reward: f32,
    pub ticks: u64,
    /// Per-unit ability token IDs (unit_id → list of token ID lists).
    pub unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>>,
    pub steps: Vec<RlStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlStep {
    pub tick: u64,
    pub unit_id: u32,
    pub game_state: Vec<f32>,
    pub action: usize,
    pub log_prob: f32,
    pub mask: Vec<bool>,
    pub step_reward: f32,
    // V2 game state: variable-length entities + threats
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entities: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_types: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threats: Option<Vec<Vec<f32>>>,
    // V3 pointer action space: position tokens + hierarchical action
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_type: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_idx: Option<usize>,
}

// ---------------------------------------------------------------------------
// LCG + softmax
// ---------------------------------------------------------------------------

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

fn masked_softmax_sample(
    logits: &[f32],
    mask: &[bool],
    temperature: f32,
    rng: &mut u64,
) -> (usize, f32) {
    let n = logits.len();
    let temp = temperature.max(0.01);

    let mut max_val = f32::NEG_INFINITY;
    for i in 0..n {
        if mask[i] {
            let scaled = logits[i] / temp;
            if scaled > max_val { max_val = scaled; }
        }
    }

    let mut probs = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for i in 0..n {
        if mask[i] {
            let e = ((logits[i] / temp) - max_val).exp();
            probs[i] = e;
            sum += e;
        }
    }

    if sum > 0.0 {
        for p in &mut probs { *p /= sum; }
    } else {
        let valid = mask.iter().filter(|&&m| m).count() as f32;
        for (i, p) in probs.iter_mut().enumerate() {
            *p = if mask[i] { 1.0 / valid } else { 0.0 };
        }
    }

    let r = lcg_f32(rng);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return (i, p.max(1e-8).ln());
        }
    }
    // Fallback
    (n - 1, probs[n - 1].max(1e-8).ln())
}

// ---------------------------------------------------------------------------
// Policy abstraction
// ---------------------------------------------------------------------------

const NUM_ACTIONS: usize = 14;
const MAX_ABILITIES: usize = 8;

/// Either actor-critic weights (full policy), legacy transformer weights (bootstrap),
/// or the combined ability-eval + squad AI system.
enum Policy {
    ActorCritic(bevy_game::ai::core::ability_transformer::ActorCriticWeights),
    ActorCriticV2(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV2),
    ActorCriticV3(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV3),
    Legacy(bevy_game::ai::core::ability_transformer::AbilityTransformerWeights),
    /// Uses existing squad AI (force-based + ability eval + student) — no transformer.
    /// Records decisions in the same format for distillation / warmstarting.
    Combined,
}

impl Policy {
    fn load(path: &std::path::Path) -> Result<Self, String> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        // Try v3 actor-critic first (pointer-based)
        if json_str.contains("\"actor_critic_v3\"") {
            let ac = bevy_game::ai::core::ability_transformer::ActorCriticWeightsV3::from_json(&json_str)?;
            return Ok(Policy::ActorCriticV3(ac));
        }

        // Try v2 actor-critic
        if json_str.contains("\"actor_critic_v2\"") {
            let ac = bevy_game::ai::core::ability_transformer::ActorCriticWeightsV2::from_json(&json_str)?;
            return Ok(Policy::ActorCriticV2(ac));
        }

        // Try v1 actor-critic
        if json_str.contains("\"actor_critic\"") {
            let ac = bevy_game::ai::core::ability_transformer::ActorCriticWeights::from_json(&json_str)?;
            return Ok(Policy::ActorCritic(ac));
        }

        // Fall back to legacy transformer
        let tw = bevy_game::ai::core::ability_transformer::AbilityTransformerWeights::from_json(&json_str)?;
        Ok(Policy::Legacy(tw))
    }

    fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        match self {
            Policy::ActorCritic(ac) => ac.encode_cls(token_ids),
            Policy::ActorCriticV2(ac) => ac.encode_cls(token_ids),
            Policy::ActorCriticV3(ac) => ac.encode_cls(token_ids),
            Policy::Legacy(tw) => tw.encode_cls(token_ids),
            Policy::Combined => Vec::new(), // Not used
        }
    }

    fn needs_transformer(&self) -> bool {
        !matches!(self, Policy::Combined)
    }
}

// ---------------------------------------------------------------------------
// Episode runner
// ---------------------------------------------------------------------------

fn run_rl_episode(
    initial_sim: bevy_game::ai::core::SimState,
    initial_squad_ai: bevy_game::ai::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    policy: &Policy,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    temperature: f32,
    rng_seed: u64,
    step_interval: u64,
    student_weights: &Option<std::sync::Arc<super::training::StudentWeights>>,
    grid_nav: Option<bevy_game::ai::pathing::GridNav>,
) -> RlEpisode {
    use bevy_game::ai::core::{is_alive, step, Team, UnitIntent, FIXED_TICK_MS};
    use bevy_game::ai::core::ability_eval::{extract_game_state, extract_game_state_v2};
    use bevy_game::ai::core::self_play::actions::{action_mask, action_to_intent, intent_to_action};
    use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;
    use bevy_game::ai::squad::generate_intents;

    let mut sim = initial_sim;
    // V3 needs GridNav for position token extraction
    if let Some(nav) = grid_nav {
        sim.grid_nav = Some(nav);
    }
    let mut squad_ai = initial_squad_ai;
    let mut rng = rng_seed;
    let mut steps = Vec::new();

    let hero_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    // Pre-tokenize and cache CLS embeddings per hero ability
    let mut unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>> =
        std::collections::HashMap::new();
    let mut cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>> =
        std::collections::HashMap::new();

    for &uid in &hero_ids {
        if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
            let mut ability_tokens_list = Vec::new();
            for (idx, slot) in unit.abilities.iter().enumerate() {
                let dsl = emit_ability_dsl(&slot.def);
                let tokens = tokenizer.encode_with_cls(&dsl);
                if policy.needs_transformer() {
                    let cls = policy.encode_cls(&tokens);
                    cls_cache.insert((uid, idx), cls);
                }
                ability_tokens_list.push(tokens);
            }
            unit_abilities.insert(uid, ability_tokens_list);
        }
    }

    // Dense reward tracking
    let mut prev_hero_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
    let mut prev_enemy_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
    let total_hp_start = (prev_hero_hp + prev_enemy_hp).max(1) as f32;
    let initial_enemy_count = sim.units.iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;
    let initial_hero_count = sim.units.iter()
        .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;
    let mut pending_event_reward: f32 = 0.0;

    for tick in 0..max_ticks {
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let record = tick % step_interval == 0;

        // Compute dense step reward from events + HP differential
        let step_r = if record {
            let cur_hero_hp: i32 = sim.units.iter()
                .filter(|u| u.team == Team::Hero).map(|u| u.hp.max(0)).sum();
            let cur_enemy_hp: i32 = sim.units.iter()
                .filter(|u| u.team == Team::Enemy).map(|u| u.hp.max(0)).sum();

            // HP differential (scaled up 3x for stronger signal)
            let enemy_dmg = (prev_enemy_hp - cur_enemy_hp).max(0) as f32;
            let hero_dmg = (prev_hero_hp - cur_hero_hp).max(0) as f32;
            let hp_reward = 3.0 * (enemy_dmg - hero_dmg) / total_hp_start;

            prev_hero_hp = cur_hero_hp;
            prev_enemy_hp = cur_enemy_hp;

            // Collect accumulated event rewards and reset
            let event_r = pending_event_reward;
            pending_event_reward = 0.0;

            hp_reward + event_r
        } else {
            0.0
        };

        // For Combined policy: generate_intents gives squad AI base, then
        // student model overrides hero combat decisions (ability eval already
        // fires via squad_ai.ability_eval_weights if set).
        if matches!(policy, Policy::Combined) {
            // Replicate the 93% win rate system: ability eval interrupt + student fallthrough.
            // This overrides the squad AI's default decisions for heroes.
            if let Some(ref sw) = *student_weights {
                use bevy_game::ai::core::ability_eval::evaluate_abilities;

                for &uid in &hero_ids {
                    if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) { continue; }
                    if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                        if u.casting.is_some() || u.control_remaining_ms > 0 { continue; }
                    }

                    // Phase 1: ability eval interrupt
                    if let Some(ref ab_weights) = squad_ai.ability_eval_weights {
                        if let Some((action, _urgency)) = evaluate_abilities(
                            &sim, &squad_ai, uid, ab_weights) {
                            intents.retain(|i| i.unit_id != uid);
                            intents.push(UnitIntent { unit_id: uid, action });
                            continue;
                        }
                    }

                    // Phase 2: student model fallthrough
                    let features = bevy_game::ai::core::dataset::extract_unit_features(&sim, &squad_ai, uid);
                    let class = super::training::student_predict_combat(sw, &features);
                    if let Some(action) = super::training::combat_class_to_intent(class, uid, &sim) {
                        intents.retain(|i| i.unit_id != uid);
                        intents.push(UnitIntent { unit_id: uid, action });
                    }
                }
            }

            if record {
                for &uid in &hero_ids {
                    let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                        Some(u) => u,
                        None => continue,
                    };
                    if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                        continue;
                    }
                    let mask_arr = action_mask(&sim, uid);
                    let intent_action = intents.iter()
                        .find(|i| i.unit_id == uid)
                        .map(|i| &i.action)
                        .cloned()
                        .unwrap_or(bevy_game::ai::core::IntentAction::Hold);
                    let action = intent_to_action(&intent_action, uid, &sim);
                    let gs_v2 = extract_game_state_v2(&sim, unit);
                    let game_state = extract_game_state(&sim, unit);
                    steps.push(RlStep {
                        tick,
                        unit_id: uid,
                        game_state: game_state.to_vec(),
                        action,
                        log_prob: 0.0,
                        mask: mask_arr.to_vec(),
                        step_reward: step_r,
                        entities: Some(gs_v2.entities),
                        entity_types: Some(gs_v2.entity_types),
                        threats: Some(gs_v2.threats),
                        positions: None,
                        action_type: None,
                        target_idx: None,
                    });
                }
            }
        } else {
            // Transformer/AC policies: override hero intents with policy output
            for &uid in &hero_ids {
                let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                    Some(u) => u,
                    None => continue,
                };
                if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                    continue;
                }

                let mask_arr = action_mask(&sim, uid);
                let mask_vec: Vec<bool> = mask_arr.to_vec();

                // Extract game states (v1 only when recording, v2 for policy + recording)
                let gs_v2 = extract_game_state_v2(&sim, unit);

                // Build ability CLS list — skip abilities on cooldown (no cross-attn needed)
                let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
                let mut ability_cls_refs: Vec<Option<&[f32]>> = vec![None; MAX_ABILITIES];
                for idx in 0..n_abilities {
                    if unit.abilities[idx].cooldown_remaining_ms == 0 && mask_arr[3 + idx] {
                        if let Some(cls) = cls_cache.get(&(uid, idx)) {
                            ability_cls_refs[idx] = Some(cls.as_slice());
                        }
                    }
                }

                // V3 pointer policy uses a completely different action space
                if let Policy::ActorCriticV3(ac) = policy {
                    use bevy_game::ai::core::self_play::actions::{
                        pointer_action_to_intent, build_token_infos,
                    };

                    let ent_refs: Vec<&[f32]> = gs_v2.entities.iter()
                        .map(|e| e.as_slice()).collect();
                    let type_refs: Vec<usize> = gs_v2.entity_types.iter()
                        .map(|&t| t as usize).collect();
                    let threat_refs: Vec<&[f32]> = gs_v2.threats.iter()
                        .map(|t| t.as_slice()).collect();
                    let pos_refs: Vec<&[f32]> = gs_v2.positions.iter()
                        .map(|p| p.as_slice()).collect();

                    let ent_state = ac.encode_entities_v3(
                        &ent_refs, &type_refs, &threat_refs, &pos_refs,
                    );
                    let ptr_out = ac.pointer_logits(&ent_state, &ability_cls_refs);

                    // Build action type mask
                    let has_enemies = ent_state.type_ids.iter().any(|&t| t == 1);
                    let mut type_mask = vec![false; 11];
                    type_mask[0] = has_enemies; // attack
                    type_mask[1] = true;        // move (always valid if any non-self tokens)
                    type_mask[2] = true;        // hold
                    for idx in 0..n_abilities {
                        if mask_arr[3 + idx] {
                            type_mask[3 + idx] = true;
                        }
                    }

                    // Sample action type
                    let (action_type, type_log_prob) = masked_softmax_sample(
                        &ptr_out.type_logits, &type_mask, temperature, &mut rng,
                    );

                    // Sample target pointer based on action type
                    let (target_idx, target_log_prob, intent_action) = match action_type {
                        0 => {
                            // Attack: sample from attack pointer
                            let atk_mask: Vec<bool> = ptr_out.attack_ptr.iter()
                                .map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
                            let (idx, lp) = masked_softmax_sample(
                                &ptr_out.attack_ptr, &atk_mask, temperature, &mut rng,
                            );
                            let token_infos = build_token_infos(
                                &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                            );
                            let intent = pointer_action_to_intent(
                                action_type, idx, uid, &sim, &token_infos,
                            );
                            (idx, lp, intent)
                        }
                        1 => {
                            // Move: sample from move pointer
                            let mv_mask: Vec<bool> = ptr_out.move_ptr.iter()
                                .map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
                            let (idx, lp) = masked_softmax_sample(
                                &ptr_out.move_ptr, &mv_mask, temperature, &mut rng,
                            );
                            let token_infos = build_token_infos(
                                &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                            );
                            let intent = pointer_action_to_intent(
                                action_type, idx, uid, &sim, &token_infos,
                            );
                            (idx, lp, intent)
                        }
                        2 => {
                            // Hold: no pointer needed
                            (0, 0.0, bevy_game::ai::core::IntentAction::Hold)
                        }
                        t @ 3..=10 => {
                            // Ability: sample from ability pointer
                            let ab_idx = t - 3;
                            if let Some(Some(ab_ptr)) = ptr_out.ability_ptrs.get(ab_idx) {
                                let ab_mask: Vec<bool> = ab_ptr.iter()
                                    .map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
                                let (idx, lp) = masked_softmax_sample(
                                    ab_ptr, &ab_mask, temperature, &mut rng,
                                );
                                let token_infos = build_token_infos(
                                    &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                                );
                                let intent = pointer_action_to_intent(
                                    action_type, idx, uid, &sim, &token_infos,
                                );
                                (idx, lp, intent)
                            } else {
                                (0, 0.0, bevy_game::ai::core::IntentAction::Hold)
                            }
                        }
                        _ => (0, 0.0, bevy_game::ai::core::IntentAction::Hold),
                    };

                    // Composite log prob = log P(type) + log P(target | type)
                    let composite_log_prob = type_log_prob + target_log_prob;

                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action: intent_action });

                    if record {
                        let game_state = extract_game_state(&sim, unit);
                        steps.push(RlStep {
                            tick,
                            unit_id: uid,
                            game_state: game_state.to_vec(),
                            action: action_type, // store action_type in action field for compat
                            log_prob: composite_log_prob,
                            mask: mask_vec,
                            step_reward: step_r,
                            entities: Some(gs_v2.entities.clone()),
                            entity_types: Some(gs_v2.entity_types.clone()),
                            threats: Some(gs_v2.threats.clone()),
                            positions: Some(gs_v2.positions.clone()),
                            action_type: Some(action_type),
                            target_idx: Some(target_idx),
                        });
                    }
                    continue;
                }

                // Compute action logits (V1/V2/Legacy flat action space)
                let logits: Vec<f32> = match policy {
                    Policy::ActorCritic(ac) => {
                        let game_state = extract_game_state(&sim, unit);
                        let ent_state = ac.encode_entities(&game_state);
                        let raw = ac.action_logits(&ent_state, &ability_cls_refs);
                        raw.to_vec()
                    }
                    Policy::ActorCriticV2(ac) => {
                        let ent_refs: Vec<&[f32]> = gs_v2.entities.iter()
                            .map(|e| e.as_slice()).collect();
                        let type_refs: Vec<usize> = gs_v2.entity_types.iter()
                            .map(|&t| t as usize).collect();
                        let threat_refs: Vec<&[f32]> = gs_v2.threats.iter()
                            .map(|t| t.as_slice()).collect();
                        let ent_state = ac.encode_entities_v2(
                            &ent_refs, &type_refs, &threat_refs,
                        );
                        let raw = ac.action_logits(&ent_state, &ability_cls_refs);
                        raw.to_vec()
                    }
                    Policy::Legacy(tw) => {
                        let game_state = extract_game_state(&sim, unit);
                        let mut logits = vec![0.0f32; NUM_ACTIONS];
                        if let Some(entities) = tw.encode_entities(&game_state) {
                            for idx in 0..n_abilities {
                                if let Some(cls) = cls_cache.get(&(uid, idx)) {
                                    let output = tw.predict_from_cls(cls, Some(&entities));
                                    let u = output.urgency.clamp(0.001, 0.999);
                                    logits[3 + idx] = (u / (1.0 - u)).ln();
                                }
                            }
                        }
                        logits
                    }
                    Policy::ActorCriticV3(_) | Policy::Combined => unreachable!(),
                };

                // Sample action
                let (action, log_prob) = masked_softmax_sample(
                    &logits, &mask_arr, temperature, &mut rng,
                );

                // Convert to intent
                let intent_action = action_to_intent(action, uid, &sim);
                intents.retain(|i| i.unit_id != uid);
                intents.push(UnitIntent { unit_id: uid, action: intent_action });

                if record {
                    let game_state = extract_game_state(&sim, unit);
                    steps.push(RlStep {
                        tick,
                        unit_id: uid,
                        game_state: game_state.to_vec(),
                        action,
                        log_prob,
                        mask: mask_vec,
                        step_reward: step_r,
                        entities: Some(gs_v2.entities.clone()),
                        entity_types: Some(gs_v2.entity_types.clone()),
                        threats: Some(gs_v2.threats.clone()),
                        positions: None,
                        action_type: None,
                        target_idx: None,
                    });
                }
            }
        }

        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);

        // Dense event-based rewards: kills and deaths
        for ev in &events {
            match ev {
                bevy_game::ai::core::SimEvent::UnitDied { unit_id, .. } => {
                    if let Some(dead_unit) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                        if dead_unit.team == Team::Enemy {
                            // Enemy kill: +0.3 scaled by how many enemies started
                            pending_event_reward += 0.3 / initial_enemy_count.max(1.0);
                        } else if dead_unit.team == Team::Hero {
                            // Hero death: -0.4 scaled by how many heroes started
                            pending_event_reward -= 0.4 / initial_hero_count.max(1.0);
                        }
                    }
                }
                _ => {}
            }
        }

        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: "Victory".to_string(),
                reward: 1.0,
                ticks: sim.tick,
                unit_abilities,
                steps,
            };
        }
        if heroes_alive == 0 {
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: "Defeat".to_string(),
                reward: -1.0,
                ticks: sim.tick,
                unit_abilities,
                steps,
            };
        }
    }

    let hero_hp_frac = hp_fraction(&sim, bevy_game::ai::core::Team::Hero);
    let enemy_hp_frac = hp_fraction(&sim, bevy_game::ai::core::Team::Enemy);
    let shaped = (enemy_hp_frac - hero_hp_frac).clamp(-1.0, 1.0) * 0.5;

    RlEpisode {
        scenario: scenario_name.to_string(),
        outcome: "Timeout".to_string(),
        reward: shaped,
        ticks: sim.tick,
        unit_abilities,
        steps,
    }
}

fn hp_fraction(sim: &bevy_game::ai::core::SimState, team: bevy_game::ai::core::Team) -> f32 {
    let mut lost = 0i32;
    let mut total = 0i32;
    for u in &sim.units {
        if u.team == team {
            total += u.max_hp;
            lost += u.max_hp - u.hp.max(0);
        }
    }
    if total == 0 { 0.0 } else { lost as f32 / total as f32 }
}

// ---------------------------------------------------------------------------
// CLI entry points
// ---------------------------------------------------------------------------

pub fn run_transformer_rl(args: crate::cli::TransformerRlArgs) -> ExitCode {
    match args.sub {
        crate::cli::TransformerRlSubcommand::Generate(gen_args) => run_generate(gen_args),
        crate::cli::TransformerRlSubcommand::Eval(eval_args) => run_eval(eval_args),
    }
}

fn run_generate(args: crate::cli::TransformerRlGenerateArgs) -> ExitCode {
    use bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state, run_scenario_to_state_with_room};

    // Load ability eval weights for Combined policy (also used to configure squad AI)
    let ability_eval_weights = if let Some(ref path) = args.ability_eval {
        let json_str = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => { eprintln!("Failed to read ability eval weights: {e}"); return ExitCode::from(1); }
        };
        let json_val: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to parse ability eval JSON: {e}"); return ExitCode::from(1); }
        };
        Some(std::sync::Arc::new(
            bevy_game::ai::core::ability_eval::AbilityEvalWeights::from_json(&json_val)
        ))
    } else {
        None
    };

    // Load combat student model for Combined policy
    let student_weights = if let Some(ref path) = args.student_model {
        let json_str = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => { eprintln!("Failed to read student model: {e}"); return ExitCode::from(1); }
        };
        let json_val: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to parse student model JSON: {e}"); return ExitCode::from(1); }
        };
        Some(std::sync::Arc::new(super::training::StudentWeights::from_json(&json_val)))
    } else {
        None
    };

    let policy = if args.policy == "combined" {
        Policy::Combined
    } else {
        let weights_path = match &args.weights {
            Some(p) => p,
            None => { eprintln!("--weights is required for transformer policy"); return ExitCode::from(1); }
        };
        match Policy::load(weights_path) {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to load weights: {e}"); return ExitCode::from(1); }
        }
    };
    let policy_type = match &policy {
        Policy::ActorCritic(_) => "actor-critic",
        Policy::ActorCriticV2(_) => "actor-critic-v2",
        Policy::ActorCriticV3(_) => "actor-critic-v3 (pointer)",
        Policy::Legacy(_) => "legacy (bootstrap)",
        Policy::Combined => "combined (ability-eval + squad AI)",
    };
    let is_v3 = matches!(&policy, Policy::ActorCriticV3(_));

    let tokenizer = AbilityTokenizer::new();

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    eprintln!("Generating RL episodes: {} scenarios × {} episodes, temp={:.2}, policy={}",
        paths.len(), args.episodes, args.temperature, policy_type);

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("{e}"); None }
        }
    }).collect();

    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let ability_eval_ref = &ability_eval_weights;
    let student_ref = &student_weights;
    let step_interval = args.step_interval;
    let temperature = args.temperature;
    let max_ticks_override = args.max_ticks;

    let episode_tasks: Vec<(usize, usize)> = scenarios.iter().enumerate()
        .flat_map(|(si, _)| (0..args.episodes as usize).map(move |ei| (si, ei)))
        .collect();

    let episodes: Vec<RlEpisode> = pool.install(|| {
        episode_tasks.par_iter().map(|&(si, ei)| {
            let scenario_file = &scenarios[si];
            let cfg = &scenario_file.scenario;
            let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);

            // V3 needs GridNav for position token extraction
            let (sim, mut squad_ai, grid_nav) = if is_v3 {
                let (s, ai, nav) = run_scenario_to_state_with_room(cfg);
                (s, ai, Some(nav))
            } else {
                let (s, ai) = run_scenario_to_state(cfg);
                (s, ai, None)
            };

            // Inject ability eval weights into squad AI for Combined policy
            if matches!(policy_ref, Policy::Combined) {
                if let Some(ref w) = *ability_eval_ref {
                    squad_ai.ability_eval_weights = Some((**w).clone());
                }
            }

            let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;

            run_rl_episode(
                sim, squad_ai, &cfg.name, max_ticks,
                policy_ref, tokenizer_ref,
                temperature, seed, step_interval,
                student_ref,
                grid_nav,
            )
        }).collect()
    });

    let wins = episodes.iter().filter(|e| e.outcome == "Victory").count();
    let losses = episodes.iter().filter(|e| e.outcome == "Defeat").count();
    let timeouts = episodes.iter().filter(|e| e.outcome == "Timeout").count();
    let total_steps: usize = episodes.iter().map(|e| e.steps.len()).sum();
    let mean_reward: f32 = episodes.iter().map(|e| e.reward).sum::<f32>() / episodes.len().max(1) as f32;

    eprintln!("Episodes: {}  Wins: {}  Losses: {}  Timeouts: {}  Win rate: {:.1}%",
        episodes.len(), wins, losses, timeouts,
        wins as f64 / episodes.len().max(1) as f64 * 100.0);
    eprintln!("Total steps: {}  Mean reward: {:.3}", total_steps, mean_reward);

    let file = std::fs::File::create(&args.output).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    for ep in &episodes {
        let line = serde_json::to_string(ep).unwrap();
        writeln!(writer, "{}", line).unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Wrote {} episodes to {}", episodes.len(), args.output.display());

    ExitCode::SUCCESS
}

fn run_eval(args: crate::cli::TransformerRlEvalArgs) -> ExitCode {
    use bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state, run_scenario_to_state_with_room};
    use rayon::prelude::*;

    let policy = match Policy::load(&args.weights) {
        Ok(p) => p,
        Err(e) => { eprintln!("Failed to load weights: {e}"); return ExitCode::from(1); }
    };
    let is_v3 = matches!(&policy, Policy::ActorCriticV3(_));

    let tokenizer = AbilityTokenizer::new();
    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("{e}"); None }
        }
    }).collect();

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let max_ticks_override = args.max_ticks;
    let no_student: Option<std::sync::Arc<super::training::StudentWeights>> = None;
    let student_ref = &no_student;

    let results: Vec<(String, RlEpisode)> = scenarios.par_iter().map(|scenario_file| {
        let cfg = &scenario_file.scenario;
        let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);

        let (sim, squad_ai, grid_nav) = if is_v3 {
            let (s, ai, nav) = run_scenario_to_state_with_room(cfg);
            (s, ai, Some(nav))
        } else {
            let (s, ai) = run_scenario_to_state(cfg);
            (s, ai, None)
        };

        let episode = run_rl_episode(
            sim, squad_ai, &cfg.name, max_ticks,
            policy_ref, tokenizer_ref, 0.01, 42, 1,
            student_ref,
            grid_nav,
        );
        (cfg.name.clone(), episode)
    }).collect();

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;

    for (name, episode) in &results {
        let tag = match episode.outcome.as_str() {
            "Victory" => { wins += 1; "WIN " }
            "Defeat" => { losses += 1; "LOSS" }
            _ => { timeouts += 1; "TIME" }
        };
        println!("[{tag}] {:<30} tick={:<5} reward={:.2}", name, episode.ticks, episode.reward);
    }

    let total = wins + losses + timeouts;
    if total > 1 {
        println!("\n--- Aggregate ---");
        println!("Scenarios: {total}  Wins: {wins}  Losses: {losses}  Timeouts: {timeouts}  Win rate: {:.1}%",
            if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 });
    }

    ExitCode::SUCCESS
}
