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
    /// Per-unit ability names (unit_id → list of ability names).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub unit_ability_names: std::collections::HashMap<u32, Vec<String>>,
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
    // V4 dual-head: movement direction + combat action
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_dir: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combat_type: Option<usize>,
    // V4 separate log probs per head (for V-trace importance ratios)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_move: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_combat: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_pointer: Option<f32>,
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

    // Temperature-scaled softmax for sampling
    let mut max_scaled = f32::NEG_INFINITY;
    for i in 0..n {
        if mask[i] {
            let scaled = logits[i] / temp;
            if scaled > max_scaled { max_scaled = scaled; }
        }
    }

    let mut probs = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for i in 0..n {
        if mask[i] {
            let e = ((logits[i] / temp) - max_scaled).exp();
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

    // Sample from temperature-scaled distribution
    let r = lcg_f32(rng);
    let mut cum = 0.0;
    let mut chosen = n - 1;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            chosen = i;
            break;
        }
    }

    // Return UNTEMPERED log probability (matches Python's F.log_softmax(raw_logits))
    // This is critical for PPO: old_log_prob must use the same distribution as the
    // Python training loop's compute_hierarchical_log_prob().
    let mut max_raw = f32::NEG_INFINITY;
    for i in 0..n {
        if mask[i] && logits[i] > max_raw { max_raw = logits[i]; }
    }
    let mut log_sum_exp = 0.0f32;
    for i in 0..n {
        if mask[i] {
            log_sum_exp += (logits[i] - max_raw).exp();
        }
    }
    let log_prob = logits[chosen] - max_raw - log_sum_exp.ln();

    (chosen, log_prob)
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
    ActorCriticV4(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV4),
    /// GPU inference via TCP — sends states to Python GPU server, receives actions.
    GpuServer(std::sync::Arc<bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient>),
    Legacy(bevy_game::ai::core::ability_transformer::AbilityTransformerWeights),
    /// Uses existing squad AI (force-based + ability eval + student) — no transformer.
    /// Records decisions in the same format for distillation / warmstarting.
    Combined,
    /// Uniformly random actions — no model inference.
    Random,
}

impl Policy {
    fn load(path: &std::path::Path) -> Result<Self, String> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        // Try v4 actor-critic first (dual-head: move + combat)
        if json_str.contains("\"actor_critic_v4\"") {
            let ac = bevy_game::ai::core::ability_transformer::ActorCriticWeightsV4::from_json(&json_str)?;
            return Ok(Policy::ActorCriticV4(ac));
        }

        // Try v3 actor-critic (pointer-based)
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
            Policy::ActorCriticV4(ac) => ac.encode_cls(token_ids),
            Policy::Legacy(tw) => tw.encode_cls(token_ids),
            Policy::GpuServer(_) | Policy::Combined | Policy::Random => Vec::new(),
        }
    }

    fn needs_transformer(&self) -> bool {
        !matches!(self, Policy::Combined | Policy::GpuServer(_) | Policy::Random)
    }

    fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        match self {
            Policy::ActorCritic(ac) => ac.project_external_cls(cls),
            Policy::ActorCriticV3(ac) => ac.project_external_cls(cls),
            Policy::ActorCriticV4(ac) => ac.project_external_cls(cls),
            _ => cls.to_vec(),
        }
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
    embedding_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    enemy_policy: Option<&Policy>,
    enemy_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> RlEpisode {
    use bevy_game::ai::core::{is_alive, step, distance, move_towards, position_at_range, Team, UnitIntent, FIXED_TICK_MS};
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
    let mut unit_ability_names: std::collections::HashMap<u32, Vec<String>> =
        std::collections::HashMap::new();
    let mut cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>> =
        std::collections::HashMap::new();

    for &uid in &hero_ids {
        if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
            let mut ability_tokens_list = Vec::new();
            let mut ability_names_list = Vec::new();
            for (idx, slot) in unit.abilities.iter().enumerate() {
                let dsl = emit_ability_dsl(&slot.def);
                let tokens = tokenizer.encode_with_cls(&dsl);

                // CLS: prefer registry lookup, fall back to transformer encoding
                let safe_name = slot.def.name.replace(' ', "_");
                if let Some(reg) = embedding_registry {
                    if let Some(reg_cls) = reg.get(&safe_name) {
                        // Project external CLS (128→d_model) if policy supports it
                        let projected = policy.project_external_cls(reg_cls);
                        cls_cache.insert((uid, idx), projected);
                    } else if policy.needs_transformer() {
                        let cls = policy.encode_cls(&tokens);
                        cls_cache.insert((uid, idx), cls);
                    }
                } else if policy.needs_transformer() {
                    let cls = policy.encode_cls(&tokens);
                    cls_cache.insert((uid, idx), cls);
                }

                ability_tokens_list.push(tokens);
                ability_names_list.push(slot.def.name.clone());
            }
            unit_abilities.insert(uid, ability_tokens_list);
            unit_ability_names.insert(uid, ability_names_list);
        }
    }

    // Self-play: set up enemy policy CLS cache
    let enemy_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| u.id)
        .collect();
    let mut enemy_cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>> =
        std::collections::HashMap::new();
    if let Some(ep) = enemy_policy {
        for &uid in &enemy_ids {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    let tokens = tokenizer.encode_with_cls(&dsl);
                    let safe_name = slot.def.name.replace(' ', "_");
                    if let Some(reg) = enemy_registry {
                        if let Some(reg_cls) = reg.get(&safe_name) {
                            let projected = ep.project_external_cls(reg_cls);
                            enemy_cls_cache.insert((uid, idx), projected);
                        } else if ep.needs_transformer() {
                            let cls = ep.encode_cls(&tokens);
                            enemy_cls_cache.insert((uid, idx), cls);
                        }
                    } else if ep.needs_transformer() {
                        let cls = ep.encode_cls(&tokens);
                        enemy_cls_cache.insert((uid, idx), cls);
                    }
                }
            }
        }
    }

    // Dense reward tracking
    let mut prev_hero_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
    let mut prev_enemy_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
    let n_units = sim.units.iter().filter(|u| u.hp > 0).count().max(1) as f32;
    let avg_unit_hp = (prev_hero_hp + prev_enemy_hp) as f32 / n_units;
    let initial_enemy_count = sim.units.iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;
    let initial_hero_count = sim.units.iter()
        .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;
    let mut pending_event_reward: f32 = 0.0;

    for tick in 0..max_ticks {
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let record = tick % step_interval == 0;

        // Compute dense step reward (HP differential normalized by avg unit HP)
        let step_r = if record {
            let cur_hero_hp: i32 = sim.units.iter()
                .filter(|u| u.team == Team::Hero).map(|u| u.hp.max(0)).sum();
            let cur_enemy_hp: i32 = sim.units.iter()
                .filter(|u| u.team == Team::Enemy).map(|u| u.hp.max(0)).sum();

            let enemy_dmg = (prev_enemy_hp - cur_enemy_hp).max(0) as f32;
            let hero_dmg = (prev_hero_hp - cur_hero_hp).max(0) as f32;
            let hp_reward = (enemy_dmg - hero_dmg) / avg_unit_hp.max(1.0);

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
                use bevy_game::ai::core::self_play::actions::{
                    build_token_infos, intent_to_v3_action, intent_to_v4_action,
                };
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
                    // Convert oracle intent to V3 pointer format
                    let token_infos = build_token_infos(
                        &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                    );
                    let (v3_action_type, v3_target_idx) = intent_to_v3_action(
                        &intent_action, uid, &sim, &token_infos,
                    ).unwrap_or((2, 0)); // default to hold
                    // Convert oracle intent to V4 dual-head format
                    let (v4_move_dir, v4_combat_type, _v4_target_idx) = intent_to_v4_action(
                        &intent_action, uid, &sim, &token_infos,
                    ).unwrap_or((8, 1, 0)); // default: stay + hold
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
                        positions: Some(gs_v2.positions),
                        action_type: Some(v3_action_type),
                        target_idx: Some(v3_target_idx),
                        move_dir: Some(v4_move_dir),
                        combat_type: Some(v4_combat_type),
                        lp_move: None,
                        lp_combat: None,
                        lp_pointer: None,
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

                // Random policy: skip all model inference, pick uniformly random actions
                if matches!(policy, Policy::Random) {
                    use bevy_game::ai::core::self_play::actions::{
                        move_dir_to_intent, combat_action_to_intent, build_token_infos,
                        NUM_MOVE_DIRS, COMBAT_TYPE_ATTACK, COMBAT_TYPE_HOLD,
                    };
                    let gs_v2 = extract_game_state_v2(&sim, unit);
                    let n_abilities = unit.abilities.len().min(MAX_ABILITIES);

                    // Random movement direction (0-8, where 8 = stay)
                    let move_dir = (lcg_f32(&mut rng) * NUM_MOVE_DIRS as f32) as usize;

                    // Build combat mask
                    let has_enemies = sim.units.iter().any(|e| e.team == Team::Enemy && e.hp > 0);
                    let n_combat = 2 + n_abilities; // attack, hold, ability_0..N
                    let mut combat_mask = vec![false; n_combat];
                    combat_mask[COMBAT_TYPE_ATTACK] = has_enemies;
                    combat_mask[COMBAT_TYPE_HOLD] = true;
                    for idx in 0..n_abilities {
                        if mask_arr[3 + idx] { combat_mask[2 + idx] = true; }
                    }
                    let valid_combat: Vec<usize> = combat_mask.iter().enumerate()
                        .filter(|(_, &v)| v).map(|(i, _)| i).collect();
                    let combat_type = valid_combat[(lcg_f32(&mut rng) * valid_combat.len() as f32) as usize % valid_combat.len()];

                    // Random target pointer
                    let n_entities = gs_v2.entities.len();
                    let target_idx = if n_entities > 0 {
                        (lcg_f32(&mut rng) * n_entities as f32) as usize % n_entities
                    } else { 0 };

                    let token_infos = build_token_infos(
                        &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                    );
                    let move_intent = move_dir_to_intent(move_dir, uid, &sim);
                    let combat_intent = combat_action_to_intent(
                        combat_type, target_idx, uid, &sim, &token_infos,
                    );
                    let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
                        combat_intent
                    } else {
                        move_intent
                    };

                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action: final_intent });

                    if record {
                        let game_state = extract_game_state(&sim, unit);
                        steps.push(RlStep {
                            tick, unit_id: uid,
                            game_state: game_state.to_vec(),
                            action: combat_type,
                            log_prob: 0.0,
                            mask: mask_vec,
                            step_reward: step_r,
                            entities: Some(gs_v2.entities),
                            entity_types: Some(gs_v2.entity_types),
                            threats: Some(gs_v2.threats),
                            positions: Some(gs_v2.positions),
                            action_type: Some(combat_type),
                            target_idx: Some(target_idx),
                            move_dir: Some(move_dir),
                            combat_type: Some(combat_type),
                            lp_move: Some(0.0),
                            lp_combat: Some(0.0),
                            lp_pointer: Some(0.0),
                        });
                    }
                    continue;
                }

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

                    // Engagement heuristic: if model says hold but no enemy in
                    // attack range and enemies exist, move toward nearest enemy.
                    // The BC data has 0% move actions (oracle movement happens
                    // via squad AI before recording), so the model never learned
                    // to move. This heuristic provides baseline engagement.
                    let final_intent = if matches!(intent_action, bevy_game::ai::core::IntentAction::Hold) {
                        let unit_pos = unit.position;
                        let atk_range = unit.attack_range;
                        let has_enemy_in_range = sim.units.iter().any(|e| {
                            e.team == Team::Enemy && e.hp > 0
                                && distance(unit_pos, e.position) <= atk_range * 1.1
                        });
                        if !has_enemy_in_range {
                            // Find nearest living enemy and move toward them
                            if let Some(nearest) = sim.units.iter()
                                .filter(|e| e.team == Team::Enemy && e.hp > 0)
                                .min_by(|a, b| {
                                    distance(unit_pos, a.position)
                                        .partial_cmp(&distance(unit_pos, b.position))
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                            {
                                let step = unit.move_speed_per_sec * 0.1;
                                let desired = position_at_range(
                                    unit_pos, nearest.position, atk_range * 0.9,
                                );
                                let next = move_towards(unit_pos, desired, step);
                                bevy_game::ai::core::IntentAction::MoveTo { position: next }
                            } else {
                                intent_action
                            }
                        } else {
                            intent_action
                        }
                    } else {
                        intent_action
                    };

                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action: final_intent });

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
                            move_dir: None,
                            combat_type: None,
                            lp_move: None,
                            lp_combat: None,
                            lp_pointer: None,
                        });
                    }
                    continue;
                }

                // GPU server policy: send state to Python GPU, receive actions
                if let Policy::GpuServer(gpu) = policy {
                    use bevy_game::ai::core::self_play::actions::{
                        move_dir_to_intent, combat_action_to_intent, build_token_infos,
                    };
                    use bevy_game::ai::core::ability_transformer::gpu_client::InferenceRequest;

                    // Build combat mask
                    let has_enemies = gs_v2.entity_types.iter().any(|&t| t == 1);
                    let mut combat_mask_vec = vec![false; 10];
                    combat_mask_vec[0] = has_enemies; // attack
                    combat_mask_vec[1] = true; // hold
                    for idx in 0..n_abilities {
                        if mask_arr[3 + idx] {
                            combat_mask_vec[2 + idx] = true;
                        }
                    }

                    // Build ability CLS list for the request
                    let ability_cls_for_req: Vec<Option<Vec<f32>>> = (0..MAX_ABILITIES)
                        .map(|i| {
                            ability_cls_refs.get(i)
                                .and_then(|opt| opt.map(|s| s.to_vec()))
                        })
                        .collect();

                    let req = InferenceRequest {
                        entities: gs_v2.entities.clone(),
                        entity_types: gs_v2.entity_types.clone(),
                        threats: gs_v2.threats.clone(),
                        positions: gs_v2.positions.clone(),
                        combat_mask: combat_mask_vec,
                        ability_cls: ability_cls_for_req,
                    };

                    match gpu.infer(req) {
                        Ok(result) => {
                            let move_dir = result.move_dir as usize;
                            let combat_type = result.combat_type as usize;
                            let target_idx = result.target_idx as usize;

                            let move_intent = move_dir_to_intent(move_dir, uid, &sim);
                            let token_infos = build_token_infos(
                                &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                            );
                            let combat_intent = combat_action_to_intent(
                                combat_type, target_idx, uid, &sim, &token_infos,
                            );

                            let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
                                combat_intent
                            } else {
                                move_intent
                            };

                            intents.retain(|i| i.unit_id != uid);
                            intents.push(UnitIntent { unit_id: uid, action: final_intent });

                            if record {
                                let game_state = extract_game_state(&sim, unit);
                                let composite_lp = result.lp_move + result.lp_combat + result.lp_pointer;
                                steps.push(RlStep {
                                    tick,
                                    unit_id: uid,
                                    game_state: game_state.to_vec(),
                                    action: combat_type,
                                    log_prob: composite_lp,
                                    mask: mask_vec,
                                    step_reward: step_r,
                                    entities: Some(gs_v2.entities.clone()),
                                    entity_types: Some(gs_v2.entity_types.clone()),
                                    threats: Some(gs_v2.threats.clone()),
                                    positions: Some(gs_v2.positions.clone()),
                                    action_type: Some(combat_type),
                                    target_idx: Some(target_idx),
                                    move_dir: Some(move_dir),
                                    combat_type: Some(combat_type),
                                    lp_move: Some(result.lp_move),
                                    lp_combat: Some(result.lp_combat),
                                    lp_pointer: Some(result.lp_pointer),
                                });
                            }
                        }
                        Err(e) => {
                            eprintln!("GPU inference error for unit {uid}: {e}");
                        }
                    }
                    continue;
                }

                // V4 dual-head policy: separate movement direction + combat targeting
                if let Policy::ActorCriticV4(ac) = policy {
                    use bevy_game::ai::core::self_play::actions::{
                        move_dir_to_intent, combat_action_to_intent, build_token_infos,
                        NUM_MOVE_DIRS, COMBAT_TYPE_ATTACK, COMBAT_TYPE_HOLD,
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
                    let dual = ac.dual_head_logits(&ent_state, &ability_cls_refs);

                    // Sample movement direction (9-way)
                    let move_mask = vec![true; NUM_MOVE_DIRS];
                    let (move_dir, move_lp) = masked_softmax_sample(
                        &dual.move_logits, &move_mask, temperature, &mut rng,
                    );

                    // Sample combat action type
                    let has_enemies = ent_state.type_ids.iter().any(|&t| t == 1);
                    let mut combat_mask = vec![false; dual.combat_logits.len()];
                    combat_mask[COMBAT_TYPE_ATTACK] = has_enemies;
                    combat_mask[COMBAT_TYPE_HOLD] = true;
                    for idx in 0..n_abilities {
                        if mask_arr[3 + idx] {
                            combat_mask[2 + idx] = true; // ability slots at 2..9
                        }
                    }
                    let (combat_type, combat_lp) = masked_softmax_sample(
                        &dual.combat_logits, &combat_mask, temperature, &mut rng,
                    );

                    // Sample target pointer for combat action
                    let (target_idx, target_lp, combat_intent) = match combat_type {
                        t if t == COMBAT_TYPE_ATTACK => {
                            let atk_mask: Vec<bool> = dual.attack_ptr.iter()
                                .map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
                            let (idx, lp) = masked_softmax_sample(
                                &dual.attack_ptr, &atk_mask, temperature, &mut rng,
                            );
                            let token_infos = build_token_infos(
                                &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                            );
                            let intent = combat_action_to_intent(
                                combat_type, idx, uid, &sim, &token_infos,
                            );
                            (idx, lp, intent)
                        }
                        t if t == COMBAT_TYPE_HOLD => {
                            (0, 0.0, bevy_game::ai::core::IntentAction::Hold)
                        }
                        t @ 2..=9 => {
                            let ab_idx = t - 2;
                            if let Some(Some(ab_ptr)) = dual.ability_ptrs.get(ab_idx) {
                                let ab_mask: Vec<bool> = ab_ptr.iter()
                                    .map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
                                let (idx, lp) = masked_softmax_sample(
                                    ab_ptr, &ab_mask, temperature, &mut rng,
                                );
                                let token_infos = build_token_infos(
                                    &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                                );
                                let intent = combat_action_to_intent(
                                    combat_type, idx, uid, &sim, &token_infos,
                                );
                                (idx, lp, intent)
                            } else {
                                (0, 0.0, bevy_game::ai::core::IntentAction::Hold)
                            }
                        }
                        _ => (0, 0.0, bevy_game::ai::core::IntentAction::Hold),
                    };

                    // Combine both heads: combat overrides movement
                    let move_intent = move_dir_to_intent(move_dir, uid, &sim);
                    let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
                        combat_intent
                    } else {
                        move_intent
                    };

                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action: final_intent });

                    if record {
                        let game_state = extract_game_state(&sim, unit);
                        let composite_lp = move_lp + combat_lp + target_lp;
                        steps.push(RlStep {
                            tick,
                            unit_id: uid,
                            game_state: game_state.to_vec(),
                            action: combat_type,
                            log_prob: composite_lp,
                            mask: mask_vec,
                            step_reward: step_r,
                            entities: Some(gs_v2.entities.clone()),
                            entity_types: Some(gs_v2.entity_types.clone()),
                            threats: Some(gs_v2.threats.clone()),
                            positions: Some(gs_v2.positions.clone()),
                            action_type: Some(combat_type),
                            target_idx: Some(target_idx),
                            move_dir: Some(move_dir),
                            combat_type: Some(combat_type),
                            lp_move: Some(move_lp),
                            lp_combat: Some(combat_lp),
                            lp_pointer: Some(target_lp),
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
                    Policy::ActorCriticV3(_) | Policy::ActorCriticV4(_) | Policy::GpuServer(_) | Policy::Combined | Policy::Random => unreachable!(),
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
                        move_dir: None,
                        combat_type: None,
                        lp_move: None,
                        lp_combat: None,
                        lp_pointer: None,
                    });
                }
            }
        }

        // Self-play: override enemy intents with enemy policy
        if let Some(ep) = enemy_policy {
            if let Policy::ActorCritic(ac) = ep {
                for &uid in &enemy_ids {
                    let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                        Some(u) => u,
                        None => continue,
                    };
                    if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                        continue;
                    }

                    let mask_arr = action_mask(&sim, uid);

                    let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
                    let mut ability_cls_refs: Vec<Option<&[f32]>> = vec![None; MAX_ABILITIES];
                    for idx in 0..n_abilities {
                        if unit.abilities[idx].cooldown_remaining_ms == 0 && mask_arr[3 + idx] {
                            if let Some(cls) = enemy_cls_cache.get(&(uid, idx)) {
                                ability_cls_refs[idx] = Some(cls.as_slice());
                            }
                        }
                    }

                    let game_state = extract_game_state(&sim, unit);
                    let ent_state = ac.encode_entities(&game_state);
                    let raw = ac.action_logits(&ent_state, &ability_cls_refs);
                    let logits: Vec<f32> = raw.to_vec();

                    let (action, _log_prob) = masked_softmax_sample(
                        &logits, &mask_arr, temperature, &mut rng,
                    );

                    let intent_action = action_to_intent(action, uid, &sim);
                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action: intent_action });
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
                unit_ability_names,
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
                unit_ability_names,
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
        unit_ability_names,
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
// Multiplexed GPU episode runner
// ---------------------------------------------------------------------------

static DUMMY_ATOMIC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// State for one active sim in the multiplexed GPU pipeline.
/// Per-hero snapshot taken before sim step for action-specific reward computation.
#[derive(Clone)]
struct HeroPreStepState {
    unit_id: u32,
    position: bevy_game::ai::core::SimVec2,
    hp: i32,
    nearest_enemy_dist: f32,
    move_dir: usize,
    combat_type: usize,
}

struct ActiveSim {
    sim: bevy_game::ai::core::SimState,
    squad_ai: bevy_game::ai::squad::SquadAiState,
    scenario_name: String,
    max_ticks: u64,
    rng: u64,
    steps: Vec<RlStep>,
    unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>>,
    unit_ability_names: std::collections::HashMap<u32, Vec<String>>,
    cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>>,
    hero_ids: Vec<u32>,
    enemy_ids: Vec<u32>,
    // Dense reward state
    prev_hero_hp: i32,
    prev_enemy_hp: i32,
    avg_unit_hp: f32,
    initial_enemy_count: f32,
    initial_hero_count: f32,
    pending_event_reward: f32,
    // Per-hero pre-step snapshots for action-specific rewards
    hero_pre_step: Vec<HeroPreStepState>,
    // Steps recorded this tick (indices into self.steps) for retroactive reward
    steps_recorded_this_tick: Vec<usize>,
    // Per-tick state
    tick: u64,
    step_interval: u64,
    temperature: f32,
    intents: Vec<bevy_game::ai::core::UnitIntent>,
    // GPU inference state
    pending_units: Vec<PendingUnit>,
    phase: SimPhase,
}

struct PendingUnit {
    unit_id: u32,
    token: bevy_game::ai::core::ability_transformer::gpu_client::InferenceToken,
    gs_v2: bevy_game::ai::core::ability_eval::GameStateV2,
    mask_vec: Vec<bool>,
    n_abilities: usize,
    step_reward: f32,
    resolved: bool,
}

#[derive(PartialEq)]
enum SimPhase {
    NeedsTick,
    WaitingGpu,
}

impl ActiveSim {
    fn new(
        sim: bevy_game::ai::core::SimState,
        squad_ai: bevy_game::ai::squad::SquadAiState,
        scenario_name: String,
        max_ticks: u64,
        rng_seed: u64,
        step_interval: u64,
        temperature: f32,
        tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
        embedding_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    ) -> Self {
        use bevy_game::ai::core::Team;
        use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;

        let hero_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.id).collect();
        let enemy_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.id).collect();

        let mut unit_abilities = std::collections::HashMap::new();
        let mut unit_ability_names = std::collections::HashMap::new();
        let mut cls_cache = std::collections::HashMap::new();

        for &uid in &hero_ids {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                let mut ability_tokens_list = Vec::new();
                let mut ability_names_list = Vec::new();
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    let tokens = tokenizer.encode_with_cls(&dsl);
                    let safe_name = slot.def.name.replace(' ', "_");
                    if let Some(reg) = embedding_registry {
                        if let Some(reg_cls) = reg.get(&safe_name) {
                            cls_cache.insert((uid, idx), reg_cls.to_vec());
                        }
                    }
                    ability_tokens_list.push(tokens);
                    ability_names_list.push(slot.def.name.clone());
                }
                unit_abilities.insert(uid, ability_tokens_list);
                unit_ability_names.insert(uid, ability_names_list);
            }
        }

        let prev_hero_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
        let prev_enemy_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
        let n_units = sim.units.iter().filter(|u| u.hp > 0).count().max(1) as f32;
        let avg_unit_hp = (prev_hero_hp + prev_enemy_hp) as f32 / n_units;
        let initial_enemy_count = sim.units.iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;
        let initial_hero_count = sim.units.iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;

        ActiveSim {
            sim, squad_ai, scenario_name, max_ticks,
            rng: rng_seed, steps: Vec::new(),
            unit_abilities, unit_ability_names, cls_cache,
            hero_ids, enemy_ids,
            prev_hero_hp, prev_enemy_hp, avg_unit_hp,
            initial_enemy_count, initial_hero_count,
            pending_event_reward: 0.0,
            hero_pre_step: Vec::new(),
            steps_recorded_this_tick: Vec::new(),
            tick: 0, step_interval, temperature,
            intents: Vec::new(), pending_units: Vec::new(),
            phase: SimPhase::NeedsTick,
        }
    }

    /// Prepare a new tick: generate intents, compute rewards, submit GPU requests.
    fn prepare_and_submit(
        &mut self,
        gpu: &bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient,
    ) -> Result<(), String> {
        self.prepare_and_submit_profiled(gpu, &DUMMY_ATOMIC, &DUMMY_ATOMIC, &DUMMY_ATOMIC, &DUMMY_ATOMIC)
    }

    fn prepare_and_submit_profiled(
        &mut self,
        gpu: &bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient,
        p_intent: &std::sync::atomic::AtomicU64,
        p_extract: &std::sync::atomic::AtomicU64,
        p_submit: &std::sync::atomic::AtomicU64,
        p_serialize: &std::sync::atomic::AtomicU64,
    ) -> Result<(), String> {
        use bevy_game::ai::core::{is_alive, FIXED_TICK_MS};
        use bevy_game::ai::core::ability_eval::extract_game_state_v2;
        use bevy_game::ai::core::self_play::actions::action_mask;
        use bevy_game::ai::core::ability_transformer::gpu_client::InferenceRequest;
        use bevy_game::ai::squad::generate_intents;
        use std::sync::atomic::Ordering;

        let t0 = std::time::Instant::now();
        self.intents = generate_intents(&self.sim, &mut self.squad_ai, FIXED_TICK_MS);
        p_intent.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let record = self.tick % self.step_interval == 0;

        // Compute dense step reward (HP differential normalized by avg unit HP)
        let step_r = if record {
            let cur_hero_hp: i32 = self.sim.units.iter()
                .filter(|u| u.team == bevy_game::ai::core::Team::Hero).map(|u| u.hp.max(0)).sum();
            let cur_enemy_hp: i32 = self.sim.units.iter()
                .filter(|u| u.team == bevy_game::ai::core::Team::Enemy).map(|u| u.hp.max(0)).sum();
            let enemy_dmg = (self.prev_enemy_hp - cur_enemy_hp).max(0) as f32;
            let hero_dmg = (self.prev_hero_hp - cur_hero_hp).max(0) as f32;
            // Normalize by per-unit average HP so 10x HP doesn't kill the signal
            let hp_reward = (enemy_dmg - hero_dmg) / self.avg_unit_hp.max(1.0);
            self.prev_hero_hp = cur_hero_hp;
            self.prev_enemy_hp = cur_enemy_hp;
            let event_r = self.pending_event_reward;
            self.pending_event_reward = 0.0;
            hp_reward + event_r
        } else {
            0.0
        };

        self.pending_units.clear();
        self.hero_pre_step.clear();
        self.steps_recorded_this_tick.clear();

        for &uid in &self.hero_ids.clone() {
            let unit = match self.sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                Some(u) => u,
                None => continue,
            };

            // Snapshot pre-step state for action-specific rewards
            if record {
                let nearest_enemy_dist = self.sim.units.iter()
                    .filter(|e| e.team == bevy_game::ai::core::Team::Enemy && e.hp > 0)
                    .map(|e| bevy_game::ai::core::distance(unit.position, e.position))
                    .fold(f32::MAX, f32::min);
                self.hero_pre_step.push(HeroPreStepState {
                    unit_id: uid,
                    position: unit.position,
                    hp: unit.hp,
                    nearest_enemy_dist,
                    move_dir: 0,     // filled after GPU result
                    combat_type: 0,  // filled after GPU result
                });
            }

            if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                continue;
            }

            let t_ext = std::time::Instant::now();
            let mask_arr = action_mask(&self.sim, uid);
            let mask_vec: Vec<bool> = mask_arr.to_vec();
            let gs_v2 = extract_game_state_v2(&self.sim, unit);
            let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
            p_extract.fetch_add(t_ext.elapsed().as_nanos() as u64, Ordering::Relaxed);

            // Build combat mask
            let has_enemies = gs_v2.entity_types.iter().any(|&t| t == 1);
            let mut combat_mask_vec = vec![false; 10];
            combat_mask_vec[0] = has_enemies;
            combat_mask_vec[1] = true;
            for idx in 0..n_abilities {
                if mask_arr[3 + idx] {
                    combat_mask_vec[2 + idx] = true;
                }
            }

            // Build ability CLS
            let ability_cls_for_req: Vec<Option<Vec<f32>>> = (0..MAX_ABILITIES)
                .map(|i| {
                    if i < n_abilities && unit.abilities[i].cooldown_remaining_ms == 0 && mask_arr[3 + i] {
                        self.cls_cache.get(&(uid, i)).cloned()
                    } else {
                        None
                    }
                })
                .collect();

            let t_ser = std::time::Instant::now();
            let req = InferenceRequest {
                entities: gs_v2.entities.clone(),
                entity_types: gs_v2.entity_types.clone(),
                threats: gs_v2.threats.clone(),
                positions: gs_v2.positions.clone(),
                combat_mask: combat_mask_vec,
                ability_cls: ability_cls_for_req,
            };
            p_serialize.fetch_add(t_ser.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let t_sub = std::time::Instant::now();
            let token = gpu.submit(req)?;
            p_submit.fetch_add(t_sub.elapsed().as_nanos() as u64, Ordering::Relaxed);

            self.pending_units.push(PendingUnit {
                unit_id: uid,
                token,
                gs_v2,
                mask_vec,
                n_abilities,
                step_reward: step_r,
                resolved: false,
            });
        }

        self.phase = if self.pending_units.is_empty() {
            // No heroes needed GPU — just step directly
            SimPhase::NeedsTick
        } else {
            SimPhase::WaitingGpu
        };

        // If no pending, step sim immediately
        if self.pending_units.is_empty() {
            self.step_sim();
        }

        Ok(())
    }

    /// Poll pending GPU results. Returns true if all done and sim was stepped.
    fn poll_gpu(
        &mut self,
        gpu: &bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient,
    ) -> Result<bool, String> {
        use bevy_game::ai::core::UnitIntent;
        use bevy_game::ai::core::ability_eval::extract_game_state;
        use bevy_game::ai::core::self_play::actions::{
            move_dir_to_intent, combat_action_to_intent, build_token_infos,
        };

        let record = self.tick % self.step_interval == 0;
        let mut all_done = true;

        for pu in &mut self.pending_units {
            if pu.resolved { continue; }
            match gpu.try_recv(pu.token) {
                Ok(Some(result)) => {
                    pu.resolved = true;
                    let uid = pu.unit_id;
                    let move_dir = result.move_dir as usize;
                    let combat_type = result.combat_type as usize;
                    let target_idx = result.target_idx as usize;

                    let move_intent = move_dir_to_intent(move_dir, uid, &self.sim);
                    let token_infos = build_token_infos(
                        &self.sim, uid, &pu.gs_v2.entity_types, &pu.gs_v2.positions,
                    );
                    let combat_intent = combat_action_to_intent(
                        combat_type, target_idx, uid, &self.sim, &token_infos,
                    );

                    let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
                        combat_intent
                    } else {
                        move_intent
                    };

                    self.intents.retain(|i| i.unit_id != uid);
                    self.intents.push(UnitIntent { unit_id: uid, action: final_intent });

                    if record {
                        if let Some(unit) = self.sim.units.iter().find(|u| u.id == uid) {
                            let game_state = extract_game_state(&self.sim, unit);
                            let composite_lp = result.lp_move + result.lp_combat + result.lp_pointer;
                            let step_idx = self.steps.len();
                            self.steps.push(RlStep {
                                tick: self.tick,
                                unit_id: uid,
                                game_state: game_state.to_vec(),
                                action: combat_type,
                                log_prob: composite_lp,
                                mask: pu.mask_vec.clone(),
                                step_reward: pu.step_reward,
                                entities: Some(pu.gs_v2.entities.clone()),
                                entity_types: Some(pu.gs_v2.entity_types.clone()),
                                threats: Some(pu.gs_v2.threats.clone()),
                                positions: Some(pu.gs_v2.positions.clone()),
                                action_type: Some(combat_type),
                                target_idx: Some(target_idx),
                                move_dir: Some(move_dir),
                                combat_type: Some(combat_type),
                                lp_move: Some(result.lp_move),
                                lp_combat: Some(result.lp_combat),
                                lp_pointer: Some(result.lp_pointer),
                            });
                            self.steps_recorded_this_tick.push(step_idx);
                            // Update pre-step snapshot with chosen action
                            if let Some(ps) = self.hero_pre_step.iter_mut().find(|p| p.unit_id == uid) {
                                ps.move_dir = move_dir;
                                ps.combat_type = combat_type;
                            }
                        }
                    }
                }
                Ok(None) => { all_done = false; }
                Err(e) => {
                    eprintln!("GPU inference error for unit {}: {e}", pu.unit_id);
                    pu.resolved = true;
                }
            }
        }

        if all_done {
            self.pending_units.clear();
            self.step_sim();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Advance the sim by one tick after all intents are resolved.
    fn step_sim(&mut self) {
        use bevy_game::ai::core::{step, Team, FIXED_TICK_MS};
        use bevy_game::ai::core::distance;

        let (new_sim, events) = step(self.sim.clone(), &self.intents, FIXED_TICK_MS);

        for ev in &events {
            if let bevy_game::ai::core::SimEvent::UnitDied { unit_id, .. } = ev {
                if let Some(dead_unit) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                    if dead_unit.team == Team::Enemy {
                        self.pending_event_reward += 0.3 / self.initial_enemy_count.max(1.0);
                    } else if dead_unit.team == Team::Hero {
                        self.pending_event_reward -= 0.4 / self.initial_hero_count.max(1.0);
                    }
                }
            }
        }

        // Compute action-specific rewards for steps recorded this tick
        for &step_idx in &self.steps_recorded_this_tick {
            let uid = self.steps[step_idx].unit_id;
            if let Some(pre) = self.hero_pre_step.iter().find(|p| p.unit_id == uid) {
                let mut action_reward: f32 = 0.0;

                if let Some(post_unit) = new_sim.units.iter().find(|u| u.id == uid) {
                    let post_nearest_enemy = new_sim.units.iter()
                        .filter(|e| e.team == Team::Enemy && e.hp > 0)
                        .map(|e| distance(post_unit.position, e.position))
                        .fold(f32::MAX, f32::min);

                    // --- Approach reward: closing distance to nearest enemy ---
                    // Scale: approaching by ~50 units (attack range) gives ~0.1 reward
                    if pre.nearest_enemy_dist < f32::MAX && post_nearest_enemy < f32::MAX {
                        let dist_delta = pre.nearest_enemy_dist - post_nearest_enemy;
                        action_reward += 0.002 * dist_delta; // +reward for closing, -for retreating
                    }

                    // --- Engagement reward: being in attack range of enemy ---
                    // Encourage staying in combat range (~50-100 units)
                    if post_nearest_enemy < 100.0 {
                        action_reward += 0.01; // small bonus for being near enemies
                    }

                    // --- Damage dealt reward (per-hero, from HP loss on enemies) ---
                    // This complements the global HP differential with per-hero attribution
                    let dmg_dealt = (pre.hp - post_unit.hp.max(0)).max(0) as f32;
                    if dmg_dealt > 0.0 {
                        // Hero took damage — slight penalty already in global HP reward
                    }
                }

                // --- Combat action reward: penalize hold when enemies exist ---
                // combat_type: 0=attack, 1=hold, 2+=ability
                if pre.combat_type == 1 && pre.nearest_enemy_dist < 150.0 {
                    action_reward -= 0.02; // holding near enemies is bad
                }

                // --- Attack/ability reward: bonus for choosing combat ---
                if pre.combat_type == 0 || pre.combat_type >= 2 {
                    action_reward += 0.01; // encourage attacking/using abilities
                }

                self.steps[step_idx].step_reward += action_reward;
            }
        }
        self.steps_recorded_this_tick.clear();

        self.sim = new_sim;
        self.tick += 1;
        self.phase = SimPhase::NeedsTick;
    }

    /// Check if episode is over.
    fn is_done(&self) -> bool {
        use bevy_game::ai::core::Team;
        let heroes_alive = self.sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = self.sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        enemies_alive == 0 || heroes_alive == 0 || self.tick >= self.max_ticks
    }

    fn into_episode(self) -> RlEpisode {
        use bevy_game::ai::core::Team;
        let heroes_alive = self.sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = self.sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();

        let (outcome, reward) = if enemies_alive == 0 {
            ("Victory".to_string(), 1.0)
        } else if heroes_alive == 0 {
            ("Defeat".to_string(), -1.0)
        } else {
            let hero_frac = hp_fraction(&self.sim, Team::Hero);
            let enemy_frac = hp_fraction(&self.sim, Team::Enemy);
            let shaped = (enemy_frac - hero_frac).clamp(-1.0, 1.0) * 0.5;
            ("Timeout".to_string(), shaped)
        };

        RlEpisode {
            scenario: self.scenario_name,
            outcome,
            reward,
            ticks: self.sim.tick,
            unit_abilities: self.unit_abilities,
            unit_ability_names: self.unit_ability_names,
            steps: self.steps,
        }
    }
}

/// Run GPU-multiplexed episode generation: multiple sims per thread, non-blocking inference.
fn run_gpu_multiplexed(
    gpu: &std::sync::Arc<bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient>,
    scenarios: &[bevy_game::scenario::ScenarioFile],
    episode_tasks: &[(usize, usize)],
    threads: usize,
    sims_per_thread: usize,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    temperature: f32,
    step_interval: u64,
    max_ticks_override: Option<u64>,
    registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> Vec<RlEpisode> {
    use crossbeam_channel::{bounded, Sender, Receiver};
    use std::sync::atomic::{AtomicU64, Ordering};

    // Task queue: prefill a channel with all (scenario_idx, episode_idx)
    let (task_tx, task_rx): (Sender<(usize, usize)>, Receiver<(usize, usize)>) =
        bounded(episode_tasks.len());
    for &task in episode_tasks {
        task_tx.send(task).unwrap();
    }
    drop(task_tx); // close sender so receivers know when empty

    // Result collection
    let results: std::sync::Arc<std::sync::Mutex<Vec<RlEpisode>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(episode_tasks.len())));

    let scenarios_arc = std::sync::Arc::new(scenarios.to_vec());

    // Global profiling counters (nanoseconds)
    let prof_intent_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_extract_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_submit_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_poll_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_step_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_record_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_ticks = std::sync::Arc::new(AtomicU64::new(0));
    let prof_polls = std::sync::Arc::new(AtomicU64::new(0));
    let prof_poll_misses = std::sync::Arc::new(AtomicU64::new(0));
    let prof_gpu_wait_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_make_sim_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_serialize_ns = std::sync::Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..threads).map(|_| {
        let gpu = gpu.clone();
        let task_rx = task_rx.clone();
        let results = results.clone();
        let scenarios = scenarios_arc.clone();
        let tok = tokenizer.clone();
        let reg_data: Option<std::sync::Arc<bevy_game::ai::core::ability_transformer::EmbeddingRegistry>> =
            registry.map(|r| std::sync::Arc::new(r.clone()));

        let p_intent = prof_intent_ns.clone();
        let p_extract = prof_extract_ns.clone();
        let p_submit = prof_submit_ns.clone();
        let p_poll = prof_poll_ns.clone();
        let p_step = prof_step_ns.clone();
        let p_record = prof_record_ns.clone();
        let p_ticks = prof_ticks.clone();
        let p_polls = prof_polls.clone();
        let p_poll_misses = prof_poll_misses.clone();
        let p_gpu_wait = prof_gpu_wait_ns.clone();
        let p_make_sim = prof_make_sim_ns.clone();
        let p_serialize = prof_serialize_ns.clone();

        std::thread::spawn(move || {
            let reg_ref = reg_data.as_deref();
            let mut active: Vec<ActiveSim> = Vec::with_capacity(sims_per_thread);

            // Fill initial batch
            while active.len() < sims_per_thread {
                match task_rx.try_recv() {
                    Ok((si, ei)) => {
                        let t0 = std::time::Instant::now();
                        if let Some(asim) = make_active_sim(
                            &scenarios[si], si, ei, max_ticks_override,
                            temperature, step_interval, &tok, reg_ref,
                        ) {
                            active.push(asim);
                        }
                        p_make_sim.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    Err(_) => break,
                }
            }

            while !active.is_empty() {
                let mut completed_indices = Vec::new();

                for (i, asim) in active.iter_mut().enumerate() {
                    match asim.phase {
                        SimPhase::NeedsTick => {
                            if asim.is_done() {
                                completed_indices.push(i);
                                continue;
                            }
                            p_ticks.fetch_add(1, Ordering::Relaxed);
                            if let Err(e) = asim.prepare_and_submit_profiled(
                                &gpu, &p_intent, &p_extract, &p_submit, &p_serialize,
                            ) {
                                eprintln!("GPU submit error: {e}");
                                completed_indices.push(i);
                            }
                        }
                        SimPhase::WaitingGpu => {
                            let t0 = std::time::Instant::now();
                            p_polls.fetch_add(1, Ordering::Relaxed);
                            match asim.poll_gpu(&gpu) {
                                Ok(true) => {
                                    p_poll.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                                    // Stepped — check done
                                    if asim.is_done() {
                                        completed_indices.push(i);
                                    }
                                }
                                Ok(false) => {
                                    p_poll_misses.fetch_add(1, Ordering::Relaxed);
                                    p_gpu_wait.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    eprintln!("GPU poll error: {e}");
                                    completed_indices.push(i);
                                }
                            }
                        }
                    }
                }

                // Remove completed (reverse order to preserve indices)
                completed_indices.sort_unstable();
                completed_indices.dedup();
                let mut result_batch = Vec::new();
                for &idx in completed_indices.iter().rev() {
                    let done = active.swap_remove(idx);
                    result_batch.push(done.into_episode());
                }
                if !result_batch.is_empty() {
                    results.lock().unwrap().extend(result_batch);
                }

                // Refill from queue
                while active.len() < sims_per_thread {
                    match task_rx.try_recv() {
                        Ok((si, ei)) => {
                            let t0 = std::time::Instant::now();
                            if let Some(asim) = make_active_sim(
                                &scenarios[si], si, ei, max_ticks_override,
                                temperature, step_interval, &tok, reg_ref,
                            ) {
                                active.push(asim);
                            }
                            p_make_sim.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        Err(_) => break,
                    }
                }

                // Park until next batch completes instead of busy-spinning
                if active.iter().all(|s| s.phase == SimPhase::WaitingGpu) {
                    let epoch = gpu.batch_epoch();
                    gpu.wait_for_batch(epoch);
                }
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    // Print profiling summary
    let ticks = prof_ticks.load(Ordering::Relaxed).max(1);
    let polls = prof_polls.load(Ordering::Relaxed).max(1);
    let poll_misses = prof_poll_misses.load(Ordering::Relaxed);
    let intent_ms = prof_intent_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let extract_ms = prof_extract_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let submit_ms = prof_submit_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let poll_ms = prof_poll_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let step_ms = prof_step_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let record_ms = prof_record_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let gpu_wait_ms = prof_gpu_wait_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let make_sim_ms = prof_make_sim_ns.load(Ordering::Relaxed) as f64 / 1e6;
    let serialize_ms = prof_serialize_ns.load(Ordering::Relaxed) as f64 / 1e6;

    eprintln!("\n--- GPU Multiplexed Profiling ({} ticks, {} polls, {} poll misses) ---", ticks, polls, poll_misses);
    eprintln!("  make_sim:    {:>8.1}ms total, {:>6.3}ms/tick", make_sim_ms, make_sim_ms / ticks as f64);
    eprintln!("  intents:     {:>8.1}ms total, {:>6.3}ms/tick", intent_ms, intent_ms / ticks as f64);
    eprintln!("  extract:     {:>8.1}ms total, {:>6.3}ms/tick", extract_ms, extract_ms / ticks as f64);
    eprintln!("  serialize:   {:>8.1}ms total, {:>6.3}ms/tick", serialize_ms, serialize_ms / ticks as f64);
    eprintln!("  submit:      {:>8.1}ms total, {:>6.3}ms/tick", submit_ms, submit_ms / ticks as f64);
    eprintln!("  poll (hit):  {:>8.1}ms total, {:>6.3}ms/poll", poll_ms, poll_ms / (polls - poll_misses).max(1) as f64);
    eprintln!("  gpu_wait:    {:>8.1}ms total ({} misses)", gpu_wait_ms, poll_misses);
    eprintln!("  step_sim:    {:>8.1}ms total, {:>6.3}ms/tick", step_ms, step_ms / ticks as f64);
    eprintln!("  record:      {:>8.1}ms total, {:>6.3}ms/tick", record_ms, record_ms / ticks as f64);

    std::sync::Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}

fn run_single_episode(
    scenario_file: &bevy_game::scenario::ScenarioFile,
    si: usize,
    ei: usize,
    max_ticks_override: Option<u64>,
    is_v3: bool,
    policy: &Policy,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    temperature: f32,
    step_interval: u64,
    student_weights: &Option<std::sync::Arc<super::training::StudentWeights>>,
    ability_eval_weights: &Option<std::sync::Arc<bevy_game::ai::core::ability_eval::AbilityEvalWeights>>,
    registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    enemy_policy: Option<&Policy>,
    enemy_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> RlEpisode {
    use bevy_game::scenario::{run_scenario_to_state, run_scenario_to_state_with_room};

    let cfg = &scenario_file.scenario;
    let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);

    let (sim, mut squad_ai, grid_nav) = if is_v3 {
        let (s, ai, nav) = run_scenario_to_state_with_room(cfg);
        (s, ai, Some(nav))
    } else {
        let (s, ai) = run_scenario_to_state(cfg);
        (s, ai, None)
    };

    if matches!(policy, Policy::Combined) {
        if let Some(ref w) = *ability_eval_weights {
            squad_ai.ability_eval_weights = Some((**w).clone());
        }
    }

    let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;

    run_rl_episode(
        sim, squad_ai, &cfg.name, max_ticks,
        policy, tokenizer,
        temperature, seed, step_interval,
        student_weights,
        grid_nav, registry,
        enemy_policy, enemy_registry,
    )
}

fn make_active_sim(
    scenario_file: &bevy_game::scenario::ScenarioFile,
    si: usize,
    ei: usize,
    max_ticks_override: Option<u64>,
    temperature: f32,
    step_interval: u64,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> Option<ActiveSim> {
    use bevy_game::scenario::run_scenario_to_state_with_room;

    let cfg = &scenario_file.scenario;
    let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);
    let (sim, squad_ai, nav) = run_scenario_to_state_with_room(cfg);
    let mut sim = sim;
    sim.grid_nav = Some(nav);
    let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;

    Some(ActiveSim::new(
        sim, squad_ai, cfg.name.clone(), max_ticks,
        seed, step_interval, temperature, tokenizer, registry,
    ))
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

    let policy = if args.random_policy {
        Policy::Random
    } else if let Some(ref shm_path) = args.gpu_shm {
        use bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient;
        match GpuInferenceClient::new(shm_path, 1024, 1) {
            Ok(client) => Policy::GpuServer(client),
            Err(e) => { eprintln!("Failed to open GPU SHM at {shm_path}: {e}"); return ExitCode::from(1); }
        }
    } else if args.policy == "combined" {
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
        Policy::ActorCriticV4(_) => "actor-critic-v4 (dual-head)",
        Policy::GpuServer(_) => "gpu-server (dual-head)",
        Policy::Legacy(_) => "legacy (bootstrap)",
        Policy::Combined => "combined (ability-eval + squad AI)",
        Policy::Random => "random",
    };
    let is_v3 = matches!(&policy, Policy::ActorCriticV3(_) | Policy::ActorCriticV4(_) | Policy::GpuServer(_));

    let tokenizer = AbilityTokenizer::new();

    let paths: Vec<_> = args.path.iter().flat_map(|p| collect_toml_paths(p)).collect();
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

    // Load embedding registry if provided
    let registry = if let Some(ref reg_path) = args.embedding_registry {
        match bevy_game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded embedding registry: {} abilities, hash={}",
                    r.len(), r.model_hash);
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    // Load enemy policy for self-play
    let enemy_policy: Option<Policy> = if let Some(ref ew_path) = args.enemy_weights {
        match Policy::load(ew_path) {
            Ok(p) => {
                eprintln!("Loaded enemy policy from {}", ew_path.display());
                Some(p)
            }
            Err(e) => { eprintln!("Failed to load enemy policy: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };
    let enemy_registry = if let Some(ref reg_path) = args.enemy_registry {
        match bevy_game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded enemy embedding registry: {} abilities", r.len());
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load enemy registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let ability_eval_ref = &ability_eval_weights;
    let student_ref = &student_weights;
    let registry_ref = registry.as_ref();
    let enemy_policy_ref = enemy_policy.as_ref();
    let enemy_registry_ref = enemy_registry.as_ref();
    let step_interval = args.step_interval;
    let temperature = args.temperature;
    let max_ticks_override = args.max_ticks;
    let episode_tasks: Vec<(usize, usize)> = scenarios.iter().enumerate()
        .flat_map(|(si, _)| (0..args.episodes as usize).map(move |ei| (si, ei)))
        .collect();

    // Use multiplexed GPU path when sims_per_thread > 1 and policy is GpuServer
    let episodes: Vec<RlEpisode> = if args.sims_per_thread > 1 {
        if let Policy::GpuServer(ref gpu) = policy {
            eprintln!("GPU multiplexed: {} threads × {} sims/thread", threads, args.sims_per_thread);
            run_gpu_multiplexed(
                gpu, &scenarios, &episode_tasks,
                threads, args.sims_per_thread,
                tokenizer_ref, temperature, step_interval,
                max_ticks_override, registry_ref,
            )
        } else {
            eprintln!("Warning: --sims-per-thread > 1 only works with --gpu-shm, falling back to par_iter");
            pool.install(|| {
                episode_tasks.par_iter().map(|&(si, ei)| {
                    run_single_episode(
                        &scenarios[si], si, ei, max_ticks_override, is_v3,
                        policy_ref, tokenizer_ref, temperature, step_interval,
                        student_ref, ability_eval_ref, registry_ref,
                        enemy_policy_ref, enemy_registry_ref,
                    )
                }).collect()
            })
        }
    } else {
        pool.install(|| {
            episode_tasks.par_iter().map(|&(si, ei)| {
                run_single_episode(
                    &scenarios[si], si, ei, max_ticks_override, is_v3,
                    policy_ref, tokenizer_ref, temperature, step_interval,
                    student_ref, ability_eval_ref, registry_ref,
                    enemy_policy_ref, enemy_registry_ref,
                )
            }).collect()
        })
    };

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
    let is_v3 = matches!(&policy, Policy::ActorCriticV3(_) | Policy::ActorCriticV4(_));

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

    // Load embedding registry if provided
    let registry = if let Some(ref reg_path) = args.embedding_registry {
        match bevy_game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded embedding registry: {} abilities, hash={}",
                    r.len(), r.model_hash);
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    // Load enemy policy for self-play eval
    let enemy_policy: Option<Policy> = if let Some(ref ew_path) = args.enemy_weights {
        match Policy::load(ew_path) {
            Ok(p) => {
                eprintln!("Loaded enemy policy from {}", ew_path.display());
                Some(p)
            }
            Err(e) => { eprintln!("Failed to load enemy policy: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };
    let enemy_registry = if let Some(ref reg_path) = args.enemy_registry {
        match bevy_game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded enemy embedding registry: {} abilities", r.len());
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load enemy registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let registry_ref = registry.as_ref();
    let enemy_policy_ref = enemy_policy.as_ref();
    let enemy_registry_ref = enemy_registry.as_ref();
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
            registry_ref,
            enemy_policy_ref,
            enemy_registry_ref,
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
