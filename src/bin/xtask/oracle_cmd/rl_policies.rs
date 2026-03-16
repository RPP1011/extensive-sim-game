//! Policy-specific application functions and helpers for RL episode runner.

use super::transformer_rl::{Policy, RlEpisode, RlStep, lcg_f32, masked_softmax_sample, apply_action_mask, MAX_ABILITIES};

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_random_policy(
    sim: &bevy_game::ai::core::SimState,
    unit: &bevy_game::ai::core::UnitState,
    uid: u32,
    mask_arr: &[bool],
    mask_vec: &[bool],
    scenario_action_mask: Option<&str>,
    record: bool,
    step_r: f32,
    tick: u64,
    rng: &mut u64,
    intents: &mut Vec<bevy_game::ai::core::UnitIntent>,
    steps: &mut Vec<RlStep>,
    vis_map: Option<&bevy_game::ai::goap::spatial::VisibilityMap>,
    nav: Option<&bevy_game::ai::pathing::GridNav>,
) {
    use bevy_game::ai::core::{Team, UnitIntent};
    use bevy_game::ai::core::ability_eval::{extract_game_state, extract_game_state_v2, extract_game_state_v2_spatial};
    use bevy_game::ai::core::self_play::actions::{
        move_dir_to_intent, combat_action_to_intent, build_token_infos,
        NUM_MOVE_DIRS, COMBAT_TYPE_ATTACK, COMBAT_TYPE_HOLD,
    };

    let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
    let move_dir = (lcg_f32(rng) * NUM_MOVE_DIRS as f32) as usize;
    let has_enemies = sim.units.iter().any(|e| e.team == Team::Enemy && e.hp > 0);
    let n_combat = 2 + n_abilities;
    let mut combat_mask = vec![false; n_combat];
    combat_mask[COMBAT_TYPE_ATTACK] = has_enemies;
    combat_mask[COMBAT_TYPE_HOLD] = true;
    for idx in 0..n_abilities {
        if mask_arr[3 + idx] { combat_mask[2 + idx] = true; }
    }
    apply_action_mask(&mut combat_mask, scenario_action_mask);
    let valid_combat: Vec<usize> = combat_mask.iter().enumerate()
        .filter(|(_, &v)| v).map(|(i, _)| i).collect();
    let combat_type = valid_combat[(lcg_f32(rng) * valid_combat.len() as f32) as usize % valid_combat.len()];

    // Only extract full game state on recording ticks
    let gs_v2 = if record {
        Some(match (vis_map, nav) {
            (Some(vm), Some(n)) => extract_game_state_v2_spatial(sim, unit, vm, n),
            _ => extract_game_state_v2(sim, unit),
        })
    } else {
        None
    };

    let n_entities = gs_v2.as_ref().map_or(sim.units.len(), |g| g.entities.len());
    let target_idx = if n_entities > 0 {
        (lcg_f32(rng) * n_entities as f32) as usize % n_entities
    } else { 0 };

    // For intent generation, we only need token_infos when we have gs_v2
    // Otherwise fall back to simple nearest-enemy targeting
    if let Some(ref gs) = gs_v2 {
        let token_infos = build_token_infos(sim, uid, &gs.entity_types, &gs.positions);
        let move_intent = move_dir_to_intent(move_dir, uid, sim);
        let combat_intent = combat_action_to_intent(combat_type, target_idx, uid, sim, &token_infos);
        let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
            combat_intent
        } else {
            move_intent
        };
        intents.retain(|i| i.unit_id != uid);
        intents.push(UnitIntent { unit_id: uid, action: final_intent });
    } else {
        // Non-recording tick: lightweight random intent without full extraction
        let move_intent = move_dir_to_intent(move_dir, uid, sim);
        // For combat, just pick nearest enemy as target
        let combat_intent = if combat_type == COMBAT_TYPE_HOLD {
            bevy_game::ai::core::IntentAction::Hold
        } else if combat_type == COMBAT_TYPE_ATTACK {
            if let Some(target) = sim.units.iter()
                .filter(|e| e.team == Team::Enemy && e.hp > 0)
                .min_by(|a, b| {
                    let da = (a.position.x - unit.position.x).powi(2) + (a.position.y - unit.position.y).powi(2);
                    let db = (b.position.x - unit.position.x).powi(2) + (b.position.y - unit.position.y).powi(2);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                bevy_game::ai::core::IntentAction::Attack { target_id: target.id }
            } else {
                bevy_game::ai::core::IntentAction::Hold
            }
        } else {
            // Ability usage
            let ability_idx = combat_type - 2;
            if let Some(target) = sim.units.iter()
                .filter(|e| e.team == Team::Enemy && e.hp > 0)
                .next() {
                bevy_game::ai::core::IntentAction::UseAbility {
                    ability_index: ability_idx,
                    target: bevy_game::ai::effects::AbilityTarget::Unit(target.id),
                }
            } else {
                bevy_game::ai::core::IntentAction::Hold
            }
        };
        let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
            combat_intent
        } else {
            move_intent
        };
        intents.retain(|i| i.unit_id != uid);
        intents.push(UnitIntent { unit_id: uid, action: final_intent });
    }

    if let Some(gs_v2) = gs_v2 {
        let game_state = extract_game_state(sim, unit);
        steps.push(RlStep {
            tick, unit_id: uid,
            game_state: game_state.to_vec(),
            action: combat_type, log_prob: 0.0,
            mask: mask_vec.to_vec(), step_reward: step_r,
            entities: Some(gs_v2.entities), entity_types: Some(gs_v2.entity_types),
            threats: Some(gs_v2.threats), positions: Some(gs_v2.positions),
            zones: Some(gs_v2.zones),
            action_type: Some(combat_type), target_idx: Some(target_idx),
            move_dir: Some(move_dir), combat_type: Some(combat_type),
            lp_move: Some(0.0), lp_combat: Some(0.0),
            lp_pointer: Some(0.0),
            aggregate_features: if gs_v2.aggregate_features.is_empty() { None } else { Some(gs_v2.aggregate_features) },
            target_move_pos: None,
            teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_v5_policy(
    ac: &bevy_game::ai::core::ability_transformer::ActorCriticWeightsV5,
    sim: &bevy_game::ai::core::SimState,
    unit: &bevy_game::ai::core::UnitState,
    uid: u32,
    gs_v2: &bevy_game::ai::core::ability_eval::GameStateV2,
    ability_cls_refs: &[Option<&[f32]>],
    mask_arr: &[bool],
    mask_vec: &[bool],
    scenario_action_mask: Option<&str>,
    temperature: f32,
    record: bool,
    step_r: f32,
    tick: u64,
    rng: &mut u64,
    intents: &mut Vec<bevy_game::ai::core::UnitIntent>,
    steps: &mut Vec<RlStep>,
) {
    use bevy_game::ai::core::UnitIntent;
    use bevy_game::ai::core::ability_eval::extract_game_state;
    use bevy_game::ai::core::self_play::actions::{
        move_dir_to_intent, combat_action_to_intent, build_token_infos,
        NUM_MOVE_DIRS, COMBAT_TYPE_ATTACK, COMBAT_TYPE_HOLD,
    };

    let ent_refs: Vec<&[f32]> = gs_v2.entities.iter().map(|e| e.as_slice()).collect();
    let type_refs: Vec<usize> = gs_v2.entity_types.iter().map(|&t| t as usize).collect();
    let threat_refs: Vec<&[f32]> = gs_v2.threats.iter().map(|t| t.as_slice()).collect();
    let pos_refs: Vec<&[f32]> = gs_v2.positions.iter().map(|p| p.as_slice()).collect();
    let zone_refs: Vec<&[f32]> = gs_v2.zones.iter().map(|z| z.as_slice()).collect();
    let agg_ref: Option<&[f32]> = if gs_v2.aggregate_features.is_empty() {
        None
    } else {
        Some(gs_v2.aggregate_features.as_slice())
    };

    let ent_state = ac.encode_entities_with_zones(&ent_refs, &type_refs, &threat_refs, &pos_refs, &zone_refs, agg_ref);
    let dual = ac.dual_head_logits(&ent_state, ability_cls_refs);

    let move_mask = vec![true; NUM_MOVE_DIRS];
    let (move_dir, move_lp) = masked_softmax_sample(&dual.move_logits, &move_mask, temperature, rng);

    let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
    let has_enemies = ent_state.type_ids.iter().any(|&t| t == 1);
    let mut combat_mask = vec![false; dual.combat_logits.len()];
    combat_mask[COMBAT_TYPE_ATTACK] = has_enemies;
    combat_mask[COMBAT_TYPE_HOLD] = true;
    for idx in 0..n_abilities {
        if mask_arr[3 + idx] { combat_mask[2 + idx] = true; }
    }
    apply_action_mask(&mut combat_mask, scenario_action_mask);
    let (combat_type, combat_lp) = masked_softmax_sample(&dual.combat_logits, &combat_mask, temperature, rng);

    let (target_idx, target_lp, combat_intent) = match combat_type {
        t if t == COMBAT_TYPE_ATTACK => {
            let atk_mask: Vec<bool> = dual.attack_ptr.iter().map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
            let (idx, lp) = masked_softmax_sample(&dual.attack_ptr, &atk_mask, temperature, rng);
            let ti = build_token_infos(sim, uid, &gs_v2.entity_types, &gs_v2.positions);
            (idx, lp, combat_action_to_intent(combat_type, idx, uid, sim, &ti))
        }
        t if t == COMBAT_TYPE_HOLD => {
            (0, 0.0, bevy_game::ai::core::IntentAction::Hold)
        }
        t @ 2..=9 => {
            let ab_idx = t - 2;
            if let Some(Some(ab_ptr)) = dual.ability_ptrs.get(ab_idx) {
                let ab_mask: Vec<bool> = ab_ptr.iter().map(|&v| v > f32::NEG_INFINITY + 1.0).collect();
                let (idx, lp) = masked_softmax_sample(ab_ptr, &ab_mask, temperature, rng);
                let ti = build_token_infos(sim, uid, &gs_v2.entity_types, &gs_v2.positions);
                (idx, lp, combat_action_to_intent(combat_type, idx, uid, sim, &ti))
            } else {
                (0, 0.0, bevy_game::ai::core::IntentAction::Hold)
            }
        }
        _ => (0, 0.0, bevy_game::ai::core::IntentAction::Hold),
    };

    let move_intent = move_dir_to_intent(move_dir, uid, sim);
    let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
        combat_intent
    } else {
        move_intent
    };

    intents.retain(|i| i.unit_id != uid);
    intents.push(UnitIntent { unit_id: uid, action: final_intent });

    if record {
        let game_state = extract_game_state(sim, unit);
        let composite_lp = move_lp + combat_lp + target_lp;
        steps.push(RlStep {
            tick, unit_id: uid,
            game_state: game_state.to_vec(),
            action: combat_type, log_prob: composite_lp,
            mask: mask_vec.to_vec(), step_reward: step_r,
            entities: Some(gs_v2.entities.clone()),
            entity_types: Some(gs_v2.entity_types.clone()),
            threats: Some(gs_v2.threats.clone()),
            positions: Some(gs_v2.positions.clone()),
            zones: Some(gs_v2.zones.clone()),
            action_type: Some(combat_type), target_idx: Some(target_idx),
            move_dir: Some(move_dir), combat_type: Some(combat_type),
            lp_move: Some(move_lp), lp_combat: Some(combat_lp),
            lp_pointer: Some(target_lp),
            aggregate_features: Some(gs_v2.aggregate_features.clone()),
            target_move_pos: None,
            teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
        });
    }
}

// ---------------------------------------------------------------------------
// Drill objective checking
// ---------------------------------------------------------------------------

pub(crate) fn check_drill_objective(
    obj: Option<&bevy_game::scenario::ObjectiveDef>,
    sim: &bevy_game::ai::core::SimState,
    heroes_alive: usize,
    enemies_alive: usize,
) -> Option<(String, f32)> {
    let obj = obj?;
    match obj.objective_type.as_str() {
        "reach_position" => {
            if let (Some(target), Some(radius)) = (obj.position, obj.radius) {
                for unit in sim.units.iter().filter(|u| u.team == bevy_game::ai::core::Team::Hero && u.hp > 0) {
                    let dx = unit.position.x - target[0];
                    let dy = unit.position.y - target[1];
                    if (dx * dx + dy * dy).sqrt() <= radius {
                        return Some(("Victory".to_string(), 1.0));
                    }
                }
            }
            None
        }
        "reach_entity" => {
            let radius = obj.radius.unwrap_or(1.5);
            for hero in sim.units.iter().filter(|u| u.team == bevy_game::ai::core::Team::Hero && u.hp > 0) {
                for enemy in sim.units.iter().filter(|u| u.team == bevy_game::ai::core::Team::Enemy && u.hp > 0) {
                    let dx = hero.position.x - enemy.position.x;
                    let dy = hero.position.y - enemy.position.y;
                    if (dx * dx + dy * dy).sqrt() <= radius {
                        return Some(("Victory".to_string(), 1.0));
                    }
                }
            }
            None
        }
        "kill_all" | "kill_target" => {
            if enemies_alive == 0 { Some(("Victory".to_string(), 1.0)) } else { None }
        }
        "survive" => {
            if heroes_alive == 0 { Some(("Defeat".to_string(), -1.0)) } else { None }
        }
        _ => {
            if enemies_alive == 0 { Some(("Victory".to_string(), 1.0)) } else { None }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn hp_fraction(sim: &bevy_game::ai::core::SimState, team: bevy_game::ai::core::Team) -> f32 {
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

pub(crate) fn run_single_episode(
    pre: &super::transformer_rl::PrecomputedScenario,
    si: usize,
    ei: usize,
    _is_v3: bool,
    policy: &Policy,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    temperature: f32,
    step_interval: u64,
    registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    enemy_policy: Option<&Policy>,
    enemy_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> RlEpisode {
    let mut sim = pre.sim.clone();
    let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;
    sim.rng_state = seed;

    let grid_nav = sim.grid_nav.clone();
    let squad_ai = pre.squad_ai.clone();

    super::rl_episode::run_rl_episode(
        sim, squad_ai, &pre.scenario_name, pre.max_ticks,
        policy, tokenizer,
        temperature, seed, step_interval,
        grid_nav, registry,
        enemy_policy, enemy_registry,
        pre.objective.as_ref(),
        pre.action_mask.as_deref(),
        &pre.behavior_trees,
    )
}
