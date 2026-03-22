use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use crate::sim::{
    position_at_range, run_replay, step, IntentAction, ReplayResult, SimState, Team, UnitIntent,
    UnitStore,
};

#[derive(Debug, Clone)]
pub struct UtilityAiConfig {
    pub distance_weight: f32,
    pub damage_weight: f32,
    pub overkill_penalty_weight: f32,
    pub stickiness_bonus: f32,
    pub target_lock_ticks: u32,
}

impl Default for UtilityAiConfig {
    fn default() -> Self {
        Self {
            distance_weight: 4.0,
            damage_weight: 1.3,
            overkill_penalty_weight: 1.6,
            stickiness_bonus: 7.0,
            target_lock_ticks: 3,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct UnitMemory {
    sticky_target: Option<u32>,
    retarget_lock_ticks: u32,
}

#[derive(Debug, Clone, Default)]
pub struct UtilityAiState {
    memory: HashMap<u32, UnitMemory>,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    action: IntentAction,
    target_id: u32,
    score: f32,
    action_rank: u8,
}

pub fn generate_intents(
    state: &SimState,
    ai_state: &mut UtilityAiState,
    cfg: &UtilityAiConfig,
    dt_ms: u32,
) -> Vec<UnitIntent> {
    let mut intents = Vec::new();
    let mut acting_ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    acting_ids.sort_unstable();

    for unit_id in acting_ids {
        let Some(unit_idx) = state.units.iter().position(|u| u.id == unit_id) else {
            continue;
        };
        let unit = &state.units[unit_idx];

        let mut enemy_ids = state
            .units
            .iter()
            .filter(|other| other.hp > 0 && other.team != unit.team)
            .map(|other| other.id)
            .collect::<Vec<_>>();
        enemy_ids.sort_unstable();
        if enemy_ids.is_empty() {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        }

        let memory = ai_state.memory.entry(unit_id).or_default();
        if let Some(sticky) = memory.sticky_target {
            let sticky_alive = enemy_ids.contains(&sticky);
            if !sticky_alive {
                memory.sticky_target = None;
                memory.retarget_lock_ticks = 0;
            }
        }

        let target_candidates = if memory.retarget_lock_ticks > 0 {
            if let Some(sticky) = memory.sticky_target {
                vec![sticky]
            } else {
                enemy_ids
            }
        } else {
            enemy_ids
        };

        let mut candidates = Vec::new();
        for target_id in target_candidates {
            let Some(target_idx) = state.units.iter().position(|u| u.id == target_id) else {
                continue;
            };
            let target = &state.units[target_idx];
            let distance = crate::sim::distance(unit.position, target.position);
            let sticky = memory.sticky_target == Some(target_id);
            let stickiness = if sticky { cfg.stickiness_bonus } else { 0.0 };

            let attack_overkill = (unit.attack_damage - target.hp).max(0) as f32;
            let attack_score = 18.0 + unit.attack_damage as f32 * cfg.damage_weight
                - attack_overkill * cfg.overkill_penalty_weight
                - distance * cfg.distance_weight
                + stickiness;
            candidates.push(Candidate {
                action: IntentAction::Attack { target_id },
                target_id,
                score: attack_score,
                action_rank: 1,
            });

            let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
            let dist_to_attack_range = (distance - unit.attack_range).max(0.0);
            let post_move_dist = (dist_to_attack_range - max_step).max(0.0);
            let closure = dist_to_attack_range - post_move_dist;
            let move_score = 10.0 + closure * 5.5 - post_move_dist * 4.0 + stickiness;
            let move_target_pos =
                position_at_range(unit.position, target.position, unit.attack_range * 0.9);
            candidates.push(Candidate {
                action: IntentAction::MoveTo {
                    position: move_target_pos,
                },
                target_id,
                score: move_score,
                action_rank: 2,
            });

            let ability_ready = unit.ability_cooldown_remaining_ms == 0 && unit.ability_damage > 0;
            let ability_in_range = distance <= unit.ability_range;
            if ability_ready && ability_in_range {
                let ability_overkill = (unit.ability_damage - target.hp).max(0) as f32;
                let ability_score = 28.0 + unit.ability_damage as f32 * (cfg.damage_weight + 0.4)
                    - ability_overkill * (cfg.overkill_penalty_weight + 0.5)
                    - distance * (cfg.distance_weight * 0.7)
                    + stickiness;
                candidates.push(Candidate {
                    action: IntentAction::CastAbility { target_id },
                    target_id,
                    score: ability_score,
                    action_rank: 0,
                });
            }
        }

        let selected = candidates.into_iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.action_rank.cmp(&a.action_rank))
                .then_with(|| b.target_id.cmp(&a.target_id))
        });

        let chosen = selected.map_or(IntentAction::Hold, |c| c.action);
        let chosen_target = match chosen {
            IntentAction::Attack { target_id } => Some(target_id),
            IntentAction::CastAbility { target_id } => Some(target_id),
            _ => None,
        };

        if let Some(target_id) = chosen_target {
            if memory.sticky_target == Some(target_id) {
                if memory.retarget_lock_ticks > 0 {
                    memory.retarget_lock_ticks -= 1;
                }
            } else {
                memory.sticky_target = Some(target_id);
                memory.retarget_lock_ticks = cfg.target_lock_ticks;
            }
        }

        intents.push(UnitIntent {
            unit_id,
            action: chosen,
        });
    }

    intents
}

pub fn sample_phase1_skirmish_state(seed: u64) -> SimState {
    let mut state = crate::sim::sample_duel_state(seed);
    state.units = UnitStore::new(vec![
        crate::sim::UnitState {
            id: 1,
            team: Team::Hero,
            hp: 110,
            max_hp: 110,
            position: crate::sim::sim_vec2(-1.0, -0.6),
            move_speed_per_sec: 4.2,
            attack_damage: 12,
            attack_range: 1.4,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 24,
            ability_range: 2.0,
            ability_cooldown_ms: 3_000,
            ability_cast_time_ms: 500,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },
        crate::sim::UnitState {
            id: 2,
            team: Team::Hero,
            hp: 90,
            max_hp: 90,
            position: crate::sim::sim_vec2(-3.0, 0.5),
            move_speed_per_sec: 4.6,
            attack_damage: 10,
            attack_range: 1.3,
            attack_cooldown_ms: 600,
            attack_cast_time_ms: 250,
            cooldown_remaining_ms: 0,
            ability_damage: 18,
            ability_range: 1.8,
            ability_cooldown_ms: 2_500,
            ability_cast_time_ms: 400,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },
        crate::sim::UnitState {
            id: 3,
            team: Team::Enemy,
            hp: 105,
            max_hp: 105,
            position: crate::sim::sim_vec2(8.5, -0.6),
            move_speed_per_sec: 3.8,
            attack_damage: 11,
            attack_range: 1.2,
            attack_cooldown_ms: 800,
            attack_cast_time_ms: 350,
            cooldown_remaining_ms: 0,
            ability_damage: 20,
            ability_range: 1.7,
            ability_cooldown_ms: 3_200,
            ability_cast_time_ms: 550,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },
        crate::sim::UnitState {
            id: 4,
            team: Team::Enemy,
            hp: 95,
            max_hp: 95,
            position: crate::sim::sim_vec2(10.5, 0.5),
            move_speed_per_sec: 4.0,
            attack_damage: 9,
            attack_range: 1.4,
            attack_cooldown_ms: 650,
            attack_cast_time_ms: 280,
            cooldown_remaining_ms: 0,
            ability_damage: 16,
            ability_range: 2.2,
            ability_cooldown_ms: 2_800,
            ability_cast_time_ms: 450,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },
    ]);
    state.units.sort_by_key(|u| u.id);
    state
}

pub fn generate_scripted_intents(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
) -> Vec<Vec<UnitIntent>> {
    let cfg = UtilityAiConfig::default();
    let mut ai_state = UtilityAiState::default();
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);

    for _ in 0..ticks {
        let intents = generate_intents(&state, &mut ai_state, &cfg, dt_ms);
        script.push(intents.clone());
        let (new_state, _) = step(state, &intents, dt_ms);
        state = new_state;
    }
    script
}

pub fn run_phase1_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let initial = sample_phase1_skirmish_state(seed);
    let script = generate_scripted_intents(&initial, ticks, dt_ms);
    run_replay(initial, &script, ticks, dt_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::FIXED_TICK_MS;

    #[test]
    fn phase1_sample_is_deterministic() {
        let a = run_phase1_sample(11, 160, FIXED_TICK_MS);
        let b = run_phase1_sample(11, 160, FIXED_TICK_MS);
        assert_eq!(a.event_log_hash, b.event_log_hash);
        assert_eq!(a.final_state_hash, b.final_state_hash);
        assert_eq!(a.per_tick_state_hashes, b.per_tick_state_hashes);
    }

    #[test]
    fn ability_candidate_is_never_emitted_out_of_range() {
        let initial = sample_phase1_skirmish_state(5);
        let script = generate_scripted_intents(&initial, 40, FIXED_TICK_MS);
        let mut state = initial.clone();

        for intents in script {
            for intent in &intents {
                if let IntentAction::CastAbility { target_id } = intent.action {
                    let src = state.units.iter().find(|u| u.id == intent.unit_id).unwrap();
                    let tgt = state.units.iter().find(|u| u.id == target_id).unwrap();
                    let distance = crate::sim::distance(src.position, tgt.position);
                    assert!(distance <= src.ability_range);
                }
            }
            let (new_state, _) = step(state, &intents, FIXED_TICK_MS);
            state = new_state;
        }
    }

    #[test]
    fn stickiness_reduces_retargeting() {
        let initial = sample_phase1_skirmish_state(9);
        let script = generate_scripted_intents(&initial, 60, FIXED_TICK_MS);
        let mut retargets = 0_u32;
        let mut last_target_by_unit: HashMap<u32, u32> = HashMap::new();

        for intents in script {
            for intent in intents {
                let maybe_target = match intent.action {
                    IntentAction::Attack { target_id } => Some(target_id),
                    IntentAction::CastAbility { target_id } => Some(target_id),
                    _ => None,
                };
                if let Some(target_id) = maybe_target {
                    if let Some(last_target) = last_target_by_unit.get(&intent.unit_id) {
                        if *last_target != target_id {
                            retargets += 1;
                        }
                    }
                    last_target_by_unit.insert(intent.unit_id, target_id);
                }
            }
        }

        assert!(retargets <= 16);
    }

    #[test]
    fn phase1_fight_is_competent_in_small_skirmish() {
        let result = run_phase1_sample(11, 200, FIXED_TICK_MS);
        assert!(result.metrics.casts_started > 0);
        assert!(result.metrics.casts_completed > 0);
        assert_eq!(result.metrics.invariant_violations, 0);
        assert!(result.metrics.winner.is_some());
    }

    #[test]
    fn phase1_regression_snapshot() {
        let result = run_phase1_sample(11, 200, FIXED_TICK_MS);
        assert_eq!(result.event_log_hash, 0x4153_af9a_8241_744b);
        assert_eq!(result.final_state_hash, 0xf227_f31f_c6ad_debc);
        assert_eq!(result.metrics.winner, Some(Team::Hero));
        assert_eq!(
            result.metrics.final_hp_by_unit,
            vec![(1, 3), (2, 71), (3, 0), (4, 0)]
        );
        assert_eq!(result.metrics.casts_started, 26);
        assert_eq!(result.metrics.casts_completed, 25);
        assert_eq!(result.metrics.invariant_violations, 0);
    }
}
