use std::cmp::Ordering;
use std::collections::HashMap;

use crate::ai::core::{
    distance, move_away, move_towards, position_at_range, run_replay, step, IntentAction,
    ReplayResult, SimState, SimVec2, Team, UnitIntent,
};
use crate::ai::roles::{default_roles, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormationMode {
    Hold,
    Advance,
    Retreat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquadBlackboard {
    pub focus_target: Option<u32>,
    pub mode: FormationMode,
}

#[derive(Debug, Clone, Copy)]
struct RoleProfile {
    preferred_range_min: f32,
    preferred_range_max: f32,
    leash_distance: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct UnitMemory {
    anchor_position: SimVec2,
    sticky_target: Option<u32>,
    lock_ticks: u32,
}

#[derive(Debug, Clone)]
pub struct SquadAiState {
    role_by_unit: HashMap<u32, Role>,
    memory: HashMap<u32, UnitMemory>,
    blackboard_by_team: HashMap<Team, SquadBlackboard>,
    eval_every_ticks: u64,
}

impl SquadAiState {
    pub fn new(initial: &SimState, role_by_unit: HashMap<u32, Role>) -> Self {
        let memory = initial
            .units
            .iter()
            .map(|u| {
                (
                    u.id,
                    UnitMemory {
                        anchor_position: u.position,
                        sticky_target: None,
                        lock_ticks: 0,
                    },
                )
            })
            .collect();

        let blackboard_by_team = HashMap::from([
            (
                Team::Hero,
                SquadBlackboard {
                    focus_target: None,
                    mode: FormationMode::Hold,
                },
            ),
            (
                Team::Enemy,
                SquadBlackboard {
                    focus_target: None,
                    mode: FormationMode::Hold,
                },
            ),
        ]);

        Self {
            role_by_unit,
            memory,
            blackboard_by_team,
            eval_every_ticks: 5,
        }
    }

    fn role_for(&self, unit_id: u32) -> Role {
        *self.role_by_unit.get(&unit_id).unwrap_or(&Role::Dps)
    }

    pub fn evaluate_blackboards_if_needed(&mut self, state: &SimState) {
        if state.tick % self.eval_every_ticks != 0 {
            return;
        }
        self.blackboard_by_team
            .insert(Team::Hero, evaluate_blackboard(state, Team::Hero));
        self.blackboard_by_team
            .insert(Team::Enemy, evaluate_blackboard(state, Team::Enemy));
    }

    fn blackboard(&self, team: Team) -> SquadBlackboard {
        self.blackboard_by_team
            .get(&team)
            .copied()
            .unwrap_or(SquadBlackboard {
                focus_target: None,
                mode: FormationMode::Hold,
            })
    }
}

fn role_profile(role: Role) -> RoleProfile {
    match role {
        Role::Tank => RoleProfile {
            preferred_range_min: 0.8,
            preferred_range_max: 1.5,
            leash_distance: 14.0,
        },
        Role::Dps => RoleProfile {
            preferred_range_min: 1.3,
            preferred_range_max: 2.3,
            leash_distance: 17.0,
        },
        Role::Healer => RoleProfile {
            preferred_range_min: 1.8,
            preferred_range_max: 3.8,
            leash_distance: 19.0,
        },
    }
}

fn evaluate_blackboard(state: &SimState, team: Team) -> SquadBlackboard {
    let allies = alive_by_team(state, team);
    let enemies = alive_by_team(state, opposite(team));

    let focus_target = enemies.iter().copied().min_by(|a, b| {
        let hpa = state
            .units
            .iter()
            .find(|u| u.id == *a)
            .map_or(i32::MAX, |u| u.hp);
        let hpb = state
            .units
            .iter()
            .find(|u| u.id == *b)
            .map_or(i32::MAX, |u| u.hp);
        hpa.cmp(&hpb).then_with(|| a.cmp(b))
    });

    let ally_avg = average_hp_pct(state, &allies);
    let enemy_avg = average_hp_pct(state, &enemies);

    let mode =
        if allies.len() * 2 < enemies.len() || (ally_avg < 0.35 && enemy_avg > ally_avg + 0.10) {
            FormationMode::Retreat
        } else if ally_avg > enemy_avg + 0.12 {
            FormationMode::Advance
        } else {
            FormationMode::Hold
        };

    SquadBlackboard { focus_target, mode }
}

fn average_hp_pct(state: &SimState, ids: &[u32]) -> f32 {
    if ids.is_empty() {
        return 0.0;
    }
    let sum = ids
        .iter()
        .filter_map(|id| state.units.iter().find(|u| u.id == *id))
        .map(|u| u.hp.max(0) as f32 / u.max_hp.max(1) as f32)
        .sum::<f32>();
    sum / ids.len() as f32
}

fn opposite(team: Team) -> Team {
    match team {
        Team::Hero => Team::Enemy,
        Team::Enemy => Team::Hero,
    }
}

fn alive_by_team(state: &SimState, team: Team) -> Vec<u32> {
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

pub fn generate_intents(state: &SimState, ai: &mut SquadAiState, dt_ms: u32) -> Vec<UnitIntent> {
    ai.evaluate_blackboards_if_needed(state);

    let mut intents = Vec::new();
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();

    for unit_id in ids {
        let Some(unit_idx) = state.units.iter().position(|u| u.id == unit_id) else {
            continue;
        };
        let unit = &state.units[unit_idx];
        let role = ai.role_for(unit_id);
        let profile = role_profile(role);
        let board = ai.blackboard(unit.team);

        let (anchor, sticky_target, lock_ticks) = {
            let mem = ai.memory.get(&unit_id).copied().unwrap_or_default();
            (mem.anchor_position, mem.sticky_target, mem.lock_ticks)
        };

        let leash = match board.mode {
            FormationMode::Advance => profile.leash_distance + 3.0,
            FormationMode::Retreat => profile.leash_distance - 2.0,
            FormationMode::Hold => profile.leash_distance,
        };
        if distance(unit.position, anchor) > leash {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::MoveTo { position: anchor },
            });
            continue;
        }

        if role == Role::Healer {
            if let Some(heal_intent) = healer_intent(state, unit_id, board.mode, dt_ms) {
                intents.push(heal_intent);
                continue;
            }
        }

        let enemies = alive_by_team(state, opposite(unit.team));
        if enemies.is_empty() {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        }

        let target = choose_target(
            state,
            unit_id,
            role,
            board,
            sticky_target,
            lock_ticks,
            &enemies,
        );
        let Some(target_id) = target else {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        };

        if let Some(mem) = ai.memory.get_mut(&unit_id) {
            if sticky_target == Some(target_id) {
                mem.lock_ticks = lock_ticks.saturating_sub(1);
            } else {
                mem.sticky_target = Some(target_id);
                mem.lock_ticks = 4;
            }
        }

        let action = choose_action(state, unit_id, target_id, role, board.mode, dt_ms);
        intents.push(UnitIntent { unit_id, action });
    }

    intents
}

fn healer_intent(
    state: &SimState,
    healer_id: u32,
    mode: FormationMode,
    dt_ms: u32,
) -> Option<UnitIntent> {
    let healer = state.units.iter().find(|u| u.id == healer_id)?;
    if healer.heal_amount <= 0 {
        return None;
    }

    let allies = alive_by_team(state, healer.team);
    let triage = allies
        .iter()
        .filter_map(|ally_id| {
            let ally = state.units.iter().find(|u| u.id == *ally_id)?;
            let missing = ally.max_hp - ally.hp;
            if missing <= 0 {
                return None;
            }
            let hp_pct = ally.hp as f32 / ally.max_hp.max(1) as f32;
            Some((*ally_id, missing, hp_pct))
        })
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let Some((ally_id, _missing, hp_pct)) = triage else {
        return None;
    };

    let threshold = match mode {
        FormationMode::Advance => 0.38,
        FormationMode::Hold => 0.48,
        FormationMode::Retreat => 0.62,
    };

    if hp_pct > threshold {
        return None;
    }

    let ally = state.units.iter().find(|u| u.id == ally_id)?;
    let dist = distance(healer.position, ally.position);
    if healer.heal_cooldown_remaining_ms == 0 && dist <= healer.heal_range {
        return Some(UnitIntent {
            unit_id: healer_id,
            action: IntentAction::CastHeal { target_id: ally_id },
        });
    }

    if dist > healer.heal_range {
        let max_step = healer.move_speed_per_sec * (dt_ms as f32 / 1000.0);
        let desired_pos =
            position_at_range(healer.position, ally.position, healer.heal_range * 0.9);
        let next_pos = move_towards(healer.position, desired_pos, max_step);
        return Some(UnitIntent {
            unit_id: healer_id,
            action: IntentAction::MoveTo { position: next_pos },
        });
    }

    None
}

fn choose_target(
    state: &SimState,
    unit_id: u32,
    role: Role,
    board: SquadBlackboard,
    sticky_target: Option<u32>,
    lock_ticks: u32,
    enemies: &[u32],
) -> Option<u32> {
    if lock_ticks > 0 {
        if let Some(sticky) = sticky_target {
            if enemies.contains(&sticky) {
                return Some(sticky);
            }
        }
    }

    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let focus = board.focus_target;

    enemies.iter().copied().max_by(|a, b| {
        let score_a = target_score(state, unit, role, *a, focus);
        let score_b = target_score(state, unit, role, *b, focus);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.cmp(a))
    })
}

fn target_score(
    state: &SimState,
    unit: &crate::ai::core::UnitState,
    role: Role,
    target_id: u32,
    focus: Option<u32>,
) -> f32 {
    let Some(target) = state.units.iter().find(|u| u.id == target_id) else {
        return f32::MIN;
    };

    let dist = crate::ai::core::distance(unit.position, target.position);
    let hp_factor = (target.max_hp - target.hp).max(0) as f32 * 0.2;
    let focus_bonus = if focus == Some(target_id) { 7.0 } else { 0.0 };
    let role_bias = match role {
        Role::Tank => -dist * 1.0,
        Role::Dps => -dist * 1.8,
        Role::Healer => -dist * 0.8,
    };
    hp_factor + focus_bonus + role_bias
}

fn choose_action(
    state: &SimState,
    unit_id: u32,
    target_id: u32,
    role: Role,
    mode: FormationMode,
    dt_ms: u32,
) -> IntentAction {
    let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
        return IntentAction::Hold;
    };
    let Some(target) = state.units.iter().find(|u| u.id == target_id) else {
        return IntentAction::Hold;
    };

    let dist = distance(unit.position, target.position);
    if unit.control_duration_ms > 0
        && unit.control_cooldown_remaining_ms == 0
        && dist <= unit.control_range
        && target.control_remaining_ms == 0
        && mode != FormationMode::Retreat
    {
        return IntentAction::CastControl { target_id };
    }
    if unit.ability_cooldown_remaining_ms == 0
        && unit.ability_damage > 0
        && dist <= unit.ability_range
        && target.hp > unit.attack_damage
        && mode != FormationMode::Retreat
    {
        return IntentAction::CastAbility { target_id };
    }

    if dist <= unit.attack_range {
        return IntentAction::Attack { target_id };
    }

    let profile = role_profile(role);
    let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
    let range_center = (profile.preferred_range_min + profile.preferred_range_max) * 0.5;

    let desired_pos = match mode {
        FormationMode::Advance => {
            position_at_range(unit.position, target.position, range_center * 0.85)
        }
        FormationMode::Hold => position_at_range(unit.position, target.position, range_center),
        FormationMode::Retreat => move_away(unit.position, target.position, range_center * 0.7),
    };

    let next_pos = move_towards(unit.position, desired_pos, max_step);
    IntentAction::MoveTo { position: next_pos }
}

#[derive(Debug, Clone)]
pub struct Phase3Run {
    pub replay: ReplayResult,
    pub board_history: Vec<HashMap<Team, SquadBlackboard>>,
}

pub fn sample_phase3_party_state(seed: u64) -> SimState {
    crate::ai::roles::sample_phase2_party_state(seed)
}

pub fn generate_scripted_intents(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    roles: HashMap<u32, Role>,
) -> (Vec<Vec<UnitIntent>>, Vec<HashMap<Team, SquadBlackboard>>) {
    let mut ai = SquadAiState::new(initial, roles);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);
    let mut board_history = Vec::with_capacity(ticks as usize);

    for _ in 0..ticks {
        let intents = generate_intents(&state, &mut ai, dt_ms);
        board_history.push(ai.blackboard_by_team.clone());
        script.push(intents.clone());
        let (new_state, _events) = step(state, &intents, dt_ms);
        state = new_state;
    }

    (script, board_history)
}

pub fn run_phase3_sample(seed: u64, ticks: u32, dt_ms: u32) -> Phase3Run {
    let initial = sample_phase3_party_state(seed);
    let (script, board_history) =
        generate_scripted_intents(&initial, ticks, dt_ms, default_roles());
    let replay = run_replay(initial, &script, ticks, dt_ms);
    Phase3Run {
        replay,
        board_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::FIXED_TICK_MS;
    use std::collections::HashSet;

    fn run_phase3_with_seed(seed: u64) -> Phase3Run {
        run_phase3_sample(seed, 280, FIXED_TICK_MS)
    }

    #[test]
    fn phase3_is_deterministic() {
        let a = run_phase3_with_seed(23);
        let b = run_phase3_with_seed(23);
        assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
        assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
        assert_eq!(
            a.replay.per_tick_state_hashes,
            b.replay.per_tick_state_hashes
        );
    }

    #[test]
    fn blackboard_focus_drives_target_coherence() {
        let initial = sample_phase3_party_state(23);
        let (script, history) =
            generate_scripted_intents(&initial, 120, FIXED_TICK_MS, default_roles());

        let mut focused_actions = 0_u32;
        let mut total_offensive = 0_u32;
        for (tick, intents) in script.iter().enumerate() {
            let boards = &history[tick];
            for intent in intents {
                let Some(unit) = initial.units.iter().find(|u| u.id == intent.unit_id) else {
                    continue;
                };
                let focus = boards.get(&unit.team).and_then(|b| b.focus_target);
                let target = match intent.action {
                    IntentAction::Attack { target_id } => Some(target_id),
                    IntentAction::CastAbility { target_id } => Some(target_id),
                    _ => None,
                };
                if let Some(target_id) = target {
                    total_offensive += 1;
                    if Some(target_id) == focus {
                        focused_actions += 1;
                    }
                }
            }
        }

        assert!(total_offensive > 0);
        assert!((focused_actions as f32 / total_offensive as f32) >= 0.55);
    }

    #[test]
    fn squad_mode_changes_over_time() {
        let run = run_phase3_with_seed(23);
        let mut hero_modes = run
            .board_history
            .iter()
            .filter_map(|m| m.get(&Team::Hero).map(|b| b.mode))
            .collect::<Vec<_>>();
        hero_modes.dedup();
        assert!(hero_modes.len() >= 2);
    }

    #[test]
    fn phase3_competent_party_behavior() {
        let run = run_phase3_with_seed(23);
        assert!(run.replay.metrics.winner.is_some());
        assert_eq!(run.replay.metrics.invariant_violations, 0);
        assert!(run.replay.metrics.casts_completed + run.replay.metrics.heals_completed > 0);
    }

    #[test]
    fn phase3_regression_snapshot() {
        let run = run_phase3_with_seed(23);
        assert_eq!(run.replay.event_log_hash, 0xdb95_9f8e_2e84_480a);
        assert_eq!(run.replay.final_state_hash, 0x673b_d6b7_96db_9a40);
        assert_eq!(run.replay.metrics.winner, Some(Team::Hero));
        assert_eq!(
            run.replay.metrics.final_hp_by_unit,
            vec![(1, 145), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        );
        assert_eq!(run.replay.metrics.casts_completed, 38);
        assert_eq!(run.replay.metrics.heals_completed, 2);
        assert_eq!(run.replay.metrics.invariant_violations, 0);
    }

    #[test]
    fn blackboard_updates_only_on_eval_cadence() {
        let run = run_phase3_with_seed(23);
        for tick in 1..run.board_history.len() {
            if tick % 5 != 0 {
                assert_eq!(run.board_history[tick], run.board_history[tick - 1]);
            }
        }
    }

    #[test]
    fn phase3_multi_seed_metrics_bands() {
        let seeds = [23_u64, 29, 31, 37, 41, 43];
        let mut hero_wins = 0_u32;
        let mut focus_ratio_sum = 0.0_f32;

        for seed in seeds {
            let initial = sample_phase3_party_state(seed);
            let (script, history) =
                generate_scripted_intents(&initial, 200, FIXED_TICK_MS, default_roles());
            let replay = run_replay(initial.clone(), &script, 200, FIXED_TICK_MS);
            if replay.metrics.winner == Some(Team::Hero) {
                hero_wins += 1;
            }
            assert_eq!(replay.metrics.invariant_violations, 0);

            let mut focused_actions = 0_u32;
            let mut offensive_actions = 0_u32;
            for (tick, intents) in script.iter().enumerate() {
                let boards = &history[tick];
                for intent in intents {
                    let Some(unit) = initial.units.iter().find(|u| u.id == intent.unit_id) else {
                        continue;
                    };
                    let focus = boards.get(&unit.team).and_then(|b| b.focus_target);
                    let target = match intent.action {
                        IntentAction::Attack { target_id } => Some(target_id),
                        IntentAction::CastAbility { target_id } => Some(target_id),
                        _ => None,
                    };
                    if let Some(target_id) = target {
                        offensive_actions += 1;
                        if Some(target_id) == focus {
                            focused_actions += 1;
                        }
                    }
                }
            }
            if offensive_actions > 0 {
                focus_ratio_sum += focused_actions as f32 / offensive_actions as f32;
            }
        }

        let hero_win_rate = hero_wins as f32 / 6.0;
        let avg_focus_ratio = focus_ratio_sum / 6.0;
        assert!(hero_win_rate >= 0.35 && hero_win_rate <= 1.0);
        assert!(avg_focus_ratio >= 0.50);
    }

    #[test]
    fn tie_break_target_selection_is_deterministic() {
        let mut state = sample_phase3_party_state(55);
        {
            let enemy4 = state.units.iter_mut().find(|u| u.id == 4).unwrap();
            enemy4.hp = 100;
            enemy4.position = crate::ai::core::sim_vec2(8.0, 0.0);
        }
        {
            let enemy5 = state.units.iter_mut().find(|u| u.id == 5).unwrap();
            enemy5.hp = 100;
            enemy5.position = crate::ai::core::sim_vec2(8.0, 0.0);
        }

        let mut ai = SquadAiState::new(&state, default_roles());
        let intents_a = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        let intents_b = generate_intents(&state, &mut ai, FIXED_TICK_MS);

        let pick_targets = |intents: &[UnitIntent]| -> HashSet<u32> {
            intents
                .iter()
                .filter_map(|intent| match intent.action {
                    IntentAction::Attack { target_id } => Some(target_id),
                    IntentAction::CastAbility { target_id } => Some(target_id),
                    _ => None,
                })
                .collect()
        };

        assert_eq!(pick_targets(&intents_a), pick_targets(&intents_b));
    }

    #[test]
    fn no_enemies_produces_hold_for_entire_squad() {
        let mut state = sample_phase3_party_state(57);
        state.units.retain(|u| u.team == Team::Hero);
        let mut ai = SquadAiState::new(&state, default_roles());
        let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        assert!(intents
            .iter()
            .all(|intent| matches!(intent.action, IntentAction::Hold)));
    }

    #[test]
    fn hash_changes_with_small_parameter_mutation() {
        let baseline = run_phase3_with_seed(23);
        let mut initial = sample_phase3_party_state(23);
        let hero_tank = initial.units.iter_mut().find(|u| u.id == 1).unwrap();
        hero_tank.attack_damage += 1;
        let (script, _) = generate_scripted_intents(&initial, 280, FIXED_TICK_MS, default_roles());
        let mutated = run_replay(initial, &script, 280, FIXED_TICK_MS);
        assert_ne!(baseline.replay.event_log_hash, mutated.event_log_hash);
    }

    #[test]
    fn fuzz_invariants_hold_for_phase3_seed_sweep() {
        for seed in 70_u64..80_u64 {
            let run = run_phase3_with_seed(seed);
            assert_eq!(run.replay.metrics.invariant_violations, 0);
            assert_eq!(run.replay.metrics.dead_source_attack_intents, 0);
            assert_eq!(run.replay.per_tick_state_hashes.len(), 280);
        }
    }
}
