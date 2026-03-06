use std::collections::HashMap;

use crate::ai::control::{
    generate_scripted_intents as phase4_generate_scripted_intents, CcReservation,
};
use crate::ai::core::{
    distance, move_away, move_towards, position_at_range, run_replay, step, IntentAction,
    ReplayResult, SimState, UnitIntent,
};
use crate::ai::roles::{default_roles, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitMode {
    Aggressive,
    Defensive,
    Protector,
    Controller,
}

#[derive(Debug, Clone, Copy)]
pub struct PersonalityProfile {
    pub aggression: f32,
    pub risk_tolerance: f32,
    pub discipline: f32,
    pub control_bias: f32,
    pub altruism: f32,
    pub patience: f32,
}

impl PersonalityProfile {
    pub fn vanguard() -> Self {
        Self {
            aggression: 0.85,
            risk_tolerance: 0.75,
            discipline: 0.55,
            control_bias: 0.35,
            altruism: 0.35,
            patience: 0.40,
        }
    }

    pub fn guardian() -> Self {
        Self {
            aggression: 0.45,
            risk_tolerance: 0.30,
            discipline: 0.85,
            control_bias: 0.55,
            altruism: 0.80,
            patience: 0.75,
        }
    }

    pub fn tactician() -> Self {
        Self {
            aggression: 0.55,
            risk_tolerance: 0.45,
            discipline: 0.80,
            control_bias: 0.85,
            altruism: 0.50,
            patience: 0.85,
        }
    }
}

#[derive(Debug, Clone)]
struct UnitPersonalityState {
    mode: UnitMode,
    mode_lock_ticks: u32,
}

#[derive(Debug, Clone)]
pub struct PersonalityAiState {
    roles: HashMap<u32, Role>,
    personality_by_unit: HashMap<u32, PersonalityProfile>,
    unit_state: HashMap<u32, UnitPersonalityState>,
}

#[derive(Debug, Clone)]
pub struct Phase5Run {
    pub replay: ReplayResult,
    pub mode_history: Vec<Vec<(u32, UnitMode)>>,
}

pub fn sample_phase5_party_state(seed: u64) -> SimState {
    let mut state = crate::ai::control::sample_phase4_party_state(seed);
    for unit in &mut state.units {
        unit.attack_damage += 2;
        if unit.ability_damage > 0 {
            unit.ability_damage += 3;
        }
    }
    state
}

pub fn default_personalities() -> HashMap<u32, PersonalityProfile> {
    HashMap::from([
        (1, PersonalityProfile::guardian()),
        (2, PersonalityProfile::vanguard()),
        (3, PersonalityProfile::guardian()),
        (4, PersonalityProfile::tactician()),
        (5, PersonalityProfile::vanguard()),
        (6, PersonalityProfile::guardian()),
    ])
}

fn initial_mode_for_role(role: Role) -> UnitMode {
    match role {
        Role::Tank => UnitMode::Controller,
        Role::Dps => UnitMode::Aggressive,
        Role::Healer => UnitMode::Protector,
    }
}

impl PersonalityAiState {
    fn new(
        state: &SimState,
        roles: HashMap<u32, Role>,
        personalities: HashMap<u32, PersonalityProfile>,
    ) -> Self {
        let mut unit_state = HashMap::new();
        for unit in &state.units {
            let role = *roles.get(&unit.id).unwrap_or(&Role::Dps);
            unit_state.insert(
                unit.id,
                UnitPersonalityState {
                    mode: initial_mode_for_role(role),
                    mode_lock_ticks: 0,
                },
            );
        }
        Self {
            roles,
            personality_by_unit: personalities,
            unit_state,
        }
    }

    fn role(&self, unit_id: u32) -> Role {
        *self.roles.get(&unit_id).unwrap_or(&Role::Dps)
    }

    fn personality(&self, unit_id: u32) -> PersonalityProfile {
        *self
            .personality_by_unit
            .get(&unit_id)
            .unwrap_or(&PersonalityProfile::vanguard())
    }

    fn mode(&self, unit_id: u32) -> UnitMode {
        self.unit_state
            .get(&unit_id)
            .map(|s| s.mode)
            .unwrap_or(UnitMode::Aggressive)
    }

    fn maybe_update_mode(
        &mut self,
        state: &SimState,
        unit_id: u32,
        reservations: &[CcReservation],
    ) {
        let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
            return;
        };
        let role = self.role(unit_id);
        let p = self.personality(unit_id);
        let hp_pct = unit.hp.max(0) as f32 / unit.max_hp.max(1) as f32;

        let ally_min_hp = state
            .units
            .iter()
            .filter(|u| u.team == unit.team && u.hp > 0)
            .map(|u| u.hp as f32 / u.max_hp.max(1) as f32)
            .fold(1.0_f32, f32::min);

        let has_reservation = reservations.iter().any(|r| r.caster_id == unit_id);

        let lock = self
            .unit_state
            .get(&unit_id)
            .map_or(0, |s| s.mode_lock_ticks);
        if lock > 0 {
            if let Some(s) = self.unit_state.get_mut(&unit_id) {
                s.mode_lock_ticks -= 1;
            }
            return;
        }

        let defensive_threshold = 0.42 - p.risk_tolerance * 0.18;
        let protector_threshold = 0.55 + p.altruism * 0.20;
        let aggressive_threshold = 0.62 - p.patience * 0.15;

        let new_mode = if has_reservation && p.control_bias > 0.55 {
            UnitMode::Controller
        } else if hp_pct <= defensive_threshold {
            UnitMode::Defensive
        } else if role == Role::Healer && ally_min_hp <= protector_threshold {
            UnitMode::Protector
        } else if role != Role::Healer && hp_pct >= aggressive_threshold && p.aggression > 0.5 {
            UnitMode::Aggressive
        } else {
            match role {
                Role::Tank => UnitMode::Controller,
                Role::Dps => UnitMode::Aggressive,
                Role::Healer => UnitMode::Protector,
            }
        };

        if let Some(s) = self.unit_state.get_mut(&unit_id) {
            if s.mode != new_mode {
                s.mode = new_mode;
                s.mode_lock_ticks = (2.0 + p.discipline * 4.0).round() as u32;
            }
        }
    }
}

fn nearest_enemy(state: &SimState, unit_id: u32) -> Option<u32> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .min_by(|a, b| {
            let da = distance(unit.position, a.position);
            let db = distance(unit.position, b.position);
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        })
        .map(|u| u.id)
}

fn team_focus_target(state: &SimState, unit_id: u32) -> Option<u32> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .min_by(|a, b| a.hp.cmp(&b.hp).then_with(|| a.id.cmp(&b.id)))
        .map(|u| u.id)
}

fn focused_offensive_action(
    state: &SimState,
    unit_id: u32,
    target_id: u32,
    dt_ms: u32,
) -> Option<IntentAction> {
    let unit = state.units.iter().find(|u| u.id == unit_id && u.hp > 0)?;
    let target = state.units.iter().find(|u| u.id == target_id && u.hp > 0)?;
    let dist = distance(unit.position, target.position);

    if unit.control_duration_ms > 0
        && unit.control_cooldown_remaining_ms == 0
        && dist <= unit.control_range
        && target.control_remaining_ms == 0
    {
        return Some(IntentAction::CastControl { target_id });
    }
    if unit.ability_damage > 0
        && unit.ability_cooldown_remaining_ms == 0
        && dist <= unit.ability_range * 0.95
    {
        return Some(IntentAction::CastAbility { target_id });
    }
    if dist <= unit.attack_range {
        return Some(IntentAction::Attack { target_id });
    }

    let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
    let desired = position_at_range(unit.position, target.position, unit.attack_range * 0.9);
    let next = move_towards(unit.position, desired, max_step);
    Some(IntentAction::MoveTo { position: next })
}

fn weakest_ally_below_threshold(state: &SimState, unit_id: u32, threshold: f32) -> Option<u32> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == unit.team)
        .map(|u| (u.id, u.hp as f32 / u.max_hp.max(1) as f32))
        .filter(|(_, hp_pct)| *hp_pct <= threshold)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
}

fn apply_personality_overrides(
    state: &SimState,
    ai: &mut PersonalityAiState,
    base: &[UnitIntent],
    reservations: &[CcReservation],
    dt_ms: u32,
) -> (Vec<UnitIntent>, Vec<(u32, UnitMode)>) {
    let mut out = Vec::with_capacity(base.len());
    let mut mode_snapshot = Vec::with_capacity(base.len());

    for intent in base {
        let unit_id = intent.unit_id;
        ai.maybe_update_mode(state, unit_id, reservations);
        let mode = ai.mode(unit_id);
        let p = ai.personality(unit_id);
        mode_snapshot.push((unit_id, mode));

        let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
            out.push(*intent);
            continue;
        };

        let mut action = intent.action;
        match mode {
            UnitMode::Aggressive => {
                if matches!(action, IntentAction::Hold | IntentAction::MoveTo { .. }) {
                    if let Some(target_id) = nearest_enemy(state, unit_id) {
                        let target = state.units.iter().find(|u| u.id == target_id).unwrap();
                        let dist = distance(unit.position, target.position);
                        if dist <= unit.attack_range || p.aggression > 0.7 {
                            action = IntentAction::Attack { target_id };
                        }
                    }
                }
            }
            UnitMode::Defensive => {
                if let Some(target_id) = nearest_enemy(state, unit_id) {
                    let target = state.units.iter().find(|u| u.id == target_id).unwrap();
                    let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                    let away_pos = move_away(unit.position, target.position, max_step);
                    action = IntentAction::MoveTo { position: away_pos };
                }
            }
            UnitMode::Protector => {
                if unit.heal_amount > 0 {
                    let threshold = 0.65 + p.altruism * 0.2;
                    if let Some(ally_id) = weakest_ally_below_threshold(state, unit_id, threshold) {
                        let ally = state.units.iter().find(|u| u.id == ally_id).unwrap();
                        let dist = distance(unit.position, ally.position);
                        if unit.heal_cooldown_remaining_ms == 0 && dist <= unit.heal_range {
                            action = IntentAction::CastHeal { target_id: ally_id };
                        } else {
                            let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                            let desired = position_at_range(
                                unit.position,
                                ally.position,
                                unit.heal_range * 0.9,
                            );
                            let next = move_towards(unit.position, desired, max_step);
                            action = IntentAction::MoveTo { position: next };
                        }
                    }
                }
            }
            UnitMode::Controller => {
                let reservation = reservations.iter().find(|r| r.caster_id == unit_id);
                if p.control_bias >= 0.5 {
                    if let Some(res) = reservation {
                        if let Some(target) = state.units.iter().find(|u| u.id == res.target_id) {
                            let dist = distance(unit.position, target.position);
                            if unit.ability_cooldown_remaining_ms == 0 && dist <= unit.ability_range
                            {
                                action = IntentAction::CastAbility {
                                    target_id: res.target_id,
                                };
                            } else {
                                let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                                let desired = position_at_range(
                                    unit.position,
                                    target.position,
                                    unit.ability_range * 0.9,
                                );
                                let next = move_towards(unit.position, desired, max_step);
                                action = IntentAction::MoveTo { position: next };
                            }
                        }
                    }
                }
                if reservation.is_none() && matches!(action, IntentAction::CastAbility { .. }) {
                    // No reservation: default to DPS unless control-bias is very high.
                    if p.control_bias < 0.8 {
                        if let IntentAction::CastAbility { target_id } = action {
                            action = IntentAction::Attack { target_id };
                        }
                    }
                }
            }
        }

        let stalemate_break_tick = 120;
        if state.tick >= stalemate_break_tick {
            if let Some(focus_id) = team_focus_target(state, unit_id) {
                let ally_critical = state
                    .units
                    .iter()
                    .filter(|u| u.hp > 0 && u.team == unit.team)
                    .any(|u| (u.hp as f32 / u.max_hp.max(1) as f32) <= 0.30);
                if mode != UnitMode::Protector || !ally_critical {
                    if let Some(focus_action) =
                        focused_offensive_action(state, unit_id, focus_id, dt_ms)
                    {
                        action = focus_action;
                    }
                }
            }
        }

        out.push(UnitIntent { unit_id, action });
    }

    mode_snapshot.sort_by_key(|(unit_id, _)| *unit_id);
    (out, mode_snapshot)
}

pub fn generate_scripted_intents(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    roles: HashMap<u32, Role>,
    personalities: HashMap<u32, PersonalityProfile>,
) -> (Vec<Vec<UnitIntent>>, Vec<Vec<(u32, UnitMode)>>) {
    let (base_script, reservation_history, _windows) =
        phase4_generate_scripted_intents(initial, ticks, dt_ms, roles.clone());

    let mut ai = PersonalityAiState::new(initial, roles, personalities);
    let mut state = initial.clone();
    let mut final_script = Vec::with_capacity(ticks as usize);
    let mut mode_history = Vec::with_capacity(ticks as usize);

    for tick in 0..ticks as usize {
        let base = base_script.get(tick).cloned().unwrap_or_default();
        let reservations = reservation_history.get(tick).cloned().unwrap_or_default();
        let (intents, modes) =
            apply_personality_overrides(&state, &mut ai, &base, &reservations, dt_ms);
        final_script.push(intents.clone());
        mode_history.push(modes);
        let (new_state, _) = step(state, &intents, dt_ms);
        state = new_state;
    }

    (final_script, mode_history)
}

pub fn run_phase5_sample(seed: u64, ticks: u32, dt_ms: u32) -> Phase5Run {
    let initial = sample_phase5_party_state(seed);
    let roles = default_roles();
    let personalities = default_personalities();
    let (script, mode_history) =
        generate_scripted_intents(&initial, ticks, dt_ms, roles, personalities);
    let replay = run_replay(initial, &script, ticks, dt_ms);
    Phase5Run {
        replay,
        mode_history,
    }
}

pub fn run_phase5_with_personality_overrides(
    seed: u64,
    ticks: u32,
    dt_ms: u32,
    personality_by_unit: HashMap<u32, PersonalityProfile>,
) -> Phase5Run {
    let initial = sample_phase5_party_state(seed);
    let roles = default_roles();
    let (script, mode_history) =
        generate_scripted_intents(&initial, ticks, dt_ms, roles, personality_by_unit);
    let replay = run_replay(initial, &script, ticks, dt_ms);
    Phase5Run {
        replay,
        mode_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::FIXED_TICK_MS;
    use std::collections::HashSet;

    fn run_phase5(seed: u64) -> Phase5Run {
        run_phase5_sample(seed, 320, FIXED_TICK_MS)
    }

    fn all_vanguard() -> HashMap<u32, PersonalityProfile> {
        (1_u32..=6_u32)
            .map(|id| (id, PersonalityProfile::vanguard()))
            .collect()
    }

    fn all_guardian() -> HashMap<u32, PersonalityProfile> {
        (1_u32..=6_u32)
            .map(|id| (id, PersonalityProfile::guardian()))
            .collect()
    }

    fn all_tactician() -> HashMap<u32, PersonalityProfile> {
        (1_u32..=6_u32)
            .map(|id| (id, PersonalityProfile::tactician()))
            .collect()
    }

    fn hero_aggressive_enemy_defensive() -> HashMap<u32, PersonalityProfile> {
        HashMap::from([
            (1, PersonalityProfile::vanguard()),
            (2, PersonalityProfile::vanguard()),
            (3, PersonalityProfile::vanguard()),
            (4, PersonalityProfile::guardian()),
            (5, PersonalityProfile::guardian()),
            (6, PersonalityProfile::guardian()),
        ])
    }

    fn hero_controller_enemy_aggressive() -> HashMap<u32, PersonalityProfile> {
        HashMap::from([
            (1, PersonalityProfile::tactician()),
            (2, PersonalityProfile::guardian()),
            (3, PersonalityProfile::tactician()),
            (4, PersonalityProfile::vanguard()),
            (5, PersonalityProfile::vanguard()),
            (6, PersonalityProfile::vanguard()),
        ])
    }

    fn run_profile(seed: u64, profile: HashMap<u32, PersonalityProfile>) -> Phase5Run {
        run_phase5_with_personality_overrides(seed, 320, FIXED_TICK_MS, profile)
    }

    #[test]
    fn phase5_is_deterministic() {
        let a = run_phase5(31);
        let b = run_phase5(31);
        assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
        assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
    }

    #[test]
    fn mode_state_machine_transitions_exist() {
        let run = run_phase5(31);
        let mut changes = 0_u32;
        let mut prev: Option<Vec<(u32, UnitMode)>> = None;
        for snapshot in &run.mode_history {
            if let Some(p) = &prev {
                if p != snapshot {
                    changes += 1;
                }
            }
            prev = Some(snapshot.clone());
        }
        assert!(changes >= 2);
    }

    #[test]
    fn personalities_produce_different_outcomes() {
        let base = run_phase5(31);
        let alt = run_profile(31, all_vanguard());

        assert_ne!(base.replay.event_log_hash, alt.replay.event_log_hash);
    }

    #[test]
    fn phase5_competent_and_safe() {
        let run = run_phase5(31);
        let any_damage = run
            .replay
            .metrics
            .total_damage_by_unit
            .iter()
            .map(|(_, dmg)| *dmg)
            .sum::<i32>()
            > 0;
        assert!(any_damage);
        assert_eq!(run.replay.metrics.invariant_violations, 0);
        assert!(run.replay.metrics.casts_completed + run.replay.metrics.heals_completed > 0);
    }

    #[test]
    fn phase5_regression_snapshot() {
        let run = run_phase5(31);
        assert_eq!(run.replay.event_log_hash, 0x1609_d64c_eeae_0632);
        assert_eq!(run.replay.final_state_hash, 0xa90d_5423_a84b_0e9f);
        assert_eq!(run.replay.metrics.winner, Some(crate::ai::core::Team::Hero));
        assert_eq!(
            run.replay.metrics.final_hp_by_unit,
            vec![(1, 130), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        );
        assert_eq!(run.replay.metrics.invariant_violations, 0);
    }

    #[test]
    fn personality_matrix_produces_distinct_signatures() {
        let profiles = vec![
            ("vanguard", all_vanguard()),
            ("guardian", all_guardian()),
            ("tactician", all_tactician()),
            ("hero_aggr_enemy_def", hero_aggressive_enemy_defensive()),
            ("hero_ctrl_enemy_aggr", hero_controller_enemy_aggressive()),
        ];

        let mut seen = HashSet::new();
        for (_name, profile) in profiles {
            let run = run_profile(31, profile);
            assert_eq!(run.replay.metrics.invariant_violations, 0);
            seen.insert(run.replay.event_log_hash);
        }

        assert!(seen.len() >= 4);
    }

    #[test]
    fn each_personality_preset_is_deterministic() {
        let presets = vec![
            all_vanguard(),
            all_guardian(),
            all_tactician(),
            hero_aggressive_enemy_defensive(),
            hero_controller_enemy_aggressive(),
        ];

        for preset in presets {
            let a = run_profile(37, preset.clone());
            let b = run_profile(37, preset);
            assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
            assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
        }
    }
}
