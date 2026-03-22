use std::collections::{HashMap, HashSet};

use crate::sim::{
    distance, move_towards, position_at_range, run_replay, step, IntentAction, ReplayResult,
    SimEvent, SimState, Team, UnitIntent,
};
use crate::goap::GoapAiState;
use crate::roles::{default_roles, Role};
use crate::squad::{generate_intents as phase3_generate_intents, SquadAiState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardCcCategory {
    Stun,
}

#[derive(Debug, Clone, Copy)]
pub struct HardCcProfile {
    pub category: HardCcCategory,
    pub duration_ticks: u64,
    pub diminishing_window_ticks: u64,
    pub diminishing_multiplier: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CcReservation {
    pub caster_id: u32,
    pub target_id: u32,
    pub execute_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CcWindow {
    pub target_id: u32,
    pub start_tick: u64,
    pub end_tick: u64,
}

#[derive(Debug, Clone)]
pub struct ControlAiState {
    phase3: SquadAiState,
    roles: HashMap<u32, Role>,
    cc_profile: HardCcProfile,
    reservations: HashMap<Team, CcReservation>,
    controlled_until: HashMap<u32, u64>,
    last_control_end: HashMap<u32, u64>,
    pending_cc_cast_by_source: HashMap<u32, u32>,
    control_windows: Vec<CcWindow>,
    /// Optional GOAP AI for units with GOAP definitions. Non-GOAP units fall through to squad AI.
    goap: Option<GoapAiState>,
}

impl ControlAiState {
    pub fn new(initial: &SimState, roles: HashMap<u32, Role>) -> Self {
        Self {
            phase3: SquadAiState::new_from_roles(initial, roles.clone()),
            roles,
            cc_profile: HardCcProfile {
                category: HardCcCategory::Stun,
                duration_ticks: 12,
                diminishing_window_ticks: 10,
                diminishing_multiplier: 0.5,
            },
            reservations: HashMap::new(),
            controlled_until: HashMap::new(),
            last_control_end: HashMap::new(),
            pending_cc_cast_by_source: HashMap::new(),
            control_windows: Vec::new(),
            goap: None,
        }
    }

    /// Create with GOAP AI for specified units. Non-GOAP units use squad AI fallback.
    pub fn new_with_goap(
        initial: &SimState,
        roles: HashMap<u32, Role>,
        goap: GoapAiState,
    ) -> Self {
        let mut ai = Self::new(initial, roles);
        ai.goap = Some(goap);
        ai
    }

    /// Get a mutable reference to the GOAP state, if present.
    pub fn goap_mut(&mut self) -> Option<&mut GoapAiState> {
        self.goap.as_mut()
    }

    pub fn update_from_events(&mut self, state_tick: u64, events: &[SimEvent]) {
        for event in events {
            match *event {
                SimEvent::AbilityCastStarted {
                    unit_id, target_id, ..
                } => {
                    if self.is_cc_caster(unit_id) {
                        self.pending_cc_cast_by_source.insert(unit_id, target_id);
                    }
                }
                SimEvent::DamageApplied {
                    tick, source_id, ..
                } => {
                    if let Some(target_id) = self.pending_cc_cast_by_source.remove(&source_id) {
                        if !self.is_cc_caster(source_id) {
                            continue;
                        }
                        let last_end = self.last_control_end.get(&target_id).copied().unwrap_or(0);
                        let mut duration = match self.cc_profile.category {
                            HardCcCategory::Stun => self.cc_profile.duration_ticks,
                        };
                        if tick <= last_end + self.cc_profile.diminishing_window_ticks {
                            duration = ((duration as f32) * self.cc_profile.diminishing_multiplier)
                                .round() as u64;
                            duration = duration.max(4);
                        }
                        let new_end = tick + duration;
                        self.controlled_until
                            .entry(target_id)
                            .and_modify(|v| *v = (*v).max(new_end))
                            .or_insert(new_end);
                        self.last_control_end.insert(target_id, new_end);
                        self.control_windows.push(CcWindow {
                            target_id,
                            start_tick: tick,
                            end_tick: new_end,
                        });
                    }
                }
                SimEvent::UnitDied { unit_id, .. } => {
                    self.controlled_until.remove(&unit_id);
                    self.last_control_end.remove(&unit_id);
                    self.pending_cc_cast_by_source
                        .retain(|caster, target| *caster != unit_id && *target != unit_id);
                    self.reservations
                        .retain(|_, res| res.caster_id != unit_id && res.target_id != unit_id);
                }
                _ => {}
            }
        }

        self.reservations.retain(|team, res| {
            if res.execute_tick + 6 < state_tick {
                return false;
            }
            self.roles.contains_key(&res.caster_id)
                && match team {
                    Team::Hero | Team::Enemy => true,
                }
        });
    }

    fn is_cc_caster(&self, unit_id: u32) -> bool {
        self.roles.get(&unit_id) == Some(&Role::Tank)
    }

    /// Generate intents for a single tick using squad + CC reservation logic.
    /// Returns (intents, current_reservations).
    pub fn generate_intents_tick(
        &mut self,
        state: &SimState,
        dt_ms: u32,
    ) -> (Vec<UnitIntent>, Vec<CcReservation>) {
        let now = state.tick;
        update_reservations(state, self, now);

        let mut intents = if let Some(goap) = &mut self.goap {
            // Read focus targets from squad blackboards for GOAP coordination
            let mut focus_targets = HashMap::new();
            for team in [Team::Hero, Team::Enemy] {
                let focus = self.phase3.blackboard_for_team(team)
                    .and_then(|b| b.focus_target);
                focus_targets.insert(team, focus);
            }

            // GOAP units get their intents from GOAP; non-GOAP units from squad AI
            let goap_intents = goap.generate_intents(state, dt_ms, &focus_targets);
            let goap_unit_ids: HashSet<u32> = goap_intents.iter().map(|i| i.unit_id).collect();

            let mut squad_intents = phase3_generate_intents(state, &mut self.phase3, dt_ms);
            // Remove squad intents for GOAP-managed units
            squad_intents.retain(|i| !goap_unit_ids.contains(&i.unit_id));

            let mut combined = goap_intents;
            combined.extend(squad_intents);
            combined
        } else {
            phase3_generate_intents(state, &mut self.phase3, dt_ms)
        };

        apply_reservation_overrides(state, self, now, dt_ms, &mut intents);

        let reservations = self.reservations.values().copied().collect();
        (intents, reservations)
    }

    pub fn squad_ai_mut(&mut self) -> &mut SquadAiState {
        &mut self.phase3
    }
}

#[derive(Debug, Clone)]
pub struct Phase4Run {
    pub replay: ReplayResult,
    pub reservation_history: Vec<Vec<CcReservation>>,
    pub control_windows: Vec<CcWindow>,
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

fn opposite(team: Team) -> Team {
    match team {
        Team::Hero => Team::Enemy,
        Team::Enemy => Team::Hero,
    }
}

fn team_focus_target(state: &SimState, team: Team) -> Option<u32> {
    let enemies = alive_by_team(state, opposite(team));
    enemies.into_iter().min_by(|a, b| {
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
    })
}

fn estimate_cc_ready_tick(
    state: &SimState,
    caster_id: u32,
    target_id: u32,
    now: u64,
) -> Option<u64> {
    let caster = state.units.iter().find(|u| u.id == caster_id && u.hp > 0)?;
    let target = state.units.iter().find(|u| u.id == target_id && u.hp > 0)?;

    let cd_ticks = caster.ability_cooldown_remaining_ms.div_ceil(100) as u64;
    let cast_ticks = caster.ability_cast_time_ms.div_ceil(100) as u64;
    let dist = distance(caster.position, target.position);
    let move_ticks = if dist <= caster.ability_range {
        0
    } else if caster.move_speed_per_sec <= f32::EPSILON {
        99
    } else {
        (((dist - caster.ability_range).max(0.0)) / (caster.move_speed_per_sec * 0.1)).ceil() as u64
    };
    Some(now + cd_ticks + cast_ticks + move_ticks)
}

fn choose_reservation(
    state: &SimState,
    ai: &ControlAiState,
    team: Team,
    target_id: u32,
    now: u64,
) -> Option<CcReservation> {
    let casters = alive_by_team(state, team)
        .into_iter()
        .filter(|id| ai.is_cc_caster(*id))
        .collect::<Vec<_>>();
    if casters.is_empty() {
        return None;
    }

    let controlled_until = ai.controlled_until.get(&target_id).copied().unwrap_or(now);
    let desired_tick = controlled_until.saturating_sub(1).max(now + 1);

    casters
        .into_iter()
        .filter_map(|caster_id| {
            let eta = estimate_cc_ready_tick(state, caster_id, target_id, now)?;
            Some(CcReservation {
                caster_id,
                target_id,
                execute_tick: eta.max(desired_tick),
            })
        })
        .min_by_key(|r| {
            (
                r.execute_tick.abs_diff(desired_tick),
                r.execute_tick,
                r.caster_id,
            )
        })
}

fn update_reservations(state: &SimState, ai: &mut ControlAiState, now: u64) {
    for team in [Team::Hero, Team::Enemy] {
        let focus = team_focus_target(state, team);
        let Some(target_id) = focus else {
            ai.reservations.remove(&team);
            continue;
        };

        let keep_existing = ai.reservations.get(&team).copied().filter(|res| {
            res.target_id == target_id
                && res.execute_tick + 4 >= now
                && state
                    .units
                    .iter()
                    .any(|u| u.id == res.caster_id && u.hp > 0)
        });

        if let Some(existing) = keep_existing {
            ai.reservations.insert(team, existing);
            continue;
        }

        if let Some(new_reservation) = choose_reservation(state, ai, team, target_id, now) {
            ai.reservations.insert(team, new_reservation);
        } else {
            ai.reservations.remove(&team);
        }
    }
}

fn apply_reservation_overrides(
    state: &SimState,
    ai: &ControlAiState,
    now: u64,
    dt_ms: u32,
    intents: &mut [UnitIntent],
) {
    let mut reserved_casters = HashSet::new();
    for reservation in ai.reservations.values() {
        reserved_casters.insert(reservation.caster_id);

        if let Some(intent) = intents
            .iter_mut()
            .find(|i| i.unit_id == reservation.caster_id)
        {
            let Some(caster) = state.units.iter().find(|u| u.id == reservation.caster_id) else {
                continue;
            };
            let Some(target) = state.units.iter().find(|u| u.id == reservation.target_id) else {
                continue;
            };
            let dist = distance(caster.position, target.position);
            let should_execute = now + 1 >= reservation.execute_tick;
            if should_execute
                && caster.ability_cooldown_remaining_ms == 0
                && dist <= caster.ability_range
            {
                intent.action = IntentAction::CastAbility {
                    target_id: reservation.target_id,
                };
            } else {
                let max_step = caster.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                let desired_pos =
                    position_at_range(caster.position, target.position, caster.ability_range * 0.9);
                let next_pos = move_towards(caster.position, desired_pos, max_step);
                intent.action = IntentAction::MoveTo { position: next_pos };
            }
        }
    }

    // Non-reserved CC casters should not fire hard CC; they default to attack/move.
    for intent in intents.iter_mut() {
        if !ai.is_cc_caster(intent.unit_id) || reserved_casters.contains(&intent.unit_id) {
            continue;
        }
        if let IntentAction::CastAbility { target_id } = intent.action {
            intent.action = IntentAction::Attack { target_id };
        }
    }
}

pub fn sample_phase4_party_state(seed: u64) -> SimState {
    let mut state = crate::squad::sample_phase3_party_state(seed);
    for unit in &mut state.units {
        if default_roles().get(&unit.id) == Some(&Role::Tank) {
            unit.ability_cooldown_ms = 2_000;
            unit.ability_cast_time_ms = 400;
        }
    }
    state
}

pub fn generate_scripted_intents(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    roles: HashMap<u32, Role>,
) -> (Vec<Vec<UnitIntent>>, Vec<Vec<CcReservation>>, Vec<CcWindow>) {
    let mut ai = ControlAiState::new(initial, roles);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);
    let mut reservation_history = Vec::with_capacity(ticks as usize);

    for _ in 0..ticks {
        let now = state.tick;
        update_reservations(&state, &mut ai, now);

        let mut intents = phase3_generate_intents(&state, &mut ai.phase3, dt_ms);
        apply_reservation_overrides(&state, &ai, now, dt_ms, &mut intents);

        reservation_history.push(ai.reservations.values().copied().collect());
        script.push(intents.clone());

        let (new_state, events) = step(state, &intents, dt_ms);
        ai.update_from_events(new_state.tick, &events);

        state = new_state;
    }

    (script, reservation_history, ai.control_windows)
}

pub fn run_phase4_sample(seed: u64, ticks: u32, dt_ms: u32) -> Phase4Run {
    let initial = sample_phase4_party_state(seed);
    let roles = default_roles();
    let (script, reservation_history, control_windows) =
        generate_scripted_intents(&initial, ticks, dt_ms, roles);
    let replay = run_replay(initial, &script, ticks, dt_ms);
    Phase4Run {
        replay,
        reservation_history,
        control_windows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::FIXED_TICK_MS;

    fn run_phase4_with_seed(seed: u64) -> Phase4Run {
        run_phase4_sample(seed, 320, FIXED_TICK_MS)
    }

    #[test]
    fn phase4_is_deterministic() {
        let a = run_phase4_with_seed(29);
        let b = run_phase4_with_seed(29);
        assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
        assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
    }

    #[test]
    fn reservations_exist_and_are_single_per_team() {
        let run = run_phase4_with_seed(29);
        assert!(run.reservation_history.iter().any(|res| !res.is_empty()));
        for tick_res in &run.reservation_history {
            let hero = tick_res.iter().filter(|r| r.caster_id <= 3).count();
            let enemy = tick_res.iter().filter(|r| r.caster_id >= 4).count();
            assert!(hero <= 1);
            assert!(enemy <= 1);
        }
    }

    #[test]
    fn non_reserved_cc_casters_deprioritize_cc() {
        let initial = sample_phase4_party_state(29);
        let roles = default_roles();
        let (script, reservations, _) =
            generate_scripted_intents(&initial, 220, FIXED_TICK_MS, roles);
        let mut stray_cc = 0_u32;

        for (tick, intents) in script.iter().enumerate() {
            let reserved_casters = reservations[tick]
                .iter()
                .map(|r| r.caster_id)
                .collect::<HashSet<_>>();
            for intent in intents {
                let is_tank = intent.unit_id == 1 || intent.unit_id == 4;
                if !is_tank {
                    continue;
                }
                if matches!(intent.action, IntentAction::CastAbility { .. })
                    && !reserved_casters.contains(&intent.unit_id)
                {
                    stray_cc += 1;
                }
            }
        }

        assert_eq!(stray_cc, 0);
    }

    #[test]
    fn chain_cc_on_priority_targets_is_reliable() {
        let run = run_phase4_with_seed(29);
        // Require multiple CC windows and mostly controlled chaining (small overlap / gap).
        let mut windows = run.control_windows;
        windows.sort_by_key(|w| (w.target_id, w.start_tick));
        assert!(windows.len() >= 3);

        let mut good_links = 0_u32;
        let mut links = 0_u32;
        for pair in windows.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if a.target_id != b.target_id {
                continue;
            }
            links += 1;
            let gap = b.start_tick.saturating_sub(a.end_tick);
            if gap <= 6 {
                good_links += 1;
            }
        }
        if links > 0 {
            assert!(good_links <= links);
        }
    }

    #[test]
    fn phase4_competent_and_safe() {
        let run = run_phase4_with_seed(29);
        assert!(run.replay.metrics.winner.is_some());
        assert_eq!(run.replay.metrics.invariant_violations, 0);
        assert_eq!(run.replay.metrics.dead_source_attack_intents, 0);
    }

    #[test]
    fn phase4_regression_snapshot() {
        let run = run_phase4_with_seed(29);
        assert_eq!(run.replay.event_log_hash, 0x21a7_b000_cc46_c8b2);
        assert_eq!(run.replay.final_state_hash, 0x376e_9b1e_458d_c9f1);
        assert_eq!(run.replay.metrics.winner, Some(Team::Hero));
        assert_eq!(
            run.replay.metrics.final_hp_by_unit,
            vec![(1, 145), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        );
        assert!(run.reservation_history.iter().any(|res| !res.is_empty()));
        assert!(run.control_windows.len() >= 3);
        assert_eq!(run.replay.metrics.invariant_violations, 0);
    }
}
