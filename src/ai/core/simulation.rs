use std::collections::HashMap;
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::ai::effects::StatusKind;

use super::types::*;
use super::events::SimEvent;
use super::math::*;
use super::helpers::*;
use super::resolve::resolve_cast;
use super::intent::*;
use super::tick_systems::*;
use super::tick_world::*;
use super::metrics::compute_metrics;
use super::verify::verify_tick;

pub fn step(mut state: SimState, intents: &[UnitIntent], dt_ms: u32) -> (SimState, Vec<SimEvent>) {
    state.tick += 1;
    let tick = state.tick;
    let mut events = Vec::new();
    let intents_by_unit = collect_intents(intents);

    tick_hero_cooldowns(&mut state, dt_ms);
    tick_status_effects(&mut state, tick, dt_ms, &mut events);
    advance_projectiles(&mut state, tick, dt_ms, &mut events);
    tick_periodic_passives(&mut state, tick, dt_ms, &mut events);
    tick_zones(&mut state, tick, dt_ms, &mut events);
    tick_channels(&mut state, tick, dt_ms, &mut events);
    tick_tethers(&mut state, tick, dt_ms, &mut events);

    for unit in &mut state.units {
        if unit.hp > 0 {
            unit.state_history.push_back((state.tick as u32, unit.position, unit.hp));
            if unit.state_history.len() > 50 {
                unit.state_history.pop_front();
            }
        }
    }

    // Shuffle unit processing order to avoid first-mover advantage
    let mut unit_order: Vec<usize> = (0..state.units.len()).collect();
    for i in (1..unit_order.len()).rev() {
        let j = (next_rand_u32(&mut state) as usize) % (i + 1);
        unit_order.swap(i, j);
    }

    for &idx in &unit_order {
        if !is_alive(&state.units[idx]) {
            continue;
        }

        // Directed summons don't act independently — skip intent processing
        if state.units[idx].directed {
            continue;
        }

        if state.units[idx].cooldown_remaining_ms > 0 {
            state.units[idx].cooldown_remaining_ms =
                state.units[idx].cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].ability_cooldown_remaining_ms > 0 {
            state.units[idx].ability_cooldown_remaining_ms = state.units[idx]
                .ability_cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].heal_cooldown_remaining_ms > 0 {
            state.units[idx].heal_cooldown_remaining_ms = state.units[idx]
                .heal_cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].control_cooldown_remaining_ms > 0 {
            state.units[idx].control_cooldown_remaining_ms = state.units[idx]
                .control_cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].control_remaining_ms > 0 {
            // Check if unit is casting an unstoppable ability — if so, ignore CC
            let is_unstoppable = state.units[idx].casting.as_ref().map_or(false, |cast| {
                if let CastKind::HeroAbility(ai) = cast.kind {
                    state.units[idx].abilities.get(ai).map_or(false, |s| s.def.unstoppable)
                } else {
                    false
                }
            });
            if is_unstoppable {
                // Clear the CC, cast continues
                state.units[idx].control_remaining_ms = 0;
            } else {
                state.units[idx].control_remaining_ms =
                    state.units[idx].control_remaining_ms.saturating_sub(dt_ms);
                state.units[idx].casting = None;
                events.push(SimEvent::UnitControlled {
                    tick, unit_id: state.units[idx].id,
                });
                continue;
            }
        }

        if state.units[idx].channeling.is_some() {
            continue;
        }

        if state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Banish)) {
            state.units[idx].casting = None;
            continue;
        }

        if let Some(fear_source) = state.units[idx].status_effects.iter().find_map(|s| {
            if let StatusKind::Fear { source_pos } = s.kind { Some(source_pos) } else { None }
        }) {
            state.units[idx].casting = None;
            let max_delta = state.units[idx].move_speed_per_sec * (dt_ms as f32 / 1000.0);
            let start = state.units[idx].position;
            let next = move_away(start, fear_source, max_delta);
            if distance(start, next) > f32::EPSILON {
                state.units[idx].position = next;
                events.push(SimEvent::Moved {
                    tick, unit_id: state.units[idx].id,
                    from_x100: to_x100(start.x), from_y100: to_x100(start.y),
                    to_x100: to_x100(next.x), to_y100: to_x100(next.y),
                });
            }
            continue;
        }

        let is_polymorphed = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Polymorph));
        if is_polymorphed {
            state.units[idx].casting = None;
        }

        if let Some(mut cast) = state.units[idx].casting {
            cast.remaining_ms = cast.remaining_ms.saturating_sub(dt_ms);
            if cast.remaining_ms == 0 {
                state.units[idx].casting = None;
                resolve_cast(idx, cast.target_id, cast.target_pos, cast.kind, tick, &mut state, &mut events);
            } else {
                state.units[idx].casting = Some(cast);
            }
            continue;
        }

        let mut intent = intents_by_unit
            .iter()
            .find(|(unit_id, _)| *unit_id == state.units[idx].id)
            .map(|(_, action)| *action)
            .unwrap_or(IntentAction::Hold);

        let is_rooted = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Root));
        let is_silenced = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Silence));
        let is_confused = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Confuse));

        if is_rooted {
            if let IntentAction::MoveTo { .. } = intent {
                intent = IntentAction::Hold;
            }
        }

        if is_silenced || is_polymorphed {
            match intent {
                IntentAction::CastAbility { .. } | IntentAction::CastControl { .. }
                | IntentAction::UseAbility { .. } => {
                    intent = IntentAction::Hold;
                }
                _ => {}
            }
        }

        if is_polymorphed {
            if let IntentAction::Attack { .. } = intent {
                intent = IntentAction::Hold;
            }
        }

        if let Some(taunter_id) = state.units[idx].status_effects.iter().find_map(|s| {
            if let StatusKind::Taunt { taunter_id } = s.kind { Some(taunter_id) } else { None }
        }) {
            match intent {
                IntentAction::Attack { .. } => {
                    intent = IntentAction::Attack { target_id: taunter_id };
                }
                _ => {}
            }
        }

        if is_confused {
            let alive: Vec<u32> = state.units.iter()
                .filter(|u| u.hp > 0 && u.id != state.units[idx].id)
                .map(|u| u.id)
                .collect();
            if !alive.is_empty() {
                let random_target = alive[(next_rand_u32(&mut state) as usize) % alive.len()];
                match intent {
                    IntentAction::Attack { .. } => {
                        intent = IntentAction::Attack { target_id: random_target };
                    }
                    IntentAction::CastAbility { .. } => {
                        intent = IntentAction::CastAbility { target_id: random_target };
                    }
                    _ => {}
                }
            }
        }

        match intent {
            IntentAction::Hold => {}
            IntentAction::MoveTo { position } => {
                move_towards_position(idx, position, tick, &mut state, dt_ms, &mut events);
                clamp_leash(idx, &mut state);
            }
            IntentAction::Attack { target_id } => {
                try_start_attack(idx, target_id, tick, dt_ms, &mut state, &mut events);
            }
            IntentAction::CastAbility { target_id } => {
                try_start_ability(idx, target_id, tick, &mut state, &mut events);
            }
            IntentAction::CastHeal { target_id } => {
                try_start_heal(idx, target_id, tick, &mut state, &mut events);
            }
            IntentAction::CastControl { target_id } => {
                try_start_control(idx, target_id, tick, &mut state, &mut events);
            }
            IntentAction::UseAbility { ability_index, target } => {
                try_start_hero_ability(idx, ability_index, target, tick, &mut state, &mut events);
            }
        }
    }

    // Runtime verification: check invariants after every tick in debug/test builds.
    #[cfg(debug_assertions)]
    {
        let report = verify_tick(&state);
        debug_assert!(
            report.is_ok(),
            "Runtime verification failed at tick {}: {:?}",
            state.tick,
            report.violations,
        );
    }

    (state, events)
}

pub fn run_replay(
    initial_state: SimState,
    scripted_intents: &[Vec<UnitIntent>],
    ticks: u32,
    dt_ms: u32,
) -> ReplayResult {
    let mut state = initial_state.clone();
    let mut all_events = Vec::new();
    let mut per_tick_state_hashes = Vec::with_capacity(ticks as usize);

    for tick in 0..ticks {
        let intents = scripted_intents
            .get(tick as usize)
            .map_or(&[][..], |v| v.as_slice());
        let (new_state, events) = step(state, intents, dt_ms);
        state = new_state;
        all_events.extend(events);
        per_tick_state_hashes.push(hash_sim_state(&state));
    }

    let event_log_hash = hash_event_log(&all_events);
    let final_state_hash = hash_sim_state(&state);
    let metrics = compute_metrics(
        &initial_state, &state, scripted_intents, &all_events, ticks, dt_ms,
    );

    ReplayResult {
        final_state: state,
        events: all_events,
        event_log_hash,
        final_state_hash,
        per_tick_state_hashes,
        metrics,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    pub final_state: SimState,
    pub events: Vec<SimEvent>,
    pub event_log_hash: u64,
    pub final_state_hash: u64,
    pub per_tick_state_hashes: Vec<u64>,
    pub metrics: SimMetrics,
}

pub fn sample_duel_state(seed: u64) -> SimState {
    let mut units = vec![
        UnitState {
            id: 1,
            team: Team::Hero,
            hp: 100,
            max_hp: 100,
            position: sim_vec2(0.0, 0.0),
            move_speed_per_sec: 4.0,
            attack_damage: 14,
            attack_range: 1.4,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 20,
            ability_range: 1.6,
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
        UnitState {
            id: 2,
            team: Team::Enemy,
            hp: 100,
            max_hp: 100,
            position: sim_vec2(8.0, 0.0),
            move_speed_per_sec: 3.5,
            attack_damage: 12,
            attack_range: 1.2,
            attack_cooldown_ms: 900,
            attack_cast_time_ms: 400,
            cooldown_remaining_ms: 0,
            ability_damage: 16,
            ability_range: 1.4,
            ability_cooldown_ms: 3_200,
            ability_cast_time_ms: 600,
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
    ];
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units,
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

pub fn sample_duel_script(ticks: u32) -> Vec<Vec<UnitIntent>> {
    let mut script = Vec::with_capacity(ticks as usize);
    for _ in 0..ticks {
        script.push(vec![
            UnitIntent {
                unit_id: 1,
                action: IntentAction::Attack { target_id: 2 },
            },
            UnitIntent {
                unit_id: 2,
                action: IntentAction::Attack { target_id: 1 },
            },
        ]);
    }
    script
}

// ---------------------------------------------------------------------------
// Determinism verification
// ---------------------------------------------------------------------------

/// Result of a determinism check — compares two replays of the same input.
#[derive(Debug, Clone)]
pub struct DeterminismReport {
    /// True if both runs produced identical results.
    pub is_deterministic: bool,
    /// If divergence occurred, the first tick index where hashes differ.
    pub first_divergent_tick: Option<usize>,
    /// Hash from run A at the divergent tick (if any).
    pub hash_a: Option<u64>,
    /// Hash from run B at the divergent tick (if any).
    pub hash_b: Option<u64>,
}

/// Run the same replay twice and compare per-tick state hashes.
/// Returns a report indicating whether the simulation is fully deterministic
/// for the given initial state and intent script.
pub fn verify_determinism(
    initial_state: &SimState,
    scripted_intents: &[Vec<UnitIntent>],
    ticks: u32,
    dt_ms: u32,
) -> DeterminismReport {
    let result_a = run_replay(initial_state.clone(), scripted_intents, ticks, dt_ms);
    let result_b = run_replay(initial_state.clone(), scripted_intents, ticks, dt_ms);

    // Compare per-tick hashes
    for (i, (ha, hb)) in result_a
        .per_tick_state_hashes
        .iter()
        .zip(result_b.per_tick_state_hashes.iter())
        .enumerate()
    {
        if ha != hb {
            return DeterminismReport {
                is_deterministic: false,
                first_divergent_tick: Some(i),
                hash_a: Some(*ha),
                hash_b: Some(*hb),
            };
        }
    }

    // Also check final aggregates
    if result_a.event_log_hash != result_b.event_log_hash
        || result_a.final_state_hash != result_b.final_state_hash
    {
        return DeterminismReport {
            is_deterministic: false,
            first_divergent_tick: None,
            hash_a: Some(result_a.final_state_hash),
            hash_b: Some(result_b.final_state_hash),
        };
    }

    DeterminismReport {
        is_deterministic: true,
        first_divergent_tick: None,
        hash_a: None,
        hash_b: None,
    }
}

/// Run a replay against a known set of expected per-tick hashes.
/// Returns the first tick where the actual hash diverges from the expected.
pub fn verify_replay_against_hashes(
    initial_state: &SimState,
    scripted_intents: &[Vec<UnitIntent>],
    expected_hashes: &[u64],
    dt_ms: u32,
) -> Option<(usize, u64, u64)> {
    let mut state = initial_state.clone();
    for (tick, expected) in expected_hashes.iter().enumerate() {
        let intents = scripted_intents
            .get(tick)
            .map_or(&[][..], |v| v.as_slice());
        let (new_state, _) = step(state, intents, dt_ms);
        state = new_state;
        let actual = hash_sim_state(&state);
        if actual != *expected {
            return Some((tick, *expected, actual));
        }
    }
    None
}

pub fn hash_event_log(events: &[SimEvent]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;
    let mut hash = FNV_OFFSET;
    for event in events {
        let line = format!("{event:?}");
        for byte in line.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

pub fn hash_sim_state(state: &SimState) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut entries = state.units.iter()
        .map(|u| {
            format!(
                "id={} team={:?} hp={} max_hp={} pos=({}, {}) cd={} acd={} hcd={} ccd={} ctrl={} cast={:?}",
                u.id, u.team, u.hp, u.max_hp,
                to_x100(u.position.x), to_x100(u.position.y),
                u.cooldown_remaining_ms, u.ability_cooldown_remaining_ms,
                u.heal_cooldown_remaining_ms, u.control_cooldown_remaining_ms,
                u.control_remaining_ms, u.casting
            )
        })
        .collect::<Vec<_>>();
    entries.sort();

    let header = format!("tick={} rng={}", state.tick, state.rng_state);
    for byte in header.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for line in entries {
        for byte in line.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
