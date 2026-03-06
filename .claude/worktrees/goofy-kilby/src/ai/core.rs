use std::cmp::Ordering;
use std::collections::HashMap;

pub const FIXED_TICK_MS: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Team {
    Hero,
    Enemy,
}

#[derive(Debug, Clone)]
pub struct UnitState {
    pub id: u32,
    pub team: Team,
    pub hp: i32,
    pub max_hp: i32,
    pub position: SimVec2,
    pub move_speed_per_sec: f32,
    pub attack_damage: i32,
    pub attack_range: f32,
    pub attack_cooldown_ms: u32,
    pub attack_cast_time_ms: u32,
    pub cooldown_remaining_ms: u32,
    pub ability_damage: i32,
    pub ability_range: f32,
    pub ability_cooldown_ms: u32,
    pub ability_cast_time_ms: u32,
    pub ability_cooldown_remaining_ms: u32,
    pub heal_amount: i32,
    pub heal_range: f32,
    pub heal_cooldown_ms: u32,
    pub heal_cast_time_ms: u32,
    pub heal_cooldown_remaining_ms: u32,
    pub control_range: f32,
    pub control_duration_ms: u32,
    pub control_cooldown_ms: u32,
    pub control_cast_time_ms: u32,
    pub control_cooldown_remaining_ms: u32,
    pub control_remaining_ms: u32,
    pub casting: Option<CastState>,
}

#[derive(Debug, Clone, Copy)]
pub struct CastState {
    pub target_id: u32,
    pub remaining_ms: u32,
    pub kind: CastKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    Attack,
    Ability,
    Heal,
    Control,
}

#[derive(Debug, Clone)]
pub struct SimState {
    pub tick: u64,
    pub rng_state: u64,
    pub units: Vec<UnitState>,
}

#[derive(Debug, Clone, Copy)]
pub enum IntentAction {
    Attack { target_id: u32 },
    CastAbility { target_id: u32 },
    CastHeal { target_id: u32 },
    CastControl { target_id: u32 },
    MoveTo { position: SimVec2 },
    Hold,
}

#[derive(Debug, Clone, Copy)]
pub struct UnitIntent {
    pub unit_id: u32,
    pub action: IntentAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimEvent {
    Moved {
        tick: u64,
        unit_id: u32,
        from_x100: i32,
        from_y100: i32,
        to_x100: i32,
        to_y100: i32,
    },
    CastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    CastFailedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    DamageApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        amount: i32,
        target_hp_before: i32,
        target_hp_after: i32,
    },
    UnitDied {
        tick: u64,
        unit_id: u32,
    },
    AttackBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    AbilityBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    AttackBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AbilityBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AttackRepositioned {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AbilityBlockedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AbilityCastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    HealBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    HealBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    HealBlockedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    HealCastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    ControlBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlBlockedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlCastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        duration_ms: u32,
    },
    UnitControlled {
        tick: u64,
        unit_id: u32,
    },
    HealApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        amount: i32,
        target_hp_before: i32,
        target_hp_after: i32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SimVec2 {
    pub x: f32,
    pub y: f32,
}

pub const fn sim_vec2(x: f32, y: f32) -> SimVec2 {
    SimVec2 { x, y }
}

pub fn distance(a: SimVec2, b: SimVec2) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

pub fn move_towards(from: SimVec2, to: SimVec2, max_delta: f32) -> SimVec2 {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON || max_delta <= f32::EPSILON {
        return from;
    }
    let step = len.min(max_delta);
    let nx = dx / len;
    let ny = dy / len;
    sim_vec2(from.x + nx * step, from.y + ny * step)
}

pub fn move_away(from: SimVec2, threat: SimVec2, max_delta: f32) -> SimVec2 {
    let dx = from.x - threat.x;
    let dy = from.y - threat.y;
    let len = (dx * dx + dy * dy).sqrt();
    if max_delta <= f32::EPSILON {
        return from;
    }
    if len <= f32::EPSILON {
        return sim_vec2(from.x + max_delta, from.y);
    }
    let nx = dx / len;
    let ny = dy / len;
    sim_vec2(from.x + nx * max_delta, from.y + ny * max_delta)
}

pub fn position_at_range(from: SimVec2, target: SimVec2, desired_range: f32) -> SimVec2 {
    let dx = from.x - target.x;
    let dy = from.y - target.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON {
        return sim_vec2(target.x + desired_range, target.y);
    }
    let nx = dx / len;
    let ny = dy / len;
    sim_vec2(target.x + nx * desired_range, target.y + ny * desired_range)
}

#[derive(Debug, Clone)]
pub struct SimMetrics {
    pub ticks_elapsed: u32,
    pub seconds_elapsed: f32,
    pub winner: Option<Team>,
    pub tick_to_first_death: Option<u64>,
    pub final_hp_by_unit: Vec<(u32, i32)>,
    pub total_damage_by_unit: Vec<(u32, i32)>,
    pub damage_taken_by_unit: Vec<(u32, i32)>,
    pub dps_by_unit: Vec<(u32, f32)>,
    pub overkill_damage_total: i32,
    pub casts_started: u32,
    pub casts_completed: u32,
    pub casts_failed_out_of_range: u32,
    pub avg_cast_delay_ms: f32,
    pub heals_started: u32,
    pub heals_completed: u32,
    pub total_healing_by_unit: Vec<(u32, i32)>,
    pub attack_intents: u32,
    pub executed_attack_intents: u32,
    pub blocked_cooldown_intents: u32,
    pub blocked_invalid_target_intents: u32,
    pub dead_source_attack_intents: u32,
    pub reposition_for_range_events: u32,
    pub focus_fire_ticks: u32,
    pub max_targeters_on_single_target: u32,
    pub target_switches_by_unit: Vec<(u32, u32)>,
    pub movement_distance_x100_by_unit: Vec<(u32, i32)>,
    pub in_range_ticks_by_unit: Vec<(u32, u32)>,
    pub out_of_range_ticks_by_unit: Vec<(u32, u32)>,
    pub chase_ticks_by_unit: Vec<(u32, u32)>,
    pub invariant_violations: u32,
}

pub fn step(mut state: SimState, intents: &[UnitIntent], dt_ms: u32) -> (SimState, Vec<SimEvent>) {
    state.tick += 1;
    let tick = state.tick;
    let mut events = Vec::new();
    let intents_by_unit = collect_intents(intents);

    for idx in 0..state.units.len() {
        if !is_alive(&state.units[idx]) {
            continue;
        }

        if state.units[idx].cooldown_remaining_ms > 0 {
            state.units[idx].cooldown_remaining_ms =
                state.units[idx].cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].ability_cooldown_remaining_ms > 0 {
            state.units[idx].ability_cooldown_remaining_ms = state.units[idx]
                .ability_cooldown_remaining_ms
                .saturating_sub(dt_ms);
        }
        if state.units[idx].heal_cooldown_remaining_ms > 0 {
            state.units[idx].heal_cooldown_remaining_ms = state.units[idx]
                .heal_cooldown_remaining_ms
                .saturating_sub(dt_ms);
        }
        if state.units[idx].control_cooldown_remaining_ms > 0 {
            state.units[idx].control_cooldown_remaining_ms = state.units[idx]
                .control_cooldown_remaining_ms
                .saturating_sub(dt_ms);
        }
        if state.units[idx].control_remaining_ms > 0 {
            state.units[idx].control_remaining_ms =
                state.units[idx].control_remaining_ms.saturating_sub(dt_ms);
            state.units[idx].casting = None;
            events.push(SimEvent::UnitControlled {
                tick,
                unit_id: state.units[idx].id,
            });
            continue;
        }

        if let Some(mut cast) = state.units[idx].casting {
            cast.remaining_ms = cast.remaining_ms.saturating_sub(dt_ms);
            if cast.remaining_ms == 0 {
                state.units[idx].casting = None;
                resolve_cast(
                    idx,
                    cast.target_id,
                    cast.kind,
                    tick,
                    &mut state,
                    &mut events,
                );
            } else {
                state.units[idx].casting = Some(cast);
            }
            continue;
        }

        let intent = intents_by_unit
            .iter()
            .find(|(unit_id, _)| *unit_id == state.units[idx].id)
            .map(|(_, action)| *action)
            .unwrap_or(IntentAction::Hold);

        match intent {
            IntentAction::Hold => {}
            IntentAction::MoveTo { position } => {
                move_towards_position(idx, position, tick, &mut state, dt_ms, &mut events);
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
        }
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
        &initial_state,
        &state,
        scripted_intents,
        &all_events,
        ticks,
        dt_ms,
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

#[derive(Debug, Clone)]
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
        },
    ];
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units,
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
    let mut entries = state
        .units
        .iter()
        .map(|u| {
            format!(
                "id={} team={:?} hp={} max_hp={} pos=({}, {}) cd={} acd={} hcd={} ccd={} ctrl={} cast={:?}",
                u.id,
                u.team,
                u.hp,
                u.max_hp,
                to_x100(u.position.x),
                to_x100(u.position.y),
                u.cooldown_remaining_ms,
                u.ability_cooldown_remaining_ms,
                u.heal_cooldown_remaining_ms,
                u.control_cooldown_remaining_ms,
                u.control_remaining_ms,
                u.casting
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

fn collect_intents(intents: &[UnitIntent]) -> Vec<(u32, IntentAction)> {
    let mut stable = intents
        .iter()
        .map(|intent| (intent.unit_id, intent.action))
        .collect::<Vec<_>>();
    stable.sort_by_key(|entry| entry.0);
    stable
}

fn is_alive(unit: &UnitState) -> bool {
    unit.hp > 0
}

fn move_towards_position(
    idx: usize,
    target_pos: SimVec2,
    tick: u64,
    state: &mut SimState,
    dt_ms: u32,
    events: &mut Vec<SimEvent>,
) {
    let start = state.units[idx].position;
    let max_delta = state.units[idx].move_speed_per_sec * (dt_ms as f32 / 1000.0);
    let next = move_towards(start, target_pos, max_delta);
    if distance(start, next) <= f32::EPSILON {
        return;
    }
    state.units[idx].position = next;
    events.push(SimEvent::Moved {
        tick,
        unit_id: state.units[idx].id,
        from_x100: to_x100(start.x),
        from_y100: to_x100(start.y),
        to_x100: to_x100(state.units[idx].position.x),
        to_y100: to_x100(state.units[idx].position.y),
    });
}

fn try_start_attack(
    attacker_idx: usize,
    target_id: u32,
    tick: u64,
    dt_ms: u32,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    try_start_cast(
        attacker_idx,
        target_id,
        CastKind::Attack,
        tick,
        dt_ms,
        state,
        events,
    );
}

fn try_start_ability(
    attacker_idx: usize,
    target_id: u32,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.units[attacker_idx].ability_cooldown_remaining_ms > 0 {
        events.push(SimEvent::AbilityBlockedCooldown {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
            cooldown_remaining_ms: state.units[attacker_idx].ability_cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::AbilityBlockedInvalidTarget {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
        });
        return;
    };

    if !is_alive(&state.units[target_idx]) {
        events.push(SimEvent::AbilityBlockedInvalidTarget {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
        });
        return;
    }

    if !target_in_range_for_kind(attacker_idx, target_idx, state, CastKind::Ability) {
        events.push(SimEvent::AbilityBlockedOutOfRange {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        remaining_ms: state.units[attacker_idx].ability_cast_time_ms,
        kind: CastKind::Ability,
    };
    state.units[attacker_idx].casting = Some(cast);
    events.push(SimEvent::AbilityCastStarted {
        tick,
        unit_id: state.units[attacker_idx].id,
        target_id,
    });
}

fn try_start_heal(
    healer_idx: usize,
    target_id: u32,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.units[healer_idx].heal_cooldown_remaining_ms > 0 {
        events.push(SimEvent::HealBlockedCooldown {
            tick,
            unit_id: state.units[healer_idx].id,
            target_id,
            cooldown_remaining_ms: state.units[healer_idx].heal_cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::HealBlockedInvalidTarget {
            tick,
            unit_id: state.units[healer_idx].id,
            target_id,
        });
        return;
    };

    if !is_alive(&state.units[target_idx])
        || state.units[target_idx].team != state.units[healer_idx].team
        || state.units[target_idx].hp >= state.units[target_idx].max_hp
    {
        events.push(SimEvent::HealBlockedInvalidTarget {
            tick,
            unit_id: state.units[healer_idx].id,
            target_id,
        });
        return;
    }

    if !target_in_range_for_kind(healer_idx, target_idx, state, CastKind::Heal) {
        events.push(SimEvent::HealBlockedOutOfRange {
            tick,
            unit_id: state.units[healer_idx].id,
            target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        remaining_ms: state.units[healer_idx].heal_cast_time_ms,
        kind: CastKind::Heal,
    };
    state.units[healer_idx].casting = Some(cast);
    events.push(SimEvent::HealCastStarted {
        tick,
        unit_id: state.units[healer_idx].id,
        target_id,
    });
}

fn try_start_control(
    caster_idx: usize,
    target_id: u32,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.units[caster_idx].control_duration_ms == 0
        || state.units[caster_idx].control_range <= 0.0
    {
        events.push(SimEvent::ControlBlockedInvalidTarget {
            tick,
            unit_id: state.units[caster_idx].id,
            target_id,
        });
        return;
    }
    if state.units[caster_idx].control_cooldown_remaining_ms > 0 {
        events.push(SimEvent::ControlBlockedCooldown {
            tick,
            unit_id: state.units[caster_idx].id,
            target_id,
            cooldown_remaining_ms: state.units[caster_idx].control_cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::ControlBlockedInvalidTarget {
            tick,
            unit_id: state.units[caster_idx].id,
            target_id,
        });
        return;
    };
    if !is_alive(&state.units[target_idx])
        || state.units[target_idx].team == state.units[caster_idx].team
    {
        events.push(SimEvent::ControlBlockedInvalidTarget {
            tick,
            unit_id: state.units[caster_idx].id,
            target_id,
        });
        return;
    }
    if !target_in_range_for_kind(caster_idx, target_idx, state, CastKind::Control) {
        events.push(SimEvent::ControlBlockedOutOfRange {
            tick,
            unit_id: state.units[caster_idx].id,
            target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        remaining_ms: state.units[caster_idx].control_cast_time_ms,
        kind: CastKind::Control,
    };
    state.units[caster_idx].casting = Some(cast);
    events.push(SimEvent::ControlCastStarted {
        tick,
        unit_id: state.units[caster_idx].id,
        target_id,
    });
}

fn try_start_cast(
    attacker_idx: usize,
    target_id: u32,
    kind: CastKind,
    tick: u64,
    dt_ms: u32,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.units[attacker_idx].cooldown_remaining_ms > 0 {
        events.push(SimEvent::AttackBlockedCooldown {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
            cooldown_remaining_ms: state.units[attacker_idx].cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::AttackBlockedInvalidTarget {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
        });
        return;
    };

    if !is_alive(&state.units[target_idx]) {
        events.push(SimEvent::AttackBlockedInvalidTarget {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
        });
        return;
    }

    if !target_in_range_for_kind(attacker_idx, target_idx, state, kind) {
        let target_pos = state.units[target_idx].position;
        move_towards_position(attacker_idx, target_pos, tick, state, dt_ms, events);
        events.push(SimEvent::AttackRepositioned {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        remaining_ms: state.units[attacker_idx].attack_cast_time_ms,
        kind,
    };
    state.units[attacker_idx].casting = Some(cast);
    events.push(SimEvent::CastStarted {
        tick,
        unit_id: state.units[attacker_idx].id,
        target_id,
    });
}

fn resolve_cast(
    attacker_idx: usize,
    target_id: u32,
    kind: CastKind,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    let Some(target_idx) = find_unit_idx(state, target_id) else {
        return;
    };
    if !is_alive(&state.units[target_idx]) {
        return;
    }

    if !target_in_range_for_kind(attacker_idx, target_idx, state, kind) {
        match kind {
            CastKind::Heal => events.push(SimEvent::HealBlockedOutOfRange {
                tick,
                unit_id: state.units[attacker_idx].id,
                target_id,
            }),
            CastKind::Control => events.push(SimEvent::ControlBlockedOutOfRange {
                tick,
                unit_id: state.units[attacker_idx].id,
                target_id,
            }),
            _ => events.push(SimEvent::CastFailedOutOfRange {
                tick,
                unit_id: state.units[attacker_idx].id,
                target_id,
            }),
        }
        return;
    }

    if kind == CastKind::Heal {
        let base_heal = state.units[attacker_idx].heal_amount;
        let variance_percent = 95 + (next_rand_u32(state) % 11) as i32;
        let heal_amount = ((base_heal * variance_percent) + 99) / 100;
        let current_hp = state.units[target_idx].hp;
        let max_hp = state.units[target_idx].max_hp;
        let new_hp = (current_hp + heal_amount).min(max_hp);
        let actual_healed = new_hp - current_hp;
        state.units[target_idx].hp = new_hp;
        state.units[attacker_idx].heal_cooldown_remaining_ms =
            state.units[attacker_idx].heal_cooldown_ms;
        events.push(SimEvent::HealApplied {
            tick,
            source_id: state.units[attacker_idx].id,
            target_id,
            amount: actual_healed,
            target_hp_before: current_hp,
            target_hp_after: new_hp,
        });
        return;
    }
    if kind == CastKind::Control {
        let duration_ms = state.units[attacker_idx].control_duration_ms;
        state.units[target_idx].control_remaining_ms = state.units[target_idx]
            .control_remaining_ms
            .max(duration_ms);
        state.units[target_idx].casting = None;
        state.units[attacker_idx].control_cooldown_remaining_ms =
            state.units[attacker_idx].control_cooldown_ms;
        events.push(SimEvent::ControlApplied {
            tick,
            source_id: state.units[attacker_idx].id,
            target_id,
            duration_ms,
        });
        return;
    }

    let variance_percent = 90 + (next_rand_u32(state) % 21) as i32;
    let base_damage = match kind {
        CastKind::Attack => state.units[attacker_idx].attack_damage,
        CastKind::Ability => state.units[attacker_idx].ability_damage,
        CastKind::Heal => 0,
        CastKind::Control => 0,
    };
    let damage = ((base_damage * variance_percent) + 99) / 100;
    let current_hp = state.units[target_idx].hp;
    let new_hp = (current_hp - damage).max(0);
    state.units[target_idx].hp = new_hp;
    match kind {
        CastKind::Attack => {
            state.units[attacker_idx].cooldown_remaining_ms =
                state.units[attacker_idx].attack_cooldown_ms;
        }
        CastKind::Ability => {
            state.units[attacker_idx].ability_cooldown_remaining_ms =
                state.units[attacker_idx].ability_cooldown_ms;
        }
        CastKind::Heal => {
            state.units[attacker_idx].heal_cooldown_remaining_ms =
                state.units[attacker_idx].heal_cooldown_ms;
        }
        CastKind::Control => {
            state.units[attacker_idx].control_cooldown_remaining_ms =
                state.units[attacker_idx].control_cooldown_ms;
        }
    }

    events.push(SimEvent::DamageApplied {
        tick,
        source_id: state.units[attacker_idx].id,
        target_id,
        amount: damage,
        target_hp_before: current_hp,
        target_hp_after: new_hp,
    });

    if new_hp == 0 {
        events.push(SimEvent::UnitDied {
            tick,
            unit_id: target_id,
        });
    }
}

fn find_unit_idx(state: &SimState, unit_id: u32) -> Option<usize> {
    state.units.iter().position(|unit| unit.id == unit_id)
}

fn target_in_range_for_kind(
    attacker_idx: usize,
    target_idx: usize,
    state: &SimState,
    kind: CastKind,
) -> bool {
    let attacker = &state.units[attacker_idx];
    let target = &state.units[target_idx];
    let range = match kind {
        CastKind::Attack => attacker.attack_range,
        // Small hysteresis margin for abilities to reduce edge oscillation.
        CastKind::Ability => attacker.ability_range * 0.95,
        CastKind::Heal => attacker.heal_range,
        CastKind::Control => attacker.control_range,
    };
    let dist = distance(attacker.position, target.position);
    matches!(
        dist.partial_cmp(&range),
        Some(Ordering::Less | Ordering::Equal)
    )
}

fn to_x100(v: f32) -> i32 {
    (v * 100.0).round() as i32
}

fn next_rand_u32(state: &mut SimState) -> u32 {
    let mut x = state.rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.rng_state = x;
    (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
}

fn compute_metrics(
    initial_state: &SimState,
    final_state: &SimState,
    scripted_intents: &[Vec<UnitIntent>],
    events: &[SimEvent],
    ticks: u32,
    dt_ms: u32,
) -> SimMetrics {
    let mut total_damage_by_unit: HashMap<u32, i32> = HashMap::new();
    let mut damage_taken_by_unit: HashMap<u32, i32> = HashMap::new();
    let mut total_healing_by_unit: HashMap<u32, i32> = HashMap::new();
    let mut movement_distance_x100_by_unit: HashMap<u32, i32> = HashMap::new();
    let mut in_range_ticks_by_unit: HashMap<u32, u32> = HashMap::new();
    let mut out_of_range_ticks_by_unit: HashMap<u32, u32> = HashMap::new();
    let mut chase_ticks_by_unit: HashMap<u32, u32> = HashMap::new();
    let mut target_switches_by_unit: HashMap<u32, u32> = HashMap::new();

    let mut unit_hp: HashMap<u32, i32> = initial_state.units.iter().map(|u| (u.id, u.hp)).collect();
    let max_hp_by_unit: HashMap<u32, i32> = initial_state
        .units
        .iter()
        .map(|u| (u.id, u.max_hp))
        .collect();

    let mut casts_started = 0_u32;
    let mut casts_completed = 0_u32;
    let mut casts_failed_out_of_range = 0_u32;
    let mut heals_started = 0_u32;
    let mut heals_completed = 0_u32;
    let mut blocked_cooldown_intents = 0_u32;
    let mut blocked_invalid_target_intents = 0_u32;
    let mut reposition_for_range_events = 0_u32;
    let mut dead_source_attack_intents = 0_u32;
    let mut attack_intents = 0_u32;
    let mut executed_attack_intents = 0_u32;
    let mut focus_fire_ticks = 0_u32;
    let mut max_targeters_on_single_target = 0_u32;
    let mut overkill_damage_total = 0_i32;
    let mut invariant_violations = 0_u32;
    let mut tick_to_first_death = None;

    let mut cast_started_tick_by_unit: HashMap<u32, u64> = HashMap::new();
    let mut last_hostile_target_by_unit: HashMap<u32, u32> = HashMap::new();
    let mut total_cast_delay_ticks = 0_u64;
    let mut resolved_casts = 0_u64;

    let mut state_for_range = initial_state.clone();
    for tick in 0..ticks {
        let intents = scripted_intents
            .get(tick as usize)
            .map_or(&[][..], |v| v.as_slice());
        let mut hostile_targeters_by_target: HashMap<u32, u32> = HashMap::new();

        for intent in intents {
            let (target_id, intent_kind) = match intent.action {
                IntentAction::Attack { target_id } => (target_id, CastKind::Attack),
                IntentAction::CastAbility { target_id } => (target_id, CastKind::Ability),
                IntentAction::CastHeal { target_id } => (target_id, CastKind::Heal),
                IntentAction::CastControl { target_id } => (target_id, CastKind::Control),
                _ => continue,
            };
            attack_intents += 1;

            let source_idx = find_unit_idx(&state_for_range, intent.unit_id);
            let target_idx = find_unit_idx(&state_for_range, target_id);

            if let Some(src_idx) = source_idx {
                let source = &state_for_range.units[src_idx];
                if source.hp <= 0 {
                    dead_source_attack_intents += 1;
                    continue;
                }

                if let Some(tgt_idx) = target_idx {
                    let target = &state_for_range.units[tgt_idx];
                    let range = match intent_kind {
                        CastKind::Attack => source.attack_range,
                        CastKind::Ability => source.ability_range,
                        CastKind::Heal => source.heal_range,
                        CastKind::Control => source.control_range,
                    };
                    let dist = distance(source.position, target.position);
                    if dist <= range {
                        *in_range_ticks_by_unit.entry(source.id).or_insert(0) += 1;
                    } else {
                        *out_of_range_ticks_by_unit.entry(source.id).or_insert(0) += 1;
                        *chase_ticks_by_unit.entry(source.id).or_insert(0) += 1;
                    }

                    if target.team != source.team
                        && matches!(
                            intent_kind,
                            CastKind::Attack | CastKind::Ability | CastKind::Control
                        )
                    {
                        *hostile_targeters_by_target.entry(target.id).or_insert(0) += 1;
                        if let Some(last_target) = last_hostile_target_by_unit.get(&source.id) {
                            if *last_target != target.id {
                                *target_switches_by_unit.entry(source.id).or_insert(0) += 1;
                            }
                        }
                        last_hostile_target_by_unit.insert(source.id, target.id);
                    }
                }
            }
        }

        let max_targeters_this_tick = hostile_targeters_by_target
            .values()
            .copied()
            .max()
            .unwrap_or(0);
        if max_targeters_this_tick >= 2 {
            focus_fire_ticks += 1;
        }
        max_targeters_on_single_target =
            max_targeters_on_single_target.max(max_targeters_this_tick);

        let (next_state, _) = step(state_for_range, intents, dt_ms);
        state_for_range = next_state;
    }

    for event in events {
        match *event {
            SimEvent::Moved {
                unit_id,
                from_x100,
                from_y100,
                to_x100,
                to_y100,
                ..
            } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                let dx = (to_x100 - from_x100) as f32;
                let dy = (to_y100 - from_y100) as f32;
                let d = (dx * dx + dy * dy).sqrt().round() as i32;
                *movement_distance_x100_by_unit.entry(unit_id).or_insert(0) += d;
            }
            SimEvent::CastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                casts_started += 1;
                executed_attack_intents += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::HealCastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                heals_started += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::AbilityCastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                casts_started += 1;
                executed_attack_intents += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::ControlCastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                casts_started += 1;
                executed_attack_intents += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::CastFailedOutOfRange { tick, unit_id, .. } => {
                casts_failed_out_of_range += 1;
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&unit_id) {
                    total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    resolved_casts += 1;
                }
            }
            SimEvent::DamageApplied {
                tick,
                source_id,
                target_id,
                amount,
                target_hp_before,
                target_hp_after,
            } => {
                if unit_hp.get(&source_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                if target_hp_after > target_hp_before {
                    invariant_violations += 1;
                }
                if target_hp_after < 0 {
                    invariant_violations += 1;
                }
                if let Some(max_hp) = max_hp_by_unit.get(&target_id) {
                    if target_hp_after > *max_hp {
                        invariant_violations += 1;
                    }
                }

                casts_completed += 1;
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&source_id) {
                    total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    resolved_casts += 1;
                }

                *total_damage_by_unit.entry(source_id).or_insert(0) += amount;
                *damage_taken_by_unit.entry(target_id).or_insert(0) += amount;

                let tracked_before = unit_hp.get(&target_id).copied().unwrap_or(target_hp_before);
                if amount > tracked_before {
                    overkill_damage_total += amount - tracked_before;
                }
                unit_hp.insert(target_id, target_hp_after);
            }
            SimEvent::UnitDied { tick, unit_id } => {
                if tick_to_first_death.is_none() {
                    tick_to_first_death = Some(tick);
                }
                if unit_hp.get(&unit_id).copied().unwrap_or(0) != 0 {
                    invariant_violations += 1;
                }
            }
            SimEvent::AttackBlockedCooldown { .. } => {
                blocked_cooldown_intents += 1;
            }
            SimEvent::AbilityBlockedCooldown { .. } => {
                blocked_cooldown_intents += 1;
            }
            SimEvent::AttackBlockedInvalidTarget { .. } => {
                blocked_invalid_target_intents += 1;
            }
            SimEvent::AbilityBlockedInvalidTarget { .. } => {
                blocked_invalid_target_intents += 1;
            }
            SimEvent::AttackRepositioned { .. } => {
                reposition_for_range_events += 1;
            }
            SimEvent::AbilityBlockedOutOfRange { .. } => {
                casts_failed_out_of_range += 1;
            }
            SimEvent::HealBlockedCooldown { .. } => {
                blocked_cooldown_intents += 1;
            }
            SimEvent::HealBlockedInvalidTarget { .. } => {
                blocked_invalid_target_intents += 1;
            }
            SimEvent::HealBlockedOutOfRange { .. } => {
                casts_failed_out_of_range += 1;
            }
            SimEvent::ControlBlockedCooldown { .. } => {
                blocked_cooldown_intents += 1;
            }
            SimEvent::ControlBlockedInvalidTarget { .. } => {
                blocked_invalid_target_intents += 1;
            }
            SimEvent::ControlBlockedOutOfRange { .. } => {
                casts_failed_out_of_range += 1;
            }
            SimEvent::ControlApplied {
                tick,
                source_id,
                target_id,
                ..
            } => {
                if unit_hp.get(&source_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                if unit_hp.get(&target_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&source_id) {
                    total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    resolved_casts += 1;
                }
            }
            SimEvent::UnitControlled { .. } => {}
            SimEvent::HealApplied {
                tick,
                source_id,
                target_id,
                amount,
                target_hp_before,
                target_hp_after,
            } => {
                if unit_hp.get(&source_id).copied().unwrap_or(0) <= 0 {
                    invariant_violations += 1;
                }
                if target_hp_after < target_hp_before {
                    invariant_violations += 1;
                }
                if let Some(max_hp) = max_hp_by_unit.get(&target_id) {
                    if target_hp_after > *max_hp {
                        invariant_violations += 1;
                    }
                }
                heals_completed += 1;
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&source_id) {
                    total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    resolved_casts += 1;
                }
                *total_healing_by_unit.entry(source_id).or_insert(0) += amount;
                unit_hp.insert(target_id, target_hp_after);
            }
        }
    }

    for unit in &final_state.units {
        if unit.hp < 0 || unit.hp > unit.max_hp {
            invariant_violations += 1;
        }
    }

    let mut hero_alive = 0_usize;
    let mut enemy_alive = 0_usize;
    for unit in &final_state.units {
        if unit.hp > 0 {
            match unit.team {
                Team::Hero => hero_alive += 1,
                Team::Enemy => enemy_alive += 1,
            }
        }
    }

    let winner = match (hero_alive > 0, enemy_alive > 0) {
        (true, false) => Some(Team::Hero),
        (false, true) => Some(Team::Enemy),
        _ => None,
    };

    let avg_cast_delay_ms = if resolved_casts == 0 {
        0.0
    } else {
        ((total_cast_delay_ticks as f32 / resolved_casts as f32) * dt_ms as f32 * 100.0).round()
            / 100.0
    };

    let dps_by_unit = compute_dps_by_unit(ticks, dt_ms, &total_damage_by_unit);
    let total_damage_by_unit = map_to_sorted_vec_i32(total_damage_by_unit.into_iter());
    let damage_taken_by_unit = map_to_sorted_vec_i32(damage_taken_by_unit.into_iter());
    let total_healing_by_unit = map_to_sorted_vec_i32(total_healing_by_unit.into_iter());

    SimMetrics {
        ticks_elapsed: ticks,
        seconds_elapsed: (ticks as f32 * dt_ms as f32) / 1000.0,
        winner,
        tick_to_first_death,
        final_hp_by_unit: map_to_sorted_vec_i32(final_state.units.iter().map(|u| (u.id, u.hp))),
        total_damage_by_unit,
        damage_taken_by_unit,
        dps_by_unit,
        overkill_damage_total,
        casts_started,
        casts_completed,
        casts_failed_out_of_range,
        avg_cast_delay_ms,
        heals_started,
        heals_completed,
        total_healing_by_unit,
        attack_intents,
        executed_attack_intents,
        blocked_cooldown_intents,
        blocked_invalid_target_intents,
        dead_source_attack_intents,
        reposition_for_range_events,
        focus_fire_ticks,
        max_targeters_on_single_target,
        target_switches_by_unit: map_to_sorted_vec_u32(target_switches_by_unit.into_iter()),
        movement_distance_x100_by_unit: map_to_sorted_vec_i32(
            movement_distance_x100_by_unit.into_iter(),
        ),
        in_range_ticks_by_unit: map_to_sorted_vec_u32(in_range_ticks_by_unit.into_iter()),
        out_of_range_ticks_by_unit: map_to_sorted_vec_u32(out_of_range_ticks_by_unit.into_iter()),
        chase_ticks_by_unit: map_to_sorted_vec_u32(chase_ticks_by_unit.into_iter()),
        invariant_violations,
    }
}

fn map_to_sorted_vec_i32(iter: impl Iterator<Item = (u32, i32)>) -> Vec<(u32, i32)> {
    let mut items = iter.collect::<Vec<_>>();
    items.sort_by_key(|(id, _)| *id);
    items
}

fn map_to_sorted_vec_u32(iter: impl Iterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let mut items = iter.collect::<Vec<_>>();
    items.sort_by_key(|(id, _)| *id);
    items
}

fn compute_dps_by_unit(
    ticks: u32,
    dt_ms: u32,
    damage_by_unit: &HashMap<u32, i32>,
) -> Vec<(u32, f32)> {
    let elapsed = (ticks as f32 * dt_ms as f32) / 1000.0;
    let mut values = damage_by_unit
        .iter()
        .map(|(unit_id, damage)| {
            let dps = if elapsed <= f32::EPSILON {
                0.0
            } else {
                *damage as f32 / elapsed
            };
            (*unit_id, (dps * 100.0).round() / 100.0)
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|(unit_id, _)| *unit_id);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_hash_is_stable_for_same_seed() {
        let ticks = 120;
        let script = sample_duel_script(ticks);

        let result_a = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
        let result_b = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);

        assert_eq!(result_a.event_log_hash, result_b.event_log_hash);
        assert_eq!(result_a.final_state_hash, result_b.final_state_hash);
        assert_eq!(
            result_a.per_tick_state_hashes,
            result_b.per_tick_state_hashes
        );
        assert_eq!(result_a.events, result_b.events);
    }

    #[test]
    fn replay_hash_changes_with_different_seed() {
        let ticks = 120;
        let script = sample_duel_script(ticks);

        let result_a = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
        let result_b = run_replay(sample_duel_state(13), &script, ticks, FIXED_TICK_MS);

        assert_ne!(result_a.event_log_hash, result_b.event_log_hash);
        assert_ne!(result_a.final_state_hash, result_b.final_state_hash);
    }

    #[test]
    fn attack_requires_range_and_uses_movement() {
        let initial = SimState {
            tick: 0,
            rng_state: 1,
            units: vec![
                UnitState {
                    id: 1,
                    team: Team::Hero,
                    hp: 50,
                    max_hp: 50,
                    position: sim_vec2(0.0, 0.0),
                    move_speed_per_sec: 5.0,
                    attack_damage: 10,
                    attack_range: 1.0,
                    attack_cooldown_ms: 500,
                    attack_cast_time_ms: 200,
                    cooldown_remaining_ms: 0,
                    ability_damage: 0,
                    ability_range: 0.0,
                    ability_cooldown_ms: 0,
                    ability_cast_time_ms: 0,
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
                },
                UnitState {
                    id: 2,
                    team: Team::Enemy,
                    hp: 50,
                    max_hp: 50,
                    position: sim_vec2(10.0, 0.0),
                    move_speed_per_sec: 0.0,
                    attack_damage: 0,
                    attack_range: 0.0,
                    attack_cooldown_ms: 0,
                    attack_cast_time_ms: 0,
                    cooldown_remaining_ms: 0,
                    ability_damage: 0,
                    ability_range: 0.0,
                    ability_cooldown_ms: 0,
                    ability_cast_time_ms: 0,
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
                },
            ],
        };

        let intent = [UnitIntent {
            unit_id: 1,
            action: IntentAction::Attack { target_id: 2 },
        }];
        let (_, events) = step(initial, &intent, FIXED_TICK_MS);
        assert!(events.iter().any(|e| matches!(e, SimEvent::Moved { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, SimEvent::AttackRepositioned { .. })));
        assert!(!events
            .iter()
            .any(|e| matches!(e, SimEvent::CastStarted { .. })));
    }

    #[test]
    fn metrics_include_core_verification_signals() {
        let ticks = 120;
        let script = sample_duel_script(ticks);
        let result = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);

        assert_eq!(result.metrics.ticks_elapsed, ticks);
        assert!(result.metrics.seconds_elapsed > 0.0);
        assert!(result.metrics.attack_intents > 0);
        assert!(result.metrics.casts_started > 0);
        assert!(result.metrics.casts_completed > 0);
        assert!(result.metrics.avg_cast_delay_ms > 0.0);
        assert!(!result.metrics.dps_by_unit.is_empty());
        assert_eq!(result.metrics.invariant_violations, 0);
        assert!(!result.metrics.final_hp_by_unit.is_empty());
    }

    #[test]
    fn sample_duel_regression_snapshot() {
        let ticks = 120;
        let script = sample_duel_script(ticks);
        let result = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);

        assert_eq!(result.event_log_hash, 0xcaa9_9255_8277_ba4d);
        assert_eq!(result.final_state_hash, 0xf7f0_48a8_2dd7_b88e);
        assert_eq!(result.metrics.winner, Some(Team::Hero));
        assert_eq!(result.metrics.tick_to_first_death, Some(73));
        assert_eq!(result.metrics.final_hp_by_unit, vec![(1, 37), (2, 0)]);
        assert_eq!(result.metrics.invariant_violations, 0);
    }

    #[test]
    fn small_param_mutation_changes_hash() {
        let ticks = 120;
        let script = sample_duel_script(ticks);
        let baseline = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
        let mut altered = sample_duel_state(7);
        altered.units[0].attack_damage += 1;
        let mutated = run_replay(altered, &script, ticks, FIXED_TICK_MS);
        assert_ne!(baseline.event_log_hash, mutated.event_log_hash);
    }

    #[test]
    fn deterministic_tie_break_for_identical_targets() {
        let mut state = sample_duel_state(5);
        state.units = vec![
            UnitState {
                id: 1,
                team: Team::Hero,
                hp: 100,
                max_hp: 100,
                position: sim_vec2(0.0, 0.0),
                move_speed_per_sec: 0.0,
                attack_damage: 10,
                attack_range: 2.0,
                attack_cooldown_ms: 0,
                attack_cast_time_ms: 0,
                cooldown_remaining_ms: 0,
                ability_damage: 0,
                ability_range: 0.0,
                ability_cooldown_ms: 0,
                ability_cast_time_ms: 0,
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
            },
            UnitState {
                id: 2,
                team: Team::Enemy,
                hp: 40,
                max_hp: 40,
                position: sim_vec2(1.0, 0.0),
                move_speed_per_sec: 0.0,
                attack_damage: 0,
                attack_range: 0.0,
                attack_cooldown_ms: 0,
                attack_cast_time_ms: 0,
                cooldown_remaining_ms: 0,
                ability_damage: 0,
                ability_range: 0.0,
                ability_cooldown_ms: 0,
                ability_cast_time_ms: 0,
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
            },
            UnitState {
                id: 3,
                team: Team::Enemy,
                hp: 40,
                max_hp: 40,
                position: sim_vec2(1.0, 0.0),
                move_speed_per_sec: 0.0,
                attack_damage: 0,
                attack_range: 0.0,
                attack_cooldown_ms: 0,
                attack_cast_time_ms: 0,
                cooldown_remaining_ms: 0,
                ability_damage: 0,
                ability_range: 0.0,
                ability_cooldown_ms: 0,
                ability_cast_time_ms: 0,
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
            },
        ];
        state.units.sort_by_key(|u| u.id);
        let intent = [UnitIntent {
            unit_id: 1,
            action: IntentAction::Attack { target_id: 2 },
        }];
        let (_, events_a) = step(state.clone(), &intent, FIXED_TICK_MS);
        let (_, events_b) = step(state, &intent, FIXED_TICK_MS);
        assert_eq!(events_a, events_b);
    }

    #[test]
    fn fuzz_invariants_hold_across_seed_sweep() {
        for seed in 1_u64..16_u64 {
            let ticks = 80;
            let script = sample_duel_script(ticks);
            let result = run_replay(sample_duel_state(seed), &script, ticks, FIXED_TICK_MS);
            assert_eq!(result.metrics.invariant_violations, 0);
            assert_eq!(result.per_tick_state_hashes.len(), ticks as usize);
        }
    }

    #[test]
    fn control_cast_locks_target_actions_temporarily() {
        let mut state = SimState {
            tick: 0,
            rng_state: 11,
            units: vec![
                UnitState {
                    id: 1,
                    team: Team::Hero,
                    hp: 100,
                    max_hp: 100,
                    position: sim_vec2(0.0, 0.0),
                    move_speed_per_sec: 0.0,
                    attack_damage: 0,
                    attack_range: 0.0,
                    attack_cooldown_ms: 0,
                    attack_cast_time_ms: 0,
                    cooldown_remaining_ms: 0,
                    ability_damage: 0,
                    ability_range: 0.0,
                    ability_cooldown_ms: 0,
                    ability_cast_time_ms: 0,
                    ability_cooldown_remaining_ms: 0,
                    heal_amount: 0,
                    heal_range: 0.0,
                    heal_cooldown_ms: 0,
                    heal_cast_time_ms: 0,
                    heal_cooldown_remaining_ms: 0,
                    control_range: 2.0,
                    control_duration_ms: 350,
                    control_cooldown_ms: 1_000,
                    control_cast_time_ms: 0,
                    control_cooldown_remaining_ms: 0,
                    control_remaining_ms: 0,
                    casting: None,
                },
                UnitState {
                    id: 2,
                    team: Team::Enemy,
                    hp: 100,
                    max_hp: 100,
                    position: sim_vec2(1.0, 0.0),
                    move_speed_per_sec: 0.0,
                    attack_damage: 10,
                    attack_range: 2.0,
                    attack_cooldown_ms: 0,
                    attack_cast_time_ms: 0,
                    cooldown_remaining_ms: 0,
                    ability_damage: 0,
                    ability_range: 0.0,
                    ability_cooldown_ms: 0,
                    ability_cast_time_ms: 0,
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
                },
            ],
        };
        state.units.sort_by_key(|u| u.id);

        let intents_t1 = vec![
            UnitIntent {
                unit_id: 1,
                action: IntentAction::CastControl { target_id: 2 },
            },
            UnitIntent {
                unit_id: 2,
                action: IntentAction::Attack { target_id: 1 },
            },
        ];
        let (state, events_t1) = step(state, &intents_t1, FIXED_TICK_MS);
        assert!(events_t1
            .iter()
            .any(|e| matches!(e, SimEvent::ControlCastStarted { .. })));

        let intents_t2 = vec![UnitIntent {
            unit_id: 2,
            action: IntentAction::Attack { target_id: 1 },
        }];
        let (_, events_t2) = step(state, &intents_t2, FIXED_TICK_MS);
        assert!(events_t2
            .iter()
            .any(|e| matches!(e, SimEvent::ControlApplied { target_id: 2, .. })));
        assert!(events_t2
            .iter()
            .any(|e| matches!(e, SimEvent::UnitControlled { unit_id: 2, .. })));
    }
}
