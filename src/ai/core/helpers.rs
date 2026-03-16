use std::cmp::Ordering;

use contracts::*;
use crate::ai::effects::{ScalingTerm, StatRef, StatusKind};

use super::types::*;
use super::events::SimEvent;
use super::math::*;

pub fn collect_intents(intents: &[UnitIntent]) -> Vec<(u32, IntentAction)> {
    let mut stable = intents
        .iter()
        .map(|intent| (intent.unit_id, intent.action))
        .collect::<Vec<_>>();
    stable.sort_by_key(|entry| entry.0);
    stable
}

pub fn is_alive(unit: &UnitState) -> bool {
    unit.hp > 0
}

/// Check if `observer` can see `target`.
/// Non-stealthed units are always visible.
/// Stealthed units are only visible within close range (3 cells),
/// further reduced by the target's cover_bonus (near walls = harder to spot).
pub fn can_see(observer: &UnitState, target: &UnitState) -> bool {
    let is_stealthed = target.status_effects.iter()
        .any(|s| matches!(s.kind, crate::ai::effects::StatusKind::Stealth { .. }));
    if !is_stealthed {
        return true;
    }
    let dist = super::math::distance(observer.position, target.position);
    // Base detection range 3.0, reduced by cover
    let detection_range = 3.0 * (1.0 - target.cover_bonus);
    dist <= detection_range
}

/// Clear casting/channeling/CC state when a unit dies.
pub fn clear_dead_unit_state(unit: &mut UnitState) {
    unit.casting = None;
    unit.channeling = None;
    unit.control_remaining_ms = 0;
}

#[requires(idx < state.units.len())]
#[requires(target_pos.x.is_finite() && target_pos.y.is_finite())]
pub fn move_towards_position(
    idx: usize,
    target_pos: SimVec2,
    tick: u64,
    state: &mut SimState,
    dt_ms: u32,
    events: &mut Vec<SimEvent>,
) {
    let start = state.units[idx].position;
    let max_delta = state.units[idx].move_speed_per_sec * (dt_ms as f32 / 1000.0);
    let next = if let Some(ref nav) = state.grid_nav {
        // Fast path: try straight-line first; only A* if that lands in a blocked cell
        let direct = move_towards(start, target_pos, max_delta);
        if nav.is_walkable_pos(direct) {
            direct
        } else {
            let waypoint = crate::ai::pathing::next_waypoint(nav, start, target_pos);
            let raw = move_towards(start, waypoint, max_delta);
            crate::ai::pathing::clamp_step_to_walkable(nav, start, raw)
        }
    } else {
        move_towards(start, target_pos, max_delta)
    };
    if distance(start, next) <= f32::EPSILON {
        return;
    }
    state.units[idx].position = next;

    // Refresh elevation after movement (cover_bonus is direction-dependent
    // and refreshed per-tick in the scenario/sim_bridge loop)
    if let Some(ref nav) = state.grid_nav {
        state.units[idx].elevation = nav.elevation_at_pos(next);
    }

    events.push(SimEvent::Moved {
        tick,
        unit_id: state.units[idx].id,
        from_x100: to_x100(start.x),
        from_y100: to_x100(start.y),
        to_x100: to_x100(state.units[idx].position.x),
        to_y100: to_x100(state.units[idx].position.y),
    });
}

/// Clamp unit position within Leash radius if applicable.
pub fn clamp_leash(idx: usize, state: &mut SimState) {
    for s in &state.units[idx].status_effects {
        if let StatusKind::Leash { anchor_pos, max_range } = s.kind {
            let dist = distance(state.units[idx].position, anchor_pos);
            if dist > max_range {
                state.units[idx].position = position_at_range(state.units[idx].position, anchor_pos, max_range);
            }
            break;
        }
    }
}

pub fn find_unit_idx(state: &SimState, unit_id: u32) -> Option<usize> {
    state.units.iter().position(|unit| unit.id == unit_id)
}

pub fn target_in_range_for_kind(
    attacker_idx: usize,
    target_idx: usize,
    state: &SimState,
    kind: CastKind,
) -> bool {
    let attacker = &state.units[attacker_idx];
    let target = &state.units[target_idx];
    let range = match kind {
        CastKind::Attack => attacker.attack_range,
        CastKind::Ability => attacker.ability_range * 0.95,
        CastKind::Heal => attacker.heal_range,
        CastKind::Control => attacker.control_range,
        CastKind::HeroAbility(idx) => {
            attacker.abilities.get(idx).map_or(0.0, |slot| slot.def.range)
        }
    };
    let dist = distance(attacker.position, target.position);
    matches!(
        dist.partial_cmp(&range),
        Some(Ordering::Less | Ordering::Equal)
    )
}

pub fn to_x100(v: f32) -> i32 {
    (v * 100.0).round() as i32
}

pub fn next_rand_u32(state: &mut SimState) -> u32 {
    let mut x = state.rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.rng_state = x;
    (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
}

pub fn find_lowest_hp_ally_in_range(state: &SimState, unit_idx: usize, range: f32) -> Option<u32> {
    let unit = &state.units[unit_idx];
    state
        .units
        .iter()
        .filter(|u| {
            u.hp > 0
                && u.team == unit.team
                && u.id != unit.id
                && distance(unit.position, u.position) <= range
        })
        .min_by_key(|u| (u.hp * 100) / u.max_hp.max(1))
        .map(|u| u.id)
}

pub fn unit_has_status(unit: &UnitState, name: &str) -> bool {
    unit.status_effects.iter().any(|s| match (&s.kind, name) {
        (StatusKind::Stun, "stun") => true,
        (StatusKind::Slow { .. }, "slow") => true,
        (StatusKind::Root, "root") => true,
        (StatusKind::Silence, "silence") => true,
        (StatusKind::Fear { .. }, "fear") => true,
        (StatusKind::Taunt { .. }, "taunt") => true,
        (StatusKind::Reflect { .. }, "reflect") => true,
        (StatusKind::Lifesteal { .. }, "lifesteal") => true,
        (StatusKind::DamageModify { .. }, "damage_modify") => true,
        (StatusKind::Blind { .. }, "blind") => true,
        (StatusKind::Polymorph, "polymorph") => true,
        (StatusKind::Banish, "banish") => true,
        (StatusKind::Confuse, "confuse") => true,
        (StatusKind::Charm { .. }, "charm") => true,
        (StatusKind::Stealth { .. }, "stealth") => true,
        (StatusKind::Immunity { .. }, "immunity") => true,
        (StatusKind::DeathMark { .. }, "death_mark") => true,
        (StatusKind::Shield { .. }, "shield") => true,
        (StatusKind::Buff { .. }, "buff") => true,
        (StatusKind::Debuff { .. }, "debuff") => true,
        (StatusKind::Dot { .. }, "dot") => true,
        (StatusKind::Hot { .. }, "hot") => true,
        (StatusKind::Stacks { .. }, "stacks") => true,
        _ => false,
    })
}

pub fn check_tags_resisted(effect_tags: &crate::ai::effects::Tags, resistance_tags: &crate::ai::effects::Tags) -> bool {
    for (tag, &power) in effect_tags {
        if let Some(&resistance) = resistance_tags.get(tag) {
            if resistance >= power {
                return true;
            }
        }
    }
    false
}

pub fn compute_scaling(base: i32, stat: Option<&str>, pct: f32, caster_idx: usize, target_id: u32, state: &SimState) -> i32 {
    if pct == 0.0 || stat.is_none() {
        return base;
    }
    let stat_val = match stat.unwrap() {
        "caster_current_hp" => state.units[caster_idx].hp as f32,
        "caster_missing_hp" => (state.units[caster_idx].max_hp - state.units[caster_idx].hp) as f32,
        "caster_max_hp" => state.units[caster_idx].max_hp as f32,
        "target_current_hp" => {
            find_unit_idx(state, target_id).map_or(0.0, |i| state.units[i].hp as f32)
        }
        "target_missing_hp" => {
            find_unit_idx(state, target_id).map_or(0.0, |i| (state.units[i].max_hp - state.units[i].hp) as f32)
        }
        "target_max_hp" => {
            find_unit_idx(state, target_id).map_or(0.0, |i| state.units[i].max_hp as f32)
        }
        _ => 0.0,
    };
    base + (stat_val * pct / 100.0) as i32
}

/// Resolve composable scaling terms against the sim state.
/// Returns the total bonus to add to a base amount.
/// Each term contributes: stat_value * percent / 100, optionally capped by `max`.
/// If `consume` is set on a term referencing stacks, those stacks are removed.
pub fn resolve_bonus(terms: &[ScalingTerm], caster_idx: usize, target_id: u32, state: &mut SimState) -> i32 {
    if terms.is_empty() {
        return 0;
    }
    let mut total = 0i32;
    for term in terms {
        let stat_val = resolve_stat_ref(&term.stat, caster_idx, target_id, state);
        let mut contribution = (stat_val * term.percent / 100.0) as i32;
        if term.max > 0 {
            contribution = contribution.min(term.max);
        }
        total += contribution;
        if term.consume {
            consume_stacks(&term.stat, caster_idx, target_id, state);
        }
    }
    total
}

fn resolve_stat_ref(stat: &StatRef, caster_idx: usize, target_id: u32, state: &SimState) -> f32 {
    match stat {
        StatRef::CasterMaxHp => state.units[caster_idx].max_hp as f32,
        StatRef::CasterCurrentHp => state.units[caster_idx].hp as f32,
        StatRef::CasterMissingHp => (state.units[caster_idx].max_hp - state.units[caster_idx].hp) as f32,
        StatRef::TargetMaxHp => {
            find_unit_idx(state, target_id).map_or(0.0, |i| state.units[i].max_hp as f32)
        }
        StatRef::TargetCurrentHp => {
            find_unit_idx(state, target_id).map_or(0.0, |i| state.units[i].hp as f32)
        }
        StatRef::TargetMissingHp => {
            find_unit_idx(state, target_id).map_or(0.0, |i| (state.units[i].max_hp - state.units[i].hp) as f32)
        }
        StatRef::CasterAttackDamage => state.units[caster_idx].attack_damage as f32,
        StatRef::TargetStacks { ref name } => {
            find_unit_idx(state, target_id).map_or(0.0, |i| {
                state.units[i].status_effects.iter()
                    .find_map(|s| match &s.kind {
                        StatusKind::Stacks { name: n, count, .. } if n == name => Some(*count as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
        }
        StatRef::CasterStacks { ref name } => {
            state.units[caster_idx].status_effects.iter()
                .find_map(|s| match &s.kind {
                    StatusKind::Stacks { name: n, count, .. } if n == name => Some(*count as f32),
                    _ => None,
                })
                .unwrap_or(0.0)
        }
    }
}

fn consume_stacks(stat: &StatRef, _caster_idx: usize, target_id: u32, state: &mut SimState) {
    match stat {
        StatRef::TargetStacks { ref name } => {
            if let Some(i) = find_unit_idx(state, target_id) {
                state.units[i].status_effects.retain(|s| {
                    !matches!(&s.kind, StatusKind::Stacks { name: n, .. } if n == name)
                });
            }
        }
        StatRef::CasterStacks { ref name } => {
            state.units[_caster_idx].status_effects.retain(|s| {
                !matches!(&s.kind, StatusKind::Stacks { name: n, .. } if n == name)
            });
        }
        _ => {} // Only stack refs can be consumed
    }
}
