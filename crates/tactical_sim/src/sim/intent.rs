use crate::effects::{AbilityTarget, StatusKind, Trigger};

use super::types::*;
use super::events::SimEvent;
use super::math::distance;
use super::helpers::{find_unit_idx, is_alive, target_in_range_for_kind};
use super::resolve::try_start_cast;
use super::hero::resolve_hero_ability;
use super::triggers::check_passive_triggers;

pub fn try_start_attack(
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

pub fn try_start_ability(
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
            tick, unit_id: state.units[attacker_idx].id, target_id,
        });
        return;
    };

    if !is_alive(&state.units[target_idx]) {
        events.push(SimEvent::AbilityBlockedInvalidTarget {
            tick, unit_id: state.units[attacker_idx].id, target_id,
        });
        return;
    }

    if !target_in_range_for_kind(attacker_idx, target_idx, state, CastKind::Ability) {
        events.push(SimEvent::AbilityBlockedOutOfRange {
            tick, unit_id: state.units[attacker_idx].id, target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        target_pos: None,
        remaining_ms: state.units[attacker_idx].ability_cast_time_ms,
        kind: CastKind::Ability,
        area: None,
        ability_index: None,
        effect_hint: CastEffectHint::Damage,
    };
    state.units[attacker_idx].casting = Some(cast);
    events.push(SimEvent::AbilityCastStarted {
        tick, unit_id: state.units[attacker_idx].id, target_id,
    });
}

pub fn try_start_heal(
    healer_idx: usize,
    target_id: u32,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.units[healer_idx].heal_cooldown_remaining_ms > 0 {
        events.push(SimEvent::HealBlockedCooldown {
            tick, unit_id: state.units[healer_idx].id, target_id,
            cooldown_remaining_ms: state.units[healer_idx].heal_cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::HealBlockedInvalidTarget {
            tick, unit_id: state.units[healer_idx].id, target_id,
        });
        return;
    };

    if !is_alive(&state.units[target_idx])
        || state.units[target_idx].team != state.units[healer_idx].team
        || state.units[target_idx].hp >= state.units[target_idx].max_hp
    {
        events.push(SimEvent::HealBlockedInvalidTarget {
            tick, unit_id: state.units[healer_idx].id, target_id,
        });
        return;
    }

    if !target_in_range_for_kind(healer_idx, target_idx, state, CastKind::Heal) {
        events.push(SimEvent::HealBlockedOutOfRange {
            tick, unit_id: state.units[healer_idx].id, target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        target_pos: None,
        remaining_ms: state.units[healer_idx].heal_cast_time_ms,
        kind: CastKind::Heal,
        area: None,
        ability_index: None,
        effect_hint: CastEffectHint::Heal,
    };
    state.units[healer_idx].casting = Some(cast);
    events.push(SimEvent::HealCastStarted {
        tick, unit_id: state.units[healer_idx].id, target_id,
    });
}

pub fn try_start_control(
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
            tick, unit_id: state.units[caster_idx].id, target_id,
        });
        return;
    }
    if state.units[caster_idx].control_cooldown_remaining_ms > 0 {
        events.push(SimEvent::ControlBlockedCooldown {
            tick, unit_id: state.units[caster_idx].id, target_id,
            cooldown_remaining_ms: state.units[caster_idx].control_cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::ControlBlockedInvalidTarget {
            tick, unit_id: state.units[caster_idx].id, target_id,
        });
        return;
    };
    if !is_alive(&state.units[target_idx])
        || state.units[target_idx].team == state.units[caster_idx].team
    {
        events.push(SimEvent::ControlBlockedInvalidTarget {
            tick, unit_id: state.units[caster_idx].id, target_id,
        });
        return;
    }
    if !target_in_range_for_kind(caster_idx, target_idx, state, CastKind::Control) {
        events.push(SimEvent::ControlBlockedOutOfRange {
            tick, unit_id: state.units[caster_idx].id, target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        target_pos: None,
        remaining_ms: state.units[caster_idx].control_cast_time_ms,
        kind: CastKind::Control,
        area: None,
        ability_index: None,
        effect_hint: CastEffectHint::CrowdControl,
    };
    state.units[caster_idx].casting = Some(cast);
    events.push(SimEvent::ControlCastStarted {
        tick, unit_id: state.units[caster_idx].id, target_id,
    });
}

pub fn try_start_hero_ability(
    caster_idx: usize,
    ability_index: usize,
    target: AbilityTarget,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    let unit_id = state.units[caster_idx].id;

    if state.units[caster_idx].status_effects.iter().any(|s|
        matches!(s.kind, StatusKind::Silence | StatusKind::Polymorph | StatusKind::Banish)
    ) {
        return;
    }

    let Some(slot) = state.units[caster_idx].abilities.get(ability_index) else {
        return;
    };

    if !slot.is_ready() {
        return;
    }

    // Extract what we need and drop the borrow
    let is_toggle = slot.def.is_toggle;
    let resource_cost = slot.def.resource_cost;
    let cast_time_ms = slot.def.cast_time_ms;
    let ability_name = slot.def.name.clone();
    let _ = slot; // release immutable borrow

    // Toggle abilities: flip on/off and return
    if is_toggle {
        if let Some(s) = state.units[caster_idx].abilities.get_mut(ability_index) {
            s.toggled_on = !s.toggled_on;
        }
        events.push(SimEvent::AbilityUsed {
            tick, unit_id, ability_index,
            ability_name,
        });
        // If toggling on, apply the initial effects
        if state.units[caster_idx].abilities.get(ability_index).map_or(false, |s| s.toggled_on) {
            resolve_hero_ability(caster_idx, ability_index, target, tick, state, events);
        }
        return;
    }

    if resource_cost > 0 && state.units[caster_idx].resource < resource_cost {
        return;
    }

    let (target_id, target_pos) = match target {
        AbilityTarget::Unit(id) => (id, None),
        AbilityTarget::Position(pos) => (0, Some(pos)),
        AbilityTarget::None => (unit_id, None),
    };

    if let AbilityTarget::Unit(tid) = target {
        if let Some(target_idx) = find_unit_idx(state, tid) {
            if !is_alive(&state.units[target_idx]) {
                return;
            }
            let range = state.units[caster_idx].abilities[ability_index].def.range;
            if range > 0.0 {
                let dist = distance(
                    state.units[caster_idx].position,
                    state.units[target_idx].position,
                );
                if dist > range {
                    return;
                }
            }
        } else {
            return;
        }
    }

    if cast_time_ms > 0 {
        // Enrich CastState with ability info for spatial awareness
        let ability_area = state.units[caster_idx].abilities.get(ability_index)
            .and_then(|slot| {
                // Extract area from the ability's effects or delivery
                slot.def.effects.first().and_then(|ce| ce.area)
            });
        let effect_hint = state.units[caster_idx].abilities.get(ability_index)
            .map(|slot| CastEffectHint::from_ability_def(&slot.def))
            .unwrap_or(CastEffectHint::Unknown);

        let cast = CastState {
            target_id,
            target_pos,
            remaining_ms: cast_time_ms,
            kind: CastKind::HeroAbility(ability_index),
            area: ability_area,
            ability_index: Some(ability_index),
            effect_hint,
        };
        state.units[caster_idx].casting = Some(cast);
    } else {
        resolve_hero_ability(caster_idx, ability_index, target, tick, state, events);
    }

    events.push(SimEvent::AbilityUsed {
        tick, unit_id, ability_index, ability_name,
    });

    state.units[caster_idx].status_effects.retain(|s| {
        if let StatusKind::Stealth { break_on_ability, .. } = s.kind {
            !break_on_ability
        } else { true }
    });

    check_passive_triggers(Trigger::OnAbilityUsed, caster_idx, unit_id, tick, state, events);
}
