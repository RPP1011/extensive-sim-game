use crate::ai::effects::{AbilityTarget, Trigger};

use super::types::*;
use super::events::SimEvent;
use super::math::distance;
use super::helpers::{is_alive, find_unit_idx};
use super::conditions::evaluate_condition_tracked;
use super::apply_effect::apply_effect;

/// Check event-driven passive triggers after an event occurs.
pub fn check_passive_triggers(
    trigger_kind: Trigger,
    unit_idx: usize,
    context_target_id: u32,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.passive_trigger_depth >= 3 {
        return;
    }
    state.passive_trigger_depth += 1;

    let unit_pos = state.units[unit_idx].position;

    for p_idx in 0..state.units[unit_idx].passives.len() {
        if state.units[unit_idx].passives[p_idx].cooldown_remaining_ms > 0 {
            continue;
        }

        let passive_trigger = state.units[unit_idx].passives[p_idx].def.trigger.clone();
        let trigger_matches = match (passive_trigger, trigger_kind.clone()) {
            (Trigger::OnDamageDealt, Trigger::OnDamageDealt) => true,
            (Trigger::OnDamageTaken, Trigger::OnDamageTaken) => true,
            (Trigger::OnKill, Trigger::OnKill) => true,
            (Trigger::OnDeath, Trigger::OnDeath) => true,
            (Trigger::OnAbilityUsed, Trigger::OnAbilityUsed) => true,
            (Trigger::OnHpBelow { percent: threshold }, Trigger::OnHpBelow { percent: current_pct }) => {
                current_pct <= threshold
            }
            (Trigger::OnHpAbove { percent: threshold }, Trigger::OnHpAbove { percent: current_pct }) => {
                current_pct >= threshold
            }
            (Trigger::OnShieldBroken, Trigger::OnShieldBroken) => true,
            (Trigger::OnStunExpire, Trigger::OnStunExpire) => true,
            (Trigger::OnAllyDamaged { range }, Trigger::OnAllyDamaged { .. }) => {
                if let Some(ally_idx) = find_unit_idx(state, context_target_id) {
                    distance(unit_pos, state.units[ally_idx].position) <= range
                } else {
                    false
                }
            }
            (Trigger::OnHealReceived, Trigger::OnHealReceived) => true,
            (Trigger::OnStatusApplied, Trigger::OnStatusApplied) => true,
            (Trigger::OnStatusExpired, Trigger::OnStatusExpired) => true,
            (Trigger::OnResurrect, Trigger::OnResurrect) => true,
            (Trigger::OnDodge, Trigger::OnDodge) => true,
            (Trigger::OnReflect, Trigger::OnReflect) => true,
            (Trigger::OnAllyKilled { range }, Trigger::OnAllyKilled { .. }) => {
                if let Some(ally_idx) = find_unit_idx(state, context_target_id) {
                    distance(unit_pos, state.units[ally_idx].position) <= range
                } else {
                    false
                }
            }
            (Trigger::OnAutoAttack, Trigger::OnAutoAttack) => true,
            (Trigger::OnStackReached { ref name, count: threshold }, Trigger::OnStackReached { name: ref fired_name, count: fired_count }) => {
                name == fired_name && fired_count >= threshold
            }
            _ => false,
        };

        if !trigger_matches {
            continue;
        }

        let effects = state.units[unit_idx].passives[p_idx].def.effects.clone();
        let unit_id = state.units[unit_idx].id;
        let passive_name = state.units[unit_idx].passives[p_idx].def.name.clone();
        let cd = state.units[unit_idx].passives[p_idx].def.cooldown_ms;

        events.push(SimEvent::PassiveTriggered {
            tick,
            unit_id,
            passive_name,
        });

        state.units[unit_idx].passives[p_idx].cooldown_remaining_ms = cd;

        for ce in &effects {
            if evaluate_condition_tracked(&ce.condition, unit_idx, AbilityTarget::Unit(context_target_id), state, tick, events) {
                apply_effect(&ce.effect, unit_idx, context_target_id, AbilityTarget::Unit(context_target_id), tick, &ce.tags, ce.stacking, state, events);
            }
        }
    }

    state.passive_trigger_depth -= 1;
}

/// Fire all passive triggers after a damage event.
pub fn fire_damage_triggers(
    source_idx: usize,
    target_idx: usize,
    target_id: u32,
    new_hp: i32,
    shield_broke: bool,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    let source_id = state.units[source_idx].id;
    let target_team = state.units[target_idx].team;

    // OnDamageDealt on attacker
    if is_alive(&state.units[source_idx]) {
        check_passive_triggers(Trigger::OnDamageDealt, source_idx, target_id, tick, state, events);
    }

    // OnDamageTaken on target
    if is_alive(&state.units[target_idx]) {
        check_passive_triggers(Trigger::OnDamageTaken, target_idx, source_id, tick, state, events);
    }

    // OnAllyDamaged on all teammates of target
    let teammate_indices: Vec<usize> = state.units.iter().enumerate()
        .filter(|(i, u)| *i != target_idx && u.team == target_team && u.hp > 0)
        .map(|(i, _)| i)
        .collect();
    for ally_idx in teammate_indices {
        check_passive_triggers(
            Trigger::OnAllyDamaged { range: f32::MAX },
            ally_idx,
            target_id,
            tick,
            state,
            events,
        );
    }

    // OnHpBelow on target if still alive
    if new_hp > 0 {
        let max_hp = state.units[target_idx].max_hp;
        let pct = (new_hp as f32 / max_hp as f32) * 100.0;
        check_passive_triggers(Trigger::OnHpBelow { percent: pct }, target_idx, source_id, tick, state, events);
    }

    // OnShieldBroken on target if shield went from >0 to 0
    if shield_broke && is_alive(&state.units[target_idx]) {
        check_passive_triggers(Trigger::OnShieldBroken, target_idx, source_id, tick, state, events);
    }

    // OnKill on attacker + OnDeath on target if killed
    if new_hp == 0 {
        check_passive_triggers(Trigger::OnKill, source_idx, target_id, tick, state, events);
        check_passive_triggers(Trigger::OnDeath, target_idx, source_id, tick, state, events);
    }
}
