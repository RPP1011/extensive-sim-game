use crate::ai::effects::{
    AbilityTarget, Area, ConditionalEffect, Delivery,
};

use super::super::types::*;
use super::super::events::SimEvent;
use super::super::helpers::{find_unit_idx, check_tags_resisted};
use super::super::conditions::evaluate_condition_tracked;
use super::super::targeting::resolve_targets;
use super::super::apply_effect::apply_effect;
use super::super::damage::resolve_chain_delivery;
use super::reactions::{check_zone_reactions, apply_morph, apply_form_swap};

pub fn resolve_hero_ability(
    caster_idx: usize,
    ability_index: usize,
    target: AbilityTarget,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    let Some(slot) = state.units[caster_idx].abilities.get(ability_index).cloned() else {
        return;
    };

    // Put ability on cooldown (or consume a charge)
    if let Some(s) = state.units[caster_idx].abilities.get_mut(ability_index) {
        if s.def.max_charges > 0 {
            // Charge-based: consume one charge, start recharge if not already recharging
            s.charges = s.charges.saturating_sub(1);
            if s.charge_recharge_remaining_ms == 0 && s.charges < s.def.max_charges {
                s.charge_recharge_remaining_ms = s.def.charge_recharge_ms;
            }
        } else if s.def.recast_count > 0 && s.recasts_remaining == 0 {
            // First cast of a recast ability — open recast window instead of full cooldown
            s.recasts_remaining = s.def.recast_count;
            s.recast_window_remaining_ms = s.def.recast_window_ms;
        } else if s.recasts_remaining > 0 {
            // Mid-recast: consume one recast
            s.recasts_remaining -= 1;
            if s.recasts_remaining == 0 {
                // All recasts used — put on full cooldown
                s.recast_window_remaining_ms = 0;
                s.cooldown_remaining_ms = s.def.cooldown_ms;
            }
        } else {
            s.cooldown_remaining_ms = s.def.cooldown_ms;
        }
    }

    let caster_id = state.units[caster_idx].id;
    let caster_pos = state.units[caster_idx].position;
    let caster_team = state.units[caster_idx].team;

    // Determine which recast index we're on (0 = first cast, 1+ = recasts)
    let recast_index = if slot.def.recast_count > 0 {
        let remaining_after = state.units[caster_idx].abilities.get(ability_index)
            .map(|s| s.recasts_remaining).unwrap_or(0);
        // We already decremented above, so figure out which recast we just consumed
        if remaining_after < slot.def.recast_count {
            (slot.def.recast_count - remaining_after) as usize
        } else {
            0 // first cast (recasts_remaining was just set)
        }
    } else {
        0
    };

    // Deduct resource cost
    let resource_cost = slot.def.resource_cost;
    if resource_cost > 0 {
        state.units[caster_idx].resource -= resource_cost;
    }

    // Select effects: use recast_effects if mid-recast, otherwise normal effects
    let active_effects: &Vec<ConditionalEffect> = if recast_index > 0 && !slot.def.recast_effects.is_empty() {
        let re_idx = (recast_index - 1).min(slot.def.recast_effects.len() - 1);
        &slot.def.recast_effects[re_idx]
    } else {
        &slot.def.effects
    };
    // Clone so we can use it across the branches below
    let active_effects = active_effects.clone();

    // --- Zone delivery ---
    if let Some(Delivery::Zone { duration_ms, tick_interval_ms }) = slot.def.delivery {
        let zone_pos = match target {
            AbilityTarget::Position(p) => p,
            AbilityTarget::Unit(tid) => find_unit_idx(state, tid)
                .map(|i| state.units[i].position)
                .unwrap_or(caster_pos),
            AbilityTarget::None => caster_pos,
        };
        let zone_id = state.tick as u32 * 1000 + caster_id;
        let area = active_effects.first()
            .and_then(|ce| ce.area.clone())
            .unwrap_or(Area::Circle { radius: 2.0 });
        // Compute blocked cells for Obstacle effects
        let mut blocked_cells = Vec::new();
        for ce in &active_effects {
            if let crate::ai::effects::Effect::Obstacle { width, height } = &ce.effect {
                if let Some(ref mut nav) = state.grid_nav {
                    let cells = nav.cells_in_rect(zone_pos, *width, *height);
                    for &cell in &cells {
                        nav.blocked.insert(cell);
                    }
                    blocked_cells.extend(cells);
                }
            }
        }
        let zone_tag = slot.def.zone_tag.clone();
        state.zones.push(ActiveZone {
            id: zone_id,
            source_id: caster_id,
            source_team: caster_team,
            position: zone_pos,
            area,
            effects: active_effects.clone(),
            remaining_ms: duration_ms,
            tick_interval_ms,
            tick_elapsed_ms: 0,
            trigger_on_enter: false,
            invisible: false,
            triggered: false,
            arm_time_ms: 0,
            blocked_cells,
            zone_tag: zone_tag.clone(),
        });
        events.push(SimEvent::ZoneCreated { tick, zone_id, source_id: caster_id });

        // Zone reactions: check for overlapping tagged zones from the same caster
        if let Some(ref new_tag) = zone_tag {
            check_zone_reactions(state, caster_id, caster_team, zone_pos, new_tag, tick, events);
        }

        apply_morph(caster_idx, ability_index, &slot, state);
        return;
    }

    // --- Trap delivery ---
    if let Some(Delivery::Trap { duration_ms, trigger_radius, arm_time_ms }) = slot.def.delivery {
        let zone_pos = match target {
            AbilityTarget::Position(p) => p,
            AbilityTarget::Unit(tid) => find_unit_idx(state, tid)
                .map(|i| state.units[i].position)
                .unwrap_or(caster_pos),
            AbilityTarget::None => caster_pos,
        };
        let zone_id = state.tick as u32 * 1000 + caster_id;
        state.zones.push(ActiveZone {
            id: zone_id,
            source_id: caster_id,
            source_team: caster_team,
            position: zone_pos,
            area: Area::Circle { radius: trigger_radius },
            effects: active_effects.clone(),
            remaining_ms: duration_ms,
            tick_interval_ms: 0,
            tick_elapsed_ms: 0,
            trigger_on_enter: true,
            invisible: true,
            triggered: false,
            arm_time_ms,
            blocked_cells: Vec::new(),
            zone_tag: slot.def.zone_tag.clone(),
        });
        events.push(SimEvent::ZoneCreated { tick, zone_id, source_id: caster_id });
        apply_morph(caster_idx, ability_index, &slot, state);
        return;
    }

    // --- Channel delivery ---
    if let Some(Delivery::Channel { duration_ms, tick_interval_ms }) = slot.def.delivery {
        let ch_target_id = match target {
            AbilityTarget::Unit(tid) => tid,
            _ => caster_id,
        };
        let ch_target_pos = match target {
            AbilityTarget::Position(p) => Some(p),
            _ => None,
        };
        state.units[caster_idx].channeling = Some(ChannelState {
            ability_index,
            target_id: ch_target_id,
            target_pos: ch_target_pos,
            remaining_ms: duration_ms,
            tick_interval_ms,
            tick_elapsed_ms: 0,
        });
        events.push(SimEvent::ChannelStarted {
            tick,
            unit_id: caster_id,
            ability_name: slot.def.name.clone(),
        });
        apply_morph(caster_idx, ability_index, &slot, state);
        return;
    }

    // --- Tether delivery ---
    if let Some(Delivery::Tether { max_range, tick_interval_ms, ref on_complete }) = slot.def.delivery {
        let tether_target = match target {
            AbilityTarget::Unit(tid) => tid,
            _ => { apply_morph(caster_idx, ability_index, &slot, state); return; }
        };
        state.tethers.push(ActiveTether {
            source_id: caster_id,
            target_id: tether_target,
            remaining_ms: slot.def.cooldown_ms.max(3000),
            max_range,
            tick_effects: active_effects.clone(),
            on_complete: on_complete.clone(),
            tick_interval_ms,
            tick_elapsed_ms: 0,
        });
        events.push(SimEvent::TetherFormed { tick, source_id: caster_id, target_id: tether_target });
        apply_morph(caster_idx, ability_index, &slot, state);
        return;
    }

    // Chain delivery
    if let Some(Delivery::Chain {
        bounces, bounce_range, falloff, ref on_hit,
    }) = slot.def.delivery
    {
        resolve_chain_delivery(
            caster_idx, target, bounces, bounce_range, falloff,
            on_hit, &active_effects, tick, state, events,
        );
        apply_morph(caster_idx, ability_index, &slot, state);
        return;
    }

    // Projectile delivery
    if let Some(Delivery::Projectile {
        speed, pierce, width, ref on_hit, ref on_arrival,
    }) = slot.def.delivery
    {
        let (proj_target_id, proj_target_pos, proj_max_travel) = match target {
            AbilityTarget::Unit(tid) => {
                let tpos = find_unit_idx(state, tid)
                    .map(|i| state.units[i].position)
                    .unwrap_or(caster_pos);
                (tid, tpos, 0.0)
            }
            AbilityTarget::Position(pos) => {
                let dx = pos.x - caster_pos.x;
                let dy = pos.y - caster_pos.y;
                let len = (dx * dx + dy * dy).sqrt().max(f32::EPSILON);
                let range = slot.def.range.max(1.0);
                let endpoint = sim_vec2(
                    caster_pos.x + (dx / len) * range,
                    caster_pos.y + (dy / len) * range,
                );
                (0, endpoint, range)
            }
            AbilityTarget::None => {
                apply_morph(caster_idx, ability_index, &slot, state);
                return;
            }
        };

        let dx = proj_target_pos.x - caster_pos.x;
        let dy = proj_target_pos.y - caster_pos.y;
        let len = (dx * dx + dy * dy).sqrt().max(f32::EPSILON);
        let dir = sim_vec2(dx / len, dy / len);

        use crate::ai::effects::Projectile;
        state.projectiles.push(Projectile {
            source_id: caster_id,
            target_id: proj_target_id,
            position: caster_pos,
            direction: dir,
            speed,
            pierce,
            width,
            on_hit: on_hit.clone(),
            on_arrival: on_arrival.clone(),
            already_hit: vec![],
            target_position: proj_target_pos,
            max_travel_distance: proj_max_travel,
            distance_traveled: 0.0,
        });
        events.push(SimEvent::ProjectileSpawned {
            tick,
            source_id: caster_id,
            target_id: proj_target_id,
        });
        apply_morph(caster_idx, ability_index, &slot, state);
        return;
    }

    // Dispatch instant effects
    for ce in &active_effects {
        if !evaluate_condition_tracked(&ce.condition, caster_idx, target, state, tick, events) {
            continue;
        }

        let targets = resolve_targets(
            ce.area.as_ref(),
            caster_idx,
            target,
            caster_team,
            &ce.effect,
            state,
        );

        for &tid in &targets {
            if !ce.tags.is_empty() {
                if let Some(tidx) = find_unit_idx(state, tid) {
                    if check_tags_resisted(&ce.tags, &state.units[tidx].resistance_tags) {
                        events.push(SimEvent::EffectResisted {
                            tick,
                            unit_id: tid,
                            resisted_tag: ce.tags.keys().next().cloned().unwrap_or_default(),
                        });
                        continue;
                    }
                }
            }

            apply_effect(&ce.effect, caster_idx, tid, target, tick, &ce.tags, ce.stacking, state, events);
        }
    }

    apply_morph(caster_idx, ability_index, &slot, state);

    // Form swap: if this ability has swap_form, morph all abilities with matching form tag
    if let Some(ref form_tag) = slot.def.swap_form {
        apply_form_swap(caster_idx, form_tag, state);
    }
}
