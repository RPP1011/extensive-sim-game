use crate::ai::effects::{AbilityTarget, Area, Effect, StatusKind};

use super::types::*;
use super::events::SimEvent;
use super::math::distance;
use super::helpers::find_unit_idx;
use super::conditions::evaluate_condition_tracked;
use super::targeting::resolve_targets;
use super::apply_effect::apply_effect;

/// Unblock cells owned by the expiring zone, but only if no other active zone
/// also claims that cell.
fn unblock_zone_cells(zone_idx: usize, state: &mut SimState) {
    if state.zones[zone_idx].blocked_cells.is_empty() {
        return;
    }
    let Some(ref mut nav) = state.grid_nav else { return };
    let cells = std::mem::take(&mut state.zones[zone_idx].blocked_cells);
    for cell in &cells {
        let still_blocked = state.zones.iter().enumerate().any(|(j, z)| {
            j != zone_idx && z.blocked_cells.contains(cell)
        });
        if !still_blocked {
            nav.blocked.remove(cell);
        }
    }
    // Restore cells vec so the zone struct is valid until removal
    state.zones[zone_idx].blocked_cells = cells;
}

/// Tick all active zones: apply effects at interval, expire when done.
pub fn tick_zones(state: &mut SimState, tick: u64, dt_ms: u32, events: &mut Vec<SimEvent>) {
    let mut i = 0;
    while i < state.zones.len() {
        state.zones[i].remaining_ms = state.zones[i].remaining_ms.saturating_sub(dt_ms);

        if state.zones[i].arm_time_ms > 0 {
            state.zones[i].arm_time_ms = state.zones[i].arm_time_ms.saturating_sub(dt_ms);
            if state.zones[i].remaining_ms == 0 {
                unblock_zone_cells(i, state);
                events.push(SimEvent::ZoneExpired { tick, zone_id: state.zones[i].id });
                state.zones.remove(i);
                continue;
            }
            i += 1;
            continue;
        }

        // Trap logic
        if state.zones[i].trigger_on_enter && !state.zones[i].triggered {
            let zone_pos = state.zones[i].position;
            let zone_team = state.zones[i].source_team;
            let source_id = state.zones[i].source_id;
            let area = &state.zones[i].area;
            let radius = match area {
                Area::Circle { radius } => *radius,
                _ => 2.0,
            };
            let has_enemy = state.units.iter().any(|u| {
                u.hp > 0 && u.team != zone_team && distance(zone_pos, u.position) <= radius
            });
            if has_enemy {
                state.zones[i].triggered = true;
                let effects = state.zones[i].effects.clone();
                let source_idx = find_unit_idx(state, source_id);
                if let Some(src) = source_idx {
                    let targets: Vec<u32> = state.units.iter()
                        .filter(|u| u.hp > 0 && u.team != zone_team && distance(zone_pos, u.position) <= radius)
                        .map(|u| u.id)
                        .collect();
                    for ce in &effects {
                        for &tid in &targets {
                            if evaluate_condition_tracked(&ce.condition, src, AbilityTarget::Unit(tid), state, tick, events) {
                                apply_effect(&ce.effect, src, tid, AbilityTarget::Unit(tid), tick, &ce.tags, ce.stacking, state, events);
                            }
                        }
                    }
                }
                unblock_zone_cells(i, state);
                let zone_id = state.zones[i].id;
                events.push(SimEvent::ZoneExpired { tick, zone_id });
                state.zones.remove(i);
                continue;
            }
        }

        // Regular zone tick
        if !state.zones[i].trigger_on_enter && state.zones[i].tick_interval_ms > 0 {
            state.zones[i].tick_elapsed_ms += dt_ms;
            if state.zones[i].tick_elapsed_ms >= state.zones[i].tick_interval_ms {
                state.zones[i].tick_elapsed_ms -= state.zones[i].tick_interval_ms;
                let zone_id = state.zones[i].id;
                let zone_pos = state.zones[i].position;
                let zone_team = state.zones[i].source_team;
                let source_id = state.zones[i].source_id;
                let effects = state.zones[i].effects.clone();
                let area = state.zones[i].area.clone();

                events.push(SimEvent::ZoneTick { tick, zone_id });

                let source_idx = find_unit_idx(state, source_id);
                if let Some(src) = source_idx {
                    let radius = match &area {
                        Area::Circle { radius } => *radius,
                        _ => 2.0,
                    };
                    let targets_allies = effects.iter().any(|ce| matches!(ce.effect,
                        Effect::Heal { .. } | Effect::Shield { .. } | Effect::Buff { .. }
                    ));
                    let targets: Vec<u32> = state.units.iter()
                        .filter(|u| u.hp > 0)
                        .filter(|u| if targets_allies { u.team == zone_team } else { u.team != zone_team })
                        .filter(|u| distance(zone_pos, u.position) <= radius)
                        .map(|u| u.id)
                        .collect();
                    for ce in &effects {
                        for &tid in &targets {
                            if evaluate_condition_tracked(&ce.condition, src, AbilityTarget::Unit(tid), state, tick, events) {
                                apply_effect(&ce.effect, src, tid, AbilityTarget::Unit(tid), tick, &ce.tags, ce.stacking, state, events);
                            }
                        }
                    }
                }
            }
        }

        if state.zones[i].remaining_ms == 0 {
            unblock_zone_cells(i, state);
            let zone_id = state.zones[i].id;
            events.push(SimEvent::ZoneExpired { tick, zone_id });
            state.zones.remove(i);
        } else {
            i += 1;
        }
    }
}

/// Tick all active channels.
pub fn tick_channels(state: &mut SimState, tick: u64, dt_ms: u32, events: &mut Vec<SimEvent>) {
    for idx in 0..state.units.len() {
        let Some(mut channel) = state.units[idx].channeling.take() else {
            continue;
        };

        if state.units[idx].hp <= 0 {
            events.push(SimEvent::ChannelInterrupted { tick, unit_id: state.units[idx].id });
            continue;
        }

        let interrupted = state.units[idx].status_effects.iter().any(|s| matches!(s.kind,
            StatusKind::Stun | StatusKind::Silence | StatusKind::Fear { .. }
            | StatusKind::Polymorph | StatusKind::Banish
        ));
        if interrupted {
            events.push(SimEvent::ChannelInterrupted { tick, unit_id: state.units[idx].id });
            continue;
        }

        channel.remaining_ms = channel.remaining_ms.saturating_sub(dt_ms);

        if channel.tick_interval_ms > 0 {
            channel.tick_elapsed_ms += dt_ms;
            if channel.tick_elapsed_ms >= channel.tick_interval_ms {
                channel.tick_elapsed_ms -= channel.tick_interval_ms;
                let unit_id = state.units[idx].id;
                events.push(SimEvent::ChannelTick { tick, unit_id });

                let ability_index = channel.ability_index;
                let effects = state.units[idx].abilities.get(ability_index)
                    .map(|s| s.def.effects.clone())
                    .unwrap_or_default();
                let target = if let Some(pos) = channel.target_pos {
                    AbilityTarget::Position(pos)
                } else if channel.target_id == unit_id {
                    AbilityTarget::None
                } else {
                    AbilityTarget::Unit(channel.target_id)
                };
                let caster_team = state.units[idx].team;
                for ce in &effects {
                    if evaluate_condition_tracked(&ce.condition, idx, target, state, tick, events) {
                        let targets = resolve_targets(ce.area.as_ref(), idx, target, caster_team, &ce.effect, state);
                        for &tid in &targets {
                            apply_effect(&ce.effect, idx, tid, target, tick, &ce.tags, ce.stacking, state, events);
                        }
                    }
                }
            }
        }

        if channel.remaining_ms == 0 {
            events.push(SimEvent::ChannelCompleted { tick, unit_id: state.units[idx].id });
        } else {
            state.units[idx].channeling = Some(channel);
        }
    }
}

/// Tick all active tethers.
pub fn tick_tethers(state: &mut SimState, tick: u64, dt_ms: u32, events: &mut Vec<SimEvent>) {
    let mut i = 0;
    while i < state.tethers.len() {
        let tether = &mut state.tethers[i];
        tether.remaining_ms = tether.remaining_ms.saturating_sub(dt_ms);

        let source_id = tether.source_id;
        let target_id = tether.target_id;
        let source_idx = find_unit_idx(state, source_id);
        let target_idx = find_unit_idx(state, target_id);

        let should_break = match (source_idx, target_idx) {
            (Some(si), Some(ti)) => {
                if state.units[si].hp <= 0 || state.units[ti].hp <= 0 {
                    true
                } else {
                    distance(state.units[si].position, state.units[ti].position) > state.tethers[i].max_range
                }
            }
            _ => true,
        };

        if should_break {
            events.push(SimEvent::TetherBroken { tick, source_id, target_id });
            state.tethers.remove(i);
            continue;
        }

        let tether = &mut state.tethers[i];
        if tether.tick_interval_ms > 0 {
            tether.tick_elapsed_ms += dt_ms;
            if tether.tick_elapsed_ms >= tether.tick_interval_ms {
                tether.tick_elapsed_ms -= tether.tick_interval_ms;
                let effects = tether.tick_effects.clone();
                let src = source_idx.unwrap();
                for ce in &effects {
                    if evaluate_condition_tracked(&ce.condition, src, AbilityTarget::Unit(target_id), state, tick, events) {
                        apply_effect(&ce.effect, src, target_id, AbilityTarget::Unit(target_id), tick, &ce.tags, ce.stacking, state, events);
                    }
                }
            }
        }

        if state.tethers[i].remaining_ms == 0 {
            let on_complete = state.tethers[i].on_complete.clone();
            let src = source_idx.unwrap();
            for ce in &on_complete {
                if evaluate_condition_tracked(&ce.condition, src, AbilityTarget::Unit(target_id), state, tick, events) {
                    apply_effect(&ce.effect, src, target_id, AbilityTarget::Unit(target_id), tick, &ce.tags, ce.stacking, state, events);
                }
            }
            events.push(SimEvent::TetherCompleted { tick, source_id, target_id });
            state.tethers.remove(i);
        } else {
            i += 1;
        }
    }
}
