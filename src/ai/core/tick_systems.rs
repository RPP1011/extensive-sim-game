use crate::ai::effects::{AbilityTarget, StatusKind, Trigger};

use super::types::*;
use super::events::SimEvent;
use super::math::distance;
use super::helpers::{clear_dead_unit_state, find_unit_idx, find_lowest_hp_ally_in_range};
use super::conditions::evaluate_condition_tracked;
use super::apply_effect::apply_effect;
use super::targeting::resolve_targets;
use super::triggers::check_passive_triggers;

/// Decrement cooldowns on all hero ability/passive slots.
pub fn tick_hero_cooldowns(state: &mut SimState, dt_ms: u32) {
    for unit in &mut state.units {
        if unit.hp <= 0 {
            continue;
        }
        let cdr_factor: f32 = unit.status_effects.iter()
            .filter_map(|s| {
                if let StatusKind::Buff { ref stat, factor } = s.kind {
                    if stat == "cooldown_reduction" { return Some(factor); }
                }
                None
            })
            .fold(0.0f32, |acc, f| acc + f);
        let effective_dt = if cdr_factor > 0.0 {
            (dt_ms as f32 * (1.0 + cdr_factor)) as u32
        } else {
            dt_ms
        };

        for slot in &mut unit.abilities {
            if slot.cooldown_remaining_ms > 0 {
                slot.cooldown_remaining_ms = slot.cooldown_remaining_ms.saturating_sub(effective_dt);
            }
            if slot.morph_remaining_ms > 0 {
                slot.morph_remaining_ms = slot.morph_remaining_ms.saturating_sub(dt_ms);
                if slot.morph_remaining_ms == 0 {
                    if let Some(base) = slot.base_def.take() {
                        slot.def = *base;
                    }
                }
            }
            // Charge recharge
            if slot.def.max_charges > 0 && slot.charges < slot.def.max_charges {
                if slot.charge_recharge_remaining_ms > 0 {
                    slot.charge_recharge_remaining_ms = slot.charge_recharge_remaining_ms.saturating_sub(effective_dt);
                }
                if slot.charge_recharge_remaining_ms == 0 {
                    slot.charges += 1;
                    if slot.charges < slot.def.max_charges {
                        slot.charge_recharge_remaining_ms = slot.def.charge_recharge_ms;
                    }
                }
            }
            // Recast window countdown
            if slot.recast_window_remaining_ms > 0 {
                slot.recast_window_remaining_ms = slot.recast_window_remaining_ms.saturating_sub(dt_ms);
                if slot.recast_window_remaining_ms == 0 {
                    // Recast window expired — put on full cooldown
                    slot.recasts_remaining = 0;
                    slot.cooldown_remaining_ms = slot.def.cooldown_ms;
                }
            }
        }
        // Toggle drain: consume resource per second while toggled on
        for slot in &mut unit.abilities {
            if slot.toggled_on && slot.def.toggle_cost_per_sec > 0.0 {
                let cost = (slot.def.toggle_cost_per_sec * dt_ms as f32 / 1000.0) as i32;
                if unit.resource >= cost {
                    unit.resource -= cost;
                } else {
                    // Not enough resource — auto-toggle off
                    slot.toggled_on = false;
                }
            }
        }
        for slot in &mut unit.passives {
            if slot.cooldown_remaining_ms > 0 {
                slot.cooldown_remaining_ms = slot.cooldown_remaining_ms.saturating_sub(effective_dt);
            }
        }

        if unit.max_resource > 0 && unit.resource_regen_per_sec > 0.0 {
            let regen = (unit.resource_regen_per_sec * dt_ms as f32 / 1000.0) as i32;
            unit.resource = (unit.resource + regen).min(unit.max_resource);
        }
    }
}

/// Tick all active status effects: DoT/HoT damage/heal, duration countdown, expiry.
pub fn tick_status_effects(state: &mut SimState, tick: u64, dt_ms: u32, events: &mut Vec<SimEvent>) {
    let mut deferred_triggers: Vec<(Trigger, usize, u32)> = Vec::new();

    for idx in 0..state.units.len() {
        if state.units[idx].hp <= 0 {
            continue;
        }

        let unit_id = state.units[idx].id;

        let mut i = 0;
        while i < state.units[idx].status_effects.len() {
            state.units[idx].status_effects[i].remaining_ms = state.units[idx]
                .status_effects[i].remaining_ms.saturating_sub(dt_ms);

            enum TickAction {
                None,
                DotTick { dmg: i32, source_id: u32 },
                HotTick { heal: i32, source_id: u32 },
                ShieldExpire { amount: i32 },
            }

            let remaining = state.units[idx].status_effects[i].remaining_ms;
            let source_id = state.units[idx].status_effects[i].source_id;
            let is_stun = matches!(state.units[idx].status_effects[i].kind, StatusKind::Stun);
            let action = match &mut state.units[idx].status_effects[i].kind {
                StatusKind::Dot { amount_per_tick, tick_interval_ms, tick_elapsed_ms } => {
                    *tick_elapsed_ms += dt_ms;
                    if *tick_elapsed_ms >= *tick_interval_ms {
                        *tick_elapsed_ms -= *tick_interval_ms;
                        TickAction::DotTick { dmg: *amount_per_tick, source_id }
                    } else {
                        TickAction::None
                    }
                }
                StatusKind::Hot { amount_per_tick, tick_interval_ms, tick_elapsed_ms } => {
                    *tick_elapsed_ms += dt_ms;
                    if *tick_elapsed_ms >= *tick_interval_ms {
                        *tick_elapsed_ms -= *tick_interval_ms;
                        TickAction::HotTick { heal: *amount_per_tick, source_id }
                    } else {
                        TickAction::None
                    }
                }
                StatusKind::Shield { amount } => {
                    if *amount <= 0 || remaining == 0 {
                        TickAction::ShieldExpire { amount: *amount }
                    } else {
                        TickAction::None
                    }
                }
                _ => TickAction::None,
            };

            match action {
                TickAction::DotTick { dmg, source_id } => {
                    let hp_before = state.units[idx].hp;
                    state.units[idx].hp = (hp_before - dmg).max(0);
                    let hp_after = state.units[idx].hp;
                    events.push(SimEvent::DamageApplied {
                        tick, source_id, target_id: unit_id,
                        amount: hp_before - hp_after,
                        target_hp_before: hp_before, target_hp_after: hp_after,
                    });
                    if hp_after == 0 {
                        clear_dead_unit_state(&mut state.units[idx]);
                        events.push(SimEvent::UnitDied { tick, unit_id });
                        deferred_triggers.push((Trigger::OnDeath, idx, source_id));
                        if let Some(src_idx) = state.units.iter().position(|u| u.id == source_id) {
                            deferred_triggers.push((Trigger::OnKill, src_idx, unit_id));
                        }
                    } else {
                        deferred_triggers.push((Trigger::OnDamageTaken, idx, source_id));
                        let max_hp = state.units[idx].max_hp;
                        let pct = (hp_after as f32 / max_hp as f32) * 100.0;
                        deferred_triggers.push((Trigger::OnHpBelow { percent: pct }, idx, source_id));
                    }
                }
                TickAction::HotTick { heal, source_id } => {
                    let hp_before = state.units[idx].hp;
                    let max_hp = state.units[idx].max_hp;
                    state.units[idx].hp = (hp_before + heal).min(max_hp);
                    let hp_after = state.units[idx].hp;
                    events.push(SimEvent::HealApplied {
                        tick, source_id, target_id: unit_id,
                        amount: hp_after - hp_before,
                        target_hp_before: hp_before, target_hp_after: hp_after,
                    });
                    if hp_after > hp_before {
                        let pct = (hp_after as f32 / max_hp as f32) * 100.0;
                        deferred_triggers.push((Trigger::OnHpAbove { percent: pct }, idx, source_id));
                    }
                }
                TickAction::ShieldExpire { amount } => {
                    state.units[idx].shield_hp = (state.units[idx].shield_hp - amount.max(0)).max(0);
                    events.push(SimEvent::StatusEffectExpired {
                        tick, unit_id, effect_name: format!("shield"),
                    });
                    state.units[idx].status_effects.remove(i);
                    continue;
                }
                TickAction::None => {}
            }

            if remaining == 0 {
                if is_stun {
                    deferred_triggers.push((Trigger::OnStunExpire, idx, unit_id));
                }

                enum ExpireAction {
                    None,
                    AbsorbHeal { heal: i32, shield_sub: i32 },
                    DeathMarkBurst { burst: i32 },
                    RestoreTeam { team: Team },
                }
                let expire_action = match &state.units[idx].status_effects[i].kind {
                    StatusKind::AbsorbShield { amount, heal_percent } => {
                        let heal = (*amount as f32 * heal_percent / 100.0) as i32;
                        ExpireAction::AbsorbHeal { heal, shield_sub: (*amount).max(0) }
                    }
                    StatusKind::DeathMark { accumulated_damage, damage_percent } => {
                        let burst = (*accumulated_damage as f32 * damage_percent / 100.0) as i32;
                        ExpireAction::DeathMarkBurst { burst }
                    }
                    StatusKind::Charm { original_team } => {
                        ExpireAction::RestoreTeam { team: *original_team }
                    }
                    _ => ExpireAction::None,
                };
                match expire_action {
                    ExpireAction::AbsorbHeal { heal, shield_sub } => {
                        if heal > 0 {
                            let hp_before = state.units[idx].hp;
                            let max_hp = state.units[idx].max_hp;
                            state.units[idx].hp = (hp_before + heal).min(max_hp);
                            events.push(SimEvent::HealApplied {
                                tick, source_id: unit_id, target_id: unit_id,
                                amount: state.units[idx].hp - hp_before,
                                target_hp_before: hp_before,
                                target_hp_after: state.units[idx].hp,
                            });
                        }
                        state.units[idx].shield_hp = (state.units[idx].shield_hp - shield_sub).max(0);
                    }
                    ExpireAction::DeathMarkBurst { burst } => {
                        if burst > 0 {
                            let hp_before = state.units[idx].hp;
                            state.units[idx].hp = (hp_before - burst).max(0);
                            events.push(SimEvent::DeathMarkDetonated {
                                tick, unit_id, damage: burst,
                            });
                            if state.units[idx].hp == 0 {
                                clear_dead_unit_state(&mut state.units[idx]);
                                events.push(SimEvent::UnitDied { tick, unit_id });
                            }
                        }
                    }
                    ExpireAction::RestoreTeam { team } => {
                        state.units[idx].team = team;
                    }
                    ExpireAction::None => {}
                }

                deferred_triggers.push((Trigger::OnStatusExpired, idx, unit_id));

                let effect_name = format!("{:?}", state.units[idx].status_effects[i].kind);
                events.push(SimEvent::StatusEffectExpired {
                    tick, unit_id, effect_name,
                });
                state.units[idx].status_effects.remove(i);
            } else {
                i += 1;
            }
        }
    }

    for (trigger, unit_idx, context_id) in deferred_triggers {
        if unit_idx < state.units.len() && state.units[unit_idx].hp > 0 {
            check_passive_triggers(trigger, unit_idx, context_id, tick, state, events);
        }
    }

    // Sync attached units to their host positions
    let attach_pairs: Vec<(u32, u32)> = state.units.iter()
        .flat_map(|u| u.status_effects.iter().filter_map(move |s| {
            if let StatusKind::Attached { host_id } = s.kind {
                Some((u.id, host_id))
            } else { None }
        }))
        .collect();
    for (attached_id, host_id) in attach_pairs {
        let host_pos = state.units.iter().find(|u| u.id == host_id).map(|u| u.position);
        if let Some(pos) = host_pos {
            if let Some(a_idx) = state.units.iter().position(|u| u.id == attached_id) {
                state.units[a_idx].position = pos;
            }
        }
    }
}

/// Advance all in-flight projectiles.
pub fn advance_projectiles(state: &mut SimState, tick: u64, dt_ms: u32, events: &mut Vec<SimEvent>) {
    let mut i = 0;
    while i < state.projectiles.len() {
        let dt_sec = dt_ms as f32 / 1000.0;
        let move_dist = state.projectiles[i].speed * dt_sec;

        state.projectiles[i].position.x += state.projectiles[i].direction.x * move_dist;
        state.projectiles[i].position.y += state.projectiles[i].direction.y * move_dist;
        state.projectiles[i].distance_traveled += move_dist;

        let proj_pos = state.projectiles[i].position;
        let target_pos = state.projectiles[i].target_position;
        let source_id = state.projectiles[i].source_id;

        if state.projectiles[i].pierce {
            let width = state.projectiles[i].width;
            let already_hit = state.projectiles[i].already_hit.clone();
            let mut hit_ids = Vec::new();

            for unit in &state.units {
                if unit.hp <= 0 || unit.id == source_id || already_hit.contains(&unit.id) {
                    continue;
                }
                if distance(proj_pos, unit.position) <= width + 0.5 {
                    hit_ids.push(unit.id);
                }
            }

            for hit_id in &hit_ids {
                state.projectiles[i].already_hit.push(*hit_id);
                events.push(SimEvent::ProjectileHit { tick, target_id: *hit_id });
                let on_hit = state.projectiles[i].on_hit.clone();
                let source_idx = find_unit_idx(state, source_id);
                let target_idx = find_unit_idx(state, *hit_id);
                if let (Some(src), Some(_tgt)) = (source_idx, target_idx) {
                    for ce in &on_hit {
                        if evaluate_condition_tracked(&ce.condition, src, AbilityTarget::Unit(*hit_id), state, tick, events) {
                            apply_effect(&ce.effect, src, *hit_id, AbilityTarget::Unit(*hit_id), tick, &ce.tags, ce.stacking, state, events);
                        }
                    }
                }
            }
        }

        let dist_to_target = distance(proj_pos, target_pos);
        let is_skillshot = state.projectiles[i].max_travel_distance > 0.0;
        if dist_to_target <= move_dist + 0.5 || (is_skillshot && state.projectiles[i].distance_traveled >= state.projectiles[i].max_travel_distance) {
            events.push(SimEvent::ProjectileArrived { tick });

            let on_arrival = state.projectiles[i].on_arrival.clone();
            let source_idx = find_unit_idx(state, source_id);
            let _target_id = state.projectiles[i].target_id;
            if let Some(src) = source_idx {
                for ce in &on_arrival {
                    let caster_team = state.units[src].team;
                    let targets = resolve_targets(ce.area.as_ref(), src, AbilityTarget::Position(target_pos), caster_team, &ce.effect, state);
                    for tid in &targets {
                        if evaluate_condition_tracked(&ce.condition, src, AbilityTarget::Unit(*tid), state, tick, events) {
                            apply_effect(&ce.effect, src, *tid, AbilityTarget::Position(target_pos), tick, &ce.tags, ce.stacking, state, events);
                        }
                    }
                }
            }

            state.projectiles.remove(i);
        } else {
            i += 1;
        }
    }
}

/// Fire periodic passives whose interval has elapsed.
pub fn tick_periodic_passives(state: &mut SimState, tick: u64, dt_ms: u32, events: &mut Vec<SimEvent>) {
    for idx in 0..state.units.len() {
        if state.units[idx].hp <= 0 {
            continue;
        }

        for p_idx in 0..state.units[idx].passives.len() {
            if state.units[idx].passives[p_idx].cooldown_remaining_ms > 0 {
                continue;
            }

            let trigger = state.units[idx].passives[p_idx].def.trigger.clone();
            if let Trigger::Periodic { interval_ms } = trigger {
                let elapsed = state.units[idx].passives[p_idx].periodic_elapsed_ms + dt_ms;
                if elapsed >= interval_ms {
                    let effects = state.units[idx].passives[p_idx].def.effects.clone();
                    let passive_range = state.units[idx].passives[p_idx].def.range;
                    let passive_name = state.units[idx].passives[p_idx].def.name.clone();
                    let unit_id = state.units[idx].id;
                    let cd = state.units[idx].passives[p_idx].def.cooldown_ms;

                    events.push(SimEvent::PassiveTriggered { tick, unit_id, passive_name });

                    let target_id = find_lowest_hp_ally_in_range(state, idx, passive_range)
                        .unwrap_or(unit_id);

                    let caster_team = state.units[idx].team;
                    for ce in &effects {
                        let target = AbilityTarget::Unit(target_id);
                        if evaluate_condition_tracked(&ce.condition, idx, target, state, tick, events) {
                            if ce.area.is_some() {
                                let targets = resolve_targets(ce.area.as_ref(), idx, target, caster_team, &ce.effect, state);
                                for &tid in &targets {
                                    apply_effect(&ce.effect, idx, tid, target, tick, &ce.tags, ce.stacking, state, events);
                                }
                            } else {
                                apply_effect(&ce.effect, idx, target_id, target, tick, &ce.tags, ce.stacking, state, events);
                            }
                        }
                    }

                    state.units[idx].passives[p_idx].periodic_elapsed_ms = elapsed - interval_ms;
                    state.units[idx].passives[p_idx].cooldown_remaining_ms = cd;
                } else {
                    state.units[idx].passives[p_idx].periodic_elapsed_ms = elapsed;
                }
            }
        }
    }
}
