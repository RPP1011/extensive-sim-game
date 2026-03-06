use std::collections::HashMap;

use super::types::*;
use super::events::SimEvent;
use super::math::distance;
use super::helpers::find_unit_idx;
use super::simulation::step;

pub fn compute_metrics(
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
    let max_hp_by_unit: HashMap<u32, i32> = initial_state.units.iter().map(|u| (u.id, u.max_hp)).collect();

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
        let intents = scripted_intents.get(tick as usize).map_or(&[][..], |v| v.as_slice());
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
                        CastKind::HeroAbility(idx) => {
                            source.abilities.get(idx).map_or(0.0, |s| s.def.range)
                        }
                    };
                    let dist = distance(source.position, target.position);
                    if dist <= range {
                        *in_range_ticks_by_unit.entry(source.id).or_insert(0) += 1;
                    } else {
                        *out_of_range_ticks_by_unit.entry(source.id).or_insert(0) += 1;
                        *chase_ticks_by_unit.entry(source.id).or_insert(0) += 1;
                    }

                    if target.team != source.team
                        && matches!(intent_kind, CastKind::Attack | CastKind::Ability | CastKind::Control)
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

        let max_targeters_this_tick = hostile_targeters_by_target.values().copied().max().unwrap_or(0);
        if max_targeters_this_tick >= 2 {
            focus_fire_ticks += 1;
        }
        max_targeters_on_single_target = max_targeters_on_single_target.max(max_targeters_this_tick);

        let (next_state, _) = step(state_for_range, intents, dt_ms);
        state_for_range = next_state;
    }

    process_events(events, &mut unit_hp, &max_hp_by_unit,
        &mut total_damage_by_unit, &mut damage_taken_by_unit, &mut total_healing_by_unit,
        &mut movement_distance_x100_by_unit, &mut casts_started, &mut casts_completed,
        &mut casts_failed_out_of_range, &mut heals_started, &mut heals_completed,
        &mut blocked_cooldown_intents, &mut blocked_invalid_target_intents,
        &mut reposition_for_range_events, &mut executed_attack_intents,
        &mut overkill_damage_total, &mut invariant_violations, &mut tick_to_first_death,
        &mut cast_started_tick_by_unit, &mut total_cast_delay_ticks, &mut resolved_casts);

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
        ((total_cast_delay_ticks as f32 / resolved_casts as f32) * dt_ms as f32 * 100.0).round() / 100.0
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
        movement_distance_x100_by_unit: map_to_sorted_vec_i32(movement_distance_x100_by_unit.into_iter()),
        in_range_ticks_by_unit: map_to_sorted_vec_u32(in_range_ticks_by_unit.into_iter()),
        out_of_range_ticks_by_unit: map_to_sorted_vec_u32(out_of_range_ticks_by_unit.into_iter()),
        chase_ticks_by_unit: map_to_sorted_vec_u32(chase_ticks_by_unit.into_iter()),
        invariant_violations,
    }
}

#[allow(clippy::too_many_arguments)]
fn process_events(
    events: &[SimEvent],
    unit_hp: &mut HashMap<u32, i32>,
    max_hp_by_unit: &HashMap<u32, i32>,
    total_damage_by_unit: &mut HashMap<u32, i32>,
    damage_taken_by_unit: &mut HashMap<u32, i32>,
    total_healing_by_unit: &mut HashMap<u32, i32>,
    movement_distance_x100_by_unit: &mut HashMap<u32, i32>,
    casts_started: &mut u32,
    casts_completed: &mut u32,
    casts_failed_out_of_range: &mut u32,
    heals_started: &mut u32,
    heals_completed: &mut u32,
    blocked_cooldown_intents: &mut u32,
    blocked_invalid_target_intents: &mut u32,
    reposition_for_range_events: &mut u32,
    executed_attack_intents: &mut u32,
    overkill_damage_total: &mut i32,
    invariant_violations: &mut u32,
    tick_to_first_death: &mut Option<u64>,
    cast_started_tick_by_unit: &mut HashMap<u32, u64>,
    total_cast_delay_ticks: &mut u64,
    resolved_casts: &mut u64,
) {
    for event in events {
        match *event {
            SimEvent::Moved { unit_id, from_x100, from_y100, to_x100, to_y100, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                let dx = (to_x100 - from_x100) as f32;
                let dy = (to_y100 - from_y100) as f32;
                let d = (dx * dx + dy * dy).sqrt().round() as i32;
                *movement_distance_x100_by_unit.entry(unit_id).or_insert(0) += d;
            }
            SimEvent::CastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                *casts_started += 1; *executed_attack_intents += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::HealCastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                *heals_started += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::AbilityCastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                *casts_started += 1; *executed_attack_intents += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::ControlCastStarted { tick, unit_id, .. } => {
                if unit_hp.get(&unit_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                *casts_started += 1; *executed_attack_intents += 1;
                cast_started_tick_by_unit.insert(unit_id, tick);
            }
            SimEvent::CastFailedOutOfRange { tick, unit_id, .. } => {
                *casts_failed_out_of_range += 1;
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&unit_id) {
                    *total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    *resolved_casts += 1;
                }
            }
            SimEvent::DamageApplied { tick, source_id, target_id, amount, target_hp_before, target_hp_after } => {
                if unit_hp.get(&source_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                if target_hp_after > target_hp_before { *invariant_violations += 1; }
                if target_hp_after < 0 { *invariant_violations += 1; }
                if let Some(max_hp) = max_hp_by_unit.get(&target_id) {
                    if target_hp_after > *max_hp { *invariant_violations += 1; }
                }
                *casts_completed += 1;
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&source_id) {
                    *total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    *resolved_casts += 1;
                }
                *total_damage_by_unit.entry(source_id).or_insert(0) += amount;
                *damage_taken_by_unit.entry(target_id).or_insert(0) += amount;
                let tracked_before = unit_hp.get(&target_id).copied().unwrap_or(target_hp_before);
                if amount > tracked_before { *overkill_damage_total += amount - tracked_before; }
                unit_hp.insert(target_id, target_hp_after);
            }
            SimEvent::UnitDied { tick, unit_id } => {
                if tick_to_first_death.is_none() { *tick_to_first_death = Some(tick); }
                if unit_hp.get(&unit_id).copied().unwrap_or(0) != 0 { *invariant_violations += 1; }
            }
            SimEvent::AttackBlockedCooldown { .. } | SimEvent::AbilityBlockedCooldown { .. }
            | SimEvent::HealBlockedCooldown { .. } | SimEvent::ControlBlockedCooldown { .. } => {
                *blocked_cooldown_intents += 1;
            }
            SimEvent::AttackBlockedInvalidTarget { .. } | SimEvent::AbilityBlockedInvalidTarget { .. }
            | SimEvent::HealBlockedInvalidTarget { .. } | SimEvent::ControlBlockedInvalidTarget { .. } => {
                *blocked_invalid_target_intents += 1;
            }
            SimEvent::AttackRepositioned { .. } => { *reposition_for_range_events += 1; }
            SimEvent::AbilityBlockedOutOfRange { .. } | SimEvent::HealBlockedOutOfRange { .. }
            | SimEvent::ControlBlockedOutOfRange { .. } => {
                *casts_failed_out_of_range += 1;
            }
            SimEvent::ControlApplied { tick, source_id, target_id, .. } => {
                if unit_hp.get(&source_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                if unit_hp.get(&target_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&source_id) {
                    *total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    *resolved_casts += 1;
                }
            }
            SimEvent::HealApplied { tick, source_id, target_id, amount, target_hp_before, target_hp_after } => {
                if unit_hp.get(&source_id).copied().unwrap_or(0) <= 0 { *invariant_violations += 1; }
                if target_hp_after < target_hp_before { *invariant_violations += 1; }
                if let Some(max_hp) = max_hp_by_unit.get(&target_id) {
                    if target_hp_after > *max_hp { *invariant_violations += 1; }
                }
                *heals_completed += 1;
                if let Some(start_tick) = cast_started_tick_by_unit.remove(&source_id) {
                    *total_cast_delay_ticks += tick.saturating_sub(start_tick);
                    *resolved_casts += 1;
                }
                *total_healing_by_unit.entry(source_id).or_insert(0) += amount;
                unit_hp.insert(target_id, target_hp_after);
            }
            _ => {}
        }
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
    let mut values = damage_by_unit.iter()
        .map(|(unit_id, damage)| {
            let dps = if elapsed <= f32::EPSILON { 0.0 } else { *damage as f32 / elapsed };
            (*unit_id, (dps * 100.0).round() / 100.0)
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|(unit_id, _)| *unit_id);
    values
}
