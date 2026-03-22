use contracts::*;
use crate::effects::{
    AbilityTarget, ConditionalEffect, DamageType, Effect, StatusKind, Trigger,
};

use super::types::*;
use super::events::SimEvent;
use super::math::distance;
use super::helpers::{clear_dead_unit_state, find_unit_idx, is_alive, next_rand_u32};
use super::triggers::{check_passive_triggers, fire_damage_triggers};
use super::conditions::evaluate_condition_tracked;
use super::apply_effect::apply_effect;
use super::targeting::resolve_targets;

pub fn apply_damage_to_unit(
    source_idx: usize,
    target_id: u32,
    base_amount: i32,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    apply_typed_damage(source_idx, target_id, base_amount, DamageType::Physical, tick, state, events);
}

#[requires(base_amount >= 0)]
#[requires(source_idx < state.units.len())]
pub fn apply_typed_damage(
    source_idx: usize,
    target_id: u32,
    base_amount: i32,
    damage_type: DamageType,
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

    // Banish check
    if state.units[target_idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Banish)) {
        return;
    }

    // Blind miss check
    let miss_chance = state.units[source_idx].status_effects.iter()
        .find_map(|s| if let StatusKind::Blind { miss_chance } = s.kind { Some(miss_chance) } else { None })
        .unwrap_or(0.0);
    let blind_miss = if miss_chance > 0.0 {
        let roll = (next_rand_u32(state) % 100) as f32 / 100.0;
        roll < miss_chance
    } else { false };
    if blind_miss {
        events.push(SimEvent::AttackMissed {
            tick, source_id: state.units[source_idx].id, target_id,
        });
        return;
    }

    let variance_percent = 90 + (next_rand_u32(state) % 21) as i32;
    let mut damage = ((base_amount * variance_percent) + 99) / 100;

    // Armor/MR reduction: reduction = resist / (100 + resist). True damage bypasses.
    match damage_type {
        DamageType::Physical => {
            let armor = state.units[target_idx].armor;
            if armor > 0.0 {
                let reduction = armor / (100.0 + armor);
                damage = (damage as f32 * (1.0 - reduction)) as i32;
            }
        }
        DamageType::Magic => {
            let mr = state.units[target_idx].magic_resist;
            if mr > 0.0 {
                let reduction = mr / (100.0 + mr);
                damage = (damage as f32 * (1.0 - reduction)) as i32;
            }
        }
        DamageType::True => {
            // True damage bypasses all reductions — skip cover, elevation, DamageModify too
        }
    }

    // Apply damage_output buff/debuff on source
    let damage_output_factor: f32 = state.units[source_idx].status_effects.iter()
        .filter_map(|s| {
            if let StatusKind::Buff { ref stat, factor } = s.kind {
                if stat == "damage_output" { return Some(factor); }
            }
            if let StatusKind::Debuff { ref stat, factor } = s.kind {
                if stat == "damage_output" { return Some(factor); }
            }
            None
        })
        .fold(1.0f32, |acc, f| acc * f);
    damage = (damage as f32 * damage_output_factor) as i32;

    // Cover damage reduction: units near obstacles take less damage.
    // Outnumbered teams get amplified cover (defensive stance bonus).
    // True damage bypasses cover/elevation/damage modifiers.
    let cover = state.units[target_idx].cover_bonus;
    if cover > 0.0 && damage_type != DamageType::True {
        let target_team = state.units[target_idx].team;
        let allies_alive = state.units.iter().filter(|u| u.team == target_team && u.hp > 0).count();
        let enemies_alive = state.units.iter().filter(|u| u.team != target_team && u.hp > 0).count();
        let outnumber_ratio = if allies_alive > 0 {
            (enemies_alive as f32 / allies_alive as f32).clamp(1.0, 4.0)
        } else {
            1.0
        };
        // Scale cover effectiveness: base cover * sqrt(outnumber ratio)
        // e.g. 4v1 → cover is 2x as effective; 2v1 → 1.4x; 1v1 → 1x
        let effective_cover = (cover * outnumber_ratio.sqrt()).min(0.7);
        damage = (damage as f32 * (1.0 - effective_cover)) as i32;
    }

    // Elevation advantage: attacker on high ground deals more, defender on high ground takes less.
    // Also scaled by outnumber ratio for the defender.
    let elev_diff = state.units[source_idx].elevation - state.units[target_idx].elevation;
    if elev_diff.abs() > 0.1 && damage_type != DamageType::True {
        let target_team = state.units[target_idx].team;
        let allies_alive = state.units.iter().filter(|u| u.team == target_team && u.hp > 0).count();
        let enemies_alive = state.units.iter().filter(|u| u.team != target_team && u.hp > 0).count();
        let defender_bonus = if allies_alive > 0 && elev_diff < 0.0 {
            // Defender on high ground and outnumbered: amplify elevation advantage
            let ratio = (enemies_alive as f32 / allies_alive as f32).clamp(1.0, 4.0);
            ratio.sqrt()
        } else {
            1.0
        };
        let elev_factor = 1.0 + elev_diff * 0.15 * defender_bonus;
        damage = (damage as f32 * elev_factor.clamp(0.5, 1.3)) as i32;
    }

    // DamageModify on target
    let damage_mod_factor: f32 = state.units[target_idx].status_effects.iter()
        .filter_map(|s| {
            if let StatusKind::DamageModify { factor } = s.kind { Some(factor) } else { None }
        })
        .fold(1.0f32, |acc, f| acc * f);
    damage = (damage as f32 * damage_mod_factor) as i32;

    // Redirect check
    let redirect_info: Option<(usize, u32)> = state.units[target_idx].status_effects.iter()
        .enumerate()
        .find_map(|(i, s)| {
            if let StatusKind::Redirect { protector_id, charges } = s.kind {
                if charges > 0 { Some((i, protector_id)) } else { None }
            } else { None }
        });
    if let Some((redirect_se_idx, protector_id)) = redirect_info {
        if let Some(p_idx) = find_unit_idx(state, protector_id) {
            if is_alive(&state.units[p_idx]) {
                if let StatusKind::Redirect { ref mut charges, .. } = state.units[target_idx].status_effects[redirect_se_idx].kind {
                    *charges -= 1;
                }
                let source_id = state.units[source_idx].id;
                let current_hp = state.units[p_idx].hp;
                let new_hp = (current_hp - damage).max(0);
                state.units[p_idx].hp = new_hp;
                events.push(SimEvent::DamageApplied {
                    tick, source_id, target_id: protector_id,
                    amount: damage, target_hp_before: current_hp, target_hp_after: new_hp,
                });
                if new_hp == 0 {
                    clear_dead_unit_state(&mut state.units[p_idx]);
                    events.push(SimEvent::UnitDied { tick, unit_id: protector_id });
                }
                return;
            }
        }
    }

    // Shield absorption
    let shield_before = state.units[target_idx].shield_hp;
    if state.units[target_idx].shield_hp > 0 {
        let absorbed = damage.min(state.units[target_idx].shield_hp).max(0);
        state.units[target_idx].shield_hp = (state.units[target_idx].shield_hp - absorbed).max(0);
        damage -= absorbed;
        events.push(SimEvent::ShieldAbsorbed {
            tick, unit_id: target_id, absorbed,
            remaining: state.units[target_idx].shield_hp,
        });
    }

    let current_hp = state.units[target_idx].hp;
    let new_hp = (current_hp - damage).max(0);
    state.units[target_idx].hp = new_hp;
    let actual_damage = current_hp - new_hp;
    state.units[source_idx].total_damage_done += actual_damage;

    events.push(SimEvent::DamageApplied {
        tick, source_id: state.units[source_idx].id, target_id,
        amount: damage, target_hp_before: current_hp, target_hp_after: new_hp,
    });

    // DeathMark accumulation
    for s in &mut state.units[target_idx].status_effects {
        if let StatusKind::DeathMark { ref mut accumulated_damage, .. } = s.kind {
            *accumulated_damage += actual_damage;
        }
    }

    // Reflect
    let reflect_damage: i32 = state.units[target_idx].status_effects.iter()
        .filter_map(|s| {
            if let StatusKind::Reflect { percent } = s.kind {
                Some((actual_damage as f32 * percent / 100.0) as i32)
            } else { None }
        })
        .sum();
    if reflect_damage > 0 && source_idx != target_idx {
        let src_hp_before = state.units[source_idx].hp;
        state.units[source_idx].hp = (src_hp_before - reflect_damage).max(0);
        events.push(SimEvent::ReflectDamage {
            tick, source_id: target_id, target_id: state.units[source_idx].id,
            amount: reflect_damage,
        });
        if state.units[source_idx].hp == 0 {
            clear_dead_unit_state(&mut state.units[source_idx]);
            events.push(SimEvent::UnitDied { tick, unit_id: state.units[source_idx].id });
        }
    }

    // Lifesteal
    let lifesteal_heal: i32 = state.units[source_idx].status_effects.iter()
        .filter_map(|s| {
            if let StatusKind::Lifesteal { percent } = s.kind {
                Some((actual_damage as f32 * percent / 100.0) as i32)
            } else { None }
        })
        .sum();
    if lifesteal_heal > 0 {
        let src_hp = state.units[source_idx].hp;
        let src_max = state.units[source_idx].max_hp;
        let healed_to = (src_hp + lifesteal_heal).min(src_max);
        state.units[source_idx].hp = healed_to;
        events.push(SimEvent::LifestealHeal {
            tick, unit_id: state.units[source_idx].id, amount: healed_to - src_hp,
        });
    }

    // Break stealth on damage received
    state.units[target_idx].status_effects.retain(|s| {
        if let StatusKind::Stealth { break_on_damage, .. } = s.kind {
            !break_on_damage
        } else { true }
    });

    // Link
    if state.passive_trigger_depth < 3 {
        let link_entries: Vec<(u32, f32)> = state.units[target_idx].status_effects.iter()
            .filter_map(|s| {
                if let StatusKind::Link { partner_id, share_percent } = s.kind {
                    Some((partner_id, share_percent))
                } else { None }
            })
            .collect();
        for (partner_id, share_pct) in link_entries {
            if let Some(p_idx) = find_unit_idx(state, partner_id) {
                if is_alive(&state.units[p_idx]) {
                    let shared = (actual_damage as f32 * share_pct / 100.0) as i32;
                    if shared > 0 {
                        state.passive_trigger_depth += 1;
                        let p_hp = state.units[p_idx].hp;
                        state.units[p_idx].hp = (p_hp - shared).max(0);
                        events.push(SimEvent::DamageApplied {
                            tick, source_id: state.units[source_idx].id,
                            target_id: partner_id, amount: shared,
                            target_hp_before: p_hp, target_hp_after: state.units[p_idx].hp,
                        });
                        if state.units[p_idx].hp == 0 {
                            clear_dead_unit_state(&mut state.units[p_idx]);
                            events.push(SimEvent::UnitDied { tick, unit_id: partner_id });
                        }
                        state.passive_trigger_depth -= 1;
                    }
                }
            }
        }
    }

    if new_hp == 0 {
        clear_dead_unit_state(&mut state.units[target_idx]);
        events.push(SimEvent::UnitDied { tick, unit_id: target_id });
    }

    let shield_broke = shield_before > 0 && state.units[target_idx].shield_hp == 0;
    fire_damage_triggers(source_idx, target_idx, target_id, new_hp, shield_broke, tick, state, events);

    // Re-clear dead unit state after triggers: OnDeath passives may set casting/channeling.
    if new_hp == 0 {
        clear_dead_unit_state(&mut state.units[target_idx]);
    }
}

#[requires(base_amount >= 0)]
pub fn apply_heal_to_unit(
    source_idx: usize,
    target_id: u32,
    base_amount: i32,
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

    let variance_percent = 95 + (next_rand_u32(state) % 11) as i32;
    let mut heal_amount = ((base_amount * variance_percent) + 99) / 100;

    let heal_power_factor: f32 = state.units[source_idx].status_effects.iter()
        .filter_map(|s| {
            match &s.kind {
                StatusKind::Buff { stat, factor } if stat == "heal_power" => Some(1.0 + factor),
                StatusKind::Debuff { stat, factor } if stat == "heal_power" => Some(1.0 - factor),
                _ => None,
            }
        })
        .fold(1.0f32, |acc, f| acc * f);
    heal_amount = (heal_amount as f32 * heal_power_factor) as i32;

    let current_hp = state.units[target_idx].hp;
    let max_hp = state.units[target_idx].max_hp;
    let new_hp = (current_hp + heal_amount).min(max_hp);
    let actual = new_hp - current_hp;
    let overflow = heal_amount - actual;
    state.units[target_idx].hp = new_hp;

    if overflow > 0 {
        let conversion: f32 = state.units[target_idx].status_effects.iter()
            .filter_map(|s| {
                if let StatusKind::OverhealShield { conversion_percent } = s.kind {
                    Some(conversion_percent)
                } else { None }
            })
            .next()
            .unwrap_or(0.0);
        if conversion > 0.0 {
            let shield_gain = (overflow as f32 * conversion / 100.0) as i32;
            state.units[target_idx].shield_hp += shield_gain;
            events.push(SimEvent::ShieldApplied {
                tick, unit_id: target_id, amount: shield_gain,
            });
        }
    }

    if actual > 0 {
        state.units[source_idx].total_healing_done += actual;
    }

    events.push(SimEvent::HealApplied {
        tick, source_id: state.units[source_idx].id, target_id,
        amount: actual, target_hp_before: current_hp, target_hp_after: new_hp,
    });

    if actual > 0 {
        let pct = (new_hp as f32 / max_hp as f32) * 100.0;
        check_passive_triggers(Trigger::OnHpAbove { percent: pct }, target_idx, state.units[source_idx].id, tick, state, events);
        check_passive_triggers(Trigger::OnHealReceived, target_idx, state.units[source_idx].id, tick, state, events);
    }
}

pub fn resolve_chain_delivery(
    caster_idx: usize,
    target: AbilityTarget,
    bounces: u32,
    bounce_range: f32,
    falloff: f32,
    on_hit: &[ConditionalEffect],
    primary_effects: &[ConditionalEffect],
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    let caster_id = state.units[caster_idx].id;
    let caster_team = state.units[caster_idx].team;

    let first_target = match target {
        AbilityTarget::Unit(tid) => tid,
        _ => return,
    };

    let mut already_hit = vec![first_target];

    for ce in primary_effects {
        if evaluate_condition_tracked(&ce.condition, caster_idx, target, state, tick, events) {
            let targets = resolve_targets(ce.area.as_ref(), caster_idx, target, caster_team, &ce.effect, state);
            for &tid in &targets {
                apply_effect(&ce.effect, caster_idx, tid, target, tick, &ce.tags, ce.stacking, state, events);
            }
        }
    }
    for ce in on_hit {
        if evaluate_condition_tracked(&ce.condition, caster_idx, AbilityTarget::Unit(first_target), state, tick, events) {
            apply_effect(&ce.effect, caster_idx, first_target, target, tick, &ce.tags, ce.stacking, state, events);
        }
    }

    let mut current_pos = find_unit_idx(state, first_target)
        .map(|i| state.units[i].position)
        .unwrap_or_default();

    let targets_allies = on_hit.iter().any(|ce| matches!(ce.effect,
        Effect::Heal { .. } | Effect::Shield { .. } | Effect::Buff { .. }
    ));

    for bounce_num in 1..=bounces {
        let scale = (1.0 - falloff).powi(bounce_num as i32);

        let next_target = state.units.iter()
            .filter(|u| u.hp > 0 && !already_hit.contains(&u.id) && u.id != caster_id)
            .filter(|u| {
                if targets_allies { u.team == caster_team } else { u.team != caster_team }
            })
            .filter(|u| distance(current_pos, u.position) <= bounce_range)
            .min_by(|a, b| {
                distance(current_pos, a.position)
                    .partial_cmp(&distance(current_pos, b.position))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|u| u.id);

        let Some(next_id) = next_target else { break; };
        already_hit.push(next_id);

        events.push(SimEvent::ChainBounce {
            tick, source_id: caster_id, target_id: next_id, bounce_num,
        });

        for ce in on_hit {
            if evaluate_condition_tracked(&ce.condition, caster_idx, AbilityTarget::Unit(next_id), state, tick, events) {
                let scaled_effect = scale_effect(&ce.effect, scale);
                apply_effect(&scaled_effect, caster_idx, next_id, AbilityTarget::Unit(next_id), tick, &ce.tags, ce.stacking, state, events);
            }
        }

        if let Some(next_idx) = find_unit_idx(state, next_id) {
            current_pos = state.units[next_idx].position;
        }
    }
}

pub fn scale_effect(effect: &Effect, scale: f32) -> Effect {
    match effect {
        Effect::Damage { amount, amount_per_tick, duration_ms, tick_interval_ms, scaling_stat, scaling_percent, damage_type, ref bonus } => {
            Effect::Damage {
                amount: ((*amount as f32) * scale) as i32,
                amount_per_tick: ((*amount_per_tick as f32) * scale) as i32,
                duration_ms: ((*duration_ms as f32) * scale) as u32,
                tick_interval_ms: *tick_interval_ms,
                scaling_stat: scaling_stat.clone(),
                scaling_percent: *scaling_percent,
                damage_type: *damage_type,
                bonus: bonus.clone(),
            }
        }
        Effect::Heal { amount, amount_per_tick, duration_ms, tick_interval_ms, scaling_stat, scaling_percent, ref bonus } => {
            Effect::Heal {
                amount: ((*amount as f32) * scale) as i32,
                amount_per_tick: ((*amount_per_tick as f32) * scale) as i32,
                duration_ms: ((*duration_ms as f32) * scale) as u32,
                tick_interval_ms: *tick_interval_ms,
                scaling_stat: scaling_stat.clone(),
                scaling_percent: *scaling_percent,
                bonus: bonus.clone(),
            }
        }
        Effect::Stun { duration_ms } => Effect::Stun {
            duration_ms: ((*duration_ms as f32) * scale) as u32,
        },
        Effect::Slow { factor, duration_ms } => Effect::Slow {
            factor: *factor * scale,
            duration_ms: ((*duration_ms as f32) * scale) as u32,
        },
        other => other.clone(),
    }
}
