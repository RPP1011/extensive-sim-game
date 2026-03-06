use crate::ai::core::{
    distance, move_away, move_towards, position_at_range, IntentAction, SimState,
};
use crate::ai::effects::{AbilityTarget, AbilityTargeting, Area, Effect};

use crate::ai::squad::personality::Personality;
use crate::ai::squad::state::{personality_movement_profile, FormationMode, TickContext};

use super::sorted_tags;

pub(in crate::ai::squad) fn evaluate_hero_ability(
    state: &SimState,
    unit_id: u32,
    target_id: u32,
    mode: FormationMode,
    ctx: &TickContext,
) -> Option<IntentAction> {
    let unit = ctx.unit(state, unit_id)?;
    let target = ctx.unit(state, target_id)?;
    let dist = distance(unit.position, target.position);

    // Pre-compute context for scoring
    let hp_pct = unit.hp as f32 / unit.max_hp.max(1) as f32;
    let target_hp_pct = target.hp as f32 / target.max_hp.max(1) as f32;
    let _nearby_enemies = state.units.iter()
        .filter(|u| u.hp > 0 && u.team != unit.team && distance(unit.position, u.position) < 4.0)
        .count() as f32;
    let nearby_allies = state.units.iter()
        .filter(|u| u.hp > 0 && u.team == unit.team && u.id != unit.id && distance(unit.position, u.position) < 5.0)
        .count() as f32;
    let target_is_controlled = target.control_remaining_ms > 0;

    // --- Threat model: compute target's DPS for threat-reduction scoring ---
    let target_dps = if target.attack_cooldown_ms > 0 {
        target.attack_damage as f32 / (target.attack_cooldown_ms as f32 / 1000.0)
    } else {
        target.attack_damage as f32
    };

    // Pre-compute: what conditional bonus damage could we deal if the target
    // were stunned/CC'd? This is the "setup value" of applying CC first.
    let conditional_bonus_damage: f32 = unit.abilities.iter()
        .filter(|s| s.cooldown_remaining_ms == 0 || s.cooldown_remaining_ms <= 3000)
        .flat_map(|s| s.def.effects.iter())
        .filter(|ce| ce.condition.is_some())
        .filter_map(|ce| match &ce.effect {
            Effect::Damage { amount, .. } => Some(*amount as f32),
            _ => None,
        })
        .sum();

    // Does this unit have a CC ability ready right now?
    // Uses the ai_hint tag rather than hardcoding effect types.
    let has_cc_ready = unit.abilities.iter().any(|s| {
        s.cooldown_remaining_ms == 0
            && matches!(s.def.ai_hint.as_str(), "crowd_control" | "control")
    });

    let mut best: Option<(usize, f32)> = None;

    for (i, slot) in unit.abilities.iter().enumerate() {
        if slot.cooldown_remaining_ms > 0 {
            continue;
        }

        let target_ok = match slot.def.targeting {
            AbilityTargeting::TargetEnemy => target.team != unit.team,
            AbilityTargeting::TargetAlly => target.team == unit.team,
            AbilityTargeting::SelfCast | AbilityTargeting::SelfAoe => true,
            AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector | AbilityTargeting::Global => true,
        };
        if !target_ok {
            continue;
        }

        if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
            continue;
        }

        // Range check: skip if target is beyond ability range.
        // For direction/ground_target projectiles, range = max travel distance,
        // so we need to check range for those too.
        let skip_range = matches!(
            slot.def.targeting,
            AbilityTargeting::SelfCast | AbilityTargeting::SelfAoe | AbilityTargeting::Global
        );
        if !skip_range && slot.def.range > 0.0 && dist > slot.def.range {
            continue;
        }

        // --- Analyze ability effects ---
        let has_aoe = slot.def.effects.iter().any(|ce| {
            matches!(ce.area, Some(Area::Circle { .. }) | Some(Area::Cone { .. }) | Some(Area::Spread { .. }))
        });
        let aoe_radius = slot.def.effects.iter().filter_map(|ce| match &ce.area {
            Some(Area::Circle { radius }) => Some(*radius),
            Some(Area::Cone { radius, .. }) => Some(*radius),
            Some(Area::Spread { radius, .. }) => Some(*radius),
            _ => None,
        }).fold(0.0f32, f32::max);
        let has_shield = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Shield { .. }));
        let has_buff = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Buff { .. }));
        let has_debuff = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Debuff { .. } | Effect::DamageModify { .. }));
        let has_dash = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Dash { .. }));

        // Estimate AoE targets hit (centered on target for ground-target, on caster for self_aoe)
        let aoe_center = match slot.def.targeting {
            AbilityTargeting::GroundTarget | AbilityTargeting::Direction => target.position,
            _ => unit.position,
        };
        let aoe_targets = if has_aoe && aoe_radius > 0.0 {
            state.units.iter()
                .filter(|u| u.hp > 0 && u.team != unit.team && distance(aoe_center, u.position) <= aoe_radius)
                .count().max(1) as f32
        } else {
            1.0
        };

        // --- Threat reduction scoring ---
        // Compute how much "threat" this ability removes from the battlefield.
        // Threat = enemy damage potential. Removing it = CC (temporary) or kill (permanent).

        // Sum up direct damage this ability would deal (unconditional effects only).
        // Includes both top-level effects and delivery on_hit effects.
        let damage_from_effects = |effects: &[crate::ai::effects::ConditionalEffect]| -> f32 {
            effects.iter()
                .filter(|ce| ce.condition.is_none())
                .filter_map(|ce| match &ce.effect {
                    Effect::Damage { amount, amount_per_tick, duration_ms, tick_interval_ms, .. } => {
                        let instant = *amount as f32;
                        let dot = if *tick_interval_ms > 0 && *duration_ms > 0 {
                            *amount_per_tick as f32 * (*duration_ms as f32 / *tick_interval_ms as f32)
                        } else { 0.0 };
                        Some(instant + dot)
                    }
                    Effect::ApplyStacks { count, max_stacks, .. } => {
                        Some(*count as f32 / *max_stacks as f32 * 15.0)
                    }
                    _ => None,
                })
                .sum::<f32>()
        };

        let mut unconditional_damage = damage_from_effects(&slot.def.effects);

        // Include damage from delivery payloads (projectile on_hit, tether on_complete, etc.)
        if let Some(ref delivery) = slot.def.delivery {
            use crate::ai::effects::Delivery;
            match delivery {
                Delivery::Projectile { on_hit, on_arrival, .. } => {
                    unconditional_damage += damage_from_effects(on_hit);
                    unconditional_damage += damage_from_effects(on_arrival);
                }
                Delivery::Tether { on_complete, .. } => {
                    unconditional_damage += damage_from_effects(on_complete);
                }
                _ => {}
            }
        }

        // Sum conditional damage (only counts if condition is currently met)
        let met_conditional_damage: f32 = slot.def.effects.iter()
            .filter(|ce| ce.condition.is_some())
            .filter_map(|ce| {
                let cond_met = match &ce.condition {
                    Some(crate::ai::effects::Condition::TargetIsStunned) => target_is_controlled,
                    Some(crate::ai::effects::Condition::TargetIsSlowed) => {
                        target.status_effects.iter().any(|s| matches!(s.kind, crate::ai::effects::StatusKind::Slow { .. }))
                    }
                    Some(crate::ai::effects::Condition::TargetHpBelow { percent }) => {
                        target_hp_pct * 100.0 < *percent
                    }
                    // For other conditions, assume not met (conservative)
                    _ => false,
                };
                if !cond_met { return None; }
                match &ce.effect {
                    Effect::Damage { amount, .. } => Some(*amount as f32),
                    _ => None,
                }
            })
            .sum();

        let total_damage = (unconditional_damage + met_conditional_damage) * aoe_targets;

        // CC threat reduction: damage prevented by stunning/slowing the target
        let cc_threat_reduction: f32 = slot.def.effects.iter()
            .filter(|ce| ce.condition.is_none())
            .filter_map(|ce| match &ce.effect {
                Effect::Stun { duration_ms } => {
                    if target_is_controlled { return None; } // Don't stack stuns
                    let prevented = target_dps * (*duration_ms as f32 / 1000.0);
                    // Setup value: this CC enables our conditional abilities
                    let setup = conditional_bonus_damage;
                    Some(prevented + setup)
                }
                Effect::Slow { factor, duration_ms } => {
                    // Slow reduces threat partially (fewer attacks land due to kiting)
                    let prevented = target_dps * (*duration_ms as f32 / 1000.0) * factor * 0.5;
                    Some(prevented)
                }
                _ => None,
            })
            .sum::<f32>() * aoe_targets;

        // Kill check: if this ability would kill the target, that's permanent threat removal
        let would_kill = total_damage >= target.hp as f32;
        let kill_bonus = if would_kill {
            // Permanent threat removal — worth much more than temporary CC
            target_dps * 10.0
        } else {
            0.0
        };

        let hint = slot.def.ai_hint.as_str();
        let mut score = match hint {
            "crowd_control" | "control" => {
                if target_is_controlled || mode == FormationMode::Retreat {
                    0.0
                } else {
                    // Threat reduction from CC + any damage the ability also does
                    cc_threat_reduction + total_damage
                }
            }
            "damage" | "opener" => {
                if mode == FormationMode::Retreat && !has_dash {
                    1.0
                } else {
                    let opener = if hint == "opener" { 2.0 } else { 0.0 };
                    let mut s = total_damage + kill_bonus + opener + cc_threat_reduction;
                    // Deferral: if this ability has unmet conditional damage AND we
                    // have a CC ability ready to enable it, defer so the CC fires first.
                    // This makes the rogue use Garrote before Backstab, etc.
                    let has_unmet_conditional = met_conditional_damage == 0.0
                        && slot.def.effects.iter().any(|ce| ce.condition.is_some());
                    if has_unmet_conditional && has_cc_ready && !target_is_controlled {
                        s -= conditional_bonus_damage;
                    }
                    s
                }
            }
            "defense" => {
                let base = if hp_pct < 0.35 { 10.0 }
                    else if hp_pct < 0.6 { 7.0 }
                    else { 1.5 };
                if has_aoe && (has_shield || has_buff) {
                    base * (nearby_allies + 1.0).min(4.0)
                } else {
                    base
                }
            }
            "heal" => {
                if hp_pct < 0.4 { 8.0 }
                else if hp_pct < 0.6 { 4.0 }
                else { 0.0 }
            }
            "utility" => {
                let mut base = 3.0;
                if has_buff { base += 4.0 + nearby_allies; }
                if has_debuff { base += 5.0 + nearby_allies * 1.5; }
                if has_dash && hp_pct < 0.4 { base += 5.0; }
                // Utility abilities with CC components use threat model too
                base + cc_threat_reduction + total_damage
            }
            _ => 3.0 + total_damage + cc_threat_reduction,
        };

        // Zone-reaction bonus: if this ability has a zone_tag and there's an existing
        // compatible zone from this caster, boost score for the combo potential.
        if let Some(ref tag) = slot.def.zone_tag {
            let has_compatible_zone = state.zones.iter().any(|z| {
                z.source_id == unit.id
                    && z.zone_tag.as_ref().map_or(false, |zt| {
                        zt != tag && matches!(
                            sorted_tags(tag, zt),
                            ("fire", "frost") | ("fire", "lightning") | ("frost", "lightning")
                        )
                    })
            });
            if has_compatible_zone {
                score += 25.0; // Strong incentive to trigger the combo
            }
        }

        // Penalize long cooldown abilities unless the score is high enough to justify
        if slot.def.cooldown_ms > 12000 && score < 8.0 {
            score *= 0.7;
        }

        if score > 0.0 {
            if best.map_or(true, |(_, bs)| score > bs) {
                best = Some((i, score));
            }
        }
    }

    let (ability_index, _score) = best?;
    let ability_target = match unit.abilities[ability_index].def.targeting {
        AbilityTargeting::TargetEnemy | AbilityTargeting::TargetAlly => {
            AbilityTarget::Unit(target_id)
        }
        AbilityTargeting::SelfCast | AbilityTargeting::SelfAoe | AbilityTargeting::Global => AbilityTarget::None,
        AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector => {
            let chosen = &unit.abilities[ability_index];
            // If this ability has a zone_tag, prefer placing it on a compatible existing zone
            let combo_pos = chosen.def.zone_tag.as_ref().and_then(|tag| {
                state.zones.iter()
                    .filter(|z| z.source_id == unit.id)
                    .filter(|z| z.zone_tag.as_ref().map_or(false, |zt| {
                        zt != tag && matches!(
                            sorted_tags(tag, zt),
                            ("fire", "frost") | ("fire", "lightning") | ("frost", "lightning")
                        )
                    }))
                    .filter(|z| distance(unit.position, z.position) <= chosen.def.range)
                    .map(|z| z.position)
                    .next()
            });
            AbilityTarget::Position(combo_pos.unwrap_or(target.position))
        }
    };

    Some(IntentAction::UseAbility {
        ability_index,
        target: ability_target,
    })
}

pub(in crate::ai::squad) fn choose_action(
    state: &SimState,
    unit_id: u32,
    target_id: u32,
    personality: &Personality,
    mode: FormationMode,
    dt_ms: u32,
    ctx: &TickContext,
) -> IntentAction {
    let Some(unit) = ctx.unit(state, unit_id) else {
        return IntentAction::Hold;
    };
    let Some(target) = ctx.unit(state, target_id) else {
        return IntentAction::Hold;
    };

    let dist = distance(unit.position, target.position);

    if !unit.abilities.is_empty() && mode != FormationMode::Retreat {
        if let Some(action) = evaluate_hero_ability(state, unit_id, target_id, mode, ctx) {
            return action;
        }
    }

    if unit.control_duration_ms > 0
        && unit.control_cooldown_remaining_ms == 0
        && dist <= unit.control_range
        && target.control_remaining_ms == 0
        && mode != FormationMode::Retreat
    {
        return IntentAction::CastControl { target_id };
    }
    if unit.ability_cooldown_remaining_ms == 0
        && unit.ability_damage > 0
        && dist <= unit.ability_range
        && target.hp > unit.attack_damage
        && mode != FormationMode::Retreat
    {
        return IntentAction::CastAbility { target_id };
    }

    if dist <= unit.attack_range {
        return IntentAction::Attack { target_id };
    }

    let profile = personality_movement_profile(personality, unit);
    let base_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
    let range_center = (profile.preferred_range_min + profile.preferred_range_max) * 0.5;

    let (desired_pos, max_step) = match mode {
        FormationMode::Advance => {
            (target.position, base_step)
        }
        FormationMode::Hold => {
            let r = range_center.min(unit.attack_range);
            (position_at_range(unit.position, target.position, r), base_step)
        }
        FormationMode::Retreat => {
            (
                move_away(unit.position, target.position, range_center * 0.7),
                base_step * 0.6,
            )
        }
    };

    let next_pos = move_towards(unit.position, desired_pos, max_step);
    IntentAction::MoveTo { position: next_pos }
}
