//! Effect lowering (AST EffectNode → ConditionalEffect / Effect).

use super::ast::*;
use super::lower::{lower_condition, lower_area, lower_tags};
use crate::effects::effect_enum::Effect;
use crate::effects::types::*;

pub(super) fn lower_effect(node: &EffectNode) -> Result<ConditionalEffect, String> {
    let effect = lower_effect_type(node)?;
    let condition = node.condition.as_ref().map(lower_condition).transpose()?;
    let area = node.area.as_ref().map(lower_area).transpose()?;
    let tags = lower_tags(&node.tags);
    let stacking = node.stacking.as_ref().map(|s| match s.as_str() {
        "refresh" => Stacking::Refresh,
        "extend" => Stacking::Extend,
        "strongest" => Stacking::Strongest,
        "stack" => Stacking::Stack,
        _ => Stacking::Refresh,
    }).unwrap_or_default();
    let chance = node.chance.map(|c| c as f32).unwrap_or(0.0);

    let mut else_effects = Vec::new();
    for eff in &node.else_effects {
        else_effects.push(lower_effect(eff)?);
    }

    Ok(ConditionalEffect {
        effect,
        condition,
        area,
        tags,
        stacking,
        chance,
        else_effects,
    })
}

fn lower_effect_type(node: &EffectNode) -> Result<Effect, String> {
    match node.effect_type.as_str() {
        "damage" => {
            let bonus = lower_scaling(&node.scaling);
            // Check for per-tick (DoT) syntax: damage X/tick for Ys
            if let Some(Arg::PerTick { amount, interval_ms }) = node.args.first() {
                let dur = node.duration.unwrap_or(0);
                Ok(Effect::Damage {
                    amount: 0,
                    amount_per_tick: *amount,
                    duration_ms: dur,
                    tick_interval_ms: *interval_ms,
                    scaling_stat: None,
                    scaling_percent: 0.0,
                    damage_type: DamageType::Physical,
                    bonus,
                })
            } else {
                let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
                Ok(Effect::Damage {
                    amount,
                    amount_per_tick: 0,
                    duration_ms: 0,
                    tick_interval_ms: 0,
                    scaling_stat: None,
                    scaling_percent: 0.0,
                    damage_type: DamageType::Physical,
                    bonus,
                })
            }
        }
        "heal" => {
            let bonus = lower_scaling(&node.scaling);
            // Check for per-tick (HoT) syntax: heal X/tick for Ys
            if let Some(Arg::PerTick { amount, interval_ms }) = node.args.first() {
                let dur = node.duration.unwrap_or(0);
                Ok(Effect::Heal {
                    amount: 0,
                    amount_per_tick: *amount,
                    duration_ms: dur,
                    tick_interval_ms: *interval_ms,
                    scaling_stat: None,
                    scaling_percent: 0.0,
                    bonus,
                })
            } else {
                let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
                Ok(Effect::Heal {
                    amount,
                    amount_per_tick: 0,
                    duration_ms: 0,
                    tick_interval_ms: 0,
                    scaling_stat: None,
                    scaling_percent: 0.0,
                    bonus,
                })
            }
        }
        "shield" => {
            let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Shield { amount, duration_ms: dur })
        }
        "stun" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Stun { duration_ms: dur })
        }
        "slow" => {
            let factor = node.args.first().and_then(|a| a.as_f64()).unwrap_or(0.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Slow { factor, duration_ms: dur })
        }
        "root" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Root { duration_ms: dur })
        }
        "silence" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Silence { duration_ms: dur })
        }
        "fear" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Fear { duration_ms: dur })
        }
        "taunt" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Taunt { duration_ms: dur })
        }
        "charm" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Charm { duration_ms: dur })
        }
        "polymorph" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Polymorph { duration_ms: dur })
        }
        "banish" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Banish { duration_ms: dur })
        }
        "confuse" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Confuse { duration_ms: dur })
        }
        "suppress" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Suppress { duration_ms: dur })
        }
        "grounded" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Grounded { duration_ms: dur })
        }
        "dash" => {
            let mut distance = 2.0f32;
            let mut to_target = false;
            let mut to_position = false;
            let is_blink = false;

            for arg in &node.args {
                match arg {
                    Arg::Number(n) => distance = *n as f32,
                    Arg::Ident(s) if s == "to_target" => to_target = true,
                    Arg::Ident(s) if s == "to_position" => to_position = true,
                    _ => {}
                }
            }

            Ok(Effect::Dash { to_target, distance, to_position, is_blink })
        }
        "blink" => {
            let distance = node.args.first().and_then(|a| a.as_f64()).unwrap_or(2.0) as f32;
            Ok(Effect::Dash { to_target: false, distance, to_position: false, is_blink: true })
        }
        "knockback" => {
            let dist = node.args.first().and_then(|a| a.as_f64()).unwrap_or(2.0) as f32;
            Ok(Effect::Knockback { distance: dist })
        }
        "pull" => {
            let dist = node.args.first().and_then(|a| a.as_f64()).unwrap_or(2.0) as f32;
            Ok(Effect::Pull { distance: dist })
        }
        "swap" => Ok(Effect::Swap),
        "buff" => {
            let stat = node.args.first().and_then(|a| a.as_str()).unwrap_or("").to_string();
            let factor = node.args.get(1).and_then(|a| a.as_f64()).unwrap_or(0.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(2).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Buff { stat, factor, duration_ms: dur })
        }
        "debuff" => {
            let stat = node.args.first().and_then(|a| a.as_str()).unwrap_or("").to_string();
            let factor = node.args.get(1).and_then(|a| a.as_f64()).unwrap_or(0.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(2).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Debuff { stat, factor, duration_ms: dur })
        }
        "damage_modify" => {
            let factor = node.args.first().and_then(|a| a.as_f64()).unwrap_or(1.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::DamageModify { factor, duration_ms: dur })
        }
        "reflect" => {
            let pct = node.args.first().and_then(|a| a.as_f64()).unwrap_or(0.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Reflect { percent: pct, duration_ms: dur })
        }
        "lifesteal" => {
            let pct = node.args.first().and_then(|a| a.as_f64()).unwrap_or(0.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Lifesteal { percent: pct, duration_ms: dur })
        }
        "blind" => {
            let miss = node.args.first().and_then(|a| a.as_f64()).unwrap_or(0.5) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Blind { miss_chance: miss, duration_ms: dur })
        }
        "summon" => {
            let template = node.args.first().and_then(|a| a.as_str()).unwrap_or("").to_string();
            let is_clone = template == "clone";
            let mut count = 1u32;
            for arg in &node.args[1..] {
                if let Arg::Ident(s) = arg {
                    if let Some(n_str) = s.strip_prefix('x') {
                        if let Ok(n) = n_str.parse::<u32>() {
                            count = n;
                        }
                    }
                }
            }
            Ok(Effect::Summon {
                template,
                count,
                hp_percent: 100.0,
                clone: is_clone,
                clone_damage_percent: 75.0,
                directed: false,
            })
        }
        "command_summons" => {
            let mut speed = 8.0f32;
            for arg in &node.args {
                if let Arg::Number(n) = arg {
                    speed = *n as f32;
                }
            }
            Ok(Effect::CommandSummons { speed })
        }
        "stealth" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            let break_on_damage = node.args.iter().any(|a| matches!(a, Arg::Ident(s) if s == "break_on_damage"));
            Ok(Effect::Stealth { duration_ms: dur, break_on_damage, break_on_ability: false })
        }
        "leash" => {
            let max_range = node.args.first().and_then(|a| a.as_f64()).unwrap_or(4.0) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Leash { max_range, duration_ms: dur })
        }
        "link" => {
            let share = node.args.first().and_then(|a| a.as_f64()).unwrap_or(0.5) as f32;
            let dur = node.duration.or_else(|| node.args.get(1).and_then(|a| a.as_duration_ms())).unwrap_or(0);
            Ok(Effect::Link { duration_ms: dur, share_percent: share })
        }
        "redirect" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            let mut charges = 3u32;
            // Look for "charges N" in args
            let mut i = 0;
            while i < node.args.len() {
                if let Arg::Ident(s) = &node.args[i] {
                    if s == "charges" {
                        if let Some(Arg::Number(n)) = node.args.get(i + 1) {
                            charges = *n as u32;
                        }
                    }
                }
                i += 1;
            }
            Ok(Effect::Redirect { duration_ms: dur, charges })
        }
        "rewind" => {
            let lookback = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(3000);
            Ok(Effect::Rewind { lookback_ms: lookback })
        }
        "cooldown_modify" => {
            let amount_str = node.args.first();
            let mut amount_ms = 0i32;
            if let Some(arg) = amount_str {
                match arg {
                    Arg::Duration(ms) => amount_ms = -(*ms as i32),
                    Arg::Number(n) => amount_ms = *n as i32,
                    _ => {}
                }
            }
            let ability_name = node.args.get(1).and_then(|a| a.as_str()).map(|s| s.to_string());
            Ok(Effect::CooldownModify { amount_ms, ability_name })
        }
        "apply_stacks" => {
            let name = node.args.first().and_then(|a| a.as_str()).unwrap_or("").to_string();
            let count = node.args.get(1).and_then(|a| a.as_u32()).unwrap_or(1);
            // Look for "max N"
            let mut max_stacks = 4u32;
            let mut i = 2;
            while i < node.args.len() {
                if let Arg::Ident(s) = &node.args[i] {
                    if s == "max" {
                        if let Some(n) = node.args.get(i + 1).and_then(|a| a.as_u32()) {
                            max_stacks = n;
                        }
                    }
                }
                i += 1;
            }
            let dur = node.duration.unwrap_or(0);
            Ok(Effect::ApplyStacks { name, count, max_stacks, duration_ms: dur })
        }
        "execute" => {
            let pct = node.args.first().and_then(|a| a.as_f64()).unwrap_or(15.0) as f32;
            Ok(Effect::Execute { hp_threshold_percent: pct })
        }
        "self_damage" => {
            let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
            Ok(Effect::SelfDamage { amount })
        }
        "dispel" => {
            let target_tags = node.tags.iter().map(|(k, _)| k.clone()).collect();
            Ok(Effect::Dispel { target_tags })
        }
        "immunity" => {
            let immune_to = node.tags.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
            // Also check for [tag, tag] in args
            let immune_to = if immune_to.is_empty() {
                node.args.iter().filter_map(|a| a.as_str().map(|s| s.to_string())).collect()
            } else {
                immune_to
            };
            let dur = node.duration.unwrap_or(0);
            Ok(Effect::Immunity { immune_to, duration_ms: dur })
        }
        "death_mark" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            let dmg_pct = node.args.get(1).and_then(|a| a.as_f64()).unwrap_or(50.0) as f32;
            Ok(Effect::DeathMark { duration_ms: dur, damage_percent: dmg_pct })
        }
        "resurrect" => {
            let hp_pct = node.args.first().and_then(|a| a.as_f64()).unwrap_or(50.0) as f32;
            Ok(Effect::Resurrect { hp_percent: hp_pct })
        }
        "overheal_shield" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::OverhealShield { duration_ms: dur, conversion_percent: 100.0 })
        }
        "absorb_to_heal" => {
            let shield = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
            let dur = node.duration.unwrap_or(0);
            let mut heal_pct = 50.0f32;
            // Look for "heal 0.5" in args
            let mut i = 1;
            while i < node.args.len() {
                if let Arg::Ident(s) = &node.args[i] {
                    if s == "heal" {
                        if let Some(n) = node.args.get(i + 1).and_then(|a| a.as_f64()) {
                            heal_pct = n as f32;
                        }
                    }
                }
                i += 1;
            }
            Ok(Effect::AbsorbToHeal { shield_amount: shield, duration_ms: dur, heal_percent: heal_pct })
        }
        "shield_steal" => {
            let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
            Ok(Effect::ShieldSteal { amount })
        }
        "status_clone" => {
            let mut max_count = 3u32;
            for arg in &node.args {
                if let Arg::Ident(s) = arg {
                    if s == "max" {
                        // next arg
                    }
                }
                if let Arg::Number(n) = arg {
                    max_count = *n as u32;
                }
            }
            Ok(Effect::StatusClone { max_count })
        }
        "detonate" => {
            let mult = node.args.first().and_then(|a| a.as_f64()).unwrap_or(1.0) as f32;
            Ok(Effect::Detonate { damage_multiplier: mult })
        }
        "status_transfer" => {
            let steal = node.args.iter().any(|a| matches!(a, Arg::Ident(s) if s == "steal"));
            Ok(Effect::StatusTransfer { steal_buffs: steal })
        }
        "on_hit_buff" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            let mut on_hit_effects = Vec::new();
            for child in &node.children {
                on_hit_effects.push(lower_effect(child)?);
            }
            Ok(Effect::OnHitBuff { duration_ms: dur, on_hit_effects })
        }
        "obstacle" => {
            let width = node.args.first().and_then(|a| a.as_f64()).unwrap_or(2.0) as f32;
            // Skip "x" arg
            let height = node.args.iter().filter_map(|a| a.as_f64()).nth(1).unwrap_or(1.0) as f32;
            Ok(Effect::Obstacle { width, height })
        }
        "projectile_block" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::ProjectileBlock { duration_ms: dur })
        }
        "attach" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Attach { duration_ms: dur })
        }
        "evolve_ability" => {
            let idx = node.args.first().and_then(|a| a.as_u32()).unwrap_or(0) as usize;
            Ok(Effect::EvolveAbility { ability_index: idx })
        }
        "duel" => {
            let dur = node.args.first().and_then(|a| a.as_duration_ms())
                .or(node.duration).unwrap_or(0);
            Ok(Effect::Duel { duration_ms: dur })
        }
        other => Err(format!("unknown effect type: {other}")),
    }
}

fn lower_scaling(nodes: &[ScalingNode]) -> Vec<ScalingTerm> {
    nodes.iter().map(|s| {
        let stat = match s.stat.as_str() {
            "target_max_hp" => StatRef::TargetMaxHp,
            "target_current_hp" => StatRef::TargetCurrentHp,
            "target_missing_hp" => StatRef::TargetMissingHp,
            "caster_max_hp" => StatRef::CasterMaxHp,
            "caster_current_hp" => StatRef::CasterCurrentHp,
            "caster_missing_hp" => StatRef::CasterMissingHp,
            "caster_attack_damage" => StatRef::CasterAttackDamage,
            other => {
                // Check for target_stacks("name") or caster_stacks("name")
                if let Some(rest) = other.strip_prefix("target_stacks") {
                    let name = rest.trim_start_matches('(').trim_end_matches(')').trim_matches('"').to_string();
                    StatRef::TargetStacks { name }
                } else if let Some(rest) = other.strip_prefix("caster_stacks") {
                    let name = rest.trim_start_matches('(').trim_end_matches(')').trim_matches('"').to_string();
                    StatRef::CasterStacks { name }
                } else {
                    StatRef::CasterAttackDamage // fallback
                }
            }
        };
        ScalingTerm {
            stat,
            percent: s.percent as f32,
            max: s.cap.unwrap_or(0),
            consume: s.consume,
        }
    }).collect()
}

// Condition, area, and tag lowering are in lower.rs
