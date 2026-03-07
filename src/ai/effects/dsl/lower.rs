//! Lowering pass: AST → AbilityDef / PassiveDef.

use std::collections::HashMap;

use super::ast::*;
use crate::ai::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
use crate::ai::effects::effect_enum::Effect;
use crate::ai::effects::types::*;

/// Lower an `AbilityFile` AST into concrete ability and passive definitions.
pub fn lower_file(file: &AbilityFile) -> Result<(Vec<AbilityDef>, Vec<PassiveDef>), String> {
    let mut abilities = Vec::new();
    let mut passives = Vec::new();

    for item in &file.items {
        match item {
            TopLevel::Ability(node) => abilities.push(lower_ability(node)?),
            TopLevel::Passive(node) => passives.push(lower_passive(node)?),
        }
    }

    Ok((abilities, passives))
}

fn lower_ability(node: &AbilityNode) -> Result<AbilityDef, String> {
    let mut def = AbilityDef::default();
    def.name = node.name.clone();

    // Apply properties
    for prop in &node.props {
        apply_ability_prop(&mut def, prop)?;
    }

    // Lower effects
    for eff in &node.effects {
        def.effects.push(lower_effect(eff)?);
    }

    // Lower delivery
    if let Some(delivery) = &node.delivery {
        def.delivery = Some(lower_delivery(delivery)?);
    }

    // Lower morph
    if let Some(morph) = &node.morph {
        let inner = lower_ability(&morph.inner)?;
        def.morph_into = Some(Box::new(inner));
        def.morph_duration_ms = morph.duration_ms;
    }

    // Lower recasts
    if !node.recasts.is_empty() {
        let max_idx = node.recasts.iter().map(|r| r.index).max().unwrap_or(0);
        def.recast_count = max_idx;
        def.recast_effects = Vec::new();
        for _ in 0..max_idx {
            def.recast_effects.push(Vec::new());
        }
        for recast in &node.recasts {
            let idx = (recast.index as usize).saturating_sub(1);
            if idx < def.recast_effects.len() {
                for eff in &recast.effects {
                    def.recast_effects[idx].push(lower_effect(eff)?);
                }
            }
        }
    }

    Ok(def)
}

fn apply_ability_prop(def: &mut AbilityDef, prop: &Property) -> Result<(), String> {
    match prop.key.as_str() {
        "target" | "targeting" => {
            def.targeting = match prop.value.as_ident()? {
                "enemy" => AbilityTargeting::TargetEnemy,
                "ally" => AbilityTargeting::TargetAlly,
                "self" => AbilityTargeting::SelfCast,
                "self_aoe" => AbilityTargeting::SelfAoe,
                "ground" => AbilityTargeting::GroundTarget,
                "direction" => AbilityTargeting::Direction,
                "vector" => AbilityTargeting::Vector,
                "global" => AbilityTargeting::Global,
                other => return Err(format!("unknown targeting: {other}")),
            };
        }
        "range" => def.range = prop.value.as_f64()? as f32,
        "cooldown" => def.cooldown_ms = prop.value.as_duration_ms()?,
        "cast" => def.cast_time_ms = prop.value.as_duration_ms()?,
        "hint" | "ai_hint" => def.ai_hint = prop.value.as_string()?,
        "cost" | "resource_cost" => def.resource_cost = prop.value.as_f64()? as i32,
        "zone_tag" => def.zone_tag = Some(prop.value.as_string()?),
        "charges" => def.max_charges = prop.value.as_f64()? as u32,
        "recharge" => def.charge_recharge_ms = prop.value.as_duration_ms()?,
        "toggle" => def.is_toggle = true,
        "toggle_cost" => def.toggle_cost_per_sec = prop.value.as_f64()? as f32,
        "recast" => def.recast_count = prop.value.as_f64()? as u32,
        "recast_window" => def.recast_window_ms = prop.value.as_duration_ms()?,
        "unstoppable" => def.unstoppable = true,
        "form" => def.form = Some(prop.value.as_string()?),
        "swap_form" => def.swap_form = Some(prop.value.as_string()?),
        other => return Err(format!("unknown ability property: {other}")),
    }
    Ok(())
}

fn lower_passive(node: &PassiveNode) -> Result<PassiveDef, String> {
    let mut trigger = None;
    let mut cooldown_ms = 0u32;
    let mut range = 0.0f32;

    for prop in &node.props {
        match prop.key.as_str() {
            "trigger" => trigger = Some(lower_trigger_value(&prop.value)?),
            "cooldown" => cooldown_ms = prop.value.as_duration_ms()?,
            "range" => range = prop.value.as_f64()? as f32,
            other => return Err(format!("unknown passive property: {other}")),
        }
    }

    let trigger = trigger.ok_or_else(|| format!("passive '{}' missing trigger", node.name))?;

    let mut effects = Vec::new();
    for eff in &node.effects {
        effects.push(lower_effect(eff)?);
    }

    Ok(PassiveDef {
        name: node.name.clone(),
        trigger,
        cooldown_ms,
        effects,
        range,
    })
}

fn lower_trigger_value(value: &PropValue) -> Result<Trigger, String> {
    // Trigger values come in as identifiers, possibly with args encoded
    // in the ident string (parsed as "on_hp_below(50%)" → ident "on_hp_below"
    // but we need to handle the arg form).
    // The parser stores the full form, so we need to handle both cases.
    let s = value.as_string()?;
    parse_trigger_string(&s)
}

fn parse_trigger_string(s: &str) -> Result<Trigger, String> {
    // Check for function-call form: name(args)
    if let Some(paren_pos) = s.find('(') {
        let name = &s[..paren_pos];
        let args_str = &s[paren_pos + 1..s.len() - 1]; // strip parens

        match name {
            "on_hp_below" => {
                let pct = parse_trigger_arg_percent(args_str)?;
                Ok(Trigger::OnHpBelow { percent: pct })
            }
            "on_hp_above" => {
                let pct = parse_trigger_arg_percent(args_str)?;
                Ok(Trigger::OnHpAbove { percent: pct })
            }
            "on_ally_damaged" => {
                let range = parse_trigger_kwarg_f32(args_str, "range")?;
                Ok(Trigger::OnAllyDamaged { range })
            }
            "on_ally_killed" => {
                let range = parse_trigger_kwarg_f32(args_str, "range")?;
                Ok(Trigger::OnAllyKilled { range })
            }
            "periodic" => {
                let ms = parse_trigger_arg_duration(args_str)?;
                Ok(Trigger::Periodic { interval_ms: ms })
            }
            "on_stack_reached" => {
                // on_stack_reached("poison", 4)
                let parts: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
                if parts.len() != 2 {
                    return Err(format!("on_stack_reached expects 2 args, got {}", parts.len()));
                }
                let name = parts[0].trim_matches('"').to_string();
                let count = parts[1].parse::<u32>().map_err(|e| format!("bad count: {e}"))?;
                Ok(Trigger::OnStackReached { name, count })
            }
            _ => Err(format!("unknown trigger: {name}")),
        }
    } else {
        match s {
            "on_damage_dealt" => Ok(Trigger::OnDamageDealt),
            "on_damage_taken" => Ok(Trigger::OnDamageTaken),
            "on_kill" => Ok(Trigger::OnKill),
            "on_death" => Ok(Trigger::OnDeath),
            "on_ability_used" => Ok(Trigger::OnAbilityUsed),
            "on_shield_broken" => Ok(Trigger::OnShieldBroken),
            "on_stun_expire" => Ok(Trigger::OnStunExpire),
            "on_heal_received" => Ok(Trigger::OnHealReceived),
            "on_status_applied" => Ok(Trigger::OnStatusApplied),
            "on_status_expired" => Ok(Trigger::OnStatusExpired),
            "on_resurrect" => Ok(Trigger::OnResurrect),
            "on_dodge" => Ok(Trigger::OnDodge),
            "on_reflect" => Ok(Trigger::OnReflect),
            "on_auto_attack" => Ok(Trigger::OnAutoAttack),
            _ => Err(format!("unknown trigger: {s}")),
        }
    }
}

fn parse_trigger_arg_percent(s: &str) -> Result<f32, String> {
    let s = s.trim().trim_end_matches('%');
    s.parse::<f32>().map_err(|e| format!("bad percent: {e}"))
}

fn parse_trigger_kwarg_f32(s: &str, key: &str) -> Result<f32, String> {
    // "range: 5.0" or just "5.0"
    let s = s.trim();
    if let Some(pos) = s.find(':') {
        let val_str = s[pos + 1..].trim();
        val_str.parse::<f32>().map_err(|e| format!("bad {key}: {e}"))
    } else {
        s.parse::<f32>().map_err(|e| format!("bad {key}: {e}"))
    }
}

fn parse_trigger_arg_duration(s: &str) -> Result<u32, String> {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix("ms") {
        stripped.trim().parse::<u32>().map_err(|e| format!("bad duration: {e}"))
    } else if let Some(stripped) = s.strip_suffix('s') {
        let secs: f64 = stripped.trim().parse().map_err(|e| format!("bad duration: {e}"))?;
        Ok((secs * 1000.0) as u32)
    } else {
        s.parse::<u32>().map_err(|e| format!("bad duration: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Effect lowering
// ---------------------------------------------------------------------------

fn lower_effect(node: &EffectNode) -> Result<ConditionalEffect, String> {
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
            let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
            let bonus = lower_scaling(&node.scaling);
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
        "heal" => {
            let amount = node.args.first().and_then(|a| a.as_i32()).unwrap_or(0);
            let bonus = lower_scaling(&node.scaling);
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
            let mut is_blink = false;

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

fn lower_condition(node: &ConditionNode) -> Result<Condition, String> {
    match node {
        ConditionNode::Simple { name, args } => {
            match name.as_str() {
                "target_hp_below" => {
                    let pct = args.first().map(|a| cond_arg_to_f32(a)).unwrap_or(50.0);
                    Ok(Condition::TargetHpBelow { percent: pct })
                }
                "target_hp_above" => {
                    let pct = args.first().map(|a| cond_arg_to_f32(a)).unwrap_or(50.0);
                    Ok(Condition::TargetHpAbove { percent: pct })
                }
                "caster_hp_below" => {
                    let pct = args.first().map(|a| cond_arg_to_f32(a)).unwrap_or(50.0);
                    Ok(Condition::CasterHpBelow { percent: pct })
                }
                "caster_hp_above" => {
                    let pct = args.first().map(|a| cond_arg_to_f32(a)).unwrap_or(50.0);
                    Ok(Condition::CasterHpAbove { percent: pct })
                }
                "target_is_stunned" => Ok(Condition::TargetIsStunned),
                "target_is_slowed" => Ok(Condition::TargetIsSlowed),
                "target_is_rooted" => Ok(Condition::TargetIsRooted),
                "target_is_silenced" => Ok(Condition::TargetIsSilenced),
                "target_is_feared" => Ok(Condition::TargetIsFeared),
                "target_is_taunted" => Ok(Condition::TargetIsTaunted),
                "target_is_banished" => Ok(Condition::TargetIsBanished),
                "target_is_stealthed" => Ok(Condition::TargetIsStealthed),
                "target_is_charmed" => Ok(Condition::TargetIsCharmed),
                "target_is_polymorphed" => Ok(Condition::TargetIsPolymorphed),
                "target_has_tag" => {
                    let tag = args.first().map(cond_arg_to_string).unwrap_or_default();
                    Ok(Condition::TargetHasTag { tag })
                }
                "hit_count_above" => {
                    let count = args.first().map(|a| cond_arg_to_f32(a) as u32).unwrap_or(0);
                    Ok(Condition::HitCountAbove { count })
                }
                _ => Err(format!("unknown condition: {name}")),
            }
        }
        ConditionNode::And(subs) => {
            let conditions = subs.iter().map(lower_condition).collect::<Result<Vec<_>, _>>()?;
            Ok(Condition::And { conditions })
        }
        ConditionNode::Or(subs) => {
            let conditions = subs.iter().map(lower_condition).collect::<Result<Vec<_>, _>>()?;
            Ok(Condition::Or { conditions })
        }
        ConditionNode::Not(inner) => {
            let condition = lower_condition(inner)?;
            Ok(Condition::Not { condition: Box::new(condition) })
        }
    }
}

fn cond_arg_to_f32(arg: &CondArg) -> f32 {
    match arg {
        CondArg::Number(n) => *n as f32,
        CondArg::Percent(n) => *n as f32,
        _ => 0.0,
    }
}

fn cond_arg_to_string(arg: &CondArg) -> String {
    match arg {
        CondArg::StringLit(s) | CondArg::Ident(s) => s.clone(),
        CondArg::Number(n) => n.to_string(),
        CondArg::Percent(n) => format!("{n}%"),
    }
}

fn lower_area(node: &AreaNode) -> Result<Area, String> {
    match node.shape.as_str() {
        "circle" => {
            let radius = *node.args.first().ok_or("circle needs radius")? as f32;
            Ok(Area::Circle { radius })
        }
        "cone" => {
            let radius = *node.args.first().ok_or("cone needs radius")? as f32;
            let angle_deg = *node.args.get(1).ok_or("cone needs angle")? as f32;
            Ok(Area::Cone { radius, angle_deg })
        }
        "line" => {
            let length = *node.args.first().ok_or("line needs length")? as f32;
            let width = *node.args.get(1).ok_or("line needs width")? as f32;
            Ok(Area::Line { length, width })
        }
        "ring" => {
            let inner = *node.args.first().ok_or("ring needs inner_radius")? as f32;
            let outer = *node.args.get(1).ok_or("ring needs outer_radius")? as f32;
            Ok(Area::Ring { inner_radius: inner, outer_radius: outer })
        }
        "spread" => {
            let radius = *node.args.first().ok_or("spread needs radius")? as f32;
            let max_targets = node.args.get(1).map(|n| *n as u32).unwrap_or(0);
            Ok(Area::Spread { radius, max_targets })
        }
        other => Err(format!("unknown area shape: {other}")),
    }
}

fn lower_tags(tags: &[(String, f64)]) -> Tags {
    tags.iter().map(|(k, v)| (k.clone(), *v as f32)).collect()
}

fn lower_delivery(node: &DeliveryNode) -> Result<Delivery, String> {
    match node.method.as_str() {
        "projectile" => {
            let mut speed = 8.0f32;
            let mut pierce = false;
            let mut width = 0.0f32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "speed" => speed = val.as_f64().unwrap_or(8.0) as f32,
                    "pierce" => pierce = true,
                    "width" => width = val.as_f64().unwrap_or(0.0) as f32,
                    _ => {}
                }
            }

            let on_hit = node.on_hit.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;
            let on_arrival = node.on_arrival.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;

            Ok(Delivery::Projectile { speed, pierce, width, on_hit, on_arrival })
        }
        "chain" => {
            let mut bounces = 3u32;
            let mut bounce_range = 3.0f32;
            let mut falloff = 0.0f32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "bounces" => bounces = val.as_f64().unwrap_or(3.0) as u32,
                    "range" => bounce_range = val.as_f64().unwrap_or(3.0) as f32,
                    "falloff" => falloff = val.as_f64().unwrap_or(0.0) as f32,
                    _ => {}
                }
            }

            let on_hit = node.on_hit.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;

            Ok(Delivery::Chain { bounces, bounce_range, falloff, on_hit })
        }
        "zone" => {
            let mut duration_ms = 0u32;
            let mut tick_interval_ms = 1000u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "duration" => duration_ms = val.as_duration_ms().unwrap_or(0),
                    "tick" => tick_interval_ms = val.as_duration_ms().unwrap_or(1000),
                    _ => {}
                }
            }

            // Zone uses on_hit for tick effects
            Ok(Delivery::Zone { duration_ms, tick_interval_ms })
        }
        "channel" => {
            let mut duration_ms = 0u32;
            let mut tick_interval_ms = 500u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "duration" => duration_ms = val.as_duration_ms().unwrap_or(0),
                    "tick" => tick_interval_ms = val.as_duration_ms().unwrap_or(500),
                    _ => {}
                }
            }

            Ok(Delivery::Channel { duration_ms, tick_interval_ms })
        }
        "tether" => {
            let mut max_range = 5.0f32;
            let mut tick_interval_ms = 500u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "max_range" => max_range = val.as_f64().unwrap_or(5.0) as f32,
                    "tick" => tick_interval_ms = val.as_duration_ms().unwrap_or(500),
                    _ => {}
                }
            }

            let on_complete = node.on_complete.iter().map(lower_effect).collect::<Result<Vec<_>, _>>()?;

            Ok(Delivery::Tether { max_range, tick_interval_ms, on_complete })
        }
        "trap" => {
            let mut duration_ms = 0u32;
            let mut trigger_radius = 1.5f32;
            let mut arm_time_ms = 0u32;

            for (key, val) in &node.params {
                match key.as_str() {
                    "duration" => duration_ms = val.as_duration_ms().unwrap_or(0),
                    "trigger_radius" => trigger_radius = val.as_f64().unwrap_or(1.5) as f32,
                    "arm_time" => arm_time_ms = val.as_duration_ms().unwrap_or(0),
                    _ => {}
                }
            }

            Ok(Delivery::Trap { duration_ms, trigger_radius, arm_time_ms })
        }
        other => Err(format!("unknown delivery method: {other}")),
    }
}

// ---------------------------------------------------------------------------
// PropValue helpers
// ---------------------------------------------------------------------------

impl PropValue {
    fn as_ident(&self) -> Result<&str, String> {
        match self {
            PropValue::Ident(s) => Ok(s.as_str()),
            other => Err(format!("expected identifier, got {other:?}")),
        }
    }

    fn as_f64(&self) -> Result<f64, String> {
        match self {
            PropValue::Number(n) => Ok(*n),
            PropValue::Duration(ms) => Ok(*ms as f64),
            other => Err(format!("expected number, got {other:?}")),
        }
    }

    fn as_duration_ms(&self) -> Result<u32, String> {
        match self {
            PropValue::Duration(ms) => Ok(*ms),
            PropValue::Number(n) => Ok(*n as u32),
            other => Err(format!("expected duration, got {other:?}")),
        }
    }

    fn as_string(&self) -> Result<String, String> {
        match self {
            PropValue::Ident(s) | PropValue::StringLit(s) => Ok(s.clone()),
            PropValue::Number(n) => Ok(n.to_string()),
            PropValue::Duration(ms) => Ok(ms.to_string()),
            PropValue::Bool(b) => Ok(b.to_string()),
        }
    }
}
