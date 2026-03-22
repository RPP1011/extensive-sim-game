//! Lowering pass: AST → AbilityDef / PassiveDef.

use super::ast::*;
use super::lower_effects::lower_effect;
use super::lower_delivery::lower_delivery;
use crate::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
use crate::effects::types::*;

// ---------------------------------------------------------------------------
// Condition, area, and tag lowering helpers
// ---------------------------------------------------------------------------

pub(super) fn lower_condition(node: &ConditionNode) -> Result<Condition, String> {
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

pub(super) fn lower_area(node: &AreaNode) -> Result<Area, String> {
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
        "single_target" => Ok(Area::SingleTarget),
        "self" | "self_only" => Ok(Area::SelfOnly),
        other => Err(format!("unknown area shape: {other}")),
    }
}

pub(super) fn lower_tags(tags: &[(String, f64)]) -> Tags {
    tags.iter().map(|(k, v)| (k.clone(), *v as f32)).collect()
}

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
// PropValue helpers
// ---------------------------------------------------------------------------

impl PropValue {
    pub(crate) fn as_ident(&self) -> Result<&str, String> {
        match self {
            PropValue::Ident(s) => Ok(s.as_str()),
            other => Err(format!("expected identifier, got {other:?}")),
        }
    }

    pub(crate) fn as_f64(&self) -> Result<f64, String> {
        match self {
            PropValue::Number(n) => Ok(*n),
            PropValue::Duration(ms) => Ok(*ms as f64),
            other => Err(format!("expected number, got {other:?}")),
        }
    }

    pub(crate) fn as_duration_ms(&self) -> Result<u32, String> {
        match self {
            PropValue::Duration(ms) => Ok(*ms),
            PropValue::Number(n) => Ok(*n as u32),
            other => Err(format!("expected duration, got {other:?}")),
        }
    }

    pub(crate) fn as_string(&self) -> Result<String, String> {
        match self {
            PropValue::Ident(s) | PropValue::StringLit(s) => Ok(s.clone()),
            PropValue::Number(n) => Ok(n.to_string()),
            PropValue::Duration(ms) => Ok(ms.to_string()),
            PropValue::Bool(b) => Ok(b.to_string()),
        }
    }
}
