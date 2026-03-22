//! Emit AbilityDef as DSL text (reverse of parsing).
//!
//! Used to generate DSL text for transformer training data.

use crate::effects::defs::{AbilityDef, AbilityTargeting};
use crate::effects::types::*;

use super::emit_effects::{emit_effect, emit_scaling};
use super::emit_helpers::{emit_area, emit_tags, emit_condition, fmt_duration, fmt_f32};

/// Convert an AbilityDef to its DSL text representation.
pub fn emit_ability_dsl(def: &AbilityDef) -> String {
    let mut lines = Vec::new();
    // Use underscore-joined name to avoid multi-word UNK issues in tokenizer
    let safe_name = def.name.replace(' ', "_");
    lines.push(format!("ability {safe_name} {{"));

    // Header line: target + range
    let target = match def.targeting {
        AbilityTargeting::TargetEnemy => "enemy",
        AbilityTargeting::TargetAlly => "ally",
        AbilityTargeting::SelfCast => "self",
        AbilityTargeting::SelfAoe => "self_aoe",
        AbilityTargeting::GroundTarget => "ground",
        AbilityTargeting::Direction => "direction",
        AbilityTargeting::Vector => "vector",
        AbilityTargeting::Global => "global",
    };
    if def.range > 0.0 {
        lines.push(format!("    target: {target}, range: {}", fmt_f32(def.range)));
    } else {
        lines.push(format!("    target: {target}"));
    }

    // Cooldown + cast
    let mut timing = Vec::new();
    if def.cooldown_ms > 0 {
        timing.push(format!("cooldown: {}", fmt_duration(def.cooldown_ms)));
    }
    if def.cast_time_ms > 0 {
        timing.push(format!("cast: {}", fmt_duration(def.cast_time_ms)));
    }
    if !timing.is_empty() {
        lines.push(format!("    {}", timing.join(", ")));
    }

    // Hint
    if !def.ai_hint.is_empty() {
        lines.push(format!("    hint: {}", def.ai_hint));
    }

    // Cost
    if def.resource_cost > 0 {
        lines.push(format!("    cost: {}", def.resource_cost));
    }

    // Charges
    if def.max_charges > 0 {
        lines.push(format!("    charges: {}", def.max_charges));
        if def.charge_recharge_ms > 0 {
            lines.push(format!("    recharge: {}", fmt_duration(def.charge_recharge_ms)));
        }
    }

    // Toggle
    if def.is_toggle {
        lines.push("    toggle".to_string());
    }

    // Recast
    if def.recast_count > 0 {
        lines.push(format!("    recast: {}", def.recast_count));
        if def.recast_window_ms > 0 {
            lines.push(format!("    recast_window: {}", fmt_duration(def.recast_window_ms)));
        }
    }

    // Unstoppable
    if def.unstoppable {
        lines.push("    unstoppable".to_string());
    }

    // Zone tag
    if let Some(ref tag) = def.zone_tag {
        lines.push(format!("    zone_tag: {tag}"));
    }

    // Blank line before effects
    if !def.effects.is_empty() || def.delivery.is_some() {
        lines.push(String::new());
    }

    // Delivery
    if let Some(ref delivery) = def.delivery {
        emit_delivery(delivery, &mut lines, 1);
    }

    // Top-level effects
    for eff in &def.effects {
        emit_conditional_effect(eff, &mut lines, 1);
    }

    lines.push("}".to_string());
    lines.join("\n")
}

fn emit_delivery(delivery: &Delivery, lines: &mut Vec<String>, indent: usize) {
    let pad = "    ".repeat(indent);
    match delivery {
        Delivery::Instant => {}
        Delivery::Projectile { speed, pierce, width, on_hit, on_arrival } => {
            let mut params = vec![format!("speed: {}", fmt_f32(*speed))];
            if *pierce { params.push("pierce".to_string()); }
            if *width > 0.0 { params.push(format!("width: {}", fmt_f32(*width))); }
            lines.push(format!("{pad}deliver projectile {{ {} }} {{", params.join(", ")));
            emit_delivery_hooks(on_hit, on_arrival, lines, indent + 1);
            lines.push(format!("{pad}}}"));
        }
        Delivery::Channel { duration_ms, tick_interval_ms } => {
            lines.push(format!("{pad}deliver channel {{ duration: {}, tick: {} }} {{",
                fmt_duration(*duration_ms), fmt_duration(*tick_interval_ms)));
            // Channel uses on_hit for tick effects
            lines.push(format!("{pad}}}"));
        }
        Delivery::Zone { duration_ms, tick_interval_ms } => {
            lines.push(format!("{pad}deliver zone {{ duration: {}, tick: {} }} {{",
                fmt_duration(*duration_ms), fmt_duration(*tick_interval_ms)));
            lines.push(format!("{pad}}}"));
        }
        Delivery::Tether { max_range, tick_interval_ms, on_complete } => {
            let mut params = vec![format!("range: {}", fmt_f32(*max_range))];
            if *tick_interval_ms > 0 {
                params.push(format!("tick: {}", fmt_duration(*tick_interval_ms)));
            }
            lines.push(format!("{pad}deliver tether {{ {} }} {{", params.join(", ")));
            if !on_complete.is_empty() {
                let inner_pad = "    ".repeat(indent + 1);
                lines.push(format!("{inner_pad}on_complete {{"));
                for eff in on_complete {
                    emit_conditional_effect(eff, lines, indent + 2);
                }
                lines.push(format!("{inner_pad}}}"));
            }
            lines.push(format!("{pad}}}"));
        }
        Delivery::Trap { duration_ms, trigger_radius, arm_time_ms } => {
            let mut params = vec![
                format!("duration: {}", fmt_duration(*duration_ms)),
                format!("radius: {}", fmt_f32(*trigger_radius)),
            ];
            if *arm_time_ms > 0 {
                params.push(format!("arm_time: {}", fmt_duration(*arm_time_ms)));
            }
            lines.push(format!("{pad}deliver trap {{ {} }}", params.join(", ")));
        }
        Delivery::Chain { bounces, bounce_range, falloff, on_hit } => {
            let mut params = vec![
                format!("bounces: {bounces}"),
                format!("range: {}", fmt_f32(*bounce_range)),
            ];
            if *falloff > 0.0 { params.push(format!("falloff: {}", fmt_f32(*falloff))); }
            lines.push(format!("{pad}deliver chain {{ {} }} {{", params.join(", ")));
            if !on_hit.is_empty() {
                let inner_pad = "    ".repeat(indent + 1);
                lines.push(format!("{inner_pad}on_hit {{"));
                for eff in on_hit {
                    emit_conditional_effect(eff, lines, indent + 2);
                }
                lines.push(format!("{inner_pad}}}"));
            }
            lines.push(format!("{pad}}}"));
        }
    }
}

fn emit_delivery_hooks(
    on_hit: &[ConditionalEffect],
    on_arrival: &[ConditionalEffect],
    lines: &mut Vec<String>,
    indent: usize,
) {
    let pad = "    ".repeat(indent);
    if !on_hit.is_empty() {
        lines.push(format!("{pad}on_hit {{"));
        for eff in on_hit {
            emit_conditional_effect(eff, lines, indent + 1);
        }
        lines.push(format!("{pad}}}"));
    }
    if !on_arrival.is_empty() {
        lines.push(format!("{pad}on_arrival {{"));
        for eff in on_arrival {
            emit_conditional_effect(eff, lines, indent + 1);
        }
        lines.push(format!("{pad}}}"));
    }
}

fn emit_conditional_effect(ce: &ConditionalEffect, lines: &mut Vec<String>, indent: usize) {
    let pad = "    ".repeat(indent);
    let mut parts = vec![emit_effect(&ce.effect)];

    // Area
    if let Some(ref area) = ce.area {
        parts.push(emit_area(area));
    }

    // Tags
    if !ce.tags.is_empty() {
        parts.push(emit_tags(&ce.tags));
    }

    // Condition
    if let Some(ref cond) = ce.condition {
        if !matches!(cond, Condition::Always) {
            parts.push(format!("when {}", emit_condition(cond)));
        }
    }

    // Scaling (from effect)
    let scaling = emit_scaling(&ce.effect);
    if !scaling.is_empty() {
        parts.push(scaling);
    }

    lines.push(format!("{pad}{}", parts.join(" ")));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::effect_enum::Effect;

    #[test]
    fn test_emit_simple_ability() {
        let def = AbilityDef {
            name: "Fireball".to_string(),
            targeting: AbilityTargeting::TargetEnemy,
            range: 5.0,
            cooldown_ms: 5000,
            cast_time_ms: 300,
            ai_hint: "damage".to_string(),
            effects: vec![ConditionalEffect {
                effect: Effect::Damage {
                    amount: 55,
                    amount_per_tick: 0,
                    duration_ms: 0,
                    tick_interval_ms: 0,
                    scaling_stat: None,
                    scaling_percent: 0.0,
                    damage_type: DamageType::Physical,
                    bonus: vec![],
                },
                condition: None,
                area: None,
                tags: [("FIRE".to_string(), 60.0)].into_iter().collect(),
                stacking: Stacking::default(),
                chance: 0.0,
                else_effects: vec![],
            }],
            ..Default::default()
        };

        let dsl = emit_ability_dsl(&def);
        assert!(dsl.contains("ability Fireball {"));
        assert!(dsl.contains("target: enemy, range: 5.0"));
        assert!(dsl.contains("cooldown: 5s, cast: 300ms"));
        assert!(dsl.contains("hint: damage"));
        assert!(dsl.contains("damage 55"));
        assert!(dsl.contains("[FIRE: 60]"));
    }

    #[test]
    fn test_roundtrip_hero_abilities() {
        // Parse real hero ability files, emit DSL, parse again
        let mut total = 0;
        let mut roundtripped = 0;
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest.join("../../assets/hero_templates"),
            manifest.join("../../dataset/hero_templates"),
        ];
        for dir in &candidates {
            if !dir.is_dir() { continue; }
        for entry in std::fs::read_dir(dir).expect("read_dir failed") {
            let path = entry.unwrap().path();
            if path.extension().map_or(true, |e| e != "ability") { continue; }
            let source = std::fs::read_to_string(&path).unwrap();
            let (abilities, _passives) = match crate::effects::dsl::parse_abilities(&source) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for def in &abilities {
                total += 1;
                let emitted = emit_ability_dsl(def);
                match crate::effects::dsl::parse_abilities(&emitted) {
                    Ok((reparsed, _)) => {
                        assert_eq!(reparsed.len(), 1, "Expected 1 ability from re-parse of:\n{emitted}");
                        let r = &reparsed[0];
                        assert_eq!(r.name, def.name);
                        assert_eq!(r.ai_hint, def.ai_hint);
                        assert_eq!(r.cooldown_ms, def.cooldown_ms);
                        assert_eq!(r.cast_time_ms, def.cast_time_ms);
                        roundtripped += 1;
                    }
                    Err(e) => {
                        panic!("Failed to re-parse emitted DSL for {}:\n{emitted}\nError: {e}", def.name);
                    }
                }
            }
        }
        }
        println!("Round-tripped {roundtripped}/{total} abilities from hero templates");
        assert!(total > 0, "No hero ability files found");
    }

    #[test]
    fn test_emit_duration_formatting() {
        assert_eq!(fmt_duration(1000), "1s");
        assert_eq!(fmt_duration(500), "500ms");
        assert_eq!(fmt_duration(2500), "2500ms");
        assert_eq!(fmt_duration(0), "0ms");
    }
}
