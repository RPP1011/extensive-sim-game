//! Effect emission match arms (Effect → DSL string fragments).

use crate::effects::effect_enum::Effect;
use crate::effects::types::*;
use super::emit_helpers::{fmt_duration, fmt_f32};

pub(super) fn emit_effect(effect: &Effect) -> String {
    match effect {
        Effect::Damage { amount, amount_per_tick, duration_ms, damage_type, .. } => {
            if *amount_per_tick > 0 && *duration_ms > 0 {
                let dt = match damage_type {
                    DamageType::Magic => " magic",
                    DamageType::True => " true",
                    _ => "",
                };
                format!("damage {amount_per_tick}/tick{dt} for {}", fmt_duration(*duration_ms))
            } else {
                let dt = match damage_type {
                    DamageType::Magic => " magic",
                    DamageType::True => " true",
                    _ => "",
                };
                format!("damage {amount}{dt}")
            }
        }
        Effect::Heal { amount, amount_per_tick, duration_ms, .. } => {
            if *amount_per_tick > 0 && *duration_ms > 0 {
                format!("heal {amount_per_tick}/tick for {}", fmt_duration(*duration_ms))
            } else {
                format!("heal {amount}")
            }
        }
        Effect::Shield { amount, duration_ms } => {
            format!("shield {amount} for {}", fmt_duration(*duration_ms))
        }
        Effect::Stun { duration_ms } => format!("stun {}", fmt_duration(*duration_ms)),
        Effect::Slow { factor, duration_ms } => {
            format!("slow {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Root { duration_ms } => format!("root {}", fmt_duration(*duration_ms)),
        Effect::Silence { duration_ms } => format!("silence {}", fmt_duration(*duration_ms)),
        Effect::Fear { duration_ms } => format!("fear {}", fmt_duration(*duration_ms)),
        Effect::Taunt { duration_ms } => format!("taunt {}", fmt_duration(*duration_ms)),
        Effect::Knockback { distance } => format!("knockback {}", fmt_f32(*distance)),
        Effect::Pull { distance } => format!("pull {}", fmt_f32(*distance)),
        Effect::Dash { distance, to_target, is_blink, .. } => {
            let mut s = if *is_blink {
                format!("blink {}", fmt_f32(*distance))
            } else {
                format!("dash {}", fmt_f32(*distance))
            };
            if *to_target { s.push_str(" to_target"); }
            s
        }
        Effect::Buff { stat, factor, duration_ms } => {
            format!("buff {stat} {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Debuff { stat, factor, duration_ms } => {
            format!("debuff {stat} {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Duel { duration_ms } => format!("duel {}", fmt_duration(*duration_ms)),
        Effect::Summon { template, count, hp_percent, clone, directed, .. } => {
            let mut s = format!("summon \"{template}\"");
            if *count > 1 { s.push_str(&format!(" x{count}")); }
            if *clone { s.push_str(" clone"); }
            if *directed { s.push_str(" directed"); }
            if (*hp_percent - 100.0).abs() > 0.1 {
                s.push_str(&format!(" hp:{}%", *hp_percent as i32));
            }
            s
        }
        Effect::CommandSummons { speed } => format!("command_summons speed:{}", fmt_f32(*speed)),
        Effect::Dispel { target_tags } => {
            if target_tags.is_empty() {
                "dispel".to_string()
            } else {
                // Wrap tags as string literals so tokenizer maps them to [STR]
                let tags: Vec<String> = target_tags.iter()
                    .map(|t| format!("\"{t}\""))
                    .collect();
                format!("dispel {}", tags.join(" "))
            }
        }
        Effect::Reflect { percent, duration_ms } => {
            format!("reflect {} for {}", fmt_f32(*percent), fmt_duration(*duration_ms))
        }
        Effect::Lifesteal { percent, duration_ms } => {
            format!("lifesteal {} for {}", fmt_f32(*percent), fmt_duration(*duration_ms))
        }
        Effect::DamageModify { factor, duration_ms } => {
            format!("damage_modify {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::SelfDamage { amount } => format!("self_damage {amount}"),
        Effect::Execute { hp_threshold_percent } => {
            format!("execute {}%", *hp_threshold_percent as i32)
        }
        Effect::Blind { miss_chance, duration_ms } => {
            format!("blind {} for {}", fmt_f32(*miss_chance), fmt_duration(*duration_ms))
        }
        Effect::OnHitBuff { duration_ms, .. } => {
            format!("on_hit_buff for {}", fmt_duration(*duration_ms))
        }
        Effect::Resurrect { hp_percent } => format!("resurrect {}%", *hp_percent as i32),
        Effect::OverhealShield { duration_ms, .. } => {
            format!("overheal_shield for {}", fmt_duration(*duration_ms))
        }
        Effect::AbsorbToHeal { shield_amount, duration_ms, .. } => {
            format!("absorb_to_heal {shield_amount} for {}", fmt_duration(*duration_ms))
        }
        Effect::ShieldSteal { amount } => format!("shield_steal {amount}"),
        Effect::StatusClone { max_count } => format!("status_clone {max_count}"),
        Effect::Immunity { immune_to, duration_ms } => {
            format!("immunity {} for {}", immune_to.join(","), fmt_duration(*duration_ms))
        }
        Effect::Detonate { damage_multiplier } => {
            format!("detonate {}", fmt_f32(*damage_multiplier))
        }
        Effect::StatusTransfer { steal_buffs } => {
            if *steal_buffs { "status_transfer steal".to_string() }
            else { "status_transfer".to_string() }
        }
        Effect::DeathMark { duration_ms, damage_percent } => {
            format!("death_mark {} for {}", fmt_f32(*damage_percent), fmt_duration(*duration_ms))
        }
        Effect::Polymorph { duration_ms } => format!("polymorph {}", fmt_duration(*duration_ms)),
        Effect::Banish { duration_ms } => format!("banish {}", fmt_duration(*duration_ms)),
        Effect::Confuse { duration_ms } => format!("confuse {}", fmt_duration(*duration_ms)),
        Effect::Charm { duration_ms } => format!("charm {}", fmt_duration(*duration_ms)),
        Effect::Stealth { duration_ms, break_on_damage, break_on_ability } => {
            let mut s = format!("stealth for {}", fmt_duration(*duration_ms));
            if *break_on_damage { s.push_str(" break_on_damage"); }
            if *break_on_ability { s.push_str(" break_on_ability"); }
            s
        }
        Effect::Leash { max_range, duration_ms } => {
            format!("leash {} for {}", fmt_f32(*max_range), fmt_duration(*duration_ms))
        }
        Effect::Link { duration_ms, share_percent } => {
            format!("link {}% for {}", *share_percent as i32, fmt_duration(*duration_ms))
        }
        Effect::Redirect { duration_ms, charges } => {
            format!("redirect {charges} for {}", fmt_duration(*duration_ms))
        }
        Effect::Rewind { lookback_ms } => format!("rewind {}", fmt_duration(*lookback_ms)),
        Effect::CooldownModify { amount_ms, ability_name } => {
            if let Some(ref name) = ability_name {
                format!("cooldown_modify {amount_ms}ms \"{name}\"")
            } else {
                format!("cooldown_modify {amount_ms}ms")
            }
        }
        Effect::ApplyStacks { name, count, max_stacks, duration_ms } => {
            let mut s = format!("apply_stacks \"{name}\" {count}/{max_stacks}");
            if *duration_ms > 0 {
                s.push_str(&format!(" for {}", fmt_duration(*duration_ms)));
            }
            s
        }
        Effect::Obstacle { width, height } => {
            format!("obstacle {} {}", fmt_f32(*width), fmt_f32(*height))
        }
        Effect::Suppress { duration_ms } => format!("suppress {}", fmt_duration(*duration_ms)),
        Effect::Grounded { duration_ms } => format!("grounded {}", fmt_duration(*duration_ms)),
        Effect::ProjectileBlock { duration_ms } => {
            format!("projectile_block {}", fmt_duration(*duration_ms))
        }
        Effect::Attach { duration_ms } => format!("attach for {}", fmt_duration(*duration_ms)),
        Effect::EvolveAbility { ability_index } => format!("evolve {ability_index}"),
        Effect::Swap => "swap".to_string(),
    }
}

pub(super) fn emit_scaling(effect: &Effect) -> String {
    let bonus = match effect {
        Effect::Damage { bonus, scaling_stat, scaling_percent, .. } => {
            if let Some(ref stat) = scaling_stat {
                if *scaling_percent > 0.0 {
                    return format!("+ {}% {stat}", *scaling_percent as i32);
                }
            }
            bonus
        }
        Effect::Heal { bonus, scaling_stat, scaling_percent, .. } => {
            if let Some(ref stat) = scaling_stat {
                if *scaling_percent > 0.0 {
                    return format!("+ {}% {stat}", *scaling_percent as i32);
                }
            }
            bonus
        }
        _ => return String::new(),
    };
    if bonus.is_empty() { return String::new(); }
    let terms: Vec<String> = bonus.iter().map(|t| {
        let stat_name = match &t.stat {
            StatRef::TargetMaxHp => "target_max_hp",
            StatRef::TargetCurrentHp => "target_current_hp",
            StatRef::TargetMissingHp => "target_missing_hp",
            StatRef::CasterMaxHp => "caster_max_hp",
            StatRef::CasterCurrentHp => "caster_current_hp",
            StatRef::CasterMissingHp => "caster_missing_hp",
            StatRef::CasterAttackDamage => "caster_attack_damage",
            StatRef::TargetStacks { name } => return format!("{}% target_stacks({name})", t.percent as i32),
            StatRef::CasterStacks { name } => return format!("{}% caster_stacks({name})", t.percent as i32),
        };
        format!("{}% {stat_name}", t.percent as i32)
    }).collect();
    format!("+ {}", terms.join(" + "))
}
