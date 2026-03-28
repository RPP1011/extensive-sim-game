//! Area, tag, condition, scaling, and formatting emission helpers.

use crate::effects::types::*;

pub(super) fn emit_area(area: &Area) -> String {
    match area {
        Area::SingleTarget | Area::SelfOnly => String::new(),
        Area::Circle { radius } => format!("in circle({})", fmt_f32(*radius)),
        Area::Cone { radius, angle_deg } => {
            format!("in cone({}, {})", fmt_f32(*radius), fmt_f32(*angle_deg))
        }
        Area::Line { length, width } => {
            format!("in line({}, {})", fmt_f32(*length), fmt_f32(*width))
        }
        Area::Ring { inner_radius, outer_radius } => {
            format!("in ring({}, {})", fmt_f32(*inner_radius), fmt_f32(*outer_radius))
        }
        Area::Spread { radius, max_targets } => {
            if *max_targets > 0 {
                format!("in spread({}, {})", fmt_f32(*radius), max_targets)
            } else {
                format!("in spread({})", fmt_f32(*radius))
            }
        }
    }
}

pub(super) fn emit_tags(tags: &Tags) -> String {
    if tags.is_empty() { return String::new(); }
    let mut pairs: Vec<_> = tags.iter().collect();
    pairs.sort_by_key(|(k, _)| k.to_string());
    let inner: Vec<String> = pairs.iter()
        .map(|(k, v)| format!("{k}: {}", **v as i32))
        .collect();
    format!("[{}]", inner.join(", "))
}

pub(super) fn emit_condition(cond: &Condition) -> String {
    match cond {
        Condition::Always => String::new(),
        Condition::TargetHpBelow { percent } => format!("target_hp_below({}%)", *percent as i32),
        Condition::TargetHpAbove { percent } => format!("target_hp_above({}%)", *percent as i32),
        Condition::CasterHpBelow { percent } => format!("caster_hp_below({}%)", *percent as i32),
        Condition::CasterHpAbove { percent } => format!("caster_hp_above({}%)", *percent as i32),
        Condition::TargetIsStunned => "target_is_stunned".to_string(),
        Condition::TargetIsSlowed => "target_is_slowed".to_string(),
        Condition::TargetIsRooted => "target_is_rooted".to_string(),
        Condition::TargetIsSilenced => "target_is_silenced".to_string(),
        Condition::TargetIsFeared => "target_is_feared".to_string(),
        Condition::TargetIsTaunted => "target_is_taunted".to_string(),
        Condition::TargetIsBanished => "target_is_banished".to_string(),
        Condition::TargetIsStealthed => "target_is_stealthed".to_string(),
        Condition::TargetIsCharmed => "target_is_charmed".to_string(),
        Condition::TargetIsPolymorphed => "target_is_polymorphed".to_string(),
        Condition::HitCountAbove { count } => format!("hit_count_above({count})"),
        Condition::TargetHasTag { tag } => format!("target_has_tag(\"{tag}\")"),
        Condition::CasterHasStatus { status } => format!("caster_has_status(\"{status}\")"),
        Condition::TargetHasStatus { status } => format!("target_has_status(\"{status}\")"),
        Condition::TargetDebuffCount { min_count } => format!("target_debuff_count({min_count})"),
        Condition::CasterBuffCount { min_count } => format!("caster_buff_count({min_count})"),
        Condition::AllyCountBelow { count } => format!("ally_count_below({count})"),
        Condition::EnemyCountBelow { count } => format!("enemy_count_below({count})"),
        Condition::TargetStackCount { name, min_count } => {
            format!("target_stack_count(\"{name}\", {min_count})")
        }
        Condition::TargetDistanceBelow { range } => {
            format!("target_distance_below({})", fmt_f32(*range))
        }
        Condition::TargetDistanceAbove { range } => {
            format!("target_distance_above({})", fmt_f32(*range))
        }
        Condition::CasterResourceBelow { percent } => {
            format!("caster_resource_below({}%)", *percent as i32)
        }
        Condition::CasterResourceAbove { percent } => {
            format!("caster_resource_above({}%)", *percent as i32)
        }
        // --- Campaign Conditions ---
        Condition::FactionHostile => "faction_hostile".to_string(),
        Condition::AtWar => "at_war".to_string(),
        Condition::CrisisActive => "crisis_active".to_string(),
        Condition::GoldAbove { amount } => format!("gold_above({})", fmt_f32(*amount)),
        Condition::GoldBelow { amount } => format!("gold_below({})", fmt_f32(*amount)),
        Condition::Outnumbered => "outnumbered".to_string(),
        Condition::AllyInjured => "ally_injured".to_string(),
        Condition::Alone => "alone".to_string(),
        Condition::NearDeath => "near_death".to_string(),

        Condition::And { conditions } => {
            let inner: Vec<String> = conditions.iter().map(emit_condition).collect();
            format!("and({})", inner.join(", "))
        }
        Condition::Or { conditions } => {
            let inner: Vec<String> = conditions.iter().map(emit_condition).collect();
            format!("or({})", inner.join(", "))
        }
        Condition::Not { condition } => {
            format!("not({})", emit_condition(condition))
        }
    }
}

pub(super) fn emit_target_filter(filter: &super::super::types::TargetFilter) -> String {
    use super::super::types::TargetFilter;
    match filter {
        TargetFilter::UnderCommand => "targeting under_command".to_string(),
        TargetFilter::LoyaltyAbove { threshold } => format!("targeting loyalty_above({})", fmt_f32(*threshold)),
        TargetFilter::LoyaltyBelow { threshold } => format!("targeting loyalty_below({})", fmt_f32(*threshold)),
        TargetFilter::HasClass { class_name } => format!("targeting has_class(\"{}\")", class_name),
        TargetFilter::LevelAbove { level } => format!("targeting level_above({})", level),
        TargetFilter::LevelBelow { level } => format!("targeting level_below({})", level),
        TargetFilter::HasStatus { status } => format!("targeting has_status(\"{}\")", status),
        TargetFilter::FactionMember { faction } => format!("targeting faction(\"{}\")", faction),
        TargetFilter::Injured => "targeting injured".to_string(),
        TargetFilter::Healthy => "targeting healthy".to_string(),
    }
}

pub(crate) fn fmt_duration(ms: u32) -> String {
    if ms == u32::MAX {
        "while_alive".to_string()
    } else if ms == 0 {
        "0ms".to_string()
    } else if ms % 1000 == 0 {
        format!("{}s", ms / 1000)
    } else {
        format!("{ms}ms")
    }
}

/// Format a duration with the `for` prefix, or bare `while_alive` for permanent auras.
pub(crate) fn fmt_for_duration(ms: u32) -> String {
    if ms == u32::MAX {
        "while_alive".to_string()
    } else {
        format!("for {}", fmt_duration(ms))
    }
}

pub(crate) fn fmt_tick_duration(ticks: u32) -> String {
    format!("{}t", ticks)
}

pub(crate) fn fmt_f32(v: f32) -> String {
    if v == v.floor() && v.abs() < 10000.0 {
        format!("{:.1}", v)
    } else {
        // Round to 2 decimal places for clean DSL output
        format!("{:.2}", v)
    }
}
