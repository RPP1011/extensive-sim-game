use crate::ai::effects::{AbilityTarget, Condition, StatusKind};

use super::events::SimEvent;
use super::types::*;
use super::helpers::unit_has_status;

/// Evaluate condition and emit a `ConditionalEffectApplied` event when a real
/// condition (not `None` / `Always`) evaluates to true.
pub fn evaluate_condition_tracked(
    condition: &Option<Condition>,
    caster_idx: usize,
    target: AbilityTarget,
    state: &SimState,
    tick: u64,
    events: &mut Vec<SimEvent>,
) -> bool {
    let result = evaluate_condition(condition, caster_idx, target, state);
    if result {
        if let Some(cond) = condition {
            if !matches!(cond, Condition::Always) {
                events.push(SimEvent::ConditionalEffectApplied {
                    tick,
                    unit_id: state.units[caster_idx].id,
                    condition: format!("{:?}", cond),
                });
            }
        }
    }
    result
}

pub fn evaluate_condition(
    condition: &Option<Condition>,
    caster_idx: usize,
    target: AbilityTarget,
    state: &SimState,
) -> bool {
    let Some(cond) = condition else {
        return true;
    };
    match cond {
        Condition::Always => true,
        Condition::TargetHpBelow { percent } => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return (t.hp as f32 / t.max_hp.max(1) as f32) * 100.0 < *percent;
                }
            }
            false
        }
        Condition::TargetHpAbove { percent } => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return (t.hp as f32 / t.max_hp.max(1) as f32) * 100.0 > *percent;
                }
            }
            false
        }
        Condition::TargetIsStunned => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return t.control_remaining_ms > 0
                        || t.status_effects.iter().any(|s| matches!(s.kind, StatusKind::Stun));
                }
            }
            false
        }
        Condition::TargetIsSlowed => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return t.status_effects.iter().any(|s| matches!(s.kind, StatusKind::Slow { .. }));
                }
            }
            false
        }
        Condition::CasterHpBelow { percent } => {
            let c = &state.units[caster_idx];
            (c.hp as f32 / c.max_hp.max(1) as f32) * 100.0 < *percent
        }
        Condition::CasterHpAbove { percent } => {
            let c = &state.units[caster_idx];
            (c.hp as f32 / c.max_hp.max(1) as f32) * 100.0 > *percent
        }
        Condition::HitCountAbove { count: _ } => true, // evaluated during AoE dispatch
        Condition::TargetHasTag { ref tag } => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return t.resistance_tags.contains_key(tag)
                        || t.status_effects.iter().any(|s| s.tags.contains_key(tag));
                }
            }
            false
        }
        Condition::TargetIsRooted => target_has_status_kind(target, state, "root"),
        Condition::TargetIsSilenced => target_has_status_kind(target, state, "silence"),
        Condition::TargetIsFeared => target_has_status_kind(target, state, "fear"),
        Condition::TargetIsTaunted => target_has_status_kind(target, state, "taunt"),
        Condition::TargetIsBanished => target_has_status_kind(target, state, "banish"),
        Condition::TargetIsStealthed => target_has_status_kind(target, state, "stealth"),
        Condition::TargetIsCharmed => target_has_status_kind(target, state, "charm"),
        Condition::TargetIsPolymorphed => target_has_status_kind(target, state, "polymorph"),
        Condition::CasterHasStatus { ref status } => unit_has_status(&state.units[caster_idx], status),
        Condition::TargetHasStatus { ref status } => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return unit_has_status(t, status);
                }
            }
            false
        }
        Condition::TargetDebuffCount { min_count } => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    let count = t.status_effects.iter().filter(|s| match &s.kind {
                        StatusKind::Debuff { .. } | StatusKind::Slow { .. } | StatusKind::Dot { .. }
                        | StatusKind::Blind { .. } => true,
                        StatusKind::DamageModify { factor } => *factor > 1.0,
                        _ => false,
                    }).count() as u32;
                    return count >= *min_count;
                }
            }
            false
        }
        Condition::CasterBuffCount { min_count } => {
            let count = state.units[caster_idx].status_effects.iter().filter(|s| matches!(s.kind,
                StatusKind::Buff { .. } | StatusKind::Shield { .. } | StatusKind::Hot { .. }
                | StatusKind::Lifesteal { .. } | StatusKind::Reflect { .. }
            )).count() as u32;
            count >= *min_count
        }
        Condition::AllyCountBelow { count } => {
            let caster_team = state.units[caster_idx].team;
            let allies = state.units.iter().filter(|u| u.hp > 0 && u.team == caster_team).count() as u32;
            allies < *count
        }
        Condition::EnemyCountBelow { count } => {
            let caster_team = state.units[caster_idx].team;
            let enemies = state.units.iter().filter(|u| u.hp > 0 && u.team != caster_team).count() as u32;
            enemies < *count
        }
        Condition::TargetStackCount { ref name, min_count } => {
            if let AbilityTarget::Unit(tid) = target {
                if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                    return t.status_effects.iter().any(|s| {
                        matches!(&s.kind, StatusKind::Stacks { name: n, count, .. } if n == name && *count >= *min_count)
                    });
                }
            }
            false
        }
        // --- Compound Conditions ---
        Condition::And { ref conditions } => {
            conditions.iter().all(|c| evaluate_condition(&Some(c.clone()), caster_idx, target, state))
        }
        Condition::Or { ref conditions } => {
            conditions.iter().any(|c| evaluate_condition(&Some(c.clone()), caster_idx, target, state))
        }
        Condition::Not { ref condition } => {
            !evaluate_condition(&Some(*condition.clone()), caster_idx, target, state)
        }
    }
}

fn target_has_status_kind(target: AbilityTarget, state: &SimState, kind_name: &str) -> bool {
    if let AbilityTarget::Unit(tid) = target {
        if let Some(t) = state.units.iter().find(|u| u.id == tid) {
            return unit_has_status(t, kind_name);
        }
    }
    false
}

