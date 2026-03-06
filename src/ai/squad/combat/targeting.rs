use std::cmp::Ordering;

use crate::ai::core::{distance, SimState, UnitState};

use crate::ai::squad::personality::Personality;
use crate::ai::squad::state::{SquadBlackboard, TickContext};

pub(in crate::ai::squad) fn choose_target(
    state: &SimState,
    unit_id: u32,
    personality: &Personality,
    board: SquadBlackboard,
    sticky_target: Option<u32>,
    lock_ticks: u32,
    enemies: &[u32],
    ctx: &TickContext,
) -> Option<u32> {
    if lock_ticks > 0 {
        if let Some(sticky) = sticky_target {
            if enemies.contains(&sticky) {
                return Some(sticky);
            }
        }
    }

    let unit = ctx.unit(state, unit_id)?;
    let focus = board.focus_target;

    enemies.iter().copied().max_by(|a, b| {
        let score_a = target_score(state, unit, personality, *a, focus, ctx);
        let score_b = target_score(state, unit, personality, *b, focus, ctx);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.cmp(a))
    })
}

fn target_score(
    state: &SimState,
    unit: &UnitState,
    personality: &Personality,
    target_id: u32,
    focus: Option<u32>,
    ctx: &TickContext,
) -> f32 {
    let Some(target) = ctx.unit(state, target_id) else {
        return f32::MIN;
    };

    let dist = distance(unit.position, target.position);
    let hp_factor = (target.max_hp - target.hp).max(0) as f32 * 0.2;
    let focus_bonus = if focus == Some(target_id) {
        personality.discipline * 10.0
    } else {
        0.0
    };
    let dist_bias = -dist * (0.5 + (1.0 - personality.aggression) * 1.5);

    // Combo opportunity: if target is CC'd and we have conditional abilities,
    // strongly prefer this target to exploit the CC window
    let combo_bonus = if target.control_remaining_ms > 0 {
        let has_conditional = unit.abilities.iter().any(|slot| {
            slot.cooldown_remaining_ms == 0
                && slot.def.effects.iter().any(|ce| ce.condition.is_some())
        });
        if has_conditional { 15.0 } else { 3.0 } // Even without conditionals, CC'd targets are easier
    } else {
        0.0
    };

    hp_factor + focus_bonus + dist_bias + combo_bonus
}
