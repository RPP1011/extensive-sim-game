//! GOAP goal definitions with dynamic insistence scoring.

use super::world_state::{PropValue, WorldState};

/// Comparison operator for preconditions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompOp {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

impl CompOp {
    pub fn evaluate(self, lhs: f32, rhs: f32) -> bool {
        match self {
            CompOp::Eq => (lhs - rhs).abs() < f32::EPSILON,
            CompOp::Neq => (lhs - rhs).abs() >= f32::EPSILON,
            CompOp::Lt => lhs < rhs,
            CompOp::Gt => lhs > rhs,
            CompOp::Lte => lhs <= rhs,
            CompOp::Gte => lhs >= rhs,
        }
    }
}

/// A precondition on a world state property.
#[derive(Debug, Clone)]
pub struct Precondition {
    pub op: CompOp,
    pub value: PropValue,
}

impl Precondition {
    pub fn is_satisfied(&self, actual: PropValue) -> bool {
        match (&self.value, &actual) {
            (PropValue::Bool(expected), _) => {
                let a = actual.as_bool();
                match self.op {
                    CompOp::Eq => a == *expected,
                    CompOp::Neq => a != *expected,
                    _ => a == *expected, // for bool, < > don't make sense — treat as ==
                }
            }
            (PropValue::Float(expected), _) => {
                self.op.evaluate(actual.as_float(), *expected)
            }
            (PropValue::Id(expected), PropValue::Id(actual_id)) => {
                match self.op {
                    CompOp::Eq => actual_id == expected,
                    CompOp::Neq => actual_id != expected,
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

/// Dynamic insistence function — determines goal priority.
#[derive(Debug, Clone)]
pub enum InsistenceFn {
    /// Constant priority.
    Fixed(f32),
    /// Linear function of a property: `scale * prop_value + offset`.
    Linear { prop: usize, scale: f32, offset: f32 },
    /// Step function: if prop satisfies condition, return `value`, else 0.
    Threshold { prop: usize, op: CompOp, threshold: f32, value: f32 },
}

impl InsistenceFn {
    pub fn evaluate(&self, ws: &WorldState) -> f32 {
        match self {
            InsistenceFn::Fixed(v) => *v,
            InsistenceFn::Linear { prop, scale, offset } => {
                let val = ws.get(*prop).as_float();
                (scale * val + offset).max(0.0)
            }
            InsistenceFn::Threshold { prop, op, threshold, value } => {
                let val = ws.get(*prop).as_float();
                if op.evaluate(val, *threshold) { *value } else { 0.0 }
            }
        }
    }
}

/// A goal that drives GOAP planning.
#[derive(Debug, Clone)]
pub struct Goal {
    pub name: String,
    /// Desired world state conditions — goal is satisfied when all hold.
    pub desired: Vec<(usize, Precondition)>,
    /// Dynamic priority scoring.
    pub insistence: InsistenceFn,
}

impl Goal {
    /// Check if all desired conditions are met in current world state.
    pub fn is_satisfied(&self, ws: &WorldState) -> bool {
        self.desired.iter().all(|(prop, pre)| pre.is_satisfied(ws.get(*prop)))
    }

    /// Score this goal's priority given current world state.
    pub fn score(&self, ws: &WorldState) -> f32 {
        self.insistence.evaluate(ws)
    }
}

/// Select the highest-insistence unsatisfied goal.
/// Hysteresis: new goal must exceed current by `hysteresis` margin.
pub fn select_goal<'a>(
    goals: &'a [Goal],
    ws: &WorldState,
    current_goal: Option<&str>,
    hysteresis: f32,
) -> Option<&'a Goal> {
    let current_score = current_goal
        .and_then(|name| goals.iter().find(|g| g.name == name))
        .map(|g| g.score(ws))
        .unwrap_or(0.0);

    let mut best: Option<&Goal> = None;
    let mut best_score = 0.0_f32;

    for goal in goals {
        if goal.is_satisfied(ws) {
            continue;
        }
        let score = goal.score(ws);
        if score > best_score {
            best_score = score;
            best = Some(goal);
        }
    }

    let best = best?;

    // Hysteresis: if we already have a goal, the new one must beat it by margin
    if let Some(current_name) = current_goal {
        if best.name != current_name && best_score < current_score + hysteresis {
            // Keep current goal if it's still unsatisfied
            return goals
                .iter()
                .find(|g| g.name == current_name && !g.is_satisfied(ws));
        }
    }

    Some(best)
}
