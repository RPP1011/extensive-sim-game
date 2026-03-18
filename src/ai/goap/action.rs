//! GOAP action definitions — preconditions, effects, and intent mapping.

use super::target::Target;
use super::goal::Precondition;
use super::world_state::PropValue;

/// Template for mapping a GOAP action to an IntentAction at execution time.
#[derive(Debug, Clone)]
pub enum IntentTemplate {
    AttackTarget(Target),
    ChaseTarget(Target),
    FleeTarget(Target),
    MaintainDistance(Target, f32),
    CastIfReady(usize, Target), // ability_slot, target
    Hold,
}

/// A GOAP action with preconditions, effects, cost, and intent mapping.
#[derive(Debug, Clone)]
pub struct GoapAction {
    pub name: String,
    pub cost: f32,
    pub preconditions: Vec<(usize, Precondition)>,
    pub effects: Vec<(usize, PropValue)>,
    pub intent: IntentTemplate,
    pub duration_ticks: u32,
}

impl GoapAction {
    /// Check if all preconditions are met in the given world state.
    pub fn preconditions_met(&self, ws: &super::world_state::WorldState) -> bool {
        self.preconditions.iter().all(|(prop, pre)| pre.is_satisfied(ws.get(*prop)))
    }

    /// Apply this action's effects to a world state (for planning, not execution).
    pub fn apply_effects(&self, ws: &mut super::world_state::WorldState) {
        for (prop, val) in &self.effects {
            ws.set(*prop, *val);
        }
    }

    /// Check if any of this action's effects overlap with the given property indices.
    pub fn affects_any(&self, props: &[usize]) -> bool {
        self.effects.iter().any(|(p, _)| props.contains(p))
    }
}
