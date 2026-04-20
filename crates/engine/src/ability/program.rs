//! `AbilityProgram` IR — stub for Combat Foundation Task 7.
//!
//! Task 6 introduces only the `AbilityId` newtype; the program IR, effect
//! ops, delivery / area / gate types land in Task 7. The placeholder types
//! below exist so `ability/mod.rs` can re-export them without breaking the
//! build. They will be replaced with the real definitions in the next task.

use crate::ids::AgentId as _AgentId;
use smallvec::SmallVec;

/// Maximum number of effects a single `AbilityProgram` may carry. Bounded so
/// cast-dispatch emission fits in a fixed-size smallvec on the stack. Task 7
/// pins this in the size-budget test; the value is final (changing it is a
/// schema-hash bump).
pub const MAX_EFFECTS_PER_PROGRAM: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Delivery { Instant }

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Area { SingleTarget { range: f32 } }

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Gate {
    pub cooldown_ticks: u32,
    pub hostile_only:   bool,
    pub line_of_sight:  bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TargetSelector { Target, Caster }

/// Effect-operation enum — stub. Task 7 replaces with the 8-variant real enum.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EffectOp {
    /// Placeholder variant; not used at runtime until Task 7 lands.
    Noop,
}

#[derive(Clone, Debug)]
pub struct AbilityProgram {
    pub delivery: Delivery,
    pub area:     Area,
    pub gate:     Gate,
    pub effects:  SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]>,
}

// Silence the unused-import lint in Task-6 scaffold; Task 7 references AgentId
// in effect payloads via `crate::ids`.
#[allow(dead_code)]
fn _scaffold_marker(_id: _AgentId) {}
