// crates/engine/src/scratch.rs
//
// Per-tick scratch buffers — a standalone primitive hoisted out of `step.rs`
// (deleted in Plan B1' Task 11) so the type survives as an engine-layer
// primitive. `SimScratch` holds zero rule-aware logic: it is nothing more than
// pre-allocated buffers that callers reset at the top of each tick.
//
// Rule-aware tick logic (mask-build, action-apply, cascade dispatch) will live
// in `engine_rules::step::step` once Task 11 lands. Until then, callers that
// previously drove `engine::step::step` are `#[ignore]`d (see test files).

use crate::mask::{MaskBuffer, TargetMask};
use crate::policy::Action;

/// Per-tick scratch buffers hoisted out of `step` so a steady-state tick loop
/// allocates zero bytes. Caller constructs once (capacity = `state.agent_cap()`),
/// reuses across ticks. Buffers are reset/cleared at the top of each `step`.
///
/// Moved from `engine::step` (deleted, Plan B1' Task 11) to
/// `engine::scratch` so downstream crates (`engine_gpu`, `viz`,
/// `engine_rules`) can refer to the type without pulling in the
/// (now-deleted) rule-aware step body.
pub struct SimScratch {
    pub mask:        MaskBuffer,
    /// Per-agent per-target-bound-kind candidate list. Task 138 —
    /// populated by the compiler-emitted `mask_<name>_candidates`
    /// enumerators during mask-build and consumed by the scorer.
    pub target_mask: TargetMask,
    pub actions:     Vec<Action>,
    pub shuffle_idx: Vec<u32>,
}

impl SimScratch {
    pub fn new(n_agents: usize) -> Self {
        Self {
            mask:        MaskBuffer::new(n_agents),
            target_mask: TargetMask::new(n_agents),
            actions:     Vec::with_capacity(n_agents),
            shuffle_idx: Vec::with_capacity(n_agents),
        }
    }
}
