// crates/engine/src/scratch.rs
//
// Per-tick scratch buffers — a standalone storage primitive. `SimScratch`
// holds zero rule-aware logic: it is nothing more than pre-allocated buffers
// that callers reset at the top of each tick.
//
// Rule-aware tick logic (mask-build, action-apply, cascade dispatch) lives in
// `engine_rules::step::step`.

use crate::ids::AgentId;
use crate::mask::{MaskBuffer, TargetMask};
use crate::policy::Action;

/// Per-tick scratch buffers for a steady-state tick loop that allocates zero
/// bytes per tick. Caller constructs once (capacity = `state.agent_cap()`),
/// reuses across ticks. Buffers are reset/cleared at the top of each `step`.
pub struct SimScratch {
    pub mask:        MaskBuffer,
    /// Per-agent per-target-bound-kind candidate list. Task 138 —
    /// populated by the compiler-emitted `mask_<name>_candidates`
    /// enumerators during mask-build and consumed by the scorer.
    pub target_mask: TargetMask,
    pub actions:     Vec<Action>,
    pub shuffle_idx: Vec<u32>,
    /// Reusable scratch buffer for spatial neighbor queries. Callers pass
    /// `&mut scratch.neighbors_scratch` to `SpatialHash::within_radius_into` /
    /// `within_planar_into` to avoid a per-call heap allocation.
    pub neighbors_scratch: Vec<AgentId>,
}

impl SimScratch {
    pub fn new(n_agents: usize) -> Self {
        Self {
            mask:              MaskBuffer::new(n_agents),
            target_mask:       TargetMask::new(n_agents),
            actions:           Vec::with_capacity(n_agents),
            shuffle_idx:       Vec::with_capacity(n_agents),
            neighbors_scratch: Vec::with_capacity(n_agents),
        }
    }
}
