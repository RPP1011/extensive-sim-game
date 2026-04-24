//! Resident-path (batch) state — buffers and kernels used by
//! `step_batch()` for GPU-resident execution.

#![cfg(feature = "gpu")]

use crate::cascade_resident::CascadeResidentCtx;
use crate::gpu_util::indirect::IndirectArgsBuffer;
use crate::mask::{FusedAgentUnpackKernel, MaskUnpackKernel};
use crate::scoring::ScoringUnpackKernel;
use crate::view_storage_symmetric_pair::ViewStorageSymmetricPair;

pub struct ResidentPathContext {
    /// Phase D — persistent agent SoA buffer. Allocated on first
    /// step_batch call, reused across ticks. Sync path uses per-kernel
    /// pooled buffers instead.
    pub resident_agents_buf:    Option<wgpu::Buffer>,
    pub resident_agents_cap:    u32,

    /// Phase 3 Task 3.3 — gold ledger side buffer, one i32 per agent slot.
    /// Uploaded from SimState.cold_inventory at ensure_resident_init;
    /// wired into physics via atomic add in Task 3.4; read back into
    /// cold_inventory on snapshot() in Task 3.5.
    pub gold_buf:     Option<wgpu::Buffer>,
    pub gold_buf_cap: u32,

    /// Task #79 — resident storage for the `@symmetric_pair_topk`
    /// `standing` view. Per-agent `[StandingEdge; K=8]` records + per-
    /// owner atomic counts. Uploaded from `state.views.standing` at
    /// `ensure_resident_init` (SP-3); bound into the resident physics
    /// BGL at slots 18 / 19 (SP-4); read back into
    /// `state.views.standing` on `snapshot()` (SP-5).
    pub standing_storage:     Option<ViewStorageSymmetricPair>,
    pub standing_storage_cap: u32,

    /// Phase D — indirect dispatch args for the resident cascade.
    /// MAX_CASCADE_ITERATIONS + 1 slots. Lazy-initialised in step_batch.
    pub resident_indirect_args: Option<IndirectArgsBuffer>,

    /// Phase D — Task D4: resident cascade driver context. Owns
    /// caller-side spatial/ability/physics-ring buffers the
    /// `cascade_resident::run_cascade_resident` driver consumes across
    /// ticks.
    pub resident_cascade_ctx:   Option<CascadeResidentCtx>,

    /// Phase 2 (subsystem 1 follow-up) — GPU-resident sim state buffer.
    /// Holds tick, world seed, world-scalar fields, and cache-invalidation
    /// generation counters. Kernels bind this instead of reading
    /// duplicated fields from per-kernel cfg uniforms. The tick field is
    /// atomically incremented by the seed-indirect kernel at end of each
    /// cascade iteration (Task 2.3).
    ///
    /// Lazy-initialised by `GpuBackend::ensure_resident_init` on first
    /// `step_batch` call.
    pub sim_cfg_buf:            Option<wgpu::Buffer>,

    /// Phase E: retained for backward compatibility but NOT on the hot
    /// batch path — subsumed by [`FusedAgentUnpackKernel`].
    #[allow(dead_code)]
    pub mask_unpack_kernel:     MaskUnpackKernel,

    /// Phase E: retained for backward compatibility but NOT on the hot
    /// batch path — subsumed by [`FusedAgentUnpackKernel`].
    #[allow(dead_code)]
    pub scoring_unpack_kernel:  ScoringUnpackKernel,

    /// Suspect 5 (perf gap): fused mask+scoring unpack kernel. Writes
    /// mask's SoA (pos/alive/creature_type) + scoring's
    /// `agent_data_buf` in a single dispatch, saving one compute pass
    /// begin/end + pipeline set per batch tick.
    pub fused_unpack_kernel:    FusedAgentUnpackKernel,
}

impl ResidentPathContext {
    pub fn new(
        mask_unpack_kernel:    MaskUnpackKernel,
        scoring_unpack_kernel: ScoringUnpackKernel,
        fused_unpack_kernel:   FusedAgentUnpackKernel,
    ) -> Self {
        Self {
            resident_agents_buf:    None,
            resident_agents_cap:    0,
            gold_buf:               None,
            gold_buf_cap:           0,
            standing_storage:       None,
            standing_storage_cap:   0,
            resident_indirect_args: None,
            resident_cascade_ctx:   None,
            mask_unpack_kernel,
            scoring_unpack_kernel,
            fused_unpack_kernel,
            sim_cfg_buf:            None,
        }
    }
}
