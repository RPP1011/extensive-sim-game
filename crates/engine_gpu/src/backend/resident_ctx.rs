//! Resident-path (batch) state — buffers and kernels used by
//! `step_batch()` for GPU-resident execution.

#![cfg(feature = "gpu")]

use crate::alive_bitmap::AlivePackKernel;
use crate::cascade_resident::CascadeResidentCtx;
use crate::gpu_profiling::GpuProfiler;
use crate::gpu_util::indirect::IndirectArgsBuffer;
use crate::mask::{FusedAgentUnpackKernel, MaskUnpackKernel};
use crate::scoring::ScoringUnpackKernel;
use crate::view_storage_per_entity_ring::ViewStoragePerEntityRing;
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

    /// Subsystem 2 Phase 4 — resident storage for the
    /// `@per_entity_ring(K=64)` `memory` view. Per-agent
    /// `[MemoryEventGpu; K=64]` rings + per-owner monotonic u32
    /// cursors. Uploaded from the CPU memory ring at
    /// `ensure_resident_init` (PR-3); bound into the resident physics
    /// BGL at slots 20 / 21 (PR-4); read back on `snapshot()` (PR-6).
    pub memory_storage:     Option<ViewStoragePerEntityRing>,
    pub memory_storage_cap: u32,

    /// Per-tick alive bitmap — packed `array<u32>` with one bit per
    /// agent slot. Written once at the top of each batch tick by the
    /// pack kernel; read by mask / scoring / physics kernels at
    /// binding slot 22. Reduces the per-tick `agents[x].alive`
    /// cacheline reads from O(agents × alive-calls × 64 B) to
    /// O(agents × alive-calls × 4 B bit lookups against an L1-resident
    /// buffer).
    pub alive_bitmap_buf:   Option<wgpu::Buffer>,
    pub alive_bitmap_cap:   u32,

    /// Pack kernel that reads `agents[i].alive` and packs one bit per
    /// slot into `alive_bitmap_buf`. Dispatched at the TOP of every
    /// batch tick — after the fused unpack wrote SoA fields but before
    /// mask / scoring / physics run.
    pub alive_pack_kernel:  AlivePackKernel,

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

    /// Perf Stage A.1 — GPU-resident timestamp profiler. `None` until
    /// the first `step_batch` call wires one up via
    /// `ensure_resident_init`; may be a disabled-mode profiler if the
    /// adapter lacks `TIMESTAMP_QUERY`. Read back per-phase µs via
    /// `GpuBackend::last_batch_phase_us`.
    pub profiler: Option<GpuProfiler>,

    /// Per-phase µs samples from the most recent `step_batch` call.
    /// Accumulated across all N ticks in the batch (entry `i` is the
    /// sum of the i-th phase-pair across every tick); divide by the
    /// batch's tick count in the test harness to print a mean.
    /// Empty when the profiler is disabled or `step_batch` hasn't run.
    pub last_batch_phase_us: Vec<(&'static str, u64)>,

    // -- T5: emitted-kernel resident state ----------------------------
    //
    // The plan walks the migration of GPU dispatch from hand-written
    // wrappers in `engine_gpu::*` to emitted kernels under
    // `engine_gpu_rules::*`. T5 lights up the first emitted kernel
    // (FusedMaskKernel + MaskUnpackKernel); the persistent containers
    // and lazy-init kernel slots below are the resident-side anchors
    // for the bind/record dispatch pattern. Later tasks (T6+) grow this
    // section as more kernels migrate.
    /// Persistent buffers shared across kernels for the whole batch;
    /// rebuilt when the resident context is rebuilt (e.g. agent_cap
    /// growth).
    pub path_ctx:     engine_gpu_rules::resident_context::ResidentPathContext,
    /// PingPong A/B ring containers (cascade physics).
    pub pingpong_ctx: engine_gpu_rules::pingpong_context::PingPongContext,
    /// Shape-keyed pooled buffers reused across kernels.
    pub pool:         engine_gpu_rules::pool::Pool,
    /// Lazy-initialised emitted FusedMaskKernel — built on first
    /// `step_batch` call, reused across ticks until the resident
    /// context is rebuilt.
    pub fused_mask_kernel: Option<engine_gpu_rules::fused_mask::FusedMaskKernel>,
    /// Lazy-initialised emitted MaskUnpackKernel — built on first
    /// `step_batch` call, reused across ticks. T5 wiring; the
    /// hand-written `mask_unpack_kernel` field above is kept for
    /// backward compatibility until T16 retires it.
    pub fused_mask_unpack_kernel: Option<engine_gpu_rules::mask_unpack::MaskUnpackKernel>,
    /// T7 — lazy-initialised emitted ApplyActionsKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_apply_actions_dispatch` feature is enabled;
    /// remains `None` (and the slot is just type-checked) when the
    /// feature is off. The hand-written
    /// `cascade_ctx.apply_actions.run_resident(...)` path is the only
    /// dispatch that runs by default until T16 hoists the real
    /// damage/heal/event semantics into the emitted WGSL body.
    pub apply_actions_kernel: Option<engine_gpu_rules::apply_actions::ApplyActionsKernel>,
    /// T8 — lazy-initialised emitted PickAbilityKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_pick_ability_dispatch` feature is enabled;
    /// remains `None` (and the slot is just type-checked) when the
    /// feature is off. There is no hand-written pick_ability kernel
    /// to fall back to (Subsystem 3 Group B folded into this plan);
    /// the emitted dispatch is the only version. Stays off by default
    /// until a `per_ability` row exists in the resolved scoring IR
    /// (the bootstrap currently emits a stub WGSL).
    pub pick_ability_kernel: Option<engine_gpu_rules::pick_ability::PickAbilityKernel>,
    /// T9 — lazy-initialised emitted MovementKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_movement_dispatch` feature is enabled;
    /// remains `None` (and the slot is just type-checked) when the
    /// feature is off. The hand-written
    /// `cascade_ctx.movement.run_resident(...)` path is the only
    /// dispatch that runs by default until T16 hoists the real
    /// MoveToward / Flee semantics into the emitted WGSL body.
    pub movement_kernel: Option<engine_gpu_rules::movement::MovementKernel>,
    /// T10 — lazy-initialised emitted PhysicsKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_physics_dispatch` feature is enabled;
    /// remains `None` (slot type-checked only) when the feature is off.
    /// The hand-written
    /// `cascade_resident::run_cascade_resident_with_iter_cap` is the
    /// only dispatch that runs by default until T16 retires it.
    pub physics_kernel: Option<engine_gpu_rules::physics::PhysicsKernel>,
    /// T10 — lazy-initialised emitted SeedIndirectKernel.
    pub seed_indirect_kernel: Option<engine_gpu_rules::seed_indirect::SeedIndirectKernel>,
    /// T10 — lazy-initialised emitted AppendEventsKernel.
    pub append_events_kernel: Option<engine_gpu_rules::append_events::AppendEventsKernel>,
    /// T11 — lazy-initialised emitted per-view Fold kernels. One slot
    /// per materialized view in the IR (engaged_with / threat_level /
    /// kin_fear / my_enemies / pack_focus / rally_boost / standing /
    /// memory). All built on first `step_batch` call when the
    /// `engine_gpu_emitted_view_folds_dispatch` feature is enabled;
    /// remain `None` (slot type-checked only) when off. The
    /// hand-written `engine_gpu::view_storage` fold dispatch is the
    /// only version that runs by default until T16 hoists the real
    /// fold-body WGSL into the emitted kernels.
    pub fold_engaged_with_kernel: Option<engine_gpu_rules::fold_engaged_with::FoldEngagedWithKernel>,
    pub fold_threat_level_kernel: Option<engine_gpu_rules::fold_threat_level::FoldThreatLevelKernel>,
    pub fold_kin_fear_kernel:     Option<engine_gpu_rules::fold_kin_fear::FoldKinFearKernel>,
    pub fold_my_enemies_kernel:   Option<engine_gpu_rules::fold_my_enemies::FoldMyEnemiesKernel>,
    pub fold_pack_focus_kernel:   Option<engine_gpu_rules::fold_pack_focus::FoldPackFocusKernel>,
    pub fold_rally_boost_kernel:  Option<engine_gpu_rules::fold_rally_boost::FoldRallyBoostKernel>,
    pub fold_standing_kernel:     Option<engine_gpu_rules::fold_standing::FoldStandingKernel>,
    pub fold_memory_kernel:       Option<engine_gpu_rules::fold_memory::FoldMemoryKernel>,
    /// T12 — lazy-initialised emitted SpatialHashKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_spatial_dispatch` feature is enabled;
    /// remains `None` (slot type-checked only) when off. The
    /// hand-written `cascade_resident::run_spatial_resident_pre_scoring`
    /// is the only dispatch that runs by default until T16 retires it.
    pub spatial_hash_kernel: Option<engine_gpu_rules::spatial_hash::SpatialHashKernel>,
    /// T12 — lazy-initialised emitted SpatialKinQueryKernel.
    pub spatial_kin_query_kernel: Option<engine_gpu_rules::spatial_kin_query::SpatialKinQueryKernel>,
    /// T12 — lazy-initialised emitted SpatialEngagementQueryKernel.
    pub spatial_engagement_query_kernel: Option<engine_gpu_rules::spatial_engagement_query::SpatialEngagementQueryKernel>,
    /// T13 — lazy-initialised emitted AlivePackKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_alive_pack_dispatch` feature is enabled;
    /// remains `None` (slot type-checked only) when off. The
    /// hand-written `alive_pack_kernel.encode_pack` is the only
    /// dispatch that runs by default until T16 retires it.
    pub alive_pack_kernel_emitted: Option<engine_gpu_rules::alive_pack::AlivePackKernel>,
    /// T13 — lazy-initialised emitted FusedAgentUnpackKernel from
    /// `engine_gpu_rules`. Built on first `step_batch` call when the
    /// `engine_gpu_emitted_fused_unpack_dispatch` feature is enabled;
    /// remains `None` (slot type-checked only) when off. The
    /// hand-written `fused_unpack_kernel.encode_unpack` is the only
    /// dispatch that runs by default until T16 retires it.
    pub fused_agent_unpack_kernel_emitted: Option<engine_gpu_rules::fused_agent_unpack::FusedAgentUnpackKernel>,
}

impl ResidentPathContext {
    pub fn new(
        device:                &wgpu::Device,
        agent_cap:             u32,
        mask_unpack_kernel:    MaskUnpackKernel,
        scoring_unpack_kernel: ScoringUnpackKernel,
        fused_unpack_kernel:   FusedAgentUnpackKernel,
        alive_pack_kernel:     AlivePackKernel,
    ) -> Self {
        let path_ctx = engine_gpu_rules::resident_context::ResidentPathContext::new(device, agent_cap);
        let pingpong_ctx = engine_gpu_rules::pingpong_context::PingPongContext::new(device);
        let pool = engine_gpu_rules::pool::Pool::new(device);
        Self {
            resident_agents_buf:    None,
            resident_agents_cap:    0,
            gold_buf:               None,
            gold_buf_cap:           0,
            standing_storage:       None,
            standing_storage_cap:   0,
            memory_storage:         None,
            memory_storage_cap:     0,
            alive_bitmap_buf:       None,
            alive_bitmap_cap:       0,
            alive_pack_kernel,
            resident_indirect_args: None,
            resident_cascade_ctx:   None,
            mask_unpack_kernel,
            scoring_unpack_kernel,
            fused_unpack_kernel,
            sim_cfg_buf:            None,
            profiler:               None,
            last_batch_phase_us:    Vec::new(),
            path_ctx,
            pingpong_ctx,
            pool,
            fused_mask_kernel:      None,
            fused_mask_unpack_kernel: None,
            apply_actions_kernel:   None,
            pick_ability_kernel:    None,
            movement_kernel:        None,
            physics_kernel:         None,
            seed_indirect_kernel:   None,
            append_events_kernel:   None,
            fold_engaged_with_kernel: None,
            fold_threat_level_kernel: None,
            fold_kin_fear_kernel:     None,
            fold_my_enemies_kernel:   None,
            fold_pack_focus_kernel:   None,
            fold_rally_boost_kernel:  None,
            fold_standing_kernel:     None,
            fold_memory_kernel:       None,
            spatial_hash_kernel:               None,
            spatial_kin_query_kernel:          None,
            spatial_engagement_query_kernel:   None,
            alive_pack_kernel_emitted:         None,
            fused_agent_unpack_kernel_emitted: None,
        }
    }
}
