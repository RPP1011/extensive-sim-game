//! Resident-path (batch) state — buffers and emitted-kernel slots used
//! by `step_batch()` for GPU-resident execution.
//!
//! Post-T16 every kernel referenced here lives in `engine_gpu_rules::*`.
//! The hand-written kernels and their orchestration (cascade,
//! cascade_resident, apply_actions, movement, mask, scoring,
//! spatial_gpu, alive_bitmap) were deleted in commit `4474566c`; the
//! SCHEDULE-driven dispatcher in `lib.rs::dispatch()` is now the sole
//! per-tick driver.

#![cfg(feature = "gpu")]

use crate::gpu_profiling::GpuProfiler;
use crate::view_storage_per_entity_ring::ViewStoragePerEntityRing;
use crate::view_storage_symmetric_pair::ViewStorageSymmetricPair;

pub struct ResidentPathContext {
    /// Persistent agent SoA buffer. Allocated on first `step_batch`
    /// call; `engine_gpu_rules::external_buffers::ExternalBuffers::agents`
    /// reads it on every dispatch.
    pub resident_agents_buf: Option<wgpu::Buffer>,
    pub resident_agents_cap: u32,

    /// Gold ledger side buffer, one `i32` per agent slot. Populated
    /// from `SimState.cold_inventory()` at resident-init.
    pub gold_buf:     Option<wgpu::Buffer>,
    pub gold_buf_cap: u32,

    /// Resident storage for the `@symmetric_pair_topk` `standing` view.
    pub standing_storage:     Option<ViewStorageSymmetricPair>,
    pub standing_storage_cap: u32,

    /// Resident storage for the `@per_entity_ring(K=64)` `memory` view.
    pub memory_storage:     Option<ViewStoragePerEntityRing>,
    pub memory_storage_cap: u32,

    /// Per-tick alive bitmap — packed `array<u32>` with one bit per
    /// agent slot. Written by `engine_gpu_rules::alive_pack` and read
    /// by mask / scoring / physics kernels.
    pub alive_bitmap_buf: Option<wgpu::Buffer>,
    pub alive_bitmap_cap: u32,

    /// `engine_gpu_rules::external_buffers::ExternalBuffers::sim_cfg`
    /// source. Allocated on first batch entry; tick is mutated on the
    /// GPU side by the SeedIndirect kernel.
    pub sim_cfg_buf: Option<wgpu::Buffer>,

    /// GPU-resident timestamp profiler. May be a disabled-mode profiler
    /// when the adapter lacks `TIMESTAMP_QUERY`.
    pub profiler: Option<GpuProfiler>,

    /// Per-phase µs samples from the most recent `step_batch` call.
    /// Empty when the profiler is disabled or `step_batch` hasn't run.
    pub last_batch_phase_us: Vec<(&'static str, u64)>,

    // Persistent containers shared across emitted kernels. Rebuilt when
    // the resident context is rebuilt (e.g. agent_cap growth).
    pub path_ctx:     engine_gpu_rules::resident_context::ResidentPathContext,
    pub pingpong_ctx: engine_gpu_rules::pingpong_context::PingPongContext,
    pub pool:         engine_gpu_rules::pool::Pool,

    // Lazy-initialised emitted kernels. Each is built on first use via
    // `<KernelTy>::new(&device)` and reused for the rest of the batch.
    // Kernel set is the CG-emitted SCHEDULE — see `engine_gpu_rules::KernelId`.
    pub mask_hold_kernel:                Option<engine_gpu_rules::mask_Hold::MaskHoldKernel>,
    pub mask_move_toward_kernel:         Option<engine_gpu_rules::mask_MoveToward::MaskMoveTowardKernel>,
    pub fused_mask_flee_kernel:          Option<engine_gpu_rules::fused_mask_Flee::FusedMaskFleeKernel>,
    pub fold_engaged_with_kernel:        Option<engine_gpu_rules::fold_engaged_with::FoldEngagedWithKernel>,
    pub fold_threat_level_kernel:        Option<engine_gpu_rules::fold_threat_level::FoldThreatLevelKernel>,
    pub fold_kin_fear_kernel:            Option<engine_gpu_rules::fold_kin_fear::FoldKinFearKernel>,
    pub fold_my_enemies_kernel:          Option<engine_gpu_rules::fold_my_enemies::FoldMyEnemiesKernel>,
    pub fold_pack_focus_kernel:          Option<engine_gpu_rules::fold_pack_focus::FoldPackFocusKernel>,
    pub fold_rally_boost_kernel:         Option<engine_gpu_rules::fold_rally_boost::FoldRallyBoostKernel>,
    pub fold_memory_kernel:              Option<engine_gpu_rules::fold_memory::FoldMemoryKernel>,
    pub fused_spatial_build_hash_kernel: Option<engine_gpu_rules::fused_spatial_build_hash::FusedSpatialBuildHashKernel>,
    pub upload_sim_cfg_kernel:           Option<engine_gpu_rules::upload_sim_cfg::UploadSimCfgKernel>,
    pub pack_agents_kernel:              Option<engine_gpu_rules::pack_agents::PackAgentsKernel>,
    pub seed_indirect_0_kernel:          Option<engine_gpu_rules::seed_indirect_0::SeedIndirect0Kernel>,
    pub unpack_agents_kernel:            Option<engine_gpu_rules::unpack_agents::UnpackAgentsKernel>,
    pub kick_snapshot_kernel:            Option<engine_gpu_rules::kick_snapshot::KickSnapshotKernel>,

    /// Placeholder scratch buffers — one distinct buffer per
    /// `TransientHandles` field. The dispatch path uses these as
    /// stand-ins until the real per-tick scratch allocation lands.
    /// Distinct buffers (rather than a single shared placeholder) are
    /// required because wgpu validates that a single buffer can't be
    /// bound with conflicting STORAGE_READ_ONLY / READ_WRITE usages
    /// in the same dispatch's BindGroup.
    ///
    /// Each is sized to be large enough to satisfy any single-row
    /// BGL min_binding_size check (256 bytes covers our struct array
    /// strides through K=8 SymmetricPairTopK and K=64 PerEntityRing
    /// for a single agent slot).
    pub transient_placeholders: TransientPlaceholders,
}

/// One placeholder buffer per `TransientHandles` field. The real
/// per-tick scratch allocation will replace these once the dispatch
/// path mutates state through dedicated scratch buffers; for now
/// they exist purely so wgpu validation sees distinct resources at
/// each binding slot (vs a single shared placeholder, which triggers
/// "conflicting usages" errors when one slot is read-only and
/// another is read-write).
pub struct TransientPlaceholders {
    pub mask_bitmaps: wgpu::Buffer,
    pub mask_unpack_agents_input: wgpu::Buffer,
    pub action_buf: wgpu::Buffer,
    pub scoring_unpack_agents_input: wgpu::Buffer,
    pub cascade_current_ring: wgpu::Buffer,
    pub cascade_current_tail: wgpu::Buffer,
    pub cascade_next_ring: wgpu::Buffer,
    pub cascade_next_tail: wgpu::Buffer,
    pub cascade_indirect_args: wgpu::Buffer,
    pub fused_agent_unpack_input: wgpu::Buffer,
    pub fused_agent_unpack_mask_soa: wgpu::Buffer,
}

impl TransientPlaceholders {
    pub fn new(device: &wgpu::Device) -> Self {
        let alloc = |label: &'static str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            })
        };
        Self {
            mask_bitmaps: alloc("transient_placeholder::mask_bitmaps"),
            mask_unpack_agents_input: alloc("transient_placeholder::mask_unpack_agents_input"),
            action_buf: alloc("transient_placeholder::action_buf"),
            scoring_unpack_agents_input: alloc(
                "transient_placeholder::scoring_unpack_agents_input",
            ),
            cascade_current_ring: alloc("transient_placeholder::cascade_current_ring"),
            cascade_current_tail: alloc("transient_placeholder::cascade_current_tail"),
            cascade_next_ring: alloc("transient_placeholder::cascade_next_ring"),
            cascade_next_tail: alloc("transient_placeholder::cascade_next_tail"),
            cascade_indirect_args: alloc("transient_placeholder::cascade_indirect_args"),
            fused_agent_unpack_input: alloc("transient_placeholder::fused_agent_unpack_input"),
            fused_agent_unpack_mask_soa: alloc(
                "transient_placeholder::fused_agent_unpack_mask_soa",
            ),
        }
    }
}

impl ResidentPathContext {
    pub fn new(device: &wgpu::Device, agent_cap: u32) -> Self {
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
            sim_cfg_buf:            None,
            profiler:               None,
            last_batch_phase_us:    Vec::new(),
            path_ctx,
            pingpong_ctx,
            pool,
            mask_hold_kernel:                None,
            mask_move_toward_kernel:         None,
            fused_mask_flee_kernel:          None,
            fold_engaged_with_kernel:        None,
            fold_threat_level_kernel:        None,
            fold_kin_fear_kernel:            None,
            fold_my_enemies_kernel:          None,
            fold_pack_focus_kernel:          None,
            fold_rally_boost_kernel:         None,
            fold_memory_kernel:              None,
            fused_spatial_build_hash_kernel: None,
            upload_sim_cfg_kernel:           None,
            pack_agents_kernel:              None,
            seed_indirect_0_kernel:          None,
            unpack_agents_kernel:            None,
            kick_snapshot_kernel:            None,
            transient_placeholders: TransientPlaceholders::new(device),
        }
    }
}
