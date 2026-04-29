//! GPU backend for the engine — Phase 2 (fused mask kernel).
//!
//! See `docs/plans/gpu_megakernel_plan.md`. Phase 0 shipped a pure
//! CPU-forwarding stub. Phase 1 wired the Attack mask end-to-end via a
//! single-kernel dispatch. Phase 2 generalises: one WGSL module with
//! one `cs_fused_masks` entry point writes every supported mask's
//! bitmap in a single dispatch.
//!
//! ## What runs on GPU in Phase 2
//!
//! Seven of the engine's eight masks lower cleanly to the Phase 2
//! emitter subset and run in the fused kernel:
//!
//!   * Attack / MoveToward (target-bound, radius-filtered)
//!   * Hold / Flee / Eat / Drink / Rest (self-only, alive gate)
//!
//! Cast is skipped — its `(ability: AbilityId)` parametric head and
//! view / cooldown dependencies need the Phase 4+ storage layer. The
//! fused kernel is built from the subset the emitter accepts; the
//! skipped name is documented alongside the emitter and in
//! `engine_gpu::mask`.
//!
//! ## Step semantics
//!
//! As in Phase 1, `GpuBackend::step` still forwards to
//! `engine::step::step` (CPU kernel) and ADDITIONALLY runs the fused
//! mask kernel. The seven GPU-computed bitmaps are not yet spliced
//! into the engine's scratch buffers — they live on the backend and
//! are exposed via [`GpuBackend::last_mask_bitmaps`] for the parity
//! harness to compare against CPU references (one reference per mask,
//! computed by `engine_gpu::mask::cpu_mask_bitmap`).
//!
//! Phase 3 feeds the GPU bitmaps back into the argmax scoring path;
//! that's when the CPU mask-build becomes dead code.
//!
//! ## `#[cfg(feature = "gpu")]` boundary
//!
//! Without the `gpu` feature, `GpuBackend` is the Phase 0 stub — zero
//! wgpu/naga compile cost, `GpuBackend::new()` is infallible, and
//! `step` forwards to CPU. With the feature on, the type grows wgpu
//! handles and `new()` returns a `Result` because device creation can
//! fail on headless CI without a GPU.

use engine::{
    backend::ComputeBackend,
    cascade::CascadeRegistry,
    event::EventRing,
    policy::PolicyBackend,
    state::SimState,
    step::SimScratch,
};
#[cfg(feature = "gpu")]
use engine_data::events::Event;

#[cfg(feature = "gpu")]
pub mod backend;

#[cfg(feature = "gpu")]
pub mod gpu_util;

#[cfg(feature = "gpu")]
pub mod sim_cfg;

/// Phase 4 — per-view GPU storage + fold kernels. Not wired into the
/// backend's tick loop yet; the follow-up integration task swaps
/// scoring's stub views for reads against this module.
#[cfg(feature = "gpu")]
pub mod view_storage;

/// Task #79 — GPU resident-path storage for `@symmetric_pair_topk`
/// views (the `standing` view, Task 3.1). Owns per-agent `[StandingEdge;
/// K=8]` arrays + counts, uploaded from `state.views.standing` on
/// `ensure_resident_init` and read back in `snapshot()`. Bound into the
/// resident physics BGL at slots 18 / 19.
#[cfg(feature = "gpu")]
pub mod view_storage_symmetric_pair;

/// Subsystem 2 Phase 4 — GPU resident-path storage for
/// `@per_entity_ring` views (the `memory` view). Owns per-agent
/// `[MemoryEventGpu; K=64]` ring buffers + monotonic u32 write
/// cursors. Uploaded from the CPU memory ring on
/// `ensure_resident_init`, read back in `snapshot()`. Bound into the
/// resident physics BGL at slots 20 / 21. See
/// `docs/superpowers/plans/2026-04-23-gpu-per-entity-ring-driver-memory.md`.
#[cfg(feature = "gpu")]
pub mod view_storage_per_entity_ring;


/// Phase 6b — GPU event ring primitive. Fixed-capacity device buffer
/// of `EventRecord` slots + atomic tail counter. Kernels emit events
/// via the `gpu_emit_event_*` helpers in `event_ring::EVENT_RING_WGSL`;
/// the host drains back into `engine::event::EventRing` per
/// sub-dispatch. Not yet wired into the backend's tick loop — the
/// physics WGSL emitter (task 187) is the first consumer; the
/// integration task after this plumbs both together.
#[cfg(feature = "gpu")]
pub mod event_ring;


/// Pod helpers (`GpuAgentSlot`, `pack_agent_slots`, alive-bitmap
/// allocator) that survived the T16 hand-written-kernel deletion. Real
/// working code, not stubs — used by `snapshot::GpuSnapshot` decoding
/// and by `ensure_resident_init`'s upload path.
#[cfg(feature = "gpu")]
pub mod sync_helpers;

/// Phase D (task D1) — double-buffered snapshot staging primitives.
/// `GpuStaging` owns one side of an agents+events staging pair;
/// `GpuSnapshot` is the read-only observed state returned to callers.
/// Task D3 consumes these inside `GpuBackend::snapshot()`.
#[cfg(feature = "gpu")]
pub mod snapshot;

/// Perf Stage A — GPU-resident timestamp instrumentation for
/// `step_batch`. Enabled only when the adapter advertises
/// `TIMESTAMP_QUERY` + `TIMESTAMP_QUERY_INSIDE_ENCODERS`; otherwise a
/// cheap no-op so CI on software adapters still works.
#[cfg(feature = "gpu")]
pub mod gpu_profiling;


/// Phase 1 GPU backend.
///
/// With `feature = "gpu"` this owns a `wgpu::Device`/`wgpu::Queue` pair
/// and the compiled Attack-mask compute pipeline. Without the feature
/// it's a zero-sized stub, matching the Phase 0 contract so CPU-only
/// consumers don't pay the wgpu compile cost.
#[cfg(not(feature = "gpu"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuBackend {
    _phase0_marker: (),
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Sync-path state: kernels + view storage + diagnostic fields used
    /// exclusively by `ComputeBackend::step`. See
    /// `backend::sync_ctx::SyncPathContext`.
    pub sync: crate::backend::SyncPathContext,

    /// Resident-path (batch) state: persistent buffers + unpack kernels
    /// used by `step_batch()`. See
    /// `backend::resident_ctx::ResidentPathContext`.
    pub resident: crate::backend::ResidentPathContext,

    /// Snapshot read-back state: double-buffered staging + ring
    /// watermarks consumed by `GpuBackend::snapshot()`. See
    /// `backend::snapshot_ctx::SnapshotContext`.
    pub snapshot: crate::backend::SnapshotContext,
}

/// Phase 9 (task 195): per-tick GPU pipeline phase timings in
/// microseconds. Reset at the top of each `step` and populated as the
/// phase completes. Useful for `perf_n100`-style callers that want to
/// know WHICH submit dominates without cargo-instruments.
///
/// Task 197 repurposed `cpu_phases_1_3_us` — it now measures the GPU
/// mask + scoring dispatch + readback (the CPU mask build + policy
/// evaluate it replaced no longer runs). Kept the old name so existing
/// harnesses don't break; `mask_scoring_us` is the current-accurate
/// alias returned by [`PhaseTimings::mask_scoring_us`].
#[cfg(feature = "gpu")]
#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseTimings {
    /// GPU mask + scoring dispatch (formerly the CPU phases 1-3 cost).
    /// Task 197 replaced the 170-700 ms CPU mask-build at N=1000 with a
    /// ~70-90 ms GPU dispatch + readback; the field name is load-
    /// bearing for existing perf harnesses — use
    /// [`PhaseTimings::mask_scoring_us`] in new code.
    pub cpu_phases_1_3_us: u64,
    /// Task 200 repurposed this field — it now measures the GPU
    /// `apply_actions` + `movement` kernel dispatch + event ring drain
    /// + agent SoA unpack. Kept the old name for harness compatibility;
    /// `apply_us()` is the current-accurate alias.
    pub cpu_apply_actions_us: u64,
    pub gpu_cascade_us: u64,
    pub gpu_seed_fold_us: u64,
    pub cpu_cold_state_us: u64,
    pub cpu_view_fold_all_us: u64,
    pub cpu_finalize_us: u64,
    pub gpu_sidecar_us: u64,
}

#[cfg(feature = "gpu")]
impl PhaseTimings {
    /// Task 197 alias for `cpu_phases_1_3_us`, which now measures the
    /// GPU mask + scoring dispatch (the CPU mask-build+evaluate it
    /// replaced no longer runs). New code should prefer this name.
    pub fn mask_scoring_us(&self) -> u64 {
        self.cpu_phases_1_3_us
    }

    /// Task 200 alias for `cpu_apply_actions_us`, which now measures
    /// the GPU `apply_actions` + `movement` kernel dispatch + drain
    /// (the CPU `apply_actions` it replaced no longer runs on the
    /// hot path). New code should prefer this name.
    pub fn apply_us(&self) -> u64 {
        self.cpu_apply_actions_us
    }
}

/// Initial agent capacity the view storage is provisioned for. Grows on
/// demand to match whatever SimState the first `step` uses. Picked so
/// the common parity test (agent_cap = 8) fits without re-allocation,
/// while small enough that init is cheap on machines that never step
/// through larger worlds.
#[cfg(feature = "gpu")]
const INITIAL_VIEW_AGENT_CAP: u32 = 32;

/// Errors surfaced by `GpuBackend::new`.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub enum GpuInitError {
    /// No compatible GPU adapter available. On a headless box wgpu falls
    /// back to a software backend (LLVMpipe on Linux); this fires only
    /// when even the software backend refuses.
    NoAdapter,
    /// Device request failed — driver bug, feature mismatch, etc.
    RequestDevice(String),
    /// View storage init failed — fold shader compile / buffer
    /// allocation.
    ViewStorage(view_storage::ViewStorageError),
}

#[cfg(feature = "gpu")]
impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuInitError::NoAdapter => write!(f, "no compatible GPU adapter"),
            GpuInitError::RequestDevice(s) => write!(f, "request_device: {s}"),
            GpuInitError::ViewStorage(e) => write!(f, "view_storage init: {e}"),
        }
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for GpuInitError {}

#[cfg(feature = "gpu")]
impl From<view_storage::ViewStorageError> for GpuInitError {
    fn from(e: view_storage::ViewStorageError) -> Self {
        GpuInitError::ViewStorage(e)
    }
}

// -----------------------------------------------------------------------
// Phase 0 stub impl (no `gpu` feature) — retained verbatim so CPU-only
// builds don't require a `Result` return from `new()`.
// -----------------------------------------------------------------------

#[cfg(not(feature = "gpu"))]
impl GpuBackend {
    /// Phase 0 stub constructor — infallible, zero-cost. Matches the
    /// gpu-feature-off case of `GpuBackend::default()`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(not(feature = "gpu"))]
impl ComputeBackend for GpuBackend {
    type Event = engine_data::events::Event;
    type Views = ();

    #[inline]
    fn step<B: PolicyBackend>(
        &mut self,
        state: &mut SimState,
        scratch: &mut SimScratch,
        events: &mut EventRing<Self::Event>,
        _views: &mut Self::Views,
        policy: &B,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
    ) {
        engine::step::step(state, scratch, events, policy, cascade);
    }

    fn reset_mask(&mut self, buf: &mut engine::mask::MaskBuffer) {
        buf.reset(); // Phase 1 stub: CPU pass-through. Plan 5d dispatches GPU clear kernel.
    }

    fn set_mask_bit(&mut self, buf: &mut engine::mask::MaskBuffer, slot: usize, kind: engine::mask::MicroKind) {
        buf.set(slot, kind, true); // Phase 1 stub.
    }

    fn commit_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {
        // Phase 1 stub: no-op (no GPU mirror to flush yet).
    }

    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        views:   &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Phase 5b stub: CPU pass-through. Plan 5e wires GPU cascade kernel.
        cascade.run_fixed_point(state, views, events);
    }

    fn view_fold(
        &mut self,
        _views:         &mut Self::Views,
        _events:        &EventRing<Self::Event>,
        _events_before: usize,
        _tick:          u32,
    ) {
        // Phase 5c stub: no-op. Plan 5e dispatches GPU fold kernels via view_storage.
    }

    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &engine::scratch::SimScratch,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Phase 5d stub: CPU pass-through. Plan 5e dispatches cs_apply_actions + cs_movement.
        engine_rules::step::apply_actions_pub(state, scratch, events);
    }
}

// -----------------------------------------------------------------------
// Phase 1 impl (`gpu` feature) — real wgpu + Attack mask kernel.
// -----------------------------------------------------------------------

/// Test-only helper — spin up a fresh wgpu device + queue pair with
/// the same adapter-selection logic as [`GpuBackend::new`], blocking on
/// the async setup via `pollster::block_on`. Used by standalone unit
/// tests (e.g. `gpu_prefix_scan`) that need a device but don't want to
/// pay for the full backend (mask + scoring + view storage
/// pipelines).
///
/// Returns the same `GpuInitError::NoAdapter` / `::RequestDevice`
/// variants that production init would, so tests can forward the
/// error reason if they care.
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub fn test_device() -> Result<(wgpu::Device, wgpu::Queue), GpuInitError> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuInitError::NoAdapter)?;
        let adapter_limits = adapter.limits();
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine_gpu::test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::RequestDevice(format!("{e}")))
    })
}



#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Spin up a wgpu instance, request an adapter + device, and
    /// allocate the view storage + emitted-kernel infrastructure used
    /// by `step_batch()`. Blocks on the async setup via
    /// `pollster::block_on` so callers stay synchronous — the engine
    /// tick loop is strictly sync.
    ///
    /// Post-T16 (commit `4474566c`) this no longer compiles any
    /// hand-written kernels. Each emitted kernel in
    /// `engine_gpu_rules::*` is lazy-built on first dispatch.
    pub fn new() -> Result<Self, GpuInitError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuInitError::NoAdapter)?;
        let backend_label = format!("{:?}", adapter.get_info().backend);
        let adapter_limits = adapter.limits();

        // Perf Stage A.1 — opt-in timestamp-query feature request.
        let adapter_features = adapter.features();
        let timestamps_supported = crate::gpu_profiling::adapter_supports_timestamps(adapter_features);
        let required_features = if timestamps_supported {
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine_gpu::device"),
                required_features,
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::RequestDevice(format!("{e}")))?;

        let view_storage = view_storage::ViewStorage::new(&device, INITIAL_VIEW_AGENT_CAP)?;
        let sync = crate::backend::SyncPathContext::new(view_storage, backend_label);

        let mut resident = crate::backend::ResidentPathContext::new(
            &device,
            INITIAL_VIEW_AGENT_CAP,
        );
        // Allocate the timestamp profiler up front. Disabled-mode
        // profiler when the adapter lacked the timestamp features;
        // every subsequent `mark` call is a cheap no-op.
        resident.profiler = Some(crate::gpu_profiling::GpuProfiler::new(
            &device,
            &queue,
            timestamps_supported,
        ));

        let snapshot = crate::backend::SnapshotContext::new();

        Ok(Self {
            device,
            queue,
            sync,
            resident,
            snapshot,
        })
    }

    /// Per-tick phase timings from the most recent `step`. Reset at
    /// the top of each step; populated as each phase finishes. Fields
    /// are in microseconds.
    pub fn last_phase_timings(&self) -> PhaseTimings {
        self.sync.last_phase_us
    }

    /// Per-phase GPU µs from the most recent `step_batch` call,
    /// summed across every tick in the batch.
    pub fn last_batch_phase_us(&self) -> &[(&'static str, u64)] {
        &self.resident.last_batch_phase_us
    }

    /// True iff the GPU profiler is live (adapter advertised
    /// `TIMESTAMP_QUERY` + `TIMESTAMP_QUERY_INSIDE_ENCODERS` at init).
    pub fn gpu_profiler_enabled(&self) -> bool {
        self.resident
            .profiler
            .as_ref()
            .map(|p| p.is_enabled())
            .unwrap_or(false)
    }

    /// Borrow the backing view storage. Tests and integration callers
    /// use this to fold events into the view cells before a scoring
    /// dispatch.
    pub fn view_storage(&self) -> &view_storage::ViewStorage {
        &self.sync.view_storage
    }

    /// Mutable borrow of the view storage.
    pub fn view_storage_mut(&mut self) -> &mut view_storage::ViewStorage {
        &mut self.sync.view_storage
    }

    /// The backend's wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The backend's wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Resize the view storage to match the given `agent_cap`,
    /// preserving zeroed state.
    pub fn rebuild_view_storage(
        &mut self,
        agent_cap: u32,
    ) -> Result<(), view_storage::ViewStorageError> {
        self.sync.view_storage = view_storage::ViewStorage::new(&self.device, agent_cap)?;
        Ok(())
    }

    /// Human-readable name of the wgpu backend the device is running on.
    pub fn backend_label(&self) -> &str {
        &self.sync.backend_label
    }

    /// Grow `view_storage` to match the SimState's `agent_cap` if the
    /// current storage is too small. Preserves zero-state on resize.
    fn ensure_view_storage_cap(
        &mut self,
        agent_cap: u32,
    ) -> Result<(), view_storage::ViewStorageError> {
        if self.sync.view_storage.agent_cap() < agent_cap {
            let vs = view_storage::ViewStorage::new(&self.device, agent_cap)?;
            self.sync.view_storage = vs;
        }
        Ok(())
    }

    /// Batched step API. Runs `n_ticks` ticks driven by the
    /// SCHEDULE-loop dispatcher in [`Self::dispatch`]; submits once
    /// per call after recording every tick into a single command
    /// encoder.
    ///
    /// Post-T16 the SCHEDULE-driven path is the sole authoritative
    /// driver. Each kernel in `engine_gpu_rules::*` is currently a
    /// no-op WGSL stub (the bodies were retired alongside the hand-
    /// written kernels), so this method advances `state.tick` on the
    /// CPU side via the supplied `cascade` registry's CPU fallback to
    /// keep the rest of the engine's tick semantics intact while the
    /// emitted shader bodies are filled in.
    ///
    /// ### Fallback
    ///
    /// If resident init fails (GPU allocation error, etc.) the batch
    /// path falls back to calling `engine::step::step` N times so the
    /// tick loop still advances on the CPU.
    pub fn step_batch<B: PolicyBackend>(
        &mut self,
        state: &mut SimState,
        scratch: &mut SimScratch,
        events: &mut EventRing<Event>,
        policy: &B,
        cascade: &CascadeRegistry<Event, ()>,
        n_ticks: u32,
    ) {
        if let Err(e) = self.ensure_resident_init(state) {
            eprintln!(
                "engine_gpu::step_batch: resident init failed ({e}), falling back to CPU loop"
            );
            for _ in 0..n_ticks {
                engine::step::step(state, scratch, events, policy, cascade);
            }
            return;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine_gpu::step_batch::enc"),
            });

        if let Some(p) = self.resident.profiler.as_mut() {
            p.begin_frame();
        }

        for _tick_idx in 0..n_ticks {
            // SCHEDULE-driven dispatch: walk every op the emitter
            // recorded and dispatch its matching kernel arm. Today's
            // emitted WGSL is a no-op stub for every kernel, so the
            // pass advances `state.tick` (via the CPU `engine::step`
            // forward below) but doesn't yet mutate state on the GPU
            // — the dispatch is wiring-only until each kernel's WGSL
            // body is hoisted in a follow-up.
            for op in engine_gpu_rules::schedule::SCHEDULE {
                self.dispatch(op, &mut encoder, state)
                    .expect("emitted schedule dispatch");
            }

            // CPU forward: while the emitted kernel bodies are stubs,
            // keep the engine's authoritative tick state in sync by
            // running the CPU step. Honest fallback — once each WGSL
            // body is filled in, this call will be replaced with a
            // GPU-resident commit.
            engine::step::step(state, scratch, events, policy, cascade);
        }

        if let Some(p) = self.resident.profiler.as_ref() {
            p.finish_frame(&mut encoder);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);

        // Read per-phase µs back from the profiler. Empty when the
        // adapter lacked TIMESTAMP_QUERY at init.
        let phase_samples_opt = self
            .resident
            .profiler
            .as_ref()
            .map(|p| p.read_phase_us(&self.device, &self.queue));
        if let Some(phase_samples) = phase_samples_opt {
            use std::collections::BTreeMap;
            let mut totals: BTreeMap<&'static str, u64> = BTreeMap::new();
            let mut order: Vec<&'static str> = Vec::new();
            for (label, us) in phase_samples {
                if !totals.contains_key(label) {
                    order.push(label);
                }
                *totals.entry(label).or_insert(0) += us;
            }
            self.resident.last_batch_phase_us = order
                .into_iter()
                .map(|l| (l, *totals.get(l).unwrap_or(&0)))
                .collect();
        } else {
            self.resident.last_batch_phase_us.clear();
        }
    }

    /// SCHEDULE-driven kernel dispatcher. One arm per `KernelId` the
    /// emitter references. Each arm lazy-builds its kernel slot on
    /// first call, builds a per-tick `cfg` uniform from `SimState`,
    /// and records into `encoder`.
    ///
    /// ### P10 (no-runtime-panic) note
    /// Every `DispatchOp` variant is matched explicitly. The
    /// `unreachable!()` arms only fire if a future kernel emitter
    /// adds a `KernelId` variant without updating both this match AND
    /// the SCHEDULE — both of which are compile-time. They are not
    /// user-reachable failure modes on the deterministic per-tick
    /// path.
    fn dispatch(
        &mut self,
        op: &engine_gpu_rules::schedule::DispatchOp,
        encoder: &mut wgpu::CommandEncoder,
        state: &SimState,
    ) -> Result<(), ()> {
        use engine_gpu_rules::binding_sources::BindingSources;
        use engine_gpu_rules::external_buffers::ExternalBuffers;
        use engine_gpu_rules::schedule::DispatchOp;
        use engine_gpu_rules::transient_handles::TransientHandles;
        use engine_gpu_rules::{Kernel as _, KernelId};
        use wgpu::util::DeviceExt as _;

        let agent_cap = state.agent_cap();
        let agents_buf = self
            .resident
            .resident_agents_buf
            .as_ref()
            .expect("resident_agents_buf ensured by ensure_resident_init");
        let sim_cfg_ref = self
            .resident
            .sim_cfg_buf
            .as_ref()
            .expect("sim_cfg_buf ensured by ensure_resident_init");

        // Build transient/external aggregates ONCE per dispatch call.
        // Every emitted kernel today is a no-op WGSL stub, so the
        // binding values don't matter for behaviour, only for BGL
        // type-check survival. T16 hoists the real bodies and these
        // placeholders get replaced with the live ring/indirect-args
        // buffers.
        let transient = TransientHandles {
            mask_bitmaps:                sim_cfg_ref,
            mask_unpack_agents_input:    sim_cfg_ref,
            action_buf:                  sim_cfg_ref,
            scoring_unpack_agents_input: sim_cfg_ref,
            cascade_current_ring:        sim_cfg_ref,
            cascade_current_tail:        sim_cfg_ref,
            cascade_next_ring:           sim_cfg_ref,
            cascade_next_tail:           sim_cfg_ref,
            cascade_indirect_args:       sim_cfg_ref,
            fused_agent_unpack_input:    agents_buf,
            fused_agent_unpack_mask_soa: sim_cfg_ref,
            _phantom: std::marker::PhantomData,
        };
        let external = ExternalBuffers {
            agents:           agents_buf,
            sim_cfg:          sim_cfg_ref,
            ability_registry: sim_cfg_ref,
            tag_values:       sim_cfg_ref,
            _phantom:         std::marker::PhantomData,
        };
        let sources = BindingSources {
            resident:  &self.resident.path_ctx,
            pingpong:  &self.resident.pingpong_ctx,
            pool:      &self.resident.pool,
            transient: &transient,
            external:  &external,
        };

        macro_rules! dispatch_kernel {
            ($field:ident, $kernel_ty:path, $label:expr) => {{
                let kernel = self.resident.$field
                    .get_or_insert_with(|| <$kernel_ty>::new(&self.device));
                let cfg = kernel.build_cfg(state);
                let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some($label),
                    contents: bytemuck::cast_slice(&[cfg]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bindings = kernel.bind(&sources, &cfg_buf);
                kernel.record(&self.device, encoder, &bindings, agent_cap);
            }};
        }

        match op {
            DispatchOp::Kernel(KernelId::FusedAgentUnpack) =>
                dispatch_kernel!(fused_agent_unpack_kernel_emitted, engine_gpu_rules::fused_agent_unpack::FusedAgentUnpackKernel, "engine_gpu_rules::fused_agent_unpack::cfg"),
            DispatchOp::Kernel(KernelId::AlivePack) =>
                dispatch_kernel!(alive_pack_kernel_emitted, engine_gpu_rules::alive_pack::AlivePackKernel, "engine_gpu_rules::alive_pack::cfg"),
            DispatchOp::Kernel(KernelId::SpatialHash) =>
                dispatch_kernel!(spatial_hash_kernel, engine_gpu_rules::spatial_hash::SpatialHashKernel, "engine_gpu_rules::spatial_hash::cfg"),
            DispatchOp::Kernel(KernelId::SpatialKinQuery) =>
                dispatch_kernel!(spatial_kin_query_kernel, engine_gpu_rules::spatial_kin_query::SpatialKinQueryKernel, "engine_gpu_rules::spatial_kin_query::cfg"),
            DispatchOp::Kernel(KernelId::SpatialEngagementQuery) =>
                dispatch_kernel!(spatial_engagement_query_kernel, engine_gpu_rules::spatial_engagement_query::SpatialEngagementQueryKernel, "engine_gpu_rules::spatial_engagement_query::cfg"),
            DispatchOp::Kernel(KernelId::FusedMask) =>
                dispatch_kernel!(fused_mask_kernel, engine_gpu_rules::fused_mask::FusedMaskKernel, "engine_gpu_rules::fused_mask::cfg"),
            DispatchOp::Kernel(KernelId::MaskUnpack) =>
                dispatch_kernel!(fused_mask_unpack_kernel, engine_gpu_rules::mask_unpack::MaskUnpackKernel, "engine_gpu_rules::mask_unpack::cfg"),
            DispatchOp::Kernel(KernelId::PickAbility) =>
                dispatch_kernel!(pick_ability_kernel, engine_gpu_rules::pick_ability::PickAbilityKernel, "engine_gpu_rules::pick_ability::cfg"),
            DispatchOp::Kernel(KernelId::ApplyActions) =>
                dispatch_kernel!(apply_actions_kernel, engine_gpu_rules::apply_actions::ApplyActionsKernel, "engine_gpu_rules::apply_actions::cfg"),
            DispatchOp::Kernel(KernelId::Movement) =>
                dispatch_kernel!(movement_kernel, engine_gpu_rules::movement::MovementKernel, "engine_gpu_rules::movement::cfg"),
            DispatchOp::Kernel(KernelId::Scoring) =>
                dispatch_kernel!(scoring_kernel, engine_gpu_rules::scoring::ScoringKernel, "engine_gpu_rules::scoring::cfg"),
            DispatchOp::Kernel(KernelId::ScoringUnpack) =>
                dispatch_kernel!(scoring_unpack_kernel, engine_gpu_rules::scoring_unpack::ScoringUnpackKernel, "engine_gpu_rules::scoring_unpack::cfg"),
            DispatchOp::Kernel(KernelId::FoldEngagedWith) =>
                dispatch_kernel!(fold_engaged_with_kernel, engine_gpu_rules::fold_engaged_with::FoldEngagedWithKernel, "engine_gpu_rules::fold_engaged_with::cfg"),
            DispatchOp::Kernel(KernelId::FoldThreatLevel) =>
                dispatch_kernel!(fold_threat_level_kernel, engine_gpu_rules::fold_threat_level::FoldThreatLevelKernel, "engine_gpu_rules::fold_threat_level::cfg"),
            DispatchOp::Kernel(KernelId::FoldKinFear) =>
                dispatch_kernel!(fold_kin_fear_kernel, engine_gpu_rules::fold_kin_fear::FoldKinFearKernel, "engine_gpu_rules::fold_kin_fear::cfg"),
            DispatchOp::Kernel(KernelId::FoldMyEnemies) =>
                dispatch_kernel!(fold_my_enemies_kernel, engine_gpu_rules::fold_my_enemies::FoldMyEnemiesKernel, "engine_gpu_rules::fold_my_enemies::cfg"),
            DispatchOp::Kernel(KernelId::FoldPackFocus) =>
                dispatch_kernel!(fold_pack_focus_kernel, engine_gpu_rules::fold_pack_focus::FoldPackFocusKernel, "engine_gpu_rules::fold_pack_focus::cfg"),
            DispatchOp::Kernel(KernelId::FoldRallyBoost) =>
                dispatch_kernel!(fold_rally_boost_kernel, engine_gpu_rules::fold_rally_boost::FoldRallyBoostKernel, "engine_gpu_rules::fold_rally_boost::cfg"),
            DispatchOp::Kernel(KernelId::FoldStanding) =>
                dispatch_kernel!(fold_standing_kernel, engine_gpu_rules::fold_standing::FoldStandingKernel, "engine_gpu_rules::fold_standing::cfg"),
            DispatchOp::Kernel(KernelId::FoldMemory) =>
                dispatch_kernel!(fold_memory_kernel, engine_gpu_rules::fold_memory::FoldMemoryKernel, "engine_gpu_rules::fold_memory::cfg"),
            DispatchOp::Kernel(KernelId::AppendEvents) =>
                dispatch_kernel!(append_events_kernel, engine_gpu_rules::append_events::AppendEventsKernel, "engine_gpu_rules::append_events::cfg"),

            DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter } => {
                use engine_gpu_rules::physics::{PhysicsCfg, PhysicsKernel as EmittedPhysicsKernel};

                let kernel = self
                    .resident
                    .physics_kernel
                    .get_or_insert_with(|| EmittedPhysicsKernel::new(&self.device));
                for iter in 0..*max_iter {
                    let transient_iter = TransientHandles {
                        mask_bitmaps:                sim_cfg_ref,
                        mask_unpack_agents_input:    sim_cfg_ref,
                        action_buf:                  sim_cfg_ref,
                        scoring_unpack_agents_input: sim_cfg_ref,
                        cascade_current_ring:        sim_cfg_ref,
                        cascade_current_tail:        sim_cfg_ref,
                        cascade_next_ring:           sim_cfg_ref,
                        cascade_next_tail:           sim_cfg_ref,
                        cascade_indirect_args:       sim_cfg_ref,
                        fused_agent_unpack_input:    agents_buf,
                        fused_agent_unpack_mask_soa: sim_cfg_ref,
                        _phantom: std::marker::PhantomData,
                    };
                    let external_iter = ExternalBuffers {
                        agents:           agents_buf,
                        sim_cfg:          sim_cfg_ref,
                        ability_registry: sim_cfg_ref,
                        tag_values:       sim_cfg_ref,
                        _phantom:         std::marker::PhantomData,
                    };
                    let sources_iter = BindingSources {
                        resident:  &self.resident.path_ctx,
                        pingpong:  &self.resident.pingpong_ctx,
                        pool:      &self.resident.pool,
                        transient: &transient_iter,
                        external:  &external_iter,
                    };
                    let cfg = PhysicsCfg {
                        agent_cap,
                        iter_idx: iter,
                        max_iter: *max_iter,
                        event_ring_capacity: 4096,
                    };
                    let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("engine_gpu_rules::physics::cfg"),
                        contents: bytemuck::cast_slice(&[cfg]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                    let bindings = kernel.bind(&sources_iter, &cfg_buf);
                    kernel.record(&self.device, encoder, &bindings, agent_cap);
                }
            }

            DispatchOp::Indirect { kernel: KernelId::SeedIndirect, args_buf: _ } =>
                dispatch_kernel!(seed_indirect_kernel, engine_gpu_rules::seed_indirect::SeedIndirectKernel, "engine_gpu_rules::seed_indirect::cfg"),

            // No-runtime-panic guarantee (P10): every variant the
            // SCHEDULE actually references is matched explicitly above.
            // Reaching these would be an emitter regression caught at
            // compile-time on the closed `KernelId` enum, well before
            // any binary ships.
            DispatchOp::Kernel(other) => {
                unreachable!("KernelId {other:?} has no dispatch arm; emitter regression")
            }
            DispatchOp::FixedPoint { kernel: other, .. } => {
                unreachable!("FixedPoint {other:?} has no dispatch arm")
            }
            DispatchOp::Indirect { kernel: other, .. } => {
                unreachable!("Indirect {other:?} has no dispatch arm")
            }
            DispatchOp::GatedBy { kernel: other, .. } => {
                unreachable!("GatedBy {other:?} has no dispatch arm")
            }
        }
        Ok(())
    }

    /// Lazy-init for the resident batch path. Allocates the agent
    /// buffer, sim_cfg uniform, gold ledger, view storages, alive
    /// bitmap, and the emitted-kernel `path_ctx` infrastructure.
    /// Idempotent on a stable `agent_cap`.
    fn ensure_resident_init(&mut self, state: &SimState) -> Result<(), String> {
        let agent_cap = state.agent_cap();

        // SimCfg buffer.
        if self.resident.sim_cfg_buf.is_none() {
            let buf = crate::sim_cfg::create_sim_cfg_buffer(&self.device);
            let cfg = crate::sim_cfg::SimCfg::from_state(state);
            crate::sim_cfg::upload_sim_cfg(&self.queue, &buf, &cfg);
            self.resident.sim_cfg_buf = Some(buf);
        }

        // View storage must cover agent_cap.
        self.ensure_view_storage_cap(agent_cap)
            .map_err(|e| format!("ensure_view_storage_cap: {e}"))?;

        // Resident agent buffer. Allocate-or-grow; (re)upload the
        // current SimState agent SoA on (re)allocate.
        let need_alloc = match &self.resident.resident_agents_buf {
            Some(_) => self.resident.resident_agents_cap < agent_cap,
            None => true,
        };
        if need_alloc {
            let bytes = (agent_cap as u64)
                * (std::mem::size_of::<crate::sync_helpers::GpuAgentSlot>() as u64);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine_gpu::resident_agents_buf"),
                size: bytes.max(std::mem::size_of::<crate::sync_helpers::GpuAgentSlot>() as u64),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let packed = crate::sync_helpers::pack_agent_slots(state);
            self.queue
                .write_buffer(&buf, 0, bytemuck::cast_slice(&packed));
            self.resident.resident_agents_buf = Some(buf);
            self.resident.resident_agents_cap = agent_cap;
        }

        // Gold ledger — one i32 per agent slot.
        let need_gold_alloc = match &self.resident.gold_buf {
            Some(_) => self.resident.gold_buf_cap < agent_cap,
            None => true,
        };
        if need_gold_alloc {
            let bytes = (agent_cap as u64) * 4;
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine_gpu::gold_buf"),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let gold_vec: Vec<i32> = (0..agent_cap as usize)
                .map(|slot| {
                    state.cold_inventory()
                        .get(slot)
                        .map(|inv| inv.gold)
                        .unwrap_or(0)
                })
                .collect();
            self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&gold_vec));
            self.resident.gold_buf = Some(buf);
            self.resident.gold_buf_cap = agent_cap;
        }

        // Standing view storage. Bootstrap to a zero-state snapshot —
        // the state.views surface that previously seeded this buffer
        // moved out of `SimState` per Plan B; the SCHEDULE-loop's
        // `FoldStanding` arm fills the storage from its own emitted
        // pass once the WGSL body is hoisted.
        let need_standing_alloc = match &self.resident.standing_storage {
            Some(_) => self.resident.standing_storage_cap < agent_cap,
            None => true,
        };
        if need_standing_alloc {
            let k = crate::view_storage_symmetric_pair::STANDING_K;
            let storage = crate::view_storage_symmetric_pair::ViewStorageSymmetricPair::new(
                &self.device,
                agent_cap,
                k,
            );
            self.resident.standing_storage = Some(storage);
            self.resident.standing_storage_cap = agent_cap;
        }

        // Memory view storage. Same bootstrap pattern as standing.
        let need_memory_alloc = match &self.resident.memory_storage {
            Some(_) => self.resident.memory_storage_cap < agent_cap,
            None => true,
        };
        if need_memory_alloc {
            let k = crate::view_storage_per_entity_ring::MEMORY_K;
            let storage = crate::view_storage_per_entity_ring::ViewStoragePerEntityRing::new(
                &self.device,
                agent_cap,
                k,
            );
            self.resident.memory_storage = Some(storage);
            self.resident.memory_storage_cap = agent_cap;
        }

        // Alive bitmap.
        let need_alive_alloc = match &self.resident.alive_bitmap_buf {
            Some(_) => self.resident.alive_bitmap_cap < agent_cap,
            None => true,
        };
        if need_alive_alloc {
            let buf = crate::sync_helpers::create_alive_bitmap_buffer(&self.device, agent_cap);
            self.resident.alive_bitmap_buf = Some(buf);
            self.resident.alive_bitmap_cap = agent_cap;
        }

        Ok(())
    }

    /// Cheap snapshot of the resident agent SoA + tick. Returns
    /// [`crate::snapshot::GpuSnapshot::empty`] before the first
    /// `step_batch` call (no buffers to read).
    pub fn snapshot(
        &mut self,
        state: &mut SimState,
    ) -> Result<crate::snapshot::GpuSnapshot, crate::snapshot::SnapshotError> {
        let agents_buf = match self.resident.resident_agents_buf.as_ref() {
            Some(b) => b,
            None => return Ok(crate::snapshot::GpuSnapshot::empty()),
        };

        // Lazy-init the staging pair on first call.
        let event_ring_cap = crate::event_ring::DEFAULT_CAPACITY;
        let chronicle_cap_u32 = crate::event_ring::DEFAULT_CHRONICLE_CAPACITY;
        if self.snapshot.snapshot_front.is_none() && self.snapshot.snapshot_back.is_none() {
            let caps = crate::snapshot::StagingCaps {
                agent: self.resident.resident_agents_cap,
                event_ring: event_ring_cap,
                chronicle_ring: chronicle_cap_u32,
            };
            self.snapshot.snapshot_front =
                Some(crate::snapshot::GpuStaging::new(&self.device, caps));
            self.snapshot.snapshot_back =
                Some(crate::snapshot::GpuStaging::new(&self.device, caps));
        }

        let caps = crate::snapshot::StagingCaps {
            agent: self.resident.resident_agents_cap,
            event_ring: event_ring_cap,
            chronicle_ring: chronicle_cap_u32,
        };
        if let Some(front) = self.snapshot.snapshot_front.as_mut() {
            front.ensure_cap(&self.device, caps);
        }
        if let Some(back) = self.snapshot.snapshot_back.as_mut() {
            back.ensure_cap(&self.device, caps);
        }

        let snap = self
            .snapshot
            .snapshot_front
            .as_mut()
            .expect("snapshot_front lazy-inited above")
            .take_snapshot(&self.device, self.resident.resident_agents_cap as usize)?;

        // Read GPU tick from sim_cfg (4 B readback).
        let gpu_tick: u32 = if let Some(sim_cfg_buf) = self.resident.sim_cfg_buf.as_ref() {
            let vec: Vec<u32> = crate::gpu_util::readback::readback_typed::<u32>(
                &self.device,
                &self.queue,
                sim_cfg_buf,
                4,
            )
            .map_err(crate::snapshot::SnapshotError::Ring)?;
            *vec.first().unwrap_or(&0)
        } else {
            0
        };

        // Kick a copy of just the agents buffer into the BACK staging.
        // The event/chronicle rings are observability-only post-T16
        // and not yet wired into the resident path; leave the slices
        // empty (kick_copy with start==end produces a no-op).
        let agent_bytes = (self.resident.resident_agents_cap as u64)
            * (std::mem::size_of::<crate::sync_helpers::GpuAgentSlot>() as u64);
        // Allocate placeholder rings so kick_copy's `&GpuEventRing` /
        // `&GpuChronicleRing` parameters resolve. The slices are 0..0
        // so neither buffer is actually read.
        let placeholder_event_ring = crate::event_ring::GpuEventRing::new(&self.device, event_ring_cap);
        let placeholder_chronicle_ring =
            crate::event_ring::GpuChronicleRing::new(&self.device, chronicle_cap_u32);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine_gpu::snapshot::kick_copy"),
            });
        self.snapshot
            .snapshot_back
            .as_mut()
            .expect("snapshot_back lazy-inited above")
            .kick_copy(
                &mut encoder,
                agents_buf,
                agent_bytes,
                &placeholder_event_ring,
                0,
                0,
                &placeholder_chronicle_ring,
                0,
                0,
                gpu_tick,
            );
        self.queue.submit(Some(encoder.finish()));

        // Gold readback into state.cold_inventory.
        if let Some(gold_buf) = self.resident.gold_buf.as_ref() {
            let cap = self.resident.gold_buf_cap as usize;
            if cap > 0 {
                let bytes = cap * std::mem::size_of::<i32>();
                let gold_vec: Vec<i32> = crate::gpu_util::readback::readback_typed::<i32>(
                    &self.device,
                    &self.queue,
                    gold_buf,
                    bytes,
                )
                .map_err(crate::snapshot::SnapshotError::Ring)?;

                let inv = state.cold_inventory_mut();
                let n = gold_vec.len().min(inv.len());
                for slot in 0..n {
                    inv[slot].gold = gold_vec[slot];
                }
            }
        }

        // Swap front / back.
        std::mem::swap(
            &mut self.snapshot.snapshot_front,
            &mut self.snapshot.snapshot_back,
        );

        Ok(snap)
    }
}

#[cfg(feature = "gpu")]
impl ComputeBackend for GpuBackend {
    type Event = Event;
    type Views = ();

    fn step<B: PolicyBackend>(
        &mut self,
        state:    &mut SimState,
        scratch:  &mut SimScratch,
        events:   &mut EventRing<Self::Event>,
        _views:   &mut Self::Views,
        policy:   &B,
        cascade:  &CascadeRegistry<Self::Event, Self::Views>,
    ) {
        // SCHEDULE-loop entry point. Forwards to step_batch(n_ticks=1)
        // so every per-tick consumer of ComputeBackend::step exercises
        // the emitted dispatcher. step_batch internally CPU-forwards
        // inside its tick body until each emitted WGSL body lands
        // (Stream B); when those bodies replace the CPU forward, this
        // path becomes GPU-authoritative without any further change
        // here.
        self.step_batch(state, scratch, events, policy, cascade, 1);
    }

    fn reset_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {
        // GPU mask is computed on the GPU; CPU MaskBuffer is not the
        // source of truth on the GPU path. No-op.
    }

    fn set_mask_bit(
        &mut self,
        _buf:  &mut engine::mask::MaskBuffer,
        _slot: usize,
        _kind: engine::mask::MicroKind,
    ) {
        // Per-bit CPU writes are not dispatched to GPU.
    }

    fn commit_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {
        // Trait-surface no-op. The emitted FusedMaskKernel is
        // dispatched by step_batch; per-bit CPU writes go nowhere.
    }

    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        views:   &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Mirror `step()`'s honest CPU forward — same rationale.
        cascade.run_fixed_point(state, views, events);
    }

    fn view_fold(
        &mut self,
        _views:         &mut Self::Views,
        _events:        &EventRing<Self::Event>,
        _events_before: usize,
        _tick:          u32,
    ) {
        // Views = () for this backend; the SCHEDULE-loop's Fold<View>
        // arms drive view_storage on the GPU side directly.
    }

    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &engine::scratch::SimScratch,
        events:  &mut EventRing<Self::Event>,
    ) {
        // CPU forward — apply+movement was retired with the hand-
        // written kernels in T16; the emitted ApplyActions+Movement
        // kernels in step_batch are the going-forward path.
        engine_rules::step::apply_actions_pub(state, scratch, events);
    }
}

// Task 193 (Phase 6g) retired the previous Piece 1 `mirror_cpu_views_
// into_gpu_view_storage` bridge. The GPU cascade now folds both seed
// events (apply_actions' pushes) and cascade-emitted events into
// `view_storage` directly inside `step`; the CPU `state.views`
// registry is kept in sync by a parallel CPU `fold_all` call, but
// scoring reads from the GPU-owned `view_storage` atomics without
// needing a cross-device copy.
