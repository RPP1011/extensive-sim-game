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
    backend::SimBackend,
    cascade::CascadeRegistry,
    event::EventRing,
    policy::PolicyBackend,
    state::SimState,
    step::SimScratch,
};
#[cfg(feature = "gpu")]
use engine::event::Event;

#[cfg(feature = "gpu")]
pub mod backend;

#[cfg(feature = "gpu")]
pub mod gpu_util;

#[cfg(feature = "gpu")]
pub mod mask;

/// Phase 3 — scoring kernel + deterministic argmax. Consumes fused
/// mask bitmaps and produces one `ScoreOutput` per agent. View calls
/// currently stubbed to 0.0 — follow-up integration task wires them
/// to `view_storage`.
#[cfg(feature = "gpu")]
pub mod scoring;

/// Phase 4 — per-view GPU storage + fold kernels. Not wired into the
/// backend's tick loop yet; the follow-up integration task swaps
/// scoring's stub views for reads against this module.
#[cfg(feature = "gpu")]
pub mod view_storage;

/// Phase 5 — GPU spatial hash + nearest-hostile / nearby-kin queries.
/// Not yet consumed by the backend's step loop; the scoring / physics
/// kernels will call into `spatial_gpu::SPATIAL_WGSL` helpers.
#[cfg(feature = "gpu")]
pub mod spatial_gpu;

/// Phase 6b — GPU event ring primitive. Fixed-capacity device buffer
/// of `EventRecord` slots + atomic tail counter. Kernels emit events
/// via the `gpu_emit_event_*` helpers in `event_ring::EVENT_RING_WGSL`;
/// the host drains back into `engine::event::EventRing` per
/// sub-dispatch. Not yet wired into the backend's tick loop — the
/// physics WGSL emitter (task 187) is the first consumer; the
/// integration task after this plumbs both together.
#[cfg(feature = "gpu")]
pub mod event_ring;

/// Phase 6e — GPU physics kernel (Piece 2 of the cascade megakernel).
/// Drives task 187's emitted WGSL against the event ring + agent SoA.
/// Processes one cascade iteration per `run_batch` call — the cascade
/// loop (Piece 3) drives re-dispatches until the ring stops growing.
#[cfg(feature = "gpu")]
pub mod physics;

/// Phase 6f — cascade sub-dispatch loop (Piece 3). Drives `physics::run_batch`
/// in a fixed-point loop, folds each iteration's emitted events into
/// `view_storage`, and returns the aggregated output. Not yet authoritative
/// inside `GpuBackend::step` — Piece 4 replaces the CPU forward with a
/// cascade call.
#[cfg(feature = "gpu")]
pub mod cascade;

/// Task 199 — GPU `apply_actions` kernel. WGSL port of the hot subset
/// of `engine::step::apply_actions`: Attack damage + AgentAttacked /
/// AgentDied event emission, one thread per agent slot. Needs /
/// opportunity attacks / engagement-slow / announce are documented as
/// out-of-scope in the module header. Not yet wired into
/// `GpuBackend::step` as authoritative — scaffolding for a follow-up
/// that removes the CPU `apply_actions` call entirely once the
/// engagement-slow + opportunity-attack gaps close.
#[cfg(feature = "gpu")]
pub mod apply_actions;

/// Task 199 — GPU movement kernel. WGSL port of the MoveToward / Flee
/// position updates, one thread per agent slot. Pure-away flee only
/// (no kin-flee-bias) in the initial landing — deer herding is a
/// follow-up. Not yet wired into `GpuBackend::step` as authoritative.
#[cfg(feature = "gpu")]
pub mod movement;

/// Phase C (task C1) — GPU-resident cascade driver. Composes Phase B's
/// resident kernels into one indirect-dispatch sequence per tick with
/// no Rust-side convergence check and no per-iter readback. Scaffolding
/// for Phase D's `step_batch(n)`; not yet consumed by `GpuBackend::step`
/// which still drives the sync cascade in `cascade.rs`.
#[cfg(feature = "gpu")]
pub mod cascade_resident;

/// Phase D (task D1) — double-buffered snapshot staging primitives.
/// `GpuStaging` owns one side of an agents+events staging pair;
/// `GpuSnapshot` is the read-only observed state returned to callers.
/// Task D3 consumes these inside `GpuBackend::snapshot()`.
#[cfg(feature = "gpu")]
pub mod snapshot;

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
    mask_kernel: mask::FusedMaskKernel,
    /// Phase 3 scoring kernel. Consumes the fused mask bitmaps + packed
    /// agent SoA and produces one `ScoreOutput` per agent slot.
    scoring_kernel: scoring::ScoringKernel,
    /// Phase 6d: GPU-resident view storage. Scoring reads from these
    /// buffers directly; fold kernels (driven by the integration loop
    /// below) write them in response to events. Initial `agent_cap`
    /// defaults to `INITIAL_VIEW_AGENT_CAP`; it grows on demand when a
    /// larger `SimState` is stepped through (see `ensure_view_storage`).
    view_storage: view_storage::ViewStorage,
    /// Phase 6g (task 193): lazy-initialised cascade context. Holds the
    /// physics kernel + spatial hash + packed ability registry + the
    /// DSL `Compilation` the physics kernel was compiled against.
    /// Construction reads `assets/sim/*.sim` + compiles a WGSL module,
    /// which is ~10-30 ms on a warm disk; done on first `step` to keep
    /// `GpuBackend::new` cheap for callers that never actually step
    /// (scoring parity tests, for example).
    cascade_ctx: Option<cascade::CascadeCtx>,
    /// Name of the backend wgpu selected (Vulkan / Metal / DX12 / GL /
    /// BrowserWebGpu / llvmpipe (GL/Vulkan software)). Captured at
    /// construction so tests can report which path they exercised.
    backend_label: String,
    /// Per-mask bitmaps from the most recent `step`, in the emitter's
    /// binding order (see `FusedMaskKernel::bindings`). Each inner
    /// `Vec<u32>` is a packed bitmap: bit `i` of word `i/32` is set
    /// iff slot `i`'s agent passed that mask's predicate this tick.
    /// Empty before the first `step`.
    last_mask_bitmaps: Vec<Vec<u32>>,
    /// Per-agent scoring outputs from the most recent `step` — one
    /// `ScoreOutput` per agent slot carrying the GPU's argmax choice
    /// (chosen_action, chosen_target, best_score_bits). Empty before
    /// the first `step` and when the mask / scoring dispatch fails.
    /// Phase 6 will feed this into physics dispatch; at Phase 3 it's
    /// read out by the parity harness.
    last_scoring_outputs: Vec<scoring::ScoreOutput>,
    /// Phase 6g (task 193): iteration count from the most recent GPU
    /// cascade dispatch. `None` before the first `step` or when the
    /// cascade failed (falls back to CPU cascade — the step still
    /// completes so state stays live, but this field records the
    /// failure so parity tests can surface it).
    last_cascade_iterations: Option<u32>,
    /// Phase 6g (task 193): set iff the most recent cascade dispatch
    /// failed. The backend falls back to the CPU cascade in that case
    /// so the tick still advances with correct semantics; the parity
    /// test's byte-exact assertion treats this as an acceptable outcome
    /// — the state is authoritative on CPU, just not GPU-driven this
    /// tick.
    last_cascade_error: Option<String>,
    /// Phase 9 (task 195): skip the post-step mask/scoring sidecar.
    /// Default is `true` for normal runs — the sidecar is pure
    /// diagnostics and duplicates a full mask+scoring dispatch the
    /// `SimBackend::step` contract doesn't require. Tests that assert on
    /// `last_mask_bitmaps` / `last_scoring_outputs` explicitly re-enable
    /// it.
    skip_scoring_sidecar: bool,
    /// Phase 9 (task 195): cumulative phase timings for diagnostic use
    /// by perf harnesses. Zeroed per-tick at `step` entry; readable
    /// via `last_phase_timings`.
    last_phase_us: PhaseTimings,

    // -------------------------------------------------------------------
    // Phase D — Task D2: persistent buffers + snapshot staging fields
    // for the GPU-resident cascade driver. Declared here in D2; consumed
    // by D3 (`snapshot()`) and D4 (`step_batch` rewrite).
    // -------------------------------------------------------------------
    /// Phase D — persistent agent SoA buffer for the resident path.
    /// Allocated on first step_batch call, reused across ticks. Sync
    /// path still uses per-kernel pooled buffers.
    resident_agents_buf: Option<wgpu::Buffer>,
    resident_agents_cap: u32,

    /// Phase D — indirect dispatch args for the resident cascade.
    /// MAX_CASCADE_ITERATIONS + 1 slots (one seed slot + one per
    /// iter). Lazy-initialised in step_batch.
    resident_indirect_args: Option<crate::gpu_util::indirect::IndirectArgsBuffer>,

    /// Phase D — double-buffered snapshot staging. `front` is the one
    /// that will be read next (filled by the previous `snapshot()`
    /// call); `back` is the one currently filling for the NEXT call.
    /// Both `None` until first snapshot() call lazy-inits them.
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    snapshot_front: Option<crate::snapshot::GpuStaging>,
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    snapshot_back: Option<crate::snapshot::GpuStaging>,

    /// Phase D — watermark tracking what portion of the main event /
    /// chronicle rings have been snapshotted. Advances monotonically.
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    snapshot_event_ring_read: u64,
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    snapshot_chronicle_ring_read: u64,

    /// Phase D — the most recent tick recorded by `step_batch`.
    /// Exposed via snapshot for the observer. Updated each tick the
    /// batch loop advances.
    latest_recorded_tick: u32,

    /// Phase D — Task D4: resident cascade driver context. Holds the
    /// caller-owned spatial/ability/physics-ring buffers the
    /// `cascade_resident::run_cascade_resident` driver consumes across
    /// ticks. Lazy-initialised inside `ensure_resident_init` on first
    /// `step_batch` call.
    resident_cascade_ctx: Option<crate::cascade_resident::CascadeResidentCtx>,

    /// Phase E: GPU-side unpack kernel for the mask's SoA buffers.
    /// Derives `agent_pos` / `agent_alive` / `agent_creature_type` from
    /// the resident `GpuAgentSlot` AoS each batch tick, replacing the
    /// host-side pack-and-upload path in the batch loop. The sync
    /// `step()` still uses `FusedMaskKernel::upload_soa_from_state`
    /// from `SimState` — this kernel is only used by `step_batch`.
    ///
    /// Retained for backward compatibility but NOT called on the
    /// step_batch hot path — subsumed by [`mask::FusedAgentUnpackKernel`].
    #[allow(dead_code)]
    mask_unpack_kernel: mask::MaskUnpackKernel,

    /// Phase E: GPU-side unpack kernel for scoring's `agent_data_buf`
    /// AoS. Derives the mutable subset (pos/hp/shield/alive/
    /// creature_type/hp_pct) from the resident `GpuAgentSlot` AoS each
    /// batch tick. Static fields (attack_range, hunger, thirst,
    /// fatigue) keep their tick-0 values written by
    /// `ScoringKernel::initialize_for_batch`.
    ///
    /// Retained for backward compatibility but NOT called on the
    /// step_batch hot path — the fused [`mask::FusedAgentUnpackKernel`]
    /// subsumes both mask and scoring unpacks into one dispatch.
    #[allow(dead_code)]
    scoring_unpack_kernel: scoring::ScoringUnpackKernel,
    /// Suspect 5: fused mask+scoring unpack kernel. Writes mask's SoA
    /// (pos/alive/creature_type) + scoring's `agent_data_buf` in a
    /// single dispatch, saving one compute pass begin/end + one
    /// pipeline set per batch tick.
    fused_unpack_kernel: mask::FusedAgentUnpackKernel,
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
    /// Kernel init failed — WGSL emit or naga parse.
    Kernel(mask::KernelError),
    /// Phase 3 scoring kernel init failed — WGSL compile / pipeline
    /// construction.
    Scoring(scoring::ScoringError),
    /// Phase 6d view storage init failed — fold shader compile / buffer
    /// allocation.
    ViewStorage(view_storage::ViewStorageError),
}

#[cfg(feature = "gpu")]
impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuInitError::NoAdapter => write!(f, "no compatible GPU adapter"),
            GpuInitError::RequestDevice(s) => write!(f, "request_device: {s}"),
            GpuInitError::Kernel(e) => write!(f, "kernel init: {e}"),
            GpuInitError::Scoring(e) => write!(f, "scoring init: {e}"),
            GpuInitError::ViewStorage(e) => write!(f, "view_storage init: {e}"),
        }
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for GpuInitError {}

#[cfg(feature = "gpu")]
impl From<mask::KernelError> for GpuInitError {
    fn from(e: mask::KernelError) -> Self {
        GpuInitError::Kernel(e)
    }
}

#[cfg(feature = "gpu")]
impl From<scoring::ScoringError> for GpuInitError {
    fn from(e: scoring::ScoringError) -> Self {
        GpuInitError::Scoring(e)
    }
}

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
impl SimBackend for GpuBackend {
    #[inline]
    fn step<B: PolicyBackend>(
        &mut self,
        state: &mut SimState,
        scratch: &mut SimScratch,
        events: &mut EventRing,
        policy: &B,
        cascade: &CascadeRegistry,
    ) {
        engine::step::step(state, scratch, events, policy, cascade);
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
    /// compile the Attack mask kernel. Blocks on the async setup via
    /// `pollster::block_on` so callers stay synchronous — the engine
    /// tick loop is strictly sync.
    ///
    /// On a headless / no-GPU machine wgpu falls back to LLVMpipe
    /// (Linux software Vulkan) or equivalent, which is slow but correct.
    /// CI that lacks a real GPU should still see this succeed.
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
        // `downlevel_defaults` caps `max_storage_buffers_per_shader_stage`
        // at 4 — fine for the Phase 1 Attack-only kernel (4 storage
        // buffers + 1 uniform) but the Phase 2 fused kernel needs 10
        // (3 SoA reads + 7 per-mask bitmap writes). The adapter's real
        // limit on every Vulkan / Metal / DX12 target we care about is
        // at least 16; lean on that by asking for the adapter's full
        // limits. On CI-style software adapters (LLVMpipe) this still
        // works: LLVMpipe reports >= 8 and we only need 10. If some
        // future target returns a smaller limit we'll have to cap N
        // and split the dispatch — but that's a Phase 3+ concern.
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine_gpu::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::RequestDevice(format!("{e}")))?;

        let mask_kernel = mask::FusedMaskKernel::new(&device)?;
        let scoring_kernel = scoring::ScoringKernel::new(&device, &queue)?;
        let view_storage = view_storage::ViewStorage::new(&device, INITIAL_VIEW_AGENT_CAP)?;
        let mask_unpack_kernel = mask::MaskUnpackKernel::new(&device)?;
        let scoring_unpack_kernel = scoring::ScoringUnpackKernel::new(&device)?;
        let fused_unpack_kernel = mask::FusedAgentUnpackKernel::new(&device)?;

        Ok(Self {
            device,
            queue,
            mask_kernel,
            scoring_kernel,
            view_storage,
            cascade_ctx: None,
            backend_label,
            last_mask_bitmaps: Vec::new(),
            last_scoring_outputs: Vec::new(),
            last_cascade_iterations: None,
            last_cascade_error: None,
            skip_scoring_sidecar: true,
            last_phase_us: PhaseTimings::default(),
            // Phase D — Task D2: persistent buffers + snapshot staging.
            // All `None` / `0` at construction; lazy-initialised on
            // first `step_batch` (D4) / `snapshot()` (D3) call.
            resident_agents_buf: None,
            resident_agents_cap: 0,
            resident_indirect_args: None,
            snapshot_front: None,
            snapshot_back: None,
            snapshot_event_ring_read: 0,
            snapshot_chronicle_ring_read: 0,
            latest_recorded_tick: 0,
            resident_cascade_ctx: None,
            mask_unpack_kernel,
            scoring_unpack_kernel,
            fused_unpack_kernel,
        })
    }

    /// Lazily build the cascade context on first `step`. Caches the
    /// result on the backend; subsequent ticks are amortised — no more
    /// DSL parsing or WGSL compilation. Exposed as its own fn so tests
    /// can ask for the init cost explicitly before a timing loop starts.
    pub fn ensure_cascade_initialized(&mut self) -> Result<(), cascade::CascadeCtxError> {
        if self.cascade_ctx.is_none() {
            let ctx = cascade::CascadeCtx::new(&self.device)?;
            self.cascade_ctx = Some(ctx);
        }
        Ok(())
    }

    /// Iteration count from the most recent GPU cascade. `None` before
    /// the first `step` or when cascade init failed on that tick.
    pub fn last_cascade_iterations(&self) -> Option<u32> {
        self.last_cascade_iterations
    }

    /// Enable or disable the post-step scoring sidecar. The sidecar
    /// runs a full mask + scoring kernel dispatch against post-step
    /// state to populate `last_mask_bitmaps` / `last_scoring_outputs`
    /// for diagnostic callers. Perf harnesses turn it off to shave
    /// ~3-10 ms/tick at N=1000 (the sidecar dispatches are pure
    /// duplication since the backend doesn't use its own scoring
    /// output — the CPU `step_phases_1_to_3` already decided actions).
    pub fn set_skip_scoring_sidecar(&mut self, skip: bool) {
        self.skip_scoring_sidecar = skip;
    }

    /// True iff the scoring sidecar is currently disabled. Default
    /// is `false` so existing tests keep working.
    pub fn skip_scoring_sidecar(&self) -> bool {
        self.skip_scoring_sidecar
    }

    /// Per-tick phase timings from the most recent `step`. Reset at
    /// the top of each step; populated as each phase finishes. Fields
    /// are in microseconds. A zero field means the phase was skipped
    /// on that tick (e.g. sidecar with `skip_scoring_sidecar = true`).
    pub fn last_phase_timings(&self) -> PhaseTimings {
        self.last_phase_us
    }

    /// Task 203 — drain the GPU chronicle ring into the CPU event
    /// ring. The chronicle ring is written to by physics every tick
    /// (via `emit ChronicleEntry` rules) but is NOT drained by the
    /// cascade driver — it accumulates across ticks until this method
    /// runs. Every currently-resident record is pushed into `events`
    /// as an `Event::ChronicleEntry`, then the ring's tail atomic is
    /// reset to 0 so the next session starts with a fresh window.
    ///
    /// Returns the chronicle drain outcome, or `None` if the cascade
    /// context hasn't been initialised yet (no step has run and
    /// there's nothing to drain).
    ///
    /// ## Relationship to the CPU `cold_state_replay` path
    ///
    /// The authoritative source of `Event::ChronicleEntry` in the CPU
    /// event ring today is `cascade::cold_state_replay`, which walks
    /// the drained seed + cascade events once per tick and dispatches
    /// the 8 chronicle rules CPU-side. That path runs inside
    /// `GpuBackend::step` unconditionally — flushing the GPU
    /// chronicle ring on top of that would double-count every
    /// narrative entry.
    ///
    /// So `flush_chronicle` is opt-in for callers that DISABLE
    /// `cold_state_replay` (e.g., a future observability tool that
    /// wants to read chronicles off the GPU without round-tripping
    /// through the CPU cold-state handler). In the default step
    /// pipeline the chronicle ring is a write-only observability
    /// channel that tests/tools can peek at without perturbing the
    /// CPU ring's contents.
    pub fn flush_chronicle(
        &mut self,
        events: &mut EventRing,
    ) -> Option<crate::event_ring::ChronicleDrainOutcome> {
        let cascade_ctx = self.cascade_ctx.as_mut()?;
        let outcome = match cascade_ctx
            .physics
            .chronicle_ring()
            .drain(&self.device, &self.queue, events)
        {
            Ok(o) => o,
            Err(e) => {
                eprintln!("engine_gpu::flush_chronicle: drain failed: {e}");
                return None;
            }
        };
        // Reset the tail so subsequent ticks don't re-see already-
        // drained records. The records buffer stays populated (stale
        // slots beyond the new tail are invisible to the next drain).
        cascade_ctx.physics.chronicle_ring().reset(&self.queue);
        Some(outcome)
    }

    /// Set iff the most recent `step` fell back to the CPU cascade
    /// (init or dispatch error). The backend records the error string
    /// rather than returning it so the `SimBackend::step` signature
    /// stays byte-for-byte compatible with `CpuBackend`.
    pub fn last_cascade_error(&self) -> Option<&str> {
        self.last_cascade_error.as_deref()
    }

    /// Borrow the backing view storage. Tests and integration callers
    /// use this to fold events into the view cells before a scoring
    /// dispatch.
    pub fn view_storage(&self) -> &view_storage::ViewStorage {
        &self.view_storage
    }

    /// Mutable borrow of the view storage — the integration layer uses
    /// this to dispatch fold kernels against the backend's device/queue.
    pub fn view_storage_mut(&mut self) -> &mut view_storage::ViewStorage {
        &mut self.view_storage
    }

    /// The backend's wgpu device — exposed so tests / integration can
    /// dispatch `view_storage` fold kernels against the same device the
    /// scoring kernel runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The backend's wgpu queue — paired with `device()` for uploads
    /// and fold dispatches from the test harness.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Resize the view storage to match the given agent_cap, preserving
    /// zeroed state. Called before the first step if the SimState's
    /// agent_cap exceeds `INITIAL_VIEW_AGENT_CAP`. This is not called
    /// inside `step` to avoid silently dropping fold state mid-run.
    pub fn rebuild_view_storage(
        &mut self,
        agent_cap: u32,
    ) -> Result<(), view_storage::ViewStorageError> {
        self.view_storage = view_storage::ViewStorage::new(&self.device, agent_cap)?;
        Ok(())
    }

    /// Human-readable name of the wgpu backend the device is running
    /// on — one of `Vulkan`, `Metal`, `Dx12`, `Gl`, `BrowserWebGpu`, or
    /// `Empty`. Captured at init so tests can log which path they
    /// actually exercised (useful when Linux falls back to `Gl`/LLVMpipe).
    pub fn backend_label(&self) -> &str {
        &self.backend_label
    }

    /// Per-mask packed bitmaps from the most recent `step`, in the
    /// same order as `mask_bindings()`. Bit `i` of word `i/32` of
    /// `last_mask_bitmaps()[k]` is set iff slot `i`'s agent passed
    /// mask `k`'s predicate this tick. Empty before the first `step`;
    /// `last_mask_bitmaps()[k]` is `None`-empty if the kernel dispatch
    /// failed the most recent tick.
    pub fn last_mask_bitmaps(&self) -> &[Vec<u32>] {
        &self.last_mask_bitmaps
    }

    /// Per-mask metadata (name, index, shape) in fused-kernel order.
    /// Stable across ticks — set at `GpuBackend::new` when the
    /// pipeline compiles. Callers use this to pair a bitmap from
    /// `last_mask_bitmaps` with its DSL mask name for diagnostics or
    /// mask-specific handling.
    pub fn mask_bindings(&self) -> &[dsl_compiler::emit_mask_wgsl::FusedMaskBinding] {
        self.mask_kernel.bindings()
    }

    /// Convenience lookup — scan `mask_bindings()` for the slot
    /// matching `name` and return the corresponding bitmap from the
    /// most recent `step`, or `None` if the mask isn't in the fused
    /// kernel (Cast, future non-agent masks) or `step` hasn't run yet.
    pub fn last_bitmap_for(&self, name: &str) -> Option<&[u32]> {
        let bindings = self.mask_kernel.bindings();
        let idx = bindings.iter().position(|b| b.mask_name == name)?;
        self.last_mask_bitmaps.get(idx).map(|v| v.as_slice())
    }

    /// Run the fused-mask kernel against the current `state` without
    /// touching the CPU step — used by the parity test's direct
    /// "known-state spawn check" mode. Returns every bitmap in the
    /// emitter's binding order; caller pairs each against
    /// `mask::cpu_mask_bitmap(state, name)`.
    pub fn verify_masks_on_gpu(
        &mut self,
        state: &SimState,
    ) -> Result<Vec<Vec<u32>>, mask::KernelError> {
        self.mask_kernel.run_and_readback(&self.device, &self.queue, state)
    }

    /// Per-agent scoring outputs from the most recent `step` — one
    /// `ScoreOutput` per agent slot. Carries the GPU's argmax decision
    /// (chosen_action, chosen_target). Empty before the first `step`
    /// or when the scoring kernel dispatch failed.
    pub fn last_scoring_outputs(&self) -> &[scoring::ScoreOutput] {
        &self.last_scoring_outputs
    }

    /// Run the scoring kernel against `state`'s current snapshot
    /// (without advancing the CPU step). Used by the parity test's
    /// "known-state spawn check" mode — same shape as
    /// `verify_masks_on_gpu` but for scoring. Internally runs the
    /// fused mask kernel first to get the bitmaps the scoring kernel
    /// reads.
    ///
    /// Phase 6d: resets the view storage to zero before dispatching so
    /// the scoring kernel reads deterministic empty views. Callers that
    /// need pre-folded view state should use `run_step_once` (which
    /// preserves cross-call fold state) or call `view_storage_mut()`
    /// directly to seed cells.
    pub fn verify_scoring_on_gpu(
        &mut self,
        state: &SimState,
    ) -> Result<Vec<scoring::ScoreOutput>, scoring::ScoringError> {
        self.ensure_view_storage_cap(state.agent_cap())?;
        self.view_storage.reset(&self.queue);
        let bitmaps = self
            .mask_kernel
            .run_and_readback(&self.device, &self.queue, state)?;
        self.scoring_kernel.run_and_readback(
            &self.device,
            &self.queue,
            state,
            &self.mask_kernel,
            &bitmaps,
            &self.view_storage,
        )
    }

    /// Grow `view_storage` to match the SimState's agent_cap if the
    /// current storage is too small. Preserves zero-state on resize
    /// (fresh allocation). Used internally by `step` and
    /// `verify_scoring_on_gpu`.
    fn ensure_view_storage_cap(
        &mut self,
        agent_cap: u32,
    ) -> Result<(), scoring::ScoringError> {
        if self.view_storage.agent_cap() < agent_cap {
            view_storage::ViewStorage::new(&self.device, agent_cap)
                .map(|vs| self.view_storage = vs)
                .map_err(|e| scoring::ScoringError::Dispatch(format!(
                    "view_storage rebuild for agent_cap={agent_cap} failed: {e}"
                )))?;
        }
        Ok(())
    }

    /// Same shape as `verify_scoring_on_gpu` but **does not** reset the
    /// view storage. Callers that have pre-folded view cells (via
    /// direct `view_storage.fold_*_events` calls) use this to exercise
    /// the scoring kernel against the post-fold state without the
    /// reset wiping their writes. Used by the Piece 1 test harness.
    pub fn verify_scoring_on_gpu_preserving_views(
        &mut self,
        state: &SimState,
    ) -> Result<Vec<scoring::ScoreOutput>, scoring::ScoringError> {
        self.ensure_view_storage_cap(state.agent_cap())?;
        let bitmaps = self
            .mask_kernel
            .run_and_readback(&self.device, &self.queue, state)?;
        self.scoring_kernel.run_and_readback(
            &self.device,
            &self.queue,
            state,
            &self.mask_kernel,
            &bitmaps,
            &self.view_storage,
        )
    }

    /// Phase 9 (task 195) — batched step API. Runs `n_ticks` ticks in
    /// a row without any CPU-side work between them beyond what
    /// `SimBackend::step` already does per tick. The API exists to
    /// give callers a single entry point for "just advance the sim N
    /// ticks"; the per-tick pipeline still submits + waits on GPU work
    /// individually.
    ///
    /// This API is a scaffolding deliverable for the full megakernel
    /// plan: the next milestone would be to share a single command
    /// buffer across all N ticks (no wgpu submit between ticks), with
    /// a GPU-resident cascade range buffer so physics can iterate on
    /// its own output without CPU readback. That requires:
    ///   * a WGSL `update_cascade_range` kernel that reads the
    ///     event-ring tail + writes the next iteration's (start, count)
    ///     into a storage buffer physics reads,
    ///   * a physics shader rebind that consumes its events-in slice
    ///     from the shared event ring rather than a separate
    ///     `events_in_buf`,
    ///   * GPU-side apply_actions + movement kernels so phases 4a and
    ///     movement don't need CPU-side state mutation.
    ///
    /// The scope of all three changes is ~800-1200 LOC of WGSL + Rust
    /// glue. Task 195 lands the per-tick wins (pooled staging, fused
    /// submits, sidecar opt-out, diagnostic surface) that unblock the
    /// measurement work; task 196+ consumes those + lands the
    /// cross-tick single-submit pipeline.
    /// Phase D — Task D4: GPU-resident batched step. Records `n_ticks`
    /// ticks of resident cascade execution into a single command
    /// encoder, submits once, polls once.
    ///
    /// Unlike the per-tick `step`, this path is explicitly
    /// non-deterministic and does NOT keep `state` byte-for-byte in
    /// lockstep with the GPU buffer mid-batch — the caller's `state` is
    /// slightly stale at end-of-batch (CPU-side `state.tick` is advanced,
    /// but agent HP / position / alive fields are NOT read back from
    /// the GPU `resident_agents_buf` until a future integration task
    /// lands a commit step). Observers consume the GPU-side snapshot
    /// via [`Self::snapshot`] instead.
    ///
    /// ### Unused parameters (batch path)
    ///
    /// * `scratch` — kept for signature compatibility with the sync
    ///   `SimBackend::step` trait. The resident kernels don't use the
    ///   CPU scratch buffers.
    /// * `events` — the batch path does NOT push to the CPU event ring
    ///   per-tick. GPU-emitted events stay resident in the physics
    ///   rings and are observed via `snapshot()`.
    /// * `policy` — the batch path has no policy hook. The GPU scoring
    ///   kernel runs its own deterministic argmax.
    /// * `cascade` — the batch path uses its own `cascade_resident`
    ///   driver with the engine-builtin cascade physics compiled into
    ///   the lazy `cascade_ctx`.
    ///
    /// ### Fallback
    ///
    /// If resident init fails (GPU allocation error, cascade DSL load
    /// failure, etc.) the batch path falls back to calling the sync
    /// `SimBackend::step` N times so the tick loop still advances.
    /// Mid-batch kernel failures panic via `expect(...)` — the batch
    /// path is committed once init succeeds.
    pub fn step_batch<B: PolicyBackend>(
        &mut self,
        state: &mut SimState,
        scratch: &mut SimScratch,
        events: &mut EventRing,
        policy: &B,
        cascade: &CascadeRegistry,
        n_ticks: u32,
    ) {
        if let Err(e) = self.ensure_resident_init(state) {
            eprintln!(
                "engine_gpu::step_batch: resident init failed ({e}), falling back to sync loop"
            );
            for _ in 0..n_ticks {
                <Self as SimBackend>::step(self, state, scratch, events, policy, cascade);
            }
            return;
        }

        // Shadow unused params so the reader sees they're intentional
        // no-ops on the batch path (doc-commented above).
        let _ = (scratch, events, policy, cascade);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine_gpu::step_batch::enc"),
            });

        // C1 + C2 fix: reset the batch-scoped event accumulator and the
        // chronicle ring at the START of each batch. Both rings are
        // append-only within a batch — the batch events ring collects
        // apply+movement events across every tick (per-tick-reset of
        // `apply_event_ring` would otherwise drop all but the last
        // tick's events from `snapshot()`'s view), and the chronicle
        // ring collects narrative entries the physics kernel emits.
        // Without these resets the chronicle ring would grow
        // unboundedly across back-to-back `step_batch` calls in a
        // long-running session and eventually overflow its
        // `DEFAULT_CHRONICLE_CAPACITY`.
        //
        // Also zero the snapshot watermark — the batch ring is reset,
        // so any watermark recorded during the previous batch is
        // meaningless against the new tail.
        {
            let resident_ctx = self
                .resident_cascade_ctx
                .as_ref()
                .expect("resident_cascade_ctx ensured by ensure_resident_init");
            resident_ctx.reset_batch_events_ring(&mut encoder);
            resident_ctx.reset_chronicle_ring(&mut encoder);
        }
        self.snapshot_event_ring_read = 0;

        let agent_cap = state.agent_cap();
        for _ in 0..n_ticks {
            let agents_buf = self
                .resident_agents_buf
                .as_ref()
                .expect("resident_agents_buf ensured by ensure_resident_init");

            // 1. Fused unpack: one dispatch writes both mask's SoA
            //    (pos/alive/creature_type) and scoring's
            //    `agent_data_buf` (mutable subset: pos/hp/shield/alive/
            //    creature_type/hp_pct). Merges what used to be two
            //    separate unpack dispatches into one — saves a compute
            //    pass begin/end + pipeline set per tick. Also emits
            //    the per-tick mask-bitmap clears (mask kernel
            //    atomicOr's bits in, so stale bits would poison
            //    subsequent ticks).
            self.fused_unpack_kernel
                .encode_unpack(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &mut self.mask_kernel,
                    &mut self.scoring_kernel,
                    agents_buf,
                    agent_cap,
                )
                .expect("fused unpack dispatch");
            self.mask_kernel
                .run_resident(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    agents_buf,
                    agent_cap,
                )
                .expect("mask resident dispatch");

            // 2. Scoring: reads mask_bitmaps + agent_data (both
            //    populated above). Only the `tick` field of cfg is
            //    refreshed; other cfg scalars are stable across the
            //    batch.
            self.scoring_kernel
                .refresh_tick_cfg_for_resident(&self.queue, state)
                .expect("scoring refresh_tick_cfg_for_resident");
            self.scoring_kernel
                .run_resident(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    agents_buf,
                    self.mask_kernel.mask_bitmaps_buf(),
                    agent_cap,
                )
                .expect("scoring resident dispatch");

            // 3. apply_actions + movement: both read from the scoring
            //    output buffer, mutate `resident_agents_buf`, and append
            //    events into the shared apply_event_ring. Reset the
            //    ring's tail inside the encoder so the reset is ordered
            //    relative to this tick's kernels (not all ticks'
            //    kernels — queue.write_buffer would collapse).
            //
            //    Cfg uniforms for both kernels were uploaded once in
            //    `ensure_resident_init` (they carry `tick`, which the
            //    batch path lets drift per the step_batch
            //    non-determinism contract — see the plan's Open
            //    Question #1).
            let cascade_ctx = self
                .cascade_ctx
                .as_mut()
                .expect("cascade_ctx ensured by ensure_resident_init");
            encoder.clear_buffer(cascade_ctx.apply_event_ring.tail_buffer(), 0, None);

            cascade_ctx
                .apply_actions
                .run_resident(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    agents_buf,
                    self.scoring_kernel.scoring_buf(),
                    &cascade_ctx.apply_event_ring,
                    agent_cap,
                )
                .expect("apply_actions resident dispatch");

            cascade_ctx
                .movement
                .run_resident(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    agents_buf,
                    self.scoring_kernel.scoring_buf(),
                    &cascade_ctx.apply_event_ring,
                    agent_cap,
                )
                .expect("movement resident dispatch");

            // 3b. C1 fix: append apply+movement events into the
            //     batch-scoped accumulator BEFORE the cascade seed
            //     clears-and-consumes `apply_event_ring.tail`. Reads
            //     apply_tail atomically and copies records 0..tail into
            //     `batch_events_ring`, advancing its atomic tail. This
            //     is what `snapshot()` later reads from — so the
            //     snapshot sees events from EVERY tick in the batch,
            //     not just the last one.
            //
            //     Ordered inside the encoder (and between compute
            //     passes) so it lands AFTER movement writes into
            //     `apply_event_ring` and BEFORE the cascade seed reads
            //     the tail.
            {
                let resident_ctx = self
                    .resident_cascade_ctx
                    .as_mut()
                    .expect("resident_cascade_ctx ensured by ensure_resident_init");
                resident_ctx.encode_append_apply_events(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &cascade_ctx.apply_event_ring,
                    agent_cap,
                );
            }

            // 4. Cascade: 2× spatial queries + seed + N physics iterations.
            //    The resident driver records all of it into the current
            //    encoder.
            //
            //    Split borrow: `run_cascade_resident` takes `&mut
            //    CascadeCtx` AND `&GpuEventRing` (the apply ring that
            //    lives inside that same CascadeCtx). The borrow checker
            //    can't see that the driver only touches
            //    `physics`/`spatial`/`abilities` through the `&mut`
            //    path and only reads `apply_event_ring`'s buffer
            //    handles through the `&` path, so a naive pair of
            //    references rejects. We lift the shared `&` out of the
            //    `&mut` by reborrowing via a raw pointer — safe because
            //    the driver guarantees no aliased writes between the
            //    two fields within a single call.
            let resident_ctx = self
                .resident_cascade_ctx
                .as_mut()
                .expect("resident_cascade_ctx ensured by ensure_resident_init");
            let indirect_args = self
                .resident_indirect_args
                .as_ref()
                .expect("resident_indirect_args ensured by ensure_resident_init");
            // SAFETY: `cascade_ctx` is a `&mut CascadeCtx`. We reborrow
            // `apply_event_ring` through a `*const` so the compiler
            // doesn't consider the `&mut cascade_ctx` later in the
            // call to alias it. The driver's body reads `tail_buffer()`
            // + `records_buffer()` on the ring (both immutable &
            // accessors) and never reaches into `cascade_ctx.
            // apply_event_ring` through the `&mut` path, so there is
            // no data race or aliased mutation. The pointer's lifetime
            // is bounded by the same block.
            let apply_ring_ptr: *const crate::event_ring::GpuEventRing =
                &cascade_ctx.apply_event_ring;
            let apply_ring_ref: &crate::event_ring::GpuEventRing =
                unsafe { &*apply_ring_ptr };
            // Heuristic cap: if a recent sync tick observed cascade
            // convergence at N iterations, record only N+2 dispatches
            // this tick — saves the per-iter encode cost for the 6+
            // no-op iters typical on low-convergence workloads. When
            // `last_cascade_iterations` is `None` (no prior sync tick,
            // or cascade failed) fall back to the full
            // MAX_CASCADE_ITERATIONS for safety. The +2 margin
            // tolerates modest run-to-run variance; workloads with
            // deeper cascades pay the same cost as today.
            let iter_cap = match self.last_cascade_iterations {
                Some(n) => (n + 2).min(crate::cascade::MAX_CASCADE_ITERATIONS),
                None => crate::cascade::MAX_CASCADE_ITERATIONS,
            };
            crate::cascade_resident::run_cascade_resident_with_iter_cap(
                &self.device,
                &self.queue,
                &mut encoder,
                state,
                cascade_ctx,
                resident_ctx,
                agents_buf,
                apply_ring_ref,
                indirect_args,
                iter_cap,
            )
            .expect("cascade_resident dispatch");

            // 5. Advance the tick counter on CPU. GPU-side tick / rng
            //    advance is deferred (plan Open Question #1) — the
            //    batch path's non-determinism disclaimer covers it.
            state.tick = state.tick.wrapping_add(1);
            self.latest_recorded_tick = state.tick;
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    /// Phase D — Task D4: Lazy-init for the resident batch path.
    ///
    /// Allocates:
    ///   * `resident_agents_buf` sized to `state.agent_cap() *
    ///     size_of::<GpuAgentSlot>()` bytes (STORAGE | COPY_SRC |
    ///     COPY_DST), uploaded with the initial agent SoA via
    ///     `physics::pack_agent_slots(state)`.
    ///   * `resident_indirect_args` at `MAX_CASCADE_ITERATIONS + 1` slots.
    ///   * `cascade_ctx` via `ensure_cascade_initialized` if not already.
    ///   * `resident_cascade_ctx` via `CascadeResidentCtx::new`.
    ///
    /// Also grows `view_storage` if `state.agent_cap()` exceeds the
    /// current cap, and resizes `resident_agents_buf` / re-uploads the
    /// initial state on agent_cap grow.
    ///
    /// Idempotent on a stable `agent_cap` — no allocation or upload
    /// happens after the first successful call until the cap changes.
    fn ensure_resident_init(&mut self, state: &SimState) -> Result<(), String> {
        let agent_cap = state.agent_cap();

        // Cascade context (DSL load + physics WGSL compile). Idempotent.
        self.ensure_cascade_initialized()
            .map_err(|e| format!("ensure_cascade_initialized: {e}"))?;

        // View storage must cover agent_cap so the scoring kernel's
        // view bindings don't go OOB.
        self.ensure_view_storage_cap(agent_cap)
            .map_err(|e| format!("ensure_view_storage_cap: {e}"))?;

        // Resident agent buffer. Allocate-or-grow; (re)upload the
        // current SimState agent SoA on (re)allocate so the kernels see
        // a well-defined starting state.
        let need_alloc = match &self.resident_agents_buf {
            Some(_) => self.resident_agents_cap < agent_cap,
            None => true,
        };
        if need_alloc {
            let bytes = (agent_cap as u64)
                * (std::mem::size_of::<crate::physics::GpuAgentSlot>() as u64);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine_gpu::resident_agents_buf"),
                size: bytes.max(std::mem::size_of::<crate::physics::GpuAgentSlot>() as u64),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let packed = crate::physics::pack_agent_slots(state);
            self.queue
                .write_buffer(&buf, 0, bytemuck::cast_slice(&packed));
            self.resident_agents_buf = Some(buf);
            self.resident_agents_cap = agent_cap;
        }

        // Indirect args buffer. Sized for `MAX_CASCADE_ITERATIONS + 1`
        // slots (seed + one per cascade iter).
        if self.resident_indirect_args.is_none() {
            let slots = crate::cascade::MAX_CASCADE_ITERATIONS + 1;
            self.resident_indirect_args = Some(
                crate::gpu_util::indirect::IndirectArgsBuffer::new(&self.device, slots),
            );
        }

        // Resident cascade driver context. Allocates the ping-pong
        // physics rings, the ability buffers, the seed kernel, and the
        // chronicle ring.
        if self.resident_cascade_ctx.is_none() {
            let ctx = crate::cascade_resident::CascadeResidentCtx::new(&self.device)
                .map_err(|e| format!("CascadeResidentCtx::new: {e}"))?;
            self.resident_cascade_ctx = Some(ctx);
        }

        // Phase E: seed the kernel-internal buffers the batch path no
        // longer refreshes per-tick.
        //
        //   * Mask SoA (pos / alive / creature_type) — the tick-0
        //     values matter only as a bootstrap; the `MaskUnpackKernel`
        //     overwrites them every tick from `resident_agents_buf`.
        //     Calling `upload_soa_from_state` here also allocates the
        //     mask's pool at `agent_cap` and writes the cfg uniform
        //     (radii), which IS stable across a batch.
        //   * Scoring `GpuAgentData` — full tick-0 pack populates both
        //     the mutable fields (the unpack kernel refreshes these
        //     per-tick) and the static fields (attack_range, hunger,
        //     thirst, fatigue — not mutated by any GPU kernel today,
        //     so tick-0 values suffice across the batch). Also
        //     snapshots `view_storage` buffer handles into the scoring
        //     pool, so the per-tick scoring dispatch does not need
        //     `&ViewStorage`.
        //   * Apply_actions + movement cfg uniforms — 32 bytes each.
        //     They carry `tick`, which drifts in the batch per the
        //     step_batch non-determinism contract, but re-uploading
        //     them per tick cost the batch path a `write_buffer` each
        //     for no correctness benefit.
        self.mask_kernel
            .upload_soa_from_state(&self.device, &self.queue, state);
        self.scoring_kernel
            .initialize_for_batch(&self.device, &self.queue, state, &self.view_storage)
            .map_err(|e| format!("scoring initialize_for_batch: {e}"))?;
        {
            let cascade_ctx = self
                .cascade_ctx
                .as_mut()
                .expect("cascade_ctx ensured above");
            cascade_ctx
                .apply_actions
                .upload_soa_from_state(&self.device, &self.queue, state);
            cascade_ctx
                .movement
                .upload_soa_from_state(&self.device, &self.queue, state);
        }

        Ok(())
    }

    /// Cheap non-blocking snapshot via double-buffered staging. First
    /// call returns [`crate::snapshot::GpuSnapshot::empty`] (no previous
    /// frame to map). Subsequent calls return state as-of the *previous*
    /// snapshot call (one-frame lag — acceptable for rendering, which
    /// will interpolate via its own delta mechanism).
    ///
    /// One `device.poll(Wait)` per call.
    ///
    /// Must be called from a context where `step_batch` has been invoked
    /// at least once — before `step_batch` there's no
    /// `resident_agents_buf` to snapshot, so this returns
    /// `Ok(GpuSnapshot::empty())` without error. Similarly returns empty
    /// if the cascade context has not yet been initialised (no physics
    /// event ring to read).
    ///
    /// # Event ring choice
    ///
    /// Snapshots `resident_cascade_ctx.batch_events_ring` — the
    /// batch-scoped accumulator into which `step_batch` copies every
    /// tick's apply+movement events (via a small GPU append kernel,
    /// see [`crate::cascade_resident::CascadeResidentCtx`]). That ring
    /// is reset at the top of each `step_batch` call and accumulates
    /// monotonically within the batch, so a snapshot taken after
    /// `step_batch(N)` observes events from all N ticks — not just the
    /// last one.
    ///
    /// The physics `apply_event_ring` itself can't be snapshotted
    /// because its tail is cleared at the top of every tick inside the
    /// batch (required for cascade-seed correctness: the seed kernel
    /// consumes `apply_tail` for the initial event count and re-seeing
    /// last tick's events would double-count). The `batch_events_ring`
    /// is append-only across the batch so it preserves the multi-tick
    /// view `snapshot()` callers need.
    ///
    /// The `snapshot_event_ring_read` watermark is a cross-snapshot
    /// cursor — reset to 0 whenever `step_batch` is called (since the
    /// underlying ring was reset) and advanced to the current tail
    /// after each successful snapshot. Back-to-back snapshots within a
    /// single batch yield disjoint event slices.
    pub fn snapshot(
        &mut self,
    ) -> Result<crate::snapshot::GpuSnapshot, crate::snapshot::SnapshotError> {
        // Case 1: `step_batch` hasn't allocated the resident agents
        // buffer yet. Nothing to snapshot — return empty.
        let agents_buf = match self.resident_agents_buf.as_ref() {
            Some(b) => b,
            None => return Ok(crate::snapshot::GpuSnapshot::empty()),
        };

        // Case 2: resident cascade context not initialised — no batch
        // events ring exists to read either. `step_batch` path always
        // ensures this before allocating `resident_agents_buf`, but
        // defend against a misuse / future reorder.
        let resident_ctx = match self.resident_cascade_ctx.as_ref() {
            Some(c) => c,
            None => return Ok(crate::snapshot::GpuSnapshot::empty()),
        };

        // Lazy-init the staging pair on first call. Size for the
        // current resident agent capacity + the event ring's default
        // capacity envelope (the physics ring is sized at the same
        // default so this is a conservative upper bound).
        if self.snapshot_front.is_none() && self.snapshot_back.is_none() {
            let event_cap = crate::event_ring::DEFAULT_CAPACITY;
            self.snapshot_front = Some(crate::snapshot::GpuStaging::new(
                &self.device,
                self.resident_agents_cap,
                event_cap,
            ));
            self.snapshot_back = Some(crate::snapshot::GpuStaging::new(
                &self.device,
                self.resident_agents_cap,
                event_cap,
            ));
        }

        // Grow staging if the resident capacity changed. `ensure_cap`
        // resets the filled flag, so a freshly-grown front returns
        // empty from `take_snapshot` — correct behaviour, next call
        // populates at the new size.
        let event_cap = crate::event_ring::DEFAULT_CAPACITY;
        if let Some(front) = self.snapshot_front.as_mut() {
            front.ensure_cap(&self.device, self.resident_agents_cap, event_cap);
        }
        if let Some(back) = self.snapshot_back.as_mut() {
            back.ensure_cap(&self.device, self.resident_agents_cap, event_cap);
        }

        // 1. Take a snapshot of the FRONT (filled by the previous
        //    snapshot call — returns empty on the first real call).
        let snap = self
            .snapshot_front
            .as_mut()
            .expect("snapshot_front lazy-inited above")
            .take_snapshot(&self.device, self.resident_agents_cap as usize)?;

        // 2. Read the batch events ring's current tail so we know
        //    which slice to copy into the BACK. One 4-byte blocking
        //    readback per snapshot — acceptable per plan Option (a).
        //    This ring is reset at the top of each `step_batch` call
        //    and accumulates apply+movement events across every tick
        //    in that batch, so its tail reflects the cumulative event
        //    count for the current batch.
        let main_event_ring = resident_ctx.batch_events_ring();
        let tail_vec: Vec<u32> = crate::gpu_util::readback::readback_typed::<u32>(
            &self.device,
            &self.queue,
            main_event_ring.tail_buffer(),
            4,
        )
        .map_err(crate::snapshot::SnapshotError::Ring)?;
        let event_ring_tail = *tail_vec.first().unwrap_or(&0) as u64;

        // 3. Kick the copy into the BACK (filling for the next call).
        //    Encoder + submit live entirely inside this method.
        let agent_bytes = (self.resident_agents_cap as u64)
            * (std::mem::size_of::<crate::physics::GpuAgentSlot>() as u64);
        // The batch events ring is append-only across the batch.
        // Observers who snapshot more than once per batch get the
        // delta since their last call via the `snapshot_event_ring_read`
        // watermark. The watermark is reset to 0 at the top of each
        // `step_batch` (since the ring itself is reset), so the first
        // in-batch snapshot starts at 0 and captures the entire ring
        // up to the current tail. If the tail has somehow regressed
        // relative to the watermark (a new batch was started since the
        // last snapshot), `end.max(start)` guards against a negative
        // slice length — in that case the caller will just observe an
        // empty slice for that call (the test accessor resets the
        // watermark to 0 anyway on new-batch detection).
        let start: u64 = self.snapshot_event_ring_read;
        let event_ring_cap = main_event_ring.capacity() as u64;
        let end = event_ring_tail.min(event_ring_cap).max(start);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine_gpu::snapshot::kick_copy"),
            });
        self.snapshot_back
            .as_mut()
            .expect("snapshot_back lazy-inited above")
            .kick_copy(
                &mut encoder,
                agents_buf,
                agent_bytes,
                main_event_ring,
                start,
                end,
                self.latest_recorded_tick,
            );
        self.queue.submit(Some(encoder.finish()));
        // Advance the watermark. The batch events ring is append-only
        // within a batch, so this is a true watermark — the next
        // snapshot reads records in `[end, new_tail)`. Reset to 0 at
        // the top of each `step_batch` call.
        self.snapshot_event_ring_read = end;

        // 4. Swap front / back — next call takes the one we just
        //    filled, and fills the one we just took from.
        std::mem::swap(&mut self.snapshot_front, &mut self.snapshot_back);

        Ok(snap)
    }

    /// Test-only accessor for the snapshot event-ring watermark. Used
    /// by the Phase D6 end-to-end test to assert the observer's read
    /// pointer advances each call. Not part of the stable public API.
    #[doc(hidden)]
    pub fn snapshot_event_ring_read_for_test(&self) -> u64 {
        self.snapshot_event_ring_read
    }
}

#[cfg(feature = "gpu")]
impl SimBackend for GpuBackend {
    fn step<B: PolicyBackend>(
        &mut self,
        state: &mut SimState,
        scratch: &mut SimScratch,
        events: &mut EventRing,
        policy: &B,
        cascade: &CascadeRegistry,
    ) {
        // Task 197 — eliminate the dominant N=1000 cost (CPU mask build
        // + policy evaluate, 170-700 ms/tick per task 195's
        // `PhaseTimings`). Prior shape ran `engine::step::step_phases_
        // 1_to_3` on CPU and then ran the GPU mask + scoring kernels as
        // a post-tick diagnostic SIDECAR — pure duplicate work.
        //
        // New shape:
        //   1. Ensure cascade context (lazy DSL load + WGSL compile on
        //      first call).
        //   2. Run GPU mask + scoring kernels (formerly the post-tick
        //      sidecar). Output: one `ScoreOutput` per slot — the same
        //      per-agent argmax the CPU backend computes, but ~2 ms on
        //      GPU vs. ~170-700 ms on CPU at N=1000.
        //   3. Convert scoring outputs → `Vec<Action>` (CPU, O(N)) and
        //      deterministically shuffle with the same per-tick seed
        //      the CPU backend uses (task 197 exposes
        //      `engine::step::shuffle_actions_in_place` for this).
        //   4. Phase 4a: CPU `apply_actions` on the action list — emits
        //      this tick's seed events (AgentAttacked, AgentMoved,
        //      opportunity attacks, announce broadcasts, …). Byte-
        //      compatible with the CPU backend's event emission.
        //   5. Phase 4b: GPU cascade runs against the seed events.
        //   6. Commit GPU final slots back to `SimState`; push GPU-
        //      emitted events into the CPU event ring.
        //   7. Phase 4c: cold-state replay — walk seed + GPU-emitted
        //      events once on CPU, dispatching the 11 rules the GPU
        //      stubs (chronicles, transfer_gold, modify_standing,
        //      record_memory).
        //   8. Phase 5: CPU view-fold on the full this-tick slice so
        //      `state.views` stays in sync with `view_storage`.
        //   9. Phase 6: invariants + telemetry, increment tick.
        //
        // The `skip_scoring_sidecar` flag is now a no-op on the fast
        // path — the mask + scoring kernels ALWAYS run at the top of
        // the tick, because their output IS the action source.
        // `last_mask_bitmaps` / `last_scoring_outputs` are populated
        // as a byproduct of the new flow, so diagnostic callers still
        // see them regardless of the flag.
        //
        // On cascade init / dispatch failure the backend falls back to
        // the full CPU pipeline (mask + evaluate + apply + CPU cascade)
        // for THIS tick so state stays live; the per-tick diagnostic
        // surface still populates via a sidecar on the fallback path.

        let t_start = std::time::Instant::now();

        // Lazy cascade ctx init. If this fails, fall back to CPU
        // end-to-end — the tick still advances correctly, we just
        // don't exercise GPU cascade this time.
        if let Err(e) = self.ensure_cascade_initialized() {
            self.last_cascade_error = Some(format!("init: {e}"));
            self.last_cascade_iterations = None;
            engine::step::step(state, scratch, events, policy, cascade);
            // Run scoring / view mirror so the per-tick diagnostic
            // surface still populates.
            if !self.skip_scoring_sidecar {
                self.run_scoring_sidecar(state);
            }
            return;
        }

        if let Err(_e) = self.ensure_view_storage_cap(state.agent_cap()) {
            self.last_cascade_error = Some("view_storage resize failed".to_string());
            self.last_cascade_iterations = None;
            engine::step::step(state, scratch, events, policy, cascade);
            if !self.skip_scoring_sidecar {
                self.run_scoring_sidecar(state);
            }
            return;
        }

        self.last_phase_us = PhaseTimings::default();

        // Phase 1-2 (NEW): GPU mask + scoring dispatch at the TOP of
        // the tick. This is the work that used to run as the post-tick
        // sidecar; moving it up front lets us skip
        // `step_phases_1_to_3`'s CPU mask build + policy evaluate
        // (which at N=1000 dominates the tick, per task 195's phase
        // timings). Output: one `ScoreOutput` per agent slot carrying
        // `(chosen_action, chosen_target)` the scorer's argmax picked.
        //
        // On dispatch failure we fall back to the CPU pipeline for the
        // whole tick — the GPU scoring output is load-bearing for the
        // skipped CPU phases, so a half-GPU tick isn't recoverable.
        let t_ph13 = std::time::Instant::now();
        let bitmaps_and_scoring = match self
            .mask_kernel
            .run_and_readback(&self.device, &self.queue, state)
        {
            Ok(bitmaps) => match self.scoring_kernel.run_and_readback(
                &self.device,
                &self.queue,
                state,
                &self.mask_kernel,
                &bitmaps,
                &self.view_storage,
            ) {
                Ok(scores) => Some((bitmaps, scores)),
                Err(e) => {
                    eprintln!("engine_gpu: scoring dispatch failed, falling back to CPU: {e}");
                    None
                }
            },
            Err(e) => {
                eprintln!("engine_gpu: mask dispatch failed, falling back to CPU: {e}");
                None
            }
        };

        if bitmaps_and_scoring.is_none() {
            // Full CPU fallback: mask-build + evaluate + apply + CPU
            // cascade. Diagnostic fields stay empty for this tick.
            self.last_cascade_error =
                Some("mask/scoring dispatch failed; CPU fallback".to_string());
            self.last_cascade_iterations = None;
            self.last_mask_bitmaps.clear();
            self.last_scoring_outputs.clear();
            engine::step::step(state, scratch, events, policy, cascade);
            return;
        }

        let (bitmaps, scoring_outputs) = bitmaps_and_scoring.unwrap();

        // Cache for diagnostic callers. Done now rather than at tick
        // end because the sidecar is gone — the mask + scoring we just
        // ran IS the authoritative output for this tick.
        self.last_mask_bitmaps = bitmaps;
        self.last_scoring_outputs = scoring_outputs;

        self.last_phase_us.cpu_phases_1_3_us = t_ph13.elapsed().as_micros() as u64;

        // Phase 4a — GPU `apply_actions` + `movement` kernels. Task 200
        // retires the CPU bridge (`apply_scoring::scoring_outputs_to_
        // actions` + `engine::step::apply_actions`) by dispatching two
        // WGSL kernels that consume the scoring outputs in place:
        //
        //   * apply_actions_kernel — Attack damage + AgentAttacked /
        //     AgentDied emission. One thread per agent slot.
        //   * movement_kernel — MoveToward / Flee pos updates +
        //     AgentMoved / AgentFled emission. One thread per slot.
        //
        // Both kernels append events into a shared GPU event ring; we
        // drain the ring once after both dispatches to recover the seed
        // events the cascade consumes. Agent SoA is piped from kernel
        // to kernel (apply_actions' readback → movement's input) so
        // movement sees the post-apply hp/alive state.
        //
        // The CPU backend's Fisher-Yates-shuffled action order is
        // REPLACED by the GPU's workgroup-scheduling order. The drain
        // sorts by `(tick, kind, payload[0])` before pushing to the
        // CPU ring, which reinstates a deterministic push order that's
        // byte-comparable modulo the stable-sort tie-breaking.
        // Downstream parity tests use multiset equality on events, so
        // the push-order shift is benign.
        let t_apply = std::time::Instant::now();
        let events_before = events.total_pushed();

        if !self.run_gpu_apply_and_movement(state, events) {
            // Kernel failure: fall back to the full CPU pipeline for
            // this tick so state stays live. The GPU mask + scoring
            // outputs we already cached are discarded — the CPU
            // `engine::step::step` re-runs phases 1-4 end-to-end.
            self.last_cascade_error =
                Some("apply_actions/movement kernel dispatch failed; full CPU fallback".to_string());
            self.last_cascade_iterations = None;
            engine::step::step(state, scratch, events, policy, cascade);
            return;
        }
        let events_after_apply = events.total_pushed();

        self.last_phase_us.cpu_apply_actions_us = t_apply.elapsed().as_micros() as u64;

        // Collect the seed events (apply_actions + movement pushed)
        // in push order. These feed both the GPU cascade's input
        // batch AND the cold-state replay's iteration.
        let seed_events: Vec<Event> = (events_before..events_after_apply)
            .filter_map(|i| events.get_pushed(i))
            .collect();
        let initial_records = cascade::pack_initial_events(&seed_events);

        // Phase 4b — GPU cascade.
        let cascade_ctx = self
            .cascade_ctx
            .as_mut()
            .expect("cascade_ctx ensured above");
        let emit_ctx = dsl_compiler::emit_physics_wgsl::EmitContext {
            events: &cascade_ctx.comp.events,
            event_tags: &cascade_ctx.comp.event_tags,
        };

        // `view_storage` is NOT reset between ticks. It accumulates
        // across the whole session — fold kernels update cells in place
        // (CAS saturating-add for pair_map scalars, max-decay for the
        // decay cells) so scoring on tick N reads the union of every
        // event ever folded. The `cascade_parity.rs` harness resets
        // because each test seeds a fresh isolated state; here the
        // backend tick loop mirrors CPU `state.views` semantics, which
        // never clears either.

        let t_cascade = std::time::Instant::now();
        let cascade_out = cascade::run_cascade(
            &self.device,
            &self.queue,
            state,
            &mut cascade_ctx.physics,
            &mut self.view_storage,
            &mut cascade_ctx.spatial,
            &cascade_ctx.abilities,
            &initial_records,
            cascade::DEFAULT_KIN_RADIUS,
            &emit_ctx,
        );
        self.last_phase_us.gpu_cascade_us = t_cascade.elapsed().as_micros() as u64;

        // Fold the seed events into view_storage too — the cascade
        // driver only folds what ITS kernel emits; the apply-phase
        // events (AgentAttacked, AgentMoved, etc.) also carry view
        // updates (my_enemies / threat_level / engaged_with) that
        // scoring on the next tick needs to see.
        let t_seed_fold = std::time::Instant::now();
        if let Err(e) = cascade::fold_iteration_events(
            &self.device,
            &self.queue,
            &mut self.view_storage,
            &initial_records,
        ) {
            eprintln!("engine_gpu: seed-fold failed: {e}");
        }
        self.last_phase_us.gpu_seed_fold_us = t_seed_fold.elapsed().as_micros() as u64;

        match cascade_out {
            Ok(out) => {
                self.last_cascade_iterations = Some(out.iterations);
                self.last_cascade_error = None;

                // Commit GPU-mutated agent SoA to SimState.
                cascade::apply_final_slots(state, &out.final_agent_slots);

                // Drain GPU-emitted events into the CPU ring.
                cascade::events_into_ring(&out.all_emitted_events, events);

                // Collect GPU-emitted events we just pushed so the
                // cold-state replay walks them in push order.
                let gpu_events: Vec<Event> = out
                    .all_emitted_events
                    .iter()
                    .filter_map(crate::event_ring::unpack_record)
                    .collect();

                // Phase 4c — cold-state replay on (seed + GPU-emitted)
                // events. Seeds first, matching the CPU cascade's
                // event-order semantics (apply_actions pushes before
                // cascade runs).
                //
                // Chronicle rules push ChronicleEntry events; those
                // re-enter the ring and don't need further replay
                // (ChronicleEntry is non-replayable and nothing
                // dispatches on it).
                let t_cold = std::time::Instant::now();
                cascade::cold_state_replay(state, events, &seed_events);
                cascade::cold_state_replay(state, events, &gpu_events);
                self.last_phase_us.cpu_cold_state_us = t_cold.elapsed().as_micros() as u64;
            }
            Err(e) => {
                // Cascade dispatch failed mid-tick. Fall back to the
                // CPU cascade on the events already in the ring so
                // state stays correct. The fallback is idempotent —
                // cascade.run_fixed_point_tel resumes from
                // events.dispatched() so events we've already
                // dispatched (none yet, this is the first cascade
                // call this tick) aren't redispatched.
                self.last_cascade_error = Some(format!("dispatch: {e}"));
                self.last_cascade_iterations = None;
                eprintln!("engine_gpu: GPU cascade failed, falling back to CPU cascade: {e}");
                cascade.run_fixed_point_tel(state, events, &engine::telemetry::NullSink);
            }
        }

        let events_emitted = events.total_pushed().saturating_sub(events_before);

        // Phase 5 — CPU view-fold on the full this-tick event slice.
        // The GPU view_storage was folded above (per iteration + seed
        // events); this keeps the CPU `state.views` registry in sync
        // so any future CPU-path dispatch (fallback, or a test that
        // pokes `state.views`) sees the same values. Cheap for the
        // 50-tick parity test; O(this-tick events).
        let t_cpu_fold_all = std::time::Instant::now();
        state.views.fold_all(events, events_before, state.tick);
        self.last_phase_us.cpu_view_fold_all_us =
            t_cpu_fold_all.elapsed().as_micros() as u64;

        // Phase 6 — invariants + telemetry + tick++. Reuses the CPU
        // helper so every backend reports the same telemetry metric
        // shape.
        let t_final = std::time::Instant::now();
        let empty_invariants = engine::invariant::InvariantRegistry::new();
        engine::step::finalize_tick(
            state,
            scratch,
            events,
            &empty_invariants,
            &engine::telemetry::NullSink,
            t_start,
            events_emitted,
        );
        self.last_phase_us.cpu_finalize_us = t_final.elapsed().as_micros() as u64;

        // Task 197: the mask + scoring kernels at the top of the tick
        // ARE the authoritative action source — diagnostic fields
        // (`last_mask_bitmaps`, `last_scoring_outputs`) are already
        // populated from that dispatch. However, the
        // `parity_with_cpu.rs` test suite compares GPU bitmaps against
        // a CPU reference computed on POST-step state (i.e. the state
        // `step()` leaves behind), so we retain a post-step sidecar
        // dispatch that overwrites those fields for tests that need
        // byte-parity against a post-step CPU bitmap.
        //
        // Perf harnesses opt out via `set_skip_scoring_sidecar(true)`.
        // Default-off (sidecar runs) preserves every existing parity
        // test's assertion surface.
        if !self.skip_scoring_sidecar {
            let t_side = std::time::Instant::now();
            self.run_scoring_sidecar(state);
            self.last_phase_us.gpu_sidecar_us = t_side.elapsed().as_micros() as u64;
        } else {
            self.last_phase_us.gpu_sidecar_us = 0;
        }
    }
}

/// Internal helper: run the mask kernel + scoring kernel against the
/// current state and cache their outputs on the backend. Used at the
/// end of `step` and on the CPU-fallback paths so diagnostic fields
/// populate identically regardless of which cascade path ran.
#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Task 200 — dispatch the GPU `apply_actions` + `movement` kernels
    /// against the cached scoring outputs, drain their shared event
    /// ring into the CPU event ring, and commit the mutated agent SoA
    /// onto `state`. Returns `false` iff a kernel dispatch failed —
    /// caller is expected to fall back to the CPU `apply_actions` path
    /// for this tick.
    ///
    /// This is the GPU replacement for the Task 197 bridge:
    ///   * `apply_scoring::scoring_outputs_to_actions` (CPU O(N) adapt)
    ///   * `engine::step::shuffle_actions_in_place` (CPU Fisher-Yates)
    ///   * `engine::step::apply_actions` (CPU apply loop with event emits)
    ///
    /// The kernels consume `self.last_scoring_outputs` (populated at
    /// the top of `step` by the scoring kernel), so the caller must
    /// invoke this AFTER running mask + scoring and BEFORE the cascade
    /// dispatch picks up the seed events.
    fn run_gpu_apply_and_movement(
        &mut self,
        state: &mut SimState,
        events: &mut engine::event::EventRing,
    ) -> bool {
        let cascade_ctx = match self.cascade_ctx.as_mut() {
            Some(c) => c,
            None => return false,
        };
        let scoring_outputs = &self.last_scoring_outputs;
        if scoring_outputs.is_empty() {
            return false;
        }

        let agent_slots_in = crate::physics::pack_agent_slots(state);
        cascade_ctx.apply_event_ring.reset(&self.queue);

        let apply_cfg = crate::apply_actions::cfg_from_state(state);
        let slots_after_apply = match cascade_ctx.apply_actions.run_and_readback(
            &self.device,
            &self.queue,
            &agent_slots_in,
            scoring_outputs,
            apply_cfg,
            &cascade_ctx.apply_event_ring,
        ) {
            Ok(slots) => slots,
            Err(e) => {
                eprintln!(
                    "engine_gpu: apply_actions kernel failed ({e}); \
                     falling back to CPU apply for this tick"
                );
                return false;
            }
        };

        let movement_cfg = crate::movement::cfg_from_state(state);
        let slots_final = match cascade_ctx.movement.run_and_readback(
            &self.device,
            &self.queue,
            &slots_after_apply,
            scoring_outputs,
            movement_cfg,
            &cascade_ctx.apply_event_ring,
        ) {
            Ok(slots) => slots,
            Err(e) => {
                eprintln!(
                    "engine_gpu: movement kernel failed ({e}); committing \
                     apply_actions result without movement this tick"
                );
                // apply_actions already mutated agent hp/alive on GPU;
                // without movement the pos stays at tick-N value. Next
                // tick re-dispatches with the updated alive set.
                slots_after_apply
            }
        };

        // Drain apply event ring into the CPU events ring. The drain
        // sorts records by `(tick, kind, payload[0])` before pushing,
        // reinstating a deterministic order independent of GPU thread
        // scheduling.
        if let Err(e) = cascade_ctx
            .apply_event_ring
            .drain(&self.device, &self.queue, events)
        {
            eprintln!(
                "engine_gpu: apply_event_ring drain failed ({e}); \
                 seed events may be lost this tick"
            );
        }

        // Commit mutated agent SoA onto SimState. apply_actions touched
        // hp/alive; movement touched pos. Shield/stun/slow/cooldown/
        // engaged_with are untouched by both kernels, so those field
        // writes in `unpack_agent_slots` are no-ops (identity writes of
        // the pre-step values we packed in).
        crate::physics::unpack_agent_slots(state, &slots_final);
        true
    }

    fn run_scoring_sidecar(&mut self, state: &SimState) {
        match self.mask_kernel.run_and_readback(&self.device, &self.queue, state) {
            Ok(bitmaps) => {
                match self.scoring_kernel.run_and_readback(
                    &self.device,
                    &self.queue,
                    state,
                    &self.mask_kernel,
                    &bitmaps,
                    &self.view_storage,
                ) {
                    Ok(scores) => self.last_scoring_outputs = scores,
                    Err(_e) => self.last_scoring_outputs.clear(),
                }
                self.last_mask_bitmaps = bitmaps;
            }
            Err(_e) => {
                self.last_mask_bitmaps.clear();
                self.last_scoring_outputs.clear();
            }
        }
    }
}

// Task 193 (Phase 6g) retired the previous Piece 1 `mirror_cpu_views_
// into_gpu_view_storage` bridge. The GPU cascade now folds both seed
// events (apply_actions' pushes) and cascade-emitted events into
// `view_storage` directly inside `step`; the CPU `state.views`
// registry is kept in sync by a parallel CPU `fold_all` call, but
// scoring reads from the GPU-owned `view_storage` atomics without
// needing a cross-device copy.
