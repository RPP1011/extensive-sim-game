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
    /// Default is `false` so existing tests that assert on
    /// `last_mask_bitmaps` / `last_scoring_outputs` keep working. Perf
    /// harnesses and batch callers flip this to `true` — the sidecar
    /// duplicates a full mask+scoring dispatch (~3-10 ms at N=1000)
    /// that the `SimBackend::step` contract doesn't require.
    skip_scoring_sidecar: bool,
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
            skip_scoring_sidecar: false,
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
        // Phase 6g — GPU-authoritative step. Pieces 1–3 (tasks 190–192)
        // landed on `world-sim-bench`; this is Piece 4 of the task-190
        // recipe, wiring the cascade into the backend's tick loop.
        //
        // Shape:
        //   1. Ensure cascade context (lazy DSL load + WGSL compile on
        //      first call).
        //   2. Phases 1–3: run CPU mask-build / evaluate / shuffle via
        //      `step::step_phases_1_to_3` (same call the CPU backend
        //      uses — byte-exact same actions).
        //   3. Phase 4a: run CPU `apply_actions` to seed the event ring
        //      with this tick's root-cause events.
        //   4. Phase 4b: GPU cascade runs against those seed events,
        //      mutating its agent SoA + emitting follow-on events.
        //   5. Commit GPU final slots back to `SimState` via
        //      `apply_final_slots`; push GPU-emitted events into the
        //      CPU event ring via `events_into_ring`.
        //   6. Phase 4c: cold-state replay — walk the GPU-emitted +
        //      seed events once on CPU, dispatching the 11 rules the
        //      GPU stubs (chronicles, transfer_gold, modify_standing,
        //      record_memory).
        //   7. Phase 5: CPU view-fold (same as step_full).
        //   8. Phase 6: invariants + telemetry, increment tick.
        //
        // On cascade init / dispatch failure the backend falls back to
        // the CPU cascade for THIS tick so state stays live. That
        // keeps long-running sessions resilient; the parity test's
        // byte-exact assertion catches silent divergence via the
        // CPU-vs-GPU state fingerprint.

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

        // Phases 1–3: CPU mask build, evaluate, shuffle.
        engine::step::step_phases_1_to_3(state, scratch, policy);

        // Phase 4a — apply actions on CPU, emitting this tick's seed
        // events. `events_before` snapshots the ring's total_pushed
        // cursor so we can locate just this tick's additions for the
        // GPU cascade's `initial_events` + the cold-state replay.
        let events_before = events.total_pushed();
        engine::step::apply_actions(state, scratch, events);
        let events_after_apply = events.total_pushed();

        // Collect the seed events (apply_actions' pushes this tick) in
        // push order. These feed both the GPU cascade's input batch
        // AND the cold-state replay's iteration.
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

        // Fold the seed events into view_storage too — the cascade
        // driver only folds what ITS kernel emits; the apply-phase
        // events (AgentAttacked, AgentMoved, etc.) also carry view
        // updates (my_enemies / threat_level / engaged_with) that
        // scoring on the next tick needs to see.
        if let Err(e) = cascade::fold_iteration_events(
            &self.device,
            &self.queue,
            &mut self.view_storage,
            &initial_records,
        ) {
            eprintln!("engine_gpu: seed-fold failed: {e}");
        }

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
                cascade::cold_state_replay(state, events, &seed_events);
                cascade::cold_state_replay(state, events, &gpu_events);
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
        state.views.fold_all(events, events_before, state.tick);

        // Phase 6 — invariants + telemetry + tick++. Reuses the CPU
        // helper so every backend reports the same telemetry metric
        // shape.
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

        // Sidecar: run the scoring + mask kernels against the new
        // post-step state so `last_scoring_outputs` / `last_mask_bitmaps`
        // continue to populate for diagnostic callers (e.g. the
        // existing `gpu_backend_matches_cpu_on_canonical_fixture`
        // parity test, which asserts per-mask GPU-vs-CPU bitmaps).
        //
        // Perf harnesses opt out via `set_skip_scoring_sidecar(true)`
        // — on a warm N=1000 workload the sidecar is ~3-10 ms of pure
        // duplicate work (the backend doesn't consume its own scoring
        // output; `step_phases_1_to_3` already picked the CPU action
        // set before the GPU cascade ran).
        if !self.skip_scoring_sidecar {
            self.run_scoring_sidecar(state);
        }
    }
}

/// Internal helper: run the mask kernel + scoring kernel against the
/// current state and cache their outputs on the backend. Used at the
/// end of `step` and on the CPU-fallback paths so diagnostic fields
/// populate identically regardless of which cascade path ran.
#[cfg(feature = "gpu")]
impl GpuBackend {
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
