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
            backend_label,
            last_mask_bitmaps: Vec::new(),
            last_scoring_outputs: Vec::new(),
        })
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
        // Phase 2: still forward the CPU step — the fused mask kernel
        // runs alongside and every output is compared against CPU by
        // the parity harness. Phase 3 feeds the GPU bitmaps back into
        // the argmax scoring path.
        //
        // Phase 6d: view storage is now the scoring kernel's read
        // source. We keep the CPU step as authoritative for fold state
        // and mirror the post-step `SimState::views` into
        // `view_storage` via `mirror_cpu_views_into_gpu_view_storage`
        // before dispatching the GPU scoring kernel. That preserves
        // the Phase 6c byte-exact parity contract (same view values as
        // CPU scoring sees) while proving out the direct-read binding
        // layout end-to-end. Pieces 2/3 replace this mirror with GPU
        // fold dispatch; until then scoring sees fold results via a
        // one-way CPU-to-GPU copy rather than the CPU-uploaded
        // DecayCell buffers the Phase 6c path used.
        engine::step::step(state, scratch, events, policy, cascade);

        if let Err(_e) = self.ensure_view_storage_cap(state.agent_cap()) {
            self.last_mask_bitmaps.clear();
            self.last_scoring_outputs.clear();
            return;
        }
        // Seed view_storage's atomic buffers with the CPU view registry's
        // post-step contents. Temporary bridge for Piece 1 parity — a
        // later piece replaces this with native GPU folds driven by
        // the CPU-pushed event slice.
        mirror_cpu_views_into_gpu_view_storage(
            state,
            &self.queue,
            &mut self.view_storage,
        );

        // Run the fused mask kernel on the post-step state. Capture
        // every per-mask bitmap on the backend so callers (parity
        // test) can read them out without re-running the kernel.
        // Errors are not fatal — if the dispatch failed we clear the
        // bitmap vec, which the parity harness surfaces as an
        // assertion failure ("GPU bitmaps empty after step").
        match self.mask_kernel.run_and_readback(&self.device, &self.queue, state) {
            Ok(bitmaps) => {
                // Phase 3: feed the GPU bitmaps into the scoring kernel.
                // The CPU has already produced its own actions during
                // the `step()` call above; this pass exists for parity
                // assertion only at Phase 3 and becomes the engine's
                // actual decision source in Phase 6.
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

/// Bridge: copy the post-step CPU `state.views` contents into
/// `view_storage`'s GPU-resident atomic buffers. Used by
/// `GpuBackend::step` so Piece 1's byte-exact parity harness can
/// exercise scoring's direct-read path against real fold state without
/// Pieces 2/3 (GPU physics + fold dispatch) in place yet.
///
/// Writes are `queue.write_buffer` — the WGSL-side reads issue
/// `atomicLoad`, which is compatible with a CPU-side plain write as
/// long as no concurrent GPU writer touches the same buffer. Inside
/// `GpuBackend::step` the fold kernels haven't been dispatched yet, so
/// this is safe.
#[cfg(feature = "gpu")]
fn mirror_cpu_views_into_gpu_view_storage(
    state: &engine::state::SimState,
    queue: &wgpu::Queue,
    view_storage: &mut view_storage::ViewStorage,
) {
    use engine::ids::AgentId;

    let agent_cap = view_storage.agent_cap();
    let n = agent_cap as usize;

    // engaged_with — slot_map<u32>, sized for agent_cap slots.
    if let Some(buf) = view_storage.primary_buffer("engaged_with") {
        let mut slots = vec![0u32; n];
        for slot in 0..n.min(state.agent_cap() as usize) {
            let id = match AgentId::new(slot as u32 + 1) {
                Some(id) => id,
                None => continue,
            };
            if let Some(partner) = state.views.engaged_with.get(id) {
                slots[slot] = partner.raw();
            }
        }
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&slots));
    }

    let live_n = state.agent_cap() as usize;

    // my_enemies — pair_map<atomic<u32>>, cells hold f32 bits.
    if let Some(buf) = view_storage.primary_buffer("my_enemies") {
        let mut cells = vec![0u32; n * n];
        for observer_slot in 0..live_n.min(n) {
            let observer = match AgentId::new(observer_slot as u32 + 1) {
                Some(id) => id,
                None => continue,
            };
            for attacker_slot in 0..live_n.min(n) {
                let attacker = match AgentId::new(attacker_slot as u32 + 1) {
                    Some(id) => id,
                    None => continue,
                };
                let v = state.views.my_enemies.get(observer, attacker);
                cells[observer_slot * n + attacker_slot] = v.to_bits();
            }
        }
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&cells));
    }

    // Decay views — values + anchors. We stamp anchor = state.tick and
    // value = already-decayed, matching the trick the Phase 6c CPU
    // mirror used: read-tick will equal stamp-tick so dt=0 →
    // the read snippet returns the stamped value verbatim. Byte-exact
    // with the CPU scoring path (which applies the same dt=0 short
    // circuit in its predicate evaluator).
    //
    // One closure per decay view because their concrete types differ
    // (generated per-view structs, no shared trait).
    mirror_decay_view(
        view_storage, queue, live_n, n, state.tick, "threat_level",
        |a, b| state.views.threat_level.get(a, b, state.tick),
    );
    mirror_decay_view(
        view_storage, queue, live_n, n, state.tick, "kin_fear",
        |a, b| state.views.kin_fear.get(a, b, state.tick),
    );
    mirror_decay_view(
        view_storage, queue, live_n, n, state.tick, "pack_focus",
        |a, b| state.views.pack_focus.get(a, b, state.tick),
    );
    mirror_decay_view(
        view_storage, queue, live_n, n, state.tick, "rally_boost",
        |a, b| state.views.rally_boost.get(a, b, state.tick),
    );
}

#[cfg(feature = "gpu")]
fn mirror_decay_view<F>(
    view_storage: &view_storage::ViewStorage,
    queue: &wgpu::Queue,
    live_n: usize,
    n: usize,
    tick: u32,
    name: &str,
    mut getter: F,
) where
    F: FnMut(engine::ids::AgentId, engine::ids::AgentId) -> f32,
{
    use engine::ids::AgentId;
    let primary = match view_storage.primary_buffer(name) {
        Some(b) => b,
        None => return,
    };
    let anchor = match view_storage.anchor_buffer(name) {
        Some(b) => b,
        None => return,
    };
    let mut values = vec![0u32; n * n];
    let mut anchors = vec![0u32; n * n];
    for observer_slot in 0..live_n.min(n) {
        let observer = match AgentId::new(observer_slot as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        for attacker_slot in 0..live_n.min(n) {
            let attacker = match AgentId::new(attacker_slot as u32 + 1) {
                Some(id) => id,
                None => continue,
            };
            let v = getter(observer, attacker);
            values[observer_slot * n + attacker_slot] = v.to_bits();
            anchors[observer_slot * n + attacker_slot] = tick;
        }
    }
    queue.write_buffer(primary, 0, bytemuck::cast_slice(&values));
    queue.write_buffer(anchor, 0, bytemuck::cast_slice(&anchors));
}
