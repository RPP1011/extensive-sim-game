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

/// Phase 4 — per-view GPU storage + fold kernels. Not wired into the
/// backend's tick loop yet; the follow-up integration task swaps
/// scoring's stub views for reads against this module.
#[cfg(feature = "gpu")]
pub mod view_storage;

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
}

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
}

#[cfg(feature = "gpu")]
impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuInitError::NoAdapter => write!(f, "no compatible GPU adapter"),
            GpuInitError::RequestDevice(s) => write!(f, "request_device: {s}"),
            GpuInitError::Kernel(e) => write!(f, "kernel init: {e}"),
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

        Ok(Self {
            device,
            queue,
            mask_kernel,
            backend_label,
            last_mask_bitmaps: Vec::new(),
        })
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
        engine::step::step(state, scratch, events, policy, cascade);

        // Run the fused mask kernel on the post-step state. Capture
        // every per-mask bitmap on the backend so callers (parity
        // test) can read them out without re-running the kernel.
        // Errors are not fatal — if the dispatch failed we clear the
        // bitmap vec, which the parity harness surfaces as an
        // assertion failure ("GPU bitmaps empty after step").
        match self.mask_kernel.run_and_readback(&self.device, &self.queue, state) {
            Ok(bitmaps) => self.last_mask_bitmaps = bitmaps,
            Err(_e) => {
                self.last_mask_bitmaps.clear();
            }
        }
    }
}
