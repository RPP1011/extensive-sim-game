//! GPU backend for the engine — Phase 1 (Attack mask E2E).
//!
//! See `docs/plans/gpu_megakernel_plan.md`. Phase 0 shipped a pure
//! CPU-forwarding stub. Phase 1 adds:
//!
//!   * Real `wgpu::Device` + `wgpu::Queue` handles, created lazily on
//!     `GpuBackend::new()` via `pollster::block_on` (kept sync —
//!     callers don't need to be async).
//!   * A WGSL compute pipeline for the Attack mask, emitted at
//!     construction time by `dsl_compiler::emit_mask_wgsl` and parsed
//!     through `naga` at runtime.
//!   * Per-tick storage-buffer uploads, dispatch, and readback of the
//!     Attack mask bitmap (one bit per agent slot).
//!
//! ## Phase 1 scope — what the backend actually does each tick
//!
//! Per the plan's "strongly prefer the alternative approach" guidance,
//! `GpuBackend::step` still forwards to `engine::step::step` (CPU
//! kernel) and ADDITIONALLY runs the Attack mask on GPU. The GPU result
//! is not yet spliced into the engine's scratch buffers — it lives on
//! the backend and is exposed via
//! [`GpuBackend::last_attack_mask_bitmap`], which the parity harness
//! calls and compares against a CPU reference bitmap computed by
//! [`attack_mask::cpu_attack_mask_bitmap`]. That's the Phase 1 proof
//! point: WGSL emit → naga parse → dispatch → readback all work, and
//! the kernel produces byte-identical output to the CPU attack mask.
//!
//! Phase 2 swaps the GPU result into `SimScratch.mask` directly, at
//! which point the CPU mask-build for Attack becomes dead code and can
//! be deleted.
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
pub mod attack_mask;

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
    attack_mask: attack_mask::AttackMaskKernel,
    /// Name of the backend wgpu selected (Vulkan / Metal / DX12 / GL /
    /// BrowserWebGpu / llvmpipe (GL/Vulkan software)). Captured at
    /// construction so tests can report which path they exercised.
    backend_label: String,
    /// GPU-computed Attack mask bitmap from the most recent `step`, in
    /// word order (`bit i of word i/32` set iff slot `i`'s agent passed
    /// the Attack predicate). Empty before the first `step`.
    last_attack_bitmap: Vec<u32>,
}

/// Errors surfaced by `GpuBackend::new` at Phase 1.
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
    Kernel(attack_mask::KernelError),
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
impl From<attack_mask::KernelError> for GpuInitError {
    fn from(e: attack_mask::KernelError) -> Self {
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine_gpu::device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::RequestDevice(format!("{e}")))?;

        let attack_mask = attack_mask::AttackMaskKernel::new(&device)?;

        Ok(Self {
            device,
            queue,
            attack_mask,
            backend_label,
            last_attack_bitmap: Vec::new(),
        })
    }

    /// Human-readable name of the wgpu backend the device is running
    /// on — one of `Vulkan`, `Metal`, `Dx12`, `Gl`, `BrowserWebGpu`, or
    /// `Empty`. Captured at init so tests can log which path they
    /// actually exercised (useful when Linux falls back to `Gl`/LLVMpipe).
    pub fn backend_label(&self) -> &str {
        &self.backend_label
    }

    /// Packed u32 bitmap from the most recent `step`. Bit `i` of word
    /// `i/32` is `1` iff slot `i`'s agent passed the Attack-mask
    /// predicate against at least one target. Empty before the first
    /// `step` — check `.is_empty()` before indexing.
    pub fn last_attack_mask_bitmap(&self) -> &[u32] {
        &self.last_attack_bitmap
    }

    /// Run the Attack-mask kernel on the current `state` without
    /// touching the CPU step — used by the parity test's
    /// "compare GPU to CPU at every tick" mode. Returns the packed
    /// bitmap; caller compares against `attack_mask::cpu_attack_mask_bitmap`.
    pub fn verify_attack_mask_on_gpu(
        &mut self,
        state: &SimState,
    ) -> Result<Vec<u32>, attack_mask::KernelError> {
        self.attack_mask.run_and_readback(&self.device, &self.queue, state)
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
        // Phase 1: still forward the CPU step — the GPU kernel runs
        // alongside and its output is validated against CPU by the
        // parity harness. Phase 2 replaces the Attack mask portion of
        // CPU mask-build with the GPU bitmap.
        engine::step::step(state, scratch, events, policy, cascade);

        // Run the Attack mask on GPU using the post-step state. Capture
        // the result on the backend so callers (parity test) can read
        // it out without re-running the kernel. Errors are not fatal —
        // if the dispatch failed we clear the bitmap, which the parity
        // harness surfaces as an assertion failure ("GPU bitmap empty
        // after step").
        match self.attack_mask.run_and_readback(&self.device, &self.queue, state) {
            Ok(bits) => self.last_attack_bitmap = bits,
            Err(_e) => {
                self.last_attack_bitmap.clear();
            }
        }
    }
}
