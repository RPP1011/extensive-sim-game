//! GPU backend for the engine — Phase 0 stub.
//!
//! See `docs/plans/gpu_megakernel_plan.md`. This crate exists so later phases
//! can add `wgpu` / `naga` dependencies, buffer pools, and compute shaders
//! behind a single façade (`GpuBackend`) without touching `engine`'s public
//! surface or the xtask CLI.
//!
//! ## Phase 0 invariant
//!
//! [`GpuBackend::step`] forwards to [`engine::step::step`] unchanged. This is
//! deliberate: the parity harness in `tests/parity_with_cpu.rs` runs N ticks
//! through `CpuBackend` and N ticks through `GpuBackend` against cloned
//! fixtures and asserts byte-equal event logs. In Phase 0 the assertion is
//! trivially true; once Phase 1 replaces the body with real GPU dispatch, the
//! same harness catches any divergence.
//!
//! The CPU path stays the reference implementation and the default for every
//! binary, library consumer, and test that doesn't explicitly opt into GPU.

use engine::{
    backend::SimBackend,
    cascade::CascadeRegistry,
    event::EventRing,
    policy::PolicyBackend,
    state::SimState,
    step::SimScratch,
};

/// Phase 0 stub backend.
///
/// Constructed via `GpuBackend::default()` (or `GpuBackend::new()`); holds no
/// state today. Future phases will park wgpu `Device`/`Queue` handles, the
/// buffer pool, and compiled WGSL shader modules on this struct — the trait
/// impl below is the only surface callers touch, so adding those fields is a
/// non-breaking change.
///
/// Implements [`engine::backend::SimBackend`] by forwarding to the CPU step
/// kernel. Swapping this for a GPU dispatch loop is an internal detail
/// contained inside `impl SimBackend for GpuBackend`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuBackend {
    // Phase 1+ will add `device: wgpu::Device`, `queue: wgpu::Queue`,
    // shader module handles, and persistent buffers here. The Default impl
    // will change to spin up the wgpu instance lazily; for Phase 0 the type
    // is zero-sized and Default is a no-op.
    _phase0_marker: (),
}

impl GpuBackend {
    /// Construct a new Phase 0 stub backend. Identical to
    /// `GpuBackend::default()` — exists only so call sites can use
    /// `GpuBackend::new()` without importing `Default`.
    #[inline]
    pub fn new() -> Self { Self::default() }
}

impl SimBackend for GpuBackend {
    #[inline]
    fn step<B: PolicyBackend>(
        &mut self,
        state:   &mut SimState,
        scratch: &mut SimScratch,
        events:  &mut EventRing,
        policy:  &B,
        cascade: &CascadeRegistry,
    ) {
        // Phase 0: pure forward. Phase 1 replaces the body with the
        // dispatch cascade; the signature — and therefore every caller — is
        // frozen from here on.
        engine::step::step(state, scratch, events, policy, cascade);
    }
}
