//! Simulation backend abstraction (GPU-megakernel plan, Phase 0).
//!
//! The engine has historically had exactly one tick driver: `step::step` (and
//! its richer sibling `step::step_full`). Task #159 — the GPU megakernel port
//! — needs a second tick driver that dispatches ~3-5 WGSL sub-kernels instead
//! of running the CPU kernel. Both drivers consume the same inputs
//! (`SimState`, `SimScratch`, `EventRing`, `PolicyBackend`, `CascadeRegistry`)
//! and produce the same observable outputs (mutated state + appended events),
//! so this module defines a narrow trait, [`SimBackend`], that both can
//! implement. The xtask and tests can then swap CPU ↔ GPU without touching the
//! tick loop itself.
//!
//! Phase 0 (this commit) ships only the CPU wrapper and the trait. The GPU
//! side lives in `crates/engine_gpu/` and initially also forwards to the CPU
//! kernel so the parity harness compares the CPU path against itself — once
//! real GPU work lands the harness catches divergence.
//!
//! ## Trait shape & genericity
//!
//! `step::step` is generic over `B: PolicyBackend` (`B` is `Sized`). To avoid
//! modifying the step kernel — which is explicitly out of scope for Phase 0
//! (see `docs/plans/gpu_megakernel_plan.md`) — the trait method is also
//! generic over `B`. This keeps the backend call a straight forward, at the
//! cost of `dyn SimBackend` not being object-safe. Callers hold concrete
//! backend types (`CpuBackend` directly, or `engine_gpu::GpuBackend` behind a
//! `cfg(feature = "gpu")` branch), so object-safety isn't needed.
//!
//! Later phases that do real GPU dispatch won't change this signature; the
//! `B: PolicyBackend` bound is consumed inside the kernel only, and the GPU
//! backend reads the mask + target_mask from `state`/`scratch` the same way
//! the CPU one does.

use crate::cascade::CascadeRegistry;
use crate::event::EventRing;
use crate::policy::PolicyBackend;
use crate::state::SimState;
use crate::step::SimScratch;

/// A tick driver. Implementations must advance `state.tick` by exactly one
/// and leave the per-tick scratch buffers (`scratch.mask`, `scratch.actions`,
/// `scratch.shuffle_idx`, `scratch.target_mask`) populated for the tick that
/// just ran — matching `step::step`'s contract. Backends consume
/// `PolicyBackend` by reference because action evaluation doesn't mutate the
/// policy; the only mutable surfaces are `state`, `scratch`, and `events`.
///
/// The method is generic over `B: PolicyBackend` to match `step::step`'s
/// signature verbatim; this is not a dyn-dispatch trait.
pub trait SimBackend {
    fn step<B: PolicyBackend>(
        &mut self,
        state:   &mut SimState,
        scratch: &mut SimScratch,
        events:  &mut EventRing,
        policy:  &B,
        cascade: &CascadeRegistry,
    );
}

/// The default CPU tick driver — a thin wrapper around [`crate::step::step`]
/// that threads the same five arguments through. Stateless: holds no per-tick
/// buffers (those live on `SimScratch`), no GPU handles, nothing. Cheap to
/// construct ad-hoc; feel free to `CpuBackend::default()` inside a loop.
///
/// Mirrors the Phase 0 contract in `docs/plans/gpu_megakernel_plan.md`: the
/// CPU path remains the default and reference implementation; the GPU backend
/// asserts byte-parity against it.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl SimBackend for CpuBackend {
    #[inline]
    fn step<B: PolicyBackend>(
        &mut self,
        state:   &mut SimState,
        scratch: &mut SimScratch,
        events:  &mut EventRing,
        policy:  &B,
        cascade: &CascadeRegistry,
    ) {
        crate::step::step(state, scratch, events, policy, cascade);
    }
}
