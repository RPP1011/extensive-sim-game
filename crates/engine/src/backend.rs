//! Simulation backend abstraction.
//!
//! Defines [`ComputeBackend`] — the narrow trait both CPU and GPU tick drivers
//! implement. Concrete implementations (`SerialBackend` for CPU) live in
//! per-fixture runtime crates; the real tick driver is `engine_rules::step::step`.
//!
//! ## Trait shape & genericity
//!
//! The trait method is generic over `B: PolicyBackend` to avoid modifying the
//! step kernel signature (see `docs/plans/gpu_megakernel_plan.md`). This means
//! `dyn ComputeBackend` is not object-safe — callers hold concrete backend types.

use crate::cascade::CascadeRegistry;
use crate::event::{EventLike, EventRing};
use crate::mask::{MaskBuffer, MicroKind};
use crate::policy::PolicyBackend;
use crate::scratch::SimScratch;
use crate::state::SimState;

/// A tick driver. Implementations must advance `state.tick` by exactly one
/// and leave the per-tick scratch buffers (`scratch.mask`, `scratch.actions`,
/// `scratch.shuffle_idx`, `scratch.target_mask`) populated for the tick that
/// just ran. Backends consume `PolicyBackend` by reference because action
/// evaluation doesn't mutate the policy.
///
/// The method is generic over `B: PolicyBackend`; this is not a dyn-dispatch
/// trait.
pub trait ComputeBackend {
    /// The event type this backend drives (e.g. `engine_data::events::Event`).
    type Event: EventLike;
    /// The views type threaded through cascade handlers
    /// (e.g. `engine_rules::ViewRegistry`).
    type Views;

    fn step<B: PolicyBackend>(
        &mut self,
        state:   &mut SimState,
        scratch: &mut SimScratch,
        events:  &mut EventRing<Self::Event>,
        views:   &mut Self::Views,
        policy:  &B,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
    );

    /// Reset every bit in the mask buffer. SerialBackend implementation
    /// is `buf.reset()`; GpuBackend dispatches a clear kernel.
    fn reset_mask(&mut self, buf: &mut MaskBuffer);

    /// Set a single mask bit. SerialBackend: `buf.set(slot, kind, true)`.
    /// GpuBackend: enqueues a per-bit write into the buffer's GPU mirror;
    /// Phase 5d batches these into kernel dispatches.
    fn set_mask_bit(&mut self, buf: &mut MaskBuffer, slot: usize, kind: MicroKind);

    /// Sync any pending mask writes. Serial: no-op. GPU: flushes the
    /// per-bit-set queue into the buffer mirror.
    fn commit_mask(&mut self, buf: &mut MaskBuffer);

    /// Run the cascade fixed-point for this tick.
    ///
    /// SerialBackend: delegates to `cascade.run_fixed_point(state, views, events)`.
    /// GpuBackend: dispatches the GPU cascade driver (`cascade_gpu` in
    /// `engine_gpu::cascade`). Phase 5b stub forwards to Serial-equivalent.
    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        views:   &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    );

    /// Fold this tick's events into the view registry.
    ///
    /// SerialBackend: delegates to `views.fold_all(events, events_before, tick)`.
    /// GpuBackend: dispatches GPU view-fold kernels against `view_storage`.
    /// Phase 5c stub forwards to the CPU fold path.
    ///
    /// `events_before` is the `EventRing::total_pushed()` watermark captured
    /// before Phase 4a — it identifies the slice of events that belong to
    /// this tick.
    fn view_fold(
        &mut self,
        views:         &mut Self::Views,
        events:        &EventRing<Self::Event>,
        events_before: usize,
        tick:          u32,
    );

    /// Execute Phase 4a — walk the shuffled action list and emit root-cause
    /// events (AgentAttacked, AgentMoved, AgentFled, AgentAte, AgentDied, …).
    ///
    /// SerialBackend: calls the internal `step::apply_actions` helper which
    /// iterates `scratch.shuffle_idx`, matches on `ActionKind`, mutates `state`,
    /// and pushes events into `events`.
    ///
    /// GpuBackend: dispatches `cs_apply_actions` + `cs_movement` kernels,
    /// drains the GPU event ring into `events`, and commits the mutated agent
    /// SoA onto `state`. Phase 5d stub delegates to the CPU helper.
    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &SimScratch,
        events:  &mut EventRing<Self::Event>,
    );
}
