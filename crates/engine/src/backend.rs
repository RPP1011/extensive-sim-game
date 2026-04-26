//! Simulation backend abstraction (GPU-megakernel plan, Phase 0).
//!
//! Defines [`ComputeBackend`] — the narrow trait both CPU and GPU tick drivers
//! implement. `CpuBackend` (the thin wrapper around `step::step`) is DELETED
//! in Plan B1' Task 11; `engine_rules` will emit `SerialBackend` as the CPU
//! implementation. Until then callers that need a CPU tick driver should call
//! `engine_rules::step::step` directly once Task 11 lands.
//!
//! ## Trait shape & genericity
//!
//! The trait method is generic over `B: PolicyBackend` to avoid modifying the
//! step kernel signature (see `docs/plans/gpu_megakernel_plan.md`). This means
//! `dyn ComputeBackend` is not object-safe — callers hold concrete backend types.

use crate::cascade::CascadeRegistry;
use crate::event::{EventLike, EventRing};
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
///
/// NOTE: `CpuBackend` is deleted (Plan B1' Task 11). `engine_rules` will emit
/// `SerialBackend` implementing this trait once Task 11 lands.
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
}
