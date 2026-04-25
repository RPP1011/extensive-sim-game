use super::{FailureMode, Invariant, Violation};
use crate::event::{EventLike, EventRing};
use crate::mask::MicroKind;
use crate::policy::ActionKind;
use crate::state::SimState;
use crate::step::SimScratch;

/// Every action emitted by a policy must have a `true` bit in the mask that
/// was passed to its `evaluate` call. Regression guard against mask/policy
/// divergence.
pub struct MaskValidityInvariant;

impl MaskValidityInvariant {
    pub fn new() -> Self { Self }

    /// Scratch-aware check used by tests and the tick pipeline (which has
    /// access to last-tick mask + actions in `SimScratch`). The generic
    /// `Invariant::check` variant can't see scratch and returns `None`.
    pub fn check_with_scratch(
        &self, _state: &SimState, scratch: &SimScratch,
    ) -> Option<Violation> {
        let n_kinds = MicroKind::ALL.len();
        for action in &scratch.actions {
            let slot = (action.agent.raw() - 1) as usize;
            match action.kind {
                ActionKind::Micro { kind, .. } => {
                    let bit = slot * n_kinds + kind as usize;
                    if !scratch.mask.micro_kind.get(bit).copied().unwrap_or(false) {
                        return Some(Violation {
                            invariant: "mask_validity",
                            tick: 0,
                            message: format!("action {:?} violates mask", action),
                            payload: None,
                        });
                    }
                }
                ActionKind::Macro(_) => { /* macro mask head deferred */ }
            }
        }
        None
    }
}

impl Default for MaskValidityInvariant { fn default() -> Self { Self::new() } }

impl<E: EventLike> Invariant<E> for MaskValidityInvariant {
    fn name(&self) -> &'static str { "mask_validity" }
    fn failure_mode(&self) -> FailureMode {
        #[cfg(debug_assertions)] { FailureMode::Panic }
        #[cfg(not(debug_assertions))] { FailureMode::Log }
    }
    fn check(&self, _s: &SimState, _e: &EventRing<E>) -> Option<Violation> { None }
}

/// No agent slot can be both alive and in the freelist, and the freelist
/// must contain no duplicates. Delegates to `SimState::pool_is_consistent`
/// which walks the `Pool<T>` alive vec + freelist.
pub struct PoolNonOverlapInvariant;

impl<E: EventLike> Invariant<E> for PoolNonOverlapInvariant {
    fn name(&self) -> &'static str { "pool_non_overlap" }
    fn failure_mode(&self) -> FailureMode {
        #[cfg(debug_assertions)] { FailureMode::Panic }
        #[cfg(not(debug_assertions))] { FailureMode::Log }
    }
    fn check(&self, state: &SimState, _events: &EventRing<E>) -> Option<Violation> {
        if state.pool_is_consistent() {
            None
        } else {
            Some(Violation {
                invariant: "pool_non_overlap",
                tick:      state.tick,
                message:   "agent pool: alive/freelist overlap or freelist duplicate".into(),
                payload:   None,
            })
        }
    }
}
