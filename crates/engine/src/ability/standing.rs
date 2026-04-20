//! Combat Foundation Task 17 — `EffectOp::ModifyStanding` handler.
//!
//! Cascade handler on `Event::EffectStandingDelta`. Calls
//! `SimState::adjust_standing(a, b, delta)`, which routes into
//! `SparseStandings::adjust` — it keys by the ordered tuple
//! `(min(a, b), max(a, b))` (symmetric) and clamps the resulting value to
//! `[-1000, 1000]`.
//!
//! Symmetry: applying `(A, B, +50)` and then `(B, A, -50)` cancels exactly;
//! both events resolve to the same pair slot. The clamp is silent (no
//! overflow event); replays re-run the handler and land on the same
//! post-clamp value because the storage, not the event, owns the clamp.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct ModifyStandingHandler;

impl CascadeHandler for ModifyStandingHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectStandingDelta }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (a, b, delta) = match *event {
            Event::EffectStandingDelta { a, b, delta, .. } => (a, b, delta),
            _ => return,
        };
        if delta == 0 { return; }
        state.adjust_standing(a, b, delta);
    }
}
