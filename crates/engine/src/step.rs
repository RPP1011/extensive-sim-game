// crates/engine/src/step.rs
use crate::event::EventRing;
use crate::mask::MaskBuffer;
use crate::policy::{Action, PolicyBackend};
use crate::state::SimState;

pub fn step<B: PolicyBackend>(state: &mut SimState, _events: &mut EventRing, backend: &B) {
    // 1. Build mask (for MVP, only Hold allowed).
    let mut mask = MaskBuffer::new(state.agent_cap() as usize);
    mask.mark_hold_allowed(state);

    // 2. Policy evaluate.
    let mut actions: Vec<Action> = Vec::with_capacity(state.agent_cap() as usize);
    backend.evaluate(state, &mask, &mut actions);

    // 3. Apply actions (Hold = no-op for MVP).
    for _action in &actions { /* Hold does nothing; richer actions in Task 11. */ }

    // 4. Advance tick.
    state.tick += 1;
}
