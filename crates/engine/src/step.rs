// crates/engine/src/step.rs
//
// COMPILE-ONLY STUBS — Plan B1' Task 11 deleted the real step/step_full
// implementations. This file exists solely so the many #[ignore]d tests that
// still import `engine::step::{step, SimScratch, ...}` compile without error.
// Every function in this file is `unimplemented!()` and will panic if called.
//
// Remove this file when Task 11 lands and `engine_rules::step` is the
// canonical tick driver. The test imports will be updated to
// `engine_rules::step::step` at that point.

pub use crate::scratch::SimScratch;

use crate::cascade::CascadeRegistry;
use crate::event::EventRing;
use crate::invariant::InvariantRegistry;
use crate::policy::PolicyBackend;
use crate::state::SimState;
use crate::telemetry::TelemetrySink;
use crate::view::MaterializedView;
use engine_data::events::Event;

/// DELETED — Plan B1' Task 11. `unimplemented!()` stub for test compilation.
/// Re-enable after B1' Task 11 emits engine_rules::step::step.
pub fn step<B: PolicyBackend, V>(
    _state:   &mut SimState,
    _scratch: &mut SimScratch,
    _events:  &mut EventRing<Event>,
    _backend: &B,
    _cascade: &CascadeRegistry<Event, V>,
) {
    unimplemented!(
        "engine::step::step is DELETED (Plan B1' Task 11). \
         Re-enable after engine_rules::step::step is emitted."
    )
}

/// DELETED — Plan B1' Task 11. `unimplemented!()` stub for test compilation.
#[allow(clippy::too_many_arguments)]
pub fn step_full<B: PolicyBackend, V>(
    _state:      &mut SimState,
    _scratch:    &mut SimScratch,
    _events:     &mut EventRing<Event>,
    _backend:    &B,
    _cascade:    &CascadeRegistry<Event, V>,
    _views:      &mut [&mut dyn MaterializedView<Event>],
    _invariants: &InvariantRegistry<Event>,
    _telemetry:  &dyn TelemetrySink,
) {
    unimplemented!(
        "engine::step::step_full is DELETED (Plan B1' Task 11). \
         Re-enable after engine_rules::step::step is emitted."
    )
}

/// DELETED — Plan B1' Task 11. Stub for test compilation.
pub fn step_phases_1_to_3<B: PolicyBackend>(
    _state:   &mut SimState,
    _scratch: &mut SimScratch,
    _backend: &B,
) {
    unimplemented!("engine::step::step_phases_1_to_3 DELETED — Plan B1' Task 11")
}

/// DELETED — Plan B1' Task 11. Stub for test compilation.
pub fn apply_actions(
    _state:   &mut SimState,
    _scratch: &SimScratch,
    _events:  &mut EventRing<Event>,
) {
    unimplemented!("engine::step::apply_actions DELETED — Plan B1' Task 11")
}

/// DELETED — Plan B1' Task 11. Stub for test compilation.
pub fn finalize_tick(
    _state:          &mut SimState,
    _scratch:        &SimScratch,
    _events:         &EventRing<Event>,
    _invariants:     &InvariantRegistry<Event>,
    _telemetry:      &dyn TelemetrySink,
    _t_start:        std::time::Instant,
    _events_emitted: usize,
) {
    unimplemented!("engine::step::finalize_tick DELETED — Plan B1' Task 11")
}

/// DELETED — Plan B1' Task 11. Stub for test compilation.
pub fn shuffle_actions_in_place(
    _world_seed:  u64,
    _tick:        u32,
    _actions:     &[crate::policy::Action],
    _shuffle_idx: &mut Vec<u32>,
) {
    unimplemented!("engine::step::shuffle_actions_in_place DELETED — Plan B1' Task 11")
}
