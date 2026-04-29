//! End-to-end smoke: `step_batch(n)` runs without panicking and
//! advances the tick counter by `n`. Works under default features
//! (CPU forward via engine_rules::step::step) and under `--features
//! gpu` (SCHEDULE-loop dispatcher + CPU forward).

mod common;

use common::smoke_fixture_n4;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::step::SimScratch;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn step_batch_n1_advances_tick() {
    let mut state = smoke_fixture_n4();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();

    let mut gpu = engine_gpu::GpuBackend::new();
    let tick0 = state.tick;
    gpu.step_batch(
        &mut state, &mut scratch, &mut events,
        &mut views, &policy, &cascade, 1,
    );
    assert_eq!(state.tick, tick0 + 1);
}

#[test]
fn step_batch_n10_advances_tick() {
    let mut state = smoke_fixture_n4();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(4096);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();

    let mut gpu = engine_gpu::GpuBackend::new();
    let tick0 = state.tick;
    gpu.step_batch(
        &mut state, &mut scratch, &mut events,
        &mut views, &policy, &cascade, 10,
    );
    assert_eq!(state.tick, tick0 + 10);
}
