//! Smoke: per-tick state.tick advances correctly under step_batch.
//! Works under both default + `--features gpu`.

mod common;

use common::smoke_fixture_n4;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::step::SimScratch;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn tick_advances_through_step_batch_calls() {
    let mut state = smoke_fixture_n4();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(4096);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();

    let mut gpu = engine_gpu::GpuBackend::new();

    // Three batched calls: 1 + 5 + 7 ticks = 13 ticks total.
    let tick0 = state.tick;
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 1);
    assert_eq!(state.tick, tick0 + 1, "after step_batch(1)");
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 5);
    assert_eq!(state.tick, tick0 + 6, "after step_batch(5)");
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 7);
    assert_eq!(state.tick, tick0 + 13, "after step_batch(7)");
}
