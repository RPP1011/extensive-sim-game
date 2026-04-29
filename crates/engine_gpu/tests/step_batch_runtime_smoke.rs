//! Runtime gate for `engine_gpu::step_batch`. Must pass for any future
//! GPU work to be meaningful — without this, "step_batch works" is
//! vacuously true at compile time but false at runtime (Plan
//! 2026-04-28-step-batch-runtime-fix discovery).
//!
//! NOT cfg-gated — runs under default features (the no-gpu Phase 0
//! stub which forwards to `engine_rules::step::step` via the same
//! ComputeBackend impl path the gpu-feature side uses).

use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;

#[test]
fn step_batch_n1_runs_without_panic() {
    let mut state = SimState::new(8, 42);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        max_hp: 100.0,
    });

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();

    let mut gpu = engine_gpu::GpuBackend::new();
    let tick0 = state.tick;
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &mut views,
        &policy,
        &cascade,
        1,
    );
    assert_eq!(state.tick, tick0 + 1, "step_batch(1) advanced tick by 1");
}

#[test]
fn step_batch_n5_advances_tick_count() {
    let mut state = SimState::new(8, 42);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        max_hp: 100.0,
    });

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();

    let mut gpu = engine_gpu::GpuBackend::new();
    let tick0 = state.tick;
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &mut views,
        &policy,
        &cascade,
        5,
    );
    assert_eq!(state.tick, tick0 + 5, "step_batch(5) advanced tick by 5");
}
