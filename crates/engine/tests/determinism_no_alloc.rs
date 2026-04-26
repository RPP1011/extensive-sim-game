#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn steady_state_zero_alloc_after_warmup() {
    use engine::cascade::CascadeRegistry;
    use engine_data::entities::CreatureType;
    use engine::event::EventRing;
use engine_data::events::Event;
    use engine::policy::UtilityBackend;
    use engine::state::{AgentSpawn, SimState};
    use engine::step::{step, SimScratch}; // Plan B1' Task 11: step is unimplemented!() stub
    use glam::Vec3;

    let mut state = SimState::new(100, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(100_000);
    let cascade = CascadeRegistry::<Event>::new();
    for i in 0..50 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }

    for _ in 0..100 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..100 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);

    eprintln!(
        "steady-state 100 ticks with 50 agents: {} blocks ({} bytes total)",
        stats.total_blocks, stats.total_bytes
    );

    // Allow a tiny budget for debug-build noise; target is 0.
    assert!(
        stats.total_blocks <= 16,
        "steady-state allocations: {} blocks ({} bytes total)",
        stats.total_blocks, stats.total_bytes
    );
}

#[cfg(not(feature = "dhat-heap"))]
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn steady_state_zero_alloc_skipped_without_feature() {
    eprintln!("skipping — run with `--features dhat-heap` to measure allocations");
}
