use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::invariant::{InvariantRegistry, PoolNonOverlapInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::{metrics, VecSink};
use engine::view::{DamageTaken, MaterializedView};
use glam::Vec3;

#[test]
fn six_phase_pipeline_runs_clean() {
    let mut state = SimState::new(20, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(10_000);
    let cascade = CascadeRegistry::<Event>::new();
    let mut invariants = InvariantRegistry::<Event>::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));
    let telemetry = VecSink::new();

    // Spawn 8 agents well separated (200m apart) so no-one can close the
    // gap at MOVE_SPEED_MPS=1.0 × 50 ticks (max closure 100m per pair)
    // to within the 2m ATTACK_RANGE. agent_alive stays at 8 across all
    // 50 ticks — the invariant we pin below.
    for i in 0..8 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new((i as f32) * 200.0, 0.0, 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }

    let mut dmg = DamageTaken::new(state.agent_cap() as usize);
    for _ in 0..50 {
        let mut views: Vec<&mut dyn MaterializedView<Event>> = vec![&mut dmg];
        step_full(
            &mut state,
            &mut scratch,
            &mut events,
            &UtilityBackend,
            &cascade,
            &mut views[..],
            &invariants,
            &telemetry,
        );
    }

    // Telemetry received built-in metrics every tick.
    let rows = telemetry.drain();
    let tick_ms_rows:  Vec<_> = rows.iter().filter(|r| r.metric == metrics::TICK_MS).collect();
    let event_rows:    Vec<_> = rows.iter().filter(|r| r.metric == metrics::EVENT_COUNT).collect();
    let alive_rows:    Vec<_> = rows.iter().filter(|r| r.metric == metrics::AGENT_ALIVE).collect();
    let mask_rows:     Vec<_> = rows.iter().filter(|r| r.metric == metrics::MASK_TRUE_FRAC).collect();
    // Audit fix HIGH #5: CASCADE_ITERATIONS must be emitted once per tick.
    let cascade_rows:  Vec<_> = rows.iter().filter(|r| r.metric == metrics::CASCADE_ITERATIONS).collect();
    assert_eq!(tick_ms_rows.len(), 50);
    assert_eq!(event_rows.len(),   50);
    assert_eq!(alive_rows.len(),   50);
    assert_eq!(mask_rows.len(),    50);
    assert_eq!(cascade_rows.len(), 50, "cascade_iterations must emit once per tick");
    // Iteration count is always within [0, MAX_CASCADE_ITERATIONS].
    for (i, r) in cascade_rows.iter().enumerate() {
        assert!(r.value >= 0.0 && r.value <= 8.0,
            "cascade_iterations[{}] = {} outside [0, 8]", i, r.value);
    }

    // Value inspection — not just counts. A broken impl that emitted
    // constant zeros or infinities would pass the count-only assertions.
    //
    // agent_alive must be exactly 8 (spawn count) on every row. The test
    // scenario doesn't kill anyone, so alive count is invariant.
    for (i, r) in alive_rows.iter().enumerate() {
        assert!((r.value - 8.0).abs() < 1e-9,
            "agent_alive[{}] = {}, expected 8.0 (spawn count)", i, r.value);
    }
    // event_count must be non-negative on every row.
    for (i, r) in event_rows.iter().enumerate() {
        assert!(r.value >= 0.0, "event_count[{}] = {} < 0", i, r.value);
    }
    // mask_true_frac must be in [0, 1] on every row (it's a fraction).
    for (i, r) in mask_rows.iter().enumerate() {
        assert!((0.0..=1.0).contains(&r.value),
            "mask_true_frac[{}] = {} outside [0, 1]", i, r.value);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Pre")]
fn step_full_panics_when_scratch_undersized() {
    use engine::cascade::CascadeRegistry;
    use engine_data::entities::CreatureType;
    use engine::event::EventRing;
use engine_data::events::Event;
    use engine::invariant::InvariantRegistry;
    use engine::policy::UtilityBackend;
    use engine::state::{AgentSpawn, SimState};
    use engine::step::{step_full, SimScratch};
    use engine::telemetry::NullSink;
    use glam::Vec3;

    let mut state = SimState::new(8, 42);
    // Deliberately mis-sized: 2 instead of 8.
    let mut scratch = SimScratch::new(2);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();
    let invariants = InvariantRegistry::<Event>::new();

    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    });

    // `step_full` debug_requires scratch capacity == state.agent_cap() * 18.
    // Undersized scratch violates the pre-condition — panic in debug.
    step_full(
        &mut state, &mut scratch, &mut events,
        &UtilityBackend, &cascade, &mut [], &invariants, &NullSink,
    );
}
