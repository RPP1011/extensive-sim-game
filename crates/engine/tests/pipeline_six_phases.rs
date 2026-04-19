use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
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
    let mut events = EventRing::with_cap(10_000);
    let cascade = CascadeRegistry::new();
    let mut invariants = InvariantRegistry::new();
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
        });
    }

    let mut dmg = DamageTaken::new(state.agent_cap() as usize);
    for _ in 0..50 {
        let mut views: Vec<&mut dyn MaterializedView> = vec![&mut dmg];
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
    assert_eq!(tick_ms_rows.len(), 50);
    assert_eq!(event_rows.len(),   50);
    assert_eq!(alive_rows.len(),   50);
    assert_eq!(mask_rows.len(),    50);

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
