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

    for i in 0..8 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
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
    let tick_ms_count = rows.iter().filter(|r| r.metric == metrics::TICK_MS).count();
    let event_count   = rows.iter().filter(|r| r.metric == metrics::EVENT_COUNT).count();
    let alive_count   = rows.iter().filter(|r| r.metric == metrics::AGENT_ALIVE).count();
    let mask_frac     = rows.iter().filter(|r| r.metric == metrics::MASK_TRUE_FRAC).count();
    assert_eq!(tick_ms_count, 50);
    assert_eq!(event_count,   50);
    assert_eq!(alive_count,   50);
    assert_eq!(mask_frac,     50);
}
