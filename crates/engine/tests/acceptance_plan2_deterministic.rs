//! 100 agents × 1000 ticks running step_full with views + invariants +
//! telemetry. Same seed twice → identical replayable hash. Different seeds
//! → different hashes. Release mode ≤ 2 s budget preserved.

use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::invariant::{InvariantRegistry, PoolNonOverlapInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch}; // Plan B1' Task 11: step_full is unimplemented!() stub
use engine::telemetry::NullSink;
use engine::view::{DamageTaken, MaterializedView};
use glam::Vec3;

fn run(seed: u64) -> [u8; 32] {
    let mut state = SimState::new(110, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1_000_000);
    let cascade = CascadeRegistry::<Event>::new();
    let mut invariants = InvariantRegistry::<Event>::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));
    let telemetry = NullSink;
    let mut dmg = DamageTaken::new(state.agent_cap() as usize);

    for i in 0..100u32 {
        let angle = (i as f32 / 100.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    for _ in 0..1000 {
        let mut views: Vec<&mut dyn MaterializedView<Event>> = vec![&mut dmg];
        step_full(
            &mut state, &mut scratch, &mut events, &UtilityBackend, &cascade,
            &mut views[..], &invariants, &telemetry,
        );
    }
    events.replayable_sha256()
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn same_seed_same_hash() {
    assert_eq!(run(42), run(42));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn different_seed_different_hash() {
    assert_ne!(run(42), run(43));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn full_pipeline_under_two_seconds_release() {
    let t0 = std::time::Instant::now();
    let _ = run(42);
    let elapsed = t0.elapsed();
    eprintln!("plan2 full pipeline: {:?}", elapsed);
    #[cfg(not(debug_assertions))]
    assert!(elapsed.as_secs_f64() <= 2.0, "full pipeline over 2s: {:?}", elapsed);
}
