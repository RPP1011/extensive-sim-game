use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::UtilityBackend;
use engine::state::{SimState, AgentSpawn};
use engine_data::entities::CreatureType;
use engine::step::{step, SimScratch};
use glam::Vec3;

#[test]
fn step_advances_tick_and_emits_no_events_for_hold() {
    let mut state = SimState::new(5, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(100);
    let cascade = CascadeRegistry::<Event>::new();
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
        ..Default::default()
    });
    step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    assert_eq!(state.tick, 1);
    assert_eq!(events.iter().count(), 0, "Hold produces no events");
}

#[test]
fn step_is_reproducible_for_same_seed() {
    // Seed-divergence assertion deferred to Task 12, once MoveToward (Task 11)
    // introduces non-empty event traces. At Task 10, Hold-only ticks produce
    // empty event rings for ALL seeds — the replayable hash can't distinguish
    // seeds without emitted events.
    fn trace(seed: u64) -> [u8; 32] {
        let mut state = SimState::new(10, seed);
        let mut scratch = SimScratch::new(state.agent_cap() as usize);
        let mut events = EventRing::<Event>::with_cap(1000);
        let cascade = CascadeRegistry::<Event>::new();
        for i in 0..5 {
            state.spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32, 0.0, 10.0), hp: 100.0,
                ..Default::default()
            });
        }
        for _ in 0..100 { step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade); }
        events.replayable_sha256()
    }
    assert_eq!(trace(42), trace(42), "same seed → same trace (Hold-only is trivially deterministic)");
}
