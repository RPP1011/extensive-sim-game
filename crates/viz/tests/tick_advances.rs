use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

#[test]
fn step_advances_tick() {
    let mut sim = SimState::new(4, 42);
    let mut scratch = SimScratch::new(4);
    let mut events  = EventRing::with_cap(64);
    let cascade = CascadeRegistry::new();
    let backend = UtilityBackend;

    sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(3.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();

    assert_eq!(sim.tick, 0);
    for expected in 1..=20u32 {
        step(&mut sim, &mut scratch, &mut events, &backend, &cascade);
        assert_eq!(sim.tick, expected, "tick should advance by 1 per step");
    }
}

#[test]
fn wolf_moves_toward_human_across_20_ticks() {
    let mut sim = SimState::new(4, 42);
    let mut scratch = SimScratch::new(4);
    let mut events  = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();
    let backend = UtilityBackend;

    let human = sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let wolf = sim.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(20.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();

    let wolf_start = sim.agent_pos(wolf).unwrap();
    for _ in 0..20 {
        step(&mut sim, &mut scratch, &mut events, &backend, &cascade);
    }
    let wolf_end = sim.agent_pos(wolf).unwrap();
    let human_pos = sim.agent_pos(human).unwrap_or(Vec3::ZERO);
    let d_before = wolf_start.distance(human_pos);
    let d_after  = wolf_end.distance(human_pos);
    assert!(
        d_after < d_before,
        "wolf should have closed distance — before={:.2} after={:.2}", d_before, d_after,
    );
}
