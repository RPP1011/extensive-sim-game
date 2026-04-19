use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::step;
use engine::trajectory::{TrajectoryReader, TrajectoryWriter};
use glam::Vec3;

#[test]
fn emit_and_reload_trajectory() {
    let mut state = SimState::new(20, 42);
    let mut events = EventRing::with_cap(10_000);
    for i in 0..5 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let mut writer = TrajectoryWriter::new(5, 50);
    for _ in 0..50 {
        step(&mut state, &mut events, &UtilityBackend);
        writer.record_tick(&state);
    }
    let tmp = std::env::temp_dir().join("engine_traj_test.safetensors");
    writer.write(&tmp).unwrap();

    let loaded = TrajectoryReader::load(&tmp).unwrap();
    assert_eq!(loaded.n_agents(), 5);
    assert_eq!(loaded.n_ticks(), 50);

    std::fs::remove_file(&tmp).ok();
}
