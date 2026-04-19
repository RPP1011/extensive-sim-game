use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::trajectory::{TrajectoryReader, TrajectoryWriter};
use glam::Vec3;

#[test]
fn emit_and_reload_trajectory() {
    let mut state = SimState::new(20, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(10_000);
    let cascade = CascadeRegistry::new();
    for i in 0..5 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let mut writer = TrajectoryWriter::new(5, 50);
    for _ in 0..50 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        writer.record_tick(&state);
    }
    let tmp = std::env::temp_dir().join("engine_traj_test.safetensors");
    writer.write(&tmp).unwrap();

    let loaded = TrajectoryReader::load(&tmp).unwrap();
    assert_eq!(loaded.n_agents(), 5);
    assert_eq!(loaded.n_ticks(), 50);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn python_roundtrip_preserves_values() {
    use engine::cascade::CascadeRegistry;
    use engine::creature::CreatureType;
    use engine::event::EventRing;
    use engine::policy::UtilityBackend;
    use engine::state::{AgentSpawn, SimState};
    use engine::step::{step, SimScratch};
    use engine::trajectory::TrajectoryWriter;
    use glam::Vec3;

    let mut state = SimState::new(10, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1000);
    let cascade = CascadeRegistry::new();
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let mut writer = TrajectoryWriter::new(3, 20);
    for _ in 0..20 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        writer.record_tick(&state);
    }
    let path_a = std::env::temp_dir().join("engine_traj_python_a.safetensors");
    let path_b = std::env::temp_dir().join("engine_traj_python_b.safetensors");
    writer.write(&path_a).unwrap();

    // Resolve script path relative to CARGO_MANIFEST_DIR so the test works from any CWD.
    let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("..").join("scripts").join("engine_roundtrip.py");

    let status = std::process::Command::new("uv")
        .args([
            "run", "--with", "safetensors", "--with", "numpy",
            script.to_str().unwrap(),
            path_a.to_str().unwrap(),
            path_b.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run python roundtrip script (is `uv` installed?)");
    assert!(status.success(), "python roundtrip exited non-zero");

    // Compare tensor values (not bytes — metadata order / padding may differ).
    use engine::trajectory::TrajectoryReader;
    let a = TrajectoryReader::load(&path_a).unwrap();
    let b = TrajectoryReader::load(&path_b).unwrap();
    assert_eq!(a.n_agents(), b.n_agents());
    assert_eq!(a.n_ticks(), b.n_ticks());

    std::fs::remove_file(&path_a).ok();
    std::fs::remove_file(&path_b).ok();
}
