use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::snapshot::{load_snapshot, save_snapshot};
use engine::state::{AgentSpawn, MovementMode, SimState};
use glam::Vec3;

fn tmp_path(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("engine_snap_{}_{}_{}.bin", name, pid, nonce))
}

#[test]
fn save_then_load_produces_identical_state() {
    let mut state = SimState::new(8, 42);
    let events = EventRing::<Event>::with_cap(64);

    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 2.0, 3.0),
            hp: 50.0,
            max_hp: 100.0,
        })
        .unwrap();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(-4.0, 5.0, -6.0),
            hp: 75.0,
            max_hp: 100.0,
        })
        .unwrap();
    state.set_agent_hunger(a, 0.3);
    state.set_agent_thirst(a, 0.8);
    state.set_agent_rest_timer(b, 0.5);
    state.set_agent_movement_mode(b, MovementMode::Fly);
    state.tick = 100;

    let path = tmp_path("rt");
    save_snapshot(&state, &events, &path).unwrap();
    let (state2, _events2) = load_snapshot::<Event>(&path).unwrap();

    assert_eq!(state2.tick, 100);
    assert_eq!(state2.seed, 42);
    assert_eq!(state2.agent_pos(a), state.agent_pos(a));
    assert_eq!(state2.agent_pos(b), state.agent_pos(b));
    assert_eq!(state2.agent_hp(a), state.agent_hp(a));
    assert_eq!(state2.agent_max_hp(a), state.agent_max_hp(a));
    assert_eq!(state2.agent_hunger(a), Some(0.3));
    assert_eq!(state2.agent_thirst(a), Some(0.8));
    assert_eq!(state2.agent_rest_timer(b), Some(0.5));
    assert_eq!(state2.agent_movement_mode(b), Some(MovementMode::Fly));
    assert_eq!(state2.agent_creature_type(a), Some(CreatureType::Human));
    assert_eq!(state2.agent_creature_type(b), Some(CreatureType::Wolf));
    assert!(state2.agent_alive(a));
    assert!(state2.agent_alive(b));

    std::fs::remove_file(&path).ok();
}

#[test]
fn save_then_load_preserves_freelist_reuse() {
    let mut state = SimState::new(4, 42);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    state.kill_agent(a);

    let events = EventRing::<Event>::with_cap(16);
    let path = tmp_path("fl");
    save_snapshot(&state, &events, &path).unwrap();
    let (mut state2, _) = load_snapshot::<Event>(&path).unwrap();

    // After load, next spawn should reuse slot 1 — proving freelist survived.
    let b = state2
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::X,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    assert_eq!(b.raw(), a.raw(), "freelist reused after load");

    std::fs::remove_file(&path).ok();
}

#[test]
fn save_then_load_preserves_spatial_index() {
    let mut state = SimState::new(8, 7);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    let _b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();

    let events = EventRing::<Event>::with_cap(16);
    let path = tmp_path("sp");
    save_snapshot(&state, &events, &path).unwrap();
    let (state2, _) = load_snapshot::<Event>(&path).unwrap();

    // Spatial index was rebuilt on load; querying within 5m of agent a
    // should find agent a (distance 0) AND agent b (distance 3).
    let hits = state2
        .spatial()
        .within_radius(&state2, Vec3::new(0.0, 0.0, 0.0), 5.0);
    assert_eq!(hits.len(), 2, "spatial index returned 2 agents in 5m radius");
    // Sanity: agent a is in the set.
    let a_raw = a.raw();
    assert!(hits.iter().any(|id| id.raw() == a_raw));

    std::fs::remove_file(&path).ok();
}
