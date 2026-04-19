//! End-to-end acceptance — exercises every primitive named in the plan's
//! acceptance criteria. If this passes, the MVP is done.

use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::trajectory::TrajectoryWriter;
use engine::view::materialized::{DamageTaken, MaterializedView};
use glam::Vec3;

#[test]
fn mvp_acceptance() {
    let seed = 42u64;
    let n_agents: u32 = 100;
    let ticks: u32 = 1000;

    let mut state = SimState::new(n_agents + 10, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1_000_000);
    let mut dmg = DamageTaken::new(state.agent_cap() as usize);
    let mut writer = TrajectoryWriter::new(n_agents as usize, ticks as usize);

    for i in 0..n_agents {
        let angle = (i as f32 / n_agents as f32) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }

    let t0 = std::time::Instant::now();
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend);
        writer.record_tick(&state);
    }
    let elapsed = t0.elapsed();

    dmg.fold(&events);
    let trace_hash = events.replayable_sha256();

    // Acceptance criteria:
    assert_eq!(state.tick, ticks, "tick counter advanced correctly");
    assert!(
        elapsed.as_secs_f64() <= 2.0,
        "elapsed {:?} exceeds 2s budget", elapsed
    );
    // Sanity checks — primitives are exercised (not just defined).
    assert_eq!(trace_hash.len(), 32, "sha256 hash is 32 bytes");
    let _: f32 = dmg.value(engine::ids::AgentId::new(1).unwrap());
    let tmp = std::env::temp_dir().join("engine_acceptance_traj.safetensors");
    writer.write(&tmp).expect("trajectory writable");
    std::fs::remove_file(&tmp).ok();

    println!("mvp_acceptance: elapsed = {:?}", elapsed);
}
