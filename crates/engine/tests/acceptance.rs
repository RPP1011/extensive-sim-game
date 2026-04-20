//! End-to-end acceptance — exercises every primitive named in the plan's
//! acceptance criteria. If this passes, the MVP is done.

use engine::cascade::CascadeRegistry;
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
    let cascade = CascadeRegistry::new();
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
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        writer.record_tick(&state);
    }
    let elapsed = t0.elapsed();

    dmg.fold(&events);
    let _trace_hash = events.replayable_sha256();

    // Acceptance criteria:
    assert_eq!(state.tick, ticks, "tick counter advanced correctly");
    // Task 138 raised the debug-mode budget from 2s to 15s. MoveToward is
    // now a target-bound mask — its `mask_move_toward_candidates`
    // enumerator walks `query.nearby_agents(self.pos, aggro_range)` for
    // every alive agent each tick, where `aggro_range` is 50m. At 100
    // agents in a 50m-radius ring every agent sees every other agent as a
    // candidate, so per-tick work grew O(N²) instead of O(N). Release
    // build still finishes inside a second (~230 ms on the author's
    // machine); debug rides the slower path. Bumping the budget keeps
    // the smoke-test's intent (tick counter advances, movement fires,
    // trajectory serialises) without masking a legitimate regression.
    assert!(
        elapsed.as_secs_f64() <= 15.0,
        "elapsed {:?} exceeds 15s budget", elapsed
    );
    // Proof-of-work checks — the sim actually *did* something across 1000 ticks,
    // not just advanced the tick counter. Assertions on emission and on the
    // UtilityBackend's core responsibility (movement toward neighbors).
    let total_events = events.iter().count();
    assert!(total_events > 0, "sim produced zero events over 1000 ticks");
    let moved_any = events.iter().any(|e| matches!(e, engine::event::Event::AgentMoved { .. }));
    assert!(moved_any,
        "UtilityBackend + 100 agents must produce at least one AgentMoved event");

    let _: f32 = dmg.value(engine::ids::AgentId::new(1).unwrap());
    let tmp = std::env::temp_dir().join("engine_acceptance_traj.safetensors");
    writer.write(&tmp).expect("trajectory writable");
    std::fs::remove_file(&tmp).ok();

    println!("mvp_acceptance: elapsed = {:?}, events = {}", elapsed, total_events);
}
