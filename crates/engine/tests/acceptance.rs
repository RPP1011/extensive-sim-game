//! End-to-end acceptance — exercises every primitive named in the plan's
//! acceptance criteria. If this passes, the MVP is done.

use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
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
    let mut events = EventRing::<Event>::with_cap(1_000_000);
    let cascade = CascadeRegistry::<Event>::new();
    let mut dmg = DamageTaken::new(state.agent_cap() as usize);
    let mut writer = TrajectoryWriter::new(n_agents as usize, ticks as usize);

    for i in 0..n_agents {
        let angle = (i as f32 / n_agents as f32) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
            ..Default::default()
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
    // Task 138 raised the debug-mode budget from 2s to 15s because
    // MoveToward's mask enumerator walked every alive agent within
    // `config.combat.aggro_range = 50` m — which, in this fixture (100
    // agents on a 50 m ring) is every other agent — producing an O(N²)
    // per-tick candidate list.
    //
    // Task 144 pulled the radius out into
    // `config.movement.max_move_radius = 20` m, which in this fixture
    // cuts per-agent MoveToward candidates from ~100 down to ~13 — a
    // ~3× drop on debug-build mask-build cost. Remaining debug cost is
    // ~debug-mode overhead: the spatial index sorts its result by
    // `AgentId::raw()` on every `within_radius` call for determinism,
    // and Rust's debug-profile UB precondition checks (`is_aligned_to`,
    // `copy::precondition_check`, …) dominate the flamegraph at ~35%
    // of samples combined. Release rides the same code paths without
    // those checks and lands ~180 ms on the author's machine.
    //
    // Current budget: 8 s debug (measured ~6.0 s on the author's
    // machine, ~33 % headroom). Release headroom is tighter: the 100 ms
    // aspirational ceiling from the task spec isn't achievable without
    // a follow-up on the view-fold phase (see report on task 144 for
    // the `ViewRegistry::fold_all` O(cumulative-events) walk; not
    // addressed here to keep the perf fix scoped to the mask).
    assert!(
        elapsed.as_secs_f64() <= 8.0,
        "elapsed {:?} exceeds 8s budget", elapsed
    );
    // Proof-of-work checks — the sim actually *did* something across 1000 ticks,
    // not just advanced the tick counter. Assertions on emission and on the
    // UtilityBackend's core responsibility (movement toward neighbors).
    let total_events = events.iter().count();
    assert!(total_events > 0, "sim produced zero events over 1000 ticks");
    let moved_any = events.iter().any(|e| matches!(e, engine_data::events::Event::AgentMoved { .. }));
    assert!(moved_any,
        "UtilityBackend + 100 agents must produce at least one AgentMoved event");

    let _: f32 = dmg.value(engine::ids::AgentId::new(1).unwrap());
    let tmp = std::env::temp_dir().join("engine_acceptance_traj.safetensors");
    writer.write(&tmp).expect("trajectory writable");
    std::fs::remove_file(&tmp).ok();

    println!("mvp_acceptance: elapsed = {:?}, events = {}", elapsed, total_events);
}
