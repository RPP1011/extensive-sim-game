//! Integration test for the agent_history per-agent state delta tracker (Plan 4 Task 6).
//!
//! Verifies that after N ticks with an `AgentHistory` installed in `DebugConfig`:
//!   - snapshots are populated for the alive agents
//!   - the ring-buffer trims at `max_ticks`
//!   - `at_tick` lookup and `agent_trajectory` iterator work correctly
//!   - no agent_history installed is a zero-cost no-op

use engine::debug::{
    agent_history::{AgentHistory, Filter},
    DebugConfig,
};
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::scratch::SimScratch;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;
use std::sync::{Arc, Mutex};

const TICK_COUNT: usize = 10;

/// Spin up a small combat scene (Wolf vs Human), run N ticks with agent_history
/// installed, then verify snapshots look correct.
#[test]
fn agent_history_records_snapshots_for_alive_agents() {
    let mut state = SimState::new(4, 42);
    let wolf_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("spawn wolf");
    let human_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(5.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("spawn human");

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();

    let history = Arc::new(Mutex::new(AgentHistory::new(Filter {
        agents: None,
        max_ticks: 64,
    })));
    let cfg = DebugConfig {
        agent_history: Some(Arc::clone(&history)),
        ..Default::default()
    };

    for _ in 0..TICK_COUNT {
        engine_rules::step::step(
            &mut engine_rules::backend::SerialBackend,
            &mut state,
            &mut scratch,
            &mut events,
            &mut views,
            &UtilityBackend,
            &cascade,
            &cfg,
        );
    }

    let h = history.lock().unwrap();

    // Should have exactly TICK_COUNT snapshots (well within max_ticks=64).
    assert_eq!(
        h.len(),
        TICK_COUNT,
        "expected {} snapshots, got {}",
        TICK_COUNT,
        h.len()
    );

    // Tick 0 snapshot should contain both agents.
    let snap0 = h.at_tick(0).expect("snapshot for tick 0 should exist");
    assert!(
        snap0.per_agent.contains_key(&wolf_id),
        "tick 0 snapshot should contain wolf"
    );
    assert!(
        snap0.per_agent.contains_key(&human_id),
        "tick 0 snapshot should contain human"
    );

    // Agents should be alive in tick 0 snapshot (10 ticks is not enough to
    // kill either side with 100/80 hp).
    let wolf_snap = &snap0.per_agent[&wolf_id];
    assert!(wolf_snap.alive, "wolf should be alive at tick 0");
    assert!(wolf_snap.hp > 0.0, "wolf hp should be positive");

    // Trajectory iterator should yield TICK_COUNT entries for wolf.
    let traj: Vec<_> = h.agent_trajectory(wolf_id).collect();
    assert_eq!(
        traj.len(),
        TICK_COUNT,
        "wolf trajectory should span {} ticks",
        TICK_COUNT
    );

    // Ticks should be in ascending order (0, 1, 2, ...).
    for (i, &(tick, _)) in traj.iter().enumerate() {
        assert_eq!(tick, i as u32, "trajectory tick {} should equal {}", tick, i);
    }
}

/// Ring-buffer trims: when `max_ticks=5` and we run 8 ticks, only the last 5
/// tick snapshots should be retained.
#[test]
fn agent_history_ring_buffer_trims_at_max_ticks() {
    let max_ticks: usize = 5;
    let run_ticks: usize = 8;

    let mut state = SimState::new(4, 7);
    let _ = state
        .spawn_agent(AgentSpawn {
            hp: 100.0,
            ..Default::default()
        })
        .expect("spawn agent");

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();

    let history = Arc::new(Mutex::new(AgentHistory::new(Filter {
        agents: None,
        max_ticks,
    })));
    let cfg = DebugConfig {
        agent_history: Some(Arc::clone(&history)),
        ..Default::default()
    };

    for _ in 0..run_ticks {
        engine_rules::step::step(
            &mut engine_rules::backend::SerialBackend,
            &mut state,
            &mut scratch,
            &mut events,
            &mut views,
            &UtilityBackend,
            &cascade,
            &cfg,
        );
    }

    let h = history.lock().unwrap();

    // Ring buffer should have trimmed to max_ticks.
    assert_eq!(
        h.len(),
        max_ticks,
        "ring buffer should retain at most {} snapshots, got {}",
        max_ticks,
        h.len()
    );

    // Early ticks (0..2) should have been evicted; ticks 3..7 should remain.
    assert!(h.at_tick(0).is_none(), "tick 0 should have been evicted");
    assert!(h.at_tick(2).is_none(), "tick 2 should have been evicted");
    // Ticks run_ticks-max_ticks .. run_ticks-1 should be present.
    let first_retained = (run_ticks - max_ticks) as u32;
    let last_retained = (run_ticks - 1) as u32;
    assert!(
        h.at_tick(first_retained).is_some(),
        "tick {} should be retained",
        first_retained
    );
    assert!(
        h.at_tick(last_retained).is_some(),
        "tick {} should be retained",
        last_retained
    );
}

/// Filter by agent: only the listed agent should appear in snapshots.
#[test]
fn agent_history_filter_limits_to_specified_agents() {
    let mut state = SimState::new(4, 42);
    let wolf_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .expect("spawn wolf");
    let _human_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(5.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("spawn human");

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();

    // Only track the wolf.
    let history = Arc::new(Mutex::new(AgentHistory::new(Filter {
        agents: Some(vec![wolf_id]),
        max_ticks: 64,
    })));
    let cfg = DebugConfig {
        agent_history: Some(Arc::clone(&history)),
        ..Default::default()
    };

    for _ in 0..5 {
        engine_rules::step::step(
            &mut engine_rules::backend::SerialBackend,
            &mut state,
            &mut scratch,
            &mut events,
            &mut views,
            &UtilityBackend,
            &cascade,
            &cfg,
        );
    }

    let h = history.lock().unwrap();
    let snap0 = h.at_tick(0).expect("tick 0 should exist");
    assert!(
        snap0.per_agent.contains_key(&wolf_id),
        "wolf should be in filtered snapshot"
    );
    assert_eq!(
        snap0.per_agent.len(),
        1,
        "only wolf should appear in snapshot (filter active)"
    );
}

/// With no `agent_history` installed, the tick pipeline should still advance
/// normally with no crashes.
#[test]
fn agent_history_not_installed_is_zero_cost_no_op() {
    let mut state = SimState::new(4, 99);
    let _ = state.spawn_agent(AgentSpawn { hp: 50.0, ..Default::default() });
    let tick_before = state.tick;

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();
    let cfg = DebugConfig::default(); // agent_history = None

    engine_rules::step::step(
            &mut engine_rules::backend::SerialBackend,
        &mut state,
        &mut scratch,
        &mut events,
        &mut views,
        &UtilityBackend,
        &cascade,
        &cfg,
    );

    assert_eq!(state.tick, tick_before + 1, "tick should advance even without agent_history");
}
