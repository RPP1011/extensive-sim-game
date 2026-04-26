//! Integration test for the tick_stepper per-phase pause harness (Plan 4 Task 4).
//!
//! Verifies that `engine_rules::step::step` fires a checkpoint for each
//! pipeline phase when a `StepperHandle` is installed in `DebugConfig`.

use engine::debug::DebugConfig;
use engine::debug::tick_stepper::{Phase, Step, StepperHandle};
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::scratch::SimScratch;
use engine::state::{AgentSpawn, SimState};
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;
use std::thread;

#[test]
fn stepper_checkpoints_each_phase() {
    // Build a minimal 2-agent SimState (wolf vs human so the policy has
    // something to evaluate; engagement physics fires).
    let mut state = SimState::new(4, 42);
    let _ = state.spawn_agent(AgentSpawn {
        creature_type: engine_data::entities::CreatureType::Wolf,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    });
    let _ = state.spawn_agent(AgentSpawn {
        creature_type: engine_data::entities::CreatureType::Human,
        pos: Vec3::new(1.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    });

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();

    let (handle, step_tx, phase_rx) = StepperHandle::new();
    let cfg = DebugConfig {
        tick_stepper: Some(handle),
        ..Default::default()
    };

    // Driver thread: run one tick under the cfg.
    // We move the engine state into the thread; it's single-threaded inside.
    let driver = thread::spawn(move || {
        engine_rules::step::step(
            &mut state,
            &mut scratch,
            &mut events,
            &mut views,
            &UtilityBackend,
            &cascade,
            &cfg,
        );
    });

    // Controller: collect phase events; advance after each.
    let mut phases_seen: Vec<Phase> = Vec::new();
    while let Ok(phase) = phase_rx.recv() {
        phases_seen.push(phase);
        step_tx.send(Step::Continue).unwrap();
        if phase == Phase::TickEnd {
            break;
        }
    }
    driver.join().unwrap();

    // All 7 phases must have been signalled in order.
    assert!(
        phases_seen.contains(&Phase::BeforeViewFold),
        "missing BeforeViewFold; saw: {phases_seen:?}"
    );
    assert!(
        phases_seen.contains(&Phase::AfterMaskFill),
        "missing AfterMaskFill; saw: {phases_seen:?}"
    );
    assert!(
        phases_seen.contains(&Phase::AfterScoring),
        "missing AfterScoring; saw: {phases_seen:?}"
    );
    assert!(
        phases_seen.contains(&Phase::AfterActionSelect),
        "missing AfterActionSelect; saw: {phases_seen:?}"
    );
    assert!(
        phases_seen.contains(&Phase::AfterCascadeDispatch),
        "missing AfterCascadeDispatch; saw: {phases_seen:?}"
    );
    assert!(
        phases_seen.contains(&Phase::AfterViewFold),
        "missing AfterViewFold; saw: {phases_seen:?}"
    );
    assert!(
        phases_seen.contains(&Phase::TickEnd),
        "missing TickEnd; saw: {phases_seen:?}"
    );
    assert_eq!(phases_seen.len(), 7, "expected exactly 7 phases, got: {phases_seen:?}");
}

#[test]
fn abort_on_second_phase_stops_tick() {
    // Verify that sending Abort at AfterMaskFill causes step() to return
    // early (the thread should finish) and subsequent phases are NOT seen.
    let mut state = SimState::new(4, 99);
    let _ = state.spawn_agent(AgentSpawn {
        hp: 50.0,
        ..Default::default()
    });
    let tick_before = state.tick;

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();

    let (handle, step_tx, phase_rx) = StepperHandle::new();
    let cfg = DebugConfig {
        tick_stepper: Some(handle),
        ..Default::default()
    };

    let driver = thread::spawn(move || {
        engine_rules::step::step(
            &mut state,
            &mut scratch,
            &mut events,
            &mut views,
            &UtilityBackend,
            &cascade,
            &cfg,
        );
        // Return tick so we can check it didn't advance.
        state.tick
    });

    let mut phases_seen: Vec<Phase> = Vec::new();
    while let Ok(phase) = phase_rx.recv() {
        phases_seen.push(phase);
        if phase == Phase::AfterMaskFill {
            // Abort after first real phase — tick advance should be skipped.
            step_tx.send(Step::Abort).unwrap();
            break;
        } else {
            step_tx.send(Step::Continue).unwrap();
        }
    }

    let tick_after = driver.join().unwrap();
    // After Abort, the tick counter must NOT have advanced.
    assert_eq!(
        tick_after, tick_before,
        "tick should not advance after Abort"
    );
    assert!(
        !phases_seen.contains(&Phase::TickEnd),
        "TickEnd should not fire after Abort"
    );
}
