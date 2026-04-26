//! Integration test for the tick_profile phase timing histogram (Plan 4 Task 5).
//!
//! Verifies that after N ticks with a `TickProfile` installed in `DebugConfig`,
//! every phase produces exactly N nanosecond samples.

use engine::debug::DebugConfig;
use engine::debug::tick_profile::TickProfile;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::scratch::SimScratch;
use engine::state::{AgentSpawn, SimState};
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;
use std::sync::{Arc, Mutex};

const TICK_COUNT: usize = 10;
const EXPECTED_PHASES: &[&str] = &[
    "mask_fill",
    "scoring",
    "action_select",
    "cascade_dispatch",
    "view_fold",
    "tick_end",
];

#[test]
fn profile_records_all_phases_for_n_ticks() {
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

    let profile = Arc::new(Mutex::new(TickProfile::new()));
    let cfg = DebugConfig {
        tick_profile: Some(Arc::clone(&profile)),
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

    let p = profile.lock().unwrap();
    let samples = p.samples();

    for &phase in EXPECTED_PHASES {
        let vec = samples.get(phase).unwrap_or_else(|| {
            panic!("phase '{}' missing from profile samples; present: {:?}", phase, samples.keys().collect::<Vec<_>>())
        });
        assert_eq!(
            vec.len(),
            TICK_COUNT,
            "phase '{}' expected {} samples, got {}",
            phase,
            TICK_COUNT,
            vec.len()
        );
        // Sanity: all samples are non-zero nanoseconds (clocks should always
        // advance, but we allow zero only if the system timer is coarse).
        // We just verify they're all plausible (< 1 second each).
        for &ns in vec.iter() {
            assert!(
                ns < 1_000_000_000,
                "phase '{}' sample {} ns looks implausibly large (>1s)",
                phase,
                ns
            );
        }
    }

    assert_eq!(
        samples.len(),
        EXPECTED_PHASES.len(),
        "unexpected extra phases in profile: {:?}",
        samples.keys().collect::<Vec<_>>()
    );
}

#[test]
fn profile_not_installed_is_zero_cost_no_op() {
    // Confirm that DebugConfig::default() (tick_profile = None) compiles and
    // runs without issue — no samples collected, tick advances normally.
    let mut state = SimState::new(4, 7);
    let _ = state.spawn_agent(AgentSpawn {
        hp: 50.0,
        ..Default::default()
    });
    let tick_before = state.tick;
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();
    let cfg = DebugConfig::default();

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

    assert_eq!(state.tick, tick_before + 1, "tick should have advanced");
}
