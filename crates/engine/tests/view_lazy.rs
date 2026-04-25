//! LazyView trait-shape tests.
//!
//! NOTE (audit MEDIUM #11, 2026-04-19): `LazyView` is not yet wired into
//! `step_full`. The engine's tick pipeline folds `MaterializedView<Event>` via
//! `views: &mut [&mut dyn MaterializedView<Event>]` only — there is no automatic
//! `invalidate_on_events` dispatch against lazy views. These tests exercise
//! the trait surface (`compute`, `is_stale`, `invalidate_on_events`, and
//! the declared `invalidated_by` kinds) by driving the view manually with
//! a synthetic `EventRing<Event>`. The `#[ignore]` test `lazy_view_wired_into_step_full`
//! below is the canary: when LazyView integration lands in `step_full`, that
//! test should be un-ignored and will pass.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::view::{LazyView, NearestEnemyLazy};
use glam::Vec3;

fn spawn_two_away(state: &mut SimState) -> (AgentId, AgentId) {
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(5.0, 0.0, 10.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    (a, b)
}

#[test]
fn fresh_view_is_stale_before_first_compute() {
    let view = NearestEnemyLazy::new(8);
    assert!(view.is_stale());
    // Reading stale → None.
    assert!(view.value(AgentId::new(1).unwrap()).is_none());
}

#[test]
fn compute_populates_and_marks_fresh() {
    let mut state = SimState::new(4, 42);
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let (a, b) = spawn_two_away(&mut state);

    view.compute(&state);
    assert!(!view.is_stale());
    assert_eq!(view.value(a), Some(b));
    assert_eq!(view.value(b), Some(a));
}

#[test]
fn invalidated_by_agent_moved() {
    // LazyView declares which event kinds invalidate it. The engine's dispatch
    // logic compares new events against `invalidated_by()` and flips the
    // staleness marker. For the unit test we simulate this by calling
    // `invalidate_on_events()` directly.
    let mut state = SimState::new(4, 42);
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let (_a, _b) = spawn_two_away(&mut state);
    view.compute(&state);
    assert!(!view.is_stale());

    let mut ring = EventRing::<Event>::with_cap(8);
    ring.push(Event::AgentMoved {
        actor: AgentId::new(1).unwrap(),
        from: Vec3::ZERO,
        location: Vec3::X,
        tick: 0,
    });
    view.invalidate_on_events(&ring);
    assert!(view.is_stale());
}

#[test]
fn does_not_invalidate_on_unrelated_event() {
    let mut state = SimState::new(4, 42);
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let (_a, _b) = spawn_two_away(&mut state);
    view.compute(&state);

    let mut ring = EventRing::<Event>::with_cap(8);
    // Use agent id 1 (spawned by spawn_two_away above). The agent/target
    // payload is structurally required since task chronicle-mvp extended
    // ChronicleEntry with (agent, target) for the prose renderer.
    ring.push(Event::ChronicleEntry {
        tick: 0,
        template_id: 0,
        agent: engine::ids::AgentId::new(1).unwrap(),
        target: engine::ids::AgentId::new(1).unwrap(),
    });
    view.invalidate_on_events(&ring);
    assert!(!view.is_stale(), "chronicle events don't affect positions");
}

// Silence unused-import warnings for spec-mandated imports not directly used
// in any single test (they document the engine surface the view integrates with).
#[allow(dead_code)]
fn _unused_imports_anchor() {
    let _ = std::mem::size_of::<CascadeRegistry<Event>>();
    let _ = std::mem::size_of::<SimScratch>();
}

/// Integration canary: when `LazyView` is wired into `step_full` (so the
/// tick pipeline automatically calls `invalidate_on_events` against the
/// events emitted that tick), un-ignore this test. It currently would
/// fail because `step_full` only folds MaterializedView<Event>, not LazyView.
///
/// Shape: spawn two agents, compute the view, run one `step` with
/// UtilityBackend (which moves them toward each other, emitting
/// `AgentMoved`), then assert the view is stale. Today the view remains
/// fresh because nothing invalidates it automatically.
#[test]
#[ignore = "LazyView not wired into step_full yet (audit MEDIUM #11)"]
fn lazy_view_wired_into_step_full() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let cascade = CascadeRegistry::<Event>::new();
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let _ = spawn_two_away(&mut state);

    view.compute(&state);
    assert!(!view.is_stale());

    // TODO(LazyView integration): this step should invalidate `view` because
    // agents will move and emit AgentMoved. Today nothing wires the view in.
    step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    assert!(view.is_stale(),
        "after step with UtilityBackend causing AgentMoved, view must be stale — \
         indicates LazyView is now wired into step_full");
}
