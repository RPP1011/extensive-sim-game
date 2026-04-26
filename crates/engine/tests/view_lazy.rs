//! LazyView trait-shape tests.
//!
//! These tests exercise the trait surface (`compute`, `is_stale`,
//! `invalidate_on_events`, and the declared `invalidated_by` kinds) by driving
//! the view manually with a synthetic `EventRing<Event>`.
//!
//! `lazy_view_wired_into_step_full` is the integration canary: it calls the
//! real `engine_rules::step::step` and asserts the view is marked stale after
//! an `AgentMoved` event is emitted. This requires `ViewRegistry` to hold a
//! `NearestEnemyLazy` field and `step` to call `invalidate_lazy_views` after
//! `fold_all` (Phase 5.5).

use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::policy::UtilityBackend;
use engine::scratch::SimScratch;
use engine::state::{AgentSpawn, SimState};
use engine::view::{LazyView, NearestEnemyLazy};
use engine_rules::{step::step, SimCascadeRegistry, ViewRegistry, SerialBackend};
use engine::debug::DebugConfig;
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
    let _ = std::mem::size_of::<SimCascadeRegistry>();
}

/// Integration canary: `LazyView` is now wired into `step` via Phase 5.5
/// (`views.invalidate_lazy_views`). The tick pipeline automatically calls
/// `invalidate_on_events` on all lazy views after emitting events.
///
/// Shape: spawn two agents, compute the view (fresh), run one `step` with
/// UtilityBackend (which moves them toward each other, emitting `AgentMoved`),
/// then assert the view is now stale because `invalidate_lazy_views` ran.
#[test]
fn lazy_view_wired_into_step_full() {
    let cap = 4;
    let mut state = SimState::new(cap, 42);
    let mut scratch = SimScratch::new(cap as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let cascade = SimCascadeRegistry::new();
    let mut views = ViewRegistry::new_with_cap(cap as usize);
    let _ = spawn_two_away(&mut state);

    // Compute the lazy view so it starts fresh.
    views.nearest_enemy_lazy.compute(&state);
    assert!(!views.nearest_enemy_lazy.is_stale(),
        "view must be fresh after explicit compute");

    // Run one tick. UtilityBackend causes agents to move toward each other,
    // which emits AgentMoved — invalidating the nearest-enemy view via Phase 5.5.
    let mut backend = SerialBackend;
    step(&mut backend, &mut state, &mut scratch, &mut events, &mut views,
         &UtilityBackend, &cascade, &DebugConfig::default());

    assert!(views.nearest_enemy_lazy.is_stale(),
        "after step with UtilityBackend causing AgentMoved, view must be stale — \
         LazyView is now wired into step_full via Phase 5.5 (invalidate_lazy_views)");
}
