use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine::view::{LazyView, NearestEnemyLazy};
use glam::Vec3;

fn spawn_two_away(state: &mut SimState) -> (AgentId, AgentId) {
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(5.0, 0.0, 10.0),
            hp: 100.0,
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

    let mut ring = EventRing::with_cap(8);
    ring.push(Event::AgentMoved {
        agent_id: AgentId::new(1).unwrap(),
        from: Vec3::ZERO,
        to: Vec3::X,
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

    let mut ring = EventRing::with_cap(8);
    ring.push(Event::ChronicleEntry {
        tick: 0,
        template_id: 0,
    });
    view.invalidate_on_events(&ring);
    assert!(!view.is_stale(), "chronicle events don't affect positions");
}

// Silence unused-import warnings for spec-mandated imports not directly used
// in any single test (they document the engine surface the view integrates with).
#[allow(dead_code)]
fn _unused_imports_anchor() {
    let _ = std::mem::size_of::<CascadeRegistry>();
    let _ = std::mem::size_of::<SimScratch>();
}
