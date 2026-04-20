use engine::cascade::{CascadeHandler, CascadeRegistry, EventKindId, Lane};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting(Arc<AtomicUsize>);
impl CascadeHandler for Counting {
    fn trigger(&self) -> EventKindId { EventKindId::AgentAttacked }
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, _e: &Event, _s: &mut SimState, _r: &mut EventRing) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn registered_handler_fires_on_matching_event() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Counting(hits.clone()));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(16);
    let evt = Event::AgentAttacked { actor: a, target: a, damage: 0.0, tick: 0 };

    reg.dispatch(&evt, &mut state, &mut ring);
    assert_eq!(hits.load(Ordering::Relaxed), 1);
}

#[test]
fn handler_not_fired_for_non_matching_kind() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Counting(hits.clone()));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(16);

    reg.dispatch(&Event::AgentDied { agent_id: a, tick: 0 }, &mut state, &mut ring);
    assert_eq!(hits.load(Ordering::Relaxed), 0);
}

#[test]
fn multiple_handlers_same_kind_all_fire() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Counting(hits.clone()));
    reg.register(Counting(hits.clone()));
    reg.register(Counting(hits.clone()));

    let mut state = SimState::new(2, 42);
    let a = AgentId::new(1).unwrap();
    let mut ring = EventRing::with_cap(16);
    reg.dispatch(
        &Event::AgentAttacked { actor: a, target: a, damage: 0.0, tick: 0 },
        &mut state, &mut ring
    );
    assert_eq!(hits.load(Ordering::Relaxed), 3);
}
