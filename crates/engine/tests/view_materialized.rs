use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::state::{AgentSpawn, SimState};
use engine::view::materialized::{DamageTaken, MaterializedView};
use glam::Vec3;

#[test]
fn damage_taken_accumulates_from_agent_attacked_events() {
    let mut state = SimState::new(10, 42);
    let mut events = EventRing::with_cap(100);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO,
        hp: 100.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::X,
        hp: 100.0,
    }).unwrap();

    let mut dmg = DamageTaken::new(state.agent_cap() as usize);
    events.push(Event::AgentAttacked { actor: b, target: a, damage: 20.0, tick: 1 });
    events.push(Event::AgentAttacked { actor: b, target: a, damage: 15.0, tick: 2 });
    dmg.fold(&events);
    assert_eq!(dmg.value(a), 35.0);
    assert_eq!(dmg.value(b), 0.0);
}
