use engine::debug::causal_tree::CausalTree;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;

#[test]
fn tree_groups_caused_events_under_root() {
    let actor  = AgentId::new(1).unwrap();
    let target = AgentId::new(2).unwrap();

    let mut ring: EventRing<Event> = EventRing::with_cap(64);

    // Root: an attack at tick 1.
    let root_id = ring.push(Event::AgentAttacked {
        actor,
        target,
        damage: 10.0,
        tick: 1,
    });

    // Child: a death caused by the attack (same tick).
    let child_id = ring.push_caused_by(
        Event::AgentDied { agent_id: target, tick: 1 },
        root_id,
    );

    let tree = CausalTree::build(&ring);

    assert_eq!(tree.roots(), &[root_id]);
    assert_eq!(tree.children_of(root_id), &[child_id]);
    assert!(tree.children_of(child_id).is_empty());

    // event() accessor returns the payload
    assert!(matches!(
        tree.event(root_id),
        Some(Event::AgentAttacked { damage, .. }) if (*damage - 10.0).abs() < f32::EPSILON
    ));
    assert!(matches!(
        tree.event(child_id),
        Some(Event::AgentDied { agent_id, .. }) if *agent_id == target
    ));
}
