use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::invariant::{Invariant, PoolNonOverlapInvariant};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn pool_non_overlap_holds_for_healthy_spawns() {
    let mut state = SimState::new(4, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    let inv = PoolNonOverlapInvariant;
    let events = EventRing::<Event>::with_cap(8);
    assert!(inv.check(&state, &events).is_none());
}

#[test]
fn pool_non_overlap_holds_after_kill_and_respawn() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    state.kill_agent(a);
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::X, hp: 100.0,
        ..Default::default()
    }).unwrap();

    let inv = PoolNonOverlapInvariant;
    let events = EventRing::<Event>::with_cap(8);
    assert!(inv.check(&state, &events).is_none());
}

#[test]
fn pool_non_overlap_detects_alive_slot_also_in_freelist() {
    // Fault injection: spawn agent 1 (alive), then corrupt the freelist by
    // pushing slot 1 into it directly. Alive AND in freelist = overlap.
    // This test PROVES the invariant check actually runs — if the check
    // body is replaced with `|| true` or `return None`, this test fails.
    let mut state = SimState::new(4, 42);
    let _a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();

    // Corrupt the pool: push raw id 1 into the freelist while it's alive.
    state.pool_mut_for_test().force_push_freelist_for_test(1);

    let inv = PoolNonOverlapInvariant;
    let events = EventRing::<Event>::with_cap(8);
    let v = inv.check(&state, &events);
    assert!(v.is_some(), "invariant must flag alive/freelist overlap");
    let v = v.unwrap();
    assert_eq!(v.invariant, "pool_non_overlap");
}

#[test]
fn pool_non_overlap_detects_duplicate_freelist_entries() {
    // Fault injection: kill agent 1, then push slot 1 into the freelist
    // a SECOND time. Duplicate entries in the freelist would cause a
    // subsequent `alloc()` to hand out the same slot twice. Invariant<Event>
    // must fire.
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    state.kill_agent(a); // slot 1 now in freelist via kill
    state.pool_mut_for_test().force_push_freelist_for_test(1); // duplicate

    let inv = PoolNonOverlapInvariant;
    let events = EventRing::<Event>::with_cap(8);
    let v = inv.check(&state, &events);
    assert!(v.is_some(), "invariant must flag duplicate freelist entry");
}
