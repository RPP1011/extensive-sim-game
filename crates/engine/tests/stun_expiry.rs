//! Combat Foundation Task 3 — stun decrement + `StunExpired` emission.

use engine::ability::expire::tick_start;
use engine::event::{Event, EventRing};
use engine::state::{AgentSpawn, SimState};

fn count_stun_expired(ring: &EventRing) -> usize {
    ring.iter().filter(|e| matches!(e, Event::StunExpired { .. })).count()
}

#[test]
fn stun_decrements_to_zero_and_emits_stun_expired_once() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_stun_remaining(a, 3);

    // tick 0: 3 → 2 (no expiry)
    tick_start(&mut state, &mut events);
    assert_eq!(state.agent_stun_remaining(a), Some(2));
    assert_eq!(count_stun_expired(&events), 0);
    state.tick += 1;

    // tick 1: 2 → 1
    tick_start(&mut state, &mut events);
    assert_eq!(state.agent_stun_remaining(a), Some(1));
    assert_eq!(count_stun_expired(&events), 0);
    state.tick += 1;

    // tick 2: 1 → 0, expiry event emitted
    tick_start(&mut state, &mut events);
    assert_eq!(state.agent_stun_remaining(a), Some(0));
    assert_eq!(count_stun_expired(&events), 1);
    state.tick += 1;

    // Subsequent ticks: no re-fire. We ran 5 more ticks and verify only one
    // StunExpired ever appears in the ring.
    for _ in 0..5 {
        tick_start(&mut state, &mut events);
        state.tick += 1;
    }
    assert_eq!(count_stun_expired(&events), 1, "StunExpired must not double-fire");
}

#[test]
fn stun_expired_carries_current_tick() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_stun_remaining(a, 1);
    state.tick = 42;

    tick_start(&mut state, &mut events);
    let ev = events.iter().find_map(|e| match e {
        Event::StunExpired { agent_id, tick } => Some((*agent_id, *tick)),
        _ => None,
    });
    assert_eq!(ev, Some((a, 42)));
}
