//! Combat Foundation Task 3 — slow decrement + `SlowExpired` emission.
//! Also asserts that `slow_factor_q8` is cleared on expiry.

use engine::ability::expire::tick_start;
use engine::event::{Event, EventRing};
use engine::state::{AgentSpawn, SimState};

fn count_slow_expired(ring: &EventRing) -> usize {
    ring.iter().filter(|e| matches!(e, Event::SlowExpired { .. })).count()
}

#[test]
fn slow_decrements_and_emits_once_then_clears_factor() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_slow_remaining(a, 2);
    state.set_agent_slow_factor_q8(a, 51); // ~0.2× speed

    // tick 0: 2 → 1
    tick_start(&mut state, &mut events);
    assert_eq!(state.agent_slow_remaining(a), Some(1));
    assert_eq!(state.agent_slow_factor_q8(a), Some(51));
    assert_eq!(count_slow_expired(&events), 0);
    state.tick += 1;

    // tick 1: 1 → 0, expiry + factor zeroed.
    tick_start(&mut state, &mut events);
    assert_eq!(state.agent_slow_remaining(a), Some(0));
    assert_eq!(state.agent_slow_factor_q8(a), Some(0));
    assert_eq!(count_slow_expired(&events), 1);
    state.tick += 1;

    // No re-fire across further ticks.
    for _ in 0..3 {
        tick_start(&mut state, &mut events);
        state.tick += 1;
    }
    assert_eq!(count_slow_expired(&events), 1, "SlowExpired must not double-fire");
}

#[test]
fn slow_expired_carries_current_tick() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_slow_remaining(a, 1);
    state.set_agent_slow_factor_q8(a, 128);
    state.tick = 7;

    tick_start(&mut state, &mut events);
    let ev = events.iter().find_map(|e| match e {
        Event::SlowExpired { agent_id, tick } => Some((*agent_id, *tick)),
        _ => None,
    });
    assert_eq!(ev, Some((a, 7)));
}
