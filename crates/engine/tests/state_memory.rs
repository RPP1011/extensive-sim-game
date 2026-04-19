//! Memory ring-buffer stub (state.md §Memory). Task I. Storage only.

use engine::ids::AgentId;
use engine::state::agent_types::MemoryEvent;
use engine::state::{AgentSpawn, SimState};
use smallvec::SmallVec;

#[test]
fn spawn_defaults_memory_to_empty() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let m = state.agent_memory(a).unwrap();
    assert!(m.is_empty());
}

#[test]
fn push_and_read_memory_events() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let src = AgentId::new(2).unwrap();
    let ev = MemoryEvent {
        source:        src,
        kind:          3,
        payload:       0xCAFE_BABE_DEAD_BEEF,
        confidence_q8: 200,
        tick:          10,
    };
    state.push_agent_memory(a, ev);
    state.push_agent_memory(a, ev);
    let m = state.agent_memory(a).unwrap();
    assert_eq!(m.len(), 2);
    assert_eq!(m[0], ev);
}

#[test]
fn cold_slice_length_matches_cap() {
    let state = SimState::new(8, 42);
    let slice: &[SmallVec<[MemoryEvent; 64]>] = state.cold_memory();
    assert_eq!(slice.len(), 8);
    assert!(slice.iter().all(|v| v.is_empty()));
}

#[test]
fn kill_clears_memory() {
    let mut state = SimState::new(2, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.push_agent_memory(
        a,
        MemoryEvent {
            source:        AgentId::new(1).unwrap(),
            kind:          0,
            payload:       0,
            confidence_q8: 0,
            tick:          0,
        },
    );
    state.kill_agent(a);
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(b.raw(), a.raw());
    assert!(state.agent_memory(b).unwrap().is_empty());
}
