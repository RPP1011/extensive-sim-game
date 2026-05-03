//! Per-agent Relationships (state.md §Relationship). Task J.

use engine::ids::AgentId;
use engine::state::agent_types::Relationship;
use engine::state::{AgentSpawn, SimState};
use smallvec::SmallVec;

#[test]
fn spawn_defaults_relationships_to_empty() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let r = state.agent_relationships(a).unwrap();
    assert!(r.is_empty());
}

#[test]
fn push_and_read_relationship() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let other = AgentId::new(42).unwrap();
    let rel = Relationship {
        other,
        valence_q8:   -64,
        tenure_ticks: 500,
    };
    state.push_agent_relationship(a, rel);
    let rs = state.agent_relationships(a).unwrap();
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0], rel);
}

#[test]
fn cold_slice_length_matches_cap() {
    let state = SimState::new(8, 42);
    let slice: &[SmallVec<[Relationship; 8]>] = state.cold_relationships();
    assert_eq!(slice.len(), 8);
    assert!(slice.iter().all(|v| v.is_empty()));
}
