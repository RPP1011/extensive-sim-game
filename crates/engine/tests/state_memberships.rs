//! Memberships (state.md §Membership). Task G. Storage only.

use engine::ids::{AgentId, GroupId};
use engine::state::agent_types::{GroupRole, Membership};
use engine::state::{AgentSpawn, SimState};
use smallvec::SmallVec;

#[test]
fn spawn_defaults_memberships_to_empty() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let m = state.agent_memberships(a).unwrap();
    assert!(m.is_empty());
}

#[test]
fn push_and_read_memberships() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let g = GroupId::new(7).unwrap();
    let m = Membership {
        group:       g,
        role:        GroupRole::Leader,
        joined_tick: 100,
        standing_q8: 200,
    };
    state.push_agent_membership(a, m);
    let ms = state.agent_memberships(a).unwrap();
    assert_eq!(ms.len(), 1);
    assert_eq!(ms[0], m);
}

#[test]
fn group_role_variants_discriminants_stable() {
    assert_eq!(GroupRole::Member     as u8, 0);
    assert_eq!(GroupRole::Officer    as u8, 1);
    assert_eq!(GroupRole::Leader     as u8, 2);
    assert_eq!(GroupRole::Founder    as u8, 3);
    assert_eq!(GroupRole::Apprentice as u8, 4);
    assert_eq!(GroupRole::Outcast    as u8, 5);
}

#[test]
fn cold_slice_length_matches_cap() {
    let state = SimState::new(8, 42);
    let slice: &[SmallVec<[Membership; 4]>] = state.cold_memberships();
    assert_eq!(slice.len(), 8);
    assert!(slice.iter().all(|v| v.is_empty()));
}

#[test]
fn kill_and_respawn_clears_memberships() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.push_agent_membership(
        a,
        Membership {
            group:       GroupId::new(1).unwrap(),
            role:        GroupRole::Founder,
            joined_tick: 0,
            standing_q8: 0,
        },
    );
    assert_eq!(state.agent_memberships(a).unwrap().len(), 1);
    state.kill_agent(a);
    let _src = AgentId::new(1).unwrap(); // silence unused
    // Respawn reuses slot — and must see a fresh empty list.
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(b.raw(), a.raw());
    assert!(state.agent_memberships(b).unwrap().is_empty());
}
