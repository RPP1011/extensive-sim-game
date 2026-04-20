//! Combat Foundation Task 3 — engagement update inside the unified
//! tick-start phase. `ability::expire::tick_start` should commit mutual
//! nearest-hostile pairings only.

use engine::ability::expire::{tick_start, ENGAGEMENT_RANGE};
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use glam::Vec3;

fn run_tick_start(state: &mut SimState, events: &mut EventRing) {
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    tick_start(state, &mut scratch, events);
}

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0 }).unwrap()
}

#[test]
fn two_hostile_agents_inside_range_engage_each_other() {
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 0.0));
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.5, 0.0, 0.0));
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), Some(b));
    assert_eq!(state.agent_engaged_with(b), Some(a));
}

#[test]
fn same_species_agents_do_not_engage() {
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 0.0));
    let b = spawn(&mut state, CreatureType::Human, Vec3::new(1.0, 0.0, 0.0));
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), None);
    assert_eq!(state.agent_engaged_with(b), None);
}

#[test]
fn agents_outside_engagement_range_do_not_engage() {
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 0.0));
    // Outside 2.0m engagement range.
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));
    assert!(Vec3::new(0.0, 0.0, 0.0).distance(Vec3::new(3.0, 0.0, 0.0)) > ENGAGEMENT_RANGE);
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), None);
    assert_eq!(state.agent_engaged_with(b), None);
}

#[test]
fn engagement_clears_when_partners_separate() {
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 0.0));
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), Some(b));
    assert_eq!(state.agent_engaged_with(b), Some(a));

    // Move B to 5m away.
    state.set_agent_pos(b, Vec3::new(5.0, 0.0, 0.0));
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), None);
    assert_eq!(state.agent_engaged_with(b), None);
}

#[test]
fn three_agent_tentative_commit_with_dragon_closer_to_wolf() {
    // A (Human) at 0. B (Wolf) at 1.0. D (Dragon) at 1.4 (closer to B than A).
    // Wolf is hostile to Human AND Dragon (dragons hostile to all).
    // A is hostile to Wolf AND Dragon (dragons hostile to all).
    // Dragon is hostile to all.
    //
    // Distances: |B-A|=1.0; |B-D|=0.4; |A-D|=1.4.
    // Tentative picks (nearest hostile within 2.0m):
    //   A → B (1.0m) vs D (1.4m) → B wins
    //   B → A (1.0m) vs D (0.4m) → D wins
    //   D → A (1.4m) vs B (0.4m) → B wins
    // Mutual commit:
    //   A.picked=B but B.picked=D → A committed None
    //   B.picked=D and D.picked=B → mutual ⇒ both engage each other
    //   D.picked=B and B.picked=D → same mutual pair
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human,  Vec3::new(0.0, 0.0, 0.0));
    let b = spawn(&mut state, CreatureType::Wolf,   Vec3::new(1.0, 0.0, 0.0));
    let d = spawn(&mut state, CreatureType::Dragon, Vec3::new(1.4, 0.0, 0.0));
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), None, "A picked B but B picked D → A committed None");
    assert_eq!(state.agent_engaged_with(b), Some(d));
    assert_eq!(state.agent_engaged_with(d), Some(b));
}
