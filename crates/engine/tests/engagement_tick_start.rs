//! Combat Foundation Task 3 — engagement update.
//!
//! Task 139 retired the tick-start tentative-commit pass in favour of
//! event-driven physics (`crate::engagement::*`) plus a
//! `@materialized view engaged_with`. The legacy three-agent
//! "tentative-commit drops A when B picks C" case is no longer
//! enforced — the new pipeline commits unilaterally on the mover's
//! side. These tests exercise the post-refactor surface via the
//! `recompute_all_engagements` shim (the entry point retained for
//! fixtures that predate the event-driven pipeline).

use engine::creature::CreatureType;
use engine::engagement::recompute_all_engagements;
use engine::event::EventRing;
use engine::state::{AgentSpawn, SimState};
use engine_rules::config::Config;
use glam::Vec3;

fn run_tick_start(state: &mut SimState, events: &mut EventRing) {
    recompute_all_engagements(state, events);
}

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0, ..Default::default() }).unwrap()
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
    // Outside engagement range (default 2.0m).
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));
    let engagement_range = Config::default().combat.engagement_range;
    assert!(Vec3::new(0.0, 0.0, 0.0).distance(Vec3::new(3.0, 0.0, 0.0)) > engagement_range);
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
fn three_agent_unilateral_commit_pins_closest_pair() {
    // A (Human) at 0. B (Wolf) at 1.0. D (Dragon) at 1.4.
    //
    // Distances: |B-A|=1.0; |B-D|=0.4; |A-D|=1.4.
    // Nearest-hostile pick per agent:
    //   A → B (1.0m) vs D (1.4m) → B
    //   B → A (1.0m) vs D (0.4m) → D
    //   D → A (1.4m) vs B (0.4m) → B
    //
    // Task 139 simplification: each agent commits unilaterally to its
    // nearest hostile. The mutual-commit invariant the retired
    // `tick_start` enforced ("A.engaged == Some(B) ⇔ B.engaged ==
    // Some(A)") is softened — slot iteration order decides who
    // "wins" when three agents race.
    //
    // `recompute_all_engagements` iterates `agents_alive()` in slot
    // order (A, B, D). A commits to B (bidirectional insert) first,
    // giving `{A↔B}`. Then B recomputes and prefers D — the
    // `recompute_engagement_for` path breaks `A↔B` (since B's
    // partner changes) and commits `B↔D`, leaving A's slot cleared.
    // D recomputes last and prefers B, which matches, so `{B↔D}`
    // stands. End state: A unengaged, B↔D engaged.
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human,  Vec3::new(0.0, 0.0, 0.0));
    let b = spawn(&mut state, CreatureType::Wolf,   Vec3::new(1.0, 0.0, 0.0));
    let d = spawn(&mut state, CreatureType::Dragon, Vec3::new(1.4, 0.0, 0.0));
    run_tick_start(&mut state, &mut events);
    assert_eq!(state.agent_engaged_with(a), None, "A lost its B pairing when B picked D");
    assert_eq!(state.agent_engaged_with(b), Some(d));
    assert_eq!(state.agent_engaged_with(d), Some(b));
}
