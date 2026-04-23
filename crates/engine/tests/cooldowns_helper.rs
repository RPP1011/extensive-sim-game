//! Unit test for `SimState::can_cast_ability` — gates cast on both
//! global (GCD) and local (per-ability-slot) cooldowns.

use engine::creature::CreatureType;
use engine::state::entity_pool::AgentSlotPool;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState) -> engine::ids::AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap()
}

#[test]
fn can_cast_ability_ready_when_both_cooldowns_cleared() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state);
    // Fresh agent: both cursors at 0; tick is 0. Ready to cast.
    assert!(state.can_cast_ability(a, 0, 0), "fresh agent should be ready");
}

#[test]
fn can_cast_ability_blocked_by_global() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state);
    // Push global cooldown cursor to tick 5; local stays at 0.
    state.set_agent_cooldown_next_ready(a, 5);
    assert!(
        !state.can_cast_ability(a, 0, 3),
        "global gate should block at tick 3"
    );
    assert!(
        state.can_cast_ability(a, 0, 5),
        "global gate should clear at tick 5"
    );
}

#[test]
fn can_cast_ability_blocked_by_local() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state);
    // Local cooldown for slot 0 at tick 10; global stays at 0.
    let agent_slot = AgentSlotPool::slot_of_agent(a);
    state.ability_cooldowns[agent_slot][0] = 10;
    assert!(
        !state.can_cast_ability(a, 0, 5),
        "local slot 0 blocks at tick 5"
    );
    assert!(
        state.can_cast_ability(a, 1, 5),
        "local slot 1 is ready"
    );
    assert!(
        state.can_cast_ability(a, 0, 10),
        "local slot 0 clears at tick 10"
    );
}
