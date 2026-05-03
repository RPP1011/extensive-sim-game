//! Inventory stub (state.md §Inventory, §AgentData.Economic). Task H.

use engine::state::agent_types::Inventory;
use engine::state::{AgentSpawn, SimState};

#[test]
fn spawn_defaults_inventory_to_empty() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let inv = state.agent_inventory(a).unwrap();
    assert_eq!(inv.gold, 0);
    assert_eq!(inv.commodities, [0u16; 8]);
}

#[test]
fn set_and_read_inventory() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let inv = Inventory {
        gold:        1234,
        commodities: {
            let mut c = [0u16; 8];
            c[0] = 50;
            c[7] = 9;
            c
        },
    };
    state.set_agent_inventory(a, inv);
    let got = state.agent_inventory(a).unwrap();
    assert_eq!(got.gold, 1234);
    assert_eq!(got.commodities[0], 50);
    assert_eq!(got.commodities[7], 9);
}

#[test]
fn cold_slice_length_matches_cap() {
    let state = SimState::new(8, 42);
    let slice: &[Inventory] = state.cold_inventory();
    assert_eq!(slice.len(), 8);
    assert!(slice.iter().all(|inv| inv.gold == 0 && inv.commodities == [0; 8]));
}
