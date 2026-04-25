//! Memory view — `@per_entity_ring(K=64)`. Subsystem 2 Phase 4
//! retired the `cold_memory: SmallVec<[MemoryEvent; 64]>` shape in
//! favour of the generated `state.views.memory` ring. The tests below
//! exercise the new API surface: `Memory::push` / `Memory::entries` /
//! `Memory::cursor`.

use engine::generated::views::memory::MemoryEntry;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn spawn_defaults_memory_to_empty() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(state.views.memory.cursor(a), 0);
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn push_and_read_memory_events() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let src = AgentId::new(2).unwrap();
    let entry = MemoryEntry {
        source: src.raw(),
        value: 1.0,
        anchor_tick: 10,
    };
    state.views.memory.push(a.raw(), entry);
    state.views.memory.push(a.raw(), entry);
    assert_eq!(state.views.memory.cursor(a), 2);
    let row = state.views.memory.entries(a).expect("owner row present");
    assert_eq!(row[0], entry);
    assert_eq!(row[1], entry);
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn empty_memory_returns_none_for_unwritten_owners() {
    let state = SimState::new(8, 42);
    // Unwritten owner — the ring is grown on demand, so `entries`
    // returns None until the first `push`.
    let a = AgentId::new(1).unwrap();
    assert!(state.views.memory.entries(a).is_none());
    assert_eq!(state.views.memory.cursor(a), 0);
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn memory_does_not_auto_clear_on_respawn_into_same_slot() {
    // Contract drift vs the retired `cold_memory` shape: `kill_agent` no
    // longer wipes the view ring (the view is SoA across the whole
    // session, not per-agent-instance storage). Consumers that need a
    // fresh ring on respawn explicitly reset the slot. Documenting the
    // new contract with this test — bumping to a stronger clear-on-
    // respawn contract is tracked as a follow-up if a scenario needs it.
    let mut state = SimState::new(2, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.views.memory.push(
        a.raw(),
        MemoryEntry {
            source: AgentId::new(1).unwrap().raw(),
            value: 1.0,
            anchor_tick: 0,
        },
    );
    assert_eq!(state.views.memory.cursor(a), 1);

    state.kill_agent(a);
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(b.raw(), a.raw());
    // Cursor + row survive respawn — the view is session-scoped.
    assert_eq!(state.views.memory.cursor(b), 1);
}
