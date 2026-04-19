//! StatusEffect SoA + supporting types (state.md §StatusEffect). Task C.

use engine::ids::AgentId;
use engine::state::agent_types::{StatusEffect, StatusEffectKind};
use engine::state::{AgentSpawn, SimState};
use smallvec::SmallVec;

#[test]
fn spawn_defaults_status_effects_to_empty() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let effects = state.agent_status_effects(a).unwrap();
    assert!(effects.is_empty());
}

#[test]
fn push_and_read_status_effects() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let src = AgentId::new(1).unwrap();
    let fx = StatusEffect {
        kind:            StatusEffectKind::Slow,
        source:          src,
        remaining_ticks: 30,
        payload_q8:      64, // slow factor ≈ 0.25
    };
    state.push_agent_status_effect(a, fx);
    let effects = state.agent_status_effects(a).unwrap();
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0], fx);
}

#[test]
fn status_effect_kinds_are_exhaustive() {
    // Make sure the 8 variants state.md promises are reachable.
    let kinds: [StatusEffectKind; 8] = [
        StatusEffectKind::Stun,
        StatusEffectKind::Slow,
        StatusEffectKind::Root,
        StatusEffectKind::Silence,
        StatusEffectKind::Dot,
        StatusEffectKind::Hot,
        StatusEffectKind::Buff,
        StatusEffectKind::Debuff,
    ];
    // Different discriminants → set semantics hold.
    let mut u8s: [u8; 8] = std::array::from_fn(|i| kinds[i] as u8);
    u8s.sort_unstable();
    for (i, v) in u8s.iter().enumerate() {
        assert_eq!(*v as usize, i, "discriminants must be 0..=7");
    }
}

#[test]
fn cold_slice_length_matches_cap() {
    let state = SimState::new(8, 42);
    let slice: &[SmallVec<[StatusEffect; 8]>] = state.cold_status_effects();
    assert_eq!(slice.len(), 8);
    assert!(slice.iter().all(|v| v.is_empty()));
}
