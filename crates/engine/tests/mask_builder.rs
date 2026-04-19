use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{SimState, AgentSpawn};
use engine::creature::CreatureType;
use glam::Vec3;

#[test]
fn mask_buffer_allocates_per_agent_per_head() {
    let n_agents = 10;
    let mask = MaskBuffer::new(n_agents);
    assert_eq!(mask.micro_kind.len(), n_agents * MicroKind::ALL.len());
    assert!(mask.micro_kind.iter().all(|&b| !b), "initial all-false");
}

#[test]
fn reset_clears_all_heads() {
    let mut mask = MaskBuffer::new(4);
    // Manually set some bits.
    mask.micro_kind[0] = true;
    mask.target[3] = true;
    mask.reset();
    assert!(mask.micro_kind.iter().all(|&b| !b));
    assert!(mask.target.iter().all(|&b| !b));
}

#[test]
fn mark_hold_sets_only_hold_bit_per_alive_agent() {
    let mut state = SimState::new(5, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0), hp: 100.0,
        });
    }
    let mut mask = MaskBuffer::new(state.agent_cap() as usize);
    mask.mark_hold_allowed(&state);
    for id in state.agents_alive() {
        let slot = (id.raw() - 1) as usize;
        let offset = slot * MicroKind::ALL.len() + MicroKind::Hold as usize;
        assert!(mask.micro_kind[offset], "Hold bit must be set for alive agent");
        // Other bits in this agent's row must still be false.
        for i in 0..MicroKind::ALL.len() {
            if i == MicroKind::Hold as usize { continue; }
            let other_offset = slot * MicroKind::ALL.len() + i;
            assert!(!mask.micro_kind[other_offset], "only Hold is set");
        }
    }
    // Dead slots (index >= 3) should have all-false rows.
    for slot in 3..state.agent_cap() as usize {
        for i in 0..MicroKind::ALL.len() {
            let offset = slot * MicroKind::ALL.len() + i;
            assert!(!mask.micro_kind[offset], "dead-slot row must be all false");
        }
    }
}
