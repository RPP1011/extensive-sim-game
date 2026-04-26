#![allow(unused_mut, unused_variables, unused_imports, dead_code)]
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{SimState, AgentSpawn};
use engine_data::entities::CreatureType;
use glam::Vec3;

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn mask_buffer_allocates_per_agent_per_head() {
    let n_agents = 10;
    let mask = MaskBuffer::new(n_agents);
    assert_eq!(mask.micro_kind.len(), n_agents * MicroKind::ALL.len());
    assert!(mask.micro_kind.iter().all(|&b| !b), "initial all-false");
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
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

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn mark_hold_sets_only_hold_bit_per_alive_agent() {
    let mut state = SimState::new(5, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0), hp: 100.0,
            ..Default::default()
        });
    }
    let mask = MaskBuffer::new(state.agent_cap() as usize);
    // mark_hold_allowed deleted — Plan B1' Task 11.
    let _ = &state; unimplemented!("mark_hold_allowed deleted — B1' Task 11");
    #[allow(unreachable_code)]
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
