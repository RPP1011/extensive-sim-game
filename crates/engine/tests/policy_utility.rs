use engine::mask::{MaskBuffer, MicroKind, TargetMask};
use engine::policy::{Action, PolicyBackend, UtilityBackend};
use engine::state::{SimState, AgentSpawn};
use engine::creature::CreatureType;
use glam::Vec3;

#[test]
fn utility_picks_hold_when_only_hold_allowed() {
    let mut state = SimState::new(5, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0), hp: 50.0,
            ..Default::default()
        });
    }
    let mut mask = MaskBuffer::new(state.agent_cap() as usize);
    mask.mark_hold_allowed(&state);
    let target_mask = TargetMask::new(state.agent_cap() as usize);
    let backend = UtilityBackend;
    let mut actions: Vec<Action> = Vec::with_capacity(state.agent_cap() as usize);
    backend.evaluate(&state, &mask, &target_mask, &mut actions);
    assert_eq!(actions.len(), 3);
    for a in &actions {
        assert_eq!(a.micro_kind(), Some(MicroKind::Hold), "utility chose Hold when only Hold allowed");
    }
}

#[test]
fn utility_prefers_eat_when_hp_low_and_eat_allowed() {
    let mut state = SimState::new(2, 42);
    let id = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    // Now lower current hp to 20 (below 30% of max_hp=100 → triggers Eat).
    state.set_agent_hp(id, 20.0);

    let mut mask = MaskBuffer::new(state.agent_cap() as usize);
    let slot = (id.raw() - 1) as usize;
    let nm = MicroKind::ALL.len();
    mask.micro_kind[slot * nm + MicroKind::Hold as usize] = true;
    mask.micro_kind[slot * nm + MicroKind::Eat as usize]  = true;

    let target_mask = TargetMask::new(state.agent_cap() as usize);
    let mut actions: Vec<Action> = Vec::new();
    UtilityBackend.evaluate(&state, &mask, &target_mask, &mut actions);
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].micro_kind(), Some(MicroKind::Eat));
}
