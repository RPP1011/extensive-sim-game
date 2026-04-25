use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::invariant::MaskValidityInvariant;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{SimScratch, step};
use glam::Vec3;

#[test]
fn mask_validity_never_flags_a_clean_utility_run() {
    let mut state = SimState::new(10, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();
    for i in 0..6 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    let inv = MaskValidityInvariant::new();
    for _ in 0..20 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        let violation = inv.check_with_scratch(&state, &scratch);
        assert!(violation.is_none(), "clean run should not violate");
    }
}

#[test]
fn mask_validity_detects_forged_action() {
    use engine::mask::MicroKind;
    use engine::policy::{Action, ActionKind, MicroTarget};

    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();

    scratch.mask.reset();
    scratch.actions.clear();
    scratch.actions.push(Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::Attack,  // no mask bit set → violation
            target: MicroTarget::Agent(a),
        },
    });

    let inv = MaskValidityInvariant::new();
    let v = inv.check_with_scratch(&state, &scratch).expect("violation expected");
    assert_eq!(v.invariant, "mask_validity");
}
