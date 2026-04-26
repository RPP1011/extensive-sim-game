//! Property: for every (mask, actions) pair where some action's mask bit is
//! false, `MaskValidityInvariant::check_with_scratch` reports a violation.
//! Conversely, when all action bits are true, it reports none.
use engine_data::entities::CreatureType;
use engine::invariant::MaskValidityInvariant;
use engine::mask::MicroKind;
use engine::policy::{Action, ActionKind, MicroTarget};
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch; // Plan B1' Task 11: SimScratch re-exported from scratch
use glam::Vec3;
use proptest::prelude::*;

/// All 18 MicroKind variants in ordinal order for proptest indexing.
const ALL_MICROS: &[MicroKind] = MicroKind::ALL;

fn arb_micro_kind() -> impl Strategy<Value = MicroKind> {
    (0u8..18).prop_map(|i| ALL_MICROS[i as usize])
}

/// Set up a state with `n_agents` spawned; wire a fresh scratch sized to cap.
fn setup(n_agents: u32) -> (SimState, SimScratch) {
    let cap = n_agents + 2;
    let mut state = SimState::new(cap, 42);
    for i in 0..n_agents {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32 * 3.0, 0.0, 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    let scratch = SimScratch::new(state.agent_cap() as usize);
    (state, scratch)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 5000,
        .. ProptestConfig::default()
    })]

    /// If we hand-set a mask bit pattern and then emit an action whose
    /// bit is false, the invariant MUST report a violation. This is the
    /// contrapositive of the "clean run" test.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
    #[test]
    fn forged_action_is_always_flagged(
        n_agents in 1u32..=6,
        agent_ord in 0u32..6,
        kind in arb_micro_kind(),
    ) {
        prop_assume!(agent_ord < n_agents);
        let (state, mut scratch) = setup(n_agents);
        // Mask starts all-false — no bits true.
        scratch.mask.reset();
        scratch.actions.clear();
        // Emit an action whose mask bit is guaranteed false.
        let agent = engine::ids::AgentId::new(agent_ord + 1).unwrap();
        scratch.actions.push(Action {
            agent,
            kind: ActionKind::Micro { kind, target: MicroTarget::None },
        });

        let inv = MaskValidityInvariant::new();
        let v = inv.check_with_scratch(&state, &scratch);
        prop_assert!(
            v.is_some(),
            "expected violation for (agent={}, kind={:?}) with all-false mask",
            agent_ord + 1, kind
        );
        prop_assert_eq!(v.unwrap().invariant, "mask_validity");
    }

    /// Conversely, if we set exactly the bits for the emitted actions and
    /// nothing else, no violation should fire.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
    #[test]
    fn all_mask_bits_set_produces_no_violation(
        n_agents in 1u32..=6,
        picks in proptest::collection::vec((0u32..6, arb_micro_kind()), 1..8),
    ) {
        let (state, mut scratch) = setup(n_agents);
        scratch.mask.reset();
        scratch.actions.clear();
        let n_kinds = MicroKind::ALL.len();
        for &(agent_ord, kind) in &picks {
            if agent_ord >= n_agents { continue; }
            let agent = engine::ids::AgentId::new(agent_ord + 1).unwrap();
            let slot = (agent.raw() - 1) as usize;
            let bit = slot * n_kinds + kind as usize;
            if bit < scratch.mask.micro_kind.len() {
                scratch.mask.micro_kind[bit] = true;
                scratch.actions.push(Action {
                    agent,
                    kind: ActionKind::Micro { kind, target: MicroTarget::None },
                });
            }
        }
        let inv = MaskValidityInvariant::new();
        prop_assert!(
            inv.check_with_scratch(&state, &scratch).is_none(),
            "all-bits-set mask must produce no violation",
        );
    }

    /// Mixed: k actions with bits set, one with bit unset — invariant MUST
    /// fire. Exercises the detector's first-miss-wins property.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
    #[test]
    fn partial_mask_still_catches_forged_action(
        n_agents in 2u32..=6,
        clean_picks in proptest::collection::vec((0u32..6, arb_micro_kind()), 1..4),
        forged_agent in 0u32..6,
        forged_kind in arb_micro_kind(),
    ) {
        prop_assume!(forged_agent < n_agents);
        let (state, mut scratch) = setup(n_agents);
        scratch.mask.reset();
        scratch.actions.clear();
        let n_kinds = MicroKind::ALL.len();
        // Clean actions: set their bits, push the action.
        for &(agent_ord, kind) in &clean_picks {
            if agent_ord >= n_agents { continue; }
            let agent = engine::ids::AgentId::new(agent_ord + 1).unwrap();
            let slot = (agent.raw() - 1) as usize;
            let bit = slot * n_kinds + kind as usize;
            if bit < scratch.mask.micro_kind.len() {
                scratch.mask.micro_kind[bit] = true;
                scratch.actions.push(Action {
                    agent,
                    kind: ActionKind::Micro { kind, target: MicroTarget::None },
                });
            }
        }
        // Forged action: explicitly clear its bit, push the action.
        let f_agent = engine::ids::AgentId::new(forged_agent + 1).unwrap();
        let f_slot = (f_agent.raw() - 1) as usize;
        let f_bit = f_slot * n_kinds + forged_kind as usize;
        if f_bit < scratch.mask.micro_kind.len() {
            scratch.mask.micro_kind[f_bit] = false;
            scratch.actions.push(Action {
                agent: f_agent,
                kind: ActionKind::Micro { kind: forged_kind, target: MicroTarget::None },
            });
            let inv = MaskValidityInvariant::new();
            prop_assert!(
                inv.check_with_scratch(&state, &scratch).is_some(),
                "partial mask with one forged action must flag it",
            );
        }
    }
}
