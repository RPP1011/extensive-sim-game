//! Wave 3 ToM Phase 4 behavioral pin: the GPU dispatcher's chronicle
//! arm for `EffectOp::Decoy` (kind 37 → 68) writes the expected
//! (kind, actor, target, subject_idx, fake_pos) tuple for a single
//! self-cast.
//!
//! ## Shape
//!
//! `EffectOp::Decoy { subject_idx: u32, fake_pos: u32 }` — caster
//! writes attacker-controlled belief values into `target`'s row about
//! `subject_idx`. `fake_pos` is a packed (x_q8, y_q8, z_q8, fake_type)
//! quartet. The dispatcher writes a chronicle record (kind=68
//! EffectDecoyApplied):
//!
//!   - slot 0 = 68 (kind tag)
//!   - slot 1 = tick
//!   - slot 2 = caster_slot
//!   - slot 3 = target_slot       (the OBSERVER whose row caster writes)
//!   - slot 4 = subject_idx       (= payload_a; the agent slot the belief is ABOUT)
//!   - slot 5 = fake_pos          (= payload_b; packed quartet)
//!   - slots 6..9 = zero

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

#[test]
fn decoy_chronicle_record_carries_correct_kind_and_payloads() {
    const SUBJECT_IDX: u32 = 4;
    const FAKE_POS: u32 = 0xDEADBEEF;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 1;

    let mut builder = AbilityRegistryBuilder::new();
    let decoy_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Decoy {
            subject_idx: SUBJECT_IDX,
            fake_pos: FAKE_POS,
        }],
    ));
    let registry = builder.build();

    let per_agent_levels = vec![decoy_id.raw()];
    let per_agent_stats = vec![PerAgentStats::default(); N_AGENTS as usize];

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &per_agent_levels,
        &per_agent_stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!("[decoy_chronicle_pin] skipping: no wgpu adapter available");
            return;
        }
    };

    state.step(TICK);
    let tail = state.read_event_tail();
    let records = state.read_event_ring(tail);

    assert_eq!(
        records.len(),
        N_AGENTS as usize,
        "expected one chronicle record per agent (got {})",
        records.len(),
    );

    let r = &records[0];
    assert_eq!(r[0], 68, "Decoy: kind tag — EffectDecoyApplied");
    assert_eq!(r[1], TICK, "Decoy: tick");
    assert_eq!(r[2], 0, "Decoy: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], 0,
        "Decoy: target slot — implicit-target rule routes target = caster (= the OBSERVER)",
    );
    assert_eq!(
        r[4], SUBJECT_IDX,
        "Decoy: subject_idx at slot 4 (= payload_a; the agent slot the belief is ABOUT)",
    );
    assert_eq!(
        r[5], FAKE_POS,
        "Decoy: fake_pos at slot 5 (= payload_b; packed (x_q8, y_q8, z_q8, fake_type) quartet)",
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Decoy: tail word {i} must be zero");
    }
}
