//! Wave 3 ToM Phase 4 behavioral pin: the GPU dispatcher's chronicle
//! arm for `EffectOp::EraseBelief` (kind 38 → 69) writes the expected
//! (kind, actor, target, subject_idx, fields) tuple for a single
//! self-cast.
//!
//! ## Shape
//!
//! `EffectOp::EraseBelief { subject_idx: u32, fields: u8 }` — caster
//! clears specific fields of `target`'s beliefs about `subject_idx` per
//! the `fields` bitset (bit 0 = pos, 1 = type, 2 = tick, 3 = confidence,
//! 4 = suspicion, 5 = flags). The dispatcher writes a chronicle record
//! (kind=69 EffectEraseBeliefApplied):
//!
//!   - slot 0 = 69 (kind tag)
//!   - slot 1 = tick
//!   - slot 2 = caster_slot
//!   - slot 3 = target_slot       (the OBSERVER whose row caster clears)
//!   - slot 4 = subject_idx       (= payload_a)
//!   - slot 5 = fields            (= payload_b low byte)
//!   - slots 6..9 = zero

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

#[test]
fn erase_belief_chronicle_record_carries_correct_kind_and_payloads() {
    const SUBJECT_IDX: u32 = 4;
    const FIELDS: u8 = 0b00111111; // all 6 BeliefState columns
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 1;

    let mut builder = AbilityRegistryBuilder::new();
    let erase_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::EraseBelief {
            subject_idx: SUBJECT_IDX,
            fields: FIELDS,
        }],
    ));
    let registry = builder.build();

    let per_agent_levels = vec![erase_id.raw()];
    let per_agent_stats = vec![PerAgentStats::default(); N_AGENTS as usize];

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &per_agent_levels,
        &per_agent_stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!("[erase_belief_chronicle_pin] skipping: no wgpu adapter available");
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
    assert_eq!(r[0], 69, "EraseBelief: kind tag — EffectEraseBeliefApplied");
    assert_eq!(r[1], TICK, "EraseBelief: tick");
    assert_eq!(r[2], 0, "EraseBelief: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], 0,
        "EraseBelief: target slot — implicit-target rule routes target = caster",
    );
    assert_eq!(
        r[4], SUBJECT_IDX,
        "EraseBelief: subject_idx at slot 4 (= payload_a)",
    );
    assert_eq!(
        r[5], FIELDS as u32,
        "EraseBelief: fields bitset at slot 5 (= payload_b low byte; \
         bit 0 = pos, 1 = type, 2 = tick, 3 = confidence, 4 = suspicion, 5 = flags)",
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "EraseBelief: tail word {i} must be zero");
    }
}
