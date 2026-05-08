//! Wave 3 ToM Phase 3.5 behavioral pin: the GPU dispatcher's chronicle
//! arm for `EffectOp::Reveal` (kind 35 → 66) writes the expected
//! (kind, actor, target, subject_idx) tuple for a single self-cast.
//!
//! ## Shape
//!
//! `EffectOp::Reveal { subject_idx: u32 }` — caster broadcasts its
//! beliefs about `subject_idx` to all observers. The dispatcher writes
//! a chronicle record (kind=66 EffectRevealApplied):
//!
//!   - slot 0 = 66 (kind tag)
//!   - slot 1 = tick
//!   - slot 2 = caster_slot       (the BROADCASTER)
//!   - slot 3 = target_slot       (= subject_idx — the agent the
//!                                  broadcast is ABOUT)
//!   - slot 4 = subject_idx       (= payload_a; u32 — same as target_slot
//!                                  on the wire; redundant but kept for
//!                                  arm-symmetry)
//!   - slot 5 = 0                 (unused — fan-out target set is "all
//!                                  observers" at consume time)
//!   - slots 6..9 = zero
//!
//! The actual fan-out copy (caster's beliefs about subject → every
//! observer's beliefs about subject) lives in a downstream runtime
//! consumer (`tom_probe_runtime::reveal()` — Phase 3.5).
//!
//! Companion to `plant_belief_chronicle_pin.rs` (kind 63) and
//! `scry_chronicle_pin.rs` (kind 65) — same shape pattern.

use apply_ability_smoke_runtime::{
    ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32,
};
use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};

#[test]
fn reveal_chronicle_record_carries_correct_kind_and_payloads() {
    // Wave 3 ToM Phase 3.5 — one-to-many propagation. The GPU dispatcher
    // writes one chronicle record per cast (kind=66) with subject_idx
    // in slot 4.
    const SUBJECT_IDX: u32 = 7;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 1;

    let mut builder = AbilityRegistryBuilder::new();
    let reveal_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Reveal { subject_idx: SUBJECT_IDX }],
    ));
    let registry = builder.build();

    let per_agent_levels = vec![reveal_id.raw()];
    let per_agent_stats = vec![PerAgentStats::default(); N_AGENTS as usize];

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &per_agent_levels,
        &per_agent_stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!(
                "[reveal_chronicle_pin] skipping: no wgpu adapter available",
            );
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
    assert_eq!(r[0], 66, "Reveal: kind tag — EffectRevealApplied");
    assert_eq!(r[1], TICK, "Reveal: tick");
    assert_eq!(r[2], 0, "Reveal: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], 0,
        "Reveal: target slot — implicit-target rule routes target = caster",
    );
    assert_eq!(
        r[4], SUBJECT_IDX,
        "Reveal: subject_idx at payload word 1 (= payload_a; the agent \
         slot the broadcast is ABOUT)",
    );
    assert_eq!(
        r[5], 0,
        "Reveal: payload_b is unused — fan-out target set is `all \
         observers` at consume time",
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Reveal: tail word {i} must be zero");
    }
}
