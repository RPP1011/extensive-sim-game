//! Wave 3 ToM Phase 3.5 behavioral pin: the GPU dispatcher's chronicle
//! arm for `EffectOp::Scry` (kind 34 → 65) writes the expected
//! (kind, actor, target, target_observer, subject_idx) tuple for a
//! single self-cast.
//!
//! ## Shape
//!
//! `EffectOp::Scry { target_observer: u8, subject_idx: u32 }` — caster
//! reads `target_observer`'s beliefs about `subject_idx` and writes them
//! into the caster's beliefs about `subject_idx`. The dispatcher writes
//! a chronicle record (kind=65 EffectScryApplied):
//!
//!   - slot 0 = 65 (kind tag)
//!   - slot 1 = tick
//!   - slot 2 = caster_slot       (the OBSERVER reading C's eyes)
//!   - slot 3 = target_slot       (= subject_idx — the agent the belief
//!                                  is ABOUT — the cast's `target`
//!                                  operand IS the subject for scry)
//!   - slot 4 = target_observer   (= payload_a; u8 widened to u32 — the
//!                                  agent slot whose beliefs caster reads)
//!   - slot 5 = subject_idx       (= payload_b; u32 — same as target_slot
//!                                  on the wire; redundant but kept for
//!                                  arm-symmetry with PlantBelief)
//!   - slots 6..9 = zero
//!
//! The actual 6-column copy from `[target_observer * N + subject_idx]`
//! to `[caster * N + subject_idx]` lives in a downstream runtime
//! consumer (`tom_probe_runtime::scry()` — Phase 3.5). This pin only
//! verifies the chronicle record is shaped correctly.
//!
//! Companion to `plant_belief_chronicle_pin.rs` (kind 63) and
//! `reveal_chronicle_pin.rs` (kind 66) — same shape pattern.

use apply_ability_smoke_runtime::{
    ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32,
};
use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};

#[test]
fn scry_chronicle_record_carries_correct_kind_and_payloads() {
    // Wave 3 ToM Phase 3.5 — cross-observer access. The GPU dispatcher
    // writes one chronicle record per cast (kind=65) with target_observer
    // in slot 4 and subject_idx in slot 5.
    const TARGET_OBSERVER: u8 = 3;
    const SUBJECT_IDX: u32 = 7;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 1;

    let mut builder = AbilityRegistryBuilder::new();
    let scry_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Scry {
            target_observer: TARGET_OBSERVER,
            subject_idx: SUBJECT_IDX,
        }],
    ));
    let registry = builder.build();

    let per_agent_levels = vec![scry_id.raw()];
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
                "[scry_chronicle_pin] skipping: no wgpu adapter available",
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
    assert_eq!(r[0], 65, "Scry: kind tag — EffectScryApplied");
    assert_eq!(r[1], TICK, "Scry: tick");
    assert_eq!(r[2], 0, "Scry: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], 0,
        "Scry: target slot — implicit-target rule routes target = caster",
    );
    assert_eq!(
        r[4], TARGET_OBSERVER as u32,
        "Scry: target_observer at payload word 1 (= payload_a; the agent \
         slot whose beliefs caster reads)",
    );
    assert_eq!(
        r[5], SUBJECT_IDX,
        "Scry: subject_idx at payload word 2 (= payload_b; the agent slot \
         the belief is ABOUT)",
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Scry: tail word {i} must be zero");
    }
}
