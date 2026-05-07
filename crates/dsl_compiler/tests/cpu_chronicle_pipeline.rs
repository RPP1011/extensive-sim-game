//! Composition test for the CPU chronicle pipeline (#133).
//!
//! Combines the engine's `apply_program` (which produces ApplyEvents)
//! with `dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record`
//! (which mirrors what the GPU dispatcher writes per event). The
//! pairing forms the **CPU equivalent** of the GPU dispatcher path:
//!
//!     ability_id + (caster, target, tick) on CPU
//!       → apply_program → Vec<ApplyEvent>
//!       → for each: apply_event_to_chronicle_record(...)
//!       → Vec<[u32; 10]>   (== same chronicle records the GPU writes)
//!
//! This is the foundation for #133 (CPU↔GPU parity): once a runtime
//! crate drives the GPU dispatcher kernel, comparing the GPU's
//! `event_ring` slice against the CPU pipeline's output asserts byte
//! equality on a per-record basis.
//!
//! Today there's no GPU runtime to compare against, so the test
//! verifies the CPU pipeline alone produces the records the GPU
//! dispatcher (per the unit assertions in `wgsl_body.rs` and the
//! integration assertions in `apply_ability_smoke.rs`) is wired to
//! emit.

use dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record;
use engine::ability::apply::apply_program;
use engine::ability::program::{AbilityProgram, CasterStats, EffectOp, Gate};
use engine::ids::AgentId;

fn aid(n: u32) -> AgentId {
    AgentId::new(n).expect("AgentId::new requires non-zero u32")
}

fn run_pipeline(
    program: &AbilityProgram,
    caster: AgentId,
    target: AgentId,
    tick: u32,
) -> Vec<[u32; 10]> {
    let events = apply_program(
        program,
        caster,
        target,
        tick as u64,
        /*world_seed*/ 0xDEAD_BEEF,
        &CasterStats::default(),
    );
    events
        .into_iter()
        .filter_map(|e| apply_event_to_chronicle_record(e, tick, caster.raw()))
        .collect()
}

#[test]
fn single_damage_ability_produces_one_chronicle_record() {
    let program = AbilityProgram::new_single_target(
        /*range*/ 5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    );
    let records = run_pipeline(&program, aid(1), aid(2), 100);
    assert_eq!(records.len(), 1, "one Damage effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 26, "EffectDamageApplied kind tag");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 1, "caster slot — slice γ self-cast (caster_id=1)");
    assert_eq!(r[3], 1, "target slot — slice γ self-cast");
    assert_eq!(r[4], 30.0_f32.to_bits(), "amount as bitcast<u32>");
}

#[test]
fn single_heal_ability_produces_kind_27() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 12.5 }],
    );
    let records = run_pipeline(&program, aid(3), aid(3), 50);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0][0], 27);
    assert_eq!(records[0][4], 12.5_f32.to_bits());
}

#[test]
fn multi_effect_ability_produces_record_per_chronicle_arm() {
    // Two chronicle-bearing variants in one program: Damage + Heal.
    // The CPU pipeline should produce two records, in the order the
    // engine's apply_program emits ApplyEvents (effect-slot order).
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [
            EffectOp::Damage { amount: 30.0 },
            EffectOp::Heal   { amount: 5.0 },
        ],
    );
    let records = run_pipeline(&program, aid(7), aid(11), 200);
    assert_eq!(records.len(), 2, "Damage + Heal → 2 chronicle records");
    assert_eq!(records[0][0], 26, "first record — EffectDamageApplied");
    assert_eq!(records[0][4], 30.0_f32.to_bits());
    assert_eq!(records[1][0], 27, "second record — EffectHealApplied");
    assert_eq!(records[1][4], 5.0_f32.to_bits());
    // Both records share the same tick + caster (slice γ self-cast).
    assert_eq!(records[0][1], 200);
    assert_eq!(records[1][1], 200);
    assert_eq!(records[0][2], 7);
    assert_eq!(records[1][2], 7);
}

#[test]
fn ability_with_only_non_chronicle_effects_produces_no_records() {
    // Root + Silence have no chronicle counterparts today (CPU
    // reference returns None for both). The pipeline produces an
    // empty record vec — the dispatcher's WGSL would still emit
    // TODO-marker arms for these on GPU, but no chronicle write.
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [
            EffectOp::Root    { duration_ticks: 10 },
            EffectOp::Silence { duration_ticks: 10 },
        ],
    );
    let records = run_pipeline(&program, aid(1), aid(2), 100);
    assert!(
        records.is_empty(),
        "no chronicle counterparts → no records (got {} records: {:?})",
        records.len(),
        records,
    );
}

#[test]
fn stun_record_writes_expires_at_tick() {
    // Stun's expires_at_tick = tick + duration_ticks. Pin it.
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Stun { duration_ticks: 17 }],
    );
    let records = run_pipeline(&program, aid(4), aid(4), 100);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0][0], 29, "EffectStunApplied");
    assert_eq!(records[0][4], 117, "tick(100) + duration(17) = expires_at_tick");
}

#[test]
fn slow_record_writes_4_payload_words_with_signed_factor() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Slow { duration_ticks: 10, factor_q8: -64 }],
    );
    let records = run_pipeline(&program, aid(8), aid(5), 100);
    assert_eq!(records.len(), 1);
    let r = records[0];
    assert_eq!(r[0], 30, "EffectSlowApplied");
    assert_eq!(r[4], 110, "expires_at_tick = tick(100) + duration(10)");
    assert_eq!(r[5], (-64_i32) as u32, "factor_q8 sign-widened i16→i32→u32");
}

/// P5 — Determinism via Keyed PCG. The CPU pipeline runs through
/// `per_agent_u32(world_seed, caster, tick, purpose)` for each chance
/// gate, which is a pure function of inputs. Two runs with identical
/// `(caster, target, tick, world_seed)` MUST produce byte-identical
/// chronicle records — including when the chance gate fires (which
/// route to a chronicle write) and when it doesn't (which silently
/// skip).
///
/// Replay equivalence + GPU parity require this property to hold
/// across both backends. The GPU dispatcher will eventually consume
/// the same `chances` SoA and gate via the same RNG seed; this test
/// pins the CPU side ahead of that runtime work.
#[test]
fn cpu_pipeline_is_deterministic_under_chance_gate() {
    // 50% chance gate on Damage — exercises both halves of the
    // chance fork. The chance value `0x8000` is the canonical "50%"
    // (half of u16::MAX) — apply_program compares
    // `(per_agent_u32 & 0xFFFF) < q16` so half the seed space fires.
    let mut program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    );
    program.chances.push(Some(0x8000));

    // Sweep enough (caster, tick) combinations to hit both
    // gate-fires-and-record-emitted AND gate-suppresses-record paths
    // — without the sweep, a deterministic-but-always-firing seed
    // would silently pass the test.
    let mut any_emit = false;
    let mut any_skip = false;
    for caster_seed in [1u32, 2, 3, 5, 7, 11, 13, 17, 19, 23] {
        for tick in [10u32, 50, 100, 200, 500] {
            let run1 = run_pipeline(&program, aid(caster_seed), aid(99), tick);
            let run2 = run_pipeline(&program, aid(caster_seed), aid(99), tick);
            assert_eq!(
                run1, run2,
                "CPU pipeline must be deterministic for caster={caster_seed} \
                 tick={tick} (P5 — keyed PCG)"
            );
            if run1.is_empty() {
                any_skip = true;
            } else {
                any_emit = true;
            }
        }
    }
    // Sanity: with 50 (caster, tick) combos at 50% gate, both branches
    // should fire across the sweep — confirms the determinism check
    // exercised both halves of the fork.
    assert!(
        any_emit,
        "expected at least one chance-fire across 50 sweep combos at 50% gate"
    );
    assert!(
        any_skip,
        "expected at least one chance-skip across 50 sweep combos at 50% gate"
    );
}
