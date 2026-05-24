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
//!       → Vec<[u32; 11]>   (== same chronicle records the GPU writes, stride 11 with seq trailer)
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

use dsl_compiler::cpu_chronicle_reference::{apply_event_to_chronicle_record, CHRONICLE_RECORD_STRIDE_U32};
use engine::ability::apply::apply_program;
use engine::ability::program::{AbilityProgram, CasterStats, EffectOp, Gate};
use engine::ids::AgentId;

fn aid(n: u32) -> AgentId {
    AgentId::new(n).expect("AgentId::new requires non-zero u32")
}

/// Default ability_id for tests that don't care about the slot-6 tag.
/// Gap detective#6 (2026-05-12) added `ability_id: u32` at slot 6 of
/// every chronicle record — every dispatched effect now carries the id
/// of the ability that produced it. Tests that don't probe slot 6 use
/// this constant for consistency.
const TEST_ABILITY_ID: u32 = 1;

fn run_pipeline(
    program: &AbilityProgram,
    caster: AgentId,
    target: AgentId,
    tick: u32,
) -> Vec<[u32; CHRONICLE_RECORD_STRIDE_U32]> {
    let events = apply_program(
        program,
        caster,
        target,
        tick as u64,
        /*world_seed*/ 0xDEAD_BEEF,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    events
        .into_iter()
        // Slice ε part 1: target_id supplied separately. The pipeline
        // tests preserve the slice-γ self-cast convention by passing
        // caster=target — chronicle records keep their existing
        // per-record byte layout. New tests can vary target_id to
        // exercise the explicit-target path.
        .filter_map(|e| apply_event_to_chronicle_record(e, tick, caster.raw(), caster.raw(), TEST_ABILITY_ID, 0))
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
fn every_apply_event_produced_via_apply_program_has_chronicle_counterpart() {
    // After the slice γ closer (Summon → kind 62), every `EffectOp`
    // EngineEffectOp variant `apply_program` can emit has a chronicle
    // counterpart in the CPU reference (and the GPU dispatcher).
    // CastAbility=7 is the only EffectOp without one, by design —
    // recursive dispatch never lands as a top-level ApplyEvent.
    //
    // Wire-up index per variant family:
    // - Root/Silence/Fear/Taunt (Wave 2 piece 1) → kinds 43..46.
    // - Dash/Blink/Knockback/Pull (Wave 2 piece 2) → kinds 47..50.
    // - DoT/HoT/TimedShield (Wave 1.5+) → kinds 51..53.
    // - Stealth/Charm/Grounded/Suppress (extended-status slice) →
    //   kinds 54..57.
    // - Buff/Harvest/PlaceVoxel/Reflect (slice γ tail) → kinds 58..61.
    // - Summon (slice γ closer) → kind 62, see the
    //   `summon_pipeline_emits_kind_62_record` test below.
    //
    // Smoke-pin the Summon path here — a regression that drops the
    // dispatcher arm or the CPU reference translation will surface as
    // an empty `records` Vec.
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [
            EffectOp::Summon { template_hash: 0xDEADBEEF, count: 2, lifetime_ticks: 100 },
        ],
    );
    let records = run_pipeline(&program, aid(1), aid(2), 100);
    assert_eq!(
        records.len(),
        1,
        "one Summon effect → one chronicle record (slice γ closer);  \
         got {} records: {:?}",
        records.len(),
        records,
    );
}

/// Wave 2 piece 1 — control statuses (Root/Silence/Fear/Taunt). Each
/// flows through `apply_program` (which emits ApplyEvent::Root/Silence/
/// Fear/Taunt with `{ target, duration_ticks }`) and the CPU reference
/// (which writes a kind=43..46 record with the same 3-payload-word
/// shape as Stun: actor, target, expires_at_tick = tick + duration).
/// Mirrors the GPU dispatcher's per-status arm shapes.
#[test]
fn root_pipeline_emits_kind_43_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Root { duration_ticks: 25 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Root effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 43, "EventKindId::EffectRootApplied = 43");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 125, "expires_at_tick = tick(100) + duration(25)");
}

#[test]
fn silence_pipeline_emits_kind_44_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Silence { duration_ticks: 25 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Silence effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 44, "EventKindId::EffectSilenceApplied = 44");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 125, "expires_at_tick = tick(100) + duration(25)");
}

#[test]
fn fear_pipeline_emits_kind_45_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Fear { duration_ticks: 25 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Fear effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 45, "EventKindId::EffectFearApplied = 45");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 125, "expires_at_tick = tick(100) + duration(25)");
}

#[test]
fn taunt_pipeline_emits_kind_46_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Taunt { duration_ticks: 25 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Taunt effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 46, "EventKindId::EffectTauntApplied = 46");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 125, "expires_at_tick = tick(100) + duration(25)");
}

/// Wave 2 piece 2 — movement EffectOps (Dash/Blink/Knockback/Pull). Two
/// distinct shapes flow through `apply_program` (which emits
/// ApplyEvent::Dash/Blink with `{ source, distance }` and
/// ApplyEvent::Knockback/Pull with `{ source, target, distance }`) and
/// the CPU reference (which writes a kind=47..50 record). Dash/Blink
/// are caster-self motion: distance lands at payload word 1 (= ring
/// slot offset 3), no target field. Knockback/Pull are forced motion
/// on a target: distance at payload word 2 (= ring slot offset 4),
/// target at payload word 1. Mirrors the GPU dispatcher's per-op arm
/// shapes.
#[test]
fn dash_pipeline_emits_kind_47_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: false, line_of_sight: false },
        [EffectOp::Dash { distance: 12.0 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Dash effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 47, "EventKindId::EffectDashApplied = 47");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 12.0_f32.to_bits(), "distance at payload word 1 (no target field)");
    // No payload word 2 — engine event has no target field.
    for i in 4..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Dash: tail word {i} should be zero");
    }
}

#[test]
fn blink_pipeline_emits_kind_48_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: false, line_of_sight: false },
        [EffectOp::Blink { distance: 8.5 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Blink effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 48, "EventKindId::EffectBlinkApplied = 48");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 8.5_f32.to_bits(), "distance at payload word 1 (no target field)");
    for i in 4..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Blink: tail word {i} should be zero");
    }
}

#[test]
fn knockback_pipeline_emits_kind_49_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Knockback { distance: 5.0 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Knockback effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 49, "EventKindId::EffectKnockbackApplied = 49");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 5.0_f32.to_bits(), "distance at payload word 2 (forced-motion shape)");
    for i in 5..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Knockback: tail word {i} should be zero");
    }
}

#[test]
fn pull_pipeline_emits_kind_50_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Pull { distance: 3.5 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Pull effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 50, "EventKindId::EffectPullApplied = 50");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 3.5_f32.to_bits(), "distance at payload word 2 (forced-motion shape)");
    for i in 5..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Pull: tail word {i} should be zero");
    }
}

/// Wave 1.5+ — multi-tick effect EffectOps (DamageOverTime/HealOverTime/
/// TimedShield). All three flow through `apply_program` (which emits
/// ApplyEvent::DamageOverTime / HealOverTime / TimedShield with
/// `{ source, target, amount, duration_ticks }` and folds scale_bonus
/// into amount) and the CPU reference (which writes a kind=51..53
/// record). All three share the same 5-payload-word shape: actor +
/// target + amount (bitcast f32 → u32 at slot 4) + duration_ticks
/// (raw u32 at slot 5). Pin per-variant kind tags + the duration
/// round-trip — distinct from the q8 / expires_at_tick shapes so a
/// regression that swaps the layout surfaces here.
#[test]
fn damage_over_time_pipeline_emits_kind_51_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::DamageOverTime { amount: 6.5, duration_ticks: 30 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one DamageOverTime effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 51, "EventKindId::EffectDamageOverTimeApplied = 51");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 6.5_f32.to_bits(), "amount-per-tick at payload word 2 (bitcast f32)");
    assert_eq!(r[5], 30, "duration_ticks at payload word 3 (raw u32)");
    for i in 6..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "DamageOverTime: tail word {i} should be zero");
    }
}

#[test]
fn heal_over_time_pipeline_emits_kind_52_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
        [EffectOp::HealOverTime { amount: 4.0, duration_ticks: 50 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one HealOverTime effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 52, "EventKindId::EffectHealOverTimeApplied = 52");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 4.0_f32.to_bits(), "amount-per-tick at payload word 2 (bitcast f32)");
    assert_eq!(r[5], 50, "duration_ticks at payload word 3 (raw u32)");
    for i in 6..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "HealOverTime: tail word {i} should be zero");
    }
}

#[test]
fn timed_shield_pipeline_emits_kind_53_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
        [EffectOp::TimedShield { amount: 25.0, duration_ticks: 100 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one TimedShield effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 53, "EventKindId::EffectTimedShieldApplied = 53");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 25.0_f32.to_bits(), "amount at payload word 2 (bitcast f32)");
    assert_eq!(r[5], 100, "duration_ticks at payload word 3 (raw u32)");
    for i in 6..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "TimedShield: tail word {i} should be zero");
    }
}

/// Extended-corpus statuses (Stealth/Charm/Grounded/Suppress). Two
/// distinct shapes flow through `apply_program` (which emits
/// `ApplyEvent::Stealth { source, duration_ticks }` for the caster-self
/// shape, and `ApplyEvent::Charm/Grounded/Suppress { target,
/// duration_ticks }` for the target-cast shape) and the CPU reference
/// (which writes a kind=54..57 record). Stealth: caster-self status,
/// duration at payload word 1 (= ring slot offset 3), no target field.
/// Charm/Grounded/Suppress: target-cast statuses, 3-payload-word record
/// with duration at payload word 2 (= ring slot offset 4). Mirrors the
/// GPU dispatcher's per-status arm shapes.
#[test]
fn stealth_pipeline_emits_kind_54_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Stealth { duration_ticks: 50 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Stealth effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 54, "EventKindId::EffectStealthApplied = 54");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 50, "duration_ticks at payload word 1 (no target field)");
    // No payload word 2 — engine event has no target field.
    for i in 4..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Stealth: tail word {i} should be zero");
    }
}

#[test]
fn charm_pipeline_emits_kind_55_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Charm { duration_ticks: 30 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Charm effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 55, "EventKindId::EffectCharmApplied = 55");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 30, "duration_ticks at payload word 2 (target-cast shape)");
    for i in 5..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Charm: tail word {i} should be zero");
    }
}

#[test]
fn grounded_pipeline_emits_kind_56_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Grounded { duration_ticks: 25 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Grounded effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 56, "EventKindId::EffectGroundedApplied = 56");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 25, "duration_ticks at payload word 2 (target-cast shape)");
    for i in 5..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Grounded: tail word {i} should be zero");
    }
}

#[test]
fn suppress_pipeline_emits_kind_57_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Suppress { duration_ticks: 40 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Suppress effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 57, "EventKindId::EffectSuppressApplied = 57");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 40, "duration_ticks at payload word 2 (target-cast shape)");
    for i in 5..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Suppress: tail word {i} should be zero");
    }
}

/// Slice γ tail (Buff/Harvest/PlaceVoxel/Reflect). Four distinct shapes
/// flow through `apply_program` (which emits `ApplyEvent::Buff/Harvest/
/// PlaceVoxel/Reflect`) and the CPU reference (which writes a
/// kind=58..61 record). Buff: target-cast with packed payload, raw
/// payload_a (= stat | mag_q8 << 8) at slot 4, raw duration_ticks at
/// slot 5. Harvest: caster-self, kind_hash at slot 3, amount at slot 4.
/// PlaceVoxel: caster-self, kind_hash at slot 3 only. Reflect:
/// target-cast, duration_ticks at slot 4, payload_b's low 16 bits
/// (= fraction_q8 i16) at slot 5. Both Buff and Reflect exercise the
/// signed-packed-payload sign-cast path with negative magnitudes.
#[test]
fn buff_pipeline_emits_kind_58_record_with_signed_magnitude() {
    use engine::ability::program::BuffStat;
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Buff {
            stat: BuffStat::AttackSpeed,  // ordinal 1
            magnitude_q8: -64,            // negative — exercise sign-cast
            duration_ticks: 50,
        }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Buff effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 58, "EventKindId::EffectBuffApplied = 58");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    // Reconstruct expected packed payload_a:
    //   low byte = stat ordinal (1 = AttackSpeed)
    //   bits 8..  = magnitude_q8 sign-cast i16 → i32 → u32 << 8
    let expected_pa = 1u32 | ((-64_i32 as u32) << 8);
    assert_eq!(
        r[4], expected_pa,
        "Buff payload_a packs stat (low byte) + magnitude_q8 (bits 8..; signed); \
         negative magnitude must sign-extend i16 → i32 before shift"
    );
    assert_eq!(r[5], 50, "duration_ticks at payload word 3 (raw u32)");
    for i in 6..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Buff: tail word {i} should be zero");
    }
}

#[test]
fn harvest_pipeline_emits_kind_59_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Harvest { kind_hash: 0xCAFEBABE, amount: 5 }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Harvest effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 59, "EventKindId::EffectHarvestApplied = 59");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 0xCAFEBABE, "kind_hash at payload word 1 (no target field)");
    assert_eq!(r[4], 5, "amount at payload word 2 (u16 widened to u32)");
    for i in 5..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Harvest: tail word {i} should be zero");
    }
}

#[test]
fn place_voxel_pipeline_emits_kind_60_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::PlaceVoxel { kind_hash: 0xFACEFEED }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one PlaceVoxel effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 60, "EventKindId::EffectPlaceVoxelApplied = 60");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 0xFACEFEED, "kind_hash at payload word 1 (no target / position fields)");
    // Position is implicit from the cast's target world position;
    // chronicle record stops here.
    for i in 4..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "PlaceVoxel: tail word {i} should be zero");
    }
}

#[test]
fn reflect_pipeline_emits_kind_61_record_with_signed_fraction() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Reflect {
            duration_ticks: 50,
            fraction_q8: -64,  // negative — exercise sign-cast through u16
        }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Reflect effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 61, "EventKindId::EffectReflectApplied = 61");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 50, "duration_ticks at payload word 2 (raw u32)");
    // payload_b matches `pack_effect`'s Reflect arm: `(fraction_q8 as u16) as u32`.
    // For -64_i16: u16 wraps to 0xFFC0; u32 zero-extend → 0x0000_FFC0.
    let expected_pb = (-64_i16 as u16) as u32;
    assert_eq!(expected_pb, 0x0000_FFC0, "sanity: -64 → 0xFFC0 in low 16 bits");
    assert_eq!(
        r[5], expected_pb,
        "Reflect payload_b's low 16 bits carry fraction_q8 (i16 → u16 → u32, \
         zero-extend). Consumer sign-extends to recover the negative value."
    );
    for i in 6..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Reflect: tail word {i} should be zero");
    }
}

/// Slice γ closer — Summon (kind 24 → 62). Caster-self with packed
/// payload. 5-payload-word record: actor + template_hash + count
/// (u8 widened to u32) + lifetime_ticks. The dispatcher splits the
/// packed `payload_b` into distinct ring slots (slot 4 = count, slot
/// 5 = lifetime_ticks) so consumers don't have to redo the bit-unpack
/// on read; the engine event struct carries `count: u8` and
/// `lifetime_ticks: u32` as separate fields.
#[test]
fn summon_pipeline_emits_kind_62_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Summon {
            template_hash: 0xDEADBEEF,
            count: 3,
            lifetime_ticks: 120,
        }],
    );
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one Summon effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 62, "EventKindId::EffectSummonApplied = 62");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 0xDEADBEEF, "template_hash at payload word 1 (no target field)");
    assert_eq!(r[4], 3, "count (u8 widened to u32) at payload word 2");
    assert_eq!(r[5], 120, "lifetime_ticks (raw u32) at payload word 3");
    for i in 6..10 {
        if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
        assert_eq!(r[i], 0, "Summon: tail word {i} should be zero");
    }
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

#[test]
fn transfer_gold_pipeline_emits_kind_31_record() {
    // Full pipeline through apply_program (which now emits
    // ApplyEvent::TransferGold) → CPU reference → kind=31 chronicle
    // record. Prior to commit f881bed1 this would yield zero records
    // (apply_program skipped TransferGold and the CPU reference
    // returned None) — both sides had to land before the integration
    // worked.
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::TransferGold { amount: 100 }],
    );
    let records = run_pipeline(&program, aid(7), aid(11), 200);
    assert_eq!(records.len(), 1, "TransferGold pipeline produces one record");
    assert_eq!(records[0][0], 31, "EventKindId::EffectGoldTransfer = 31");
    assert_eq!(records[0][1], 200, "tick");
    assert_eq!(records[0][4], 100, "amount round-trips through pipeline");
}

#[test]
fn modify_standing_pipeline_emits_kind_32_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::ModifyStanding { delta: -50 }],
    );
    let records = run_pipeline(&program, aid(3), aid(4), 100);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0][0], 32, "EventKindId::EffectStandingDelta = 32");
    assert_eq!(records[0][4], (-50_i32) as u32, "delta sign-widens through pipeline");
}

/// Bleed verb swap (Task #138 follow-on, 2026-05-06): SelfDamage
/// flows through `apply_program` (which emits
/// `ApplyEvent::SelfDamage{source, amount}`) and the CPU reference
/// (which writes a kind=39 record with caster_id in BOTH actor +
/// target slots — self-damage targets caster). Mirrors the GPU
/// dispatcher's SelfDamage arm shape.
#[test]
fn self_damage_pipeline_emits_kind_39_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 50, hostile_only: false, line_of_sight: false },
        [EffectOp::SelfDamage { amount: 5.0 }],
    );
    // Self-cast: caster == target. The pipeline helper passes
    // caster_id = target_id = caster.raw() into the CPU reference,
    // mirroring the duel_abilities Bleed verb's
    // `apply_ability 4 by self target self` shape. The SelfDamage
    // arm explicitly writes caster_id into both slots regardless.
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one SelfDamage effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 39, "EventKindId::EffectSelfDamageApplied = 39");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id (the bleeder)");
    assert_eq!(r[3], 7, "target slot — caster_id (self-damage targets caster)");
    assert_eq!(r[4], 5.0_f32.to_bits(), "amount as bitcast<u32>");
}

/// Vampirize verb swap (Task #138 follow-on, mirror of Bleed at
/// `486eb08f`): LifeSteal flows through `apply_program` (which emits
/// `ApplyEvent::LifeSteal{target=caster, duration_ticks, fraction_q8}`)
/// and the CPU reference (which writes a kind=40 record with the same
/// 4-payload-word shape as Slow — actor, target, expires_at_tick =
/// tick + duration, fraction_q8 sign-widened). Mirrors the GPU
/// dispatcher's LifeSteal arm shape.
#[test]
fn life_steal_pipeline_emits_kind_40_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 80, hostile_only: false, line_of_sight: false },
        [EffectOp::LifeSteal { duration_ticks: 50, fraction_q8: 128 }],
    );
    // Self-cast: caster == target. The pipeline helper passes
    // caster_id = target_id = caster.raw() into the CPU reference,
    // mirroring the duel_abilities Vampirize verb's
    // `apply_ability 6 by self target self` shape.
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one LifeSteal effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 40, "EventKindId::EffectLifeStealApplied = 40");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 150, "expires_at_tick = tick(100) + duration(50)");
    assert_eq!(r[5], 128, "fraction_q8 = 128 (= 0.5×) sign-widened");
}

/// Reap verb swap (Task #138 follow-on, mirror of Fortify at
/// `001ae9a6`): Execute flows through `apply_program` (which emits
/// `ApplyEvent::Execute{target, hp_threshold}`) and the CPU reference
/// (which writes a kind=42 record with a 3-payload-word shape — actor,
/// target, hp_threshold (bitcast f32)). Mirrors the GPU dispatcher's
/// Execute arm shape. Closes the slice across all 8 duel_abilities
/// verbs.
///
/// SEMANTICS NOTE: the when-condition `target.hp < hp_threshold` is
/// NOT evaluated by apply_program (the `when_per_effect[i]` field
/// stays unconsulted today), so the record fires unconditionally — the
/// duel_abilities Reap verb's outer `when` clause provides the gate
/// upstream.
#[test]
fn execute_pipeline_emits_kind_42_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Execute { hp_threshold: 20.0 }],
    );
    // Distinct caster + target — Reap targets an enemy, not self.
    // The pipeline helper uses caster == target by default; this test
    // inlines its own pipeline so it can route caster/target separately,
    // mirroring how the duel_abilities Reap verb's
    // `apply_ability 5 by self target target` lowers to distinct
    // operands.
    let caster = aid(7);
    let target = aid(11);
    let tick: u32 = 100;
    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        /*world_seed*/ 0xDEAD_BEEF,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();
    assert_eq!(records.len(), 1, "one Execute effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 42, "EventKindId::EffectExecuteApplied = 42");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 11, "target slot — target_id (distinct from caster)");
    assert_eq!(r[4], 20.0_f32.to_bits(), "hp_threshold = 20.0 as bitcast<u32>");
}

/// Fortify verb swap (Task #138 follow-on, mirror of Vampirize at
/// `60115f64`): DamageModify flows through `apply_program` (which
/// emits `ApplyEvent::DamageModify{target=caster, duration_ticks,
/// multiplier_q8}`) and the CPU reference (which writes a kind=41
/// record with the same 4-payload-word shape as Slow / LifeSteal —
/// actor, target, expires_at_tick = tick + duration, multiplier_q8
/// sign-widened). Mirrors the GPU dispatcher's DamageModify arm shape.
#[test]
fn damage_modify_pipeline_emits_kind_41_record() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 80, hostile_only: false, line_of_sight: false },
        [EffectOp::DamageModify { duration_ticks: 50, multiplier_q8: 128 }],
    );
    // Self-cast: caster == target. The pipeline helper passes
    // caster_id = target_id = caster.raw() into the CPU reference,
    // mirroring the duel_abilities Fortify verb's
    // `apply_ability 7 by self target self` shape.
    let records = run_pipeline(&program, aid(7), aid(7), 100);
    assert_eq!(records.len(), 1, "one DamageModify effect → one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 41, "EventKindId::EffectDamageModifyApplied = 41");
    assert_eq!(r[1], 100, "tick");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 7, "target slot — target_id (self-cast: ==caster)");
    assert_eq!(r[4], 150, "expires_at_tick = tick(100) + duration(50)");
    assert_eq!(r[5], 128, "multiplier_q8 = 128 (= 0.5×) sign-widened");
}

/// Slice ε part 1 pipeline integration: when the caller passes
/// distinct caster + target ids, the chronicle records have
/// distinct values in slot 2 (actor) and slot 3 (target). Mirrors
/// what the GPU dispatcher writes when the source supplies
/// `apply_ability <a> by <c> target <t>`.
///
/// Distinct from the self-cast tests above (which preserve slice-γ
/// convention by passing caster=target). This test inlines its own
/// pipeline so it can route caster/target separately to the CPU
/// reference, mirroring how a runtime crate would dispatch from
/// the GPU side with explicit operands.
#[test]
fn distinct_caster_and_target_pipeline_writes_both_slots() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    );
    let caster = aid(7);
    let target = aid(11);
    let tick: u32 = 100;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        /*world_seed*/ 0xDEAD_BEEF,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    // Route caster + target separately into the CPU reference (the
    // slice-ε path the GPU dispatcher takes when source supplies
    // explicit `target <expr>`).
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "Damage produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 26, "EffectDamageApplied");
    assert_eq!(r[2], 7, "actor slot — caster_id");
    assert_eq!(r[3], 11, "target slot — target_id (distinct from caster)");
    assert_eq!(r[4], 30.0_f32.to_bits(), "amount");
}

/// Slice ε pipeline pin for Stun — distinct caster/target case +
/// the multi-payload-field shape (Stun's payload[0] is
/// expires_at_tick, distinct from Damage's payload[0] = amount).
/// Confirms slot 2 + slot 3 routing is independent of how
/// payload words 4+ are computed.
#[test]
fn distinct_caster_and_target_pipeline_handles_stun_payload_shape() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Stun { duration_ticks: 25 }],
    );
    let caster = aid(3);
    let target = aid(17);
    let tick: u32 = 200;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xCAFE_F00D,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "Stun produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 29, "EffectStunApplied");
    assert_eq!(r[2], 3, "actor slot — caster_id");
    assert_eq!(r[3], 17, "target slot — distinct from caster");
    assert_eq!(r[4], 225, "expires_at_tick = tick(200) + duration(25)");
}

/// Slice ε pipeline pin for Heal — the friendly-target chronicle
/// variant (hostile_only=false). Confirms slot 2/3 routing works
/// when caster ≠ target on a friendly cast (caster heals an ally),
/// which is the canonical use case for the explicit-target operand.
#[test]
fn distinct_caster_and_target_pipeline_handles_friendly_heal() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 12.5 }],
    );
    let caster = aid(4);
    let target = aid(8);
    let tick: u32 = 300;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xFEED_FACE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "Heal produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 27, "EffectHealApplied");
    assert_eq!(r[1], 300, "tick");
    assert_eq!(r[2], 4, "actor slot — caster_id (the healer)");
    assert_eq!(r[3], 8, "target slot — distinct target_id (the ally)");
    assert_eq!(r[4], 12.5_f32.to_bits(), "amount as bitcast<u32>");
}

/// Slice ε pipeline pin for multi-effect programs with distinct
/// caster/target. The existing `multi_effect_ability_produces_record_
/// per_chronicle_arm` test exercises self-cast routing; this
/// variant confirms that when one program emits multiple chronicle
/// records, EVERY record carries the same explicit caster + target
/// slot pair (not just the first).
///
/// A regression where slice-ε routing was applied per-record but lost
/// state between iterations would manifest as some records having
/// caster=target while others had distinct values.
#[test]
fn distinct_caster_and_target_pipeline_preserves_routing_across_multi_effect() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [
            EffectOp::Damage { amount: 30.0 },
            EffectOp::Heal   { amount: 5.0 },
            EffectOp::Stun   { duration_ticks: 12 },
        ],
    );
    let caster = aid(11);
    let target = aid(23);
    let tick: u32 = 800;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xBADD_CAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 3, "Damage + Heal + Stun → 3 chronicle records");

    // Every record carries the same distinct caster + target pair.
    for (idx, r) in records.iter().enumerate() {
        assert_eq!(
            r[2], 11,
            "record {idx} actor slot must be caster_id=11 \
             (slice-ε routing must be stable across multi-effect \
             record emission); got {}", r[2]
        );
        assert_eq!(
            r[3], 23,
            "record {idx} target slot must be target_id=23 \
             (distinct from caster); got {}", r[3]
        );
        assert_eq!(r[1], 800, "record {idx} tick");
    }

    // Kind tags appear in the order apply_program emits them
    // (effect-slot order: Damage=26, Heal=27, Stun=29).
    assert_eq!(records[0][0], 26, "record 0 — EffectDamageApplied");
    assert_eq!(records[1][0], 27, "record 1 — EffectHealApplied");
    assert_eq!(records[2][0], 29, "record 2 — EffectStunApplied");
}

/// Slice ε pipeline pin for ModifyStanding — closes per-variant
/// pipeline coverage. Standing deltas naturally bind caster (the
/// observer/initiator whose opinion is being recorded) to target
/// (the agent whose standing changes); the i16→i32→u32 sign-widening
/// payload mirrors TransferGold's shape but emits kind=32.
#[test]
fn distinct_caster_and_target_pipeline_handles_modify_standing() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::ModifyStanding { delta: -100 }],
    );
    let caster = aid(21);
    let target = aid(34);
    let tick: u32 = 750;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xF00D_BABE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "ModifyStanding produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 32, "EffectStandingDelta");
    assert_eq!(r[1], 750, "tick");
    assert_eq!(r[2], 21, "actor slot — caster_id (the observer recording opinion)");
    assert_eq!(r[3], 34, "target slot — distinct target_id (the judged agent)");
    assert_eq!(r[4], (-100_i32) as u32, "negative delta sign-widened i16→i32→u32");
}

/// Slice ε pipeline pin for TransferGold — the chronicle variant
/// where caster≠target is the *canonical* shape (gold flows from
/// source to recipient; self-transfer is a degenerate case). Confirms
/// the i32→u32 sign-widening payload coexists with explicit slot 2/3
/// routing.
#[test]
fn distinct_caster_and_target_pipeline_handles_transfer_gold() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::TransferGold { amount: 250 }],
    );
    let caster = aid(15);
    let target = aid(22);
    let tick: u32 = 500;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xABBA_C0DE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "TransferGold produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 31, "EffectGoldTransfer");
    assert_eq!(r[1], 500, "tick");
    assert_eq!(r[2], 15, "actor slot — caster_id (the sender)");
    assert_eq!(r[3], 22, "target slot — distinct target_id (the recipient)");
    assert_eq!(r[4], 250, "amount round-trips through pipeline");
}

/// Slice ε pipeline pin for negative TransferGold — exercises the
/// i32→u32 sign-widening on the caster≠target path. (Negative amounts
/// model gold being taxed/seized — source field is still the agent
/// initiating the transfer, regardless of direction.)
#[test]
fn distinct_caster_and_target_pipeline_transfer_gold_negative_amount() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::TransferGold { amount: -75 }],
    );
    let caster = aid(9);
    let target = aid(14);
    let tick: u32 = 600;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xDEAD_C0DE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1);
    let r = records[0];
    assert_eq!(r[0], 31, "EffectGoldTransfer");
    assert_eq!(r[2], 9, "caster_id");
    assert_eq!(r[3], 14, "target_id distinct from caster");
    assert_eq!(r[4], (-75_i32) as u32, "negative amount sign-widened i32→u32");
}

/// Slice ε pipeline pin for Shield — the third amount-only chronicle
/// variant (alongside Damage + Heal) but with its own EventKindId
/// (28). Confirms slot 2/3 routing is event-kind-tag independent
/// across the three structurally-identical payload shapes.
#[test]
fn distinct_caster_and_target_pipeline_handles_shield() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::Shield { amount: 45.0 }],
    );
    let caster = aid(6);
    let target = aid(13);
    let tick: u32 = 400;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xC0FFEE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "Shield produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 28, "EffectShieldApplied");
    assert_eq!(r[1], 400, "tick");
    assert_eq!(r[2], 6, "actor slot — caster_id (the shielder)");
    assert_eq!(r[3], 13, "target slot — distinct target_id (the protected ally)");
    assert_eq!(r[4], 45.0_f32.to_bits(), "amount as bitcast<u32>");
}

/// Slice ε pipeline pin for Slow — the only chronicle-bearing
/// variant with 4 payload words (others have 3). Confirms slot 2/3
/// routing is independent even when the payload extends to slot 5.
#[test]
fn distinct_caster_and_target_pipeline_handles_slow_4_field_payload() {
    let program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Slow { duration_ticks: 50, factor_q8: -32 }],
    );
    let caster = aid(2);
    let target = aid(9);
    let tick: u32 = 1000;

    let events = apply_program(
        &program,
        caster,
        target,
        tick as u64,
        0xBEEF,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    let records: Vec<_> = events
        .into_iter()
        .filter_map(|e| {
            apply_event_to_chronicle_record(e, tick, caster.raw(), target.raw(), TEST_ABILITY_ID, 0)
        })
        .collect();

    assert_eq!(records.len(), 1, "Slow produces one chronicle record");
    let r = records[0];
    assert_eq!(r[0], 30, "EffectSlowApplied");
    assert_eq!(r[2], 2, "actor slot — caster_id");
    assert_eq!(r[3], 9, "target slot — distinct from caster");
    assert_eq!(r[4], 1050, "expires_at_tick = tick(1000) + duration(50)");
    assert_eq!(r[5], (-32_i32) as u32, "factor_q8 sign-widened i16→i32→u32");
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

/// Wave 1.5#9 nested-effect dispatch (2026-05-06): a program with a
/// primary `Damage` effect and a nested `{ stun 1s }` block emits TWO
/// chronicle records — kind=26 (EffectDamageApplied) followed by
/// kind=29 (EffectStunApplied) — in declaration order.
///
/// This is the Reap-shape pin: `apply_program` produces an
/// ApplyEvent::Damage AND ApplyEvent::Stun, and
/// `apply_event_to_chronicle_record` translates each into its
/// matching record. Closes the documented gap surfaced by the Reap
/// verb swap (commit `72a35307`).
#[test]
fn nested_stun_on_damage_emits_two_chronicle_records_in_order() {
    use smallvec::smallvec;
    let mut program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    );
    program.nested_per_effect.push(smallvec![EffectOp::Stun { duration_ticks: 10 }]);
    let records = run_pipeline(&program, aid(1), aid(2), 100);
    assert_eq!(
        records.len(), 2,
        "primary Damage + nested Stun → 2 chronicle records (got {})",
        records.len(),
    );
    // First record: Damage = kind 26.
    assert_eq!(records[0][0], 26, "first record must be EffectDamageApplied (26)");
    assert_eq!(records[0][1], 100, "tick");
    assert_eq!(records[0][4], 25.0_f32.to_bits(), "amount");
    // Second record: nested Stun = kind 29 with expires_at_tick.
    assert_eq!(records[1][0], 29, "second record must be EffectStunApplied (29)");
    assert_eq!(records[1][1], 100, "tick");
    assert_eq!(records[1][4], 110, "expires_at_tick = tick(100) + duration(10)");
}

/// Wave 1.5#9: when the chance gate skips the primary, the nested
/// chronicle record is also skipped — pin both halves together so a
/// future runtime can't accidentally let nested ops outlive a gated
/// primary.
#[test]
fn nested_effect_record_is_skipped_when_chance_gate_fails() {
    use smallvec::smallvec;
    let mut program = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    );
    program.chances.push(Some(0)); // gate-out the primary
    program.nested_per_effect.push(smallvec![EffectOp::Stun { duration_ticks: 10 }]);
    let records = run_pipeline(&program, aid(1), aid(2), 100);
    assert!(
        records.is_empty(),
        "chance=0 must skip both primary AND nested chronicle records (got {})",
        records.len(),
    );
}
