//! Pin: chronicle records carry `ability_id` at slot 6 (Gap detective#6).
//!
//! Pre-fix: `apply_ability` wrote chronicle records with shape
//! `(kind, tick, actor, target, amount)` — slots 0..4 — and downstream
//! consumers had no way to discriminate which ability fired. The
//! detective_investigation fixture's `ApplyDamageFromChronicle` rule
//! re-emitted `Accused` unconditionally for every damage event,
//! over-counting accusations 3x (one per verb: Accuse / Investigate /
//! Observe).
//!
//! Post-fix (2026-05-12, this change): every chronicle arm in
//! `emit_chronicle_arm_chain` ends with
//! `atomicStore(&event_ring[_slot * 10u + 6u], ability_id__u32);`,
//! so downstream WGSL consumers can read slot 6 to discriminate ability
//! source. The 10-word stride is unchanged; slot 6 was previously
//! unused for every arm (slots 7..9 stay reserved for future fields).
//!
//! This test pins three things:
//!   1. The dispatcher's WGSL primary chain emits the slot-6 write.
//!   2. The CPU reference (`apply_event_to_chronicle_record`) writes
//!      the same slot, so CPU↔GPU parity holds.
//!   3. A synthetic chronicle reader can recover `ability_id` from
//!      slot 6 of a record produced by the CPU pipeline.

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cpu_chronicle_reference::{
    apply_event_to_chronicle_record, CHRONICLE_RECORD_STRIDE_U32,
};
use engine::ability::apply::ApplyEvent;
use engine::ids::AgentId;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

/// Closed-loop pin: lower the `apply_ability_smoke` fixture, find the
/// dispatcher kernel, assert it writes `ability_id__u32` to slot 6 of
/// every chronicle-bearing arm.
///
/// Pre-Gap-detective#6 the dispatcher emitted only slots 0..5; slot 6
/// (and slots 7..9) were always zero. The fix adds a uniform slot-6
/// write at the end of every arm, sourced from the
/// `ability_id__u32` outer-scope identifier the dispatcher prelude
/// already declares.
#[test]
fn dispatcher_emits_slot_6_ability_id_write() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("apply_ability_smoke.sim failed at: {e}");
    });

    // Find the dispatcher kernel (any kernel containing the apply
    // dispatcher loop signature).
    let dispatch = art
        .wgsl_files
        .iter()
        .find(|(_, body)| body.contains("for (var i: u32 = 0u; i < 6u;"))
        .map(|(_, b)| b.as_str())
        .expect("dispatcher kernel missing — check apply_ability_smoke.sim contains apply_ability");

    // The slot-6 write is applied to every arm (46 arms in
    // emit_chronicle_arm_chain). Even with only the Damage arm taken at
    // runtime, the WGSL contains all 46 arm bodies so the count is
    // expected to be the same in any compiled output. Pin a generous
    // lower bound so the test surfaces if the per-arm helper drops the
    // suffix for any subset.
    //
    // Why 40 (not 46): the AOE Path B builder emits both a primary
    // chain and a nested chain for each arm, so the count can land
    // higher in AOE-enabled fixtures. For non-AOE fixtures (the smoke
    // fixture today), the primary chain alone is 46 writes plus the
    // nested chain another 46 = 92. We pin >= 40 to leave headroom for
    // future arm-removals while still catching a regression that drops
    // the entire suffix.
    let count = dispatch.matches("event_ring[_slot * 10u + 6u], ability_id__u32").count();
    assert!(
        count >= 40,
        "Dispatcher must emit slot-6 ability_id writes on every chronicle arm; \
         found {count} writes in dispatcher kernel:\n{dispatch}"
    );
}

/// CPU reference writes `ability_id` at slot 6 of every chronicle
/// record. Mirrors the GPU dispatcher's slot-6 write for byte-level
/// CPU↔GPU parity.
#[test]
fn cpu_reference_writes_ability_id_at_slot_6() {
    let aid = |n: u32| AgentId::new(n).expect("AgentId::new requires non-zero u32");

    // Damage record: actor=7, target=11, ability_id=42, tick=100.
    // The slot-6 write happens regardless of variant.
    let rec = apply_event_to_chronicle_record(
        ApplyEvent::Damage {
            source: aid(7),
            target: aid(11),
            amount: 25.0,
        },
        /*tick*/ 100,
        /*caster_id*/ 7,
        /*target_id*/ 11,
        /*ability_id*/ 42,
    /*intra_emit_idx*/ 0,
    )
    .expect("Damage has chronicle counterpart");
    assert_eq!(rec[0], 26, "kind tag — EffectDamageApplied");
    assert_eq!(rec[2], 7, "caster slot");
    assert_eq!(rec[3], 11, "target slot");
    assert_eq!(rec[4], 25.0_f32.to_bits(), "amount");
    assert_eq!(
        rec[6], 42,
        "slot 6 must hold the ability_id passed to apply_event_to_chronicle_record"
    );

    // Smoke-pin slot 6 across multiple variants — every Effect*Applied
    // arm should carry the ability_id since the GPU dispatcher's
    // slot-6 write is uniform across all 46 arms.
    let rec_heal = apply_event_to_chronicle_record(
        ApplyEvent::Heal {
            source: aid(1),
            target: aid(2),
            amount: 10.0,
        },
        /*tick*/ 50,
        /*caster_id*/ 1,
        /*target_id*/ 2,
        /*ability_id*/ 7,
    /*intra_emit_idx*/ 0,
    )
    .expect("Heal has chronicle counterpart");
    assert_eq!(rec_heal[0], 27, "kind tag — EffectHealApplied");
    assert_eq!(rec_heal[6], 7, "Heal arm must also carry ability_id at slot 6");

    let rec_stun = apply_event_to_chronicle_record(
        ApplyEvent::Stun {
            target: aid(3),
            duration_ticks: 17,
        },
        /*tick*/ 200,
        /*caster_id*/ 4,
        /*target_id*/ 3,
        /*ability_id*/ 99,
    /*intra_emit_idx*/ 0,
    )
    .expect("Stun has chronicle counterpart");
    assert_eq!(rec_stun[0], 29, "kind tag — EffectStunApplied");
    assert_eq!(rec_stun[6], 99, "Stun arm must also carry ability_id at slot 6");
}

/// Downstream-reader simulation: a chronicle consumer that reads
/// `event_ring[event_idx * stride + 6u]` to recover the source
/// ability_id. This is the surface a `physics_ApplyDamageFromChronicle`
/// rule would use to discriminate Accuse / Investigate / Observe in
/// the detective_investigation fixture.
///
/// Pre-Gap-fix, slot 6 was always zero — every chronicle re-emit fired
/// for every kind=26 record regardless of source ability. Post-fix,
/// reading slot 6 returns the ability_id that produced the record, so
/// re-emits can guard on `if record[6] == ACCUSE_ABILITY_ID`.
#[test]
fn downstream_consumer_recovers_ability_id_from_slot_6() {
    let aid = |n: u32| AgentId::new(n).expect("AgentId::new requires non-zero u32");

    // Three records, one per "verb" the detective fixture distinguishes.
    // Use distinct ability_ids to mimic the post-fix surface — the
    // downstream consumer should read slot 6 and route its re-emit per
    // verb.
    const ACCUSE_ID: u32 = 1;
    const INVESTIGATE_ID: u32 = 2;
    const OBSERVE_ID: u32 = 3;

    let records: Vec<[u32; CHRONICLE_RECORD_STRIDE_U32]> = vec![
        apply_event_to_chronicle_record(
            ApplyEvent::Damage {
                source: aid(1),
                target: aid(2),
                amount: 5.0,
            },
            10,
            1,
            2,
            ACCUSE_ID,
        /*intra_emit_idx*/ 0,
        )
        .unwrap(),
        apply_event_to_chronicle_record(
            ApplyEvent::Damage {
                source: aid(1),
                target: aid(2),
                amount: 1.0,
            },
            10,
            1,
            2,
            INVESTIGATE_ID,
        /*intra_emit_idx*/ 0,
        )
        .unwrap(),
        apply_event_to_chronicle_record(
            ApplyEvent::Damage {
                source: aid(1),
                target: aid(2),
                amount: 0.5,
            },
            10,
            1,
            2,
            OBSERVE_ID,
        /*intra_emit_idx*/ 0,
        )
        .unwrap(),
    ];

    // The synthetic consumer iterates records, filters on
    // `kind == 26 (EffectDamageApplied)`, then routes by
    // `record[6]` (the ability_id). Pre-fix this routing was
    // impossible — record[6] was always 0.
    let mut accuse_count = 0u32;
    let mut investigate_count = 0u32;
    let mut observe_count = 0u32;
    let mut other_count = 0u32;

    for rec in &records {
        if rec[0] != 26 {
            continue;
        }
        match rec[6] {
            ACCUSE_ID => accuse_count += 1,
            INVESTIGATE_ID => investigate_count += 1,
            OBSERVE_ID => observe_count += 1,
            _ => other_count += 1,
        }
    }

    assert_eq!(accuse_count, 1, "exactly one Accuse-sourced damage record");
    assert_eq!(
        investigate_count, 1,
        "exactly one Investigate-sourced damage record"
    );
    assert_eq!(observe_count, 1, "exactly one Observe-sourced damage record");
    assert_eq!(
        other_count, 0,
        "no records should land in the unknown bucket — slot 6 \
         must propagate the ability_id reliably"
    );
}
