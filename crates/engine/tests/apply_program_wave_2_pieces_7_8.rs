//! CPU-oracle dispatch coverage for the seven `EffectOp` variants
//! added by Wave 2 pieces 7 (`Stealth`) and 8 (`Charm` / `Grounded` /
//! `Suppress` / `Reflect`), plus the existing #129 `RecastKind` capture
//! that lifts the LoL `recast: <int|dur>` headers into program fields.
//!
//! These verbs all lower cleanly today (LoL canary saturated at
//! 172/172), but apply semantics are deferred — apply handlers wire
//! the per-agent timer SoA later alongside the registry-driven
//! dispatch (#125 family). The CPU oracle's job in the meantime is
//! to faithfully translate each `EffectOp::*` into the matching
//! `ApplyEvent::*` so when a runtime sim wires up a drain target
//! (a la duel_abilities's ApplyDamage), every variant flows through
//! without a silent-no-op gap.
//!
//! These tests pin the dispatch shape so a future rename / refactor
//! can't accidentally introduce a fall-through at the apply level
//! (the kind of footgun #139 found in the deliver-body capture and
//! the initial honesty-audit pass surfaced for 27% of the corpus).

use engine::ability::apply::{apply_program, ApplyEvent};
use engine::ability::program::{
    AbilityProgram, CasterStats, EffectOp, Gate, RecastKind,
};
use engine::ids::AgentId;

fn caster() -> AgentId { AgentId::new(1).unwrap() }
fn target() -> AgentId { AgentId::new(2).unwrap() }

fn single(effect: EffectOp) -> AbilityProgram {
    AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [effect],
    )
}

// ---- Wave 2 piece 7: stealth (self-cast, single duration) -----------

#[test]
fn stealth_dispatches_to_apply_event_with_caster_as_source() {
    let prog = single(EffectOp::Stealth { duration_ticks: 30 });
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 1, "stealth produces exactly one event");
    match events[0] {
        ApplyEvent::Stealth { source, duration_ticks } => {
            assert_eq!(source, caster(), "stealth is self-cast — source must be caster");
            assert_eq!(duration_ticks, 30);
        }
        ref other => panic!("expected ApplyEvent::Stealth; got {other:?}"),
    }
}

// ---- Wave 2 piece 8: charm / grounded / suppress (target + duration) --

#[test]
fn charm_dispatches_to_apply_event_with_target() {
    let prog = single(EffectOp::Charm { duration_ticks: 12 });
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 1);
    match events[0] {
        ApplyEvent::Charm { target: t, duration_ticks } => {
            assert_eq!(t, target());
            assert_eq!(duration_ticks, 12);
        }
        ref other => panic!("expected ApplyEvent::Charm; got {other:?}"),
    }
}

#[test]
fn grounded_dispatches_to_apply_event_with_target() {
    let prog = single(EffectOp::Grounded { duration_ticks: 20 });
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 1);
    match events[0] {
        ApplyEvent::Grounded { target: t, duration_ticks } => {
            assert_eq!(t, target());
            assert_eq!(duration_ticks, 20);
        }
        ref other => panic!("expected ApplyEvent::Grounded; got {other:?}"),
    }
}

#[test]
fn suppress_dispatches_to_apply_event_with_target() {
    let prog = single(EffectOp::Suppress { duration_ticks: 15 });
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 1);
    match events[0] {
        ApplyEvent::Suppress { target: t, duration_ticks } => {
            assert_eq!(t, target());
            assert_eq!(duration_ticks, 15);
        }
        ref other => panic!("expected ApplyEvent::Suppress; got {other:?}"),
    }
}

// ---- Wave 2 piece 8: reflect (target + duration + q8 fraction) --------

#[test]
fn reflect_dispatches_with_q8_fraction_intact() {
    // 0.3 fraction packs as round(0.3 * 256) = 77 (matches LifeSteal /
    // DamageModify q8 convention).
    let frac_q8: i16 = (0.3_f32 * 256.0).round() as i16;
    let prog = single(EffectOp::Reflect { duration_ticks: 30, fraction_q8: frac_q8 });
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 1);
    match events[0] {
        ApplyEvent::Reflect { target: t, duration_ticks, fraction_q8 } => {
            assert_eq!(t, target());
            assert_eq!(duration_ticks, 30);
            assert_eq!(fraction_q8, frac_q8);
        }
        ref other => panic!("expected ApplyEvent::Reflect; got {other:?}"),
    }
}

// ---- Multi-effect ordering: a multi-CC ult dispatches every effect ---

#[test]
fn multi_cc_program_emits_every_event_in_source_order() {
    // Synthesizes the kind of multi-CC ult the LoL corpus has — Sett
    // (suppress + damage), Warwick (suppress + damage_over_time),
    // Singed (grounded + slow). Verifies no variant silently drops
    // when stacked together.
    let prog = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [
            EffectOp::Damage   { amount: 50.0 },
            EffectOp::Suppress { duration_ticks: 15 },
            EffectOp::Grounded { duration_ticks: 20 },
            EffectOp::Charm    { duration_ticks: 8 },
        ],
    );
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 4, "all 4 effects must dispatch");
    assert!(matches!(events[0], ApplyEvent::Damage   { .. }));
    assert!(matches!(events[1], ApplyEvent::Suppress { .. }));
    assert!(matches!(events[2], ApplyEvent::Grounded { .. }));
    assert!(matches!(events[3], ApplyEvent::Charm    { .. }));
}

// ---- #129 sanity: recast headers are captured as program fields, ----
//      not consumed by apply_program (apply path is verb-driven, not
//      header-driven; recast wires later via cast-state tracker).

#[test]
fn recast_header_lives_on_program_fields_not_apply_events() {
    let mut prog = single(EffectOp::Damage { amount: 10.0 });
    prog.recast = Some(RecastKind::Count(3));
    prog.recast_window_ticks = Some(150);

    // The damage event still fires; recast doesn't add or remove
    // events at apply time — it'll be the cast-state tracker's job
    // to count recasts and issue the next program invocation.
    let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
    assert_eq!(events.len(), 1, "recast headers don't affect apply-time event count");
    assert!(matches!(events[0], ApplyEvent::Damage { .. }));

    // The captured fields stayed populated through the apply call.
    assert_eq!(prog.recast, Some(RecastKind::Count(3)));
    assert_eq!(prog.recast_window_ticks, Some(150));
}
