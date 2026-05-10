//! Plan G option D — engine-level pin proving the immediate-cast +
//! deferred-resolution sequence works end-to-end at the apply layer.
//!
//! This is the engine-side analog of what `firebolt_probe_runtime`
//! exercises on real wgpu hardware. The flow:
//!
//!   1. Tick T0:  `apply_program(prog)` fires for the cast initiation.
//!                Should return exactly one `ApplyEvent::CastBegin`
//!                carrying the duration; per-fixture sim consumers
//!                stamp `agents.busy_until_tick(self) = T0 + duration`
//!                from this.
//!   2. Tick T0+duration:  the busy-resolution kernel detects the
//!                expiry, calls `apply_pending_program(prog)`. Should
//!                return the full deferred-effects stream — for our
//!                Firebolt fixture: one Damage(25) event.
//!
//! Replay equivalence (P5): both calls take the same `(world_seed, tick,
//! caster, target)` shape; `pending_program` is bit-stable across
//! replay because it carries no RNG draws today (modifier slots
//! deferred per option D). When the parallel aggregator slots
//! (`pending_chances`, etc.) land, the same per-effect chance gate the
//! immediate path uses MUST be replicated here so deferred resolution
//! stays deterministic.

use engine::ability::apply::{apply_pending_program, apply_program, ApplyEvent};
use engine::ability::program::{AbilityProgram, CasterStats, EffectOp, Gate};
use engine::ids::AgentId;

fn caster() -> AgentId {
    AgentId::new(1).unwrap()
}
fn target() -> AgentId {
    AgentId::new(2).unwrap()
}

#[test]
fn cast_then_resolve_emits_castbegin_then_damage() {
    // Build a Firebolt-shaped program: CastBegin in `effects` (the
    // immediate path) and one Damage(25) in `pending_program` (the
    // deferred path). Mirrors the IR shape `lower_ability_decl`
    // produces for a `cast { duration: 3t } effect { damage 25 }`
    // ability.
    let mut prog = AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 80, hostile_only: true, line_of_sight: false },
        [EffectOp::CastBegin {
            ability_id:     1,
            duration_ticks: 3,
            target_slot:    0,
            target_x_q8:    0,
            target_y_q8:    0,
        }],
    );
    prog.pending_program.push(EffectOp::Damage { amount: 25.0 });

    // -- Immediate cast at T0 --
    let immediate = apply_program(
        &prog, caster(), target(), 0, 0xCAFE,
        &CasterStats::default(), &CasterStats::default(),
    );
    assert_eq!(immediate.len(), 1, "immediate path emits exactly one CastBegin op");
    match immediate[0] {
        ApplyEvent::CastBegin { source, duration_ticks, .. } => {
            assert_eq!(source, caster());
            assert_eq!(duration_ticks, 3);
        }
        ref other => panic!("expected ApplyEvent::CastBegin; got {other:?}"),
    }

    // -- Deferred resolution at T0+3 (the busy-resolution kernel's
    //    trigger tick). At this point `firebolt_probe.sim`'s
    //    ResolveBusy rule would fire `apply_pending_program`. --
    let deferred = apply_pending_program(
        &prog, caster(), target(), 3, 0xCAFE,
        &CasterStats::default(), &CasterStats::default(),
    );
    assert_eq!(deferred.len(), 1, "deferred path emits the one pending Damage op");
    match deferred[0] {
        ApplyEvent::Damage { source, target: t, amount } => {
            assert_eq!(source, caster());
            assert_eq!(t, target());
            assert!((amount - 25.0).abs() < 1e-3);
        }
        ref other => panic!("expected ApplyEvent::Damage; got {other:?}"),
    }
}

/// Replay equivalence (P5) — same inputs at different runs produce
/// the same event stream. The pin guards future regressions when we
/// wire `pending_chances` / scaling — both `apply_pending_program`
/// calls under the same seed/tick/caster/target must remain
/// bit-equal.
#[test]
fn deferred_resolution_is_replayable_under_p5() {
    let mut prog = AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 80, hostile_only: true, line_of_sight: false },
        [EffectOp::CastBegin {
            ability_id: 1, duration_ticks: 3, target_slot: 0,
            target_x_q8: 0, target_y_q8: 0,
        }],
    );
    prog.pending_program.push(EffectOp::Damage { amount: 25.0 });
    prog.pending_program.push(EffectOp::Stun { duration_ticks: 10 });

    let a = apply_pending_program(
        &prog, caster(), target(), 42, 0xDEADBEEF,
        &CasterStats::default(), &CasterStats::default(),
    );
    let b = apply_pending_program(
        &prog, caster(), target(), 42, 0xDEADBEEF,
        &CasterStats::default(), &CasterStats::default(),
    );
    assert_eq!(a.len(), b.len(), "same inputs → same event count");
    for (i, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
        // Compare via Debug; bit-equal events render identically.
        assert_eq!(format!("{lhs:?}"), format!("{rhs:?}"),
            "deferred event {i} differs across replay");
    }
}
