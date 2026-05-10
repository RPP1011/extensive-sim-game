//! Plan G G2.5 — engine-level pin proving the cast → interrupt →
//! no-resolve sequence works at the apply layer.
//!
//! This is the engine-side analog of what
//! `firebolt_interrupt_probe_runtime` exercises on real wgpu hardware
//! (in flight). The flow models the per-fixture sim semantics:
//!
//!   1. Tick T0:  cast initiation. `apply_program(prog)` returns
//!                `ApplyEvent::CastBegin`. Per-fixture consumer would
//!                stamp `busy_until_tick = T0 + duration`.
//!   2. Tick T0+1:  external Damaged event lands on the caster. The
//!                  per-fixture consumer queries
//!                  `should_interrupt(busy, Damage, current_tick)` to
//!                  decide whether to clear busy. With
//!                  `interrupts: standard` (the firebolt_probe default),
//!                  the answer is true → busy cleared.
//!   3. Tick T0+duration:  the busy-resolution kernel SHOULD NOT fire
//!                         `apply_pending_program` because
//!                         busy_until_tick was cleared. Pin: hp does
//!                         NOT take the deferred damage.
//!
//! No GPU dispatch — this test uses the in-Rust apply layer +
//! interrupt-decision helper. The .sim's per-fixture rule shape is
//! validated separately by
//! `crates/dsl_compiler/tests/firebolt_interrupt_probe_lower.rs`;
//! the GPU lifecycle by `firebolt_interrupt_probe_runtime`.

use engine::ability::apply::{apply_pending_program, apply_program, ApplyEvent};
use engine::ability::interrupt::{
    BusyState, InterruptKind, InterruptMask, should_interrupt,
};
use engine::ability::program::{AbilityProgram, CasterStats, EffectOp, Gate};
use engine::ids::AgentId;

fn caster() -> AgentId { AgentId::new(1).unwrap() }
fn target() -> AgentId { AgentId::new(2).unwrap() }

/// Build a Firebolt-shaped program: CastBegin in `effects`,
/// Damage(25) in `pending_program`, with `cast_interrupt_mask` set
/// to standard (the default for `interrupts: standard`).
fn firebolt_program() -> AbilityProgram {
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
    prog.cast_interrupt_mask = Some(InterruptMask::standard());
    prog.pending_program.push(EffectOp::Damage { amount: 25.0 });
    prog
}

/// Pin: when an interrupt fires mid-cast, the deferred damage NEVER
/// resolves. Models the firebolt_interrupt_probe.sim flow at the
/// apply / interrupt-helper layer.
#[test]
fn cast_interrupted_at_t1_skips_resolve_at_t3() {
    let prog = firebolt_program();
    // -- Tick 0: cast initiation. --
    let immediate = apply_program(
        &prog, caster(), target(), 0, 0xCAFE,
        &CasterStats::default(), &CasterStats::default(),
    );
    assert_eq!(immediate.len(), 1);
    let cast_duration = match immediate[0] {
        ApplyEvent::CastBegin { duration_ticks, .. } => duration_ticks as u32,
        ref other => panic!("expected CastBegin; got {other:?}"),
    };
    let resolve_tick = 0 + cast_duration; // = 3

    // -- Tick 1: Damaged event lands on caster.
    //    Per-fixture consumer's interrupt check. --
    let busy_at_t1 = BusyState {
        busy_until_tick:      resolve_tick,
        busy_with_ability_id: 1,
        busy_target_slot:     0,
        interrupt_mask:       prog.cast_interrupt_mask.expect("populated above"),
    };
    let interrupted = should_interrupt(busy_at_t1, InterruptKind::Damage, /*current_tick*/ 1);
    assert!(interrupted, "Damage in standard mask + busy at tick 1 → interrupt fires");

    // The per-fixture consumer reacts by clearing busy state.
    // Model it: busy_until_tick → 0 for the remaining ticks.
    let busy_after_interrupt = BusyState {
        busy_until_tick:      0,
        busy_with_ability_id: 0,
        busy_target_slot:     0,
        interrupt_mask:       InterruptMask::standard(), // mask is per-cast, but irrelevant when idle
    };

    // -- Tick 3: would-be resolve. The busy-resolution kernel's
    //    `where (busy_until_tick > 0 && tick >= busy_until_tick)`
    //    fails because busy_until_tick == 0. NO call to
    //    apply_pending_program. --
    assert!(!busy_after_interrupt.is_busy_at(resolve_tick),
        "post-interrupt busy state must be idle");

    // Negative pin: if we DID call apply_pending_program despite
    // the cleared busy, it would emit Damage(25). Confirms the
    // damage stream is what we'd avoid by not firing.
    let counterfactual = apply_pending_program(
        &prog, caster(), target(), resolve_tick as u64, 0xCAFE,
        &CasterStats::default(), &CasterStats::default(),
    );
    assert_eq!(counterfactual.len(), 1,
        "sanity check — without the busy-state guard, pending resolve emits Damage");
}

/// Pin: an uninterruptible cast (`interrupts: none`) ignores
/// mid-cast Damaged events — busy state stays set, deferred damage
/// fires at resolve.
#[test]
fn uninterruptible_cast_ignores_mid_cast_damage() {
    let mut prog = firebolt_program();
    prog.cast_interrupt_mask = Some(InterruptMask::none()); // BindSoul-style
    let busy_at_t1 = BusyState {
        busy_until_tick:      3,
        busy_with_ability_id: 1,
        busy_target_slot:     0,
        interrupt_mask:       prog.cast_interrupt_mask.unwrap(),
    };
    assert!(!should_interrupt(busy_at_t1, InterruptKind::Damage, 1),
        "InterruptMask::none() rejects Damage");
    assert!(!should_interrupt(busy_at_t1, InterruptKind::Stun, 1),
        "InterruptMask::none() rejects Stun");
    // The consumer takes no action; busy stays set; resolve fires.
    let resolve = apply_pending_program(
        &prog, caster(), target(), 3, 0xCAFE,
        &CasterStats::default(), &CasterStats::default(),
    );
    assert_eq!(resolve.len(), 1, "uninterruptible cast resolves normally");
    assert!(matches!(resolve[0], ApplyEvent::Damage { amount, .. } if amount == 25.0));
}

/// Pin: Movement is opt-in. A cast with `interrupts: standard`
/// does NOT cancel on Movement; one with `interrupts: standard +
/// { movement }` does.
#[test]
fn movement_interrupt_is_opt_in() {
    let busy_std = BusyState {
        busy_until_tick: 3, busy_with_ability_id: 1, busy_target_slot: 0,
        interrupt_mask: InterruptMask::standard(),
    };
    assert!(!should_interrupt(busy_std, InterruptKind::Movement, 1),
        "standard mask does NOT include Movement");

    let busy_plus = BusyState {
        interrupt_mask: InterruptMask::standard().with(InterruptKind::Movement),
        ..busy_std
    };
    assert!(should_interrupt(busy_plus, InterruptKind::Movement, 1),
        "standard + {{ movement }} cancels on caster move");
}
