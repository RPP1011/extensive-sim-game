//! CPU reference for the WGSL apply_ability dispatcher's chronicle
//! output (#136 / slice γ).
//!
//! The dispatcher in `cg::emit::wgsl_body::lower_cg_stmt_to_wgsl`'s
//! `CgStmt::ApplyAbility` arm writes one chronicle record per
//! chronicle-bearing `EffectOp` directly to the `event_ring` SoA. This
//! module provides a CPU-side reference that produces the exact same
//! 10-word records — same kind tags, same payload offsets, same
//! bitcast semantics — so a CPU↔GPU parity test can compare records
//! byte-for-byte.
//!
//! What this is NOT: a runtime apply executor. The engine-side
//! [`engine::ability::apply::apply_program`] is the source of truth
//! for *which* ApplyEvent variants a cast produces; this module
//! consumes that output and tells you what chronicle record the GPU
//! dispatcher would write for it. Pairing the two gives a complete
//! CPU pipeline equivalent to the GPU dispatch path.
//!
//! **Slice γ caveat.** The GPU dispatcher today uses `agent_id` for
//! both actor + target (self-cast assumption — `CgStmt::ApplyAbility`
//! carries no explicit target operand). This reference mirrors that
//! convention: `caster_id` is written into both the actor and target
//! payload slots regardless of what the source `ApplyEvent` carries
//! for `target`. When `CgStmt::ApplyAbility` grows an explicit target
//! operand and the dispatcher consumes it, this reference grows the
//! same operand.
//!
//! Pin contract:
//!   - Each entry of [`EFFECT_KIND_TO_EVENT_KIND_ID`] (in
//!     `cg::emit::wgsl_body`) corresponds to exactly one variant
//!     handled below; variants outside the table return `None`
//!     (no chronicle counterpart today).
//!   - Stride is 10 u32-words to match the engine's runtime ring
//!     (see `cg::lower::driver`'s `record_stride_u32: 10` constant).
//!   - Header layout: word 0 = kind tag, word 1 = tick.

use engine::ability::apply::ApplyEvent;

/// Per-record stride in u32-words. Mirrors
/// `cg::lower::driver::populate_event_kinds`'s shared
/// `record_stride_u32: 10`.
pub const CHRONICLE_RECORD_STRIDE_U32: usize = 10;

/// Translate one [`ApplyEvent`] into the chronicle record the GPU
/// dispatcher writes for it, or `None` if the variant has no
/// chronicle counterpart in the engine's `EventKindId` enum today.
///
/// Returns a `[u32; 10]` mirroring the dispatcher's per-slot
/// `atomicStore` writes:
///   - `[0]` = kind tag (e.g. `26` for EffectDamageApplied)
///   - `[1]` = tick
///   - `[2..]` = per-variant payload (see variant arms below)
///
/// `caster_id` is the per-thread agent (`agent_id` in WGSL) — used
/// for both actor + target slots per the slice-γ self-cast
/// convention. See module docs.
///
/// `tick` is the runtime tick counter at cast time; mirrors the
/// `tick` preamble local the dispatcher reads from the kernel cfg.
pub fn apply_event_to_chronicle_record(
    event: ApplyEvent,
    tick: u32,
    caster_id: u32,
) -> Option<[u32; CHRONICLE_RECORD_STRIDE_U32]> {
    let mut rec = [0u32; CHRONICLE_RECORD_STRIDE_U32];
    rec[1] = tick;
    match event {
        // --- Damage = 0 → EventKindId::EffectDamageApplied = 26.
        ApplyEvent::Damage { source: _, target: _, amount } => {
            rec[0] = 26;
            rec[2] = caster_id;
            rec[3] = caster_id; // slice γ self-cast
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- Heal = 1 → EventKindId::EffectHealApplied = 27.
        ApplyEvent::Heal { source: _, target: _, amount } => {
            rec[0] = 27;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- Shield = 2 → EventKindId::EffectShieldApplied = 28.
        ApplyEvent::Shield { source: _, target: _, amount } => {
            rec[0] = 28;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- Stun = 3 → EventKindId::EffectStunApplied = 29.
        // Dispatcher computes expires_at_tick = tick + duration.
        ApplyEvent::Stun { target: _, duration_ticks } => {
            rec[0] = 29;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = tick + duration_ticks;
            Some(rec)
        }
        // --- Slow = 4 → EventKindId::EffectSlowApplied = 30.
        // 4-field payload: actor, target, expires_at_tick, factor_q8.
        ApplyEvent::Slow { target: _, duration_ticks, factor_q8 } => {
            rec[0] = 30;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = tick + duration_ticks;
            // Sign-widen i16 → i32 → bitcast to u32.
            rec[5] = (factor_q8 as i32) as u32;
            Some(rec)
        }
        // --- TransferGold = 5 → EventKindId::EffectGoldTransfer = 31.
        // Engine's ApplyEvent doesn't carry a gold amount in its
        // variant (the EffectOp's `amount: i32` is the source); the
        // dispatcher reads payload_a from the packed registry. Until
        // an i64 chronicle field lands, we sign-extend the variant's
        // own caster/target convention.
        // NOTE: the engine ApplyEvent enum doesn't currently spell
        // `TransferGold` (the apply path defers it — see
        // `engine/src/ability/apply.rs` fall-through). Returning None
        // is the honest "no apply-event here yet" result; when the
        // engine wires `ApplyEvent::TransferGold`, this arm grows.
        // --- ModifyStanding = 6 → EventKindId::EffectStandingDelta = 32.
        // Same fall-through note — no `ApplyEvent::ModifyStanding`
        // variant on the engine side today.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::emit::wgsl_body::EFFECT_KIND_TO_EVENT_KIND_ID;
    use engine::ability::apply::ApplyEvent;
    use engine::ids::AgentId;

    fn aid(n: u32) -> AgentId {
        AgentId::new(n).expect("AgentId::new requires non-zero u32")
    }

    #[test]
    fn damage_chronicle_record_matches_dispatcher_layout() {
        // amount=42.0; sign-widen via to_bits to compare exactly.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Damage { source: aid(7), target: aid(11), amount: 42.0 },
            /*tick*/ 100,
            /*caster_id*/ 7,
        )
        .expect("Damage has chronicle counterpart");
        assert_eq!(rec[0], 26, "kind tag — EffectDamageApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "caster slot — slice γ self-cast (was source=7)");
        assert_eq!(rec[3], 7, "target slot — slice γ self-cast (was target=11)");
        assert_eq!(rec[4], 42.0_f32.to_bits(), "amount as bitcast<u32>");
        // Tail words zeroed.
        for i in 5..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }
    }

    #[test]
    fn heal_chronicle_record_uses_kind_27() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Heal { source: aid(1), target: aid(2), amount: 12.5 },
            /*tick*/ 50,
            /*caster_id*/ 3,
        )
        .expect("Heal has chronicle counterpart");
        assert_eq!(rec[0], 27);
        assert_eq!(rec[2], 3);
        assert_eq!(rec[3], 3);
        assert_eq!(rec[4], 12.5_f32.to_bits());
    }

    #[test]
    fn stun_record_writes_expires_at_tick() {
        // Stun's third payload word is `tick + duration_ticks` —
        // mirrors the dispatcher's
        //   let expires_at_tick: u32 = tick + payload_a;
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Stun { target: aid(99), duration_ticks: 17 },
            /*tick*/ 100,
            /*caster_id*/ 4,
        )
        .expect("Stun has chronicle counterpart");
        assert_eq!(rec[0], 29);
        assert_eq!(rec[4], 117, "tick(100) + duration(17) = expires_at_tick");
    }

    #[test]
    fn slow_record_writes_4_payload_fields() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Slow {
                target: aid(5),
                duration_ticks: 10,
                factor_q8: -64, // half-speed slow with sign bit set
            },
            /*tick*/ 100,
            /*caster_id*/ 8,
        )
        .expect("Slow has chronicle counterpart");
        assert_eq!(rec[0], 30);
        assert_eq!(rec[4], 110, "expires_at_tick");
        // Sign-widen i16(-64) → i32(-64) → bitcast<u32> = 0xFFFF_FFC0
        assert_eq!(rec[5], (-64_i32) as u32);
    }

    #[test]
    fn shield_chronicle_record_uses_kind_28() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Shield { source: aid(1), target: aid(2), amount: 25.0 },
            /*tick*/ 50,
            /*caster_id*/ 3,
        )
        .expect("Shield has chronicle counterpart");
        assert_eq!(rec[0], 28);
    }

    #[test]
    fn variants_without_chronicle_counterpart_return_none() {
        // Root / Silence / Fear / Taunt / movement / etc. don't have
        // 1:1 chronicle kinds in the engine today. Mirrors the
        // dispatcher's `// TODO slice γ` arms — no chronicle write,
        // so the CPU reference returns None.
        for ev in [
            ApplyEvent::Root    { target: aid(1), duration_ticks: 5 },
            ApplyEvent::Silence { target: aid(1), duration_ticks: 5 },
            ApplyEvent::Fear    { target: aid(1), duration_ticks: 5 },
            ApplyEvent::Taunt   { target: aid(1), duration_ticks: 5 },
            ApplyEvent::Dash    { source: aid(1), distance: 10.0 },
            ApplyEvent::Blink   { source: aid(1), distance: 10.0 },
            ApplyEvent::Knockback { source: aid(1), target: aid(2), distance: 5.0 },
            ApplyEvent::Pull      { source: aid(1), target: aid(2), distance: 5.0 },
            ApplyEvent::Execute   { target: aid(1), hp_threshold: 50.0 },
            ApplyEvent::SelfDamage{ source: aid(1), amount: 10.0 },
        ] {
            assert!(
                apply_event_to_chronicle_record(ev, 100, 0).is_none(),
                "variant {ev:?} should have no chronicle counterpart \
                 (dispatcher arm carries TODO marker)"
            );
        }
    }

    /// Cross-check: every `(effect_kind, event_kind_id)` entry in the
    /// dispatcher's pinned table must correspond to a CPU-reference
    /// arm that produces the matching kind tag. If a future entry
    /// gets added to the table without a CPU-reference arm, this
    /// test catches the gap.
    #[test]
    fn cpu_reference_covers_all_dispatcher_chronicle_arms() {
        // The four chronicle-bearing variants the engine actually
        // exposes via ApplyEvent today. The remaining entries in
        // EFFECT_KIND_TO_EVENT_KIND_ID (TransferGold=5,
        // ModifyStanding=6) have no ApplyEvent variant yet; they
        // surface in the GPU dispatcher's chronicle output but the
        // CPU pipeline doesn't produce ApplyEvents for them.
        let ev_for_kind = |effect_kind: u32| -> Option<ApplyEvent> {
            match effect_kind {
                0 => Some(ApplyEvent::Damage    { source: aid(1), target: aid(2), amount: 1.0 }),
                1 => Some(ApplyEvent::Heal      { source: aid(1), target: aid(2), amount: 1.0 }),
                2 => Some(ApplyEvent::Shield    { source: aid(1), target: aid(2), amount: 1.0 }),
                3 => Some(ApplyEvent::Stun      { target: aid(2), duration_ticks: 5 }),
                4 => Some(ApplyEvent::Slow      { target: aid(2), duration_ticks: 5, factor_q8: 128 }),
                5 | 6 => None, // see comment above
                _ => panic!("unexpected effect_kind in table"),
            }
        };

        for &(effect_kind, expected_event_kind_id) in EFFECT_KIND_TO_EVENT_KIND_ID {
            let Some(ev) = ev_for_kind(effect_kind) else {
                continue;
            };
            let rec = apply_event_to_chronicle_record(ev, 0, 0)
                .unwrap_or_else(|| {
                    panic!(
                        "EFFECT_KIND_TO_EVENT_KIND_ID entry effect_kind={effect_kind} \
                         (event_kind_id={expected_event_kind_id}) has no CPU-reference arm"
                    );
                });
            assert_eq!(
                rec[0], expected_event_kind_id,
                "CPU reference for effect_kind={effect_kind} must produce \
                 kind tag {expected_event_kind_id} (matching the dispatcher table)"
            );
        }
    }
}
