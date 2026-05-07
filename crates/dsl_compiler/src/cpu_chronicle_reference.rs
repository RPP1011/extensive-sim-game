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
//! **Slice ε update.** Originally (slice γ), the GPU dispatcher
//! hardcoded `agent_id` for both actor + target (self-cast). After
//! slice ε plumbed explicit `caster` + `target` operands through
//! `CgStmt::ApplyAbility` (commits `92572af8` / `d0bc37fd`), the
//! dispatcher writes whichever ids the lowering supplies.
//!
//! This reference mirrors that: takes `caster_id` and `target_id`
//! separately and writes them into actor (slot 2) and target (slot 3)
//! respectively. For self-cast callers, pass `target_id == caster_id`
//! to preserve the prior chronicle byte layout.
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
/// `caster_id` and `target_id` are written into the actor (slot 2)
/// and target (slot 3) chronicle payload words respectively. Both
/// are u32 raw AgentIds. For self-cast callers (the slice-γ
/// default), pass `target_id == caster_id`. See module docs for
/// the slice-ε plumbing context.
///
/// `tick` is the runtime tick counter at cast time; mirrors the
/// `tick` preamble local the dispatcher reads from the kernel cfg.
///
/// **Slice ε part 1 update**: now takes `target_id: u32` separately
/// from `caster_id`. The GPU dispatcher writes the `target` operand
/// (lowered from `apply_ability ... target <expr>` source) into
/// chronicle payload word 3 distinct from the caster slot in
/// payload word 2. For self-cast callers (slice-γ default), pass
/// `target_id == caster_id` to preserve the prior byte layout.
pub fn apply_event_to_chronicle_record(
    event: ApplyEvent,
    tick: u32,
    caster_id: u32,
    target_id: u32,
) -> Option<[u32; CHRONICLE_RECORD_STRIDE_U32]> {
    let mut rec = [0u32; CHRONICLE_RECORD_STRIDE_U32];
    rec[1] = tick;
    match event {
        // --- Damage = 0 → EventKindId::EffectDamageApplied = 26.
        ApplyEvent::Damage { source: _, target: _, amount } => {
            rec[0] = 26;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- Heal = 1 → EventKindId::EffectHealApplied = 27.
        ApplyEvent::Heal { source: _, target: _, amount } => {
            rec[0] = 27;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- Shield = 2 → EventKindId::EffectShieldApplied = 28.
        ApplyEvent::Shield { source: _, target: _, amount } => {
            rec[0] = 28;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- Stun = 3 → EventKindId::EffectStunApplied = 29.
        // Dispatcher computes expires_at_tick = tick + duration.
        ApplyEvent::Stun { target: _, duration_ticks } => {
            rec[0] = 29;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            Some(rec)
        }
        // --- Root = 8 → EventKindId::EffectRootApplied = 43.
        // Wave 2 piece 1 — same shape as Stun: 3 payload words
        // (actor + target + expires_at_tick = tick + duration).
        ApplyEvent::Root { target: _, duration_ticks } => {
            rec[0] = 43;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            Some(rec)
        }
        // --- Silence = 9 → EventKindId::EffectSilenceApplied = 44.
        // Wave 2 piece 1 — same shape as Stun.
        ApplyEvent::Silence { target: _, duration_ticks } => {
            rec[0] = 44;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            Some(rec)
        }
        // --- Fear = 10 → EventKindId::EffectFearApplied = 45.
        // Wave 2 piece 1 — same shape as Stun.
        ApplyEvent::Fear { target: _, duration_ticks } => {
            rec[0] = 45;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            Some(rec)
        }
        // --- Taunt = 11 → EventKindId::EffectTauntApplied = 46.
        // Wave 2 piece 1 — same shape as Stun.
        ApplyEvent::Taunt { target: _, duration_ticks } => {
            rec[0] = 46;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            Some(rec)
        }
        // --- Dash = 12 → EventKindId::EffectDashApplied = 47.
        // Wave 2 piece 2 — caster-self motion. The engine event has no
        // target field; the GPU dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = bitcast<u32>(distance)
        // — i.e. distance lands at payload word 1 (NOT word 2 like the
        // forced-motion shape). The CPU reference mirrors this exactly.
        ApplyEvent::Dash { source: _, distance } => {
            rec[0] = 47;
            rec[2] = caster_id;
            rec[3] = distance.to_bits();
            Some(rec)
        }
        // --- Blink = 13 → EventKindId::EffectBlinkApplied = 48.
        // Wave 2 piece 2 — same shape as Dash (caster-self motion).
        ApplyEvent::Blink { source: _, distance } => {
            rec[0] = 48;
            rec[2] = caster_id;
            rec[3] = distance.to_bits();
            Some(rec)
        }
        // --- Knockback = 14 → EventKindId::EffectKnockbackApplied = 49.
        // Wave 2 piece 2 — forced motion on a target. 3-payload-word
        // chronicle record: actor + target + distance (bitcast f32 →
        // u32). Same shape family as Damage/Heal/Shield/Execute.
        ApplyEvent::Knockback { source: _, target: _, distance } => {
            rec[0] = 49;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = distance.to_bits();
            Some(rec)
        }
        // --- Pull = 15 → EventKindId::EffectPullApplied = 50.
        // Wave 2 piece 2 — same shape as Knockback (forced motion).
        ApplyEvent::Pull { source: _, target: _, distance } => {
            rec[0] = 50;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = distance.to_bits();
            Some(rec)
        }
        // --- Slow = 4 → EventKindId::EffectSlowApplied = 30.
        // 4-field payload: actor, target, expires_at_tick, factor_q8.
        ApplyEvent::Slow { target: _, duration_ticks, factor_q8 } => {
            rec[0] = 30;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            // Sign-widen i16 → i32 → bitcast to u32.
            rec[5] = (factor_q8 as i32) as u32;
            Some(rec)
        }
        // --- TransferGold = 5 → EventKindId::EffectGoldTransfer = 31.
        // Engine event carries amount as i64 (host widens for ledger
        // arithmetic); the GPU dispatcher writes the EffectOp's i32
        // amount sign-widened to u32. Cascade chronicle decode reads
        // u32 + sign-extends to i64.
        ApplyEvent::TransferGold { source: _, target: _, amount } => {
            rec[0] = 31;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = (amount as i32) as u32;
            Some(rec)
        }
        // --- ModifyStanding = 6 → EventKindId::EffectStandingDelta = 32.
        // delta is i16 on EffectOp side, i32 on chronicle side. GPU
        // dispatcher sign-widens i16 → i32 → bitcast<u32> in the
        // ModifyStanding arm; this CPU reference mirrors the same.
        ApplyEvent::ModifyStanding { source: _, target: _, delta } => {
            rec[0] = 32;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = (delta as i32) as u32;
            Some(rec)
        }
        // --- SelfDamage = 17 → EventKindId::EffectSelfDamageApplied = 39.
        // Bleed verb swap (Task #138 follow-on, 2026-05-06). Self-damage
        // targets the caster, so the GPU dispatcher writes caster_slot
        // into BOTH actor (slot 2) and target (slot 3). The CPU
        // reference's call sites pass `target_id == caster_id` for
        // this variant, but we explicitly write `caster_id` into slot 3
        // so the record is correct even when the caller forgets the
        // self-target convention.
        ApplyEvent::SelfDamage { source: _, amount } => {
            rec[0] = 39;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = amount.to_bits();
            Some(rec)
        }
        // --- LifeSteal = 18 → EventKindId::EffectLifeStealApplied = 40.
        // Vampirize verb swap (Task #138 follow-on, mirror of Bleed at
        // `486eb08f`). 4-field payload: actor, target, expires_at_tick,
        // fraction_q8. Same shape as Slow (kind=30). Self-cast LifeSteal
        // targets the caster — `apply_program` already returns
        // `ApplyEvent::LifeSteal { target: caster, ... }`, but we
        // explicitly preserve the caller's `caster_id` and `target_id`
        // so the record byte-layout matches the GPU dispatcher (which
        // writes whatever `caster_slot` / `target_slot` the lowering
        // supplied).
        ApplyEvent::LifeSteal { target: _, duration_ticks, fraction_q8 } => {
            rec[0] = 40;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            // Sign-widen i16 → i32 → bitcast to u32.
            rec[5] = (fraction_q8 as i32) as u32;
            Some(rec)
        }
        // --- DamageModify = 19 → EventKindId::EffectDamageModifyApplied = 41.
        // Fortify verb swap (Task #138 follow-on, mirror of Vampirize at
        // `60115f64`). 4-field payload: actor, target, expires_at_tick,
        // multiplier_q8. Same shape as Slow (kind=30) / LifeSteal
        // (kind=40). Self-cast DamageModify targets the caster —
        // `apply_program` already returns
        // `ApplyEvent::DamageModify { target: caster, ... }`, but we
        // explicitly preserve the caller's `caster_id` and `target_id`
        // so the record byte-layout matches the GPU dispatcher (which
        // writes whatever `caster_slot` / `target_slot` the lowering
        // supplied).
        ApplyEvent::DamageModify { target: _, duration_ticks, multiplier_q8 } => {
            rec[0] = 41;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = tick + duration_ticks;
            // Sign-widen i16 → i32 → bitcast to u32.
            rec[5] = (multiplier_q8 as i32) as u32;
            Some(rec)
        }
        // --- Execute = 16 → EventKindId::EffectExecuteApplied = 42.
        // Reap verb swap (Task #138 follow-on, mirror of Fortify at
        // `001ae9a6`). 3-field payload: actor, target, hp_threshold (f32).
        // Same shape family as Damage (kind=26) / Heal (kind=27) — third
        // payload word is bitcast f32. The when-condition `target.hp <
        // hp_threshold` is NOT evaluated by apply_program (the
        // `when_per_effect[i]` field stays unconsulted today), so the
        // record fires unconditionally; downstream consumers (e.g. the
        // duel_abilities ApplyExecuteFromChronicle re-emit) ferry the
        // record into the host sim's defeat path. Reap's outer verb
        // gate provides the threshold check upstream.
        //
        // `ApplyEvent::Execute` carries only `{ target, hp_threshold }`
        // — no source field — so caster_id is supplied by the caller
        // (mirroring `ApplyEvent::Stun`'s pattern). Slot 2 = caster_id,
        // slot 3 = target_id, exactly like the dispatcher's WGSL arm.
        ApplyEvent::Execute { target: _, hp_threshold } => {
            rec[0] = 42;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = hp_threshold.to_bits();
            Some(rec)
        }
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
            /*caster_id*/ 7, /*target_id*/ 7,
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
            /*caster_id*/ 3, /*target_id*/ 3,
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
            /*caster_id*/ 4, /*target_id*/ 4,
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
            /*caster_id*/ 8, /*target_id*/ 8,
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
            /*caster_id*/ 3, /*target_id*/ 3,
        )
        .expect("Shield has chronicle counterpart");
        assert_eq!(rec[0], 28);
    }

    /// Slice ε part 1 pin: when caster_id and target_id differ, the
    /// chronicle record's actor (slot 2) and target (slot 3) words
    /// take distinct values — mirroring the GPU dispatcher's
    /// distinct `caster_slot` / `target_slot` writes.
    #[test]
    fn distinct_caster_and_target_write_distinct_slots() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Damage { source: aid(1), target: aid(2), amount: 100.0 },
            /*tick*/ 50,
            /*caster_id*/ 7,
            /*target_id*/ 11,
        )
        .unwrap();
        assert_eq!(rec[2], 7, "actor slot uses caster_id");
        assert_eq!(rec[3], 11, "target slot uses target_id (distinct from caster)");
        // Self-cast call site preserves slice-γ behavior: caster=target.
        let rec_self = apply_event_to_chronicle_record(
            ApplyEvent::Damage { source: aid(1), target: aid(2), amount: 100.0 },
            50, 7, 7,
        ).unwrap();
        assert_eq!(rec_self[2], rec_self[3], "self-cast collapses both slots");
    }

    #[test]
    fn variants_without_chronicle_counterpart_return_none() {
        // ApplyEvent variants without chronicle counterparts today.
        // After Wave 2 piece 2 (movement EffectOps Dash/Blink/Knockback/
        // Pull), every slice-γ-and-onward chronicle-bearing variant has
        // a wire-up; the variants below are all deferred-infrastructure
        // ApplyEvents (Summon/Harvest/PlaceVoxel/Stealth/Charm/etc.)
        // that emit ApplyEvents but have no engine `EventKindId` yet.
        //
        // Status effects already wired up:
        // - SelfDamage (Bleed verb swap, Task #138 follow-on,
        //   2026-05-06) → kind=39, see `self_damage_chronicle_record_uses_kind_39`.
        // - Execute (Reap verb swap, Task #138 follow-on, mirror of
        //   Fortify) → kind=42, see `execute_chronicle_record_uses_kind_42`.
        // - Root/Silence/Fear/Taunt (Wave 2 piece 1) →
        //   kinds 43..46, see `control_status_chronicle_records_use_kinds_43_46`.
        // - Dash/Blink/Knockback/Pull (Wave 2 piece 2, this slice) →
        //   kinds 47..50, see `movement_chronicle_records_use_kinds_47_50`.
        for ev in [
            ApplyEvent::Summon  { source: aid(1), template_hash: 0xDEADBEEF, count: 2, lifetime_ticks: 100 },
            ApplyEvent::Harvest { source: aid(1), kind_hash: 0xCAFEBABE, amount: 5 },
            ApplyEvent::PlaceVoxel { source: aid(1), kind_hash: 0xFACEFEED },
        ] {
            assert!(
                apply_event_to_chronicle_record(ev, 100, 0, 0).is_none(),
                "variant {ev:?} should have no chronicle counterpart \
                 (dispatcher arm carries TODO marker)"
            );
        }
    }

    /// Wave 2 piece 2 — movement EffectOps (Dash/Blink/Knockback/Pull).
    /// Two distinct shapes:
    ///   - Dash/Blink: caster-self motion. Engine event has no target
    ///     field; the chronicle record stores distance at payload word 1
    ///     (= ring slot offset 3), NOT word 2 like the forced-motion
    ///     shape. Slot 4 stays zero.
    ///   - Knockback/Pull: forced motion on a target. Same shape as
    ///     Damage/Heal/Shield/Execute — actor + target + distance
    ///     (bitcast f32 → u32) at payload word 2 (= ring slot offset 4).
    /// Pin per-variant kind tags (47..50) and per-shape distance offsets.
    #[test]
    fn movement_chronicle_records_use_kinds_47_50() {
        // Dash — caster-self motion (no target in engine event).
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Dash { source: aid(7), distance: 12.5 },
            /*tick*/ 100,
            /*caster_id*/ 7,
            /*target_id*/ 7,
        )
        .expect("Dash has chronicle counterpart");
        assert_eq!(rec[0], 47, "Dash: kind tag — EffectDashApplied");
        assert_eq!(rec[1], 100, "Dash: tick");
        assert_eq!(rec[2], 7, "Dash: actor slot — caster_id");
        assert_eq!(rec[3], 12.5_f32.to_bits(), "Dash: distance at payload word 1");
        for i in 4..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "Dash: tail word {i} should be zero");
        }

        // Blink — same shape as Dash.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Blink { source: aid(7), distance: 8.0 },
            /*tick*/ 50,
            /*caster_id*/ 7,
            /*target_id*/ 7,
        )
        .expect("Blink has chronicle counterpart");
        assert_eq!(rec[0], 48, "Blink: kind tag — EffectBlinkApplied");
        assert_eq!(rec[1], 50, "Blink: tick");
        assert_eq!(rec[2], 7, "Blink: actor slot — caster_id");
        assert_eq!(rec[3], 8.0_f32.to_bits(), "Blink: distance at payload word 1");
        for i in 4..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "Blink: tail word {i} should be zero");
        }

        // Knockback — forced motion on a target.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Knockback { source: aid(7), target: aid(11), distance: 5.0 },
            /*tick*/ 200,
            /*caster_id*/ 7,
            /*target_id*/ 11,
        )
        .expect("Knockback has chronicle counterpart");
        assert_eq!(rec[0], 49, "Knockback: kind tag — EffectKnockbackApplied");
        assert_eq!(rec[1], 200, "Knockback: tick");
        assert_eq!(rec[2], 7, "Knockback: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Knockback: target slot — target_id");
        assert_eq!(rec[4], 5.0_f32.to_bits(), "Knockback: distance at payload word 2");
        for i in 5..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "Knockback: tail word {i} should be zero");
        }

        // Pull — same shape as Knockback.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Pull { source: aid(7), target: aid(11), distance: 3.5 },
            /*tick*/ 300,
            /*caster_id*/ 7,
            /*target_id*/ 11,
        )
        .expect("Pull has chronicle counterpart");
        assert_eq!(rec[0], 50, "Pull: kind tag — EffectPullApplied");
        assert_eq!(rec[1], 300, "Pull: tick");
        assert_eq!(rec[2], 7, "Pull: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Pull: target slot — target_id");
        assert_eq!(rec[4], 3.5_f32.to_bits(), "Pull: distance at payload word 2");
        for i in 5..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "Pull: tail word {i} should be zero");
        }
    }

    /// Wave 2 piece 1 — control statuses (Root/Silence/Fear/Taunt).
    /// Each shares Stun's shape: 3 payload words (actor + target +
    /// expires_at_tick = tick + duration). Pin per-variant kind tags
    /// (43..46) and the expires_at_tick computation.
    #[test]
    fn control_status_chronicle_records_use_kinds_43_46() {
        let cases: &[(ApplyEvent, u32, &str)] = &[
            (ApplyEvent::Root    { target: aid(11), duration_ticks: 50 }, 43, "Root"),
            (ApplyEvent::Silence { target: aid(11), duration_ticks: 50 }, 44, "Silence"),
            (ApplyEvent::Fear    { target: aid(11), duration_ticks: 50 }, 45, "Fear"),
            (ApplyEvent::Taunt   { target: aid(11), duration_ticks: 50 }, 46, "Taunt"),
        ];
        for (ev, expected_kind, name) in cases {
            let rec = apply_event_to_chronicle_record(
                *ev,
                /*tick*/ 100,
                /*caster_id*/ 7,
                /*target_id*/ 11,
            )
            .unwrap_or_else(|| panic!("{name} has chronicle counterpart"));
            assert_eq!(rec[0], *expected_kind, "{name}: kind tag");
            assert_eq!(rec[1], 100, "{name}: tick");
            assert_eq!(rec[2], 7, "{name}: actor slot — caster_id");
            assert_eq!(rec[3], 11, "{name}: target slot — target_id");
            assert_eq!(rec[4], 150, "{name}: expires_at_tick = tick(100) + duration(50)");
            for i in 5..CHRONICLE_RECORD_STRIDE_U32 {
                assert_eq!(rec[i], 0, "{name}: tail word {i} should be zero");
            }
        }
    }

    /// Bleed verb swap (Task #138 follow-on, 2026-05-06): SelfDamage
    /// produces kind=39 records with caster_id in both actor (slot 2)
    /// and target (slot 3). Round-trip the amount via to_bits().
    #[test]
    fn self_damage_chronicle_record_uses_kind_39() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::SelfDamage { source: aid(7), amount: 10.0 },
            /*tick*/ 100,
            /*caster_id*/ 7, /*target_id*/ 7,
        )
        .expect("SelfDamage has chronicle counterpart");
        assert_eq!(rec[0], 39, "kind tag — EffectSelfDamageApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id (the bleeder)");
        assert_eq!(
            rec[3], 7,
            "target slot — caster_id (self-damage targets caster)",
        );
        assert_eq!(rec[4], 10.0_f32.to_bits(), "amount as bitcast<u32>");
        // Tail words zeroed.
        for i in 5..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }
    }

    /// Vampirize verb swap (Task #138 follow-on, mirror of Bleed):
    /// LifeSteal produces kind=40 records with the same 4-payload-word
    /// shape as Slow — actor, target, expires_at_tick (=tick+duration),
    /// fraction_q8 (sign-widened i16 → i32 → u32).
    #[test]
    fn life_steal_chronicle_record_uses_kind_40() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::LifeSteal {
                target: aid(7),
                duration_ticks: 50,
                fraction_q8: 128,
            },
            /*tick*/ 100,
            /*caster_id*/ 7, /*target_id*/ 7,
        )
        .expect("LifeSteal has chronicle counterpart");
        assert_eq!(rec[0], 40, "kind tag — EffectLifeStealApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id");
        assert_eq!(rec[3], 7, "target slot — target_id (self-cast: ==caster)");
        assert_eq!(rec[4], 150, "expires_at_tick = tick(100) + duration(50)");
        assert_eq!(rec[5], 128, "fraction_q8 = 128 (= 0.5×) sign-widened");
        // Tail words zeroed.
        for i in 6..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }

        // Sign preservation: negative fraction_q8 sign-extends i16 → i32 → u32.
        let rec_neg = apply_event_to_chronicle_record(
            ApplyEvent::LifeSteal {
                target: aid(3),
                duration_ticks: 10,
                fraction_q8: -64,
            },
            50, 3, 3,
        ).unwrap();
        assert_eq!(rec_neg[0], 40);
        assert_eq!(rec_neg[4], 60, "expires_at_tick = 50 + 10");
        assert_eq!(rec_neg[5], (-64_i32) as u32, "negative fraction_q8 sign-widens correctly");
    }

    /// Reap verb swap (Task #138 follow-on, mirror of Fortify at
    /// `001ae9a6`): Execute produces kind=42 records with a 3-payload-word
    /// shape — actor, target, hp_threshold (bitcast f32). The
    /// when-condition `target.hp < hp_threshold` is NOT evaluated here
    /// (apply_program doesn't consult `when_per_effect[i]` today); the
    /// record fires unconditionally and the duel_abilities Reap verb's
    /// outer `when` clause provides the threshold gate upstream.
    /// `ApplyEvent::Execute` carries only `{ target, hp_threshold }`,
    /// so the caller-supplied caster_id lands in slot 2 and target_id
    /// in slot 3 — same convention as Stun/Slow/LifeSteal/DamageModify.
    #[test]
    fn execute_chronicle_record_uses_kind_42() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Execute {
                target: aid(11),
                hp_threshold: 20.0,
            },
            /*tick*/ 100,
            /*caster_id*/ 7,
            /*target_id*/ 11,
        )
        .expect("Execute has chronicle counterpart");
        assert_eq!(rec[0], 42, "kind tag — EffectExecuteApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id");
        assert_eq!(rec[3], 11, "target slot — target_id");
        assert_eq!(rec[4], 20.0_f32.to_bits(), "hp_threshold as bitcast<u32>");
        // Tail words zeroed.
        for i in 5..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }

        // Self-cast collapse: caster==target sets both slots equal.
        let rec_self = apply_event_to_chronicle_record(
            ApplyEvent::Execute {
                target: aid(7),
                hp_threshold: 50.0,
            },
            50, 7, 7,
        ).unwrap();
        assert_eq!(rec_self[0], 42);
        assert_eq!(rec_self[2], rec_self[3], "self-cast collapses both slots");
        assert_eq!(rec_self[4], 50.0_f32.to_bits(), "different threshold round-trips correctly");
    }

    /// Fortify verb swap (Task #138 follow-on, mirror of Vampirize):
    /// DamageModify produces kind=41 records with the same 4-payload-word
    /// shape as Slow / LifeSteal — actor, target, expires_at_tick
    /// (=tick+duration), multiplier_q8 (sign-widened i16 → i32 → u32).
    #[test]
    fn damage_modify_chronicle_record_uses_kind_41() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::DamageModify {
                target: aid(7),
                duration_ticks: 50,
                multiplier_q8: 128,
            },
            /*tick*/ 100,
            /*caster_id*/ 7, /*target_id*/ 7,
        )
        .expect("DamageModify has chronicle counterpart");
        assert_eq!(rec[0], 41, "kind tag — EffectDamageModifyApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id");
        assert_eq!(rec[3], 7, "target slot — target_id (self-cast: ==caster)");
        assert_eq!(rec[4], 150, "expires_at_tick = tick(100) + duration(50)");
        assert_eq!(rec[5], 128, "multiplier_q8 = 128 (= 0.5×) sign-widened");
        // Tail words zeroed.
        for i in 6..CHRONICLE_RECORD_STRIDE_U32 {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }

        // Sign preservation: negative multiplier_q8 sign-extends i16 → i32 → u32.
        let rec_neg = apply_event_to_chronicle_record(
            ApplyEvent::DamageModify {
                target: aid(3),
                duration_ticks: 10,
                multiplier_q8: -64,
            },
            50, 3, 3,
        ).unwrap();
        assert_eq!(rec_neg[0], 41);
        assert_eq!(rec_neg[4], 60, "expires_at_tick = 50 + 10");
        assert_eq!(rec_neg[5], (-64_i32) as u32, "negative multiplier_q8 sign-widens correctly");
    }

    /// Cross-check: every `(effect_kind, event_kind_id)` entry in the
    /// dispatcher's pinned table must correspond to a CPU-reference
    /// arm that produces the matching kind tag. If a future entry
    /// gets added to the table without a CPU-reference arm, this
    /// test catches the gap.
    #[test]
    fn cpu_reference_covers_all_dispatcher_chronicle_arms() {
        // Every chronicle-bearing effect-kind entry must have a
        // matching CPU-reference arm. After wiring TransferGold +
        // ModifyStanding ApplyEvents (engine/src/ability/apply.rs),
        // SelfDamage (Bleed verb swap, Task #138 follow-on,
        // 2026-05-06), LifeSteal (Vampirize verb swap, Task #138
        // follow-on, mirror of Bleed), DamageModify (Fortify verb swap,
        // Task #138 follow-on, mirror of Vampirize), and Execute (Reap
        // verb swap, Task #138 follow-on, mirror of Fortify — closes
        // all 8 duel_abilities verbs), all 11 entries are covered —
        // no None fall-throughs.
        let ev_for_kind = |effect_kind: u32| -> ApplyEvent {
            match effect_kind {
                0  => ApplyEvent::Damage         { source: aid(1), target: aid(2), amount: 1.0 },
                1  => ApplyEvent::Heal           { source: aid(1), target: aid(2), amount: 1.0 },
                2  => ApplyEvent::Shield         { source: aid(1), target: aid(2), amount: 1.0 },
                3  => ApplyEvent::Stun           { target: aid(2), duration_ticks: 5 },
                4  => ApplyEvent::Slow           { target: aid(2), duration_ticks: 5, factor_q8: 128 },
                5  => ApplyEvent::TransferGold   { source: aid(1), target: aid(2), amount: 7 },
                6  => ApplyEvent::ModifyStanding { source: aid(1), target: aid(2), delta: 3 },
                8  => ApplyEvent::Root           { target: aid(2), duration_ticks: 5 },
                9  => ApplyEvent::Silence        { target: aid(2), duration_ticks: 5 },
                10 => ApplyEvent::Fear           { target: aid(2), duration_ticks: 5 },
                11 => ApplyEvent::Taunt          { target: aid(2), duration_ticks: 5 },
                12 => ApplyEvent::Dash           { source: aid(1), distance: 10.0 },
                13 => ApplyEvent::Blink          { source: aid(1), distance: 10.0 },
                14 => ApplyEvent::Knockback      { source: aid(1), target: aid(2), distance: 5.0 },
                15 => ApplyEvent::Pull           { source: aid(1), target: aid(2), distance: 5.0 },
                16 => ApplyEvent::Execute        { target: aid(2), hp_threshold: 20.0 },
                17 => ApplyEvent::SelfDamage     { source: aid(1), amount: 1.0 },
                18 => ApplyEvent::LifeSteal      { target: aid(1), duration_ticks: 5, fraction_q8: 128 },
                19 => ApplyEvent::DamageModify   { target: aid(1), duration_ticks: 5, multiplier_q8: 128 },
                _ => panic!("unexpected effect_kind in table"),
            }
        };

        for &(effect_kind, expected_event_kind_id) in EFFECT_KIND_TO_EVENT_KIND_ID {
            let ev = ev_for_kind(effect_kind);
            let rec = apply_event_to_chronicle_record(ev, 0, 0, 0)
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

    #[test]
    fn transfer_gold_chronicle_record_uses_kind_31() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::TransferGold { source: aid(1), target: aid(2), amount: 42 },
            /*tick*/ 100,
            /*caster_id*/ 9, /*target_id*/ 9,
        )
        .expect("TransferGold has chronicle counterpart");
        assert_eq!(rec[0], 31, "EffectGoldTransfer kind tag");
        assert_eq!(rec[2], 9, "caster slot — slice γ self-cast");
        assert_eq!(rec[3], 9);
        assert_eq!(rec[4], 42, "amount as i32 → u32 (positive value preserves bits)");

        // Sign preservation: negative amount sign-extends through the
        // bitcast — the dispatcher's `bitcast<i32>` recovers the
        // negative value.
        let rec_neg = apply_event_to_chronicle_record(
            ApplyEvent::TransferGold { source: aid(1), target: aid(2), amount: -7 },
            100, 9, 9,
        ).unwrap();
        assert_eq!(rec_neg[4], (-7_i32) as u32, "negative amount sign-widens correctly");
    }

    #[test]
    fn modify_standing_chronicle_record_uses_kind_32() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::ModifyStanding { source: aid(1), target: aid(2), delta: -25 },
            /*tick*/ 100,
            /*caster_id*/ 4, /*target_id*/ 4,
        )
        .expect("ModifyStanding has chronicle counterpart");
        assert_eq!(rec[0], 32, "EffectStandingDelta kind tag");
        assert_eq!(rec[2], 4);
        assert_eq!(rec[3], 4);
        assert_eq!(rec[4], (-25_i32) as u32, "delta sign-widens i16 → i32 → u32");
    }
}
