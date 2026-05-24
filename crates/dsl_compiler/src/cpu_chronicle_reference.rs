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
//!   - Stride is 11 u32-words to match the engine's runtime ring
//!     (see `EVENT_STRIDE_U32` in `engine::gpu::event_ring`). Word 10
//!     is the seq trailer added by the sort-then-fold pass.
//!   - Header layout: word 0 = kind tag, word 1 = tick.

use engine::ability::apply::ApplyEvent;

/// Per-record stride in u32-words.
///
/// Words 0..9 are the same 10-word header + payload layout used before
/// P11; word 10 is the new seq trailer added in P11 sort-then-fold.
/// The GPU `apply_ability` dispatcher still uses stride 10 for its
/// hardcoded `_slot * 10u + N` writes (separate code path). This
/// constant covers the CPU reference for `@replayable @gpu_amenable`
/// events emitted via `emit Xxx {}` in DSL physics rules — those use
/// stride 11 with seq at the last word.
pub const CHRONICLE_RECORD_STRIDE_U32: usize = 11;

/// Translate one [`ApplyEvent`] into the 11-word chronicle record the CPU
/// cascade dispatch writes for it, or `None` if the variant has no
/// chronicle counterpart in the engine's `EventKindId` enum today.
///
/// Record layout:
///   - `[0]`  = kind tag (e.g. `26` for EffectDamageApplied)
///   - `[1]`  = tick
///   - `[2..6]` = per-variant payload (see variant arms below)
///   - `[6]`  = ability_id (Gap detective#6, 2026-05-12)
///   - `[7..9]` = reserved (zero)
///   - `[10]` = seq trailer: `(kernel_id << 24) | (thread_id << 4) | emit_idx`
///
/// `caster_id` and `target_id` are written into the actor (slot 2)
/// and target (slot 3) chronicle payload words respectively. Both
/// are u32 raw AgentIds. For self-cast callers (the slice-γ
/// default), pass `target_id == caster_id`.
///
/// `ability_id` is the registry id of the ability dispatched by the
/// `apply_ability` call site. Written at slot 6 on every arm so
/// downstream consumers can discriminate verb source.
///
/// `intra_emit_idx` is the per-emit counter within the same handler
/// invocation (start at 0, increment per successive emit). On the CPU
/// side, `kernel_id` and `thread_id` are both 0 (no dense kernel table
/// today); seq = `intra_emit_idx & 0xF`.
pub fn apply_event_to_chronicle_record(
    event: ApplyEvent,
    tick: u32,
    caster_id: u32,
    target_id: u32,
    ability_id: u32,
    // Per-emit counter within the same handler invocation (start at 0,
    // increment per successive emit call). CPU kernel_id = 0 placeholder.
    intra_emit_idx: u32,
) -> Option<[u32; CHRONICLE_RECORD_STRIDE_U32]> {
    let mut rec = [0u32; CHRONICLE_RECORD_STRIDE_U32];
    rec[1] = tick;
    // Slot 6 = ability_id (Gap detective#6). Always written, regardless
    // of variant — mirrors the GPU dispatcher's
    // `atomicStore(&event_ring[_slot * 10u + 6u], ability_id__u32)`
    // suffix on every arm.
    rec[6] = ability_id;
    // Slot 10 = seq trailer. Matches the GPU emitter's last-word write:
    //   `(kernel_id << 24) | (thread_id << 4) | emit_idx`
    // CPU kernel_id and thread_id are 0 (no dense kernel table today).
    rec[CHRONICLE_RECORD_STRIDE_U32 - 1] = crate::seq::compute_event_seq(0, 0, intra_emit_idx);
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
        // --- DamageOverTime = 20 → EventKindId::EffectDamageOverTimeApplied = 51.
        // Wave 1.5+ — multi-tick effect. 4-payload-word chronicle record:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot
        //   slot 4 = bitcast<u32>(amount-per-tick) — scale_bonus already
        //            folded into amount by `apply_program` (see
        //            crates/engine/src/ability/apply.rs:312-313)
        //   slot 5 = duration_ticks (raw u32)
        // The cast records the magnitude + window once; a future
        // consumer rule will re-emit per-tick damage events.
        ApplyEvent::DamageOverTime { source: _, target: _, amount, duration_ticks } => {
            rec[0] = 51;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = amount.to_bits();
            rec[5] = duration_ticks;
            Some(rec)
        }
        // --- HealOverTime = 21 → EventKindId::EffectHealOverTimeApplied = 52.
        // Wave 1.5+ — same shape as DamageOverTime (per-tick healing).
        ApplyEvent::HealOverTime { source: _, target: _, amount, duration_ticks } => {
            rec[0] = 52;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = amount.to_bits();
            rec[5] = duration_ticks;
            Some(rec)
        }
        // --- TimedShield = 22 → EventKindId::EffectTimedShieldApplied = 53.
        // Wave 1.5+ — same payload shape as DoT/HoT but `amount` is the
        // one-shot shield magnitude (not per-tick), persisting for
        // `duration_ticks`. scale_bonus is folded into amount by
        // `apply_program` (see crates/engine/src/ability/apply.rs:316-317).
        ApplyEvent::TimedShield { source: _, target: _, amount, duration_ticks } => {
            rec[0] = 53;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = amount.to_bits();
            rec[5] = duration_ticks;
            Some(rec)
        }
        // --- Stealth = 27 → EventKindId::EffectStealthApplied = 54.
        // Extended-corpus status — caster-self stealth. The engine event
        // has no target field; the GPU dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = duration_ticks (raw u32, NOT bitcast — duration is
        //            already u32 on the EffectOp side, no widening)
        // Same slot layout as Dash/Blink (caster-self motion), but
        // payload word 1 carries duration rather than bitcast f32 distance.
        ApplyEvent::Stealth { source: _, duration_ticks } => {
            rec[0] = 54;
            rec[2] = caster_id;
            rec[3] = duration_ticks;
            Some(rec)
        }
        // --- Charm = 28 → EventKindId::EffectCharmApplied = 55.
        // Extended-corpus status — target-cast charm. 3-payload-word
        // chronicle record (actor + target + duration_ticks). Distinct
        // from Stun/Root/Silence/Fear/Taunt (kinds 29/43..46) which fold
        // the deadline at dispatch time as `expires_at_tick = tick +
        // duration_ticks` — the extended-status family stores the raw
        // duration so a future consumer rule can compute its own window.
        ApplyEvent::Charm { target: _, duration_ticks } => {
            rec[0] = 55;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = duration_ticks;
            Some(rec)
        }
        // --- Grounded = 29 → EventKindId::EffectGroundedApplied = 56.
        // Extended-corpus status — same shape as Charm (target-cast).
        ApplyEvent::Grounded { target: _, duration_ticks } => {
            rec[0] = 56;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = duration_ticks;
            Some(rec)
        }
        // --- Suppress = 30 → EventKindId::EffectSuppressApplied = 57.
        // Extended-corpus status — same shape as Charm/Grounded.
        ApplyEvent::Suppress { target: _, duration_ticks } => {
            rec[0] = 57;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = duration_ticks;
            Some(rec)
        }
        // --- Buff = 23 → EventKindId::EffectBuffApplied = 58.
        // Slice γ tail — target-cast with packed payload. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot
        //   slot 4 = raw payload_a (= stat_ordinal in low byte |
        //            magnitude_q8 in bits 8..; consumer sign-extends)
        //   slot 5 = raw payload_b (= duration_ticks)
        // The CPU reference reconstructs the same packed payload from
        // the typed `ApplyEvent::Buff` fields so the bytes round-trip
        // (stat ordinal in low byte, magnitude_q8 sign-cast i16 → i32
        // → u32 then shifted left by 8, masked with stat). Mirrors
        // `pack_effect`'s Buff arm in
        // `crates/engine/src/ability/packed.rs`.
        ApplyEvent::Buff { target: _, stat, magnitude_q8, duration_ticks } => {
            rec[0] = 58;
            rec[2] = caster_id;
            rec[3] = target_id;
            // Reconstruct the dispatcher's payload_a packing: low byte =
            // stat ordinal (u8), bits 8.. = magnitude_q8 (i16 sign-cast
            // i32 → u32 then shifted left 8). The OR-mask isolates each
            // half — magnitude_q8's low 8 bits are positioned bits 8..15,
            // and the stat fits in 0..7.
            let pa = (stat as u32) | ((magnitude_q8 as i32 as u32) << 8);
            rec[4] = pa;
            rec[5] = duration_ticks;
            Some(rec)
        }
        // --- Harvest = 25 → EventKindId::EffectHarvestApplied = 59.
        // Slice γ tail — caster-self resource gather. The GPU dispatcher
        // writes:
        //   slot 2 = caster_slot
        //   slot 3 = kind_hash (u32 FxHash of resource ident)
        //   slot 4 = amount (u32, widened from u16 EffectOp side)
        // No target field on the engine event.
        ApplyEvent::Harvest { source: _, kind_hash, amount } => {
            rec[0] = 59;
            rec[2] = caster_id;
            rec[3] = kind_hash;
            rec[4] = amount as u32;
            Some(rec)
        }
        // --- PlaceVoxel = 26 → EventKindId::EffectPlaceVoxelApplied = 60.
        // Slice γ tail — caster-self voxel placement. The GPU dispatcher
        // writes:
        //   slot 2 = caster_slot
        //   slot 3 = kind_hash (u32 FxHash of voxel kind ident)
        // Position is implicit from the cast's target world position
        // (not stored in the chronicle record). No target field on the
        // engine event.
        ApplyEvent::PlaceVoxel { source: _, kind_hash } => {
            rec[0] = 60;
            rec[2] = caster_id;
            rec[3] = kind_hash;
            Some(rec)
        }
        // --- Reflect = 31 → EventKindId::EffectReflectApplied = 61.
        // Slice γ tail — target-cast fraction-of-damage bounce. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot
        //   slot 4 = duration_ticks (raw u32 = payload_a)
        //   slot 5 = fraction_q8 packed in payload_b's low 16 bits;
        //            consumer sign-extends. The CPU reference mirrors
        //            `pack_effect`'s Reflect arm: zero-extend i16 → u16
        //            then to u32 so the byte layout matches the GPU
        //            packing exactly.
        // Same payload-shape family as Slow/LifeSteal/DamageModify (all
        // carry duration + signed q8). Distinct in that we store raw
        // `duration_ticks` (not `expires_at_tick`), consistent with the
        // multi-tick effect family (DoT/HoT/TimedShield, kinds 51..53).
        ApplyEvent::Reflect { target: _, duration_ticks, fraction_q8 } => {
            rec[0] = 61;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = duration_ticks;
            // Mirror `pack_effect`: `(fraction_q8 as u16) as u32`. The
            // u16 cast wraps negatives via two's complement; the u32
            // widen zero-extends. The dispatcher writes payload_b raw,
            // so the chronicle record carries the same low-16-bit form.
            rec[5] = (fraction_q8 as u16) as u32;
            Some(rec)
        }
        // --- Summon = 24 → EventKindId::EffectSummonApplied = 62.
        // Slice γ closer — caster-self with packed payload. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = template_hash (u32 = payload_a)
        //   slot 4 = count (u32, widened from u8 via `(payload_b >> 24) & 0xFF`)
        //   slot 5 = lifetime_ticks (u32 = `payload_b & 0x00FFFFFF`)
        // The CPU side writes ONE `ApplyEvent::Summon` per cast (per
        // `engine::ability::apply::apply_program`); downstream N-entity
        // spawning is a separate consumer concern. No target field on
        // the engine event. The dispatcher splits count and lifetime
        // into distinct ring slots so consumers don't have to redo
        // the bit-unpack on read — the engine event struct carries
        // `count: u8` and `lifetime_ticks: u32` as separate fields.
        ApplyEvent::Summon { source: _, template_hash, count, lifetime_ticks } => {
            rec[0] = 62;
            rec[2] = caster_id;
            rec[3] = template_hash;
            rec[4] = count as u32;
            rec[5] = lifetime_ticks;
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
        // --- PlantBelief = 32 → EventKindId::EffectPlantBeliefApplied = 63.
        // Wave 3 ToM Phase 1 bit-flag belief primitive. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot   (the belief's HOLDER agent)
        //   slot 4 = subject_idx   (= payload_a; agent slot the
        //                            belief is ABOUT)
        //   slot 5 = fact_bit_mask (= payload_b = `1u << fact_bit`,
        //                            pre-shifted at pack time so the
        //                            downstream view's `self |= b`
        //                            body doesn't re-shift)
        // The CPU oracle mirrors the same record shape: subject_idx
        // and fact_bit_mask land at slots 4 and 5 respectively, with
        // the bit-shift applied here to match the GPU `pack_effect`
        // arm (`1u32 << fact_bit as u32`).
        ApplyEvent::PlantBelief { source: _, target: _, subject_idx, fact_bit } => {
            rec[0] = 63;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = subject_idx;
            rec[5] = 1u32 << (fact_bit as u32);
            Some(rec)
        }
        // --- Observe = 33 → EventKindId::EffectObserveApplied = 64.
        // Wave 3 ToM Phase 3 self-observe-target verb. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot      (the OBSERVER)
        //   slot 3 = target_slot      (the OBSERVED)
        //   slot 4 = target_observer  (= payload_a; u8 widened to u32 —
        //                                future-extension hook for non-
        //                                self observe shapes; today only
        //                                `0` (self) is wired)
        //   slot 5 = 0                (unused — payload_b is 0)
        // The CPU oracle mirrors the same record shape: target_observer
        // lands at slot 4, slot 5 is left zero. No payload words for
        // pos / creature_type — the consumer reads them from the
        // agent SoA at consume tick.
        ApplyEvent::Observe { source: _, target: _, target_observer } => {
            rec[0] = 64;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = target_observer as u32;
            Some(rec)
        }
        // --- Scry = 34 → EventKindId::EffectScryApplied = 65.
        // Wave 3 ToM Phase 3.5 cross-observer access verb. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot       (the OBSERVER reading C's eyes)
        //   slot 3 = target_slot       (= subject_idx; the agent the
        //                                belief is ABOUT)
        //   slot 4 = target_observer   (= payload_a; u8 widened to u32
        //                                — the agent slot whose beliefs
        //                                caster reads. With `0` (self)
        //                                this collapses to the observe
        //                                shape — no behaviour difference)
        //   slot 5 = subject_idx       (= payload_b; u32 — same as
        //                                target_slot; redundant on the
        //                                wire but kept for arm-symmetry
        //                                with PlantBelief and downstream
        //                                consumer convenience)
        ApplyEvent::Scry { source: _, target: _, target_observer, subject_idx } => {
            rec[0] = 65;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = target_observer as u32;
            rec[5] = subject_idx;
            Some(rec)
        }
        // --- Reveal = 35 → EventKindId::EffectRevealApplied = 66.
        // Wave 3 ToM Phase 3.5 one-to-many propagation verb. The GPU
        // dispatcher writes:
        //   slot 2 = caster_slot       (the BROADCASTER)
        //   slot 3 = target_slot       (= subject_idx; the agent the
        //                                broadcast is ABOUT)
        //   slot 4 = subject_idx       (= payload_a; u32 — same as
        //                                target_slot; redundant on the
        //                                wire but kept for arm-symmetry)
        //   slot 5 = 0                 (unused — payload_b is 0; the
        //                                fan-out target set is "all
        //                                observers" at consume time)
        ApplyEvent::Reveal { source: _, target: _, subject_idx } => {
            rec[0] = 66;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = subject_idx;
            Some(rec)
        }
        // --- Disguise = 36 → EventKindId::EffectDisguiseApplied = 67.
        // Wave 3 ToM Phase 4 deception verb. payload_a packs
        // (duration_ticks << 8) | fake_type so the consumer can split
        // with `payload_a & 0xFF` (fake_type) and `payload_a >> 8`
        // (duration). payload_b = 0. The dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot       (= caster for self-cast)
        //   slot 4 = (duration<<8 | fake_type)
        //   slot 5 = 0
        ApplyEvent::Disguise { source: _, fake_type, duration_ticks } => {
            rec[0] = 67;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = ((duration_ticks << 8) & 0xFFFFFF00u32) | (fake_type as u32);
            Some(rec)
        }
        // --- Decoy = 37 → EventKindId::EffectDecoyApplied = 68.
        // Wave 3 ToM Phase 4 deception verb. payload_a = subject_idx
        // (the agent slot the belief is ABOUT). payload_b = pre-packed
        // (x_q8, y_q8, z_q8, fake_type) quartet. The dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot       (the OBSERVER whose row caster
        //                                writes — not the subject)
        //   slot 4 = subject_idx
        //   slot 5 = fake_pos          (packed quartet)
        ApplyEvent::Decoy { source: _, target: _, subject_idx, fake_pos } => {
            rec[0] = 68;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = subject_idx;
            rec[5] = fake_pos;
            Some(rec)
        }
        // --- EraseBelief = 38 → EventKindId::EffectEraseBeliefApplied = 69.
        // Wave 3 ToM Phase 4 deception verb. payload_a = subject_idx.
        // payload_b's low byte = fields bitset. The dispatcher writes:
        //   slot 2 = caster_slot
        //   slot 3 = target_slot       (the OBSERVER whose row caster
        //                                clears — not the subject)
        //   slot 4 = subject_idx
        //   slot 5 = fields            (low byte = bit 0 pos … bit 5 flags)
        ApplyEvent::EraseBelief { source: _, target: _, subject_idx, fields } => {
            rec[0] = 69;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = subject_idx;
            rec[5] = fields as u32;
            Some(rec)
        }
        // --- TravelTo = 39 → EventKindId::EffectTravelToApplied = 70.
        // Lift A multi-tick travel verb. payload_a packs
        // (dest_y_q8 << 16) | (dest_x_q8 & 0xFFFF). payload_b =
        // eta_ticks. The dispatcher writes:
        //   slot 2 = caster_slot (the traveler)
        //   slot 3 = caster_slot (target = caster for self-cast)
        //   slot 4 = packed (dest_y_q8 hi << 16 | dest_x_q8 lo & 0xFFFF)
        //   slot 5 = eta_ticks
        ApplyEvent::TravelTo { source: _, dest_x, dest_y, eta_ticks } => {
            // Re-pack f32 → i16 q8 → packed u32 the same way `pack_effect`
            // does in engine/src/ability/packed.rs. Round-half-to-even via
            // `.round()` then clamp into i16 range so a malformed cast
            // can't overflow the SoA cell.
            let dx = (dest_x * 256.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let dy = (dest_y * 256.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let lo = (dx as u16) as u32;
            let hi = ((dy as u16) as u32) << 16;
            rec[0] = 70;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = hi | lo;
            rec[5] = eta_ticks;
            Some(rec)
        }
        // --- Recipe = 40 → EventKindId::EffectRecipeApplied = 71.
        // Lift B production verb. payload_a packs
        // (target_tool << 16) | recipe_id. payload_b = 0. The dispatcher
        // writes:
        //   slot 2 = caster_slot (the producer)
        //   slot 3 = caster_slot (target = caster — recipes act on the
        //                          caster's inventory)
        //   slot 4 = packed (target_tool hi << 16 | recipe_id lo &
        //                    0xFFFF)
        //   slot 5 = 0
        ApplyEvent::Recipe { source: _, recipe_id, target_tool } => {
            let packed = (recipe_id as u32) | ((target_tool as u32) << 16);
            rec[0] = 71;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = packed;
            rec[5] = 0;
            Some(rec)
        }
        // --- WearTool = 41 → EventKindId::EffectWearToolApplied = 72.
        // Lift B capital-goods wear. payload_a packs
        // (amount << 8) | tool_kind. payload_b = 0. The dispatcher
        // writes:
        //   slot 2 = caster_slot (the tool's owner)
        //   slot 3 = caster_slot (target = caster — wear acts on the
        //                          caster's owned tool)
        //   slot 4 = packed (amount hi << 8 | tool_kind lo & 0xFF)
        //   slot 5 = 0
        ApplyEvent::WearTool { source: _, tool_kind, amount } => {
            let packed = (tool_kind as u32) | ((amount as u32) << 8);
            rec[0] = 72;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = packed;
            rec[5] = 0;
            Some(rec)
        }
        // Lift C — Propose. Chronicle kind 73. payload_a = contract_kind
        // (low byte). payload_b = expires_at_tick (0 = no expiry).
        // target = the agent the proposal is offered to.
        //   slot 2 = caster_slot (proposer)
        //   slot 3 = target_slot (recipient)
        //   slot 4 = contract_kind (low byte; high bits zero)
        //   slot 5 = expires_at_tick
        ApplyEvent::Propose { source: _, target: _, contract_kind, expires_at_tick } => {
            rec[0] = 73;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = contract_kind as u32;
            rec[5] = expires_at_tick;
            Some(rec)
        }
        // Lift C — Announce. Chronicle kind 74. payload_a packs
        // (announcement_kind | radius_q8 << 8). payload_b = 0. Self-
        // origin: target = caster.
        //   slot 2 = caster_slot (announcer / origin cell)
        //   slot 3 = caster_slot (target = caster — announcements
        //                          radiate from the caster's cell)
        //   slot 4 = packed (radius_q8 << 8 | announcement_kind)
        //   slot 5 = 0
        ApplyEvent::Announce { source: _, announcement_kind, radius_q8 } => {
            let packed = (announcement_kind as u32) | ((radius_q8 as u32) << 8);
            rec[0] = 74;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = packed;
            rec[5] = 0;
            Some(rec)
        }
        // Lift D — GainSkill. Chronicle kind 75. payload_a packs
        // (skill_id | amount_q8 << 8). payload_b = 0. Self-cast:
        // target = caster.
        //   slot 2 = caster_slot (the agent gaining skill)
        //   slot 3 = caster_slot (target = caster — skill grows on
        //                          the caster's per-agent SoA cell)
        //   slot 4 = packed (amount_q8 << 8 | skill_id)
        //   slot 5 = 0
        ApplyEvent::GainSkill { source: _, skill_id, amount_q8 } => {
            let packed = (skill_id as u32) | ((amount_q8 as u32) << 8);
            rec[0] = 75;
            rec[2] = caster_id;
            rec[3] = caster_id;
            rec[4] = packed;
            rec[5] = 0;
            Some(rec)
        }
        // Lift D — CreateObligation. Chronicle kind 76. payload_a
        // packs (obligation_id | kind << 16). payload_b = 0.
        //   slot 2 = caster_slot (creditor / claimant)
        //   slot 3 = target_slot (debtor / promisor)
        //   slot 4 = packed (kind << 16 | obligation_id)
        //   slot 5 = 0
        ApplyEvent::CreateObligation { source: _, target: _, obligation_id, kind } => {
            let packed = (obligation_id as u32) | ((kind as u32) << 16);
            rec[0] = 76;
            rec[2] = caster_id;
            rec[3] = target_id;
            rec[4] = packed;
            rec[5] = 0;
            Some(rec)
        }
        // Plan G (2026-05-09) — CastBegin = 46 → EventKindId::EffectCastBeginApplied = 77.
        // payload_a packs ability_id (low 16 bits) + duration_ticks
        // (high 16 bits). payload_b packs the q8 target position
        // (target_x_q8 low, target_y_q8 high). chronicle target_slot
        // is the runtime resolved target. Mirrors the GPU dispatcher
        // arm in crates/dsl_compiler/src/cg/emit/wgsl_body.rs
        // (kind == 46u arm in emit_chronicle_arm_chain).
        ApplyEvent::CastBegin { source: _, ability_id, duration_ticks, target_slot, target_x_q8, target_y_q8 } => {
            let payload_a = (ability_id as u32) | ((duration_ticks as u32) << 16);
            let payload_b = ((target_x_q8 as u16) as u32) | (((target_y_q8 as u16) as u32) << 16);
            rec[0] = 77;
            rec[2] = caster_id;
            rec[3] = target_slot;
            rec[4] = payload_a;
            rec[5] = payload_b;
            Some(rec)
        }
        // After the slice γ closer (Summon → kind 62), every
        // `ApplyEvent` variant has a chronicle counterpart — no
        // fallback `_ => None` arm needed. The closed-set match
        // also serves as a compile-time guarantee: when a future
        // engine slice adds a new `ApplyEvent` variant, the
        // unreachable-arm warning forces a deliberate decision
        // about whether the new variant should land in the
        // chronicle (add an arm) or be skipped (add `_ => None`
        // back, with a comment explaining why).
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
            /*caster_id*/ 7, /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Damage has chronicle counterpart");
        assert_eq!(rec[0], 26, "kind tag — EffectDamageApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "caster slot — slice γ self-cast (was source=7)");
        assert_eq!(rec[3], 7, "target slot — slice γ self-cast (was target=11)");
        assert_eq!(rec[4], 42.0_f32.to_bits(), "amount as bitcast<u32>");
        // Tail words zeroed.
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }
    }

    #[test]
    fn heal_chronicle_record_uses_kind_27() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Heal { source: aid(1), target: aid(2), amount: 12.5 },
            /*tick*/ 50,
            /*caster_id*/ 3, /*target_id*/ 3, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
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
            /*caster_id*/ 4, /*target_id*/ 4, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
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
            /*caster_id*/ 8, /*target_id*/ 8, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
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
            /*caster_id*/ 3, /*target_id*/ 3, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
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
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .unwrap();
        assert_eq!(rec[2], 7, "actor slot uses caster_id");
        assert_eq!(rec[3], 11, "target slot uses target_id (distinct from caster)");
        // Self-cast call site preserves slice-γ behavior: caster=target.
        let rec_self = apply_event_to_chronicle_record(
            ApplyEvent::Damage { source: aid(1), target: aid(2), amount: 100.0 },
            50, 7, 7, 1, 0,
        ).unwrap();
        assert_eq!(rec_self[2], rec_self[3], "self-cast collapses both slots");
    }

    #[test]
    fn every_apply_event_variant_has_chronicle_counterpart() {
        // After the slice γ closer (Summon → kind 62), every
        // `ApplyEvent` variant emitted by `apply_program` has a
        // chronicle counterpart in the GPU dispatcher. The earlier
        // "Summon multi-spawn semantics need a new dispatch shape"
        // deferral was misleading — per
        // `crates/engine/src/ability/apply.rs`, the CPU side writes
        // ONE `ApplyEvent::Summon` per cast carrying packed (count,
        // lifetime); downstream N-entity spawning is a separate
        // consumer concern, distinct from the dispatcher's
        // single-record chronicle write.
        //
        // Wire-up index per variant family:
        // - SelfDamage (Bleed verb swap, Task #138 follow-on,
        //   2026-05-06) → kind=39, see `self_damage_chronicle_record_uses_kind_39`.
        // - Execute (Reap verb swap, Task #138 follow-on, mirror of
        //   Fortify) → kind=42, see `execute_chronicle_record_uses_kind_42`.
        // - Root/Silence/Fear/Taunt (Wave 2 piece 1) →
        //   kinds 43..46, see `control_status_chronicle_records_use_kinds_43_46`.
        // - Dash/Blink/Knockback/Pull (Wave 2 piece 2) →
        //   kinds 47..50, see `movement_chronicle_records_use_kinds_47_50`.
        // - DamageOverTime/HealOverTime/TimedShield (Wave 1.5+) →
        //   kinds 51..53, see `multi_tick_chronicle_records_use_kinds_51_53`.
        // - Stealth/Charm/Grounded/Suppress (extended-status slice) →
        //   kinds 54..57, see
        //   `extended_status_chronicle_records_use_kinds_54_57`.
        // - Buff/Harvest/PlaceVoxel/Reflect (slice γ tail) →
        //   kinds 58..61, see
        //   `slice_gamma_tail_chronicle_records_use_kinds_58_61`.
        // - Summon (slice γ closer) → kind 62, see
        //   `summon_chronicle_record_uses_kind_62`.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Summon  { source: aid(1), template_hash: 0xDEADBEEF, count: 2, lifetime_ticks: 100 },
            /*tick*/ 50,
            /*caster_id*/ 1,
            /*target_id*/ 1, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        );
        assert!(
            rec.is_some(),
            "Summon should now have chronicle counterpart (slice γ closer)"
        );
    }

    /// Slice γ closer — Summon (kind 24 → 62). Caster-self with
    /// packed payload. 5-payload-word record: actor + template_hash
    /// + count (u8 widened to u32) + lifetime_ticks. The dispatcher
    /// splits the packed `payload_b` into distinct ring slots so
    /// consumers don't have to redo the bit-unpack on read; the
    /// engine event struct carries `count: u8` and `lifetime_ticks:
    /// u32` as separate fields.
    #[test]
    fn summon_chronicle_record_uses_kind_62() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Summon {
                source: aid(7),
                template_hash: 0xDEADBEEF,
                count: 3,
                lifetime_ticks: 120,
            },
            /*tick*/ 100,
            /*caster_id*/ 7,
            /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Summon has chronicle counterpart");
        assert_eq!(rec[0], 62, "Summon: kind tag — EffectSummonApplied");
        assert_eq!(rec[1], 100, "Summon: tick");
        assert_eq!(rec[2], 7, "Summon: actor slot — caster_id");
        assert_eq!(rec[3], 0xDEADBEEF, "Summon: template_hash at payload word 1");
        assert_eq!(rec[4], 3, "Summon: count (u8 widened to u32) at payload word 2");
        assert_eq!(rec[5], 120, "Summon: lifetime_ticks (raw u32) at payload word 3");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Summon: tail word {i} should be zero");
        }
    }

    /// Slice γ tail (Buff/Harvest/PlaceVoxel/Reflect). Four distinct
    /// shapes:
    ///   - Buff: target-cast with packed payload. 5-payload-word record:
    ///     actor + target + raw payload_a (stat | mag_q8 << 8) + raw
    ///     payload_b (= duration). Consumers sign-extend magnitude_q8.
    ///   - Harvest: caster-self. 4-payload-word record: actor + kind_hash
    ///     + amount. No target field on the engine event.
    ///   - PlaceVoxel: caster-self. 3-payload-word record: actor + kind_hash.
    ///     Position is implicit from cast's target position.
    ///   - Reflect: target-cast with packed payload. 5-payload-word
    ///     record: actor + target + raw payload_a (= duration) + raw
    ///     payload_b (low 16 bits = fraction_q8 i16). Consumers
    ///     sign-extend the fraction.
    ///
    /// Pin per-variant kind tags (58..61) and packed-payload byte layout
    /// — the signed sub-fields (Buff's magnitude_q8 and Reflect's
    /// fraction_q8) are exercised with negative values to guarantee the
    /// sign-cast path round-trips.
    #[test]
    fn slice_gamma_tail_chronicle_records_use_kinds_58_61() {
        use engine::ability::program::BuffStat;

        // Buff — target-cast with packed payload. Negative magnitude_q8
        // exercises the i16 → i32 → u32 sign-cast path.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Buff {
                target: aid(11),
                stat: BuffStat::AttackSpeed, // ordinal 1
                magnitude_q8: -64,           // negative — sign extends
                duration_ticks: 50,
            },
            /*tick*/ 100,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Buff has chronicle counterpart");
        assert_eq!(rec[0], 58, "Buff: kind tag — EffectBuffApplied");
        assert_eq!(rec[1], 100, "Buff: tick");
        assert_eq!(rec[2], 7,   "Buff: actor slot — caster_id");
        assert_eq!(rec[3], 11,  "Buff: target slot — target_id");
        // Reconstruct the expected packed payload_a:
        //   low byte = stat ordinal (1 = AttackSpeed)
        //   bits 8..  = magnitude_q8 sign-cast i16 → i32 → u32 then << 8
        let expected_pa = 1u32 | ((-64_i32 as u32) << 8);
        assert_eq!(
            rec[4], expected_pa,
            "Buff: payload_a packs stat (low byte) + magnitude_q8 (bits 8..); \
             negative magnitude must sign-extend i16 → i32 before shift"
        );
        assert_eq!(rec[5], 50, "Buff: payload_b = duration_ticks (raw u32)");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Buff: tail word {i} should be zero");
        }

        // Harvest — caster-self, 4-payload-word record.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Harvest { source: aid(7), kind_hash: 0xCAFEBABE, amount: 5 },
            /*tick*/ 200,
            /*caster_id*/ 7,
            /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Harvest has chronicle counterpart");
        assert_eq!(rec[0], 59, "Harvest: kind tag — EffectHarvestApplied");
        assert_eq!(rec[1], 200, "Harvest: tick");
        assert_eq!(rec[2], 7, "Harvest: actor slot — caster_id");
        assert_eq!(rec[3], 0xCAFEBABE, "Harvest: kind_hash at payload word 1");
        assert_eq!(rec[4], 5, "Harvest: amount at payload word 2 (u16 widened to u32)");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Harvest: tail word {i} should be zero");
        }

        // PlaceVoxel — caster-self, 3-payload-word record.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::PlaceVoxel { source: aid(7), kind_hash: 0xFACEFEED },
            /*tick*/ 300,
            /*caster_id*/ 7,
            /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("PlaceVoxel has chronicle counterpart");
        assert_eq!(rec[0], 60, "PlaceVoxel: kind tag — EffectPlaceVoxelApplied");
        assert_eq!(rec[1], 300, "PlaceVoxel: tick");
        assert_eq!(rec[2], 7, "PlaceVoxel: actor slot — caster_id");
        assert_eq!(rec[3], 0xFACEFEED, "PlaceVoxel: kind_hash at payload word 1");
        for i in 4..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
            assert_eq!(rec[i], 0, "PlaceVoxel: tail word {i} should be zero");
        }

        // Reflect — target-cast, packed payload_b. Negative fraction_q8
        // exercises the i16 → u16 zero-extend path: `(fraction_q8 as u16)
        // as u32` should produce 0x0000_FFC0 for fraction_q8 = -64.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Reflect { target: aid(11), duration_ticks: 50, fraction_q8: -64 },
            /*tick*/ 400,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Reflect has chronicle counterpart");
        assert_eq!(rec[0], 61, "Reflect: kind tag — EffectReflectApplied");
        assert_eq!(rec[1], 400, "Reflect: tick");
        assert_eq!(rec[2], 7, "Reflect: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Reflect: target slot — target_id");
        assert_eq!(rec[4], 50, "Reflect: payload_a = duration_ticks (raw u32)");
        // Match `pack_effect`'s Reflect arm: `(fraction_q8 as u16) as u32`.
        // For -64_i16: u16 wraps to 0xFFC0; u32 zero-extend → 0x0000_FFC0.
        let expected_pb = (-64_i16 as u16) as u32;
        assert_eq!(expected_pb, 0x0000_FFC0, "sanity: -64 → 0xFFC0 in low 16 bits");
        assert_eq!(
            rec[5], expected_pb,
            "Reflect: payload_b's low 16 bits carry fraction_q8 (i16 → u16 → u32, \
             zero-extend); consumer sign-extends low 16 to recover negative value"
        );
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Reflect: tail word {i} should be zero");
        }
    }

    /// Extended-corpus statuses (Stealth/Charm/Grounded/Suppress).
    /// Two distinct shapes:
    ///   - Stealth: caster-self status. Engine event has no target
    ///     field; chronicle record stores duration_ticks at payload
    ///     word 1 (= ring slot offset 3) — same family as Dash/Blink.
    ///   - Charm/Grounded/Suppress: target-cast statuses. 3-payload-word
    ///     record: actor + target + duration_ticks at ring slot offset 4.
    /// Distinct from Stun/Root/Silence/Fear/Taunt: the extended-status
    /// arms store raw `duration_ticks` (not `expires_at_tick`),
    /// consistent with the multi-tick effect family (DoT/HoT/TimedShield).
    /// Pin per-variant kind tags (54..57) and per-shape duration offsets.
    #[test]
    fn extended_status_chronicle_records_use_kinds_54_57() {
        // Stealth — caster-self status (no target in engine event).
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Stealth { source: aid(7), duration_ticks: 50 },
            /*tick*/ 100,
            /*caster_id*/ 7,
            /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Stealth has chronicle counterpart");
        assert_eq!(rec[0], 54, "Stealth: kind tag — EffectStealthApplied");
        assert_eq!(rec[1], 100, "Stealth: tick");
        assert_eq!(rec[2], 7, "Stealth: actor slot — caster_id");
        assert_eq!(rec[3], 50, "Stealth: duration_ticks at payload word 1 (raw u32)");
        for i in 4..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
            assert_eq!(rec[i], 0, "Stealth: tail word {i} should be zero");
        }

        // Charm — target-cast status (3 payload words).
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Charm { target: aid(11), duration_ticks: 30 },
            /*tick*/ 200,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Charm has chronicle counterpart");
        assert_eq!(rec[0], 55, "Charm: kind tag — EffectCharmApplied");
        assert_eq!(rec[1], 200, "Charm: tick");
        assert_eq!(rec[2], 7, "Charm: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Charm: target slot — target_id");
        assert_eq!(rec[4], 30, "Charm: duration_ticks at payload word 2 (raw u32)");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Charm: tail word {i} should be zero");
        }

        // Grounded — same shape as Charm.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Grounded { target: aid(11), duration_ticks: 25 },
            /*tick*/ 300,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Grounded has chronicle counterpart");
        assert_eq!(rec[0], 56, "Grounded: kind tag — EffectGroundedApplied");
        assert_eq!(rec[1], 300, "Grounded: tick");
        assert_eq!(rec[2], 7, "Grounded: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Grounded: target slot — target_id");
        assert_eq!(rec[4], 25, "Grounded: duration_ticks at payload word 2");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Grounded: tail word {i} should be zero");
        }

        // Suppress — same shape as Charm/Grounded.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Suppress { target: aid(11), duration_ticks: 40 },
            /*tick*/ 400,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Suppress has chronicle counterpart");
        assert_eq!(rec[0], 57, "Suppress: kind tag — EffectSuppressApplied");
        assert_eq!(rec[1], 400, "Suppress: tick");
        assert_eq!(rec[2], 7, "Suppress: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Suppress: target slot — target_id");
        assert_eq!(rec[4], 40, "Suppress: duration_ticks at payload word 2");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Suppress: tail word {i} should be zero");
        }
    }

    /// Wave 1.5+ — multi-tick effects (DamageOverTime/HealOverTime/
    /// TimedShield). All three share the same 5-payload-word shape:
    /// actor + target + amount (bitcast f32 → u32 at slot 4) +
    /// duration_ticks (raw u32 at slot 5). Pin per-variant kind tags
    /// (51..53) and confirm the duration round-trips correctly.
    #[test]
    fn multi_tick_chronicle_records_use_kinds_51_53() {
        // DamageOverTime — 4-payload-word record.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::DamageOverTime {
                source: aid(7),
                target: aid(11),
                amount: 6.5,
                duration_ticks: 30,
            },
            /*tick*/ 100,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("DamageOverTime has chronicle counterpart");
        assert_eq!(rec[0], 51, "DamageOverTime: kind tag — EffectDamageOverTimeApplied");
        assert_eq!(rec[1], 100, "DamageOverTime: tick");
        assert_eq!(rec[2], 7, "DamageOverTime: actor slot — caster_id");
        assert_eq!(rec[3], 11, "DamageOverTime: target slot — target_id");
        assert_eq!(rec[4], 6.5_f32.to_bits(), "DamageOverTime: amount-per-tick at payload word 2");
        assert_eq!(rec[5], 30, "DamageOverTime: duration_ticks at payload word 3 (raw u32)");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "DamageOverTime: tail word {i} should be zero");
        }

        // HealOverTime — same shape as DoT.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::HealOverTime {
                source: aid(7),
                target: aid(11),
                amount: 4.0,
                duration_ticks: 50,
            },
            /*tick*/ 200,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("HealOverTime has chronicle counterpart");
        assert_eq!(rec[0], 52, "HealOverTime: kind tag — EffectHealOverTimeApplied");
        assert_eq!(rec[1], 200, "HealOverTime: tick");
        assert_eq!(rec[2], 7, "HealOverTime: actor slot — caster_id");
        assert_eq!(rec[3], 11, "HealOverTime: target slot — target_id");
        assert_eq!(rec[4], 4.0_f32.to_bits(), "HealOverTime: amount-per-tick at payload word 2");
        assert_eq!(rec[5], 50, "HealOverTime: duration_ticks at payload word 3 (raw u32)");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "HealOverTime: tail word {i} should be zero");
        }

        // TimedShield — same payload shape as DoT/HoT.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::TimedShield {
                source: aid(7),
                target: aid(11),
                amount: 25.0,
                duration_ticks: 100,
            },
            /*tick*/ 300,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("TimedShield has chronicle counterpart");
        assert_eq!(rec[0], 53, "TimedShield: kind tag — EffectTimedShieldApplied");
        assert_eq!(rec[1], 300, "TimedShield: tick");
        assert_eq!(rec[2], 7, "TimedShield: actor slot — caster_id");
        assert_eq!(rec[3], 11, "TimedShield: target slot — target_id");
        assert_eq!(rec[4], 25.0_f32.to_bits(), "TimedShield: amount at payload word 2");
        assert_eq!(rec[5], 100, "TimedShield: duration_ticks at payload word 3 (raw u32)");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "TimedShield: tail word {i} should be zero");
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
            /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Dash has chronicle counterpart");
        assert_eq!(rec[0], 47, "Dash: kind tag — EffectDashApplied");
        assert_eq!(rec[1], 100, "Dash: tick");
        assert_eq!(rec[2], 7, "Dash: actor slot — caster_id");
        assert_eq!(rec[3], 12.5_f32.to_bits(), "Dash: distance at payload word 1");
        for i in 4..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
            assert_eq!(rec[i], 0, "Dash: tail word {i} should be zero");
        }

        // Blink — same shape as Dash.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Blink { source: aid(7), distance: 8.0 },
            /*tick*/ 50,
            /*caster_id*/ 7,
            /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Blink has chronicle counterpart");
        assert_eq!(rec[0], 48, "Blink: kind tag — EffectBlinkApplied");
        assert_eq!(rec[1], 50, "Blink: tick");
        assert_eq!(rec[2], 7, "Blink: actor slot — caster_id");
        assert_eq!(rec[3], 8.0_f32.to_bits(), "Blink: distance at payload word 1");
        for i in 4..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            if i == 6 { continue; } // slot 6 = ability_id (Gap detective#6)
            assert_eq!(rec[i], 0, "Blink: tail word {i} should be zero");
        }

        // Knockback — forced motion on a target.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Knockback { source: aid(7), target: aid(11), distance: 5.0 },
            /*tick*/ 200,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Knockback has chronicle counterpart");
        assert_eq!(rec[0], 49, "Knockback: kind tag — EffectKnockbackApplied");
        assert_eq!(rec[1], 200, "Knockback: tick");
        assert_eq!(rec[2], 7, "Knockback: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Knockback: target slot — target_id");
        assert_eq!(rec[4], 5.0_f32.to_bits(), "Knockback: distance at payload word 2");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "Knockback: tail word {i} should be zero");
        }

        // Pull — same shape as Knockback.
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Pull { source: aid(7), target: aid(11), distance: 3.5 },
            /*tick*/ 300,
            /*caster_id*/ 7,
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Pull has chronicle counterpart");
        assert_eq!(rec[0], 50, "Pull: kind tag — EffectPullApplied");
        assert_eq!(rec[1], 300, "Pull: tick");
        assert_eq!(rec[2], 7, "Pull: actor slot — caster_id");
        assert_eq!(rec[3], 11, "Pull: target slot — target_id");
        assert_eq!(rec[4], 3.5_f32.to_bits(), "Pull: distance at payload word 2");
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
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
                /*target_id*/ 11, /*ability_id*/ 1,
            /*intra_emit_idx*/ 0,
            )
            .unwrap_or_else(|| panic!("{name} has chronicle counterpart"));
            assert_eq!(rec[0], *expected_kind, "{name}: kind tag");
            assert_eq!(rec[1], 100, "{name}: tick");
            assert_eq!(rec[2], 7, "{name}: actor slot — caster_id");
            assert_eq!(rec[3], 11, "{name}: target slot — target_id");
            assert_eq!(rec[4], 150, "{name}: expires_at_tick = tick(100) + duration(50)");
            for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
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
            /*caster_id*/ 7, /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
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
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
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
            /*caster_id*/ 7, /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("LifeSteal has chronicle counterpart");
        assert_eq!(rec[0], 40, "kind tag — EffectLifeStealApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id");
        assert_eq!(rec[3], 7, "target slot — target_id (self-cast: ==caster)");
        assert_eq!(rec[4], 150, "expires_at_tick = tick(100) + duration(50)");
        assert_eq!(rec[5], 128, "fraction_q8 = 128 (= 0.5×) sign-widened");
        // Tail words zeroed.
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }

        // Sign preservation: negative fraction_q8 sign-extends i16 → i32 → u32.
        let rec_neg = apply_event_to_chronicle_record(
            ApplyEvent::LifeSteal {
                target: aid(3),
                duration_ticks: 10,
                fraction_q8: -64,
            },
            50, 3, 3, 1, 0,
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
            /*target_id*/ 11, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Execute has chronicle counterpart");
        assert_eq!(rec[0], 42, "kind tag — EffectExecuteApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id");
        assert_eq!(rec[3], 11, "target slot — target_id");
        assert_eq!(rec[4], 20.0_f32.to_bits(), "hp_threshold as bitcast<u32>");
        // Tail words zeroed.
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }

        // Self-cast collapse: caster==target sets both slots equal.
        let rec_self = apply_event_to_chronicle_record(
            ApplyEvent::Execute {
                target: aid(7),
                hp_threshold: 50.0,
            },
            50, 7, 7, 1, 0,
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
            /*caster_id*/ 7, /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("DamageModify has chronicle counterpart");
        assert_eq!(rec[0], 41, "kind tag — EffectDamageModifyApplied");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "actor slot — caster_id");
        assert_eq!(rec[3], 7, "target slot — target_id (self-cast: ==caster)");
        assert_eq!(rec[4], 150, "expires_at_tick = tick(100) + duration(50)");
        assert_eq!(rec[5], 128, "multiplier_q8 = 128 (= 0.5×) sign-widened");
        // Tail words zeroed.
        for i in 7..(CHRONICLE_RECORD_STRIDE_U32 - 1) {
            assert_eq!(rec[i], 0, "tail word {i} should be zero");
        }

        // Sign preservation: negative multiplier_q8 sign-extends i16 → i32 → u32.
        let rec_neg = apply_event_to_chronicle_record(
            ApplyEvent::DamageModify {
                target: aid(3),
                duration_ticks: 10,
                multiplier_q8: -64,
            },
            50, 3, 3, 1, 0,
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
                20 => ApplyEvent::DamageOverTime { source: aid(1), target: aid(2), amount: 5.0, duration_ticks: 30 },
                21 => ApplyEvent::HealOverTime   { source: aid(1), target: aid(2), amount: 3.0, duration_ticks: 30 },
                22 => ApplyEvent::TimedShield    { source: aid(1), target: aid(2), amount: 25.0, duration_ticks: 30 },
                27 => ApplyEvent::Stealth        { source: aid(1), duration_ticks: 50 },
                28 => ApplyEvent::Charm          { target: aid(2), duration_ticks: 50 },
                29 => ApplyEvent::Grounded       { target: aid(2), duration_ticks: 50 },
                30 => ApplyEvent::Suppress       { target: aid(2), duration_ticks: 50 },
                23 => ApplyEvent::Buff           {
                    target: aid(2),
                    stat: engine::ability::program::BuffStat::MoveSpeed,
                    magnitude_q8: 64,
                    duration_ticks: 50,
                },
                24 => ApplyEvent::Summon         {
                    source: aid(1),
                    template_hash: 0xDEADBEEF,
                    count: 3,
                    lifetime_ticks: 120,
                },
                25 => ApplyEvent::Harvest        { source: aid(1), kind_hash: 0xCAFEBABE, amount: 5 },
                26 => ApplyEvent::PlaceVoxel     { source: aid(1), kind_hash: 0xFACEFEED },
                31 => ApplyEvent::Reflect        { target: aid(2), duration_ticks: 50, fraction_q8: 64 },
                32 => ApplyEvent::PlantBelief    { source: aid(1), target: aid(2), subject_idx: 7, fact_bit: 5 },
                33 => ApplyEvent::Observe        { source: aid(1), target: aid(2), target_observer: 0 },
                34 => ApplyEvent::Scry           { source: aid(1), target: aid(2), target_observer: 3, subject_idx: 4 },
                35 => ApplyEvent::Reveal         { source: aid(1), target: aid(2), subject_idx: 4 },
                36 => ApplyEvent::Disguise       { source: aid(1), fake_type: 7, duration_ticks: 200 },
                37 => ApplyEvent::Decoy          { source: aid(1), target: aid(2), subject_idx: 4, fake_pos: 0xDEADBEEF },
                38 => ApplyEvent::EraseBelief    { source: aid(1), target: aid(2), subject_idx: 4, fields: 0b00111111 },
                39 => ApplyEvent::TravelTo       { source: aid(1), dest_x: 5.0, dest_y: 5.0, eta_ticks: 50 },
                40 => ApplyEvent::Recipe         { source: aid(1), recipe_id: 42, target_tool: 0xFF },
                41 => ApplyEvent::WearTool       { source: aid(1), tool_kind: 3, amount: 64 },
                42 => ApplyEvent::Propose        { source: aid(1), target: aid(2), contract_kind: 1, expires_at_tick: 0 },
                43 => ApplyEvent::Announce       { source: aid(1), announcement_kind: 7, radius_q8: 896 },
                44 => ApplyEvent::GainSkill      { source: aid(1), skill_id: 2, amount_q8: 64 },
                45 => ApplyEvent::CreateObligation { source: aid(1), target: aid(2), obligation_id: 17, kind: 0 },
                46 => ApplyEvent::CastBegin       { source: aid(1), ability_id: 1, duration_ticks: 3, target_slot: 0, target_x_q8: 0, target_y_q8: 0 },
                _ => panic!("unexpected effect_kind in table"),
            }
        };

        for &(effect_kind, expected_event_kind_id) in EFFECT_KIND_TO_EVENT_KIND_ID {
            let ev = ev_for_kind(effect_kind);
            let rec = apply_event_to_chronicle_record(ev, 0, 0, 0, 0, 0)
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
            /*caster_id*/ 9, /*target_id*/ 9, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
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
            100, 9, 9, 1, 0,
        ).unwrap();
        assert_eq!(rec_neg[4], (-7_i32) as u32, "negative amount sign-widens correctly");
    }

    #[test]
    fn modify_standing_chronicle_record_uses_kind_32() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::ModifyStanding { source: aid(1), target: aid(2), delta: -25 },
            /*tick*/ 100,
            /*caster_id*/ 4, /*target_id*/ 4, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("ModifyStanding has chronicle counterpart");
        assert_eq!(rec[0], 32, "EffectStandingDelta kind tag");
        assert_eq!(rec[2], 4);
        assert_eq!(rec[3], 4);
        assert_eq!(rec[4], (-25_i32) as u32, "delta sign-widens i16 → i32 → u32");
    }

    // Lift A — `travel_to` chronicle record. Self-cast: target slot ==
    // caster slot. payload_a packs (dest_y_q8 << 16) | (dest_x_q8 &
    // 0xFFFF); payload_b = eta_ticks. Round-trip the q8 packing rule:
    // 5.0 → i16 1280 → bit pattern 0x0500. Both halves: low = 0x0500,
    // high = 0x0500 — combined 0x05000500.
    #[test]
    fn travel_to_chronicle_record_uses_kind_70_and_packs_q8() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::TravelTo { source: aid(1), dest_x: 5.0, dest_y: 5.0, eta_ticks: 50 },
            /*tick*/ 100,
            /*caster_id*/ 7, /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("TravelTo has chronicle counterpart");
        assert_eq!(rec[0], 70, "EffectTravelToApplied kind tag");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "caster slot — self-cast");
        assert_eq!(rec[3], 7, "target slot == caster slot for self-cast travel");
        // dest_x = 5.0 → q8 1280 → low 16 bits 0x0500
        // dest_y = 5.0 → q8 1280 → high 16 bits 0x0500 << 16 = 0x05000000
        // combined: 0x05000500
        assert_eq!(rec[4], 0x0500_0500, "packed q8 dest");
        assert_eq!(rec[5], 50, "eta_ticks");
    }

    // Negative-coord packing: dest_y = -1.0 → q8 -256 → bit pattern
    // 0xFF00 (i16 sign-extend). Combined low half (dest_x = 1.0 → q8
    // 256 → 0x0100): 0xFF000100.
    #[test]
    fn travel_to_chronicle_handles_negative_q8_coords() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::TravelTo { source: aid(1), dest_x: 1.0, dest_y: -1.0, eta_ticks: 25 },
            /*tick*/ 50,
            /*caster_id*/ 3, /*target_id*/ 3, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("TravelTo has chronicle counterpart");
        assert_eq!(rec[0], 70);
        // -1.0 q8 → -256 → i16 bit pattern 0xFF00 → as u16 → u32 0xFF00.
        // Shifted to high half: 0xFF000000. Combined with low (1.0 → 256
        // → 0x0100): 0xFF000100.
        assert_eq!(rec[4], 0xFF00_0100, "q8 negative coord packs with sign bits intact");
        assert_eq!(rec[5], 25);
    }

    // Lift B — `cast_recipe` chronicle record. Self-cast: target slot ==
    // caster slot. payload_a packs (target_tool << 16) | recipe_id;
    // payload_b = 0. recipe_id=42 (0x002A), target_tool=0xFF →
    // packed = 0x00FF_002A.
    #[test]
    fn recipe_chronicle_record_uses_kind_71_and_packs_recipe_id() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Recipe { source: aid(1), recipe_id: 42, target_tool: 0xFF },
            /*tick*/ 100,
            /*caster_id*/ 7, /*target_id*/ 7, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Recipe has chronicle counterpart");
        assert_eq!(rec[0], 71, "EffectRecipeApplied kind tag");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 7, "caster slot — self-cast (recipe acts on caster's inventory)");
        assert_eq!(rec[3], 7, "target slot == caster slot for self-cast recipe");
        assert_eq!(rec[4], 0x00FF_002A, "packed (target_tool << 16) | recipe_id");
        assert_eq!(rec[5], 0, "payload_b unused");
    }

    // Lift B — `cast_recipe` with explicit target tool slot (not the
    // 0xFF sentinel). recipe_id=7 (0x0007), target_tool=3 → packed =
    // 0x0003_0007. Pins the target_tool packing position.
    #[test]
    fn recipe_chronicle_packs_explicit_target_tool() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::Recipe { source: aid(1), recipe_id: 7, target_tool: 3 },
            /*tick*/ 50,
            /*caster_id*/ 4, /*target_id*/ 4, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("Recipe has chronicle counterpart");
        assert_eq!(rec[0], 71);
        assert_eq!(rec[4], 0x0003_0007, "target_tool packs into bits 16..24");
    }

    // Lift B — `wear_tool` chronicle record. Self-cast: target slot ==
    // caster slot. payload_a packs (amount << 8) | tool_kind;
    // payload_b = 0. tool_kind=3 (0x03), amount=64 (0x40) →
    // packed = (64 << 8) | 3 = 0x0000_4003.
    #[test]
    fn wear_tool_chronicle_record_uses_kind_72_and_packs_amount() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::WearTool { source: aid(1), tool_kind: 3, amount: 64 },
            /*tick*/ 100,
            /*caster_id*/ 5, /*target_id*/ 5, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("WearTool has chronicle counterpart");
        assert_eq!(rec[0], 72, "EffectWearToolApplied kind tag");
        assert_eq!(rec[1], 100, "tick");
        assert_eq!(rec[2], 5, "caster slot — self-cast (wear acts on caster's tool)");
        assert_eq!(rec[3], 5, "target slot == caster slot for self-cast wear");
        assert_eq!(rec[4], 0x0000_4003, "packed (amount << 8) | tool_kind");
        assert_eq!(rec[5], 0, "payload_b unused");
    }

    // Lift B — `wear_tool` with the maximal amount (u16::MAX = 0xFFFF).
    // tool_kind=1, amount=0xFFFF → packed = (0xFFFF << 8) | 1 =
    // 0x00FF_FF01. Pins the amount packing position so a future widening
    // surfaces here.
    #[test]
    fn wear_tool_chronicle_packs_max_amount() {
        let rec = apply_event_to_chronicle_record(
            ApplyEvent::WearTool { source: aid(1), tool_kind: 1, amount: 0xFFFF },
            /*tick*/ 25,
            /*caster_id*/ 9, /*target_id*/ 9, /*ability_id*/ 1,
        /*intra_emit_idx*/ 0,
        )
        .expect("WearTool has chronicle counterpart");
        assert_eq!(rec[0], 72);
        assert_eq!(rec[4], 0x00FF_FF01, "amount fills bits 8..24, tool_kind in bits 0..8");
    }
}
