//! Registry-driven apply dispatch (#125, MVP slice).
//!
//! Translates `AbilityProgram` IR into a stream of typed `ApplyEvent`s
//! that downstream sims can drain into their existing event rings.
//! Honors the per-effect chance gate (`program.chances[i]`) using
//! `per_agent_u32` per the P5 keyed-PCG contract — replay equivalence
//! and cross-backend parity hold.
//!
//! # Status
//!
//! MVP scope is the per-(caster, target) translation pass. Out of
//! scope for this slice:
//!   * scaling lookup against caster stat SoA (needs sim-side stat
//!     resolver — pass via callback or context struct in a follow-up)
//!   * per_effect_areas spatial dispatch (single-target only today —
//!     AOE expansion lives in #121)
//!   * nested_per_effect cascade (deferred — #123 IR done; runtime
//!     scheduler not wired)
//!   * delivery method scheduling (Projectile travel, Channel hold —
//!     #124 IR done; runtime not wired)
//!
//! # Contract with sims
//!
//! Sims that opt into registry-driven dispatch:
//!   1. Bind their event vocabulary to engine's `ApplyEvent` (or
//!      provide a translator at the boundary).
//!   2. Replace per-verb hand-mirrored emit blocks with a single
//!      generic `apply_program(ability_id, caster, target, …)` call
//!      from each verb body.
//!   3. Keep their existing apply-physics chronicles
//!      (ApplyDamage / ApplyHeal / etc.) — those drain `ApplyEvent`s
//!      into SoA mutations exactly the same way.
//!
//! With this slice landed, a sim with N hand-mirrored verbs collapses
//! to one generic dispatcher; adding a new ability becomes a pure
//! .ability-file change.

use crate::ability::program::{AbilityProgram, BuffStat, CasterStats, EffectOp};
use crate::ids::AgentId;
use crate::rng::per_agent_u32;
use smallvec::SmallVec;

/// Typed apply-event vocabulary. Each variant matches an `EffectOp`
/// shape exactly, expanded with the caster/target context resolved at
/// dispatch time. Sims consume these via their existing apply-physics
/// chronicles (`on Damaged { … }`, `on Healed { … }`, etc.).
///
/// `source = u32::MAX` for self-only effects (Stun/Buff target the
/// passed-in target; the source field is unused but kept for shape
/// uniformity). Apply-physics handlers should match on the variant
/// they care about and ignore unused fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApplyEvent {
    Damage     { source: AgentId, target: AgentId, amount: f32 },
    Heal       { source: AgentId, target: AgentId, amount: f32 },
    Shield     { source: AgentId, target: AgentId, amount: f32 },
    Stun       { target: AgentId, duration_ticks: u32 },
    Slow       { target: AgentId, duration_ticks: u32, factor_q8: i16 },
    Root       { target: AgentId, duration_ticks: u32 },
    Silence    { target: AgentId, duration_ticks: u32 },
    Fear       { target: AgentId, duration_ticks: u32 },
    Taunt      { target: AgentId, duration_ticks: u32 },
    Dash       { source: AgentId, distance: f32 },
    Blink      { source: AgentId, distance: f32 },
    Knockback  { source: AgentId, target: AgentId, distance: f32 },
    Pull       { source: AgentId, target: AgentId, distance: f32 },
    Execute    { target: AgentId, hp_threshold: f32 },
    SelfDamage { source: AgentId, amount: f32 },
    LifeSteal  { target: AgentId, duration_ticks: u32, fraction_q8: i16 },
    DamageModify { target: AgentId, duration_ticks: u32, multiplier_q8: i16 },
    DamageOverTime { source: AgentId, target: AgentId, amount: f32, duration_ticks: u32 },
    HealOverTime   { source: AgentId, target: AgentId, amount: f32, duration_ticks: u32 },
    TimedShield    { source: AgentId, target: AgentId, amount: f32, duration_ticks: u32 },
    Buff           { target: AgentId, stat: BuffStat, magnitude_q8: i16, duration_ticks: u32 },
    /// `summon "<template>" [N] [for <duration>]` — caster spawns
    /// `count` minions of `template_hash` for `lifetime_ticks`. The
    /// template hash is the FxHash of the template ident from the
    /// .ability source (deferred resolution — apply handlers map the
    /// hash to a spawner via a registry follow-up). Captured here so
    /// downstream sims can drain the event when the spawner wires up;
    /// no runtime sim consumes it today (deferred infra mirroring the
    /// CastAbility/TransferGold/ModifyStanding fall-through pattern).
    Summon         { source: AgentId, template_hash: u32, count: u8, lifetime_ticks: u32 },
    /// `harvest "<kind>" [<amount>]` — caster gathers `amount` units of
    /// the named resource. `kind_hash` is the FxHash of the resource
    /// ident from the .ability source (deferred resolution — apply
    /// handlers map the hash to a concrete resource via a registry
    /// follow-up). Apply handlers route to AgentHarvested for organic /
    /// surface resources or AgentHarvestedVoxel for voxel-backed
    /// resources, distinguished by the registry lookup. No runtime sim
    /// consumes it today (deferred infra mirroring the Summon
    /// fall-through pattern).
    Harvest        { source: AgentId, kind_hash: u32, amount: u16 },
    /// `place_voxel "<kind>"` — caster places one voxel of `kind_hash`
    /// at the cast target's position. Apply handlers emit
    /// AgentPlacedVoxel and write the voxel into world state; deferred
    /// infrastructure today (no runtime sim consumes the event yet).
    PlaceVoxel     { source: AgentId, kind_hash: u32 },
    /// `stealth for <duration>` — self-cast invisibility for
    /// `duration_ticks`. The LoL idiom is `stealth for 3s
    /// break_on_damage` — the lifetime modifier rides the per-effect
    /// lifetime SoA and isn't reflected here. Apply handlers will
    /// gate target selection by the caster's stealth flag. No runtime
    /// sim consumes the event today (deferred — same fall-through as
    /// Summon / Harvest / PlaceVoxel).
    Stealth        { source: AgentId, duration_ticks: u32 },
    /// Wave 2 piece 8 CC verbs. Same shape as Stun (target + duration);
    /// apply handlers wire per-agent expiry tick-stamps. No runtime
    /// sim consumes them today.
    Charm          { target: AgentId, duration_ticks: u32 },
    Grounded       { target: AgentId, duration_ticks: u32 },
    Suppress       { target: AgentId, duration_ticks: u32 },
    /// `reflect <fraction> for <duration>` — fraction-of-damage
    /// bounce. Mirrors DamageModify's payload shape.
    Reflect        { target: AgentId, duration_ticks: u32, fraction_q8: i16 },
    /// `transfer_gold <amount>` — caster moves `amount` gold to
    /// target. The world-state effect (debiting caster's purse,
    /// crediting target's purse) is downstream of apply_program;
    /// this variant signals **the cast occurred** for chronicle /
    /// reaction-handler consumers. Pairs with
    /// `EventKindId::EffectGoldTransfer = 31` on the chronicle side.
    TransferGold   { source: AgentId, target: AgentId, amount: i32 },
    /// `modify_standing <delta>` — caster changes their standing
    /// with target by `delta` (i16 signed delta in standing's
    /// internal units). Same world-state-deferred shape as
    /// TransferGold. Pairs with `EventKindId::EffectStandingDelta =
    /// 32` on the chronicle side.
    ModifyStanding { source: AgentId, target: AgentId, delta: i16 },
}

/// Inline budget — most abilities have ≤4 effects (P4 says
/// `MAX_EFFECTS_PER_PROGRAM = 6` today). Heap-spill is fine for
/// the rare 5+ ult.
const APPLY_INLINE: usize = 4;

/// Translate one cast of `program` (caster → target at `tick`) into a
/// stream of ApplyEvents. Honors the per-effect chance gate AND the
/// per-effect `scalings_per_effect` modifier (`+ N% stat_ref`).
///
/// `world_seed` and `tick` together with `caster` derive the RNG
/// stream per P5 — replay equivalence holds because the same cast at
/// the same tick produces the same gate decisions.
///
/// `caster_stats` is the caster's stat snapshot at cast-decide time.
/// For each amount-bearing variant (Damage / Heal / Shield / SelfDamage
/// / DamageOverTime / HealOverTime / TimedShield), the dispatcher
/// computes `scaled = base + Σ percent * stat` from
/// `program.scalings_per_effect[i]` before emitting the event. Pass
/// `&CasterStats::default()` for legacy / non-scaling call sites —
/// all-zero stats project to a `0.0` contribution per scaling slot, so
/// the output is byte-identical to the pre-scaling apply path when the
/// program carries no scalings (or when the caster has no relevant
/// stats).
///
/// `Some(amount) = 0xFFFF` chance slot fires deterministically (max
/// q16 value — apply handlers treat as "always"); `None` slot also
/// fires deterministically (no gate authored). The runtime gate
/// compares `(per_agent_u32 & 0xFFFF) < q16` — when q16=65534
/// (canonical "100%") this is true 65534/65536 ≈ 99.997% of draws
/// (indistinguishable from "always" at 16-bit RNG resolution).
pub fn apply_program(
    program:      &AbilityProgram,
    caster:       AgentId,
    target:       AgentId,
    tick:         u64,
    world_seed:   u64,
    caster_stats: &CasterStats,
) -> SmallVec<[ApplyEvent; APPLY_INLINE]> {
    let mut out: SmallVec<[ApplyEvent; APPLY_INLINE]> = SmallVec::new();

    for (i, op) in program.effects.iter().enumerate() {
        // -- Wave 1.5#5 chance gate. --
        // The chances slice is either empty (no effect carried the
        // modifier — fire all) or per-effect Option<u16>. None within
        // a populated slice = no gate on that slot.
        if let Some(Some(q16)) = program.chances.get(i).copied() {
            // P5: derive the draw from (world_seed, caster, tick,
            // effect_slot) — purpose tag includes the slot index so
            // multi-effect abilities don't share a draw.
            let purpose = [b'c', b'h', b'a', b'n', b'c', b'e', i as u8];
            let draw = per_agent_u32(world_seed, caster, tick, &purpose) & 0xFFFF;
            if (draw as u16) >= q16 {
                continue; // gate fails — skip this effect
            }
        }
        // -- Wave 1.5#4 scaling — compute additive `Σ percent * stat`
        // bonus from `scalings_per_effect[i]`. Empty/missing slot ⇒ 0.0
        // (output bit-identical to pre-scaling behavior). Apply only to
        // amount-bearing variants in the dispatch arms below.
        let scale_bonus: f32 = program
            .scalings_per_effect
            .get(i)
            .map(|inner| {
                inner
                    .iter()
                    .map(|s| s.percent * caster_stats.get(s.stat_ref))
                    .sum::<f32>()
            })
            .unwrap_or(0.0);
        // -- Per-EffectOp dispatch. Mirrors pack_effect's variant
        // walk. Future in-shape / nested handling threads into here.
        match *op {
            EffectOp::Damage    { amount } => out.push(ApplyEvent::Damage { source: caster, target, amount: amount + scale_bonus }),
            EffectOp::Heal      { amount } => out.push(ApplyEvent::Heal   { source: caster, target, amount: amount + scale_bonus }),
            EffectOp::Shield    { amount } => out.push(ApplyEvent::Shield { source: caster, target, amount: amount + scale_bonus }),
            EffectOp::Stun      { duration_ticks } => out.push(ApplyEvent::Stun    { target, duration_ticks }),
            EffectOp::Slow      { duration_ticks, factor_q8 } =>
                out.push(ApplyEvent::Slow { target, duration_ticks, factor_q8 }),
            EffectOp::Root      { duration_ticks } => out.push(ApplyEvent::Root    { target, duration_ticks }),
            EffectOp::Silence   { duration_ticks } => out.push(ApplyEvent::Silence { target, duration_ticks }),
            EffectOp::Fear      { duration_ticks } => out.push(ApplyEvent::Fear    { target, duration_ticks }),
            EffectOp::Taunt     { duration_ticks } => out.push(ApplyEvent::Taunt   { target, duration_ticks }),
            EffectOp::Dash      { distance } => out.push(ApplyEvent::Dash  { source: caster, distance }),
            EffectOp::Blink     { distance } => out.push(ApplyEvent::Blink { source: caster, distance }),
            EffectOp::Knockback { distance } => out.push(ApplyEvent::Knockback { source: caster, target, distance }),
            EffectOp::Pull      { distance } => out.push(ApplyEvent::Pull      { source: caster, target, distance }),
            EffectOp::Execute   { hp_threshold } => out.push(ApplyEvent::Execute { target, hp_threshold }),
            EffectOp::SelfDamage{ amount } => out.push(ApplyEvent::SelfDamage { source: caster, amount: amount + scale_bonus }),
            EffectOp::LifeSteal { duration_ticks, fraction_q8 } =>
                out.push(ApplyEvent::LifeSteal { target: caster, duration_ticks, fraction_q8 }),
            EffectOp::DamageModify { duration_ticks, multiplier_q8 } =>
                out.push(ApplyEvent::DamageModify { target, duration_ticks, multiplier_q8 }),
            EffectOp::DamageOverTime { amount, duration_ticks } =>
                out.push(ApplyEvent::DamageOverTime { source: caster, target, amount: amount + scale_bonus, duration_ticks }),
            EffectOp::HealOverTime   { amount, duration_ticks } =>
                out.push(ApplyEvent::HealOverTime   { source: caster, target, amount: amount + scale_bonus, duration_ticks }),
            EffectOp::TimedShield    { amount, duration_ticks } =>
                out.push(ApplyEvent::TimedShield    { source: caster, target, amount: amount + scale_bonus, duration_ticks }),
            EffectOp::Buff { stat, magnitude_q8, duration_ticks } =>
                out.push(ApplyEvent::Buff { target, stat, magnitude_q8, duration_ticks }),
            EffectOp::Summon { template_hash, count, lifetime_ticks } =>
                out.push(ApplyEvent::Summon { source: caster, template_hash, count, lifetime_ticks }),
            // Non-combat verbs phase 1 — world primitives. No scaling
            // applies (these aren't amount-bearing in the combat sense
            // — `amount` is a resource quantity, not an HP delta).
            EffectOp::Harvest    { kind_hash, amount } =>
                out.push(ApplyEvent::Harvest    { source: caster, kind_hash, amount }),
            EffectOp::PlaceVoxel { kind_hash } =>
                out.push(ApplyEvent::PlaceVoxel { source: caster, kind_hash }),
            // Wave 2 piece 7: stealth is self-cast (apply handler
            // gates target selection by caster's stealth flag).
            EffectOp::Stealth    { duration_ticks } =>
                out.push(ApplyEvent::Stealth { source: caster, duration_ticks }),
            // Wave 2 piece 8 CC verbs — target-cast, single duration.
            EffectOp::Charm      { duration_ticks } =>
                out.push(ApplyEvent::Charm    { target, duration_ticks }),
            EffectOp::Grounded   { duration_ticks } =>
                out.push(ApplyEvent::Grounded { target, duration_ticks }),
            EffectOp::Suppress   { duration_ticks } =>
                out.push(ApplyEvent::Suppress { target, duration_ticks }),
            EffectOp::Reflect    { duration_ticks, fraction_q8 } =>
                out.push(ApplyEvent::Reflect  { target, duration_ticks, fraction_q8 }),
            // TransferGold / ModifyStanding emit chronicle-bearing
            // ApplyEvents that signal "the cast happened". The
            // world-state effects (debiting/crediting purses, mutating
            // standing tables) are downstream of apply_program — kept
            // intentionally separate so the chronicle stream stays a
            // pure function of the cast inputs (P5/P11) regardless of
            // when the world-state side-effects land. Pairs with
            // `EventKindId::EffectGoldTransfer = 31` and
            // `EffectStandingDelta = 32` respectively.
            EffectOp::TransferGold { amount } =>
                out.push(ApplyEvent::TransferGold { source: caster, target, amount }),
            EffectOp::ModifyStanding { delta } =>
                out.push(ApplyEvent::ModifyStanding { source: caster, target, delta }),
            // CastAbility is recursive (needs cascade-style
            // re-dispatch); deferred to slice δ. Skip for now.
            EffectOp::CastAbility { .. } => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::program::{EffectScaling, Gate, ScalingStatRef};
    use crate::ability::AbilityId;
    use smallvec::smallvec;

    fn caster() -> AgentId { AgentId::new(1).unwrap() }
    fn target() -> AgentId { AgentId::new(2).unwrap() }

    #[test]
    fn apply_strike_emits_damage_event() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            ApplyEvent::Damage { source, target: t, amount }
            if source == caster() && t == target() && amount == 30.0
        ));
    }

    #[test]
    fn apply_multi_effect_program_emits_in_order() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 20.0 },
                EffectOp::Stun   { duration_ticks: 10 },
            ],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ApplyEvent::Damage { .. }));
        assert!(matches!(events[1], ApplyEvent::Stun { .. }));
    }

    #[test]
    fn chance_zero_gates_out() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Stun { duration_ticks: 10 }],
        );
        // q16 = 0 → no draw can be < 0; effect never fires.
        prog.chances.push(Some(0));
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 0, "chance=0 must gate the effect out");
    }

    #[test]
    fn chance_max_always_fires() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Stun { duration_ticks: 10 }],
        );
        // q16 = 0xFFFE (canonical 100% per the chance lowering's
        // clamp(0..=65534)) — fires for any draw < 65534, i.e. all
        // but 1/65536 of draws. Try a fixed seed/tick combination to
        // verify the expected fire (deterministic).
        prog.chances.push(Some(0xFFFE));
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 1, "chance=0xFFFE must fire deterministically at this seed/tick");
    }

    #[test]
    fn chance_deterministic_replay() {
        // Same (program, caster, target, tick, seed) must produce the
        // same gate decision across calls — P5 replay equivalence.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Stun { duration_ticks: 10 }],
        );
        prog.chances.push(Some(32768)); // 50%
        let a = apply_program(&prog, caster(), target(), 42, 0xCAFE, &CasterStats::default());
        let b = apply_program(&prog, caster(), target(), 42, 0xCAFE, &CasterStats::default());
        assert_eq!(a.len(), b.len(), "same inputs → same gate decisions");
    }

    #[test]
    fn cast_ability_falls_through() {
        // CastAbility is recursive cascade — out of MVP scope. Apply
        // skips it without panicking (P10 — no panic on hot path).
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::CastAbility {
                ability:  AbilityId::new(1).unwrap(),
                selector: crate::ability::program::TargetSelector::Target,
            }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 0, "CastAbility falls through (deferred)");
    }

    #[test]
    fn transfer_gold_emits_apply_event_with_amount() {
        // EffectOp::TransferGold packs source=caster, target=target,
        // amount=raw i32. World-state effects (purse debit/credit)
        // are downstream of apply_program.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::TransferGold { amount: 42 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 1, "TransferGold emits exactly one ApplyEvent");
        match events[0] {
            ApplyEvent::TransferGold { source, target: t, amount } => {
                assert_eq!(source, caster());
                assert_eq!(t, target());
                assert_eq!(amount, 42, "amount round-trips from EffectOp");
            }
            other => panic!("expected TransferGold, got {other:?}"),
        }
    }

    #[test]
    fn transfer_gold_preserves_negative_amount() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::TransferGold { amount: -7 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        match events[0] {
            ApplyEvent::TransferGold { amount, .. } =>
                assert_eq!(amount, -7, "negative amount preserved (sign isn't lost)"),
            other => panic!("expected TransferGold, got {other:?}"),
        }
    }

    #[test]
    fn modify_standing_emits_apply_event_with_delta() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::ModifyStanding { delta: -25 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 1, "ModifyStanding emits exactly one ApplyEvent");
        match events[0] {
            ApplyEvent::ModifyStanding { source, target: t, delta } => {
                assert_eq!(source, caster());
                assert_eq!(t, target());
                assert_eq!(delta, -25, "delta round-trips from EffectOp (sign preserved)");
            }
            other => panic!("expected ModifyStanding, got {other:?}"),
        }
    }

    // -- Caster-stat scaling --------------------------------------------------

    #[test]
    fn apply_strike_with_attack_damage_scaling_adds_to_amount() {
        // Damage 30 + 50% AD; caster has 100 AD ⇒ emit 30 + 50 = 80.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.scalings_per_effect.push(smallvec![EffectScaling {
            stat_ref: ScalingStatRef::AttackDamage,
            percent:  0.50,
        }]);
        let stats = CasterStats { attack_damage: 100.0, ..Default::default() };
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats);
        assert_eq!(events.len(), 1);
        match events[0] {
            ApplyEvent::Damage { amount, .. } => {
                assert!((amount - 80.0).abs() < 1e-5, "expected 80.0, got {amount}");
            }
            other => panic!("expected Damage, got {other:?}"),
        }
    }

    #[test]
    fn apply_skipped_effect_doesnt_scale() {
        // chance=0 gates the effect out — no event emitted, scaling math
        // must not run (and certainly must not produce a side-effect).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.chances.push(Some(0));
        prog.scalings_per_effect.push(smallvec![EffectScaling {
            stat_ref: ScalingStatRef::AttackDamage,
            percent:  10.0, // would be huge if it ran
        }]);
        let stats = CasterStats { attack_damage: 1000.0, ..Default::default() };
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats);
        assert_eq!(events.len(), 0, "chance=0 must gate the effect out before scaling");
    }

    #[test]
    fn apply_no_scaling_is_bit_stable() {
        // Empty `scalings_per_effect` ⇒ output identical to the
        // pre-scaling apply path (regression guard for the B slice).
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 30.0 },
                EffectOp::Heal   { amount: 12.5 },
                EffectOp::Shield { amount:  7.0 },
            ],
        );
        // Even with massive caster stats, an empty scalings vec must
        // contribute zero — output is bit-identical to default-stats.
        let stats = CasterStats {
            attack_damage: 9999.0,
            ability_power: 9999.0,
            ..Default::default()
        };
        let with    = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats);
        let without = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(with.len(), without.len());
        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!(a, b, "with-stats vs default-stats diverged with no scalings");
        }
        // Spot-check the absolute values.
        assert!(matches!(with[0], ApplyEvent::Damage { amount, .. } if amount == 30.0));
        assert!(matches!(with[1], ApplyEvent::Heal   { amount, .. } if amount == 12.5));
        assert!(matches!(with[2], ApplyEvent::Shield { amount, .. } if amount == 7.0));
    }

    #[test]
    fn apply_summon_emits_summon_event() {
        // Verify the new EffectOp::Summon arm produces an ApplyEvent::Summon
        // with the template_hash threaded through.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Summon { template_hash: 0xDEADBEEF, count: 3, lifetime_ticks: 80 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default());
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            ApplyEvent::Summon { source, template_hash, count, lifetime_ticks }
            if source == caster() && template_hash == 0xDEADBEEF && count == 3 && lifetime_ticks == 80
        ));
    }
}
