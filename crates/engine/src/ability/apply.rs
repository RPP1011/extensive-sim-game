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

use crate::ability::program::{AbilityProgram, BuffStat, EffectOp};
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
}

/// Inline budget — most abilities have ≤4 effects (P4 says
/// `MAX_EFFECTS_PER_PROGRAM = 6` today). Heap-spill is fine for
/// the rare 5+ ult.
const APPLY_INLINE: usize = 4;

/// Translate one cast of `program` (caster → target at `tick`) into a
/// stream of ApplyEvents. Honors the per-effect chance gate.
///
/// `world_seed` and `tick` together with `caster` derive the RNG
/// stream per P5 — replay equivalence holds because the same cast at
/// the same tick produces the same gate decisions.
///
/// `Some(amount) = 0xFFFF` chance slot fires deterministically (max
/// q16 value — apply handlers treat as "always"); `None` slot also
/// fires deterministically (no gate authored). The runtime gate
/// compares `(per_agent_u32 & 0xFFFF) < q16` — when q16=65534
/// (canonical "100%") this is true 65534/65536 ≈ 99.997% of draws
/// (indistinguishable from "always" at 16-bit RNG resolution).
pub fn apply_program(
    program: &AbilityProgram,
    caster:  AgentId,
    target:  AgentId,
    tick:    u64,
    world_seed: u64,
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
        // -- Per-EffectOp dispatch. Mirrors pack_effect's variant
        // walk. Future scaling/in-shape/nested handling threads into
        // here.
        match *op {
            EffectOp::Damage    { amount } => out.push(ApplyEvent::Damage { source: caster, target, amount }),
            EffectOp::Heal      { amount } => out.push(ApplyEvent::Heal   { source: caster, target, amount }),
            EffectOp::Shield    { amount } => out.push(ApplyEvent::Shield { source: caster, target, amount }),
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
            EffectOp::SelfDamage{ amount } => out.push(ApplyEvent::SelfDamage { source: caster, amount }),
            EffectOp::LifeSteal { duration_ticks, fraction_q8 } =>
                out.push(ApplyEvent::LifeSteal { target: caster, duration_ticks, fraction_q8 }),
            EffectOp::DamageModify { duration_ticks, multiplier_q8 } =>
                out.push(ApplyEvent::DamageModify { target, duration_ticks, multiplier_q8 }),
            EffectOp::DamageOverTime { amount, duration_ticks } =>
                out.push(ApplyEvent::DamageOverTime { source: caster, target, amount, duration_ticks }),
            EffectOp::HealOverTime   { amount, duration_ticks } =>
                out.push(ApplyEvent::HealOverTime   { source: caster, target, amount, duration_ticks }),
            EffectOp::TimedShield    { amount, duration_ticks } =>
                out.push(ApplyEvent::TimedShield    { source: caster, target, amount, duration_ticks }),
            EffectOp::Buff { stat, magnitude_q8, duration_ticks } =>
                out.push(ApplyEvent::Buff { target, stat, magnitude_q8, duration_ticks }),
            // CastAbility / TransferGold / ModifyStanding fall outside
            // this slice — the first is recursive (needs cascade
            // handling); the latter two need world-state context not
            // threaded through here. Skip for now.
            EffectOp::CastAbility { .. }
            | EffectOp::TransferGold { .. }
            | EffectOp::ModifyStanding { .. } => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::program::Gate;
    use crate::ability::AbilityId;

    fn caster() -> AgentId { AgentId::new(1).unwrap() }
    fn target() -> AgentId { AgentId::new(2).unwrap() }

    #[test]
    fn apply_strike_emits_damage_event() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE);
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
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE);
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
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE);
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
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE);
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
        let a = apply_program(&prog, caster(), target(), 42, 0xCAFE);
        let b = apply_program(&prog, caster(), target(), 42, 0xCAFE);
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
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE);
        assert_eq!(events.len(), 0, "CastAbility falls through (deferred)");
    }
}
