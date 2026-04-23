//! `AbilityProgram` IR — the intermediate representation that the ability DSL
//! compiler targets and that the `CastHandler` (Task 9) interprets at runtime.
//!
//! One program = one ability. It describes **how** the cast is delivered
//! (`Delivery`), **where** its effect lands (`Area`), **when** it is allowed
//! to fire (`Gate`), and **what** happens on resolve (a bounded smallvec of
//! `EffectOp` atoms).
//!
//! Design notes:
//! - `EffectOp` is `Copy`, explicit-repr u8 discriminant, and pinned to a
//!   16-byte size budget (see `ability_program_shape.rs`). Keeping it small
//!   matters because the cast-dispatch hot-path iterates the effects smallvec
//!   once per cast; fitting 4 effects in 64 bytes keeps them cache-local.
//! - The Meta effect (`CastAbility`) is what makes the subsystem recursive.
//!   Recursion is bounded by the engine's `MAX_CASCADE_ITERATIONS = 8` —
//!   Task 18 pins the depth budget.
//! - `TransferGold.amount` is `i32` (signed) so debt / refunds round-trip
//!   through the same op. `Inventory.gold` is also `i32` (narrowed 2026-04-22
//!   for GPU atomic compatibility; was i64).

use crate::ability::AbilityId;
use smallvec::SmallVec;

/// Maximum number of effects a single `AbilityProgram` may carry. Bounded so
/// the effects smallvec stays stack-resident and cast-dispatch touches a
/// fixed-size window per cast. Pinned by the size-budget test. Changing this
/// is a schema-hash bump.
pub const MAX_EFFECTS_PER_PROGRAM: usize = 4;

/// Maximum number of ability slots per agent. Governs the inner
/// dimension of `SimState::ability_cooldowns`, which carries a
/// per-(agent, ability-slot) local cooldown cursor alongside the
/// per-agent global GCD in `hot_cooldown_next_ready_tick`.
///
/// Added 2026-04-22 with the ability-cooldowns subsystem. 8 matches
/// the Ability Registry's per-unit slot budget from prior hero
/// templates; raise if/when a subsystem needs more per-agent slots
/// (storage cost is `8 × agent_cap × 4B` for the u32 cursor array).
pub const MAX_ABILITIES: usize = 8;

/// How a cast is delivered to its target.
///
/// MVP ships `Instant` only — the effect applies on the tick the cast resolves.
/// Plan-2 ability work adds `Projectile` (travel-time) and `Zone` (persistent
/// AoE); those variants land alongside their resolver code.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Delivery {
    Instant = 0,
}

/// Where the effect lands relative to the cast's target.
///
/// MVP ships `SingleTarget`; Plan-2 adds `Cone`, `Circle`, and full AoE.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Area {
    SingleTarget { range: f32 },
}

/// Gate predicates — cast-time filters evaluated by `evaluate_cast_gate`
/// (Task 9). `cooldown_ticks` is the number of ticks after a cast resolves
/// before the caster may cast again; `hostile_only` forces targets to pass
/// `CreatureType::is_hostile_to`; `line_of_sight` reserves a bit that the
/// pathfinding plan wires up later (MVP: unused).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Gate {
    pub cooldown_ticks: u32,
    pub hostile_only:   bool,
    pub line_of_sight:  bool,
}

/// For meta (recursive) effects: which agent does the nested cast target?
/// `Target` forwards the outer cast's target; `Caster` redirects to the
/// caster (e.g. a self-buff that fires alongside a damage op).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TargetSelector {
    Target = 0,
    Caster = 1,
}

/// One effect atom. Cast-dispatch (Task 9) pattern-matches and emits one
/// `Effect*Applied` event per op, handed to a per-op cascade handler in
/// Tasks 10–18.
///
/// Explicit discriminants pin ordinals for the schema hash.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EffectOp {
    // --- Combat (5) ---
    /// Apply `amount` hp damage to the target. Shield absorbs first
    /// (Task 10); overflow lands on hp. Negative `amount` is undefined
    /// (use `Heal` for that direction).
    Damage { amount: f32 } = 0,

    /// Raise target hp by `amount`, clamped at max_hp. Emits the real applied
    /// delta so replays observe post-clamp values.
    Heal { amount: f32 } = 1,

    /// Add `amount` to target's `hot_shield_hp`. Stackable; no decay in MVP.
    Shield { amount: f32 } = 2,

    /// Set `hot_stun_remaining_ticks` on target. Longer stun wins (max rule).
    Stun { duration_ticks: u32 } = 3,

    /// Set `hot_slow_remaining_ticks` and `hot_slow_factor_q8`. Combining
    /// rule: longer-or-stronger wins (see Task 14).
    Slow { duration_ticks: u32, factor_q8: i16 } = 4,

    // --- World (2) ---
    /// Move gold between caster and target via `cold_inventory[slot].gold`.
    /// Signed i32 — debt is allowed; negative `amount` reverses the flow.
    /// Narrowed from i64 on 2026-04-22 for GPU atomic compatibility.
    TransferGold { amount: i32 } = 5,

    /// Adjust symmetric `cold_standing` between caster and target by `delta`.
    /// Clamped to `[-1000, 1000]` inside the handler.
    ModifyStanding { delta: i16 } = 6,

    // --- Meta (1) ---
    /// Re-cast another ability. `selector` picks whose slot becomes the nested
    /// target. Recursion is bounded by `MAX_CASCADE_ITERATIONS = 8`.
    CastAbility { ability: AbilityId, selector: TargetSelector } = 7,
}

/// Compiled ability — the unit `AbilityRegistry` stores and `CastHandler`
/// dispatches.
#[derive(Clone, Debug)]
pub struct AbilityProgram {
    pub delivery: Delivery,
    pub area:     Area,
    pub gate:     Gate,
    pub effects:  SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]>,
}

impl AbilityProgram {
    /// Convenience: a single-target, instant ability with the given gate
    /// and effect list. Most hand-authored test programs use this shape.
    pub fn new_single_target(
        range:   f32,
        gate:    Gate,
        effects: impl IntoIterator<Item = EffectOp>,
    ) -> Self {
        let mut v: SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        for e in effects { v.push(e); }
        Self {
            delivery: Delivery::Instant,
            area:     Area::SingleTarget { range },
            gate,
            effects:  v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_op_size_budget() {
        // 16-byte budget: largest payload is CastAbility at ~6 bytes plus a
        // u8 discriminant, padded to 16. Anything larger means we've
        // accidentally grown a variant — catch it at the first site.
        // (TransferGold was i64 pre-2026-04-22; now i32.)
        let sz = std::mem::size_of::<EffectOp>();
        assert!(sz <= 16, "EffectOp grew past 16B budget: {sz}");
    }

    #[test]
    fn construct_damage_program() {
        let p = AbilityProgram::new_single_target(
            6.0,
            Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        assert_eq!(p.effects.len(), 1);
        assert!(matches!(p.delivery, Delivery::Instant));
        assert!(matches!(p.area, Area::SingleTarget { range: 6.0 }));
    }

    #[test]
    fn construct_multi_effect_program() {
        let p = AbilityProgram::new_single_target(
            4.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 20.0 },
                EffectOp::Stun { duration_ticks: 5 },
            ],
        );
        assert_eq!(p.effects.len(), 2);
        match p.effects[1] {
            EffectOp::Stun { duration_ticks } => assert_eq!(duration_ticks, 5),
            _ => panic!("expected Stun at index 1"),
        }
    }

    #[test]
    fn construct_world_effect_program() {
        let p = AbilityProgram::new_single_target(
            3.0,
            Gate { cooldown_ticks: 100, hostile_only: false, line_of_sight: false },
            [
                EffectOp::TransferGold { amount: -50 },
                EffectOp::ModifyStanding { delta: -20 },
            ],
        );
        assert_eq!(p.effects.len(), 2);
        match p.effects[0] {
            EffectOp::TransferGold { amount } => assert_eq!(amount, -50),
            _ => panic!("expected TransferGold at index 0"),
        }
    }

    #[test]
    fn construct_recursive_chain_program() {
        let nested = AbilityId::new(7).unwrap();
        let p = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 1, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 10.0 },
                EffectOp::CastAbility { ability: nested, selector: TargetSelector::Target },
            ],
        );
        assert_eq!(p.effects.len(), 2);
        match p.effects[1] {
            EffectOp::CastAbility { ability, selector } => {
                assert_eq!(ability.raw(), 7);
                assert_eq!(selector, TargetSelector::Target);
            }
            _ => panic!("expected CastAbility at index 1"),
        }
    }
}
