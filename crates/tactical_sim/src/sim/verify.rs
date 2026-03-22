use std::collections::HashSet;

use super::types::SimState;
use super::verify_checks::{check_units, check_projectiles, check_status_values_and_world};

/// A specific invariant violation detected during a simulation tick.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    /// Unit HP exceeds its max_hp.
    HpExceedsMax { unit_id: u32, hp: i32, max_hp: i32 },
    /// Alive unit has non-positive HP (should have been marked dead).
    AliveWithNonPositiveHp { unit_id: u32, hp: i32 },
    /// Dead unit (hp <= 0) still has an active cast.
    DeadUnitCasting { unit_id: u32 },
    /// Dead unit still has an active channel.
    DeadUnitChanneling { unit_id: u32 },
    /// Shield HP is negative.
    NegativeShield { unit_id: u32, shield_hp: i32 },
    /// Resource exceeds max_resource.
    ResourceExceedsMax { unit_id: u32, resource: i32, max_resource: i32 },
    /// Position contains NaN or infinity.
    InvalidPosition { unit_id: u32, x: f32, y: f32 },
    /// Duplicate unit IDs detected.
    DuplicateUnitId { unit_id: u32 },
    /// Zone references a source unit that doesn't exist.
    ZoneOrphanedSource { zone_id: u32, source_id: u32 },
    /// Tether references a unit that doesn't exist.
    TetherOrphanedUnit { source_id: u32, target_id: u32 },
    /// Move speed is negative.
    NegativeMoveSpeed { unit_id: u32, speed: f32 },
    /// Cooldown remaining exceeds the base cooldown value.
    CooldownExceedsBase { unit_id: u32, field: &'static str, remaining: u32, base: u32 },
    /// Negative resource value.
    NegativeResource { unit_id: u32, resource: i32 },

    // --- New runtime verification checks ---

    /// Unit is simultaneously casting and channeling.
    CastingAndChanneling { unit_id: u32 },
    /// Dead unit still has active crowd-control timer.
    DeadUnitControlled { unit_id: u32, control_remaining_ms: u32 },
    /// Ability slot charges exceed max_charges.
    AbilityChargesExceedMax { unit_id: u32, ability_index: usize, charges: u32, max_charges: u32 },
    /// Ability has an active morph timer but no saved base definition.
    MorphWithoutBaseDef { unit_id: u32, ability_index: usize },
    /// Recast window is active but no recasts remain.
    RecastWindowWithoutRecasts { unit_id: u32, ability_index: usize },
    /// Directed summon without an owner.
    DirectedWithoutOwner { unit_id: u32 },
    /// Status effect references a unit that doesn't exist.
    StatusEffectOrphanedRef { unit_id: u32, kind: &'static str, referenced_id: u32 },
    /// Status effect references a dead unit (partner/protector/host is dead).
    StatusEffectDeadRef { unit_id: u32, kind: &'static str, referenced_id: u32 },
    /// Cover bonus is outside the valid range [0.0, 1.0].
    CoverBonusOutOfRange { unit_id: u32, cover_bonus: i32 },
    /// Zone remaining duration is zero but zone still exists.
    ZoneExpired { zone_id: u32 },
    /// Tether remaining duration is zero but tether still exists.
    TetherExpired { source_id: u32, target_id: u32 },

    // --- Projectile invariants ---

    /// Projectile source unit doesn't exist.
    ProjectileOrphanedSource { projectile_index: usize, source_id: u32 },
    /// Projectile position contains NaN or infinity.
    ProjectileInvalidPosition { projectile_index: usize, x: f32, y: f32 },
    /// Projectile speed is zero or negative.
    ProjectileInvalidSpeed { projectile_index: usize, speed: f32 },
    /// Projectile has traveled past its max distance.
    ProjectileOvershot { projectile_index: usize, distance_traveled_x100: i32, max_distance_x100: i32 },

    // --- Status effect value-range checks ---

    /// Status effect numeric parameter is outside expected bounds.
    StatusEffectValueOutOfRange { unit_id: u32, kind: &'static str, value_x1000: i32 },
    /// Stacks count exceeds max_stacks.
    StatusEffectStacksExceedMax { unit_id: u32, count: u32, max_stacks: u32 },
    /// Summon owner doesn't exist in the unit list.
    SummonOwnerMissing { unit_id: u32, owner_id: u32 },
    /// Negative armor or magic resist value.
    NegativeResist { unit_id: u32, field: &'static str, value_x100: i32 },
}

/// Result of running verification checks on a simulation state.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub tick: u64,
    pub violations: Vec<Violation>,
}

impl VerificationReport {
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// Run all invariant checks against the current simulation state.
/// Call this after each `step()` to detect violations early.
pub fn verify_tick(state: &SimState) -> VerificationReport {
    let mut violations = Vec::new();
    let unit_ids: HashSet<u32> = state.units.iter().map(|u| u.id).collect();

    // Build alive-unit set for status-effect reference checks
    let alive_ids: HashSet<u32> = state.units.iter()
        .filter(|u| u.hp > 0)
        .map(|u| u.id)
        .collect();

    // Check for duplicate unit IDs
    if unit_ids.len() != state.units.len() {
        let mut seen = HashSet::new();
        for u in &state.units {
            if !seen.insert(u.id) {
                violations.push(Violation::DuplicateUnitId { unit_id: u.id });
            }
        }
    }

    check_units(state, &unit_ids, &alive_ids, &mut violations);
    check_projectiles(state, &unit_ids, &mut violations);
    check_status_values_and_world(state, &unit_ids, &mut violations);

    VerificationReport {
        tick: state.tick,
        violations,
    }
}
