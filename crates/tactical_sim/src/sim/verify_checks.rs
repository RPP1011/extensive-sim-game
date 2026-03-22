use std::collections::HashSet;

use crate::effects::StatusKind;

use super::types::SimState;
use super::verify::Violation;

/// Check unit-level invariants: HP, casting, channeling, shields, resources,
/// position, move speed, cover, directed/owner, armor/MR, cooldowns, ability slots.
pub(crate) fn check_units(state: &SimState, unit_ids: &HashSet<u32>, alive_ids: &HashSet<u32>, violations: &mut Vec<Violation>) {
    for unit in &state.units {
        // HP bounds
        if unit.hp > unit.max_hp {
            violations.push(Violation::HpExceedsMax {
                unit_id: unit.id,
                hp: unit.hp,
                max_hp: unit.max_hp,
            });
        }

        let alive = unit.hp > 0;

        // Dead unit still casting
        if !alive && unit.casting.is_some() {
            violations.push(Violation::DeadUnitCasting { unit_id: unit.id });
        }

        // Dead unit still channeling
        if !alive && unit.channeling.is_some() {
            violations.push(Violation::DeadUnitChanneling { unit_id: unit.id });
        }

        // Dead unit still under crowd control
        if !alive && unit.control_remaining_ms > 0 {
            violations.push(Violation::DeadUnitControlled {
                unit_id: unit.id,
                control_remaining_ms: unit.control_remaining_ms,
            });
        }

        // Casting and channeling at the same time
        if unit.casting.is_some() && unit.channeling.is_some() {
            violations.push(Violation::CastingAndChanneling { unit_id: unit.id });
        }

        // Negative shield
        if unit.shield_hp < 0 {
            violations.push(Violation::NegativeShield {
                unit_id: unit.id,
                shield_hp: unit.shield_hp,
            });
        }

        // Resource bounds
        if unit.max_resource > 0 && unit.resource > unit.max_resource {
            violations.push(Violation::ResourceExceedsMax {
                unit_id: unit.id,
                resource: unit.resource,
                max_resource: unit.max_resource,
            });
        }
        if unit.resource < 0 {
            violations.push(Violation::NegativeResource {
                unit_id: unit.id,
                resource: unit.resource,
            });
        }

        // Position validity (NaN / Inf)
        if !unit.position.x.is_finite() || !unit.position.y.is_finite() {
            violations.push(Violation::InvalidPosition {
                unit_id: unit.id,
                x: unit.position.x,
                y: unit.position.y,
            });
        }

        // Negative move speed
        if unit.move_speed_per_sec < 0.0 {
            violations.push(Violation::NegativeMoveSpeed {
                unit_id: unit.id,
                speed: unit.move_speed_per_sec,
            });
        }

        // Cover bonus bounds
        if unit.cover_bonus < 0.0 || unit.cover_bonus > 1.0 {
            violations.push(Violation::CoverBonusOutOfRange {
                unit_id: unit.id,
                cover_bonus: (unit.cover_bonus * 1000.0) as i32,
            });
        }

        // Directed summon must have an owner
        if unit.directed && unit.owner_id.is_none() {
            violations.push(Violation::DirectedWithoutOwner { unit_id: unit.id });
        }

        // Summon owner must exist
        if let Some(owner_id) = unit.owner_id {
            if !unit_ids.contains(&owner_id) {
                violations.push(Violation::SummonOwnerMissing {
                    unit_id: unit.id,
                    owner_id,
                });
            }
        }

        // Negative armor / magic resist
        if unit.armor < 0.0 {
            violations.push(Violation::NegativeResist {
                unit_id: unit.id,
                field: "armor",
                value_x100: (unit.armor * 100.0) as i32,
            });
        }
        if unit.magic_resist < 0.0 {
            violations.push(Violation::NegativeResist {
                unit_id: unit.id,
                field: "magic_resist",
                value_x100: (unit.magic_resist * 100.0) as i32,
            });
        }

        // Cooldown sanity: remaining should not exceed base
        if unit.attack_cooldown_ms > 0
            && unit.cooldown_remaining_ms > unit.attack_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "attack",
                remaining: unit.cooldown_remaining_ms,
                base: unit.attack_cooldown_ms,
            });
        }
        if unit.ability_cooldown_ms > 0
            && unit.ability_cooldown_remaining_ms > unit.ability_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "ability",
                remaining: unit.ability_cooldown_remaining_ms,
                base: unit.ability_cooldown_ms,
            });
        }
        if unit.heal_cooldown_ms > 0
            && unit.heal_cooldown_remaining_ms > unit.heal_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "heal",
                remaining: unit.heal_cooldown_remaining_ms,
                base: unit.heal_cooldown_ms,
            });
        }
        if unit.control_cooldown_ms > 0
            && unit.control_cooldown_remaining_ms > unit.control_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "control",
                remaining: unit.control_cooldown_remaining_ms,
                base: unit.control_cooldown_ms,
            });
        }

        // --- Hero ability slot invariants ---
        for (i, slot) in unit.abilities.iter().enumerate() {
            if slot.def.max_charges > 0 && slot.charges > slot.def.max_charges {
                violations.push(Violation::AbilityChargesExceedMax {
                    unit_id: unit.id,
                    ability_index: i,
                    charges: slot.charges,
                    max_charges: slot.def.max_charges,
                });
            }
            if slot.morph_remaining_ms > 0 && slot.base_def.is_none() {
                violations.push(Violation::MorphWithoutBaseDef {
                    unit_id: unit.id,
                    ability_index: i,
                });
            }
            if slot.recast_window_remaining_ms > 0 && slot.recasts_remaining == 0 {
                violations.push(Violation::RecastWindowWithoutRecasts {
                    unit_id: unit.id,
                    ability_index: i,
                });
            }
        }

        // --- Status effect reference validation ---
        check_status_effect_refs(unit.id, &unit.status_effects, unit_ids, alive_ids, violations);
    }
}

/// Check status effect references for a single unit.
fn check_status_effect_refs(
    unit_id: u32,
    effects: &[crate::effects::ActiveStatusEffect],
    unit_ids: &HashSet<u32>,
    alive_ids: &HashSet<u32>,
    violations: &mut Vec<Violation>,
) {
    for effect in effects {
        match &effect.kind {
            StatusKind::Duel { partner_id } => {
                if !unit_ids.contains(partner_id) {
                    violations.push(Violation::StatusEffectOrphanedRef {
                        unit_id,
                        kind: "Duel",
                        referenced_id: *partner_id,
                    });
                }
            }
            StatusKind::Taunt { taunter_id } => {
                if !unit_ids.contains(taunter_id) {
                    violations.push(Violation::StatusEffectOrphanedRef {
                        unit_id,
                        kind: "Taunt",
                        referenced_id: *taunter_id,
                    });
                } else if !alive_ids.contains(taunter_id) {
                    violations.push(Violation::StatusEffectDeadRef {
                        unit_id,
                        kind: "Taunt",
                        referenced_id: *taunter_id,
                    });
                }
            }
            StatusKind::Link { partner_id, .. } => {
                if !unit_ids.contains(partner_id) {
                    violations.push(Violation::StatusEffectOrphanedRef {
                        unit_id,
                        kind: "Link",
                        referenced_id: *partner_id,
                    });
                } else if !alive_ids.contains(partner_id) {
                    violations.push(Violation::StatusEffectDeadRef {
                        unit_id,
                        kind: "Link",
                        referenced_id: *partner_id,
                    });
                }
            }
            StatusKind::Redirect { protector_id, .. } => {
                if !unit_ids.contains(protector_id) {
                    violations.push(Violation::StatusEffectOrphanedRef {
                        unit_id,
                        kind: "Redirect",
                        referenced_id: *protector_id,
                    });
                } else if !alive_ids.contains(protector_id) {
                    violations.push(Violation::StatusEffectDeadRef {
                        unit_id,
                        kind: "Redirect",
                        referenced_id: *protector_id,
                    });
                }
            }
            StatusKind::Attached { host_id } => {
                if !unit_ids.contains(host_id) {
                    violations.push(Violation::StatusEffectOrphanedRef {
                        unit_id,
                        kind: "Attached",
                        referenced_id: *host_id,
                    });
                } else if !alive_ids.contains(host_id) {
                    violations.push(Violation::StatusEffectDeadRef {
                        unit_id,
                        kind: "Attached",
                        referenced_id: *host_id,
                    });
                }
            }
            StatusKind::Charm { .. } => {
                if !unit_ids.contains(&effect.source_id) {
                    violations.push(Violation::StatusEffectOrphanedRef {
                        unit_id,
                        kind: "Charm",
                        referenced_id: effect.source_id,
                    });
                }
            }
            _ => {}
        }
    }
}

/// Check projectile invariants.
pub(crate) fn check_projectiles(state: &SimState, unit_ids: &HashSet<u32>, violations: &mut Vec<Violation>) {
    for (i, proj) in state.projectiles.iter().enumerate() {
        if !unit_ids.contains(&proj.source_id) {
            violations.push(Violation::ProjectileOrphanedSource {
                projectile_index: i,
                source_id: proj.source_id,
            });
        }
        if !proj.position.x.is_finite() || !proj.position.y.is_finite() {
            violations.push(Violation::ProjectileInvalidPosition {
                projectile_index: i,
                x: proj.position.x,
                y: proj.position.y,
            });
        }
        if proj.speed <= 0.0 {
            violations.push(Violation::ProjectileInvalidSpeed {
                projectile_index: i,
                speed: proj.speed,
            });
        }
        if proj.max_travel_distance > 0.0
            && proj.distance_traveled > proj.max_travel_distance + 1.0
        {
            violations.push(Violation::ProjectileOvershot {
                projectile_index: i,
                distance_traveled_x100: (proj.distance_traveled * 100.0) as i32,
                max_distance_x100: (proj.max_travel_distance * 100.0) as i32,
            });
        }
    }
}

/// Check status effect value ranges and zone/tether invariants.
pub(crate) fn check_status_values_and_world(state: &SimState, unit_ids: &HashSet<u32>, violations: &mut Vec<Violation>) {
    // Status effect value-range checks
    for unit in &state.units {
        for effect in &unit.status_effects {
            match &effect.kind {
                StatusKind::Slow { factor } => {
                    if *factor < 0.0 || *factor > 1.0 {
                        violations.push(Violation::StatusEffectValueOutOfRange {
                            unit_id: unit.id,
                            kind: "Slow",
                            value_x1000: (*factor * 1000.0) as i32,
                        });
                    }
                }
                StatusKind::Blind { miss_chance } => {
                    if *miss_chance < 0.0 || *miss_chance > 1.0 {
                        violations.push(Violation::StatusEffectValueOutOfRange {
                            unit_id: unit.id,
                            kind: "Blind",
                            value_x1000: (*miss_chance * 1000.0) as i32,
                        });
                    }
                }
                StatusKind::Stacks { count, max_stacks, .. } => {
                    if *max_stacks > 0 && *count > *max_stacks {
                        violations.push(Violation::StatusEffectStacksExceedMax {
                            unit_id: unit.id,
                            count: *count,
                            max_stacks: *max_stacks,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // Zone source validation
    for zone in &state.zones {
        if !unit_ids.contains(&zone.source_id) {
            violations.push(Violation::ZoneOrphanedSource {
                zone_id: zone.id,
                source_id: zone.source_id,
            });
        }
        if zone.remaining_ms == 0 && !zone.trigger_on_enter {
            violations.push(Violation::ZoneExpired { zone_id: zone.id });
        }
    }

    // Tether endpoint validation
    for tether in &state.tethers {
        if !unit_ids.contains(&tether.source_id) || !unit_ids.contains(&tether.target_id) {
            violations.push(Violation::TetherOrphanedUnit {
                source_id: tether.source_id,
                target_id: tether.target_id,
            });
        }
        if tether.remaining_ms == 0 {
            violations.push(Violation::TetherExpired {
                source_id: tether.source_id,
                target_id: tether.target_id,
            });
        }
    }
}

