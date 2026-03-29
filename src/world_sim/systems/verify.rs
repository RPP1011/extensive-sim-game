#![allow(unused)]
//! Invariant verification — every tick (no deltas produced).
//!
//! Checks world state consistency. Returns violations as a list but does
//! not emit deltas. This is a read-only diagnostic system.
//!
//! Original: `crates/headless_campaign/src/systems/verify.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Verify world state invariants. This system produces no deltas; it is
/// intended for debug/diagnostic use. Call `verify_invariants` directly
/// to get the list of violations.
pub fn compute_verify(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Verification is a read-only check — no deltas emitted.
    // Call verify_invariants() for the violation list.
}

/// Check world state consistency. Returns a list of violation descriptions.
pub fn verify_invariants(state: &WorldState) -> Vec<String> {
    let mut violations = Vec::new();

    // Entity checks
    let mut entity_ids = std::collections::HashSet::new();
    for entity in &state.entities {
        if !entity_ids.insert(entity.id) {
            violations.push(format!("Duplicate entity ID: {}", entity.id));
        }
        if entity.alive {
            if entity.hp < 0.0 {
                violations.push(format!(
                    "Entity {} has negative HP: {}",
                    entity.id, entity.hp
                ));
            }
            if entity.hp > entity.max_hp * 1.5 {
                violations.push(format!(
                    "Entity {} HP ({}) exceeds 1.5x max_hp ({})",
                    entity.id, entity.hp, entity.max_hp
                ));
            }
        }
    }

    // Settlement checks
    let mut settlement_ids = std::collections::HashSet::new();
    for settlement in &state.settlements {
        if !settlement_ids.insert(settlement.id) {
            violations.push(format!("Duplicate settlement ID: {}", settlement.id));
        }
        if settlement.treasury < -100.0 {
            violations.push(format!(
                "Settlement {} treasury severely negative: {}",
                settlement.id, settlement.treasury
            ));
        }
        for (i, &stock) in settlement.stockpile.iter().enumerate() {
            if stock < 0.0 {
                violations.push(format!(
                    "Settlement {} commodity {} stockpile negative: {}",
                    settlement.id, i, stock
                ));
            }
        }
    }

    // Grid checks
    for grid in &state.grids {
        for &entity_id in &grid.entity_ids {
            if state.entity(entity_id).is_none() {
                violations.push(format!(
                    "Grid {} references nonexistent entity {}",
                    grid.id, entity_id
                ));
            }
        }
    }

    // Region checks
    for region in &state.regions {
        if region.threat_level < 0.0 || region.threat_level > 200.0 {
            violations.push(format!(
                "Region {} threat_level out of range: {}",
                region.id, region.threat_level
            ));
        }
    }

    violations
}
