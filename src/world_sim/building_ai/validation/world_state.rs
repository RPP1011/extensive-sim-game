//! WorldState internal consistency checks (WS-* error codes).
//!
//! Validates that a generated WorldState is internally coherent:
//! grid-entity cross-references, grid bounds, entity invariants,
//! settlement invariants.

use std::collections::HashSet;

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::state::{EntityKind, WorldState};

/// Run all WorldState consistency checks. O(E + G + S).
pub fn validate_world_state(state: &WorldState) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    errors.extend(check_entity_invariants(state));
    errors.extend(check_grid_invariants(state));
    errors.extend(check_grid_entity_xref(state));
    errors.extend(check_settlement_invariants(state));
    errors
}

// ---------------------------------------------------------------------------
// WS-ENT-*: Entity invariants
// ---------------------------------------------------------------------------

fn check_entity_invariants(state: &WorldState) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut seen_ids: HashSet<u32> = HashSet::new();

    for entity in &state.entities {
        // WS-ENT-001: unique IDs
        if !seen_ids.insert(entity.id) {
            errors.push(ValidationError {
                code: "WS-ENT-001",
                severity: Severity::Fatal,
                message: format!("Duplicate entity ID {}", entity.id),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    ..Default::default()
                },
            });
        }

        // WS-ENT-002: Building entities have building data
        if entity.kind == EntityKind::Building && entity.building.is_none() {
            errors.push(ValidationError {
                code: "WS-ENT-002",
                severity: Severity::Fatal,
                message: format!("Building entity {} has no BuildingData", entity.id),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    ..Default::default()
                },
            });
        }

        // WS-ENT-003: NPC entities have npc data
        if entity.kind == EntityKind::Npc && entity.npc.is_none() {
            errors.push(ValidationError {
                code: "WS-ENT-003",
                severity: Severity::Fatal,
                message: format!("NPC entity {} has no NpcData", entity.id),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    ..Default::default()
                },
            });
        }

        // WS-ENT-004: HP sanity
        if entity.hp > entity.max_hp {
            errors.push(ValidationError {
                code: "WS-ENT-004",
                severity: Severity::Fatal,
                message: format!(
                    "Entity {} has hp ({}) > max_hp ({})",
                    entity.id, entity.hp, entity.max_hp
                ),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    field: Some("hp".into()),
                    ..Default::default()
                },
            });
        }
        if !entity.hp.is_finite() || !entity.max_hp.is_finite() {
            errors.push(ValidationError {
                code: "WS-ENT-004",
                severity: Severity::Fatal,
                message: format!(
                    "Entity {} has non-finite HP: hp={}, max_hp={}",
                    entity.id, entity.hp, entity.max_hp
                ),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    field: Some("hp".into()),
                    ..Default::default()
                },
            });
        }
        if entity.hp < 0.0 || entity.max_hp < 0.0 {
            errors.push(ValidationError {
                code: "WS-ENT-004",
                severity: Severity::Fatal,
                message: format!(
                    "Entity {} has negative HP: hp={}, max_hp={}",
                    entity.id, entity.hp, entity.max_hp
                ),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    field: Some("hp".into()),
                    ..Default::default()
                },
            });
        }

        // Building-specific checks
        if let Some(bdata) = &entity.building {
            // WS-ENT-007: construction_progress in [0.0, 1.0]
            if bdata.construction_progress < 0.0 || bdata.construction_progress > 1.0 {
                errors.push(ValidationError {
                    code: "WS-ENT-007",
                    severity: Severity::Fatal,
                    message: format!(
                        "Entity {} construction_progress {} out of [0.0, 1.0]",
                        entity.id, bdata.construction_progress
                    ),
                    context: ErrorContext {
                        entity_id: Some(entity.id),
                        field: Some("construction_progress".into()),
                        ..Default::default()
                    },
                });
            }

            // WS-ENT-008: storage values non-negative and finite
            for (i, &val) in bdata.storage.iter().enumerate() {
                if val < 0.0 || !val.is_finite() {
                    errors.push(ValidationError {
                        code: "WS-ENT-008",
                        severity: Severity::Fatal,
                        message: format!(
                            "Entity {} storage[{}] = {} (must be non-negative and finite)",
                            entity.id, i, val
                        ),
                        context: ErrorContext {
                            entity_id: Some(entity.id),
                            field: Some(format!("storage[{}]", i)),
                            ..Default::default()
                        },
                    });
                }
            }

            // WS-ENT-006: resident/worker IDs reference alive NPCs
            if entity.alive {
                for &rid in &bdata.resident_ids {
                    match state.entities.iter().find(|e| e.id == rid) {
                        None => {
                            errors.push(ValidationError {
                                code: "WS-ENT-006",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Building {} resident_id {} does not exist",
                                    entity.id, rid
                                ),
                                context: ErrorContext {
                                    entity_id: Some(entity.id),
                                    ..Default::default()
                                },
                            });
                        }
                        Some(e) if !e.alive => {
                            errors.push(ValidationError {
                                code: "WS-ENT-006",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Building {} resident_id {} is dead",
                                    entity.id, rid
                                ),
                                context: ErrorContext {
                                    entity_id: Some(entity.id),
                                    ..Default::default()
                                },
                            });
                        }
                        Some(e) if e.kind != EntityKind::Npc => {
                            errors.push(ValidationError {
                                code: "WS-ENT-006",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Building {} resident_id {} is not an NPC (kind={:?})",
                                    entity.id, rid, e.kind
                                ),
                                context: ErrorContext {
                                    entity_id: Some(entity.id),
                                    ..Default::default()
                                },
                            });
                        }
                        _ => {}
                    }
                }
                for &wid in &bdata.worker_ids {
                    match state.entities.iter().find(|e| e.id == wid) {
                        None => {
                            errors.push(ValidationError {
                                code: "WS-ENT-006",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Building {} worker_id {} does not exist",
                                    entity.id, wid
                                ),
                                context: ErrorContext {
                                    entity_id: Some(entity.id),
                                    ..Default::default()
                                },
                            });
                        }
                        Some(e) if !e.alive => {
                            errors.push(ValidationError {
                                code: "WS-ENT-006",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Building {} worker_id {} is dead",
                                    entity.id, wid
                                ),
                                context: ErrorContext {
                                    entity_id: Some(entity.id),
                                    ..Default::default()
                                },
                            });
                        }
                        Some(e) if e.kind != EntityKind::Npc => {
                            errors.push(ValidationError {
                                code: "WS-ENT-006",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Building {} worker_id {} is not an NPC (kind={:?})",
                                    entity.id, wid, e.kind
                                ),
                                context: ErrorContext {
                                    entity_id: Some(entity.id),
                                    ..Default::default()
                                },
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    errors
}


// ---------------------------------------------------------------------------
// WS-GRID-*: Grid invariants (CityGrid removed — stubs)
// ---------------------------------------------------------------------------

fn check_grid_invariants(_state: &WorldState) -> Vec<ValidationError> {
    // CityGrid has been removed. Grid invariant checks are no longer applicable.
    Vec::new()
}

// ---------------------------------------------------------------------------
// WS-GRID-ENT-*: Grid-entity cross-references (CityGrid removed — stubs)
// ---------------------------------------------------------------------------

fn check_grid_entity_xref(_state: &WorldState) -> Vec<ValidationError> {
    // CityGrid has been removed. Grid-entity cross-reference checks are no longer applicable.
    Vec::new()
}

fn check_settlement_invariants(state: &WorldState) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for settlement in &state.settlements {
        // WS-SET-001: stockpile values non-negative and finite
        for (i, &val) in settlement.stockpile.iter().enumerate() {
            if val < 0.0 || !val.is_finite() {
                errors.push(ValidationError {
                    code: "WS-SET-001",
                    severity: Severity::Fatal,
                    message: format!(
                        "Settlement {} stockpile[{}] = {} (must be non-negative and finite)",
                        settlement.id, i, val
                    ),
                    context: ErrorContext {
                        field: Some(format!("stockpile[{}]", i)),
                        ..Default::default()
                    },
                });
            }
        }

        // WS-SET-002: population count vs NPC entity count
        let npc_count = state
            .entities
            .iter()
            .filter(|e| {
                e.alive
                    && e.kind == EntityKind::Npc
                    && e.npc
                        .as_ref()
                        .map(|n| n.home_settlement_id == Some(settlement.id))
                        .unwrap_or(false)
            })
            .count() as u32;

        if settlement.population != npc_count {
            errors.push(ValidationError {
                code: "WS-SET-002",
                severity: Severity::Warning,
                message: format!(
                    "Settlement {} population field ({}) != alive NPC count ({})",
                    settlement.id, settlement.population, npc_count
                ),
                context: ErrorContext {
                    field: Some("population".into()),
                    ..Default::default()
                },
            });
        }

        // WS-SET-003: city_grid_idx check removed — CityGrid has been replaced by VoxelWorld.
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::{BuildingData, Entity, WorldState};

    fn minimal_state() -> WorldState {
        WorldState::new(42)
    }

    #[test]
    fn clean_empty_state_passes() {
        let state = minimal_state();
        let errors = validate_world_state(&state);
        let fatals: Vec<_> = errors.iter().filter(|e| e.severity == Severity::Fatal).collect();
        assert!(fatals.is_empty(), "Unexpected fatals: {:?}", fatals);
    }

    #[test]
    fn duplicate_entity_id_detected() {
        let mut state = minimal_state();
        let mut e1 = Entity::new_building(1, (0.0, 0.0));
        e1.building = Some(BuildingData::default());
        let mut e2 = Entity::new_building(1, (1.0, 1.0)); // same ID
        e2.building = Some(BuildingData::default());
        state.entities.push(e1);
        state.entities.push(e2);

        let errors = validate_world_state(&state);
        assert!(
            errors.iter().any(|e| e.code == "WS-ENT-001"),
            "Expected WS-ENT-001 error"
        );
    }

    #[test]
    fn hp_exceeds_max_detected() {
        let mut state = minimal_state();
        let mut e = Entity::new_building(1, (0.0, 0.0));
        e.hp = 600.0;
        e.max_hp = 500.0;
        e.building = Some(BuildingData::default());
        state.entities.push(e);

        let errors = validate_world_state(&state);
        assert!(
            errors.iter().any(|e| e.code == "WS-ENT-004"),
            "Expected WS-ENT-004 error"
        );
    }

    #[test]
    fn negative_stockpile_detected() {
        let mut state = minimal_state();
        let mut settlement =
            crate::world_sim::state::SettlementState::new(1, "Test".into(), (0.0, 0.0));
        settlement.stockpile[0] = -10.0;
        state.settlements.push(settlement);

        let errors = validate_world_state(&state);
        assert!(
            errors.iter().any(|e| e.code == "WS-SET-001"),
            "Expected WS-SET-001 error"
        );
    }
}
