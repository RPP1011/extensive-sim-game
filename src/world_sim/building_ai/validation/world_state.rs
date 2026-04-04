//! WorldState internal consistency checks (WS-* error codes).
//!
//! Validates that a generated WorldState is internally coherent:
//! grid-entity cross-references, grid bounds, entity invariants,
//! settlement invariants.

use std::collections::{HashMap, HashSet};

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::city_grid::{CellState, CellTerrain};
use crate::world_sim::state::{BuildingType, EntityKind, WorldState};

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

    // WS-ENT-005: dead entities not referenced by any grid cell
    let dead_ids: HashSet<u32> = state
        .entities
        .iter()
        .filter(|e| !e.alive)
        .map(|e| e.id)
        .collect();

    if !dead_ids.is_empty() {
        for grid in &state.city_grids {
            for (cell_idx, cell) in grid.cells.iter().enumerate() {
                if let Some(bid) = cell.building_id {
                    if dead_ids.contains(&bid) {
                        let col = (cell_idx % grid.cols) as u16;
                        let row = (cell_idx / grid.cols) as u16;
                        errors.push(ValidationError {
                            code: "WS-ENT-005",
                            severity: Severity::Fatal,
                            message: format!(
                                "Grid cell ({}, {}) references dead entity {}",
                                col, row, bid
                            ),
                            context: ErrorContext {
                                entity_id: Some(bid),
                                grid_cell: Some((col, row)),
                                ..Default::default()
                            },
                        });
                    }
                }
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// WS-GRID-*: Grid bounds and terrain
// ---------------------------------------------------------------------------

fn check_grid_invariants(state: &WorldState) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for (gi, grid) in state.city_grids.iter().enumerate() {
        // WS-GRID-001: cols * rows == cells.len()
        if grid.cols * grid.rows != grid.cells.len() {
            errors.push(ValidationError {
                code: "WS-GRID-001",
                severity: Severity::Fatal,
                message: format!(
                    "Grid {}: cols({}) * rows({}) = {} but cells.len() = {}",
                    gi,
                    grid.cols,
                    grid.rows,
                    grid.cols * grid.rows,
                    grid.cells.len()
                ),
                context: ErrorContext::default(),
            });
            // Can't do further grid checks if buffer is wrong size
            continue;
        }

        for (cell_idx, cell) in grid.cells.iter().enumerate() {
            let col = (cell_idx % grid.cols) as u16;
            let row = (cell_idx / grid.cols) as u16;

            // WS-GRID-002: no building on Water or Cliff terrain
            if (cell.state == CellState::Building || cell.state == CellState::Wall)
                && (cell.terrain == CellTerrain::Water || cell.terrain == CellTerrain::Cliff)
            {
                errors.push(ValidationError {
                    code: "WS-GRID-002",
                    severity: Severity::Fatal,
                    message: format!(
                        "Grid {}: building at ({}, {}) on unbuildable terrain {:?}",
                        gi, col, row, cell.terrain
                    ),
                    context: ErrorContext {
                        grid_cell: Some((col, row)),
                        ..Default::default()
                    },
                });
            }

            // WS-GRID-004: Building/Wall state with no building_id
            if (cell.state == CellState::Building || cell.state == CellState::Wall)
                && cell.building_id.is_none()
            {
                errors.push(ValidationError {
                    code: "WS-GRID-004",
                    severity: Severity::Fatal,
                    message: format!(
                        "Grid {}: cell ({}, {}) is {:?} but has no building_id",
                        gi, col, row, cell.state
                    ),
                    context: ErrorContext {
                        grid_cell: Some((col, row)),
                        ..Default::default()
                    },
                });
            }

            // WS-GRID-005: Empty state with building_id
            if cell.state == CellState::Empty && cell.building_id.is_some() {
                errors.push(ValidationError {
                    code: "WS-GRID-005",
                    severity: Severity::Fatal,
                    message: format!(
                        "Grid {}: cell ({}, {}) is Empty but has building_id {:?}",
                        gi, col, row, cell.building_id
                    ),
                    context: ErrorContext {
                        grid_cell: Some((col, row)),
                        entity_id: cell.building_id,
                        ..Default::default()
                    },
                });
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// WS-XREF-*: Grid-entity cross-reference
// ---------------------------------------------------------------------------

fn check_grid_entity_xref(state: &WorldState) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Build a map of entity_id -> &Entity for alive buildings
    let alive_buildings: HashMap<u32, &crate::world_sim::state::Entity> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .map(|e| (e.id, e))
        .collect();

    // WS-XREF-001: Every cell with building_id has a corresponding alive Building entity
    for (gi, grid) in state.city_grids.iter().enumerate() {
        if grid.cols * grid.rows != grid.cells.len() {
            continue; // Skip malformed grids (caught by WS-GRID-001)
        }

        // Track which building_ids appear at which cells for overlap detection
        let mut building_cells: HashMap<u32, Vec<(u16, u16)>> = HashMap::new();

        for (cell_idx, cell) in grid.cells.iter().enumerate() {
            let col = (cell_idx % grid.cols) as u16;
            let row = (cell_idx / grid.cols) as u16;

            if let Some(bid) = cell.building_id {
                building_cells
                    .entry(bid)
                    .or_default()
                    .push((col, row));

                if !alive_buildings.contains_key(&bid) {
                    errors.push(ValidationError {
                        code: "WS-XREF-001",
                        severity: Severity::Fatal,
                        message: format!(
                            "Grid {}: cell ({}, {}) references building_id {} but no alive Building entity exists",
                            gi, col, row, bid
                        ),
                        context: ErrorContext {
                            entity_id: Some(bid),
                            grid_cell: Some((col, row)),
                            ..Default::default()
                        },
                    });
                }

                // WS-XREF-005: cell state matches entity type
                if let Some(entity) = alive_buildings.get(&bid) {
                    if let Some(bdata) = &entity.building {
                        let expected_state = if bdata.building_type == BuildingType::Wall {
                            CellState::Wall
                        } else {
                            CellState::Building
                        };
                        if cell.state != expected_state {
                            errors.push(ValidationError {
                                code: "WS-XREF-005",
                                severity: Severity::Warning,
                                message: format!(
                                    "Grid {}: cell ({}, {}) state is {:?} but building {} is {:?} (expected {:?})",
                                    gi, col, row, cell.state, bid, bdata.building_type, expected_state
                                ),
                                context: ErrorContext {
                                    entity_id: Some(bid),
                                    grid_cell: Some((col, row)),
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }
        }

        // WS-XREF-003: no two alive building entities share the same grid cell
        // (detected via multiple different building_ids occupying same cell)
        // This is checked implicitly since each cell has one building_id.
        // But check for overlapping footprints: different building entities at same cell.
        let mut cell_owners: HashMap<(u16, u16), u32> = HashMap::new();
        for (cell_idx, cell) in grid.cells.iter().enumerate() {
            if let Some(bid) = cell.building_id {
                let col = (cell_idx % grid.cols) as u16;
                let row = (cell_idx / grid.cols) as u16;
                if let Some(&prev_bid) = cell_owners.get(&(col, row)) {
                    if prev_bid != bid {
                        errors.push(ValidationError {
                            code: "WS-XREF-003",
                            severity: Severity::Fatal,
                            message: format!(
                                "Grid {}: cell ({}, {}) claimed by both entity {} and {}",
                                gi, col, row, prev_bid, bid
                            ),
                            context: ErrorContext {
                                grid_cell: Some((col, row)),
                                ..Default::default()
                            },
                        });
                    }
                } else {
                    cell_owners.insert((col, row), bid);
                }
            }
        }
    }

    // WS-XREF-002: Every alive building entity with settlement_id has its grid position
    // pointing to a cell whose building_id == entity.id
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Building {
            continue;
        }
        let bdata = match &entity.building {
            Some(b) => b,
            None => continue,
        };
        let sid = match bdata.settlement_id {
            Some(s) => s,
            None => continue,
        };

        // Find the grid for this settlement
        let grid = state.city_grids.iter().find(|g| g.settlement_id == sid);
        let grid = match grid {
            Some(g) => g,
            None => continue, // No grid for this settlement (caught by WS-SET-003)
        };

        let col = bdata.grid_col as usize;
        let row = bdata.grid_row as usize;

        if !grid.in_bounds(col, row) {
            errors.push(ValidationError {
                code: "WS-XREF-002",
                severity: Severity::Fatal,
                message: format!(
                    "Building entity {} at ({}, {}) is out of bounds for grid ({}x{})",
                    entity.id, col, row, grid.cols, grid.rows
                ),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    grid_cell: Some((col as u16, row as u16)),
                    ..Default::default()
                },
            });
            continue;
        }

        let cell = &grid.cells[grid.idx(col, row)];
        if cell.building_id != Some(entity.id) {
            errors.push(ValidationError {
                code: "WS-XREF-002",
                severity: Severity::Fatal,
                message: format!(
                    "Building entity {} thinks it's at ({}, {}) but grid cell has building_id {:?}",
                    entity.id, col, row, cell.building_id
                ),
                context: ErrorContext {
                    entity_id: Some(entity.id),
                    grid_cell: Some((col as u16, row as u16)),
                    ..Default::default()
                },
            });
        }

        // WS-XREF-004: multi-cell footprints must have all covered cells in-bounds and marked
        if bdata.footprint_w > 1 || bdata.footprint_h > 1 {
            for dc in 0..bdata.footprint_w as usize {
                for dr in 0..bdata.footprint_h as usize {
                    let fc = col + dc;
                    let fr = row + dr;
                    if !grid.in_bounds(fc, fr) {
                        errors.push(ValidationError {
                            code: "WS-XREF-004",
                            severity: Severity::Fatal,
                            message: format!(
                                "Building {} footprint cell ({}, {}) is out of bounds",
                                entity.id, fc, fr
                            ),
                            context: ErrorContext {
                                entity_id: Some(entity.id),
                                grid_cell: Some((fc as u16, fr as u16)),
                                ..Default::default()
                            },
                        });
                        continue;
                    }
                    let fcell = &grid.cells[grid.idx(fc, fr)];
                    if fcell.building_id != Some(entity.id) {
                        errors.push(ValidationError {
                            code: "WS-XREF-004",
                            severity: Severity::Fatal,
                            message: format!(
                                "Building {} footprint cell ({}, {}) has building_id {:?} instead of {}",
                                entity.id, fc, fr, fcell.building_id, entity.id
                            ),
                            context: ErrorContext {
                                entity_id: Some(entity.id),
                                grid_cell: Some((fc as u16, fr as u16)),
                                ..Default::default()
                            },
                        });
                    }
                }
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// WS-SET-*: Settlement invariants
// ---------------------------------------------------------------------------

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

        // WS-SET-003: settlement has a city grid (required for building AI)
        match settlement.city_grid_idx {
            None => {
                errors.push(ValidationError {
                    code: "WS-SET-003",
                    severity: Severity::Warning,
                    message: format!(
                        "Settlement {} has no city_grid_idx",
                        settlement.id
                    ),
                    context: Default::default(),
                });
            }
            Some(idx) => {
                // WS-GRID-003: city_grid_idx valid and cross-linked
                if idx >= state.city_grids.len() {
                    errors.push(ValidationError {
                        code: "WS-GRID-003",
                        severity: Severity::Fatal,
                        message: format!(
                            "Settlement {} city_grid_idx {} >= city_grids.len() ({})",
                            settlement.id, idx, state.city_grids.len()
                        ),
                        context: Default::default(),
                    });
                } else if state.city_grids[idx].settlement_id != settlement.id {
                    errors.push(ValidationError {
                        code: "WS-GRID-003",
                        severity: Severity::Fatal,
                        message: format!(
                            "Settlement {} city_grid_idx {} points to grid with settlement_id {} (expected {})",
                            settlement.id, idx, state.city_grids[idx].settlement_id, settlement.id
                        ),
                        context: Default::default(),
                    });
                }
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::city_grid::{CellState, CityGrid};
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
    fn ghost_building_detected() {
        let mut state = minimal_state();
        let mut grid = CityGrid::new(3, 3, 1, "Plains", 0);
        // Mark a cell as Building with no building_id
        let cell = grid.cell_mut(1, 1);
        cell.state = CellState::Building;
        // No building_id set — ghost building
        state.city_grids.push(grid);

        let errors = validate_world_state(&state);
        assert!(
            errors.iter().any(|e| e.code == "WS-GRID-004"),
            "Expected WS-GRID-004 error"
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
