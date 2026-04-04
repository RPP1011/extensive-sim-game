//! Action validity checks (ACT-PRE-* and ACT-POST-* error codes).
//!
//! Pre-application checks validate that oracle-produced actions are applicable
//! to the current WorldState. Post-application checks confirm the action
//! produced a valid successor state.

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::building_ai::types::{
    ActionPayload, BuildingAction, BuildingObservation,
};
use crate::world_sim::city_grid::{CellState, CellTerrain};
use crate::world_sim::state::{BuildingType, EntityKind, WorldState};

// ---------------------------------------------------------------------------
// ACT-PRE-*: Pre-application checks
// ---------------------------------------------------------------------------

/// Validate all actions against the current state before application.
pub fn validate_action_batch(
    state: &WorldState,
    obs: &BuildingObservation,
    actions: &[BuildingAction],
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Find the grid for this observation's settlement
    let grid = find_grid(state, obs.settlement_id);

    for (ai, action) in actions.iter().enumerate() {
        match &action.action {
            ActionPayload::PlaceBuilding {
                building_type,
                grid_cell,
            } => {
                errors.extend(check_place_building(
                    state,
                    grid,
                    *building_type,
                    *grid_cell,
                    obs,
                    ai,
                ));
            }
            ActionPayload::Demolish { building_id } => {
                errors.extend(check_entity_ref(
                    state, *building_id, "ACT-PRE-006", ai,
                ));
            }
            ActionPayload::SetBuildPriority { building_id, .. } => {
                errors.extend(check_entity_ref(
                    state, *building_id, "ACT-PRE-007", ai,
                ));
            }
            ActionPayload::SetFootprint {
                building_id,
                dimensions,
                ..
            } => {
                errors.extend(check_entity_ref(
                    state, *building_id, "ACT-PRE-007", ai,
                ));
                // ACT-PRE-012: dimensions > 0
                if dimensions.0 == 0 || dimensions.1 == 0 {
                    errors.push(ValidationError {
                        code: "ACT-PRE-012",
                        severity: Severity::Fatal,
                        message: format!(
                            "Action {}: SetFootprint dimensions ({}, {}) must be > 0",
                            ai, dimensions.0, dimensions.1
                        ),
                        context: ErrorContext {
                            entity_id: Some(*building_id),
                            ..Default::default()
                        },
                    });
                }
                // Check footprint fits within grid bounds
                if let Some(g) = grid {
                    if let Some(entity) = state.entities.iter().find(|e| e.id == *building_id) {
                        if let Some(bdata) = &entity.building {
                            let end_col = bdata.grid_col as usize + dimensions.0 as usize;
                            let end_row = bdata.grid_row as usize + dimensions.1 as usize;
                            if end_col > g.cols || end_row > g.rows {
                                errors.push(ValidationError {
                                    code: "ACT-PRE-012",
                                    severity: Severity::Fatal,
                                    message: format!(
                                        "Action {}: SetFootprint from ({}, {}) with dims ({}, {}) overflows grid ({}x{})",
                                        ai, bdata.grid_col, bdata.grid_row,
                                        dimensions.0, dimensions.1, g.cols, g.rows
                                    ),
                                    context: ErrorContext {
                                        entity_id: Some(*building_id),
                                        ..Default::default()
                                    },
                                });
                            }
                        }
                    }
                }
            }
            ActionPayload::SetVertical {
                building_id,
                stories,
                ..
            } => {
                errors.extend(check_entity_ref(
                    state, *building_id, "ACT-PRE-007", ai,
                ));
                // ACT-PRE-011: stories > 0
                if *stories == 0 {
                    errors.push(ValidationError {
                        code: "ACT-PRE-011",
                        severity: Severity::Fatal,
                        message: format!("Action {}: SetVertical stories must be > 0", ai),
                        context: ErrorContext {
                            entity_id: Some(*building_id),
                            ..Default::default()
                        },
                    });
                }
            }
            ActionPayload::SetWallSpec {
                segment_id,
                height,
                thickness,
                ..
            } => {
                // ACT-PRE-010: height and thickness > 0
                if *height == 0 {
                    errors.push(ValidationError {
                        code: "ACT-PRE-010",
                        severity: Severity::Fatal,
                        message: format!("Action {}: SetWallSpec height must be > 0", ai),
                        context: ErrorContext {
                            entity_id: Some(*segment_id),
                            ..Default::default()
                        },
                    });
                }
                if *thickness == 0 {
                    errors.push(ValidationError {
                        code: "ACT-PRE-010",
                        severity: Severity::Fatal,
                        message: format!("Action {}: SetWallSpec thickness must be > 0", ai),
                        context: ErrorContext {
                            entity_id: Some(*segment_id),
                            ..Default::default()
                        },
                    });
                }
            }
            ActionPayload::SetRoofSpec { building_id, .. }
            | ActionPayload::SetFoundation { building_id, .. }
            | ActionPayload::SetOpenings { building_id, .. }
            | ActionPayload::SetInteriorLayout { building_id, .. }
            | ActionPayload::SetMaterial { building_id, .. }
            | ActionPayload::Renovate { building_id, .. } => {
                errors.extend(check_entity_ref(
                    state, *building_id, "ACT-PRE-007", ai,
                ));
            }
            ActionPayload::RouteRoad { waypoints } => {
                // ACT-PRE-008: all waypoints in bounds
                if let Some(g) = grid {
                    for (wi, wp) in waypoints.iter().enumerate() {
                        if !g.in_bounds(wp.0 as usize, wp.1 as usize) {
                            errors.push(ValidationError {
                                code: "ACT-PRE-008",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Action {}: RouteRoad waypoint {} ({}, {}) out of bounds ({}x{})",
                                    ai, wi, wp.0, wp.1, g.cols, g.rows
                                ),
                                context: ErrorContext {
                                    grid_cell: Some(*wp),
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }
            ActionPayload::SetZone { grid_cell, .. } => {
                // ACT-PRE-009: grid_cell in bounds
                if let Some(g) = grid {
                    if !g.in_bounds(grid_cell.0 as usize, grid_cell.1 as usize) {
                        errors.push(ValidationError {
                            code: "ACT-PRE-009",
                            severity: Severity::Fatal,
                            message: format!(
                                "Action {}: SetZone grid_cell ({}, {}) out of bounds ({}x{})",
                                ai, grid_cell.0, grid_cell.1, g.cols, g.rows
                            ),
                            context: ErrorContext {
                                grid_cell: Some(*grid_cell),
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

/// Validate PlaceBuilding action specifics.
fn check_place_building(
    state: &WorldState,
    grid: Option<&crate::world_sim::city_grid::CityGrid>,
    building_type: BuildingType,
    grid_cell: (u16, u16),
    obs: &BuildingObservation,
    action_idx: usize,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let g = match grid {
        Some(g) => g,
        None => {
            errors.push(ValidationError {
                code: "ACT-PRE-001",
                severity: Severity::Fatal,
                message: format!(
                    "Action {}: PlaceBuilding but settlement {} has no grid",
                    action_idx, obs.settlement_id
                ),
                context: ErrorContext {
                    grid_cell: Some(grid_cell),
                    ..Default::default()
                },
            });
            return errors;
        }
    };

    let (c, r) = (grid_cell.0 as usize, grid_cell.1 as usize);

    // ACT-PRE-001: in bounds
    if !g.in_bounds(c, r) {
        errors.push(ValidationError {
            code: "ACT-PRE-001",
            severity: Severity::Fatal,
            message: format!(
                "Action {}: PlaceBuilding at ({}, {}) out of bounds ({}x{})",
                action_idx, c, r, g.cols, g.rows
            ),
            context: ErrorContext {
                grid_cell: Some(grid_cell),
                ..Default::default()
            },
        });
        return errors;
    }

    let cell = &g.cells[g.idx(c, r)];

    // ACT-PRE-002: target cell must be Empty or Road
    match cell.state {
        CellState::Empty | CellState::Road => {}
        _ => {
            errors.push(ValidationError {
                code: "ACT-PRE-002",
                severity: Severity::Fatal,
                message: format!(
                    "Action {}: PlaceBuilding at ({}, {}) but cell state is {:?} (must be Empty or Road)",
                    action_idx, c, r, cell.state
                ),
                context: ErrorContext {
                    grid_cell: Some(grid_cell),
                    ..Default::default()
                },
            });
        }
    }

    // ACT-PRE-003: terrain not Water or Cliff
    if cell.terrain == CellTerrain::Water || cell.terrain == CellTerrain::Cliff {
        errors.push(ValidationError {
            code: "ACT-PRE-003",
            severity: Severity::Fatal,
            message: format!(
                "Action {}: PlaceBuilding at ({}, {}) on unbuildable terrain {:?}",
                action_idx, c, r, cell.terrain
            ),
            context: ErrorContext {
                grid_cell: Some(grid_cell),
                ..Default::default()
            },
        });
    }

    // ACT-PRE-004: resource feasibility (Warning until cost table exists)
    // Stub: emit warning for any building if stockpile is completely empty
    let total_resources: f32 = state
        .settlements
        .iter()
        .find(|s| s.id == obs.settlement_id)
        .map(|s| s.stockpile.iter().sum())
        .unwrap_or(0.0);
    if total_resources <= 0.0 {
        errors.push(ValidationError {
            code: "ACT-PRE-004",
            severity: Severity::Warning,
            message: format!(
                "Action {}: PlaceBuilding {:?} with zero total stockpile resources",
                action_idx, building_type
            ),
            context: ErrorContext {
                grid_cell: Some(grid_cell),
                ..Default::default()
            },
        });
    }

    // ACT-PRE-005: tech tier allows building type (Warning until tier table exists)
    // Stub: Iron/reinforced buildings require tech_tier >= 3
    let requires_high_tier = matches!(
        building_type,
        BuildingType::GuildHall | BuildingType::CourtHouse | BuildingType::Library
    );
    if requires_high_tier && obs.tech_tier < 3 {
        errors.push(ValidationError {
            code: "ACT-PRE-005",
            severity: Severity::Warning,
            message: format!(
                "Action {}: {:?} may require tech_tier >= 3 (current: {})",
                action_idx, building_type, obs.tech_tier
            ),
            context: ErrorContext {
                grid_cell: Some(grid_cell),
                ..Default::default()
            },
        });
    }

    errors
}

/// Check that an entity ID references an alive building entity.
fn check_entity_ref(
    state: &WorldState,
    building_id: u32,
    code: &'static str,
    action_idx: usize,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    match state.entities.iter().find(|e| e.id == building_id) {
        None => {
            errors.push(ValidationError {
                code,
                severity: Severity::Fatal,
                message: format!(
                    "Action {}: targets entity {} which does not exist",
                    action_idx, building_id
                ),
                context: ErrorContext {
                    entity_id: Some(building_id),
                    ..Default::default()
                },
            });
        }
        Some(e) if !e.alive => {
            errors.push(ValidationError {
                code,
                severity: Severity::Fatal,
                message: format!(
                    "Action {}: targets entity {} which is dead",
                    action_idx, building_id
                ),
                context: ErrorContext {
                    entity_id: Some(building_id),
                    ..Default::default()
                },
            });
        }
        Some(e) if e.kind != EntityKind::Building => {
            errors.push(ValidationError {
                code,
                severity: Severity::Fatal,
                message: format!(
                    "Action {}: targets entity {} which is not a Building (kind={:?})",
                    action_idx, building_id, e.kind
                ),
                context: ErrorContext {
                    entity_id: Some(building_id),
                    ..Default::default()
                },
            });
        }
        _ => {}
    }

    errors
}

// ---------------------------------------------------------------------------
// ACT-POST-*: Post-application checks
// ---------------------------------------------------------------------------

/// Validate the state transition after applying actions.
pub fn validate_post_action(
    pre_state: &WorldState,
    post_state: &WorldState,
    actions: &[BuildingAction],
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // ACT-POST-002: entity count did not decrease (except via Demolish)
    let demolish_count = actions
        .iter()
        .filter(|a| matches!(a.action, ActionPayload::Demolish { .. }))
        .count();
    // Alive entity count shouldn't decrease by more than demolish actions
    let pre_alive = pre_state.entities.iter().filter(|e| e.alive).count();
    let post_alive = post_state.entities.iter().filter(|e| e.alive).count();
    if post_alive + demolish_count < pre_alive {
        errors.push(ValidationError {
            code: "ACT-POST-002",
            severity: Severity::Fatal,
            message: format!(
                "Entity count dropped unexpectedly: {} alive before, {} after, {} demolishes",
                pre_alive, post_alive, demolish_count
            ),
            context: ErrorContext::default(),
        });
    }

    // ACT-POST-003: PlaceBuilding created the expected entity
    for action in actions {
        if let ActionPayload::PlaceBuilding {
            building_type,
            grid_cell,
        } = &action.action
        {
            let found = post_state.entities.iter().any(|e| {
                e.alive
                    && e.kind == EntityKind::Building
                    && e.building
                        .as_ref()
                        .map(|b| {
                            b.building_type == *building_type
                                && b.grid_col == grid_cell.0
                                && b.grid_row == grid_cell.1
                        })
                        .unwrap_or(false)
            });
            if !found {
                errors.push(ValidationError {
                    code: "ACT-POST-003",
                    severity: Severity::Fatal,
                    message: format!(
                        "PlaceBuilding {:?} at ({}, {}) but no matching entity in post-state",
                        building_type, grid_cell.0, grid_cell.1
                    ),
                    context: ErrorContext {
                        grid_cell: Some(*grid_cell),
                        ..Default::default()
                    },
                });
            }
        }
    }

    // ACT-POST-004: Demolish made entity dead and cleared grid
    for action in actions {
        if let ActionPayload::Demolish { building_id } = &action.action {
            if let Some(entity) = post_state.entities.iter().find(|e| e.id == *building_id) {
                if entity.alive {
                    errors.push(ValidationError {
                        code: "ACT-POST-004",
                        severity: Severity::Fatal,
                        message: format!(
                            "Demolish building {} but entity is still alive in post-state",
                            building_id
                        ),
                        context: ErrorContext {
                            entity_id: Some(*building_id),
                            ..Default::default()
                        },
                    });
                }
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_grid(state: &WorldState, settlement_id: u32) -> Option<&crate::world_sim::city_grid::CityGrid> {
    state
        .settlements
        .iter()
        .find(|s| s.id == settlement_id)
        .and_then(|s| s.city_grid_idx)
        .and_then(|idx| state.city_grids.get(idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::building_ai::types::*;
    use crate::world_sim::city_grid::CityGrid;
    use crate::world_sim::state::{
        BuildingType, SettlementState, WorldState,
    };
    use crate::world_sim::building_ai::features::SpatialFeatures;

    fn test_state_with_grid() -> (WorldState, u32) {
        let mut state = WorldState::new(42);
        let sid = 1u32;
        let mut settlement = SettlementState::new(sid, "Test".into(), (0.0, 0.0));
        settlement.stockpile[0] = 1000.0; // food
        settlement.stockpile[2] = 1000.0; // wood

        let mut grid = CityGrid::new(5, 5, sid, "Plains", 0);
        // Center is road
        grid.cell_mut(2, 2).state = CellState::Road;
        grid.cell_mut(2, 2).road_tier = 3;

        settlement.city_grid_idx = Some(0);
        state.settlements.push(settlement);
        state.city_grids.push(grid);

        (state, sid)
    }

    fn test_obs(settlement_id: u32) -> BuildingObservation {
        BuildingObservation {
            settlement_id,
            tick: 0,
            challenges: Vec::new(),
            memory: ConstructionMemory::new(),
            spatial: SpatialFeatures::default(),
            friendly_roster: Vec::new(),
            high_value_npcs: Vec::new(),
            settlement_level: 1,
            tech_tier: 1,
            decision_tier: DecisionTier::Strategic,
        }
    }

    #[test]
    fn valid_placement_passes() {
        let (state, sid) = test_state_with_grid();
        let obs = test_obs(sid);
        let actions = vec![BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding {
                building_type: BuildingType::House,
                grid_cell: (1, 1),
            },
            priority: 1.0,
            reasoning_tag: 0,
        }];

        let errors = validate_action_batch(&state, &obs, &actions);
        let fatals: Vec<_> = errors.iter().filter(|e| e.severity == Severity::Fatal).collect();
        assert!(fatals.is_empty(), "Unexpected fatals: {:?}", fatals);
    }

    #[test]
    fn out_of_bounds_placement_detected() {
        let (state, sid) = test_state_with_grid();
        let obs = test_obs(sid);
        let actions = vec![BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding {
                building_type: BuildingType::House,
                grid_cell: (10, 10), // grid is 5x5
            },
            priority: 1.0,
            reasoning_tag: 0,
        }];

        let errors = validate_action_batch(&state, &obs, &actions);
        assert!(
            errors.iter().any(|e| e.code == "ACT-PRE-001"),
            "Expected ACT-PRE-001 error"
        );
    }

    #[test]
    fn placement_on_occupied_cell_detected() {
        let (mut state, sid) = test_state_with_grid();
        // Place a building at (1,1)
        let grid = &mut state.city_grids[0];
        grid.cell_mut(1, 1).state = CellState::Building;
        grid.cell_mut(1, 1).building_id = Some(99);

        let obs = test_obs(sid);
        let actions = vec![BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding {
                building_type: BuildingType::House,
                grid_cell: (1, 1),
            },
            priority: 1.0,
            reasoning_tag: 0,
        }];

        let errors = validate_action_batch(&state, &obs, &actions);
        assert!(
            errors.iter().any(|e| e.code == "ACT-PRE-002"),
            "Expected ACT-PRE-002 error"
        );
    }

    #[test]
    fn zero_dimension_wall_detected() {
        let (state, sid) = test_state_with_grid();
        let obs = test_obs(sid);
        let actions = vec![BuildingAction {
            decision_type: DecisionType::WallComposition,
            tier: DecisionTier::Structural,
            action: ActionPayload::SetWallSpec {
                segment_id: 1,
                height: 0,
                thickness: 1,
                material: crate::world_sim::building_ai::types::BuildMaterial::Stone,
                features: crate::world_sim::building_ai::types::WallFeatures::default(),
            },
            priority: 1.0,
            reasoning_tag: 0,
        }];

        let errors = validate_action_batch(&state, &obs, &actions);
        assert!(
            errors.iter().any(|e| e.code == "ACT-PRE-010"),
            "Expected ACT-PRE-010 error"
        );
    }
}
