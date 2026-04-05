//! Action validity checks (ACT-PRE-* and ACT-POST-* error codes).
//!
//! Pre-application checks validate that oracle-produced actions are applicable
//! to the current WorldState. Post-application checks confirm the action
//! produced a valid successor state.

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::building_ai::types::{
    ActionPayload, BuildingAction, BuildingObservation,
};
use crate::world_sim::voxel::VoxelMaterial;
use crate::world_sim::state::{BuildingType, EntityKind, WorldState};

/// Virtual grid size used for bounds checking — matches env GRID_SIZE (128).
const VIRTUAL_GRID_SIZE: usize = 128;

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

    for (ai, action) in actions.iter().enumerate() {
        match &action.action {
            ActionPayload::PlaceBuilding {
                building_type,
                grid_cell,
            } => {
                errors.extend(check_place_building(
                    state,
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
                // Check footprint fits within virtual grid bounds
                if let Some(entity) = state.entities.iter().find(|e| e.id == *building_id) {
                    if let Some(bdata) = &entity.building {
                        let end_col = bdata.grid_col as usize + dimensions.0 as usize;
                        let end_row = bdata.grid_row as usize + dimensions.1 as usize;
                        if end_col > VIRTUAL_GRID_SIZE || end_row > VIRTUAL_GRID_SIZE {
                            errors.push(ValidationError {
                                code: "ACT-PRE-012",
                                severity: Severity::Fatal,
                                message: format!(
                                    "Action {}: SetFootprint from ({}, {}) with dims ({}, {}) overflows virtual grid ({}x{})",
                                    ai, bdata.grid_col, bdata.grid_row,
                                    dimensions.0, dimensions.1, VIRTUAL_GRID_SIZE, VIRTUAL_GRID_SIZE
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
                // ACT-PRE-008: all waypoints in virtual bounds
                for (wi, wp) in waypoints.iter().enumerate() {
                    if wp.0 as usize >= VIRTUAL_GRID_SIZE || wp.1 as usize >= VIRTUAL_GRID_SIZE {
                        errors.push(ValidationError {
                            code: "ACT-PRE-008",
                            severity: Severity::Fatal,
                            message: format!(
                                "Action {}: RouteRoad waypoint {} ({}, {}) out of virtual bounds ({}x{})",
                                ai, wi, wp.0, wp.1, VIRTUAL_GRID_SIZE, VIRTUAL_GRID_SIZE
                            ),
                            context: ErrorContext {
                                grid_cell: Some(*wp),
                                ..Default::default()
                            },
                        });
                    }
                }
            }
            ActionPayload::SetZone { grid_cell, .. } => {
                // ACT-PRE-009: grid_cell in virtual bounds
                if grid_cell.0 as usize >= VIRTUAL_GRID_SIZE || grid_cell.1 as usize >= VIRTUAL_GRID_SIZE {
                    errors.push(ValidationError {
                        code: "ACT-PRE-009",
                        severity: Severity::Fatal,
                        message: format!(
                            "Action {}: SetZone grid_cell ({}, {}) out of virtual bounds ({}x{})",
                            ai, grid_cell.0, grid_cell.1, VIRTUAL_GRID_SIZE, VIRTUAL_GRID_SIZE
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

    errors
}

/// Validate PlaceBuilding action specifics.
fn check_place_building(
    state: &WorldState,
    building_type: BuildingType,
    grid_cell: (u16, u16),
    obs: &BuildingObservation,
    action_idx: usize,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let (c, r) = (grid_cell.0 as usize, grid_cell.1 as usize);

    // ACT-PRE-001: in virtual bounds
    if c >= VIRTUAL_GRID_SIZE || r >= VIRTUAL_GRID_SIZE {
        errors.push(ValidationError {
            code: "ACT-PRE-001",
            severity: Severity::Fatal,
            message: format!(
                "Action {}: PlaceBuilding at ({}, {}) out of virtual bounds ({}x{})",
                action_idx, c, r, VIRTUAL_GRID_SIZE, VIRTUAL_GRID_SIZE
            ),
            context: ErrorContext {
                grid_cell: Some(grid_cell),
                ..Default::default()
            },
        });
        return errors;
    }

    // ACT-PRE-002: target voxel column must not already have a building
    let vx = grid_cell.0 as i32;
    let vy = grid_cell.1 as i32;
    let surface_z = state.voxel_world.surface_height(vx, vy);
    let surface_voxel = state.voxel_world.get_voxel(vx, vy, surface_z);
    if surface_voxel.building_id.is_some() {
        errors.push(ValidationError {
            code: "ACT-PRE-002",
            severity: Severity::Fatal,
            message: format!(
                "Action {}: PlaceBuilding at ({}, {}) but cell already contains building {:?}",
                action_idx, c, r, surface_voxel.building_id
            ),
            context: ErrorContext {
                grid_cell: Some(grid_cell),
                ..Default::default()
            },
        });
    }

    // ACT-PRE-003: terrain not Water or Lava
    if matches!(surface_voxel.material, VoxelMaterial::Water | VoxelMaterial::Lava) {
        errors.push(ValidationError {
            code: "ACT-PRE-003",
            severity: Severity::Fatal,
            message: format!(
                "Action {}: PlaceBuilding at ({}, {}) on unbuildable terrain {:?}",
                action_idx, c, r, surface_voxel.material
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::building_ai::types::*;
    use crate::world_sim::state::{BuildingType, SettlementState, WorldState};
    use crate::world_sim::building_ai::features::SpatialFeatures;
    use crate::world_sim::voxel::{Voxel, VoxelMaterial};

    fn test_state_with_voxels() -> (WorldState, u32) {
        let mut state = WorldState::new(42);
        let sid = 1u32;
        let mut settlement = SettlementState::new(sid, "Test".into(), (0.0, 0.0));
        settlement.stockpile[0] = 1000.0; // food
        settlement.stockpile[2] = 1000.0; // wood
        state.settlements.push(settlement);

        // Place solid stone ground at test positions so surface_height works deterministically.
        // surface_height returns the z above the topmost solid, so we place stone at z=0.
        for x in 0..10i32 {
            for y in 0..10i32 {
                state.voxel_world.set_voxel(x, y, 0, Voxel::new(VoxelMaterial::Stone));
            }
        }

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
        let (state, sid) = test_state_with_voxels();
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
        let (state, sid) = test_state_with_voxels();
        let obs = test_obs(sid);
        let actions = vec![BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding {
                building_type: BuildingType::House,
                grid_cell: (200, 200), // beyond virtual grid size
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
        let (mut state, sid) = test_state_with_voxels();
        // Mark the surface slot at (1,1) as occupied by an existing building.
        // surface_height returns z+1 above the topmost solid, so the "surface slot" is Air at that z.
        // Setting building_id on the Air voxel at that z marks it as occupied.
        let surface_z = state.voxel_world.surface_height(1, 1);
        let mut occupied_voxel = crate::world_sim::voxel::Voxel::default(); // Air
        occupied_voxel.building_id = Some(99);
        state.voxel_world.set_voxel(1, 1, surface_z, occupied_voxel);

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
        let (state, sid) = test_state_with_voxels();
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
