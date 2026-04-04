//! Spatial feature sanity checks (FEAT-NUM-* and FEAT-STRUCT-* error codes).
//!
//! Validates the SpatialFeatures struct produced by compute_spatial_features.
//! Catches NaN, Inf, out-of-range values, and structural mismatches.

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::building_ai::features::SpatialFeatures;
use crate::world_sim::state::WorldState;

/// Run all spatial feature sanity checks.
pub fn validate_features(
    features: &SpatialFeatures,
    state: &WorldState,
    settlement_id: u32,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    errors.extend(check_numeric_sanity(features));
    errors.extend(check_structural_sanity(features, state, settlement_id));
    errors.extend(check_cross_references(features, state, settlement_id));
    errors
}

// ---------------------------------------------------------------------------
// FEAT-NUM-*: Numeric sanity
// ---------------------------------------------------------------------------

fn check_numeric_sanity(features: &SpatialFeatures) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // FEAT-NUM-001: Exhaustive float walk via serde_json introspection
    errors.extend(walk_and_check_floats(features));

    // FEAT-NUM-002: Specific range checks for known fraction fields
    check_range(
        &mut errors,
        "defensive.wall_coverage",
        features.defensive.wall_coverage,
        0.0,
        1.0,
    );
    check_range(
        &mut errors,
        "defensive.watchtower_coverage",
        features.defensive.watchtower_coverage,
        0.0,
        1.0,
    );
    check_range(
        &mut errors,
        "connectivity.evacuation_reachability",
        features.connectivity.evacuation_reachability,
        0.0,
        1.0,
    );
    check_range(
        &mut errors,
        "economic.storage_utilization",
        features.economic.storage_utilization,
        0.0,
        1.0,
    );
    // housing_pressure can exceed 1.0 (overcrowded), but should be non-negative
    check_range(
        &mut errors,
        "population.housing_pressure",
        features.population.housing_pressure,
        0.0,
        100.0, // generous upper bound
    );

    // Wall segment conditions
    for (i, seg) in features.defensive.wall_segments.iter().enumerate() {
        check_range(
            &mut errors,
            &format!("defensive.wall_segments[{}].condition", i),
            seg.condition,
            0.0,
            1.0,
        );
    }

    // FEAT-NUM-004: worker count consistency
    let wc = &features.economic.worker_counts;
    let specialist_sum = wc.construction as u32 + wc.masonry as u32 + wc.labor as u32;
    if specialist_sum > wc.total as u32 {
        errors.push(ValidationError {
            code: "FEAT-NUM-004",
            severity: Severity::Warning,
            message: format!(
                "Worker counts: construction({}) + masonry({}) + labor({}) = {} > total({})",
                wc.construction, wc.masonry, wc.labor, specialist_sum, wc.total
            ),
            context: ErrorContext {
                field: Some("economic.worker_counts".into()),
                ..Default::default()
            },
        });
    }

    // FEAT-NUM-005: connected_components >= 1 when buildings exist
    let has_developed = features.defensive.wall_segments.len() > 0
        || features.economic.utilization.len() > 0;
    if has_developed && features.connectivity.connected_components == 0 {
        errors.push(ValidationError {
            code: "FEAT-NUM-005",
            severity: Severity::Warning,
            message: "connected_components is 0 but developed cells exist".into(),
            context: ErrorContext {
                field: Some("connectivity.connected_components".into()),
                ..Default::default()
            },
        });
    }

    // Stockpile values should be non-negative
    for (i, &val) in features.economic.stockpiles.iter().enumerate() {
        if val < 0.0 {
            errors.push(ValidationError {
                code: "FEAT-NUM-002",
                severity: Severity::Warning,
                message: format!("economic.stockpiles[{}] = {} (negative)", i, val),
                context: ErrorContext {
                    field: Some(format!("economic.stockpiles[{}]", i)),
                    ..Default::default()
                },
            });
        }
    }

    // Garrison coverage and response time values
    for (i, &val) in features.garrison.coverage_map.iter().enumerate() {
        if !val.is_finite() {
            errors.push(ValidationError {
                code: "FEAT-NUM-001",
                severity: Severity::Fatal,
                message: format!(
                    "garrison.coverage_map[{}] = {} (non-finite)",
                    i, val
                ),
                context: ErrorContext {
                    field: Some(format!("garrison.coverage_map[{}]", i)),
                    ..Default::default()
                },
            });
        }
    }

    // FEAT-STRUCT-004: response time values non-negative
    for (i, &val) in features.garrison.response_time_map.iter().enumerate() {
        if val < 0.0 {
            errors.push(ValidationError {
                code: "FEAT-STRUCT-004",
                severity: Severity::Fatal,
                message: format!(
                    "garrison.response_time_map[{}] = {} (negative BFS distance)",
                    i, val
                ),
                context: ErrorContext {
                    field: Some(format!("garrison.response_time_map[{}]", i)),
                    ..Default::default()
                },
            });
        }
        if !val.is_finite() {
            errors.push(ValidationError {
                code: "FEAT-NUM-001",
                severity: Severity::Fatal,
                message: format!(
                    "garrison.response_time_map[{}] = {} (non-finite)",
                    i, val
                ),
                context: ErrorContext {
                    field: Some(format!("garrison.response_time_map[{}]", i)),
                    ..Default::default()
                },
            });
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// FEAT-STRUCT-*: Structural sanity
// ---------------------------------------------------------------------------

fn check_structural_sanity(
    features: &SpatialFeatures,
    state: &WorldState,
    _settlement_id: u32,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Grid bounds — VoxelWorld doesn't have explicit cols/rows; use a large virtual grid.
    let (cols, rows) = (64u16, 64u16);

    // FEAT-STRUCT-001: key_building_paths reference existing entities
    for (i, entry) in features.connectivity.key_building_paths.iter().enumerate() {
        let a_exists = state.entities.iter().any(|e| e.id == entry.id_a && e.alive);
        let b_exists = state.entities.iter().any(|e| e.id == entry.id_b && e.alive);
        if !a_exists {
            errors.push(ValidationError {
                code: "FEAT-STRUCT-001",
                severity: Severity::Warning,
                message: format!(
                    "key_building_paths[{}].id_a {} does not reference an alive entity",
                    i, entry.id_a
                ),
                context: ErrorContext {
                    entity_id: Some(entry.id_a),
                    field: Some(format!("connectivity.key_building_paths[{}].id_a", i)),
                    ..Default::default()
                },
            });
        }
        if !b_exists {
            errors.push(ValidationError {
                code: "FEAT-STRUCT-001",
                severity: Severity::Warning,
                message: format!(
                    "key_building_paths[{}].id_b {} does not reference an alive entity",
                    i, entry.id_b
                ),
                context: ErrorContext {
                    entity_id: Some(entry.id_b),
                    field: Some(format!("connectivity.key_building_paths[{}].id_b", i)),
                    ..Default::default()
                },
            });
        }
    }

    // FEAT-STRUCT-002: wall segments within grid bounds
    if cols > 0 && rows > 0 {
        for (i, seg) in features.defensive.wall_segments.iter().enumerate() {
            if seg.start.0 >= cols || seg.start.1 >= rows {
                errors.push(ValidationError {
                    code: "FEAT-STRUCT-002",
                    severity: Severity::Fatal,
                    message: format!(
                        "wall_segments[{}].start ({}, {}) out of bounds ({}x{})",
                        i, seg.start.0, seg.start.1, cols, rows
                    ),
                    context: ErrorContext {
                        grid_cell: Some(seg.start),
                        field: Some(format!("defensive.wall_segments[{}].start", i)),
                        ..Default::default()
                    },
                });
            }
            if seg.end.0 >= cols || seg.end.1 >= rows {
                errors.push(ValidationError {
                    code: "FEAT-STRUCT-002",
                    severity: Severity::Fatal,
                    message: format!(
                        "wall_segments[{}].end ({}, {}) out of bounds ({}x{})",
                        i, seg.end.0, seg.end.1, cols, rows
                    ),
                    context: ErrorContext {
                        grid_cell: Some(seg.end),
                        field: Some(format!("defensive.wall_segments[{}].end", i)),
                        ..Default::default()
                    },
                });
            }
        }
    }

    // FEAT-STRUCT-005: synergy hotspots reference valid entities
    for (i, entry) in features.garrison.synergy_hotspots.iter().enumerate() {
        let unit_exists = state
            .entities
            .iter()
            .any(|e| e.id == entry.unit_id && e.alive);
        let structure_exists = state
            .entities
            .iter()
            .any(|e| e.id == entry.structure_id && e.alive);
        if !unit_exists {
            errors.push(ValidationError {
                code: "FEAT-STRUCT-005",
                severity: Severity::Warning,
                message: format!(
                    "synergy_hotspots[{}].unit_id {} not found as alive entity",
                    i, entry.unit_id
                ),
                context: ErrorContext {
                    entity_id: Some(entry.unit_id),
                    field: Some(format!("garrison.synergy_hotspots[{}].unit_id", i)),
                    ..Default::default()
                },
            });
        }
        if !structure_exists {
            errors.push(ValidationError {
                code: "FEAT-STRUCT-005",
                severity: Severity::Warning,
                message: format!(
                    "synergy_hotspots[{}].structure_id {} not found as alive entity",
                    i, entry.structure_id
                ),
                context: ErrorContext {
                    entity_id: Some(entry.structure_id),
                    field: Some(format!("garrison.synergy_hotspots[{}].structure_id", i)),
                    ..Default::default()
                },
            });
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// FEAT-NUM-003: Cross-reference features against state
// ---------------------------------------------------------------------------

fn check_cross_references(
    features: &SpatialFeatures,
    state: &WorldState,
    settlement_id: u32,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // FEAT-NUM-003: stockpiles match state
    if let Some(settlement) = state.settlements.iter().find(|s| s.id == settlement_id) {
        for i in 0..features.economic.stockpiles.len().min(settlement.stockpile.len()) {
            let feat_val = features.economic.stockpiles[i];
            let state_val = settlement.stockpile[i];
            // Use approximate comparison (features may be derived differently)
            if (feat_val - state_val).abs() > 1.0 {
                errors.push(ValidationError {
                    code: "FEAT-NUM-003",
                    severity: Severity::Warning,
                    message: format!(
                        "stockpiles[{}]: feature={} but state={} (mismatch > 1.0)",
                        i, feat_val, state_val
                    ),
                    context: ErrorContext {
                        field: Some(format!("economic.stockpiles[{}]", i)),
                        ..Default::default()
                    },
                });
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Exhaustive float walker via serde_json introspection (FEAT-NUM-001)
// ---------------------------------------------------------------------------

/// Walk all float values in the SpatialFeatures struct via JSON serialization.
/// Returns errors for any NaN or Inf values found.
fn walk_and_check_floats(features: &SpatialFeatures) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let json_val = match serde_json::to_value(features) {
        Ok(v) => v,
        Err(e) => {
            errors.push(ValidationError {
                code: "FEAT-NUM-001",
                severity: Severity::Fatal,
                message: format!("Failed to serialize SpatialFeatures: {}", e),
                context: ErrorContext::default(),
            });
            return errors;
        }
    };

    walk_json_floats(&json_val, String::new(), &mut errors);
    errors
}

fn walk_json_floats(
    val: &serde_json::Value,
    path: String,
    errors: &mut Vec<ValidationError>,
) {
    match val {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.is_nan() || f.is_infinite() {
                    errors.push(ValidationError {
                        code: "FEAT-NUM-001",
                        severity: Severity::Fatal,
                        message: format!("Non-finite float at {}: {}", path, f),
                        context: ErrorContext {
                            field: Some(path),
                            ..Default::default()
                        },
                    });
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                walk_json_floats(item, format!("{}[{}]", path, i), errors);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, item) in map {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };
                walk_json_floats(item, child_path, errors);
            }
        }
        _ => {} // booleans, strings, null — skip
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_range(
    errors: &mut Vec<ValidationError>,
    field: &str,
    value: f32,
    min: f32,
    max: f32,
) {
    if !value.is_finite() {
        errors.push(ValidationError {
            code: "FEAT-NUM-001",
            severity: Severity::Fatal,
            message: format!("{} = {} (non-finite)", field, value),
            context: ErrorContext {
                field: Some(field.into()),
                ..Default::default()
            },
        });
        return;
    }
    if value < min || value > max {
        errors.push(ValidationError {
            code: "FEAT-NUM-002",
            severity: Severity::Warning,
            message: format!(
                "{} = {} outside expected range [{}, {}]",
                field, value, min, max
            ),
            context: ErrorContext {
                field: Some(field.into()),
                ..Default::default()
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::building_ai::features::SpatialFeatures;

    #[test]
    fn default_features_pass_validation() {
        let state = WorldState::new(42);
        let features = SpatialFeatures::default();
        let errors = validate_features(&features, &state, 0);
        let fatals: Vec<_> = errors.iter().filter(|e| e.severity == Severity::Fatal).collect();
        assert!(fatals.is_empty(), "Unexpected fatals: {:?}", fatals);
    }

    #[test]
    fn out_of_range_wall_coverage_detected() {
        let state = WorldState::new(42);
        let mut features = SpatialFeatures::default();
        features.defensive.wall_coverage = 1.5;
        let errors = validate_features(&features, &state, 0);
        assert!(
            errors.iter().any(|e| e.code == "FEAT-NUM-002"),
            "Expected FEAT-NUM-002 warning for wall_coverage > 1.0"
        );
    }
}
