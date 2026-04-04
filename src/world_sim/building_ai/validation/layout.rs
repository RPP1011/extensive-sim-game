//! Layout validation — checks spatial layout descriptions before WorldState generation.
//!
//! Validates that a `SettlementLayout` is internally consistent and buildable.
//! Run this on TOML input before generating a WorldState from it.

use std::collections::HashSet;

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::building_ai::scenario_config::SettlementLayout;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Validate a settlement layout. Returns errors found (empty = clean).
pub fn validate_layout(layout: &SettlementLayout) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let (cols, rows) = layout.grid_size;

    // Grid size sanity
    if cols == 0 || rows == 0 {
        errors.push(ValidationError {
            code: "LAY-GRID-001",
            severity: Severity::Fatal,
            message: format!("Grid size is zero: ({}, {})", cols, rows),
            context: ErrorContext::default(),
        });
        return errors; // Can't validate positions against a zero grid
    }
    if cols > 256 || rows > 256 {
        errors.push(ValidationError {
            code: "LAY-GRID-002",
            severity: Severity::Warning,
            message: format!("Grid size very large: ({}, {})", cols, rows),
            context: ErrorContext::default(),
        });
    }

    // -----------------------------------------------------------------------
    // Building validation
    // -----------------------------------------------------------------------

    let mut occupied_cells: HashSet<(u16, u16)> = HashSet::new();

    for (i, b) in layout.buildings.iter().enumerate() {
        let (bc, br) = b.cell;
        let (fw, fh) = b.footprint;

        // Footprint sanity
        if fw == 0 || fh == 0 {
            errors.push(ValidationError {
                code: "LAY-BLD-001",
                severity: Severity::Fatal,
                message: format!(
                    "Building {} '{}' at ({},{}) has zero footprint ({},{})",
                    i, b.building_type, bc, br, fw, fh
                ),
                context: ErrorContext {
                    grid_cell: Some((bc, br)),
                    ..Default::default()
                },
            });
            continue;
        }

        // Bounds check
        if bc + fw > cols || br + fh > rows {
            errors.push(ValidationError {
                code: "LAY-BLD-002",
                severity: Severity::Fatal,
                message: format!(
                    "Building {} '{}' at ({},{}) with footprint ({},{}) extends beyond grid ({},{})",
                    i, b.building_type, bc, br, fw, fh, cols, rows
                ),
                context: ErrorContext {
                    grid_cell: Some((bc, br)),
                    ..Default::default()
                },
            });
            continue;
        }

        // Overlap check
        for dc in 0..fw {
            for dr in 0..fh {
                let cell = (bc + dc, br + dr);
                if !occupied_cells.insert(cell) {
                    errors.push(ValidationError {
                        code: "LAY-BLD-003",
                        severity: Severity::Fatal,
                        message: format!(
                            "Building {} '{}' at ({},{}) overlaps another building at cell ({},{})",
                            i, b.building_type, bc, br, cell.0, cell.1
                        ),
                        context: ErrorContext {
                            grid_cell: Some(cell),
                            ..Default::default()
                        },
                    });
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Wall circuit validation
    // -----------------------------------------------------------------------

    for (i, wc) in layout.wall_circuits.iter().enumerate() {
        if wc.waypoints.len() < 3 {
            errors.push(ValidationError {
                code: "LAY-WALL-001",
                severity: Severity::Fatal,
                message: format!(
                    "Wall circuit {} needs at least 3 waypoints to form a polygon, has {}",
                    i,
                    wc.waypoints.len()
                ),
                context: ErrorContext::default(),
            });
            continue;
        }

        // Bounds check on waypoints
        for (j, &(wc_col, wc_row)) in wc.waypoints.iter().enumerate() {
            if wc_col >= cols || wc_row >= rows {
                errors.push(ValidationError {
                    code: "LAY-WALL-002",
                    severity: Severity::Fatal,
                    message: format!(
                        "Wall circuit {} waypoint {} ({},{}) is outside grid ({},{})",
                        i, j, wc_col, wc_row, cols, rows
                    ),
                    context: ErrorContext {
                        grid_cell: Some((wc_col, wc_row)),
                        ..Default::default()
                    },
                });
            }
        }

        // Duplicate consecutive waypoints
        for j in 0..wc.waypoints.len() {
            let next = (j + 1) % wc.waypoints.len();
            if wc.waypoints[j] == wc.waypoints[next] {
                errors.push(ValidationError {
                    code: "LAY-WALL-003",
                    severity: Severity::Warning,
                    message: format!(
                        "Wall circuit {} has duplicate consecutive waypoints at index {} ({},{})",
                        i, j, wc.waypoints[j].0, wc.waypoints[j].1
                    ),
                    context: ErrorContext {
                        grid_cell: Some(wc.waypoints[j]),
                        ..Default::default()
                    },
                });
            }
        }

        // Height/thickness sanity
        if wc.height == 0 {
            errors.push(ValidationError {
                code: "LAY-WALL-004",
                severity: Severity::Warning,
                message: format!("Wall circuit {} has height 0", i),
                context: ErrorContext::default(),
            });
        }
    }

    // -----------------------------------------------------------------------
    // Wall segment validation
    // -----------------------------------------------------------------------

    for (i, ws) in layout.wall_segments.iter().enumerate() {
        // Bounds
        if ws.from.0 >= cols || ws.from.1 >= rows {
            errors.push(ValidationError {
                code: "LAY-SEG-001",
                severity: Severity::Fatal,
                message: format!(
                    "Wall segment {} 'from' ({},{}) is outside grid",
                    i, ws.from.0, ws.from.1
                ),
                context: ErrorContext {
                    grid_cell: Some(ws.from),
                    ..Default::default()
                },
            });
        }
        if ws.to.0 >= cols || ws.to.1 >= rows {
            errors.push(ValidationError {
                code: "LAY-SEG-001",
                severity: Severity::Fatal,
                message: format!(
                    "Wall segment {} 'to' ({},{}) is outside grid",
                    i, ws.to.0, ws.to.1
                ),
                context: ErrorContext {
                    grid_cell: Some(ws.to),
                    ..Default::default()
                },
            });
        }
        // Zero-length
        if ws.from == ws.to {
            errors.push(ValidationError {
                code: "LAY-SEG-002",
                severity: Severity::Warning,
                message: format!("Wall segment {} has zero length (from == to)", i),
                context: ErrorContext {
                    grid_cell: Some(ws.from),
                    ..Default::default()
                },
            });
        }
    }

    // -----------------------------------------------------------------------
    // Road validation
    // -----------------------------------------------------------------------

    for (i, road) in layout.roads.iter().enumerate() {
        if road.waypoints.len() < 2 {
            errors.push(ValidationError {
                code: "LAY-ROAD-001",
                severity: Severity::Warning,
                message: format!("Road {} needs at least 2 waypoints, has {}", i, road.waypoints.len()),
                context: ErrorContext::default(),
            });
            continue;
        }
        for (j, &(rc, rr)) in road.waypoints.iter().enumerate() {
            if rc >= cols || rr >= rows {
                errors.push(ValidationError {
                    code: "LAY-ROAD-002",
                    severity: Severity::Fatal,
                    message: format!(
                        "Road {} waypoint {} ({},{}) is outside grid ({},{})",
                        i, j, rc, rr, cols, rows
                    ),
                    context: ErrorContext {
                        grid_cell: Some((rc, rr)),
                        ..Default::default()
                    },
                });
            }
        }

        // Check if road passes through a building footprint
        for (j, &wp) in road.waypoints.iter().enumerate() {
            if occupied_cells.contains(&wp) {
                errors.push(ValidationError {
                    code: "LAY-ROAD-003",
                    severity: Severity::Warning,
                    message: format!(
                        "Road {} waypoint {} ({},{}) passes through a building footprint",
                        i, j, wp.0, wp.1
                    ),
                    context: ErrorContext {
                        grid_cell: Some(wp),
                        ..Default::default()
                    },
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Zone validation
    // -----------------------------------------------------------------------

    let valid_zones = ["residential", "industrial", "military", "commercial", "religious"];
    for (i, z) in layout.zones.iter().enumerate() {
        if !valid_zones.contains(&z.zone.as_str()) {
            errors.push(ValidationError {
                code: "LAY-ZONE-001",
                severity: Severity::Warning,
                message: format!("Zone {} has unknown type '{}'", i, z.zone),
                context: ErrorContext::default(),
            });
        }
        // from <= to
        if z.from.0 > z.to.0 || z.from.1 > z.to.1 {
            errors.push(ValidationError {
                code: "LAY-ZONE-002",
                severity: Severity::Fatal,
                message: format!(
                    "Zone {} '{}' has inverted bounds: from ({},{}) > to ({},{})",
                    i, z.zone, z.from.0, z.from.1, z.to.0, z.to.1
                ),
                context: ErrorContext::default(),
            });
        }
        // Bounds
        if z.to.0 >= cols || z.to.1 >= rows {
            errors.push(ValidationError {
                code: "LAY-ZONE-003",
                severity: Severity::Fatal,
                message: format!(
                    "Zone {} '{}' extends beyond grid: to ({},{}) vs grid ({},{})",
                    i, z.zone, z.to.0, z.to.1, cols, rows
                ),
                context: ErrorContext::default(),
            });
        }
    }

    // -----------------------------------------------------------------------
    // Gate validation
    // -----------------------------------------------------------------------

    for (i, g) in layout.gates.iter().enumerate() {
        // Bounds
        if g.cell.0 >= cols || g.cell.1 >= rows {
            errors.push(ValidationError {
                code: "LAY-GATE-001",
                severity: Severity::Fatal,
                message: format!(
                    "Gate {} at ({},{}) is outside grid ({},{})",
                    i, g.cell.0, g.cell.1, cols, rows
                ),
                context: ErrorContext {
                    grid_cell: Some(g.cell),
                    ..Default::default()
                },
            });
        }

        // Gate should be near a wall circuit
        let near_wall = layout.wall_circuits.iter().any(|wc| {
            wall_circuit_cells(wc).iter().any(|&wc_cell| {
                let dx = (g.cell.0 as i32 - wc_cell.0 as i32).unsigned_abs();
                let dy = (g.cell.1 as i32 - wc_cell.1 as i32).unsigned_abs();
                dx <= 1 && dy <= 1
            })
        });
        if !near_wall && !layout.wall_circuits.is_empty() {
            errors.push(ValidationError {
                code: "LAY-GATE-002",
                severity: Severity::Warning,
                message: format!(
                    "Gate {} at ({},{}) is not adjacent to any wall circuit",
                    i, g.cell.0, g.cell.1
                ),
                context: ErrorContext {
                    grid_cell: Some(g.cell),
                    ..Default::default()
                },
            });
        }

        // Width sanity
        if g.width == 0 {
            errors.push(ValidationError {
                code: "LAY-GATE-003",
                severity: Severity::Warning,
                message: format!("Gate {} has width 0", i),
                context: ErrorContext {
                    grid_cell: Some(g.cell),
                    ..Default::default()
                },
            });
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rasterize a wall circuit's polygon edges into grid cells (Bresenham-style).
fn wall_circuit_cells(wc: &crate::world_sim::building_ai::scenario_config::WallCircuit) -> Vec<(u16, u16)> {
    let mut cells = Vec::new();
    let n = wc.waypoints.len();
    if n < 2 {
        return cells;
    }
    for i in 0..n {
        let (x0, y0) = wc.waypoints[i];
        let (x1, y1) = wc.waypoints[(i + 1) % n];
        rasterize_line(x0 as i32, y0 as i32, x1 as i32, y1 as i32, &mut cells);
    }
    cells
}

/// Bresenham's line from (x0,y0) to (x1,y1), appending cells to `out`.
fn rasterize_line(x0: i32, y0: i32, x1: i32, y1: i32, out: &mut Vec<(u16, u16)>) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        if x >= 0 && y >= 0 {
            out.push((x as u16, y as u16));
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::building_ai::scenario_config::*;

    fn empty_layout(cols: u16, rows: u16) -> SettlementLayout {
        SettlementLayout {
            grid_size: (cols, rows),
            ..Default::default()
        }
    }

    #[test]
    fn empty_layout_passes() {
        let errors = validate_layout(&empty_layout(20, 20));
        assert!(errors.is_empty());
    }

    #[test]
    fn zero_grid_is_fatal() {
        let errors = validate_layout(&empty_layout(0, 10));
        assert!(errors.iter().any(|e| e.code == "LAY-GRID-001"));
    }

    #[test]
    fn building_out_of_bounds() {
        let mut layout = empty_layout(10, 10);
        layout.buildings.push(LayoutBuilding {
            building_type: "House".to_string(),
            cell: (9, 9),
            footprint: (2, 2),
            material: None,
            facing: None,
            tier: None,
        });
        let errors = validate_layout(&layout);
        assert!(errors.iter().any(|e| e.code == "LAY-BLD-002"));
    }

    #[test]
    fn building_overlap_detected() {
        let mut layout = empty_layout(20, 20);
        layout.buildings.push(LayoutBuilding {
            building_type: "House".to_string(),
            cell: (5, 5),
            footprint: (3, 3),
            material: None,
            facing: None,
            tier: None,
        });
        layout.buildings.push(LayoutBuilding {
            building_type: "Market".to_string(),
            cell: (7, 7),
            footprint: (2, 2),
            material: None,
            facing: None,
            tier: None,
        });
        let errors = validate_layout(&layout);
        assert!(errors.iter().any(|e| e.code == "LAY-BLD-003"));
    }

    #[test]
    fn wall_circuit_needs_3_waypoints() {
        let mut layout = empty_layout(20, 20);
        layout.wall_circuits.push(WallCircuit {
            waypoints: vec![(0, 0), (10, 0)],
            material: "stone".to_string(),
            height: 3,
            thickness: 1,
        });
        let errors = validate_layout(&layout);
        assert!(errors.iter().any(|e| e.code == "LAY-WALL-001"));
    }

    #[test]
    fn gate_not_near_wall_warned() {
        let mut layout = empty_layout(20, 20);
        layout.wall_circuits.push(WallCircuit {
            waypoints: vec![(2, 2), (18, 2), (18, 18), (2, 18)],
            material: "stone".to_string(),
            height: 3,
            thickness: 1,
        });
        layout.gates.push(GateSpec {
            cell: (10, 10), // center, far from walls
            facing: None,
            width: 1,
        });
        let errors = validate_layout(&layout);
        assert!(errors.iter().any(|e| e.code == "LAY-GATE-002"));
    }

    #[test]
    fn road_through_building_warned() {
        let mut layout = empty_layout(20, 20);
        layout.buildings.push(LayoutBuilding {
            building_type: "House".to_string(),
            cell: (5, 5),
            footprint: (3, 3),
            material: None,
            facing: None,
            tier: None,
        });
        layout.roads.push(RoadPath {
            waypoints: vec![(5, 5), (5, 10)],
        });
        let errors = validate_layout(&layout);
        assert!(errors.iter().any(|e| e.code == "LAY-ROAD-003"));
    }

    #[test]
    fn zone_inverted_bounds() {
        let mut layout = empty_layout(20, 20);
        layout.zones.push(ZoneRect {
            zone: "residential".to_string(),
            from: (10, 10),
            to: (5, 5),
        });
        let errors = validate_layout(&layout);
        assert!(errors.iter().any(|e| e.code == "LAY-ZONE-002"));
    }

    #[test]
    fn valid_layout_passes() {
        let mut layout = empty_layout(24, 24);
        // Wall circuit
        layout.wall_circuits.push(WallCircuit {
            waypoints: vec![(2, 2), (20, 2), (20, 20), (2, 20)],
            material: "stone".to_string(),
            height: 4,
            thickness: 1,
        });
        // Gate on the north wall
        layout.gates.push(GateSpec {
            cell: (10, 2),
            facing: Some("north".to_string()),
            width: 2,
        });
        // Buildings
        layout.buildings.push(LayoutBuilding {
            building_type: "Market".to_string(),
            cell: (10, 10),
            footprint: (3, 3),
            material: Some("stone".to_string()),
            facing: None,
            tier: None,
        });
        layout.buildings.push(LayoutBuilding {
            building_type: "Barracks".to_string(),
            cell: (15, 4),
            footprint: (4, 3),
            material: None,
            facing: Some("north".to_string()),
            tier: None,
        });
        // Road from gate to market
        layout.roads.push(RoadPath {
            waypoints: vec![(10, 3), (10, 5), (10, 9)],
        });
        // Zones
        layout.zones.push(ZoneRect {
            zone: "residential".to_string(),
            from: (3, 3),
            to: (9, 9),
        });
        layout.zones.push(ZoneRect {
            zone: "military".to_string(),
            from: (14, 3),
            to: (19, 8),
        });

        let errors = validate_layout(&layout);
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }
}
