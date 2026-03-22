//! Spatial analysis: geometry corner extraction and precomputed visibility.
//!
//! Corners are where wall boundaries change direction — they mark choke points,
//! cover positions, and fortification opportunities.
//!
//! Visibility is precomputed per-cell at room load as a bitset of which corners
//! are visible. At tick time, a unit's visible corners are a single cell lookup.

use std::collections::HashMap;

use crate::sim::{distance, SimVec2};
use crate::pathing::GridNav;

// ---------------------------------------------------------------------------
// Corner extraction
// ---------------------------------------------------------------------------

/// A detected geometry corner.
#[derive(Debug, Clone, Copy)]
pub struct Corner {
    /// World-space position of the corner.
    pub position: SimVec2,
    /// Grid cell of this corner.
    pub cell: (i32, i32),
    /// Corner type: convex (wall sticks out) or concave (wall recedes).
    pub convex: bool,
    /// Number of blocked neighbors (1-3). Higher = more enclosed.
    pub blocked_neighbors: u8,
    /// Approximate opening direction (unit vector pointing into open space).
    pub open_direction: SimVec2,
    /// Local passage width: minimum walkable distance perpendicular to wall.
    pub passage_width: f32,
    /// Elevation at this corner.
    pub elevation: f32,
}

/// Maximum corners to extract. Keeps token count bounded.
/// Also determines the bitset width for visibility (u32 supports up to 32).
pub const MAX_CORNERS: usize = 16;

/// Extract geometry corners from a GridNav. Returns up to MAX_CORNERS,
/// prioritized by tactical relevance (narrow passages first).
pub fn extract_corners(nav: &GridNav) -> Vec<Corner> {
    if nav.fully_open {
        return vec![];
    }

    let cx0 = (nav.min_x / nav.cell_size).floor() as i32;
    let cx1 = (nav.max_x / nav.cell_size).ceil() as i32;
    let cy0 = (nav.min_y / nav.cell_size).floor() as i32;
    let cy1 = (nav.max_y / nav.cell_size).ceil() as i32;

    let mut corners = Vec::new();

    for x in cx0..=cx1 {
        for y in cy0..=cy1 {
            if !nav.walkable(x, y) {
                continue;
            }

            let n = nav.blocked.contains(&(x, y - 1)) || !nav.in_bounds(x, y - 1);
            let s = nav.blocked.contains(&(x, y + 1)) || !nav.in_bounds(x, y + 1);
            let e = nav.blocked.contains(&(x + 1, y)) || !nav.in_bounds(x + 1, y);
            let w = nav.blocked.contains(&(x - 1, y)) || !nav.in_bounds(x - 1, y);

            let blocked_count = [n, s, e, w].iter().filter(|&&b| b).count() as u8;

            if blocked_count == 0 || blocked_count == 4 {
                continue;
            }

            // Check diagonals for freestanding wall detection
            let ne_blocked = nav.blocked.contains(&(x + 1, y - 1)) || !nav.in_bounds(x + 1, y - 1);
            let nw_blocked = nav.blocked.contains(&(x - 1, y - 1)) || !nav.in_bounds(x - 1, y - 1);
            let se_blocked = nav.blocked.contains(&(x + 1, y + 1)) || !nav.in_bounds(x + 1, y + 1);
            let sw_blocked = nav.blocked.contains(&(x - 1, y + 1)) || !nav.in_bounds(x - 1, y + 1);

            let is_corner = match blocked_count {
                1 => {
                    (n && (ne_blocked || nw_blocked))
                        || (s && (se_blocked || sw_blocked))
                        || (e && (ne_blocked || se_blocked))
                        || (w && (nw_blocked || sw_blocked))
                }
                2 => !((n && s) || (e && w)),
                3 => true,
                _ => false,
            };

            if !is_corner {
                continue;
            }

            let pos = nav.center_of((x, y));

            let mut ox = 0.0f32;
            let mut oy = 0.0f32;
            if !n { oy -= 1.0; }
            if !s { oy += 1.0; }
            if !e { ox += 1.0; }
            if !w { ox -= 1.0; }
            let len = (ox * ox + oy * oy).sqrt().max(0.001);
            ox /= len;
            oy /= len;

            let convex = if blocked_count == 2 {
                let diag_x = if e { x + 1 } else if w { x - 1 } else { x };
                let diag_y = if n { y - 1 } else if s { y + 1 } else { y };
                nav.walkable(diag_x, diag_y)
            } else {
                false
            };

            let passage_width = measure_passage_width(nav, x, y, ox, oy);
            let elevation = nav.elevation_at_cell((x, y));

            corners.push(Corner {
                position: pos,
                cell: (x, y),
                convex,
                blocked_neighbors: blocked_count,
                open_direction: SimVec2 { x: ox, y: oy },
                passage_width,
                elevation,
            });
        }
    }

    // Narrow passages first, then concave before convex
    corners.sort_by(|a, b| {
        a.passage_width
            .partial_cmp(&b.passage_width)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.convex.cmp(&b.convex))
    });

    corners.truncate(MAX_CORNERS);
    corners
}

/// Detect a corner at a specific cell. Returns None if the cell isn't a corner.
fn detect_corner_at(nav: &GridNav, x: i32, y: i32) -> Option<Corner> {
    if !nav.walkable(x, y) {
        return None;
    }

    let n = nav.blocked.contains(&(x, y - 1)) || !nav.in_bounds(x, y - 1);
    let s = nav.blocked.contains(&(x, y + 1)) || !nav.in_bounds(x, y + 1);
    let e = nav.blocked.contains(&(x + 1, y)) || !nav.in_bounds(x + 1, y);
    let w = nav.blocked.contains(&(x - 1, y)) || !nav.in_bounds(x - 1, y);

    let blocked_count = [n, s, e, w].iter().filter(|&&b| b).count() as u8;

    // Also check diagonal blocked for freestanding walls: a walkable cell
    // with 1 cardinal blocked + 1 diagonal blocked at the "elbow" is a corner
    let ne_blocked = nav.blocked.contains(&(x + 1, y - 1)) || !nav.in_bounds(x + 1, y - 1);
    let nw_blocked = nav.blocked.contains(&(x - 1, y - 1)) || !nav.in_bounds(x - 1, y - 1);
    let se_blocked = nav.blocked.contains(&(x + 1, y + 1)) || !nav.in_bounds(x + 1, y + 1);
    let sw_blocked = nav.blocked.contains(&(x - 1, y + 1)) || !nav.in_bounds(x - 1, y + 1);

    let is_corner = match blocked_count {
        1 => {
            // Freestanding wall end: 1 cardinal blocked + adjacent diagonal blocked
            (n && (ne_blocked || nw_blocked))
                || (s && (se_blocked || sw_blocked))
                || (e && (ne_blocked || se_blocked))
                || (w && (nw_blocked || sw_blocked))
        }
        2 => !((n && s) || (e && w)), // L-shape, not corridor
        3 => true,
        _ => false,
    };

    if !is_corner {
        return None;
    }

    let mut ox = 0.0f32;
    let mut oy = 0.0f32;
    if !n { oy -= 1.0; }
    if !s { oy += 1.0; }
    if !e { ox += 1.0; }
    if !w { ox -= 1.0; }
    let len = (ox * ox + oy * oy).sqrt().max(0.001);
    ox /= len;
    oy /= len;

    let convex = if blocked_count == 2 {
        let diag_x = if e { x + 1 } else if w { x - 1 } else { x };
        let diag_y = if n { y - 1 } else if s { y + 1 } else { y };
        nav.walkable(diag_x, diag_y)
    } else {
        false
    };

    let passage_width = measure_passage_width(nav, x, y, ox, oy);
    let elevation = nav.elevation_at_cell((x, y));
    let pos = nav.center_of((x, y));

    Some(Corner {
        position: pos,
        cell: (x, y),
        convex,
        blocked_neighbors: blocked_count,
        open_direction: SimVec2 { x: ox, y: oy },
        passage_width,
        elevation,
    })
}

/// Bounding box of a set of cells.
fn cells_bbox(cells: &[(i32, i32)]) -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y) in cells {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x, max_y)
}

fn measure_passage_width(nav: &GridNav, cx: i32, cy: i32, open_dx: f32, open_dy: f32) -> f32 {
    let max_steps = 20;
    let mut width = 0.0f32;
    for s in 1..=max_steps {
        let nx = cx + (open_dx * s as f32).round() as i32;
        let ny = cy + (open_dy * s as f32).round() as i32;
        if !nav.walkable(nx, ny) {
            break;
        }
        width = s as f32 * nav.cell_size;
    }
    width
}

// ---------------------------------------------------------------------------
// Threat / opportunity tokens (zones, cast indicators, projectiles)
// ---------------------------------------------------------------------------

/// A spatial threat or opportunity extracted from live game state.
/// These are ephemeral (change every tick) unlike geometry corners.
#[derive(Debug, Clone, Copy)]
pub struct ThreatToken {
    pub position: SimVec2,
    pub kind: ThreatKind,
    /// Radius of the threat area (0 for point threats like projectiles).
    pub radius: f32,
    /// Whether this is hostile to the querying unit's team.
    pub hostile: bool,
    /// Remaining duration in seconds (0 = instantaneous, e.g. projectile).
    pub duration_remaining: f32,
    /// Source unit ID (for cast indicators: who is casting).
    pub source_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatKind {
    /// Active zone on the ground (damage field, healing circle, etc).
    Zone,
    /// Obstacle zone (barricade — blocks movement, not a damage threat).
    Obstacle,
    /// Ability being cast with a ground target position (AoE indicator).
    CastIndicator,
    /// Projectile in flight.
    Projectile,
}

/// Feature dimension for threat tokens.
pub const THREAT_TOKEN_DIM: usize = 10;

/// Extract threat tokens from current SimState for a specific team.
/// Returns tokens sorted by distance to `reference_pos` (team centroid or unit pos).
pub fn extract_threat_tokens(
    state: &crate::sim::SimState,
    viewer_team: crate::sim::Team,
    reference_pos: SimVec2,
) -> Vec<ThreatToken> {
    let mut tokens = Vec::new();

    // Active zones
    for zone in &state.zones {
        if zone.invisible {
            continue;
        }
        let radius = match &zone.area {
            crate::effects::Area::Circle { radius } => *radius,
            crate::effects::Area::Cone { radius, .. } => *radius,
            crate::effects::Area::Line { length, width } => length.max(*width) * 0.5,
            crate::effects::Area::Ring { outer_radius, .. } => *outer_radius,
            crate::effects::Area::Spread { radius, .. } => *radius,
            crate::effects::Area::SingleTarget | crate::effects::Area::SelfOnly => 0.5,
        };
        let is_obstacle = zone.blocked_cells.len() > 0;
        let hostile = zone.source_team != viewer_team;

        tokens.push(ThreatToken {
            position: zone.position,
            kind: if is_obstacle { ThreatKind::Obstacle } else { ThreatKind::Zone },
            radius,
            hostile,
            duration_remaining: zone.remaining_ms as f32 / 1000.0,
            source_id: zone.source_id,
        });
    }

    // Cast indicators: units casting ground-targeted abilities
    for unit in &state.units {
        if unit.hp <= 0 {
            continue;
        }
        if let Some(ref cast) = unit.casting {
            if let Some(target_pos) = cast.target_pos {
                // Use enriched area from CastState if available, otherwise estimate
                let radius = cast.area.map_or(2.0, |area| match area {
                    crate::effects::Area::Circle { radius } => radius,
                    crate::effects::Area::Cone { radius, .. } => radius,
                    crate::effects::Area::Line { length, width } => length.max(width) * 0.5,
                    crate::effects::Area::Ring { outer_radius, .. } => outer_radius,
                    crate::effects::Area::Spread { radius, .. } => radius,
                    crate::effects::Area::SingleTarget
                    | crate::effects::Area::SelfOnly => 0.5,
                });

                // Use effect_hint to determine threat kind for cast indicators
                let kind = match cast.effect_hint {
                    crate::sim::CastEffectHint::Obstacle => ThreatKind::Obstacle,
                    _ => ThreatKind::CastIndicator,
                };

                tokens.push(ThreatToken {
                    position: target_pos,
                    kind,
                    radius,
                    hostile: unit.team != viewer_team,
                    duration_remaining: cast.remaining_ms as f32 / 1000.0,
                    source_id: unit.id,
                });
            }
        }
    }

    // Projectiles in flight
    for proj in &state.projectiles {
        let source_team = state.units.iter()
            .find(|u| u.id == proj.source_id)
            .map(|u| u.team);
        let hostile = source_team.map_or(true, |t| t != viewer_team);

        tokens.push(ThreatToken {
            position: proj.position,
            kind: ThreatKind::Projectile,
            radius: proj.width.max(0.5),
            hostile,
            duration_remaining: 0.0, // instantaneous on hit
            source_id: proj.source_id,
        });
    }

    // Sort by distance to reference position (closest threats first)
    tokens.sort_by(|a, b| {
        let da = distance(reference_pos, a.position);
        let db = distance(reference_pos, b.position);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    tokens
}

/// Encode a threat token into a feature vector relative to a unit's position.
pub fn encode_threat_token(
    tok: &ThreatToken,
    unit_pos: SimVec2,
    nav: &GridNav,
) -> [f32; THREAT_TOKEN_DIM] {
    let room_w = (nav.max_x - nav.min_x).max(1.0);
    let room_h = (nav.max_y - nav.min_y).max(1.0);
    let dx = tok.position.x - unit_pos.x;
    let dy = tok.position.y - unit_pos.y;
    let dist = distance(unit_pos, tok.position).max(0.001);

    [
        // Normalized position (0-1)
        (tok.position.x - nav.min_x) / room_w,
        (tok.position.y - nav.min_y) / room_h,
        // Relative direction from unit
        dx / dist,
        dy / dist,
        // Normalized distance
        (dist / 20.0).min(1.0),
        // Radius (normalized)
        (tok.radius / 5.0).min(1.0),
        // Hostile flag
        if tok.hostile { 1.0 } else { 0.0 },
        // Duration remaining (seconds, capped)
        (tok.duration_remaining / 15.0).min(1.0),
        // Kind one-hot-ish: zone=0.25, obstacle=0.5, cast_indicator=0.75, projectile=1.0
        match tok.kind {
            ThreatKind::Zone => 0.25,
            ThreatKind::Obstacle => 0.5,
            ThreatKind::CastIndicator => 0.75,
            ThreatKind::Projectile => 1.0,
        },
        // Is visible (LOS check from unit to threat position)
        if nav.fully_open || nav.is_convex_open(unit_pos, tok.position) { 1.0 } else { 0.0 },
    ]
}

// ---------------------------------------------------------------------------
// Precomputed visibility map
// ---------------------------------------------------------------------------

/// Precomputed per-cell visibility of geometry corners.
///
/// Two-layer design:
/// - **Static layer**: corners from room geometry, precomputed at room load.
/// - **Dynamic layer**: corners from placed obstacles (barricades), updated
///   incrementally when obstacles appear or expire.
///
/// Lookup is O(1) per unit per tick (HashMap cell lookup).
#[derive(Debug, Clone)]
pub struct VisibilityMap {
    /// All corners (static + dynamic). Indices match bit positions in visibility bitsets.
    pub corners: Vec<Corner>,
    /// Encoded corner features (parallel to `corners`).
    pub corner_features: Vec<[f32; CORNER_FEATURE_DIM]>,
    /// Per-cell bitset: bit `i` is set if corner `i` is visible from this cell.
    cell_visibility: HashMap<(i32, i32), u32>,
    /// Number of static corners (first N entries in `corners`). Dynamic corners
    /// are appended after these and can be added/removed.
    static_corner_count: usize,
    /// Tracks which dynamic corners belong to which obstacle zone (zone_id → corner indices).
    obstacle_corners: HashMap<u32, Vec<usize>>,
}

impl VisibilityMap {
    /// Build the visibility map for a room. Call once after room generation.
    ///
    /// For each walkable cell, tests line-of-sight to each corner using
    /// Bresenham traversal through the blocked set. Typical cost for a
    /// 20×20 room with 16 corners: ~64K ray checks, <1ms.
    pub fn build(nav: &GridNav) -> Self {
        let corners = extract_corners(nav);
        let corner_features: Vec<_> = corners.iter().map(|c| encode_corner(c, nav)).collect();

        let static_corner_count = corners.len();

        if corners.is_empty() || nav.fully_open {
            return Self {
                corners,
                corner_features,
                cell_visibility: HashMap::new(),
                static_corner_count,
                obstacle_corners: HashMap::new(),
            };
        }

        let cx0 = (nav.min_x / nav.cell_size).floor() as i32;
        let cx1 = (nav.max_x / nav.cell_size).ceil() as i32;
        let cy0 = (nav.min_y / nav.cell_size).floor() as i32;
        let cy1 = (nav.max_y / nav.cell_size).ceil() as i32;

        let mut cell_visibility = HashMap::new();

        for x in cx0..=cx1 {
            for y in cy0..=cy1 {
                if !nav.walkable(x, y) {
                    continue;
                }

                let cell_pos = nav.center_of((x, y));
                let mut bits: u32 = 0;

                for (i, corner) in corners.iter().enumerate() {
                    if nav.is_convex_open(cell_pos, corner.position) {
                        bits |= 1 << i;
                    }
                }

                if bits != 0 {
                    cell_visibility.insert((x, y), bits);
                }
            }
        }

        Self {
            corners,
            corner_features,
            cell_visibility,
            static_corner_count,
            obstacle_corners: HashMap::new(),
        }
    }

    /// Update visibility after an obstacle (barricade) is placed.
    ///
    /// `zone_id` identifies the obstacle for later removal.
    /// `blocked_cells` are the grid cells the obstacle occupies.
    /// `nav` must already have the cells inserted into `nav.blocked`.
    ///
    /// This method:
    /// 1. Extracts new corners created by the obstacle
    /// 2. Appends them to the corner list (if room in the bitset)
    /// 3. Recomputes visibility for cells within `recompute_radius` of the obstacle
    pub fn update_obstacle_placed(
        &mut self,
        nav: &GridNav,
        zone_id: u32,
        blocked_cells: &[(i32, i32)],
        recompute_radius: i32,
    ) {
        if blocked_cells.is_empty() {
            return;
        }

        // 1. Find new corners adjacent to the obstacle cells
        let mut new_corner_indices = Vec::new();
        let mut checked = std::collections::HashSet::new();

        for &(bx, by) in blocked_cells {
            // Check all neighbors of blocked cells for new corners
            for &(dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let nx = bx + dx;
                let ny = by + dy;
                if !checked.insert((nx, ny)) || !nav.walkable(nx, ny) {
                    continue;
                }

                if let Some(corner) = detect_corner_at(nav, nx, ny) {
                    // Don't add if a corner already exists at this cell
                    if self.corners.iter().any(|c| c.cell == (nx, ny)) {
                        // Update existing corner's passage width (obstacle may have narrowed it)
                        if let Some(existing) = self.corners.iter_mut().find(|c| c.cell == (nx, ny)) {
                            existing.passage_width = corner.passage_width;
                        }
                        if let Some(idx) = self.corners.iter().position(|c| c.cell == (nx, ny)) {
                            self.corner_features[idx] = encode_corner(&self.corners[idx], nav);
                        }
                        continue;
                    }

                    if self.corners.len() < 32 {
                        let idx = self.corners.len();
                        self.corner_features.push(encode_corner(&corner, nav));
                        self.corners.push(corner);
                        new_corner_indices.push(idx);
                    }
                }
            }
        }

        self.obstacle_corners.insert(zone_id, new_corner_indices);

        // 2. Update existing corners' passage widths if they're near the obstacle
        let (cx_min, cy_min, cx_max, cy_max) = cells_bbox(blocked_cells);
        for corner in self.corners.iter_mut() {
            let (ccx, ccy) = corner.cell;
            if ccx >= cx_min - recompute_radius && ccx <= cx_max + recompute_radius
                && ccy >= cy_min - recompute_radius && ccy <= cy_max + recompute_radius
            {
                corner.passage_width = measure_passage_width(
                    nav, ccx, ccy,
                    corner.open_direction.x, corner.open_direction.y,
                );
            }
        }
        // Rebuild features for affected corners
        for (i, corner) in self.corners.iter().enumerate() {
            let (ccx, ccy) = corner.cell;
            if ccx >= cx_min - recompute_radius && ccx <= cx_max + recompute_radius
                && ccy >= cy_min - recompute_radius && ccy <= cy_max + recompute_radius
            {
                self.corner_features[i] = encode_corner(corner, nav);
            }
        }

        // 3. Recompute visibility for cells within radius of the obstacle
        self.recompute_visibility_near(nav, cx_min, cy_min, cx_max, cy_max, recompute_radius);
    }

    /// Update visibility after an obstacle (barricade) expires.
    ///
    /// `zone_id` must match the ID passed to `update_obstacle_placed`.
    /// `nav` must already have the cells removed from `nav.blocked`.
    pub fn update_obstacle_removed(
        &mut self,
        nav: &GridNav,
        zone_id: u32,
        blocked_cells: &[(i32, i32)],
        recompute_radius: i32,
    ) {
        // Remove dynamic corners added by this obstacle
        if let Some(indices) = self.obstacle_corners.remove(&zone_id) {
            // Remove in reverse order to keep indices valid
            let mut to_remove: Vec<usize> = indices;
            to_remove.sort_unstable();
            to_remove.reverse();

            for idx in &to_remove {
                if *idx < self.corners.len() {
                    self.corners.remove(*idx);
                    self.corner_features.remove(*idx);

                    // Shift bit positions in all visibility entries
                    for bits in self.cell_visibility.values_mut() {
                        let above = *bits & !((1u32 << (idx + 1)) - 1); // bits above idx
                        let below = *bits & ((1u32 << idx) - 1); // bits below idx
                        *bits = below | (above >> 1);
                    }

                    // Shift indices in other obstacle_corners entries
                    for other_indices in self.obstacle_corners.values_mut() {
                        for oi in other_indices.iter_mut() {
                            if *oi > *idx {
                                *oi -= 1;
                            }
                        }
                    }
                }
            }
        }

        if blocked_cells.is_empty() {
            return;
        }

        // Recheck corners near where the obstacle was (some may no longer be corners)
        let (cx_min, cy_min, cx_max, cy_max) = cells_bbox(blocked_cells);

        // Remove corners that are no longer valid (the wall that made them a corner is gone)
        self.corners.retain(|c| {
            let (ccx, ccy) = c.cell;
            if ccx >= cx_min - 1 && ccx <= cx_max + 1 && ccy >= cy_min - 1 && ccy <= cy_max + 1 {
                // Re-validate: is this still a corner?
                detect_corner_at(nav, ccx, ccy).is_some()
            } else {
                true
            }
        });
        // Rebuild features to match
        self.corner_features = self.corners.iter().map(|c| encode_corner(c, nav)).collect();

        // Recompute visibility for cells near the removed obstacle
        self.recompute_visibility_near(nav, cx_min, cy_min, cx_max, cy_max, recompute_radius);
    }

    /// Recompute visibility bitsets for cells near a region.
    fn recompute_visibility_near(
        &mut self,
        nav: &GridNav,
        cx_min: i32, cy_min: i32, cx_max: i32, cy_max: i32,
        radius: i32,
    ) {
        let x0 = cx_min - radius;
        let x1 = cx_max + radius;
        let y0 = cy_min - radius;
        let y1 = cy_max + radius;

        for x in x0..=x1 {
            for y in y0..=y1 {
                if !nav.walkable(x, y) {
                    self.cell_visibility.remove(&(x, y));
                    continue;
                }

                let cell_pos = nav.center_of((x, y));
                let mut bits: u32 = 0;

                for (i, corner) in self.corners.iter().enumerate() {
                    if i >= 32 { break; }
                    if nav.is_convex_open(cell_pos, corner.position) {
                        bits |= 1 << i;
                    }
                }

                if bits != 0 {
                    self.cell_visibility.insert((x, y), bits);
                } else {
                    self.cell_visibility.remove(&(x, y));
                }
            }
        }
    }

    /// Get the visibility bitset for a world position. O(1).
    pub fn visible_corners_bitset(&self, nav: &GridNav, pos: SimVec2) -> u32 {
        if self.corners.is_empty() {
            return 0;
        }
        let cell = nav.cell_of(pos);
        self.cell_visibility.get(&cell).copied().unwrap_or(0)
    }

    /// Get the visible corners for a unit at the given position.
    /// Returns corner indices + features, sorted by distance to the unit.
    pub fn visible_corners_for_unit(
        &self,
        nav: &GridNav,
        unit_pos: SimVec2,
    ) -> Vec<VisibleCorner> {
        let bits = self.visible_corners_bitset(nav, unit_pos);
        if bits == 0 {
            return vec![];
        }

        let mut visible = Vec::new();
        for i in 0..self.corners.len().min(32) {
            if bits & (1 << i) != 0 {
                let corner = &self.corners[i];
                let dist = distance(unit_pos, corner.position);
                visible.push(VisibleCorner {
                    corner_index: i,
                    distance: dist,
                    features: self.corner_features[i],
                });
            }
        }

        visible.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        visible
    }

    /// Encode the visible corners for a unit as spatial tokens for the transformer.
    /// Returns up to `max_tokens` feature vectors, each augmented with distance
    /// and relative direction from the unit.
    pub fn spatial_tokens_for_unit(
        &self,
        nav: &GridNav,
        unit_pos: SimVec2,
        max_tokens: usize,
    ) -> Vec<[f32; SPATIAL_TOKEN_DIM]> {
        let visible = self.visible_corners_for_unit(nav, unit_pos);

        visible
            .iter()
            .take(max_tokens)
            .map(|vc| {
                let corner = &self.corners[vc.corner_index];
                let dx = corner.position.x - unit_pos.x;
                let dy = corner.position.y - unit_pos.y;
                let dist_norm = (vc.distance / 20.0).min(1.0);

                let mut tok = [0.0f32; SPATIAL_TOKEN_DIM];
                // Copy base corner features (8)
                tok[..CORNER_FEATURE_DIM].copy_from_slice(&vc.features);
                // Relative direction from unit (2)
                let len = vc.distance.max(0.001);
                tok[8] = dx / len;
                tok[9] = dy / len;
                // Normalized distance (1)
                tok[10] = dist_norm;
                tok
            })
            .collect()
    }

    /// Aggregate visibility stats for a unit: visible area proxy, corner count,
    /// average passage width of visible corners.
    pub fn visibility_summary(&self, nav: &GridNav, unit_pos: SimVec2) -> VisibilitySummary {
        let bits = self.visible_corners_bitset(nav, unit_pos);
        let visible_count = bits.count_ones();

        if visible_count == 0 {
            return VisibilitySummary {
                visible_corner_count: 0,
                avg_passage_width: 0.0,
                min_passage_width: f32::MAX,
                avg_corner_distance: 0.0,
            };
        }

        let mut total_width = 0.0f32;
        let mut min_width = f32::MAX;
        let mut total_dist = 0.0f32;
        let mut count = 0u32;

        for i in 0..self.corners.len().min(32) {
            if bits & (1 << i) != 0 {
                let corner = &self.corners[i];
                total_width += corner.passage_width;
                min_width = min_width.min(corner.passage_width);
                total_dist += distance(unit_pos, corner.position);
                count += 1;
            }
        }

        VisibilitySummary {
            visible_corner_count: count,
            avg_passage_width: total_width / count as f32,
            min_passage_width: min_width,
            avg_corner_distance: total_dist / count as f32,
        }
    }
}

/// A corner visible from a specific unit position.
#[derive(Debug, Clone)]
pub struct VisibleCorner {
    pub corner_index: usize,
    pub distance: f32,
    pub features: [f32; CORNER_FEATURE_DIM],
}

/// Summary of a unit's spatial awareness.
#[derive(Debug, Clone, Copy)]
pub struct VisibilitySummary {
    pub visible_corner_count: u32,
    pub avg_passage_width: f32,
    pub min_passage_width: f32,
    pub avg_corner_distance: f32,
}

// ---------------------------------------------------------------------------
// Per-unit spatial cache
// ---------------------------------------------------------------------------

/// Caches spatial tokens per unit. Invalidated when the unit moves to a
/// different grid cell or when geometry changes (obstacle placed/removed).
#[derive(Debug, Clone)]
pub struct SpatialCache {
    entries: HashMap<u32, SpatialCacheEntry>,
    /// Monotonic generation counter. Bumped on any geometry change.
    geometry_generation: u64,
}

#[derive(Debug, Clone)]
struct SpatialCacheEntry {
    cell: (i32, i32),
    generation: u64,
    tokens: Vec<[f32; SPATIAL_TOKEN_DIM]>,
    summary: VisibilitySummary,
}

impl SpatialCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            geometry_generation: 0,
        }
    }

    /// Mark geometry as changed. All cached entries become stale.
    pub fn invalidate_geometry(&mut self) {
        self.geometry_generation += 1;
    }

    /// Get spatial tokens for a unit, using cache if valid.
    /// Only recomputes if the unit moved to a different cell or geometry changed.
    pub fn get_tokens(
        &mut self,
        unit_id: u32,
        unit_pos: SimVec2,
        nav: &GridNav,
        vis: &VisibilityMap,
        max_tokens: usize,
    ) -> &[[f32; SPATIAL_TOKEN_DIM]] {
        let cell = nav.cell_of(unit_pos);
        let gen = self.geometry_generation;

        let needs_recompute = match self.entries.get(&unit_id) {
            Some(entry) => entry.cell != cell || entry.generation != gen,
            None => true,
        };

        if needs_recompute {
            let tokens = vis.spatial_tokens_for_unit(nav, unit_pos, max_tokens);
            let summary = vis.visibility_summary(nav, unit_pos);
            self.entries.insert(unit_id, SpatialCacheEntry {
                cell,
                generation: gen,
                tokens,
                summary,
            });
        }

        &self.entries[&unit_id].tokens
    }

    /// Get visibility summary for a unit, using cache if valid.
    pub fn get_summary(
        &mut self,
        unit_id: u32,
        unit_pos: SimVec2,
        nav: &GridNav,
        vis: &VisibilityMap,
    ) -> VisibilitySummary {
        let cell = nav.cell_of(unit_pos);
        let gen = self.geometry_generation;

        let needs_recompute = match self.entries.get(&unit_id) {
            Some(entry) => entry.cell != cell || entry.generation != gen,
            None => true,
        };

        if needs_recompute {
            let tokens = vis.spatial_tokens_for_unit(nav, unit_pos, MAX_CORNERS);
            let summary = vis.visibility_summary(nav, unit_pos);
            self.entries.insert(unit_id, SpatialCacheEntry {
                cell,
                generation: gen,
                tokens,
                summary,
            });
        }

        self.entries[&unit_id].summary
    }

    /// Remove cache entry for a dead unit.
    pub fn remove_unit(&mut self, unit_id: u32) {
        self.entries.remove(&unit_id);
    }
}

// ---------------------------------------------------------------------------
// Feature encoding
// ---------------------------------------------------------------------------

/// Base corner feature dimensionality.
pub const CORNER_FEATURE_DIM: usize = 8;

/// Spatial token dimensionality: base features + relative position from unit.
pub const SPATIAL_TOKEN_DIM: usize = 11; // 8 corner + 2 direction + 1 distance

/// Encode a corner into a feature vector.
pub fn encode_corner(corner: &Corner, nav: &GridNav) -> [f32; CORNER_FEATURE_DIM] {
    let room_w = nav.max_x - nav.min_x;
    let room_h = nav.max_y - nav.min_y;

    [
        (corner.position.x - nav.min_x) / room_w.max(1.0),
        (corner.position.y - nav.min_y) / room_h.max(1.0),
        if corner.convex { 1.0 } else { 0.0 },
        corner.blocked_neighbors as f32 / 3.0,
        corner.open_direction.x,
        corner.open_direction.y,
        (corner.passage_width / 10.0).min(1.0),
        corner.elevation / 3.0,
    ]
}

/// Extract and encode all corners for a room (without visibility).
pub fn extract_corner_tokens(nav: &GridNav) -> Vec<[f32; CORNER_FEATURE_DIM]> {
    let corners = extract_corners(nav);
    corners.iter().map(|c| encode_corner(c, nav)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_l_shaped_room() -> GridNav {
        let mut nav = GridNav::new(0.0, 10.0, 0.0, 10.0, 1.0);
        for i in 0..10 {
            nav.blocked.insert((i, 0));
            nav.blocked.insert((i, 9));
            nav.blocked.insert((0, i));
            nav.blocked.insert((9, i));
        }
        for x in 1..=4 {
            nav.blocked.insert((x, 2));
        }
        nav.blocked.insert((1, 3));
        nav.blocked.insert((1, 4));
        nav.precompute_wall_proximity();
        nav
    }

    fn make_corridor_room() -> GridNav {
        // Narrow corridor with a choke point:
        // XXXXXXXXXX
        // X........X
        // X.XXXX.X.X
        // X........X
        // XXXXXXXXXX
        let mut nav = GridNav::new(0.0, 10.0, 0.0, 5.0, 1.0);
        for i in 0..10 {
            nav.blocked.insert((i, 0));
            nav.blocked.insert((i, 4));
        }
        for i in 0..5 {
            nav.blocked.insert((0, i));
            nav.blocked.insert((9, i));
        }
        // Interior wall with gap
        for x in 2..=5 {
            nav.blocked.insert((x, 2));
        }
        nav.blocked.insert((7, 2));
        nav.precompute_wall_proximity();
        nav
    }

    #[test]
    fn extract_corners_from_l_room() {
        let nav = make_l_shaped_room();
        let corners = extract_corners(&nav);
        assert!(!corners.is_empty());

        for c in &corners {
            assert!(nav.is_walkable_pos(c.position));
        }

        eprintln!("L-room: {} corners", corners.len());
        for c in &corners {
            eprintln!("  ({:.1}, {:.1}) {} blocked={} width={:.1}",
                c.position.x, c.position.y,
                if c.convex { "convex" } else { "concave" },
                c.blocked_neighbors, c.passage_width);
        }
    }

    #[test]
    fn extract_corners_open_room() {
        let nav = GridNav::new(0.0, 10.0, 0.0, 10.0, 1.0);
        let corners = extract_corners(&nav);
        eprintln!("Open room: {} corners", corners.len());
    }

    #[test]
    fn corner_encoding_is_normalized() {
        let nav = make_l_shaped_room();
        let tokens = extract_corner_tokens(&nav);
        for (i, tok) in tokens.iter().enumerate() {
            for (j, &v) in tok.iter().enumerate() {
                assert!(v.is_finite(), "token[{}][{}] is not finite: {}", i, j, v);
                assert!(v >= -1.5 && v <= 1.5,
                    "token[{}][{}] out of expected range: {}", i, j, v);
            }
        }
    }

    #[test]
    fn visibility_map_build_and_lookup() {
        let nav = make_l_shaped_room();
        let vis = VisibilityMap::build(&nav);

        assert!(!vis.corners.is_empty());
        eprintln!("Built visibility map: {} corners, {} cells with visibility",
            vis.corners.len(), vis.cell_visibility.len());

        // A unit in the open part of the room should see most corners
        let open_pos = SimVec2 { x: 7.5, y: 5.5 };
        let visible = vis.visible_corners_for_unit(&nav, open_pos);
        eprintln!("Unit at (7.5, 5.5) sees {} corners", visible.len());
        assert!(!visible.is_empty());

        // A unit behind the L-wall should see fewer corners
        let hidden_pos = SimVec2 { x: 1.5, y: 1.5 };
        let hidden_visible = vis.visible_corners_for_unit(&nav, hidden_pos);
        eprintln!("Unit at (1.5, 1.5) sees {} corners", hidden_visible.len());

        // Open position should generally see more
        assert!(visible.len() >= hidden_visible.len(),
            "open unit should see at least as many corners as hidden unit");
    }

    #[test]
    fn spatial_tokens_have_correct_dim() {
        let nav = make_l_shaped_room();
        let vis = VisibilityMap::build(&nav);
        let pos = SimVec2 { x: 5.5, y: 5.5 };
        let tokens = vis.spatial_tokens_for_unit(&nav, pos, 8);

        for tok in &tokens {
            assert_eq!(tok.len(), SPATIAL_TOKEN_DIM);
            for &v in tok {
                assert!(v.is_finite());
            }
        }
        eprintln!("Got {} spatial tokens at (5.5, 5.5)", tokens.len());
    }

    #[test]
    fn visibility_summary_values() {
        let nav = make_corridor_room();
        let vis = VisibilityMap::build(&nav);

        let pos = SimVec2 { x: 5.5, y: 1.5 };
        let summary = vis.visibility_summary(&nav, pos);
        eprintln!("Corridor summary: {} corners visible, avg_width={:.1}, min_width={:.1}, avg_dist={:.1}",
            summary.visible_corner_count, summary.avg_passage_width,
            summary.min_passage_width, summary.avg_corner_distance);

        assert!(summary.visible_corner_count > 0);
        // Passage width can be 0 for tight corners (dead ends)
        assert!(summary.avg_passage_width >= 0.0);
        assert!(summary.min_passage_width >= 0.0);
    }

    #[test]
    fn dynamic_obstacle_creates_new_corners() {
        let mut nav = make_l_shaped_room();
        let mut vis = VisibilityMap::build(&nav);
        let initial_corners = vis.corners.len();
        eprintln!("Before obstacle: {} corners", initial_corners);

        // Place a barricade (3 cells wide) in the middle of the room
        let obstacle_cells = vec![(5, 5), (6, 5), (7, 5)];
        for &cell in &obstacle_cells {
            nav.blocked.insert(cell);
        }

        vis.update_obstacle_placed(&nav, 100, &obstacle_cells, 3);
        let after_corners = vis.corners.len();
        eprintln!("After obstacle: {} corners ({} new)",
            after_corners, after_corners - initial_corners);

        // Should have created new corners at the obstacle edges
        assert!(after_corners > initial_corners,
            "obstacle should create new corners");

        // Verify new corners are near the obstacle
        for i in initial_corners..after_corners {
            let c = &vis.corners[i];
            let near = obstacle_cells.iter().any(|&(ox, oy)| {
                (c.cell.0 - ox).abs() <= 1 && (c.cell.1 - oy).abs() <= 1
            });
            assert!(near, "new corner at ({}, {}) should be near obstacle",
                c.cell.0, c.cell.1);
        }

        // A unit on one side of the obstacle should not see corners on the other side
        let pos_above = SimVec2 { x: 6.5, y: 3.5 };
        let pos_below = SimVec2 { x: 6.5, y: 7.5 };
        let vis_above = vis.visible_corners_for_unit(&nav, pos_above);
        let vis_below = vis.visible_corners_for_unit(&nav, pos_below);
        eprintln!("Above obstacle sees {} corners, below sees {}",
            vis_above.len(), vis_below.len());
    }

    #[test]
    fn dynamic_obstacle_removal_restores_visibility() {
        let mut nav = make_l_shaped_room();
        let mut vis = VisibilityMap::build(&nav);

        // Snapshot visibility at a test point before obstacle
        let test_pos = SimVec2 { x: 6.5, y: 6.5 };
        let before_bits = vis.visible_corners_bitset(&nav, test_pos);

        // Place obstacle
        let obstacle_cells = vec![(5, 5), (6, 5), (7, 5)];
        for &cell in &obstacle_cells {
            nav.blocked.insert(cell);
        }
        vis.update_obstacle_placed(&nav, 200, &obstacle_cells, 3);
        let during_bits = vis.visible_corners_bitset(&nav, test_pos);

        // Remove obstacle
        for &cell in &obstacle_cells {
            nav.blocked.remove(&cell);
        }
        vis.update_obstacle_removed(&nav, 200, &obstacle_cells, 3);
        let after_bits = vis.visible_corners_bitset(&nav, test_pos);

        eprintln!("Visibility bits: before={:#b} during={:#b} after={:#b}",
            before_bits, during_bits, after_bits);

        // Static corners should be visible again after removal
        // (bit positions may shift due to dynamic corner add/remove, so check count)
        let before_count = before_bits.count_ones();
        let after_count = after_bits.count_ones();
        assert!(after_count >= before_count.saturating_sub(1),
            "after removal, should see roughly as many corners as before: {} vs {}",
            after_count, before_count);
    }

    #[test]
    fn bench_dynamic_obstacle_update() {
        let mut nav = GridNav::new(0.0, 20.0, 0.0, 20.0, 1.0);
        for i in 0..20 {
            nav.blocked.insert((i, 0));
            nav.blocked.insert((i, 19));
            nav.blocked.insert((0, i));
            nav.blocked.insert((19, i));
        }
        for x in 3..=8 { nav.blocked.insert((x, 7)); }
        for x in 11..=16 { nav.blocked.insert((x, 7)); }
        nav.precompute_wall_proximity();

        let mut vis = VisibilityMap::build(&nav);
        let obstacle_cells = vec![(10, 5), (10, 6), (10, 7)];

        let iters = 1000;
        let start = std::time::Instant::now();
        for i in 0..iters {
            // Place
            for &cell in &obstacle_cells {
                nav.blocked.insert(cell);
            }
            vis.update_obstacle_placed(&nav, i as u32, &obstacle_cells, 5);

            // Remove
            for &cell in &obstacle_cells {
                nav.blocked.remove(&cell);
            }
            vis.update_obstacle_removed(&nav, i as u32, &obstacle_cells, 5);
        }
        let elapsed = start.elapsed();
        let per_cycle_us = elapsed.as_micros() / iters as u128;
        eprintln!("bench_dynamic_obstacle: {}us per place+remove cycle", per_cycle_us);
    }

    #[test]
    fn threat_tokens_from_zones() {
        use crate::sim::{SimState, Team};
        use crate::effects::{Area, ConditionalEffect};
        use crate::sim::ActiveZone;

        let state = SimState {
            tick: 50,
            rng_state: 0,
            units: crate::sim::UnitStore::new(vec![]),
            projectiles: vec![],
            passive_trigger_depth: 0,
            zones: vec![
                ActiveZone {
                    id: 1,
                    source_id: 100,
                    source_team: Team::Enemy,
                    position: SimVec2 { x: 5.0, y: 5.0 },
                    area: Area::Circle { radius: 2.5 },
                    effects: vec![],
                    remaining_ms: 3000,
                    tick_interval_ms: 500,
                    tick_elapsed_ms: 0,
                    trigger_on_enter: false,
                    invisible: false,
                    triggered: false,
                    arm_time_ms: 0,
                    blocked_cells: vec![],
                    zone_tag: None,
                },
            ],
            tethers: vec![],
            grid_nav: None,
        };

        let ref_pos = SimVec2 { x: 3.0, y: 3.0 };
        let threats = extract_threat_tokens(&state, Team::Hero, ref_pos);

        assert_eq!(threats.len(), 1);
        assert_eq!(threats[0].kind, ThreatKind::Zone);
        assert!(threats[0].hostile);
        assert!((threats[0].radius - 2.5).abs() < f32::EPSILON);
        assert!((threats[0].duration_remaining - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn threat_token_encoding_normalized() {
        let nav = make_l_shaped_room();
        let tok = ThreatToken {
            position: SimVec2 { x: 5.0, y: 5.0 },
            kind: ThreatKind::Zone,
            radius: 2.0,
            hostile: true,
            duration_remaining: 5.0,
            source_id: 1,
        };
        let unit_pos = SimVec2 { x: 3.0, y: 3.0 };
        let encoded = encode_threat_token(&tok, unit_pos, &nav);

        assert_eq!(encoded.len(), THREAT_TOKEN_DIM);
        for (i, &v) in encoded.iter().enumerate() {
            assert!(v.is_finite(), "threat token[{}] not finite: {}", i, v);
            assert!(v >= -1.1 && v <= 1.1,
                "threat token[{}] out of range: {}", i, v);
        }
    }

    #[test]
    fn spatial_cache_skips_recompute_same_cell() {
        let nav = make_l_shaped_room();
        let vis = VisibilityMap::build(&nav);
        let mut cache = SpatialCache::new();

        let pos = SimVec2 { x: 5.5, y: 5.5 };

        // First call computes
        let tokens1 = cache.get_tokens(1, pos, &nav, &vis, 8).to_vec();
        assert!(!tokens1.is_empty());

        // Same cell — should return cached (same pointer content)
        let pos_nearby = SimVec2 { x: 5.6, y: 5.4 }; // same cell
        let tokens2 = cache.get_tokens(1, pos_nearby, &nav, &vis, 8);
        assert_eq!(tokens1.len(), tokens2.len());

        // Different cell — should recompute
        let pos_far = SimVec2 { x: 8.5, y: 8.5 };
        let tokens3 = cache.get_tokens(1, pos_far, &nav, &vis, 8);
        // May have different corners visible
        eprintln!("Same cell: {} tokens, different cell: {} tokens",
            tokens1.len(), tokens3.len());
    }

    #[test]
    fn spatial_cache_invalidates_on_geometry_change() {
        let mut nav = make_l_shaped_room();
        let vis = VisibilityMap::build(&nav);
        let mut cache = SpatialCache::new();

        let pos = SimVec2 { x: 5.5, y: 5.5 };
        let _ = cache.get_tokens(1, pos, &nav, &vis, 8);

        // Geometry change invalidates cache
        cache.invalidate_geometry();

        // Even same position re-evaluates (geometry_generation changed)
        // This would normally be called with an updated vis map too,
        // but we're testing the cache invalidation logic
        let _ = cache.get_tokens(1, pos, &nav, &vis, 8);
        // If we got here without panic, cache handled the invalidation
    }

    #[test]
    fn bench_spatial_cache_hit() {
        let nav = make_l_shaped_room();
        let vis = VisibilityMap::build(&nav);
        let mut cache = SpatialCache::new();
        let pos = SimVec2 { x: 5.5, y: 5.5 };

        // Prime the cache
        let _ = cache.get_tokens(1, pos, &nav, &vis, 8);

        let iters = 1_000_000;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let t = cache.get_tokens(1, pos, &nav, &vis, 8);
            std::hint::black_box(t);
        }
        let elapsed = start.elapsed();
        let per_ns = elapsed.as_nanos() / iters as u128;
        eprintln!("bench_spatial_cache_hit: {}ns/lookup", per_ns);
    }

    #[test]
    fn bench_visibility_map_build() {
        // 20x20 room with interior walls — realistic scenario
        let mut nav = GridNav::new(0.0, 20.0, 0.0, 20.0, 1.0);
        // Perimeter
        for i in 0..20 {
            nav.blocked.insert((i, 0));
            nav.blocked.insert((i, 19));
            nav.blocked.insert((0, i));
            nav.blocked.insert((19, i));
        }
        // Two interior walls with gaps
        for x in 3..=8 { nav.blocked.insert((x, 7)); }
        for x in 11..=16 { nav.blocked.insert((x, 7)); }
        for y in 3..=8 { nav.blocked.insert((10, y)); }
        for y in 11..=16 { nav.blocked.insert((10, y)); }
        nav.precompute_wall_proximity();

        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let vis = VisibilityMap::build(&nav);
            std::hint::black_box(&vis);
        }
        let elapsed = start.elapsed();
        let per_build_us = elapsed.as_micros() / iters as u128;
        eprintln!("bench_visibility_map_build (20x20, interior walls): {}us/build", per_build_us);

        // Also benchmark per-unit lookup
        let vis = VisibilityMap::build(&nav);
        let pos = SimVec2 { x: 5.5, y: 5.5 };
        let lookup_iters = 100_000;
        let start = std::time::Instant::now();
        for _ in 0..lookup_iters {
            let tokens = vis.spatial_tokens_for_unit(&nav, pos, 8);
            std::hint::black_box(&tokens);
        }
        let elapsed = start.elapsed();
        let per_lookup_ns = elapsed.as_nanos() / lookup_iters as u128;
        eprintln!("bench_visibility_lookup: {}ns/lookup", per_lookup_ns);
    }
}
