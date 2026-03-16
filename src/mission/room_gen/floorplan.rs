//! Mission floorplan generator: places multiple rooms on a 2D grid
//! connected by corridors, forming a complete mission layout.
//!
//! Output is a single large NavGrid with obstacle types, elevations,
//! room boundaries, and per-room spawn zones.

use super::lcg::{Lcg, ObstacleRegion, OBS_FLOOR, OBS_WALL};
use super::nav::NavGrid;
use super::templates;
use crate::ai::core::SimVec2;

// ---------------------------------------------------------------------------
// Floorplan types
// ---------------------------------------------------------------------------

/// A placed room within the floorplan.
#[derive(Debug, Clone)]
pub struct PlacedRoom {
    /// Top-left corner in floorplan grid coordinates.
    pub col: usize,
    pub row: usize,
    /// Interior dimensions (excluding walls).
    pub width: usize,
    pub height: usize,
    /// Role in the mission pacing.
    pub role: RoomRole,
    /// Base elevation of the room floor (0.0 = ground level).
    pub elevation: f32,
    /// Player spawn positions (world-space within floorplan).
    pub player_spawns: Vec<SimVec2>,
    /// Enemy spawn positions.
    pub enemy_spawns: Vec<SimVec2>,
}

impl PlacedRoom {
    /// Center point of the room in grid coordinates.
    fn center(&self) -> (usize, usize) {
        (self.col + self.width / 2, self.row + self.height / 2)
    }

    /// Bounding rect including 1-cell walls: [col-1, row-1] to [col+width, row+height].
    fn bounds_with_walls(&self) -> (usize, usize, usize, usize) {
        (
            self.col.saturating_sub(1),
            self.row.saturating_sub(1),
            self.col + self.width,
            self.row + self.height,
        )
    }
}

/// Role determines pacing, enemy difficulty, and obstacle density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomRole {
    Entry,
    Combat,
    Pressure,
    Recovery,
    Climax,
}

/// A corridor connecting two rooms.
#[derive(Debug, Clone)]
pub struct Corridor {
    pub from_room: usize,
    pub to_room: usize,
    /// Corridor cells (col, row) — 3 cells wide.
    pub cells: Vec<(usize, usize)>,
}

/// Complete mission floorplan.
#[derive(Debug, Clone)]
pub struct Floorplan {
    /// The full grid.
    pub nav: NavGrid,
    /// Obstacle regions for Bevy visual spawning.
    pub obstacles: Vec<ObstacleRegion>,
    /// Rooms placed in the floorplan.
    pub rooms: Vec<PlacedRoom>,
    /// Corridors connecting rooms.
    pub corridors: Vec<Corridor>,
    /// Total grid dimensions.
    pub total_width: usize,
    pub total_height: usize,
    /// RNG seed used.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Generation parameters
// ---------------------------------------------------------------------------

/// Configuration for floorplan generation.
pub struct FloorplanConfig {
    /// Number of rooms to place.
    pub room_count: usize,
    /// Min/max interior room dimensions.
    pub room_min_size: usize,
    pub room_max_size: usize,
    /// Total floorplan grid size.
    pub grid_width: usize,
    pub grid_height: usize,
    /// Corridor width (cells).
    pub corridor_width: usize,
    /// Minimum gap between rooms (cells).
    pub room_gap: usize,
}

impl Default for FloorplanConfig {
    fn default() -> Self {
        Self {
            room_count: 5,
            room_min_size: 12,
            room_max_size: 28,
            grid_width: 80,
            grid_height: 60,
            corridor_width: 3,
            room_gap: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Floorplan generation
// ---------------------------------------------------------------------------

/// Generate a complete mission floorplan.
pub fn generate_floorplan(seed: u64, config: &FloorplanConfig) -> Floorplan {
    let mut rng = Lcg::new(seed);
    let gw = config.grid_width;
    let gh = config.grid_height;

    // Start with everything as wall (blocked).
    let mut nav = NavGrid::new(gw, gh, 1.0);
    for i in 0..nav.walkable.len() {
        nav.walkable[i] = false;
    }

    // 1. Place rooms using random placement with rejection.
    let rooms = place_rooms(&mut rng, config);

    // 2. Carve room interiors into the grid.
    for room in &rooms {
        carve_room(&mut nav, room);
    }

    // 3. Connect rooms with corridors (MST + optional extra edges).
    let corridors = connect_rooms(&mut rng, &rooms, config);

    // 4. Carve corridors into the grid (with elevation ramps).
    for corridor in &corridors {
        carve_corridor(&mut nav, corridor, config.corridor_width, &rooms);
    }

    // 5. Add stealth bypass passages and alcoves.
    add_bypass_passages(&mut nav, &mut rng, &rooms, &corridors);
    add_corridor_alcoves(&mut nav, &mut rng, &corridors, config.corridor_width);

    // 7. Place tactical obstacles inside each room.
    let mut obstacles = Vec::new();
    for room in &rooms {
        let room_obs = fill_room_obstacles(&mut nav, &mut rng, room);
        obstacles.extend(room_obs);
    }

    // 8. Add intra-room multi-level features to larger rooms.
    for room in &rooms {
        if room.width >= 14 && room.height >= 14 {
            let feature = rng.next_usize_range(0, 5);
            match feature {
                0 => add_balcony(&mut nav, &mut obstacles, &mut rng, room),
                1 => add_sunken_pit(&mut nav, &mut rng, room),
                2 => add_split_level(&mut nav, &mut obstacles, &mut rng, room),
                3 => add_overlook(&mut nav, &mut obstacles, &mut rng, room),
                _ => {} // no multi-level feature (~40% of rooms stay flat)
            }
        }
    }

    // 9. Place spawns per room based on role.
    let mut rooms_with_spawns = rooms;
    for room in &mut rooms_with_spawns {
        place_room_spawns(&nav, &mut rng, room);
    }

    Floorplan {
        nav,
        obstacles,
        rooms: rooms_with_spawns,
        corridors,
        total_width: gw,
        total_height: gh,
        seed,
    }
}

// ---------------------------------------------------------------------------
// Room placement
// ---------------------------------------------------------------------------

fn place_rooms(rng: &mut Lcg, config: &FloorplanConfig) -> Vec<PlacedRoom> {
    let mut rooms = Vec::new();
    let margin = 2; // Keep rooms away from grid edge

    for i in 0..config.room_count * 20 {
        if rooms.len() >= config.room_count {
            break;
        }

        let w = rng.next_usize_range(config.room_min_size, config.room_max_size);
        let h = rng.next_usize_range(config.room_min_size, config.room_max_size);

        // Random position (with margin for walls)
        let max_col = config.grid_width.saturating_sub(w + margin + 1);
        let max_row = config.grid_height.saturating_sub(h + margin + 1);
        if max_col <= margin + 1 || max_row <= margin + 1 {
            continue;
        }

        let col = rng.next_usize_range(margin + 1, max_col);
        let row = rng.next_usize_range(margin + 1, max_row);

        // Check overlap with existing rooms (including gap).
        let gap = config.room_gap;
        let overlaps = rooms.iter().any(|r: &PlacedRoom| {
            let (ac0, ar0, ac1, ar1) = (
                col.saturating_sub(gap),
                row.saturating_sub(gap),
                col + w + gap,
                row + h + gap,
            );
            let (bc0, br0, bc1, br1) = (r.col, r.row, r.col + r.width, r.row + r.height);
            ac0 < bc1 && ac1 > bc0 && ar0 < br1 && ar1 > br0
        });

        if !overlaps {
            rooms.push(PlacedRoom {
                col,
                row,
                width: w,
                height: h,
                role: RoomRole::Combat, // assigned later
                elevation: 0.0,         // assigned later
                player_spawns: Vec::new(),
                enemy_spawns: Vec::new(),
            });
        }
    }

    // Assign roles based on position: leftmost = Entry, rightmost = Climax,
    // middle rooms alternate Combat/Pressure/Recovery.
    if !rooms.is_empty() {
        rooms.sort_by_key(|r| r.col);
        rooms[0].role = RoomRole::Entry;
        if rooms.len() > 1 {
            rooms.last_mut().unwrap().role = RoomRole::Climax;
        }
        for i in 1..rooms.len().saturating_sub(1) {
            rooms[i].role = match i % 3 {
                0 => RoomRole::Pressure,
                1 => RoomRole::Combat,
                _ => RoomRole::Recovery,
            };
        }
    }

    // Assign elevation based on role + random variation.
    // Creates a general upward progression from entry to climax
    // with variation to avoid monotony.
    let n = rooms.len();
    for (i, room) in rooms.iter_mut().enumerate() {
        let base = match room.role {
            RoomRole::Entry => 0.0,
            RoomRole::Recovery => 0.5,
            RoomRole::Combat => rng.next_f32_range(0.5, 1.5),
            RoomRole::Pressure => rng.next_f32_range(1.0, 2.0),
            RoomRole::Climax => 2.0,
        };
        // Add small random jitter
        let jitter = rng.next_f32_range(-0.25, 0.25);
        room.elevation = (base + jitter).max(0.0);
    }

    rooms
}

// ---------------------------------------------------------------------------
// Grid carving
// ---------------------------------------------------------------------------

fn carve_room(nav: &mut NavGrid, room: &PlacedRoom) {
    // Carve interior as walkable and set base elevation.
    for r in room.row..room.row + room.height {
        for c in room.col..room.col + room.width {
            if c < nav.cols && r < nav.rows {
                let idx = r * nav.cols + c;
                nav.walkable[idx] = true;
                nav.elevation[idx] = room.elevation;
            }
        }
    }
}

fn carve_corridor(
    nav: &mut NavGrid,
    corridor: &Corridor,
    width: usize,
    rooms: &[PlacedRoom],
) {
    let half = width / 2;
    let n_cells = corridor.cells.len();
    if n_cells == 0 {
        return;
    }

    // Interpolate elevation along the corridor between source and dest rooms.
    let elev_from = rooms[corridor.from_room].elevation;
    let elev_to = rooms[corridor.to_room].elevation;

    for (i, &(cc, cr)) in corridor.cells.iter().enumerate() {
        let t = if n_cells > 1 { i as f32 / (n_cells - 1) as f32 } else { 0.5 };
        let elev = elev_from + (elev_to - elev_from) * t;

        for dr in 0..width {
            for dc in 0..width {
                let c = cc.saturating_sub(half) + dc;
                let r = cr.saturating_sub(half) + dr;
                if c > 0 && r > 0 && c < nav.cols - 1 && r < nav.rows - 1 {
                    let idx = r * nav.cols + c;
                    nav.walkable[idx] = true;
                    nav.elevation[idx] = elev;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Room connectivity (MST via Prim's algorithm)
// ---------------------------------------------------------------------------

fn connect_rooms(
    rng: &mut Lcg,
    rooms: &[PlacedRoom],
    config: &FloorplanConfig,
) -> Vec<Corridor> {
    let n = rooms.len();
    if n < 2 {
        return Vec::new();
    }

    // Prim's MST on room centers.
    let mut in_tree = vec![false; n];
    let mut corridors = Vec::new();
    in_tree[0] = true;

    for _ in 1..n {
        let mut best_from = 0;
        let mut best_to = 0;
        let mut best_dist = usize::MAX;

        for i in 0..n {
            if !in_tree[i] {
                continue;
            }
            for j in 0..n {
                if in_tree[j] {
                    continue;
                }
                let (cx1, cy1) = rooms[i].center();
                let (cx2, cy2) = rooms[j].center();
                let dist = cx1.abs_diff(cx2) + cy1.abs_diff(cy2);
                if dist < best_dist {
                    best_dist = dist;
                    best_from = i;
                    best_to = j;
                }
            }
        }

        in_tree[best_to] = true;
        let cells = build_corridor_path(rooms, best_from, best_to);
        corridors.push(Corridor {
            from_room: best_from,
            to_room: best_to,
            cells,
        });
    }

    // Add extra connections for loop routes (critical for flanking/stealth).
    // Target: ~50% more edges than the MST minimum.
    let extra_count = (n / 2).max(1);
    for _ in 0..extra_count {
        let a = rng.next_usize_range(0, n - 1);
        let b = rng.next_usize_range(0, n - 1);
        if a != b {
            let already_connected = corridors
                .iter()
                .any(|c| (c.from_room == a && c.to_room == b) || (c.from_room == b && c.to_room == a));
            if !already_connected {
                let cells = build_corridor_path(rooms, a, b);
                corridors.push(Corridor {
                    from_room: a,
                    to_room: b,
                    cells,
                });
            }
        }
    }

    corridors
}

/// Build an L-shaped corridor path between two room centers.
fn build_corridor_path(
    rooms: &[PlacedRoom],
    from: usize,
    to: usize,
) -> Vec<(usize, usize)> {
    let (x1, y1) = rooms[from].center();
    let (x2, y2) = rooms[to].center();

    let mut cells = Vec::new();

    // Go horizontal first, then vertical (L-shaped).
    let (sx, ex) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
    for c in sx..=ex {
        cells.push((c, y1));
    }
    let (sy, ey) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
    for r in sy..=ey {
        cells.push((x2, r));
    }

    cells
}

// ---------------------------------------------------------------------------
// Room interior obstacle placement
// ---------------------------------------------------------------------------

fn fill_room_obstacles(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    room: &PlacedRoom,
) -> Vec<ObstacleRegion> {
    // Create a temporary sub-NavGrid for the room, generate obstacles,
    // then stamp them into the floorplan grid at the room's offset.
    let w = room.width;
    let h = room.height;
    let mut sub_nav = NavGrid::new(w, h, 1.0);

    // Perimeter of sub_nav stays walkable (room interior has no walls —
    // the floorplan grid's surrounding blocked cells act as walls).
    // But we do want obstacles in the interior.

    let margin = templates::spawn_margin(w);
    let play_lo = margin;
    let play_hi = w.saturating_sub(margin);
    let play_w = play_hi.saturating_sub(play_lo);

    if play_w < 4 || h < 6 {
        return Vec::new();
    }

    let sub_obstacles = templates::generate_tactical_obstacles(&mut sub_nav, rng);

    // Stamp sub_nav blocked cells into floorplan nav at room offset.
    let mut obstacles = Vec::new();
    for obs in &sub_obstacles {
        let fc0 = room.col + obs.col0;
        let fr0 = room.row + obs.row0;
        let fc1 = room.col + obs.col1;
        let fr1 = room.row + obs.row1;

        // Bounds check
        if fc1 >= nav.cols || fr1 >= nav.rows {
            continue;
        }

        for r in fr0..=fr1 {
            for c in fc0..=fc1 {
                let idx = r * nav.cols + c;
                nav.walkable[idx] = false;
            }
        }

        obstacles.push(ObstacleRegion {
            col0: fc0,
            col1: fc1,
            row0: fr0,
            row1: fr1,
            height: obs.height,
            obs_type: obs.obs_type,
        });
    }

    obstacles
}

// ---------------------------------------------------------------------------
// Stealth bypass passages and alcoves
// ---------------------------------------------------------------------------

/// Add narrow (1-2 cell wide) bypass passages that connect non-adjacent rooms,
/// creating alternate routes that circumvent main corridors.
fn add_bypass_passages(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    rooms: &[PlacedRoom],
    existing_corridors: &[Corridor],
) {
    let n = rooms.len();
    if n < 3 {
        return;
    }

    // Try to add 1-2 narrow bypass routes between rooms that aren't directly connected.
    let attempts = n;
    let mut added = 0usize;

    for _ in 0..attempts {
        if added >= 2 {
            break;
        }

        let a = rng.next_usize_range(0, n - 1);
        let b = rng.next_usize_range(0, n - 1);
        if a == b {
            continue;
        }

        // Only bypass rooms that are NOT directly connected (that's the point).
        let directly_connected = existing_corridors.iter().any(|c| {
            (c.from_room == a && c.to_room == b) || (c.from_room == b && c.to_room == a)
        });
        if directly_connected {
            continue;
        }

        // Build a narrow L-shaped path between room edges.
        let (x1, y1) = rooms[a].center();
        let (x2, y2) = rooms[b].center();

        // Offset the path so it doesn't overlap main corridors.
        let offset = if rng.next_u64() % 2 == 0 { 3isize } else { -3 };
        let mid_y = ((y1 as isize + y2 as isize) / 2 + offset)
            .clamp(2, nav.rows as isize - 3) as usize;

        // Carve 1-cell-wide passage (narrow = stealth only)
        let (sx, ex) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        let mut carved = false;
        for c in sx..=ex {
            if c > 0 && c < nav.cols - 1 && mid_y > 0 && mid_y < nav.rows - 1 {
                let idx = mid_y * nav.cols + c;
                if !nav.walkable[idx] {
                    nav.walkable[idx] = true;
                    nav.elevation[idx] = (rooms[a].elevation + rooms[b].elevation) / 2.0;
                    carved = true;
                }
            }
        }

        // Vertical segments connecting to the horizontal
        for &(rx, ry) in &[(x1, y1), (x2, y2)] {
            let (sy, ey) = if ry <= mid_y { (ry, mid_y) } else { (mid_y, ry) };
            for r in sy..=ey {
                if rx > 0 && rx < nav.cols - 1 && r > 0 && r < nav.rows - 1 {
                    let idx = r * nav.cols + rx;
                    if !nav.walkable[idx] {
                        nav.walkable[idx] = true;
                        // Interpolate elevation
                        let t = if ey > sy { (r - sy) as f32 / (ey - sy) as f32 } else { 0.5 };
                        let from_elev = rooms[a].elevation;
                        let to_elev = rooms[b].elevation;
                        nav.elevation[idx] = from_elev + (to_elev - from_elev) * t;
                        carved = true;
                    }
                }
            }
        }

        if carved {
            added += 1;
        }
    }
}

/// Add small alcoves (2×2 or 2×3 nooks) off corridor walls.
/// These provide hiding spots for stealth units and ambush positions.
fn add_corridor_alcoves(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    corridors: &[Corridor],
    corridor_width: usize,
) {
    for corridor in corridors {
        if corridor.cells.len() < 10 {
            continue; // too short for alcoves
        }

        // Add 1-2 alcoves per corridor
        let n_alcoves = rng.next_usize_range(1, 2);
        for _ in 0..n_alcoves {
            // Pick a random point along the corridor
            let idx = rng.next_usize_range(3, corridor.cells.len() - 3);
            let (cc, cr) = corridor.cells[idx];

            // Pick a side to place the alcove (perpendicular to corridor direction)
            let (dc, dr): (isize, isize) = if idx > 0 {
                let (pc, pr) = corridor.cells[idx - 1];
                if pc != cc {
                    // Corridor runs horizontal → alcove goes up or down
                    (0, if rng.next_u64() % 2 == 0 { -1 } else { 1 })
                } else {
                    // Corridor runs vertical → alcove goes left or right
                    (if rng.next_u64() % 2 == 0 { -1 } else { 1 }, 0)
                }
            } else {
                continue;
            };

            // Carve a 2×3 alcove perpendicular to the corridor
            let half_w = corridor_width as isize / 2;
            let alcove_depth = rng.next_usize_range(2, 3) as isize;

            for d in 1..=alcove_depth {
                for w in -1..=1 {
                    let ac = (cc as isize + dc * (half_w + d) + dr.abs() * w) as usize;
                    let ar = (cr as isize + dr * (half_w + d) + dc.abs() * w) as usize;
                    if ac > 0 && ar > 0 && ac < nav.cols - 1 && ar < nav.rows - 1 {
                        let nav_idx = ar * nav.cols + ac;
                        nav.walkable[nav_idx] = true;
                        // Match elevation of corridor cell
                        nav.elevation[nav_idx] = nav.elevation[cr * nav.cols + cc];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra-room multi-level features
// ---------------------------------------------------------------------------

/// Raised balcony along one side — elevated strip with a ledge wall
/// providing cover for units on the lower floor.
fn add_balcony(
    nav: &mut NavGrid,
    obstacles: &mut Vec<ObstacleRegion>,
    rng: &mut Lcg,
    room: &PlacedRoom,
) {
    let cols = nav.cols;
    let rows = nav.rows;
    let raised = room.elevation + 1.0;

    // Pick a side: 0=top, 1=bottom, 2=left, 3=right
    let side = rng.next_usize_range(0, 3);
    let depth = (room.width.min(room.height) / 4).clamp(3, 5);

    let (bc0, br0, bc1, br1) = match side {
        0 => (room.col + 1, room.row + 1, room.col + room.width - 2, room.row + depth),
        1 => (room.col + 1, room.row + room.height - depth - 1, room.col + room.width - 2, room.row + room.height - 2),
        2 => (room.col + 1, room.row + 1, room.col + depth, room.row + room.height - 2),
        _ => (room.col + room.width - depth - 1, room.row + 1, room.col + room.width - 2, room.row + room.height - 2),
    };

    // Set elevation on balcony cells
    for r in br0..=br1 {
        for c in bc0..=bc1 {
            if c < cols && r < rows && nav.walkable[r * cols + c] {
                nav.elevation[r * cols + c] = raised;
            }
        }
    }

    // Place ledge wall along the balcony edge facing the room interior
    let (lc0, lr0, lc1, lr1) = match side {
        0 => (bc0, br1 + 1, bc1, br1 + 1),      // bottom edge of balcony
        1 => (bc0, br0.saturating_sub(1), bc1, br0.saturating_sub(1)), // top edge
        2 => (bc1 + 1, br0, bc1 + 1, br1),       // right edge
        _ => (bc0.saturating_sub(1), br0, bc0.saturating_sub(1), br1), // left edge
    };

    if lc0 < cols && lr0 < rows && lc1 < cols && lr1 < rows {
        for r in lr0..=lr1 {
            for c in lc0..=lc1 {
                let idx = r * cols + c;
                nav.walkable[idx] = false;
            }
        }
        obstacles.push(ObstacleRegion {
            col0: lc0, col1: lc1, row0: lr0, row1: lr1,
            height: 0.6, obs_type: OBS_WALL,
        });
    }
}

/// Sunken pit in the center — lower elevation creating a bowl.
/// Ranged units on the rim have advantage, melee units fight in the pit.
fn add_sunken_pit(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    room: &PlacedRoom,
) {
    let cols = nav.cols;
    let rows = nav.rows;
    let pit_depth = rng.next_f32_range(0.5, 1.0);

    let pit_w = room.width / 3;
    let pit_h = room.height / 3;
    let pc0 = room.col + room.width / 2 - pit_w / 2;
    let pr0 = room.row + room.height / 2 - pit_h / 2;

    let lowered = (room.elevation - pit_depth).max(0.0);

    for r in pr0..pr0 + pit_h {
        for c in pc0..pc0 + pit_w {
            if c < cols && r < rows && nav.walkable[r * cols + c] {
                nav.elevation[r * cols + c] = lowered;
            }
        }
    }
}

/// Split level — room divided roughly in half at different heights
/// with a 2-cell wide ramp strip connecting them.
fn add_split_level(
    nav: &mut NavGrid,
    obstacles: &mut Vec<ObstacleRegion>,
    rng: &mut Lcg,
    room: &PlacedRoom,
) {
    let cols = nav.cols;
    let rows = nav.rows;
    let raised = room.elevation + 0.8;

    // Split horizontally or vertically
    let horizontal = rng.next_u64() % 2 == 0;

    if horizontal {
        // Top half is raised
        let split_row = room.row + room.height / 2;
        let ramp_row = split_row;

        for r in room.row + 1..split_row {
            for c in room.col + 1..room.col + room.width - 1 {
                if c < cols && r < rows && nav.walkable[r * cols + c] {
                    nav.elevation[r * cols + c] = raised;
                }
            }
        }

        // Ramp strip (2 cells tall) — gradual transition
        for c in room.col + 1..room.col + room.width - 1 {
            if c < cols && ramp_row < rows {
                let idx = ramp_row * cols + c;
                if nav.walkable[idx] {
                    nav.elevation[idx] = room.elevation + 0.4;
                }
            }
        }

        // Ledge wall along the split with gaps for ramp access
        let gap_start = room.col + room.width / 3;
        let gap_end = room.col + 2 * room.width / 3;
        let wall_row = split_row;
        for c in room.col + 2..room.col + room.width - 2 {
            if c >= gap_start && c <= gap_end {
                continue; // ramp gap
            }
            if c < cols && wall_row < rows {
                let idx = wall_row * cols + c;
                nav.walkable[idx] = false;
            }
        }
        // Record the wall segments
        if gap_start > room.col + 2 {
            obstacles.push(ObstacleRegion {
                col0: room.col + 2, col1: gap_start - 1,
                row0: wall_row, row1: wall_row,
                height: 0.5, obs_type: OBS_WALL,
            });
        }
        if gap_end < room.col + room.width - 3 {
            obstacles.push(ObstacleRegion {
                col0: gap_end + 1, col1: room.col + room.width - 3,
                row0: wall_row, row1: wall_row,
                height: 0.5, obs_type: OBS_WALL,
            });
        }
    } else {
        // Left half is raised
        let split_col = room.col + room.width / 2;

        for r in room.row + 1..room.row + room.height - 1 {
            for c in room.col + 1..split_col {
                if c < cols && r < rows && nav.walkable[r * cols + c] {
                    nav.elevation[r * cols + c] = raised;
                }
            }
        }

        // Ramp strip
        for r in room.row + 1..room.row + room.height - 1 {
            if split_col < cols && r < rows {
                let idx = r * cols + split_col;
                if nav.walkable[idx] {
                    nav.elevation[idx] = room.elevation + 0.4;
                }
            }
        }

        // Ledge wall with gap
        let gap_start = room.row + room.height / 3;
        let gap_end = room.row + 2 * room.height / 3;
        for r in room.row + 2..room.row + room.height - 2 {
            if r >= gap_start && r <= gap_end {
                continue;
            }
            if split_col < cols && r < rows {
                let idx = r * cols + split_col;
                nav.walkable[idx] = false;
            }
        }
        if gap_start > room.row + 2 {
            obstacles.push(ObstacleRegion {
                col0: split_col, col1: split_col,
                row0: room.row + 2, row1: gap_start - 1,
                height: 0.5, obs_type: OBS_WALL,
            });
        }
        if gap_end < room.row + room.height - 3 {
            obstacles.push(ObstacleRegion {
                col0: split_col, col1: split_col,
                row0: gap_end + 1, row1: room.row + room.height - 3,
                height: 0.5, obs_type: OBS_WALL,
            });
        }
    }
}

/// Elevated overlook — a narrow raised walkway along one wall,
/// like a catwalk or balcony rail.
fn add_overlook(
    nav: &mut NavGrid,
    obstacles: &mut Vec<ObstacleRegion>,
    rng: &mut Lcg,
    room: &PlacedRoom,
) {
    let cols = nav.cols;
    let rows = nav.rows;
    let raised = room.elevation + 0.8;
    let walk_width = 2;

    // Pick top or bottom wall
    let top = rng.next_u64() % 2 == 0;

    let (wr0, wr1) = if top {
        (room.row + 1, room.row + walk_width)
    } else {
        (room.row + room.height - walk_width - 1, room.row + room.height - 2)
    };

    // Set walkway elevation
    for r in wr0..=wr1 {
        for c in room.col + 1..room.col + room.width - 1 {
            if c < cols && r < rows && nav.walkable[r * cols + c] {
                nav.elevation[r * cols + c] = raised;
            }
        }
    }

    // Railing/ledge along the inner edge
    let rail_r = if top { wr1 + 1 } else { wr0.saturating_sub(1) };
    if rail_r > 0 && rail_r < rows {
        // Leave 2 gaps for stair access
        let gap1 = room.col + room.width / 4;
        let gap2 = room.col + 3 * room.width / 4;

        for c in room.col + 2..room.col + room.width - 2 {
            if (c >= gap1 && c <= gap1 + 1) || (c >= gap2 && c <= gap2 + 1) {
                // Ramp cells at gaps
                if c < cols {
                    let idx = rail_r * cols + c;
                    if idx < nav.walkable.len() {
                        nav.elevation[idx] = room.elevation + 0.4;
                    }
                }
                continue;
            }
            if c < cols {
                let idx = rail_r * cols + c;
                if idx < nav.walkable.len() {
                    nav.walkable[idx] = false;
                }
            }
        }
        obstacles.push(ObstacleRegion {
            col0: room.col + 2, col1: room.col + room.width - 3,
            row0: rail_r, row1: rail_r,
            height: 0.4, obs_type: OBS_WALL,
        });
    }
}

// ---------------------------------------------------------------------------
// Spawn placement
// ---------------------------------------------------------------------------

fn place_room_spawns(nav: &NavGrid, rng: &mut Lcg, room: &mut PlacedRoom) {
    let cols = nav.cols;

    // Place spawns near the edges of the room that face corridors.
    // Simple approach: player spawns on left side, enemy spawns on right side.
    let mid_r = room.row + room.height / 2;
    let r_lo = room.row + 2;
    let r_hi = (room.row + room.height).saturating_sub(3);

    let spawn_count = match room.role {
        RoomRole::Entry => 4,
        RoomRole::Climax => 6,
        _ => 4,
    };

    // Player spawns: left portion of room
    let p_col_lo = room.col + 1;
    let p_col_hi = room.col + room.width / 4;
    let mut p_candidates = Vec::new();
    for r in r_lo..=r_hi {
        for c in p_col_lo..=p_col_hi {
            if c < cols && r < nav.rows && nav.walkable[r * cols + c] {
                p_candidates.push(SimVec2 { x: c as f32 + 0.5, y: r as f32 + 0.5 });
            }
        }
    }
    if !p_candidates.is_empty() {
        let step = (p_candidates.len() as f32 / spawn_count as f32).max(1.0) as usize;
        room.player_spawns = p_candidates
            .iter()
            .step_by(step)
            .take(spawn_count)
            .copied()
            .collect();
    }

    // Enemy spawns: right portion of room
    let e_col_lo = room.col + 3 * room.width / 4;
    let e_col_hi = room.col + room.width - 2;
    let mut e_candidates = Vec::new();
    for r in r_lo..=r_hi {
        for c in e_col_lo..=e_col_hi {
            if c < cols && r < nav.rows && nav.walkable[r * cols + c] {
                e_candidates.push(SimVec2 { x: c as f32 + 0.5, y: r as f32 + 0.5 });
            }
        }
    }
    if !e_candidates.is_empty() {
        let step = (e_candidates.len() as f32 / spawn_count as f32).max(1.0) as usize;
        room.enemy_spawns = e_candidates
            .iter()
            .step_by(step)
            .take(spawn_count)
            .copied()
            .collect();
    }
}

// ---------------------------------------------------------------------------
// JSON export for the floorplan
// ---------------------------------------------------------------------------

impl Floorplan {
    /// Export floorplan as a multi-channel grid (same format as single rooms).
    pub fn to_grid(&self) -> super::RoomGrid {
        let w = self.total_width;
        let h = self.total_height;
        let n = w * h;
        let mut obstacle_type = vec![OBS_WALL; n]; // default: wall
        let mut height = vec![0.0f32; n];

        // Mark walkable cells as floor
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                if self.nav.walkable[idx] {
                    obstacle_type[idx] = OBS_FLOOR;
                }
            }
        }

        // Stamp obstacle types
        for obs in &self.obstacles {
            for r in obs.row0..=obs.row1 {
                for c in obs.col0..=obs.col1 {
                    if c < w && r < h {
                        let idx = r * w + c;
                        obstacle_type[idx] = obs.obs_type;
                        height[idx] = obs.height;
                    }
                }
            }
        }

        super::RoomGrid {
            width: w,
            depth: h,
            obstacle_type,
            height,
            elevation: self.nav.elevation.clone(),
        }
    }
}
