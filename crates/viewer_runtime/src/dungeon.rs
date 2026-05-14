//! Dungeon roomgen + voxel/topology seeding for `dungeon_horde`.
//!
//! Ported from `crates/sims/tests/dungeon_horde_pin.rs` so the viewer
//! constructs the same dungeon + agent layout as the pin. Kept in its
//! own module because it's load-bearing copy-paste — drift between
//! the pin and the viewer would mean the viewer renders a different
//! sim than the test asserts on.
//!
//! All functions are pure CPU; no GPU work happens here. The single
//! GPU touch is in [`seed_topology`] (writes to the runtime's agent
//! buffers via `gpu.queue.write_buffer`).
//!
//! # Constants — must stay in sync with the pin
//!
//! Hand-laid 21-room layout via `ROOM_LAYOUT` + per-template wall
//! patterns. The pin (`crates/sims/tests/dungeon_horde_pin.rs`) keeps
//! its own CA-based roomgen — the viewer is decoupled by design so
//! we can iterate on visual layout without churning the test
//! fixture's pinned scene.

use glam::IVec3;
use sims::dungeon_horde::GeneratedRuntime;

pub const N_HEROES: u32 = 5;

pub const GRID_X: u32 = 96;
pub const GRID_Y: u32 = 96;
pub const GRID_Z: u32 = 8;
pub const SLOTS_PER_ROW: u32 = 6;
pub const SLOT_WIDTH: u32 = 16;
pub const ROOM_INTERIOR_Z: u32 = 6;
pub const STONE: u8 = 1;

pub const CT_ARCHER: u32 = 0;
pub const CT_BRUTE: u32 = 1;
pub const CT_GOBLIN: u32 = 2;
pub const CT_HERO: u32 = 3;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoomSlot {
    pub rx: u32,
    pub ry: u32,
}

impl RoomSlot {
    pub fn new(rx: u32, ry: u32) -> Self {
        Self { rx, ry }
    }
    pub fn idx(&self) -> u32 {
        self.ry * SLOTS_PER_ROW + self.rx
    }
    pub fn centroid(&self) -> [f32; 3] {
        let cx = (self.rx as f32) * (SLOT_WIDTH as f32) + (SLOT_WIDTH as f32) / 2.0;
        let cy = (self.ry as f32) * (SLOT_WIDTH as f32) + (SLOT_WIDTH as f32) / 2.0;
        [cx, cy, 1.0]
    }
}

pub struct Dungeon {
    pub rooms: Vec<RoomSlot>,
    pub floor_cells: std::collections::BTreeMap<u32, Vec<(u32, u32)>>,
    pub bfs_dist: std::collections::BTreeMap<u32, u32>,
    pub spawn_room: RoomSlot,
    pub boss_room: RoomSlot,
    pub seed: u64,
}

impl Dungeon {
    pub fn total_floor_cells(&self) -> u32 {
        self.floor_cells.values().map(|v| v.len() as u32).sum()
    }
    pub fn total_agent_count(&self) -> u32 {
        N_HEROES + self.enemy_placements().len() as u32
    }
    /// Per-pin enemy distribution (BFS-distance scaled).
    pub fn enemy_placements(&self) -> Vec<(RoomSlot, u32)> {
        let mut out: Vec<(RoomSlot, u32)> = Vec::new();
        let seed32 = self.seed as u32;
        for &room in &self.rooms {
            if room == self.spawn_room {
                continue;
            }
            let idx = room.idx();
            let dist = *self.bfs_dist.get(&idx).unwrap_or(&0);
            let n_floor = self.floor_cells.get(&idx).map(|v| v.len() as u32).unwrap_or(0);
            // Density formula tuned for visual quality (~6-12 enemies per
            // shallow/mid room, ~24 in the boss chamber). Must stay in
            // sync with `dungeon_horde_pin.rs::enemy_placements` — drift
            // means viewer and pin render different sims.
            let raw_count = if room == self.boss_room {
                (n_floor / 4).min(30)
            } else if dist == 1 {
                n_floor / 16
            } else if dist == 2 {
                n_floor / 12
            } else if dist == 3 {
                n_floor / 10
            } else {
                n_floor / 8
            };
            let count = raw_count.min(n_floor.saturating_sub(4));
            for slot in 0..count {
                let ct = if room == self.boss_room {
                    let r = engine::rng::per_agent_u32_pcg(seed32, idx, slot, 0xB055_001) % 10;
                    if r < 3 { CT_BRUTE } else if r < 7 { CT_ARCHER } else { CT_GOBLIN }
                } else {
                    let r = engine::rng::per_agent_u32_pcg(seed32, idx, slot, 0xE5_E51_001) % 100;
                    if dist <= 2 {
                        if r < 80 { CT_GOBLIN } else if r < 98 { CT_ARCHER } else { CT_BRUTE }
                    } else if dist == 3 {
                        if r < 65 { CT_GOBLIN } else if r < 95 { CT_ARCHER } else { CT_BRUTE }
                    } else {
                        if r < 50 { CT_GOBLIN } else if r < 90 { CT_ARCHER } else { CT_BRUTE }
                    }
                };
                out.push((room, ct));
            }
        }
        out
    }
}

/// Per-slot interior layout — picks which walls/columns/partitions live
/// inside a room. Selected per-room in [`ROOM_LAYOUT`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoomTemplate {
    /// Hero spawn — fully open. Behaviorally identical to `Open`; tagged
    /// separately so `roll_dungeon` knows which slot is the spawn.
    Spawn,
    /// Bare rectangle. Outer ring is wall; interior is fully traversable.
    Open,
    /// Four 2×2 stone columns at the interior quarters.
    Pillared,
    /// Single horizontal partition wall across the middle with a 4-cell
    /// gap. Splits sightlines but leaves a clear chokepoint.
    Partition,
    /// Central 4×4 chest pillar plus four corner-accent stones.
    Treasure,
    /// Boss arena — six ceremonial 1×1 columns down the long axis.
    /// Tagged so `roll_dungeon` picks this slot as the boss room.
    Boss,
}

/// Hand-laid 21-room dungeon. Each entry is `(rx, ry, template)`; rooms
/// are placed in slot-adjacent positions so [`seed_voxel_dungeon`]'s
/// existing doorway carving creates 2-cell-wide passages between them.
///
/// Layout (S=Spawn, V=Treasure, P=Pillared, G=Partition, B=Boss):
/// ```text
///       0  1  2  3  4  5
///   0   S  H  P  V  .  .
///   1   H  .  .  H  .  .
///   2   H  G  H  H  P  .
///   3   .  .  .  .  H  .
///   4   V  H  H  H  H  .
///   5   H  .  .  P  H  B
/// ```text
///       0  1  2  3  4  5  6  7  8
///   0   .  .  .  .  .  .  .  .  .
///   1   .  S  H  P  V  .  .  .  .
///   2   .  H  .  .  H  .  .  .  .
///   3   .  H  G  H  H  H  .  .  .
///   4   .  .  .  .  P  .  .  .  .
///   5   .  V  H  H  H  H  .  .  .
///   6   .  H  .  .  .  H  .  .  .
///   7   .  H  H  P  H  H  .  .  .
///   8   .  .  .  .  .  B  .  .  .
/// ```
pub const ROOM_LAYOUT: &[(u32, u32, RoomTemplate)] = &[
    (0, 0, RoomTemplate::Spawn),
    (1, 0, RoomTemplate::Open),
    (2, 0, RoomTemplate::Pillared),
    (3, 0, RoomTemplate::Treasure),

    (0, 1, RoomTemplate::Open),
    (3, 1, RoomTemplate::Open),

    (0, 2, RoomTemplate::Open),
    (1, 2, RoomTemplate::Partition),
    (2, 2, RoomTemplate::Open),
    (3, 2, RoomTemplate::Open),
    (4, 2, RoomTemplate::Pillared),

    (4, 3, RoomTemplate::Open),

    (0, 4, RoomTemplate::Treasure),
    (1, 4, RoomTemplate::Open),
    (2, 4, RoomTemplate::Open),
    (3, 4, RoomTemplate::Open),
    (4, 4, RoomTemplate::Open),

    (0, 5, RoomTemplate::Open),
    (3, 5, RoomTemplate::Pillared),
    (4, 5, RoomTemplate::Open),
    (5, 5, RoomTemplate::Boss),
];

/// Build the hand-laid dungeon. The `initial_seed` no longer drives
/// the layout (it's fixed) — it's still threaded through to the
/// [`Dungeon`] struct so downstream RNG (door positions, enemy
/// composition) keeps its per-seed variation.
pub fn roll_dungeon(initial_seed: u64) -> Dungeon {
    let mut rooms: Vec<RoomSlot> = Vec::with_capacity(ROOM_LAYOUT.len());
    let mut floor_cells: std::collections::BTreeMap<u32, Vec<(u32, u32)>> =
        std::collections::BTreeMap::new();
    let mut spawn_room: Option<RoomSlot> = None;
    let mut boss_template_room: Option<RoomSlot> = None;

    for &(rx, ry, template) in ROOM_LAYOUT {
        let slot = RoomSlot::new(rx, ry);
        debug_assert!(rx < SLOTS_PER_ROW && ry < SLOTS_PER_ROW, "room {slot:?} out of slot grid");
        debug_assert!(
            !rooms.contains(&slot),
            "room {slot:?} listed twice in ROOM_LAYOUT",
        );
        rooms.push(slot);
        floor_cells.insert(slot.idx(), template_floor_cells(slot, template));
        if template == RoomTemplate::Spawn {
            spawn_room = Some(slot);
        }
        if template == RoomTemplate::Boss {
            boss_template_room = Some(slot);
        }
    }
    let spawn_room = spawn_room
        .expect("ROOM_LAYOUT must include exactly one RoomTemplate::Spawn entry");

    let present_set: std::collections::BTreeSet<RoomSlot> = rooms.iter().copied().collect();
    let mut bfs_dist = std::collections::BTreeMap::<u32, u32>::new();
    let mut queue: std::collections::VecDeque<RoomSlot> = std::collections::VecDeque::new();
    queue.push_back(spawn_room);
    bfs_dist.insert(spawn_room.idx(), 0);
    while let Some(cur) = queue.pop_front() {
        let d = *bfs_dist.get(&cur.idx()).unwrap();
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = cur.rx as i32 + dx;
            let ny = cur.ry as i32 + dy;
            if nx < 0 || ny < 0 || (nx as u32) >= SLOTS_PER_ROW || (ny as u32) >= SLOTS_PER_ROW {
                continue;
            }
            let n = RoomSlot::new(nx as u32, ny as u32);
            if !present_set.contains(&n) { continue; }
            if bfs_dist.contains_key(&n.idx()) { continue; }
            bfs_dist.insert(n.idx(), d + 1);
            queue.push_back(n);
        }
    }
    assert_eq!(
        bfs_dist.len(),
        rooms.len(),
        "ROOM_LAYOUT has disconnected rooms — every room must be slot-adjacent to at least one other"
    );

    // Boss room: the explicitly-tagged Boss slot, or fall back to the
    // BFS-furthest if none was tagged.
    let boss_room = boss_template_room.unwrap_or_else(|| {
        *rooms
            .iter()
            .max_by_key(|r| bfs_dist.get(&r.idx()).copied().unwrap_or(0))
            .unwrap()
    });

    Dungeon { rooms, floor_cells, bfs_dist, spawn_room, boss_room, seed: initial_seed }
}

/// Stamp a template's floor cells onto the slot. The outer ring is wall
/// by default — [`seed_voxel_dungeon`] punches doorways at adjacent
/// slot boundaries — so adjacent rooms are connected only via real
/// 2-cell-wide doors instead of wide-open shared edges.
fn template_floor_cells(slot: RoomSlot, template: RoomTemplate) -> Vec<(u32, u32)> {
    let x0 = slot.rx * SLOT_WIDTH;
    let y0 = slot.ry * SLOT_WIDTH;
    let s = SLOT_WIDTH as usize;
    let mut walls: Vec<Vec<bool>> = vec![vec![true; s]; s];
    for ly in 1..(s - 1) {
        for lx in 1..(s - 1) {
            walls[ly][lx] = false;
        }
    }
    fn set_wall(walls: &mut [Vec<bool>], lx: u32, ly: u32) {
        let s = walls.len();
        if (lx as usize) < s && (ly as usize) < s {
            walls[ly as usize][lx as usize] = true;
        }
    }
    fn set_block(walls: &mut [Vec<bool>], lx: u32, ly: u32, w: u32, h: u32) {
        for dy in 0..h {
            for dx in 0..w {
                set_wall(walls, lx + dx, ly + dy);
            }
        }
    }
    match template {
        RoomTemplate::Spawn | RoomTemplate::Open => {}
        RoomTemplate::Pillared => {
            // Four 1x1 columns at the interior quarters
            for &(px, py) in &[(3u32, 3u32), (SLOT_WIDTH - 4, 3), (3, SLOT_WIDTH - 4), (SLOT_WIDTH - 4, SLOT_WIDTH - 4)] {
                set_wall(&mut walls, px, py);
            }
        }
        RoomTemplate::Partition => {
            // Horizontal partition wall at the middle row, with a 3-cell
            // gap centered on the room.
            let mid = SLOT_WIDTH / 2;
            let gap_lo = SLOT_WIDTH / 2 - 1;
            let gap_hi = SLOT_WIDTH / 2 + 2;
            for lx in 2..(SLOT_WIDTH - 2) {
                if lx >= gap_lo && lx < gap_hi { continue; }
                set_wall(&mut walls, lx, mid);
            }
        }
        RoomTemplate::Treasure => {
            // Central 2x2 chest pillar
            let c = SLOT_WIDTH / 2 - 1;
            set_block(&mut walls, c, c, 2, 2);
        }
        RoomTemplate::Boss => {
            // Four ceremonial 1x1 columns broadly spaced for boss-arena feel
            for &(px, py) in &[
                (2u32, 3u32),
                (SLOT_WIDTH - 3, 3),
                (2, SLOT_WIDTH - 4),
                (SLOT_WIDTH - 3, SLOT_WIDTH - 4),
            ] {
                set_wall(&mut walls, px, py);
            }
        }
    }
    let mut floor = Vec::with_capacity(s * s);
    for ly in 0..SLOT_WIDTH {
        for lx in 0..SLOT_WIDTH {
            if !walls[ly as usize][lx as usize] {
                floor.push((x0 + lx, y0 + ly));
            }
        }
    }
    floor
}

/// Seed the runtime's voxel terrain with the dungeon's wall cells.
/// Mirrors `dungeon_horde_pin::seed_voxel_dungeon`. Returns the
/// (BTreeSet of) floor cells the viewer can use to paint a tan
/// floor color.
pub fn seed_voxel_dungeon(
    state: &mut GeneratedRuntime,
    dungeon: &Dungeon,
    seed: u64,
) -> std::collections::BTreeSet<(u32, u32)> {
    let mut floor_map = std::collections::BTreeSet::<(u32, u32)>::new();
    for cells in dungeon.floor_cells.values() {
        for &(x, y) in cells {
            floor_map.insert((x, y));
        }
    }
    let present_set: std::collections::BTreeSet<RoomSlot> =
        dungeon.rooms.iter().copied().collect();
    let seed32 = seed as u32;
    for &r in &dungeon.rooms {
        let east = RoomSlot::new(r.rx + 1, r.ry);
        if r.rx + 1 < SLOTS_PER_ROW && present_set.contains(&east) {
            let door_y_off = engine::rng::per_agent_u32_pcg(seed32, r.idx(), east.idx(), 3)
                % (SLOT_WIDTH - 4);
            let door_y = r.ry * SLOT_WIDTH + 2 + door_y_off;
            let boundary_x = (r.rx + 1) * SLOT_WIDTH;
            for dx in (boundary_x - 1)..=boundary_x {
                for dy in door_y..(door_y + 2) {
                    floor_map.insert((dx, dy));
                }
            }
        }
        let south = RoomSlot::new(r.rx, r.ry + 1);
        if r.ry + 1 < SLOTS_PER_ROW && present_set.contains(&south) {
            let door_x_off = engine::rng::per_agent_u32_pcg(seed32, r.idx(), south.idx(), 4)
                % (SLOT_WIDTH - 4);
            let door_x = r.rx * SLOT_WIDTH + 2 + door_x_off;
            let boundary_y = (r.ry + 1) * SLOT_WIDTH;
            for dy in (boundary_y - 1)..=boundary_y {
                for dx in door_x..(door_x + 2) {
                    floor_map.insert((dx, dy));
                }
            }
        }
    }
    let mut writes: Vec<(u32, u32, u32)> = Vec::new();
    for x in 0..GRID_X {
        for y in 0..GRID_Y {
            if !floor_map.contains(&(x, y)) {
                for z in 0..ROOM_INTERIOR_Z.min(GRID_Z) {
                    state.voxel_terrain.set_cell(x, y, z, STONE);
                    writes.push((x, y, z));
                }
            }
        }
    }
    for (x, y, z) in writes {
        state.voxel_mirror.mark_dirty(IVec3::new(x as i32, y as i32, z as i32));
    }
    floor_map
}

/// Mirrors `dungeon_horde_pin::seed_topology` — places enemies at
/// floor cells, heroes at the spawn room, sets per-agent runtime
/// fields (creature_type, role, target_room, expected allies, patrol).
///
/// Returns the agent slot of the first Brute placed in the boss
/// room — the viewer designates that one as "the boss" and gives it
/// 4x HP + a distinct color. `None` if no brute landed in boss_room
/// (small dungeons / unlucky rolls).
pub fn seed_topology(
    state: &mut GeneratedRuntime,
    dungeon: &Dungeon,
    seed: u64,
) -> Option<u32> {
    let agent_count = dungeon.total_agent_count() as usize;
    let mut positions = vec![[0.0f32; 4]; agent_count];
    let mut creature_type = vec![0u32; agent_count];
    let mut role = vec![0u32; agent_count];
    let mut target_room_idx = vec![0u32; agent_count];
    let mut expected_chamber_allies = vec![0u32; agent_count];
    let mut patrol_axis = vec![0u32; agent_count];
    let mut patrol_origin_x = vec![0.0f32; agent_count];
    let mut patrol_origin_y = vec![0.0f32; agent_count];
    let mut patrol_step_x = vec![0.0f32; agent_count];
    let mut patrol_step_y = vec![0.0f32; agent_count];

    let enemies = dungeon.enemy_placements();
    let archers: Vec<(RoomSlot, u32)> =
        enemies.iter().copied().filter(|(_, ct)| *ct == CT_ARCHER).collect();
    let brutes: Vec<(RoomSlot, u32)> =
        enemies.iter().copied().filter(|(_, ct)| *ct == CT_BRUTE).collect();
    let goblins: Vec<(RoomSlot, u32)> =
        enemies.iter().copied().filter(|(_, ct)| *ct == CT_GOBLIN).collect();

    let mut per_room_enemy_count: std::collections::BTreeMap<u32, u32> =
        std::collections::BTreeMap::new();
    for &(room, _ct) in &enemies {
        *per_room_enemy_count.entry(room.idx()).or_insert(0) += 1;
    }

    let mut slot = 0usize;

    let place_enemy = |slot: usize,
                           room: RoomSlot,
                           ct: u32,
                           positions: &mut Vec<[f32; 4]>,
                           creature_type: &mut Vec<u32>,
                           expected_chamber_allies: &mut Vec<u32>,
                           patrol_axis: &mut Vec<u32>,
                           patrol_origin_x: &mut Vec<f32>,
                           patrol_origin_y: &mut Vec<f32>,
                           patrol_step_x: &mut Vec<f32>,
                           patrol_step_y: &mut Vec<f32>,
                           per_purpose: u32,
                           i_in_class: u32| {
        let (px, py) = pick_floor_cell(dungeon, room, i_in_class, per_purpose, seed);
        positions[slot] = [px as f32 + 0.5, py as f32 + 0.5, 1.0, 0.0];
        creature_type[slot] = ct;
        expected_chamber_allies[slot] =
            per_room_enemy_count.get(&room.idx()).copied().unwrap_or(0);

        if ct == CT_GOBLIN {
            let r = engine::rng::per_agent_u32_pcg(seed as u32, slot as u32, 0, 0x5001) % 10;
            if r < 3 {
                patrol_axis[slot] = 1;
                patrol_origin_x[slot] = px as f32 + 0.5;
                patrol_origin_y[slot] = py as f32 + 0.5;
                let dir_sign = if (r & 1) == 0 { 1.0 } else { -1.0 };
                patrol_step_x[slot] = 0.10 * dir_sign;
                patrol_step_y[slot] = 0.0;
            }
        }
    };

    for (i, &(room, _ct)) in archers.iter().enumerate() {
        place_enemy(
            slot, room, CT_ARCHER, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            0xA5C_4E0, i as u32,
        );
        slot += 1;
    }
    let mut boss_slot: Option<u32> = None;
    for (i, &(room, _ct)) in brutes.iter().enumerate() {
        place_enemy(
            slot, room, CT_BRUTE, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            0xB23_C7E, i as u32,
        );
        if boss_slot.is_none() && room == dungeon.boss_room {
            boss_slot = Some(slot as u32);
        }
        slot += 1;
    }
    for (i, &(room, _ct)) in goblins.iter().enumerate() {
        place_enemy(
            slot, room, CT_GOBLIN, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            0x9081_AE, i as u32,
        );
        slot += 1;
    }

    let spawn_centroid = dungeon.spawn_room.centroid();
    let initial_target =
        pick_adjacent_present_room(dungeon, dungeon.spawn_room).unwrap_or(dungeon.spawn_room);
    for h in 0..N_HEROES as usize {
        let dx = ((h as f32) - 2.0) * 0.5;
        let dy = (((h + 1) % 5) as f32 - 2.0) * 0.5;
        positions[slot] = [
            spawn_centroid[0] + dx,
            spawn_centroid[1] + dy,
            spawn_centroid[2],
            0.0,
        ];
        creature_type[slot] = CT_HERO;
        role[slot] = (h as u32) + 1;
        target_room_idx[slot] = initial_target.idx();
        slot += 1;
    }

    debug_assert_eq!(slot, agent_count, "slot accounting drift");

    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&positions));
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf, 0, bytemuck::cast_slice(&creature_type),
    );
    state.gpu.queue.write_buffer(&state.agent_role_buf, 0, bytemuck::cast_slice(&role));
    state.gpu.queue.write_buffer(
        &state.agent_target_room_idx_buf, 0, bytemuck::cast_slice(&target_room_idx),
    );
    state.gpu.queue.write_buffer(
        &state.agent_expected_chamber_allies_buf, 0,
        bytemuck::cast_slice(&expected_chamber_allies),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_axis_buf, 0, bytemuck::cast_slice(&patrol_axis),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_origin_x_buf, 0, bytemuck::cast_slice(&patrol_origin_x),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_origin_y_buf, 0, bytemuck::cast_slice(&patrol_origin_y),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_step_x_buf, 0, bytemuck::cast_slice(&patrol_step_x),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_step_y_buf, 0, bytemuck::cast_slice(&patrol_step_y),
    );

    // Initial hero waypoints: each hero's spawn position. The host's
    // advance_hero_exploration overwrites these every tick from the
    // first step() onward; this just ensures heroes don't dash to
    // (0, 0) on tick 0 if the runtime's HeroExplore fires before the
    // host AI does.
    let mut hero_waypoint_x = vec![0.0f32; agent_count];
    let mut hero_waypoint_y = vec![0.0f32; agent_count];
    for h in 0..N_HEROES as usize {
        let hi = (agent_count - N_HEROES as usize) + h;
        hero_waypoint_x[hi] = positions[hi][0];
        hero_waypoint_y[hi] = positions[hi][1];
    }
    state.gpu.queue.write_buffer(
        &state.agent_hero_waypoint_x_buf, 0, bytemuck::cast_slice(&hero_waypoint_x),
    );
    state.gpu.queue.write_buffer(
        &state.agent_hero_waypoint_y_buf, 0, bytemuck::cast_slice(&hero_waypoint_y),
    );

    boss_slot
}

fn pick_floor_cell(
    dungeon: &Dungeon, room: RoomSlot, i: u32, purpose: u32, seed: u64,
) -> (u32, u32) {
    let cells = dungeon.floor_cells.get(&room.idx()).expect("room has cells");
    if cells.is_empty() {
        let c = room.centroid();
        return (c[0] as u32, c[1] as u32);
    }
    let idx = engine::rng::per_agent_u32_pcg(seed as u32, room.idx(), i, purpose) as usize
        % cells.len();
    cells[idx]
}

fn pick_adjacent_present_room(dungeon: &Dungeon, from: RoomSlot) -> Option<RoomSlot> {
    let present_set: std::collections::BTreeSet<RoomSlot> =
        dungeon.rooms.iter().copied().collect();
    for (dx, dy) in [(0i32, 1i32), (1, 0), (0, -1), (-1, 0)] {
        let nx = from.rx as i32 + dx;
        let ny = from.ry as i32 + dy;
        if nx < 0 || ny < 0 || (nx as u32) >= SLOTS_PER_ROW || (ny as u32) >= SLOTS_PER_ROW {
            continue;
        }
        let n = RoomSlot::new(nx as u32, ny as u32);
        if present_set.contains(&n) {
            return Some(n);
        }
    }
    None
}

/// Door midpoint between two slot-adjacent rooms `a` and `b`. Returns
/// the world (x, y) at the *center* of the 2-cell-wide door gap that
/// `seed_voxel_dungeon` carved between them. None if the two rooms are
/// not slot-adjacent (caller must enforce adjacency).
///
/// Used by the host hero-exploration AI to set per-tick waypoints —
/// heroes path to the door cell first, then to the target room's
/// centroid once they cross. Without this they get stuck against the
/// wall when the .sim's HeroExplore LoS check is enabled.
pub fn door_position(seed: u64, a: RoomSlot, b: RoomSlot) -> Option<(f32, f32)> {
    let seed32 = seed as u32;
    // Normalize: lower (rx, ry) is the "from" room in seed_voxel_dungeon's
    // door-key construction (it iterates rooms and writes east/south
    // doors keyed on the from-room's idx).
    let (from, to, axis) = if a.rx + 1 == b.rx && a.ry == b.ry {
        (a, b, 'e')
    } else if b.rx + 1 == a.rx && a.ry == b.ry {
        (b, a, 'e')
    } else if a.ry + 1 == b.ry && a.rx == b.rx {
        (a, b, 's')
    } else if b.ry + 1 == a.ry && a.rx == b.rx {
        (b, a, 's')
    } else {
        return None;
    };

    let _ = to;
    if axis == 'e' {
        let door_y_off = engine::rng::per_agent_u32_pcg(
            seed32, from.idx(), RoomSlot::new(from.rx + 1, from.ry).idx(), 3,
        ) % (SLOT_WIDTH - 4);
        let door_y = from.ry * SLOT_WIDTH + 2 + door_y_off;
        // Center: x = boundary, y = door_y + 0.5 (door spans door_y..door_y+2)
        let cx = ((from.rx + 1) * SLOT_WIDTH) as f32;
        let cy = door_y as f32 + 1.0;
        Some((cx, cy))
    } else {
        let door_x_off = engine::rng::per_agent_u32_pcg(
            seed32, from.idx(), RoomSlot::new(from.rx, from.ry + 1).idx(), 4,
        ) % (SLOT_WIDTH - 4);
        let door_x = from.rx * SLOT_WIDTH + 2 + door_x_off;
        let cx = door_x as f32 + 1.0;
        let cy = ((from.ry + 1) * SLOT_WIDTH) as f32;
        Some((cx, cy))
    }
}

// ---------------------------------------------------------------------
// Host-side hero exploration — ports `dungeon_horde_pin`'s
// `update_hero_exploration` + `pick_next_target` so the viewer
// advances `target_room_idx` between ticks the same way the test does.
// Without this, heroes reach their first adjacent room and stop.
// ---------------------------------------------------------------------

/// Per-hero exploration state — which rooms each hero has already
/// visited (bitmask, 1 bit per entry in `dungeon.rooms`) and their
/// last-known room. Used by the host-side advance step to decide when
/// to retarget after the hero arrives at their current target.
pub struct HeroExploreState {
    pub rooms_visited: [u32; N_HEROES as usize],
    pub current_room: [Option<u32>; N_HEROES as usize],
}

impl HeroExploreState {
    pub fn new(dungeon: &Dungeon) -> Self {
        let spawn_remap = dungeon
            .rooms
            .iter()
            .position(|r| *r == dungeon.spawn_room)
            .expect("spawn_room is in rooms");
        let initial_mask = 1u32 << spawn_remap;
        Self {
            rooms_visited: [initial_mask; N_HEROES as usize],
            current_room: [Some(dungeon.spawn_room.idx()); N_HEROES as usize],
        }
    }
}

/// Frontier-greedy room picker — prefer unvisited adjacent rooms;
/// fall back to any adjacent room if every neighbour was already
/// visited (so heroes can retreat or loop). Tied to the hero's
/// per-tick PCG so picks are deterministic across replays.
pub fn pick_next_target(
    dungeon: &Dungeon,
    from: RoomSlot,
    visited_mask: u32,
    tick: u32,
    hero_idx: u32,
    seed: u64,
) -> u32 {
    let present_set: std::collections::BTreeSet<RoomSlot> =
        dungeon.rooms.iter().copied().collect();
    let mut unvisited: Vec<RoomSlot> = Vec::with_capacity(4);
    let mut adjacent: Vec<RoomSlot> = Vec::with_capacity(4);
    for (dx, dy) in [(0i32, 1i32), (1, 0), (0, -1), (-1, 0)] {
        let nx = from.rx as i32 + dx;
        let ny = from.ry as i32 + dy;
        if nx < 0 || ny < 0 || (nx as u32) >= SLOTS_PER_ROW || (ny as u32) >= SLOTS_PER_ROW {
            continue;
        }
        let n = RoomSlot::new(nx as u32, ny as u32);
        if !present_set.contains(&n) { continue; }
        adjacent.push(n);
        let remap = dungeon.rooms.iter().position(|r| *r == n).unwrap();
        if (visited_mask & (1u32 << remap)) == 0 {
            unvisited.push(n);
        }
    }
    let pool = if !unvisited.is_empty() { &unvisited } else { &adjacent };
    if pool.is_empty() { return from.idx(); }
    let pick = engine::rng::per_agent_u32_pcg(seed as u32, hero_idx, tick, 0xFA000) as usize
        % pool.len();
    pool[pick].idx()
}
