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
//! `GRID_X/Y/Z`, `SLOTS_PER_ROW`, `SLOT_WIDTH`, `ROOM_INTERIOR_Z`,
//! `STONE`, `TARGET_ROOMS`, `CA_INIT_WALL_PCT`, `CA_ITERATIONS` —
//! same values as `dungeon_horde_pin.rs`. If the pin changes any
//! of these, the viewer's render will diverge from the test scene.

use glam::IVec3;
use sims::dungeon_horde::GeneratedRuntime;

pub const N_HEROES: u32 = 5;

pub const GRID_X: u32 = 72;
pub const GRID_Y: u32 = 72;
pub const GRID_Z: u32 = 8;
pub const SLOTS_PER_ROW: u32 = 6;
pub const SLOT_WIDTH: u32 = 12;
pub const ROOM_INTERIOR_Z: u32 = 6;
pub const STONE: u8 = 1;

pub const TARGET_ROOMS: usize = 22;
pub const CA_INIT_WALL_PCT: u32 = 38;
pub const CA_ITERATIONS: usize = 4;

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
            let raw_count = if room == self.boss_room {
                (n_floor / 2).min(100)
            } else if dist == 1 {
                n_floor / 12
            } else if dist == 2 {
                n_floor / 6
            } else if dist == 3 {
                n_floor / 4
            } else {
                n_floor / 3
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

pub fn roll_dungeon(initial_seed: u64) -> Dungeon {
    let mut seed = initial_seed;
    for _attempt in 0..20 {
        if let Some(d) = try_roll_dungeon(seed, initial_seed) {
            return d;
        }
        seed = seed.wrapping_add(1);
    }
    panic!("dungeon roomgen failed to converge in 20 re-rolls");
}

fn try_roll_dungeon(seed: u64, original_seed: u64) -> Option<Dungeon> {
    let seed32 = seed as u32;
    let mut present = std::collections::BTreeSet::<RoomSlot>::new();
    let start_rx = (engine::rng::per_agent_u32_pcg(seed32, 0, 0, 1) % SLOTS_PER_ROW)
        .clamp(1, SLOTS_PER_ROW - 2);
    let start_ry = (engine::rng::per_agent_u32_pcg(seed32, 1, 0, 1) % SLOTS_PER_ROW)
        .clamp(1, SLOTS_PER_ROW - 2);
    let start = RoomSlot::new(start_rx, start_ry);
    present.insert(start);

    let mut stack: Vec<RoomSlot> = vec![start];
    let mut step = 0u32;
    while present.len() < TARGET_ROOMS && !stack.is_empty() {
        let cur = *stack.last().unwrap();
        let mut cands: Vec<RoomSlot> = Vec::with_capacity(4);
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = cur.rx as i32 + dx;
            let ny = cur.ry as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as u32) < SLOTS_PER_ROW && (ny as u32) < SLOTS_PER_ROW {
                let n = RoomSlot::new(nx as u32, ny as u32);
                if !present.contains(&n) {
                    cands.push(n);
                }
            }
        }
        if cands.is_empty() {
            stack.pop();
            continue;
        }
        let pick = engine::rng::per_agent_u32_pcg(seed32, cur.idx(), step, 2) as usize % cands.len();
        let chosen = cands[pick];
        present.insert(chosen);
        stack.push(chosen);
        step = step.wrapping_add(1);
    }

    let rooms: Vec<RoomSlot> = present.iter().copied().collect();
    if rooms.len() < 12 {
        return None;
    }

    let mut floor_cells = std::collections::BTreeMap::<u32, Vec<(u32, u32)>>::new();
    for &r in &rooms {
        let cells = carve_room_interior(seed32, r);
        if cells.is_empty() {
            return None;
        }
        floor_cells.insert(r.idx(), cells);
    }

    let present_set: std::collections::BTreeSet<RoomSlot> = rooms.iter().copied().collect();
    let spawn_room = *rooms.iter().min_by_key(|r| r.rx + r.ry).unwrap();

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
    if bfs_dist.len() != rooms.len() {
        return None;
    }

    let boss_room = *rooms
        .iter()
        .max_by_key(|r| bfs_dist.get(&r.idx()).copied().unwrap_or(0))
        .unwrap();

    Some(Dungeon { rooms, floor_cells, bfs_dist, spawn_room, boss_room, seed: original_seed })
}

fn carve_room_interior(seed: u32, room: RoomSlot) -> Vec<(u32, u32)> {
    let x0 = room.rx * SLOT_WIDTH;
    let y0 = room.ry * SLOT_WIDTH;
    let mut grid: [[bool; SLOT_WIDTH as usize]; SLOT_WIDTH as usize] =
        [[false; SLOT_WIDTH as usize]; SLOT_WIDTH as usize];
    for ly in 1..(SLOT_WIDTH - 1) {
        for lx in 1..(SLOT_WIDTH - 1) {
            let r = engine::rng::per_agent_u32_pcg(seed, x0 + lx, y0 + ly, 1) % 100;
            grid[ly as usize][lx as usize] = r < CA_INIT_WALL_PCT;
        }
    }
    for _ in 0..CA_ITERATIONS {
        let mut next = grid;
        for ly in 1..(SLOT_WIDTH - 1) {
            for lx in 1..(SLOT_WIDTH - 1) {
                let mut walls = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = lx as i32 + dx;
                        let ny = ly as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= SLOT_WIDTH as i32 || ny >= SLOT_WIDTH as i32 {
                            walls += 1;
                            continue;
                        }
                        if grid[ny as usize][nx as usize] { walls += 1; }
                    }
                }
                let was_wall = grid[ly as usize][lx as usize];
                next[ly as usize][lx as usize] = walls >= 5 || (was_wall && walls >= 4);
            }
        }
        grid = next;
    }
    for i in 0..SLOT_WIDTH {
        grid[0][i as usize] = false;
        grid[(SLOT_WIDTH - 1) as usize][i as usize] = false;
        grid[i as usize][0] = false;
        grid[i as usize][(SLOT_WIDTH - 1) as usize] = false;
    }
    let mut floor = Vec::with_capacity((SLOT_WIDTH * SLOT_WIDTH) as usize);
    for ly in 0..SLOT_WIDTH {
        for lx in 0..SLOT_WIDTH {
            if !grid[ly as usize][lx as usize] {
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
pub fn seed_topology(state: &mut GeneratedRuntime, dungeon: &Dungeon, seed: u64) {
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
    for (i, &(room, _ct)) in brutes.iter().enumerate() {
        place_enemy(
            slot, room, CT_BRUTE, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            0xB23_C7E, i as u32,
        );
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
