//! `viewer_runtime` — windowed viewer over `sims::dungeon_horde::GeneratedRuntime`.
//!
//! Reads the GPU agent SoA back to host every sim tick, then mirrors
//! agent positions + types into a single world-spanning CPU
//! `VoxelGrid` the renderer (voxel_engine) consumes via Vulkan. Walls
//! come from the runtime's `voxel_terrain` (dense stone-vs-air);
//! floor cells from the seeded floor map; agents from the per-tick
//! readback.
//!
//! # Two GPU contexts (intentional)
//!
//! - `state.gpu` is a wgpu device used by the sim runtime to drive
//!   compute kernels.
//! - The viewer's renderer (voxel_engine) lives on a separate Vulkan
//!   instance/device created in `bin/viewer_app.rs`.
//!
//! The bridge crosses the boundary by reading sim state back to host
//! (CPU `Vec<u8>` voxel grid) and re-uploading it to the renderer's
//! Vulkan texture each frame. Cheap at 96×96×16 cells.

use anyhow::Result;
use glam::Vec3;
use sims::dungeon_horde::GeneratedRuntime;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu::{upload_grid_to_gpu, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::{MaterialPalette, MaterialType, PaletteEntry};

pub mod dungeon;

use dungeon::{Dungeon, GRID_X, GRID_Y, GRID_Z, N_HEROES, ROOM_INTERIOR_Z};

// ---------------------------------------------------------------------
// Material palette — distinct color per (creature_type, role) plus
// terrain. Indices are referenced from `voxel_grid` cells, so they
// must fit in a u8.
// ---------------------------------------------------------------------

pub const MAT_AIR: u8 = 0;
pub const MAT_FLOOR: u8 = 1;
pub const MAT_WALL: u8 = 2;
/// Floor tint for the boss room — clear visual cue that "this is the climax".
pub const MAT_BOSS_FLOOR: u8 = 3;
// Heroes: per-role base index (warrior=10, cleric=11, ranger=12, mage=13, rogue=14).
pub const MAT_HERO_BASE: u8 = 10;
// Enemies: archer=20, brute=21, goblin=22.
pub const MAT_ARCHER: u8 = 20;
pub const MAT_BRUTE: u8 = 21;
pub const MAT_GOBLIN: u8 = 22;
// Dead agent ghost color.
pub const MAT_DEAD: u8 = 30;
/// Boss agent — singular Brute in the boss room with 4x HP. Painted
/// in a brighter accent so the climax target is immediately visible.
pub const MAT_BOSS: u8 = 31;

fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    p.set(MAT_FLOOR, palette_entry(180, 160, 130)); // light tan
    p.set(MAT_WALL,  palette_entry(95,  92,  88));  // gray stone
    p.set(MAT_BOSS_FLOOR, palette_entry(160, 80, 80));  // ochre-red — boss-room floor
    // Heroes (slot+1 → role)
    p.set(MAT_HERO_BASE + 0, palette_entry(140, 95,  50));  // Warrior — brown
    p.set(MAT_HERO_BASE + 1, palette_entry(240, 240, 240)); // Cleric  — white
    p.set(MAT_HERO_BASE + 2, palette_entry(60,  170, 75));  // Ranger  — green
    p.set(MAT_HERO_BASE + 3, palette_entry(70,  130, 230)); // Mage    — blue
    p.set(MAT_HERO_BASE + 4, palette_entry(170, 80,  220)); // Rogue   — purple
    p.set(MAT_ARCHER, palette_entry(230, 130, 40));  // orange
    p.set(MAT_BRUTE,  palette_entry(150, 35,  35));  // dark red
    p.set(MAT_GOBLIN, palette_entry(95,  140, 70));  // dim green
    p.set(MAT_DEAD,   palette_entry(50,  50,  50));  // gray
    p.set(MAT_BOSS,   palette_entry(255, 60,  20));  // bright orange-red — the boss
    p
}

fn palette_entry(r: u8, g: u8, b: u8) -> PaletteEntry {
    PaletteEntry { r, g, b, roughness: 200, emissive: 0, material_type: MaterialType::Stone }
}

/// Maps a (creature_type, role) tuple to a palette index.
fn material_for(creature_type: u32, role: u32) -> u8 {
    match creature_type {
        dungeon::CT_HERO => {
            let r = role.saturating_sub(1).min(4) as u8;
            MAT_HERO_BASE + r
        }
        dungeon::CT_ARCHER => MAT_ARCHER,
        dungeon::CT_BRUTE  => MAT_BRUTE,
        dungeon::CT_GOBLIN => MAT_GOBLIN,
        _ => MAT_DEAD,
    }
}

/// Per-frame snapshot of a single agent (post-readback from sim GPU).
#[derive(Clone, Copy, Debug)]
pub struct AgentSnapshot {
    pub pos: Vec3, // world-space xyz (sim Z-up); the viewer swaps Z↔Y in the bridge.
    pub alive: bool,
    pub creature_type: u32,
    pub role: u32,
    pub hp: f32,
}

/// The viewer's host-side state. Owns the sim runtime, the CPU voxel
/// grid mirroring its dungeon + agents, and the dungeon roomgen
/// metadata so it can know where the spawn room is for camera framing.
pub struct ViewerApp {
    pub state: GeneratedRuntime,
    pub dungeon: Dungeon,
    pub seed: u64,
    /// Floor cells as 2D (x, y) tuples — painted on top of the
    /// stone fill each refresh.
    floor_cells: std::collections::BTreeSet<(u32, u32)>,
    /// Per-agent latest readback. Index = sim slot id.
    agents: Vec<AgentSnapshot>,
    pub agent_count: u32,
    /// Tick at which sim terminated (TPK or dungeon cleared); `None`
    /// while the sim is still running.
    pub terminated_at_tick: Option<u64>,
    /// Per-hero exploration state — frontier-greedy retargeting
    /// advances `target_room_idx` between ticks. Without this,
    /// heroes stop at the first adjacent room they reach.
    hero_state: dungeon::HeroExploreState,
    /// Agent slot of the singular boss enemy (first Brute placed
    /// in the boss room). `None` if no brute landed in boss_room
    /// (very small dungeons or unlucky rolls).
    pub boss_slot: Option<u32>,
}

impl ViewerApp {
    /// Roll a dungeon, build the runtime, seed walls + agents.
    /// Returns `None` if no wgpu adapter is available (headless host).
    pub fn try_new(seed: u64) -> Option<Self> {
        let dungeon = dungeon::roll_dungeon(seed);
        let agent_count = dungeon.total_agent_count();
        eprintln!(
            "[viewer_runtime] dungeon: {} rooms, spawn=slot{} boss=slot{} \
             agents={} ({} heroes + {} enemies)",
            dungeon.rooms.len(),
            dungeon.spawn_room.idx(),
            dungeon.boss_room.idx(),
            agent_count,
            N_HEROES,
            agent_count - N_HEROES,
        );
        let mut state = GeneratedRuntime::try_new(seed, agent_count)?;
        let floor_cells = dungeon::seed_voxel_dungeon(&mut state, &dungeon, seed);
        let boss_slot = dungeon::seed_topology(&mut state, &dungeon, seed);
        if let Some(slot) = boss_slot {
            // Boss has 4x HP — heroes have to commit to the climax fight.
            // Writes both hp and max_hp so cleric heals don't cap at 200.
            const BOSS_HP: f32 = 800.0;
            state.gpu.queue.write_buffer(
                &state.agent_hp_buf, (slot as u64) * 4, bytemuck::bytes_of(&BOSS_HP),
            );
            state.gpu.queue.write_buffer(
                &state.agent_max_hp_buf, (slot as u64) * 4, bytemuck::bytes_of(&BOSS_HP),
            );
            eprintln!(
                "[viewer_runtime] boss agent slot={slot} hp={BOSS_HP} (in boss room slot{})",
                dungeon.boss_room.idx(),
            );
        } else {
            eprintln!(
                "[viewer_runtime] no Brute landed in boss room slot{} — no distinct boss this roll",
                dungeon.boss_room.idx(),
            );
        }

        let hero_state = dungeon::HeroExploreState::new(&dungeon);
        let mut viewer = Self {
            state,
            dungeon,
            seed,
            floor_cells,
            agents: vec![
                AgentSnapshot {
                    pos: Vec3::ZERO,
                    alive: false,
                    creature_type: 0,
                    role: 0,
                    hp: 0.0,
                };
                agent_count as usize
            ],
            agent_count,
            terminated_at_tick: None,
            hero_state,
            boss_slot,
        };
        viewer.refresh_snapshot();
        Some(viewer)
    }

    pub fn sim_tick(&self) -> u64 {
        self.state.tick
    }

    /// Slice of the latest per-agent snapshot (post-readback).
    pub fn agents(&self) -> &[AgentSnapshot] {
        &self.agents
    }

    /// World-space centroid of the spawn room — the camera looks here.
    pub fn spawn_centroid(&self) -> Vec3 {
        let c = self.dungeon.spawn_room.centroid();
        Vec3::new(c[0], c[1], c[2])
    }

    /// Step sim one tick and refresh per-agent snapshot.
    pub fn step(&mut self) {
        if self.terminated_at_tick.is_some() {
            return;
        }
        self.state.step();
        self.advance_hero_exploration();
        self.refresh_snapshot();
        // Termination heuristic: heroes-alive==0 OR enemies-alive==0.
        let mut heroes_alive = 0u32;
        let mut enemies_alive = 0u32;
        for a in &self.agents {
            if !a.alive {
                continue;
            }
            if a.creature_type == dungeon::CT_HERO {
                heroes_alive += 1;
            } else {
                enemies_alive += 1;
            }
        }
        if (heroes_alive == 0 || enemies_alive == 0) && self.terminated_at_tick.is_none() {
            self.terminated_at_tick = Some(self.state.tick);
        }
    }

    /// Host-side hero target_room_idx advancement.
    ///
    /// Heroes are placed in role order by `seed_topology` — h=0 is
    /// Warrior, h=1 Cleric, h=2 Ranger, h=3 Mage, h=4 Rogue — so
    /// `h` doubles as the role discriminant minus one.
    ///
    /// Each tick: read positions/HPs, recompute current_room for every
    /// hero, then derive new targets via a per-role policy:
    /// * **Warrior** — leader. Standard frontier-greedy.
    /// * **Cleric / Ranger / Mage** — followers. Target = warrior's
    ///   current room (cluster with the tank).
    /// * **Rogue** — scout. Target = warrior's TARGET room (one step
    ///   ahead of the party, so LoS reveals enemies in advance).
    /// * If the warrior is dead, every surviving hero falls back to
    ///   their own frontier-greedy target.
    ///
    /// Wounded heroes (`hp < RETREAT_HP`) override the role policy
    /// and route to spawn until they heal past `RESUME_HP`. Hysteresis
    /// between the two prevents oscillation while the cleric mid-heals.
    fn advance_hero_exploration(&mut self) {
        /// Below this HP heroes break off and head back to spawn.
        const RETREAT_HP: f32 = 45.0;
        /// Once retreating, must heal above this to resume exploring.
        const RESUME_HP: f32 = 170.0;
        const H_WARRIOR: usize = 0;
        const H_ROGUE: usize = 4;

        let agent_count = self.agent_count;
        let hero_start = (agent_count - dungeon::N_HEROES) as usize;
        let positions = read_positions(&mut self.state, agent_count);
        let target_buf = self.state.agent_target_room_idx_buf.clone();
        let hp_buf = self.state.agent_hp_buf.clone();
        let alive_buf = self.state.agent_alive_buf.clone();
        let mut targets = read_agent_u32(&mut self.state, &target_buf, agent_count);
        let hps = read_agent_f32(&mut self.state, &hp_buf, agent_count);
        let alive = read_agent_u32(&mut self.state, &alive_buf, agent_count);
        let present_set: std::collections::BTreeSet<dungeon::RoomSlot> =
            self.dungeon.rooms.iter().copied().collect();
        let tick = self.state.tick as u32;
        let spawn_idx = self.dungeon.spawn_room.idx();
        let mut any_change = false;

        // Pass 1: recompute current_room + rooms_visited for each hero
        // from their current position. None means the hero isn't in any
        // present room (dead, out of bounds, or in a corridor cell).
        let mut current: [Option<dungeon::RoomSlot>; dungeon::N_HEROES as usize] =
            [None; dungeon::N_HEROES as usize];
        for h in 0..dungeon::N_HEROES as usize {
            let p = positions[hero_start + h];
            if !p[0].is_finite() || !p[1].is_finite() {
                continue;
            }
            let rx = (p[0] / dungeon::SLOT_WIDTH as f32).floor() as i32;
            let ry = (p[1] / dungeon::SLOT_WIDTH as f32).floor() as i32;
            if rx < 0 || ry < 0
                || (rx as u32) >= dungeon::SLOTS_PER_ROW
                || (ry as u32) >= dungeon::SLOTS_PER_ROW
            {
                continue;
            }
            let candidate = dungeon::RoomSlot::new(rx as u32, ry as u32);
            if !present_set.contains(&candidate) {
                continue;
            }
            current[h] = Some(candidate);
            self.hero_state.current_room[h] = Some(candidate.idx());
            let remap = self.dungeon.rooms.iter().position(|r| *r == candidate).unwrap();
            self.hero_state.rooms_visited[h] |= 1u32 << remap;
        }

        // Pass 2: compute the warrior's target first — it anchors the
        // party formation. Followers + rogue read off this in pass 3.
        let warrior_alive = alive[hero_start + H_WARRIOR] != 0;
        let warrior_current = current[H_WARRIOR];
        let warrior_target_after = {
            let hp = hps[hero_start + H_WARRIOR];
            let cur_target = targets[hero_start + H_WARRIOR];
            let retreating = cur_target == spawn_idx
                && warrior_current.map_or(true, |c| c.idx() != spawn_idx);
            match warrior_current {
                Some(cand) if warrior_alive => {
                    if retreating && hp >= RESUME_HP {
                        dungeon::pick_next_target(
                            &self.dungeon, cand,
                            self.hero_state.rooms_visited[H_WARRIOR],
                            tick, H_WARRIOR as u32, self.seed,
                        )
                    } else if !retreating && hp < RETREAT_HP {
                        spawn_idx
                    } else if !retreating && cand.idx() == cur_target {
                        dungeon::pick_next_target(
                            &self.dungeon, cand,
                            self.hero_state.rooms_visited[H_WARRIOR],
                            tick, H_WARRIOR as u32, self.seed,
                        )
                    } else {
                        cur_target
                    }
                }
                _ => cur_target,
            }
        };
        if warrior_target_after != targets[hero_start + H_WARRIOR] {
            targets[hero_start + H_WARRIOR] = warrior_target_after;
            any_change = true;
        }

        // Pass 3: every other hero. Followers cluster with the warrior;
        // the rogue scouts one step ahead. If the warrior is dead,
        // everyone falls back to per-hero frontier-greedy.
        for h in 0..dungeon::N_HEROES as usize {
            if h == H_WARRIOR { continue; }
            let cand = match current[h] {
                Some(c) => c,
                None => continue,
            };
            let hp = hps[hero_start + h];
            let cur_target = targets[hero_start + h];
            let retreating = cur_target == spawn_idx && cand.idx() != spawn_idx;

            let new_target = if retreating && hp >= RESUME_HP {
                dungeon::pick_next_target(
                    &self.dungeon, cand, self.hero_state.rooms_visited[h],
                    tick, h as u32, self.seed,
                )
            } else if !retreating && hp < RETREAT_HP {
                spawn_idx
            } else if !warrior_alive {
                // No leader — fall back to per-hero exploration.
                if cand.idx() == cur_target {
                    dungeon::pick_next_target(
                        &self.dungeon, cand, self.hero_state.rooms_visited[h],
                        tick, h as u32, self.seed,
                    )
                } else {
                    cur_target
                }
            } else if h == H_ROGUE {
                // Scout one room ahead of the party.
                warrior_target_after
            } else {
                // Cleric / Ranger / Mage — cluster with the warrior.
                warrior_current.map_or(cur_target, |c| c.idx())
            };
            if new_target != cur_target {
                targets[hero_start + h] = new_target;
                any_change = true;
            }
        }

        if any_change {
            self.state.gpu.queue.write_buffer(
                &self.state.agent_target_room_idx_buf, 0,
                bytemuck::cast_slice(&targets),
            );
        }
    }

    /// Read agent SoA back from sim GPU into the viewer's host cache.
    fn refresh_snapshot(&mut self) {
        let n = self.agent_count;
        // Clone the buffer handles up front to release the &self.state
        // borrow before re-borrowing &mut self.state in the readback
        // helper. wgpu::Buffer is Arc-backed; clone is cheap.
        let alive_buf = self.state.agent_alive_buf.clone();
        let type_buf = self.state.agent_creature_type_buf.clone();
        let role_buf = self.state.agent_role_buf.clone();
        let hp_buf = self.state.agent_hp_buf.clone();
        let positions = read_positions(&mut self.state, n);
        let alive = read_agent_u32(&mut self.state, &alive_buf, n);
        let types = read_agent_u32(&mut self.state, &type_buf, n);
        let role  = read_agent_u32(&mut self.state, &role_buf, n);
        let hps   = read_agent_f32(&mut self.state, &hp_buf, n);
        for slot in 0..n as usize {
            self.agents[slot] = AgentSnapshot {
                pos: Vec3::new(positions[slot][0], positions[slot][1], positions[slot][2]),
                alive: alive[slot] != 0,
                creature_type: types[slot],
                role: role[slot],
                hp: hps[slot],
            };
        }
    }
}

// ---------------------------------------------------------------------
// VoxelBridge — CPU-side world grid, painted from terrain + agents,
// re-uploaded to a Vulkan texture each frame for voxel_engine to
// render.
// ---------------------------------------------------------------------

/// Cell extent of the bridge's voxel grid. Matches the dungeon's
/// `GRID_X × GRID_Y × GRID_Z` plus a few cells of vertical headroom
/// for agent splats above the floor.
pub const BRIDGE_DIM_X: u32 = GRID_X;
pub const BRIDGE_DIM_Y: u32 = GRID_Z + 4; // vertical (renderer Y-up); GRID_Z + headroom for agent splats
pub const BRIDGE_DIM_Z: u32 = GRID_Y; // depth (sim's horizontal Y axis)

/// World-space `(0, 0, 0)` corner of the bridge grid. We anchor at
/// world origin; sim positions are in `[0, GRID_X)` so they map
/// directly into cells.
pub const BRIDGE_WORLD_ORIGIN: Vec3 = Vec3::new(0.0, 0.0, 0.0);
pub const BRIDGE_CELL_SIZE: f32 = 1.0;

/// Paints the dungeon walls + floor + agents into a `VoxelGrid` and
/// re-uploads it to a `GpuVoxelTexture` each frame. Single object,
/// per-cell coloring via `palette_tex[voxel_id]` in voxel_engine's
/// fragment shader.
pub struct VoxelBridge {
    cpu_grid: VoxelGrid,
    /// Cells painted by agents last frame — cleared before this
    /// frame's paint to keep the agent layer sparse.
    last_agent_cells: Vec<(u32, u32, u32)>,
    gpu_tex: Option<GpuVoxelTexture>,
    alloc: VulkanAllocator,
    palette_rgba: [[u8; 4]; 256],
}

impl VoxelBridge {
    pub fn new(ctx: &VulkanContext, app: &ViewerApp) -> Result<Self> {
        let palette = build_palette();
        let palette_rgba = palette.to_rgba();
        let mut cpu_grid = VoxelGrid::new(BRIDGE_DIM_X, BRIDGE_DIM_Y, BRIDGE_DIM_Z);

        // Paint the dungeon: floor cells get MAT_FLOOR at z=0; wall
        // cells get MAT_WALL filling z ∈ [0, ROOM_INTERIOR_Z). This
        // matches the runtime's voxel_terrain contents seeded by
        // `seed_voxel_dungeon`.
        let boss_room = app.dungeon.boss_room;
        let boss_x_lo = boss_room.rx * dungeon::SLOT_WIDTH;
        let boss_x_hi = boss_x_lo + dungeon::SLOT_WIDTH;
        let boss_y_lo = boss_room.ry * dungeon::SLOT_WIDTH;
        let boss_y_hi = boss_y_lo + dungeon::SLOT_WIDTH;
        let in_boss_room = |x: u32, y: u32| -> bool {
            x >= boss_x_lo && x < boss_x_hi && y >= boss_y_lo && y < boss_y_hi
        };
        for x in 0..GRID_X {
            for y in 0..GRID_Y {
                if app.floor_cells.contains(&(x, y)) {
                    // sim (x, y, 0) → renderer (x, 0, y): floor sits at vertical=0, depth=sim_y
                    let mat = if in_boss_room(x, y) { MAT_BOSS_FLOOR } else { MAT_FLOOR };
                    cpu_grid.set(x, 0, y, mat);
                } else {
                    // walls extend up the vertical (renderer Y) axis
                    for vert in 0..ROOM_INTERIOR_Z.min(GRID_Z) {
                        cpu_grid.set(x, vert, y, MAT_WALL);
                    }
                }
            }
        }

        let mut alloc = VulkanAllocator::new(ctx)?;
        let gpu_tex = upload_grid_to_gpu(ctx, &mut alloc, &cpu_grid, &palette_rgba)?;
        Ok(Self {
            cpu_grid,
            last_agent_cells: Vec::with_capacity(2048),
            gpu_tex: Some(gpu_tex),
            alloc,
            palette_rgba,
        })
    }

    /// Per-frame refresh: clear last frame's agent cells, paint new
    /// agent positions, re-upload texture (regenerates mips).
    pub fn refresh(&mut self, ctx: &VulkanContext, app: &ViewerApp) -> Result<()> {
        // Clear last frame's agents — restore floor or air based on
        // dungeon membership. Walls aren't in the agent layer (they
        // sit at z ∈ [0, ROOM_INTERIOR_Z)) so we don't touch them.
        for &(x, vert, depth) in &self.last_agent_cells {
            // Restore each agent cell. Note: post axis-swap, the bridge
            // grid is (x, vertical, depth). Floor occupancy is keyed on
            // sim coords (x, sim_y) which now equals (x, depth).
            if app.floor_cells.contains(&(x, depth)) {
                if vert == 0 {
                    let boss_room = app.dungeon.boss_room;
                    let in_boss = x >= boss_room.rx * dungeon::SLOT_WIDTH
                        && x < (boss_room.rx + 1) * dungeon::SLOT_WIDTH
                        && depth >= boss_room.ry * dungeon::SLOT_WIDTH
                        && depth < (boss_room.ry + 1) * dungeon::SLOT_WIDTH;
                    let mat = if in_boss { MAT_BOSS_FLOOR } else { MAT_FLOOR };
                    self.cpu_grid.set(x, vert, depth, mat);
                } else {
                    self.cpu_grid.set(x, vert, depth, MAT_AIR);
                }
            } else {
                if vert < ROOM_INTERIOR_Z.min(GRID_Z) {
                    self.cpu_grid.set(x, vert, depth, MAT_WALL);
                } else {
                    self.cpu_grid.set(x, vert, depth, MAT_AIR);
                }
            }
        }
        self.last_agent_cells.clear();

        // Paint each alive agent as a 2x2x2 splat one cell above the
        // floor (z ∈ [1, 3)) so they sit visibly on the floor without
        // sinking. Out-of-bounds cells are skipped.
        for (slot_idx, agent) in app.agents().iter().enumerate() {
            if !agent.alive {
                continue;
            }
            let mat = if app.boss_slot == Some(slot_idx as u32) {
                MAT_BOSS
            } else {
                material_for(agent.creature_type, agent.role)
            };
            let cx = agent.pos.x.floor() as i32;
            let cd = agent.pos.y.floor() as i32; // sim_y → renderer depth
            // 2x2x2 splat: span (x, x+1) × vertical (1..3) × depth (cd, cd+1)
            for dx in 0..2 {
                for d_depth in 0..2 {
                    for d_vert in 0..2 {
                        let x = cx + dx;
                        let depth = cd + d_depth;
                        let vert = 1 + d_vert;
                        if x < 0 || depth < 0 || vert < 0 {
                            continue;
                        }
                        let (x, vert, depth) = (x as u32, vert as u32, depth as u32);
                        if x >= BRIDGE_DIM_X || vert >= BRIDGE_DIM_Y || depth >= BRIDGE_DIM_Z {
                            continue;
                        }
                        self.cpu_grid.set(x, vert, depth, mat);
                        self.last_agent_cells.push((x, vert, depth));
                    }
                }
            }
        }

        // Destroy + re-upload to refresh the mip chain. voxel_engine's
        // gbuffer.frag uses mip3 for empty-block jump-skip; without a
        // refresh, agent cells painted after the initial upload would
        // render at one cell's worth of geometry only.
        if let Some(old) = self.gpu_tex.take() {
            old.destroy(ctx, &mut self.alloc);
        }
        let new_tex =
            upload_grid_to_gpu(ctx, &mut self.alloc, &self.cpu_grid, &self.palette_rgba)?;
        self.gpu_tex = Some(new_tex);
        Ok(())
    }

    /// Single render-object tuple for `Renderer::render_frame_gpu`.
    /// Per-cell color comes from the palette LUT (per-cell material
    /// id), not this tuple's `palette_color`.
    pub fn render_object(
        &self,
    ) -> Option<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])> {
        let dims = [
            BRIDGE_DIM_X as f32 * BRIDGE_CELL_SIZE,
            BRIDGE_DIM_Y as f32 * BRIDGE_CELL_SIZE,
            BRIDGE_DIM_Z as f32 * BRIDGE_CELL_SIZE,
        ];
        let pos: [f32; 3] = BRIDGE_WORLD_ORIGIN.into();
        self.gpu_tex
            .as_ref()
            .map(|t| (t, [1.0, 1.0, 1.0, 0.5], pos, dims))
    }

    /// Drop the GPU texture explicitly. winit's run_app doesn't drop
    /// the app on exit; without this the texture leaks until process
    /// teardown.
    pub fn destroy(mut self, ctx: &VulkanContext) {
        if let Some(tex) = self.gpu_tex.take() {
            tex.destroy(ctx, &mut self.alloc);
        }
    }
}

// ---------------------------------------------------------------------
// GPU readback helpers — copy from sim's wgpu buffers to host Vec.
// Mirrors `dungeon_horde_pin`'s helpers.
// ---------------------------------------------------------------------

fn read_positions(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<[f32; 4]> {
    let n = agent_count as usize;
    let bytes = (n as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer_runtime::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viewer_runtime::pos_readback") },
    );
    let buf = state.agent_pos_buf.clone();
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[[f32; 4]] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

fn read_agent_u32(
    state: &mut GeneratedRuntime,
    buf: &wgpu::Buffer,
    agent_count: u32,
) -> Vec<u32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer_runtime::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viewer_runtime::u32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn read_agent_f32(
    state: &mut GeneratedRuntime,
    buf: &wgpu::Buffer,
    agent_count: u32,
) -> Vec<f32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer_runtime::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viewer_runtime::f32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
