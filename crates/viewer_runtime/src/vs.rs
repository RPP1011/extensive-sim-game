//! Vampire Survivors voxel-viewer path — parallel to the dungeon_horde ViewerApp.
//! Sim-side here (state, seeding, step+drain, mana-band snapshot); rendering in VsBridge (VD2).
use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::seed_initial_state;
use sims::summon_alloc::{drain_summons, DrainCtx};

use anyhow::Result;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu::{upload_grid_to_gpu, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::{MaterialPalette, MaterialType, PaletteEntry};

use crate::dungeon::{GRID_X, GRID_Y};
use crate::{BRIDGE_DIM_X, BRIDGE_DIM_Y, BRIDGE_DIM_Z, MAT_AIR};

pub const VS_AGENT_COUNT: u32 = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VsRole { Player, Enemy, Spawner }

pub fn role_for_mana(mana: f32) -> VsRole {
    if mana < 1.5 { VsRole::Player } else if mana < 2.5 { VsRole::Enemy } else { VsRole::Spawner }
}

#[derive(Clone, Copy)]
pub struct VsAgent { pub pos: [f32; 3], pub hp: f32, pub role: VsRole, pub move_speed: f32 }

pub struct VsViewerApp {
    pub state: GeneratedRuntime,
    pub seed: u64,
    pub agent_count: u32,
    agents: Vec<VsAgent>,
    pub terminated_at_tick: Option<u64>,
}

impl VsViewerApp {
    pub fn try_new(seed: u64) -> Option<Self> {
        let mut state = GeneratedRuntime::try_new(seed, VS_AGENT_COUNT)?;
        seed_initial_state(&mut state);
        let mut app = Self { state, seed, agent_count: VS_AGENT_COUNT, agents: Vec::new(), terminated_at_tick: None };
        app.refresh_snapshot();
        Some(app)
    }
    pub fn sim_tick(&self) -> u64 { self.state.tick }
    pub fn agents(&self) -> &[VsAgent] { &self.agents }
    pub fn step(&mut self) {
        self.state.step();
        let _ = {
            let device = &self.state.gpu.device;
            let queue = &self.state.gpu.queue;
            let event_ring = &self.state.event_ring;
            let agent_alive_buf = &self.state.agent_alive_buf;
            let agent_pos_buf = &self.state.agent_pos_buf;
            let agent_hp_buf = &self.state.agent_hp_buf;
            let agent_move_speed_buf = &self.state.agent_move_speed_buf;
            let agent_count = self.state.agent_count;
            let seed = self.state.seed;
            let tick = self.state.tick;
            drain_summons(DrainCtx {
                device,
                queue,
                event_ring,
                agent_alive_buf,
                agent_pos_buf,
                agent_hp_buf,
                agent_move_speed_buf,
                agent_count,
                seed,
                tick,
                pool_start: sims::vampire_survivors_seed::ENEMY_POOL_START,
            })
        };
        self.refresh_snapshot();
        if self.terminated_at_tick.is_none() && !self.agents.iter().any(|a| a.role == VsRole::Player) {
            self.terminated_at_tick = Some(self.state.tick);
        }
    }
    fn refresh_snapshot(&mut self) {
        let n = self.agent_count;
        // Clone buffer handles before borrowing &mut self.state — wgpu::Buffer is Arc-backed, clone is cheap.
        let pos_buf  = self.state.agent_pos_buf.clone();
        let alive_buf = self.state.agent_alive_buf.clone();
        let hp_buf   = self.state.agent_hp_buf.clone();
        let mana_buf = self.state.agent_mana_buf.clone();
        let ms_buf   = self.state.agent_move_speed_buf.clone();
        let pos  = read_vec4(&mut self.state, &pos_buf, n);
        let alive = read_u32(&mut self.state, &alive_buf, n);
        let hp   = read_f32(&mut self.state, &hp_buf, n);
        let mana = read_f32(&mut self.state, &mana_buf, n);
        let ms   = read_f32(&mut self.state, &ms_buf, n);
        self.agents.clear();
        for i in 0..n as usize {
            if alive[i] == 1 {
                self.agents.push(VsAgent {
                    pos: [pos[i][0], pos[i][1], pos[i][2]],
                    hp: hp[i],
                    role: role_for_mana(mana[i]),
                    move_speed: ms[i],
                });
            }
        }
    }
}

// Inline staging-buffer readbacks (synchronous map via device.poll(Wait)).
fn read_raw_u32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let bytes = bytes.max(16);
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vs::rb"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    out
}

fn read_u32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<u32> {
    read_raw_u32(rt, buf, n as u64 * 4)
}

fn read_f32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<f32> {
    read_raw_u32(rt, buf, n as u64 * 4)
        .into_iter()
        .map(f32::from_bits)
        .collect()
}

/// Returns `n` vec4 values as `Vec<[f32; 4]>`. agent_pos_buf stride is 16 bytes (vec3<f32> padded to vec4).
fn read_vec4(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<[f32; 4]> {
    let raw = read_raw_u32(rt, buf, n as u64 * 16);
    raw.chunks_exact(4)
        .map(|c| [f32::from_bits(c[0]), f32::from_bits(c[1]), f32::from_bits(c[2]), f32::from_bits(c[3])])
        .collect()
}


// -----------------------------------------------------------------------
// VsBridge — flat-arena voxel renderer for VsViewerApp.
// Mirrors VoxelBridge in lib.rs but paints an open flat floor instead of
// dungeon walls/rooms, and uses mana-band role for color rather than
// creature_type.
// -----------------------------------------------------------------------

/// Palette material indices for Vampire Survivors (unused by the dungeon palette).
pub const MAT_VS_FLOOR:   u8 = 200; // grey flat floor
pub const MAT_VS_PLAYER:  u8 = 201; // cyan — the player
pub const MAT_VS_ENEMY:   u8 = 202; // orange-red — enemies
pub const MAT_VS_SPAWNER: u8 = 203; // purple — spawner beacons

/// Map a VsRole to its palette material index.
pub fn material_for_vs(role: VsRole) -> u8 {
    match role {
        VsRole::Player  => MAT_VS_PLAYER,
        VsRole::Enemy   => MAT_VS_ENEMY,
        VsRole::Spawner => MAT_VS_SPAWNER,
    }
}

/// Convert a VS world-space position (origin-centered, XY plane) to a
/// voxel cell  in the bridge grid.
///
/// The arena is origin-centered; spawners sit at radius ~40. The grid is
/// [0, GRID_X) = [0, 96) so we add GRID_X/2 = 48 to both world X and world Y
/// to center the origin in the grid. The sim's XY plane maps to renderer XZ
/// (floor plane); we splat at vertical y = 1 (one cell above the floor layer).
///
/// Returns  if the resulting cell is outside the grid bounds.
pub fn vs_world_to_voxel(pos: [f32; 3]) -> Option<(u32, u32, u32)> {
    let cx = pos[0] + (GRID_X as f32 / 2.0);
    let cz = pos[1] + (GRID_Y as f32 / 2.0);
    if cx < 0.0 || cz < 0.0 { return None; }
    let x = cx.floor() as u32;
    let z = cz.floor() as u32;
    if x >= GRID_X || z >= GRID_Y { return None; }
    // Splat one cell above the floor (vert = 1) to sit visibly on top.
    let y: u32 = 1;
    if y >= BRIDGE_DIM_Y { return None; }
    Some((x, y, z))
}

fn vs_palette_entry(r: u8, g: u8, b: u8) -> PaletteEntry {
    PaletteEntry { r, g, b, roughness: 200, emissive: 0, material_type: MaterialType::Stone }
}

fn build_vs_palette() -> [[u8; 4]; 256] {
    let mut p = MaterialPalette::new();
    p.set(MAT_VS_FLOOR,   vs_palette_entry(110, 110, 110)); // grey
    p.set(MAT_VS_PLAYER,  vs_palette_entry(0,   220, 220)); // cyan
    p.set(MAT_VS_ENEMY,   vs_palette_entry(230, 100,  20)); // orange-red
    p.set(MAT_VS_SPAWNER, vs_palette_entry(160,  60, 220)); // purple
    p.to_rgba()
}

/// Voxel-renderer bridge for VsViewerApp. Owns a CPU voxel grid + GPU
/// texture, painted from the flat arena floor + per-agent role colors.
pub struct VsBridge {
    cpu_grid: VoxelGrid,
    /// Cells occupied by agents last frame — cleared to MAT_VS_FLOOR before
    /// this frame's paint so the agent layer stays sparse.
    last_agent_cells: Vec<(u32, u32, u32)>,
    gpu_tex: Option<GpuVoxelTexture>,
    alloc: VulkanAllocator,
    palette_rgba: [[u8; 4]; 256],
}

impl VsBridge {
    /// Construct the bridge: build a flat-floor grid, upload to GPU.
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let palette_rgba = build_vs_palette();
        let mut cpu_grid = VoxelGrid::new(BRIDGE_DIM_X, BRIDGE_DIM_Y, BRIDGE_DIM_Z);

        // Flat floor: every (x, z) cell at vert=0 gets MAT_VS_FLOOR.
        // No walls — this is an open arena.
        for x in 0..GRID_X {
            for y in 0..GRID_Y {
                cpu_grid.set(x, 0, y, MAT_VS_FLOOR);
            }
        }

        let mut alloc = VulkanAllocator::new(ctx)?;
        let gpu_tex = upload_grid_to_gpu(ctx, &mut alloc, &cpu_grid, &palette_rgba)?;
        Ok(Self {
            cpu_grid,
            last_agent_cells: Vec::with_capacity(512),
            gpu_tex: Some(gpu_tex),
            alloc,
            palette_rgba,
        })
    }

    /// Per-frame refresh: clear last frame's agent cells, paint new agent
    /// positions, re-upload texture.
    pub fn refresh(&mut self, ctx: &VulkanContext, app: &VsViewerApp) -> Result<()> {
        // Clear last frame's agent splats — restore each cell to floor or air.
        for &(x, vert, depth) in &self.last_agent_cells {
            if vert == 0 {
                self.cpu_grid.set(x, vert, depth, MAT_VS_FLOOR);
            } else {
                self.cpu_grid.set(x, vert, depth, MAT_AIR);
            }
        }
        self.last_agent_cells.clear();

        // Paint each agent as a single voxel cell at its computed grid position.
        for agent in app.agents() {
            let Some((x, vert, depth)) = vs_world_to_voxel(agent.pos) else { continue };
            let mat = material_for_vs(agent.role);
            self.cpu_grid.set(x, vert, depth, mat);
            self.last_agent_cells.push((x, vert, depth));
        }

        // Destroy old texture + re-upload to refresh the mip chain, exactly as
        // VoxelBridge::refresh does. voxel_engine's jump-skip reads mip3; without
        // a fresh upload, newly painted cells wouldn't render.
        if let Some(old) = self.gpu_tex.take() {
            old.destroy(ctx, &mut self.alloc);
        }
        let new_tex = upload_grid_to_gpu(ctx, &mut self.alloc, &self.cpu_grid, &self.palette_rgba)?;
        self.gpu_tex = Some(new_tex);
        Ok(())
    }

    /// Single render-object tuple for .
    pub fn render_object(&self) -> Option<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])> {
        use crate::BRIDGE_CELL_SIZE;
        let dims = [
            BRIDGE_DIM_X as f32 * BRIDGE_CELL_SIZE,
            BRIDGE_DIM_Y as f32 * BRIDGE_CELL_SIZE,
            BRIDGE_DIM_Z as f32 * BRIDGE_CELL_SIZE,
        ];
        let pos = [0.0f32, 0.0, 0.0];
        self.gpu_tex.as_ref().map(|t| (t, [1.0, 1.0, 1.0, 0.5], pos, dims))
    }

    /// Explicitly drop GPU resources. winit's run_app doesn't drop on exit.
    pub fn destroy(mut self, ctx: &VulkanContext) {
        if let Some(tex) = self.gpu_tex.take() {
            tex.destroy(ctx, &mut self.alloc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn role_bands() {
        assert_eq!(role_for_mana(1.0), VsRole::Player);
        assert_eq!(role_for_mana(2.0), VsRole::Enemy);
        assert_eq!(role_for_mana(3.0), VsRole::Spawner);
    }
    #[test]
    fn world_to_voxel_centers_origin() {
        assert_eq!(vs_world_to_voxel([0.0, 0.0, 0.0]).map(|(x,_,z)|(x,z)), Some((48, 48)));
        // an agent at world (-40,0) (a spawner) maps inside the grid:
        let m = vs_world_to_voxel([-40.0, 0.0, 0.0]); assert!(m.is_some(), "spawner at -40 should be in-grid: {m:?}");
        // far outside is clipped:
        assert_eq!(vs_world_to_voxel([1000.0, 0.0, 0.0]), None);
    }
    #[test]
    fn material_for_vs_distinct() {
        assert_ne!(material_for_vs(VsRole::Player), material_for_vs(VsRole::Enemy));
        assert_ne!(material_for_vs(VsRole::Enemy), material_for_vs(VsRole::Spawner));
    }
}
