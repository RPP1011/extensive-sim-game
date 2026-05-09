//! `voxel_bridge` — Phase B of the viewer-runtime plan.
//!
//! Bridges [`crate::ViewerApp`]'s per-tick agent snapshot into a
//! single world-spanning [`GpuVoxelTexture`] that
//! [`voxel_engine::render::VoxelRenderer::render_frame_gpu`] consumes.
//!
//! ## Architecture: single grid + per-cell material_id
//!
//! voxel_engine's `gbuffer.frag` (`shaders/gbuffer.frag:184`) samples
//! a `palette_tex[voxel_id]` per cell — per-cell colour comes from
//! the cell's material id indexing the supplied palette, NOT from
//! the per-object `palette_color` push-constant (which only feeds
//! `roughness`). One grid with material ids 1..=4 (node/settler/
//! monster/spawner) renders multi-coloured in one draw call.
//!
//! ## Per-tick mip regeneration
//!
//! `GpuVoxelTexture::update_subregion` only writes the base 3D image;
//! mip1/mip2/mip3 stay at whatever values they had at upload time
//! (`upload_grid_to_gpu` is the only voxel_engine API that runs
//! mip generation). The fragment shader uses mip3 for an
//! empty-block-skip optimisation (8×8×8 cells at a time): if mip3
//! at a block is zero, the DDA jumps past the entire block. With
//! stale all-zero mip3, every ray skips through the volume and
//! discards — only first-iteration entry cells render.
//!
//! Cheapest fix: **destroy + recreate** the texture each tick via
//! `upload_grid_to_gpu`, which re-runs mip generation. At 128³ × 1B
//! per cell = 2 MB the per-tick cost is ~10 µs; trivially under the
//! 100 ms tick budget. A future voxel_engine change to add an
//! `update_mips()` API would make this in-place; for now it's
//! out-and-back.
//!
//! ## Scaling caveats
//!
//! Single-grid approach scales to ~256³ before bandwidth + VRAM
//! become a concern (see `project_viewer_voxel_grid_scaling.md`).
//! Beyond that, switch to chunked rendering or a Sparse Voxel DAG.

use anyhow::Result;
use glam::Vec3;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu::{upload_grid_to_gpu, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

use crate::ViewerApp;

/// Single-grid bridge: one CPU `VoxelGrid` + one GPU
/// `GpuVoxelTexture` covering the whole world. Per-cell material id
/// drives per-cell colour via the palette LUT.
pub struct VoxelBridge {
    /// CPU-side grid mutated each tick.
    cpu_grid: VoxelGrid,
    /// Cells we set last frame — used to clear them this frame
    /// before writing the new agent positions, so the per-tick
    /// rewrite is sparse (O(N_alive), not O(grid_dim³)).
    last_frame_cells: Vec<(u32, u32, u32)>,
    /// Persistent GPU texture, re-allocated each tick to refresh
    /// the mip chain.
    gpu_tex: Option<GpuVoxelTexture>,
    /// Allocator scoped to this bridge.
    alloc: VulkanAllocator,
    /// Cached palette RGBA for re-uploads.
    palette_rgba: [[u8; 4]; 256],
    /// Cell extent. World cell `(i,j,k)` covers world AABB
    /// `[origin + (i,j,k)*cell_size, origin + (i+1,j+1,k+1)*cell_size]`.
    cell_size: f32,
    /// World position of cell `(0,0,0)` — the lower corner of the
    /// grid in world space.
    world_origin: Vec3,
    /// Cached `grid_dim` for the discretisation math.
    grid_dim: u32,
}

impl VoxelBridge {
    /// Allocate the persistent CPU grid + initial GPU texture.
    ///
    /// `grid_dim`: cell count along each axis (so the grid holds
    /// `grid_dim³` cells). `world_extent`: world-space length the
    /// grid covers along each axis (so each cell is
    /// `world_extent / grid_dim` units). The grid is centred on
    /// world origin: cell `(0,0,0)` lands at
    /// `(-world_extent/2, -world_extent/2, -world_extent/2)`.
    pub fn new(
        ctx: &VulkanContext,
        palette: &MaterialPalette,
        grid_dim: u32,
        world_extent: f32,
    ) -> Result<Self> {
        let cell_size = world_extent / grid_dim as f32;
        let world_origin = Vec3::splat(-world_extent * 0.5);
        let cpu_grid = VoxelGrid::new(grid_dim, grid_dim, grid_dim);
        let mut alloc = VulkanAllocator::new(ctx)?;
        let palette_rgba = palette.to_rgba();
        let gpu_tex = upload_grid_to_gpu(ctx, &mut alloc, &cpu_grid, &palette_rgba)?;
        Ok(Self {
            cpu_grid,
            last_frame_cells: Vec::with_capacity(2048),
            gpu_tex: Some(gpu_tex),
            alloc,
            palette_rgba,
            cell_size,
            world_origin,
            grid_dim,
        })
    }

    /// Per-tick refresh: paint cells from the latest [`ViewerApp`]
    /// snapshot, then re-upload the grid (which regenerates mips,
    /// the load-bearing reason this isn't an in-place
    /// `update_subregion`).
    pub fn refresh(&mut self, ctx: &VulkanContext, app: &ViewerApp) -> Result<()> {
        // Sparse clear of last frame's painted cells.
        for &(x, y, z) in &self.last_frame_cells {
            self.cpu_grid.set(x, y, z, 0);
        }
        self.last_frame_cells.clear();

        // Paint this frame: each agent's discretised position gets
        // a cell with material_id == creature_type ordinal (1..=4
        // under `creature_material_index`). Multiple agents
        // discretising to the same cell are last-writer-wins.
        let positions = app.positions();
        let alive = app.alive();
        let creature_types = app.creature_types();
        for slot in 0..positions.len() {
            if alive[slot] == 0 {
                continue;
            }
            let Some((x, y, z)) = world_to_cell(
                positions[slot],
                self.world_origin,
                self.cell_size,
                self.grid_dim,
            ) else {
                continue;
            };
            let material = app.material_for(creature_types[slot]);
            self.cpu_grid.set(x, y, z, material);
            self.last_frame_cells.push((x, y, z));
        }

        // Destroy + recreate the GPU texture so mip1/mip2/mip3 get
        // regenerated. `upload_grid_to_gpu` is the only API in
        // voxel_engine that runs mip generation; `update_subregion`
        // leaves mips stale. Without fresh mips, the fragment
        // shader's empty-block jump-skip (gbuffer.frag:268-312)
        // skips through everything that wasn't there at upload time.
        if let Some(old) = self.gpu_tex.take() {
            old.destroy(ctx, &mut self.alloc);
        }
        let new_tex = upload_grid_to_gpu(ctx, &mut self.alloc, &self.cpu_grid, &self.palette_rgba)?;
        self.gpu_tex = Some(new_tex);
        Ok(())
    }

    /// Build the renderer object tuple for this bridge's texture.
    /// Single object — per-cell colours come from `palette_tex`
    /// inside the fragment shader, not this tuple's `palette_color`.
    /// We pass `[1, 1, 1, 0.5]` so the shader's `roughness =
    /// max(palette_color.a, 0.04)` (gbuffer.frag:200) gets a
    /// medium-roughness surface, which reads as matte in the
    /// deferred lighting.
    pub fn render_object(
        &self,
    ) -> Option<(
        &GpuVoxelTexture,
        [f32; 4], // RGBA — only .a is used (as roughness)
        [f32; 3], // world-space lower corner
        [f32; 3], // world-space dims
    )> {
        let world_extent = self.grid_dim as f32 * self.cell_size;
        let dims = [world_extent; 3];
        let pos: [f32; 3] = self.world_origin.into();
        self.gpu_tex.as_ref().map(|t| (t, [1.0, 1.0, 1.0, 0.5], pos, dims))
    }

    /// Drop the GPU texture explicitly. winit's run_app doesn't
    /// drop the app on exit (it returns from `run_app` then process
    /// exits) so the texture would leak if we relied on `Drop`.
    /// Call this from the exit path after `device.device_wait_idle()`.
    pub fn destroy(mut self, ctx: &VulkanContext) {
        if let Some(tex) = self.gpu_tex.take() {
            tex.destroy(ctx, &mut self.alloc);
        }
    }
}

/// Discretise a world-space position to a grid cell. Returns `None`
/// when the position falls outside the grid (an agent at world
/// coords outside `±world_extent/2` — shouldn't happen for
/// wave_defense, but the bridge stays robust to it).
fn world_to_cell(pos: Vec3, origin: Vec3, cell_size: f32, grid_dim: u32) -> Option<(u32, u32, u32)> {
    let local = (pos - origin) / cell_size;
    let (x, y, z) = (
        local.x.floor() as i32,
        local.y.floor() as i32,
        local.z.floor() as i32,
    );
    if x < 0 || y < 0 || z < 0 {
        return None;
    }
    let (x, y, z) = (x as u32, y as u32, z as u32);
    if x >= grid_dim || y >= grid_dim || z >= grid_dim {
        return None;
    }
    Some((x, y, z))
}
