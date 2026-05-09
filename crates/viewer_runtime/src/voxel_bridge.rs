//! `voxel_bridge` — Phase B of the viewer-runtime plan.
//!
//! Bridges [`crate::ViewerApp`]'s per-tick agent snapshot into a
//! single world-spanning [`GpuVoxelTexture`] that
//! [`voxel_engine::render::VoxelRenderer::render_frame_gpu`] consumes.
//!
//! Architecture:
//! - One persistent CPU-side [`VoxelGrid`] sized `grid_dim³` covers
//!   world bounds `[-world_extent/2, +world_extent/2]³`.
//! - One persistent [`GpuVoxelTexture`] mirrors it (via
//!   [`update_subregion`]).
//! - Per-tick refresh is **sparse**: clear last frame's set cells +
//!   set this frame's. At ~360 alive agents in wave_defense the
//!   touched-cell count stays well below the grid total, so the
//!   per-tick CPU cost is O(N_alive) not O(grid_dim³).
//! - The GPU upload still happens over the full grid extent in this
//!   slice (one `update_subregion` covering the whole grid). A
//!   per-affected-region upload is a future micro-optimisation; at
//!   128³ × 1B per cell = 2 MB the per-tick bandwidth is trivial.
//!
//! Scaling caveats noted in `project_viewer_voxel_grid_scaling.md`
//! (memory): the single-grid approach holds up to ~256³; beyond that
//! the natural fix is voxel_engine's chunked rendering or a Sparse
//! Voxel DAG for static terrain.

use anyhow::Result;
use glam::Vec3;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu::{upload_grid_to_gpu, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::MaterialPalette;

use crate::ViewerApp;

/// Single world-grid bridge for Phase B.
///
/// Owns the GPU texture + its allocator; both must outlive every
/// `render_frame_gpu` call that references the texture. Drop the
/// bridge to release VRAM.
pub struct VoxelBridge {
    /// CPU-side grid mutated each tick. `grid_dim³` cells.
    cpu_grid: VoxelGrid,
    /// Cells we set last frame — used to clear them this frame
    /// before writing the new agent positions, so the per-tick
    /// rewrite is sparse.
    last_frame_cells: Vec<(u32, u32, u32)>,
    /// Persistent GPU mirror of `cpu_grid`. Allocated once in
    /// [`Self::new`]; `update_subregion` rewrites it in place each
    /// tick.
    gpu_tex: GpuVoxelTexture,
    /// Allocator that owns `gpu_tex`'s memory. Voxel_engine's
    /// allocator is per-context; we keep our own scoped to the
    /// bridge so the texture's lifetime matches the bridge's.
    alloc: VulkanAllocator,
    /// Cell extent. World cell `(i,j,k)` covers world AABB
    /// `[origin + (i,j,k)*cell_size, origin + (i+1,j+1,k+1)*cell_size]`.
    cell_size: f32,
    /// World position of cell `(0,0,0)` — the lower corner of the
    /// grid in world space.
    world_origin: Vec3,
    /// Cached `grid_dim` for the discretisation math; pulled out of
    /// `cpu_grid` so the per-cell lookup avoids a fn call.
    grid_dim: u32,
}

impl VoxelBridge {
    /// Allocate the persistent CPU grid + GPU texture.
    ///
    /// `grid_dim` is the cell count along each axis (so the grid
    /// holds `grid_dim³` cells); `world_extent` is the world-space
    /// length the grid covers along each axis (so each cell is
    /// `world_extent / grid_dim` units). The grid is centred on
    /// world origin: cell `(0,0,0)` lands at world position
    /// `(-world_extent/2, -world_extent/2, -world_extent/2)`.
    ///
    /// For wave_defense, `grid_dim = 128`, `world_extent = 128.0`
    /// (so 1 unit per cell, ±64 from origin). Settlers spawn at
    /// radius 8, spawners at radius 60, monsters spawn at spawners
    /// and march to origin — the whole sim fits inside ±64.
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
            gpu_tex,
            alloc,
            cell_size,
            world_origin,
            grid_dim,
        })
    }

    /// Discretise a world-space position to a grid cell. Returns
    /// `None` when the position falls outside the grid (an agent at
    /// world coords outside ±world_extent/2 — shouldn't happen for
    /// wave_defense, but the bridge stays robust to it).
    fn world_to_cell(&self, pos: Vec3) -> Option<(u32, u32, u32)> {
        let local = (pos - self.world_origin) / self.cell_size;
        let (x, y, z) = (
            local.x.floor() as i32,
            local.y.floor() as i32,
            local.z.floor() as i32,
        );
        if x < 0 || y < 0 || z < 0 {
            return None;
        }
        let (x, y, z) = (x as u32, y as u32, z as u32);
        if x >= self.grid_dim || y >= self.grid_dim || z >= self.grid_dim {
            return None;
        }
        Some((x, y, z))
    }

    /// Per-tick refresh from the latest [`ViewerApp`] snapshot. Clears
    /// last frame's cells then writes one cell per alive agent at its
    /// discretised position. Material id comes from the agent's
    /// creature_type via [`ViewerApp::material_for`].
    ///
    /// If two agents discretise to the same cell, last-writer-wins —
    /// at 1-unit cell size and ~360 agents spread across the world
    /// this is rare; the visual artefact is "one of two
    /// almost-touching agents disappears for a tick", acceptable for
    /// the pilot.
    pub fn refresh(&mut self, ctx: &VulkanContext, app: &ViewerApp) -> Result<()> {
        // Clear last frame's painted cells.
        for &(x, y, z) in &self.last_frame_cells {
            self.cpu_grid.set(x, y, z, 0);
        }
        self.last_frame_cells.clear();

        // Paint this frame.
        let positions = app.positions();
        let alive = app.alive();
        let creature_types = app.creature_types();
        for slot in 0..positions.len() {
            if alive[slot] == 0 {
                continue;
            }
            let Some((x, y, z)) = self.world_to_cell(positions[slot]) else {
                continue;
            };
            let material = app.material_for(creature_types[slot]);
            self.cpu_grid.set(x, y, z, material);
            self.last_frame_cells.push((x, y, z));
        }

        // Re-upload the whole grid. At 128³ × 1B = 2 MB this is
        // negligible per-tick bandwidth; per-affected-region upload
        // is a future optimisation worth doing if 256³ ever becomes
        // the default.
        self.gpu_tex.update_subregion(
            ctx,
            &mut self.alloc,
            &self.cpu_grid,
            (0, 0, 0),
            (self.grid_dim, self.grid_dim, self.grid_dim),
        )?;
        Ok(())
    }

    /// Build the renderer object tuple for this bridge's texture.
    /// World-space placement: lower-corner at `world_origin`, full
    /// extent `world_extent` along each axis.
    pub fn render_object(
        &self,
    ) -> (
        &GpuVoxelTexture,
        [f32; 4], // palette tint (renderer multiplies; 1.0s = no tint)
        [f32; 3], // world-space lower corner
        [f32; 3], // world-space dims
    ) {
        let world_extent = self.grid_dim as f32 * self.cell_size;
        (
            &self.gpu_tex,
            [1.0, 1.0, 1.0, 1.0],
            self.world_origin.into(),
            [world_extent; 3],
        )
    }

    /// Drop the GPU texture explicitly. winit's run_app doesn't drop
    /// the app on exit (it returns from `run_app` then process exits)
    /// so the texture would leak if we relied on `Drop`. Call this
    /// before `event_loop.exit()` returns.
    pub fn destroy(self, ctx: &VulkanContext) {
        let mut alloc = self.alloc;
        self.gpu_tex.destroy(ctx, &mut alloc);
    }
}
