//! `voxel_bridge` — Phase B of the viewer-runtime plan.
//!
//! Bridges [`crate::ViewerApp`]'s per-tick agent snapshot into a
//! single world-spanning [`GpuVoxelTexture`] that
//! [`voxel_engine::render::VoxelRenderer::render_frame_gpu`] consumes.
//!
//! Architecture:
//! - **Per creature_type**: one CPU [`VoxelGrid`] + one persistent
//!   GPU [`GpuVoxelTexture`]. Voxel_engine's renderer uses a single
//!   `palette_color` push-constant per object (per-cell material
//!   indices in the texture are treated as binary occupancy, not
//!   indexed colors), so we get multi-color rendering by passing
//!   N objects each with cells of one type and that type's color.
//! - Per-tick refresh is **sparse**: track which cells we set last
//!   frame in EACH per-type grid, clear them + paint this frame.
//!   At ~360 alive agents in wave_defense the touched-cell count
//!   stays well below the grid total per-type.
//! - The GPU upload happens over the full grid extent per-type
//!   (one `update_subregion` covering each whole grid). At 128³ × 1B
//!   = 2 MB per type × 4 types = 8 MB per-tick bandwidth, trivial
//!   for any post-2010 GPU.
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

/// One per-creature-type grid: CPU + GPU + the cells written last
/// frame so the per-tick refresh can be sparse.
struct TypeGrid {
    cpu: VoxelGrid,
    gpu: GpuVoxelTexture,
    last_frame_cells: Vec<(u32, u32, u32)>,
    /// World-space colour for `render_object` (RGBA, 0..1).
    color: [f32; 4],
}

/// Per-creature-type world-grid bridge for Phase B.
///
/// Owns N GPU textures + the shared allocator; all must outlive
/// every `render_frame_gpu` call that references the textures.
/// Call [`Self::destroy`] before dropping to release VRAM cleanly.
pub struct VoxelBridge {
    /// One grid per creature_type the renderer should colour
    /// distinctly. Index = creature_type ordinal (1..=4 today;
    /// slot 0 is unused so indexing matches the .sim's ordinals).
    grids: Vec<TypeGrid>,
    /// Allocator scoped to this bridge — every per-type texture
    /// allocates against it. Single allocator keeps the per-frame
    /// fragmentation low.
    alloc: VulkanAllocator,
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
        let mut alloc = VulkanAllocator::new(ctx)?;
        let palette_rgba = palette.to_rgba();

        // One grid per creature_type material id (1..=4 for
        // node/settler/monster/spawner). Slot 0 holds a placeholder
        // so indexing by material id is direct. The placeholder
        // never gets cells set or rendered.
        let mut grids: Vec<TypeGrid> = Vec::with_capacity(5);
        // Slot 0 placeholder
        let placeholder_cpu = VoxelGrid::new(1, 1, 1);
        let placeholder_gpu =
            upload_grid_to_gpu(ctx, &mut alloc, &placeholder_cpu, &palette_rgba)?;
        grids.push(TypeGrid {
            cpu: placeholder_cpu,
            gpu: placeholder_gpu,
            last_frame_cells: Vec::new(),
            color: [0.0, 0.0, 0.0, 0.0],
        });
        for material_id in 1..=4u8 {
            let cpu = VoxelGrid::new(grid_dim, grid_dim, grid_dim);
            let gpu = upload_grid_to_gpu(ctx, &mut alloc, &cpu, &palette_rgba)?;
            let entry = palette.get(material_id);
            let color = [
                entry.r as f32 / 255.0,
                entry.g as f32 / 255.0,
                entry.b as f32 / 255.0,
                1.0,
            ];
            grids.push(TypeGrid {
                cpu,
                gpu,
                last_frame_cells: Vec::with_capacity(512),
                color,
            });
        }

        Ok(Self {
            grids,
            alloc,
            cell_size,
            world_origin,
            grid_dim,
        })
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
        // Clear last frame across all per-type grids.
        for grid in self.grids.iter_mut() {
            for &(x, y, z) in &grid.last_frame_cells {
                grid.cpu.set(x, y, z, 0);
            }
            grid.last_frame_cells.clear();
        }

        // Paint this frame: each agent goes into the grid for its
        // creature_type (= material id under our 1:1 mapping in
        // `creature_material_index`).
        let positions = app.positions();
        let alive = app.alive();
        let creature_types = app.creature_types();
        for slot in 0..positions.len() {
            if alive[slot] == 0 {
                continue;
            }
            let material = app.material_for(creature_types[slot]);
            let Some(grid) = self.grids.get_mut(material as usize) else {
                continue; // unknown material id (should be rare)
            };
            let Some((x, y, z)) = world_to_cell(
                positions[slot],
                self.world_origin,
                self.cell_size,
                self.grid_dim,
            ) else {
                continue;
            };
            grid.cpu.set(x, y, z, 1);
            grid.last_frame_cells.push((x, y, z));
        }

        // Re-upload each per-type grid. Skip slot 0 (placeholder).
        for grid in self.grids.iter_mut().skip(1) {
            grid.gpu.update_subregion(
                ctx,
                &mut self.alloc,
                &grid.cpu,
                (0, 0, 0),
                (self.grid_dim, self.grid_dim, self.grid_dim),
            )?;
        }
        Ok(())
    }

    /// Build the renderer objects array — one tuple per per-type
    /// grid (skipping the placeholder slot 0). Each object covers
    /// the same world bounds; the renderer raymarches each
    /// independently.
    pub fn render_objects(
        &self,
    ) -> Vec<(
        &GpuVoxelTexture,
        [f32; 4], // per-object colour
        [f32; 3], // world-space lower corner
        [f32; 3], // world-space dims
    )> {
        let world_extent = self.grid_dim as f32 * self.cell_size;
        let dims = [world_extent; 3];
        let pos: [f32; 3] = self.world_origin.into();
        self.grids
            .iter()
            .skip(1)
            .map(|g| (&g.gpu, g.color, pos, dims))
            .collect()
    }

    /// Drop the GPU textures explicitly. winit's run_app doesn't
    /// drop the app on exit (it returns from `run_app` then process
    /// exits) so the textures would leak if we relied on `Drop`.
    /// Call this before `event_loop.exit()` returns.
    pub fn destroy(self, ctx: &VulkanContext) {
        let mut alloc = self.alloc;
        for grid in self.grids.into_iter() {
            grid.gpu.destroy(ctx, &mut alloc);
        }
    }
}

/// Discretise a world-space position to a grid cell. Returns `None`
/// when the position falls outside the grid (an agent at world coords
/// outside `±world_extent/2` — shouldn't happen for wave_defense, but
/// the bridge stays robust to it). Free function so the per-type-grid
/// loop can call it without re-borrowing `self`.
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
