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
        let mut cpu_grid = VoxelGrid::new(grid_dim, grid_dim, grid_dim);

        // Paint the ground plane once. World y=0 lands at cell
        // y=`grid_dim/2` after the centred origin offset; one cell
        // thick covers world y ∈ [0, cell_size). Cells at this y
        // for every (x, z) get the GROUND_MATERIAL id. Painted
        // before any agent splat, then never cleared — the per-tick
        // refresh's `last_frame_cells` only tracks agent paints, so
        // the ground stays put.
        let ground_y = grid_dim / 2;
        for x in 0..grid_dim {
            for z in 0..grid_dim {
                cpu_grid.set(x, ground_y, z, crate::GROUND_MATERIAL);
            }
        }

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
        // Sparse clear of last frame's painted cells. Some of these
        // may be on the ground plane (an agent splat that overlapped
        // y=ground_y). The re-paint loop right after restores the
        // ground layer, so the visual effect is "agents leave footprints
        // → ground heals next tick".
        for &(x, y, z) in &self.last_frame_cells {
            self.cpu_grid.set(x, y, z, 0);
        }
        self.last_frame_cells.clear();

        // Re-paint the ground layer. Cheap (grid_dim² writes; already
        // dominated by the per-tick GPU re-upload of the full grid).
        let ground_y = self.grid_dim / 2;
        for x in 0..self.grid_dim {
            for z in 0..self.grid_dim {
                self.cpu_grid.set(x, ground_y, z, crate::GROUND_MATERIAL);
            }
        }

        // Paint this frame: each agent's discretised position gets
        // a `AGENT_SPLAT_DIM³` block of cells with material id ==
        // app.materials()[slot]. Bigger splat than a single cell
        // gives voxel_engine's mip-skip + DDA enough surface that
        // an entity is reliably hit (single-cell entities slip
        // through mip3 jump-skip in many ray directions).
        //
        // Multiple agents discretising to overlapping splats are
        // last-writer-wins. We paint a stationary objective
        // marker LAST so it's never obscured by an agent that
        // happens to stand on top of it.
        const AGENT_SPLAT_DIM: i32 = 2;
        let positions = app.positions();
        let alive = app.alive();
        let materials = app.materials();
        for slot in 0..positions.len() {
            if alive[slot] == 0 {
                continue;
            }
            self.splat_at(positions[slot], materials[slot], AGENT_SPLAT_DIM);
        }
        // Objective marker — painted after all agents so it stays
        // visible even when an agent stands on top of it.
        self.splat_at(
            crate::objective_world_position(),
            crate::objective_material(),
            AGENT_SPLAT_DIM + 1,
        );

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

    /// Paint a `dim³` block of cells centred on `pos` with the
    /// given material id. Out-of-bounds cells are skipped silently.
    fn splat_at(&mut self, pos: Vec3, material: u8, dim: i32) {
        let Some((cx, cy, cz)) = world_to_cell(
            pos,
            self.world_origin,
            self.cell_size,
            self.grid_dim,
        ) else {
            return;
        };
        let half = dim / 2;
        for dx in 0..dim {
            for dy in 0..dim {
                for dz in 0..dim {
                    let x = cx as i32 + dx - half;
                    let y = cy as i32 + dy - half;
                    let z = cz as i32 + dz - half;
                    if x < 0
                        || y < 0
                        || z < 0
                        || x >= self.grid_dim as i32
                        || y >= self.grid_dim as i32
                        || z >= self.grid_dim as i32
                    {
                        continue;
                    }
                    let (x, y, z) = (x as u32, y as u32, z as u32);
                    self.cpu_grid.set(x, y, z, material);
                    self.last_frame_cells.push((x, y, z));
                }
            }
        }
    }

    /// Build the renderer object tuple for this bridge's texture.
    /// Single object — per-cell colours come from `palette_tex`
    /// inside the fragment shader, not this tuple's `palette_color`.
    ///
    /// `palette_color.a` doubles as a debug-mode selector in the
    /// shader (gbuffer.frag:114-118): values > 3.5 enable
    /// normal_debug — white = camera-facing surface hit, hot pink
    /// = back-face hit. Toggle via `VIEWER_DEBUG_NORMALS=1` env
    /// var to triage the "no forward faces" issue: white-only
    /// means the lighting is dim but front faces are being hit
    /// correctly; pink-only means the DDA is consistently
    /// back-face hitting and the upstream shader needs fixing.
    pub fn render_object(
        &self,
    ) -> Option<(
        &GpuVoxelTexture,
        [f32; 4], // RGBA — .a feeds roughness OR debug-mode selector
        [f32; 3], // world-space lower corner
        [f32; 3], // world-space dims
    )> {
        let alpha = if std::env::var("VIEWER_DEBUG_NORMALS").is_ok() {
            4.0 // > 3.5 → normal_debug mode in gbuffer.frag
        } else {
            0.5 // medium roughness for matte appearance
        };
        let world_extent = self.grid_dim as f32 * self.cell_size;
        let dims = [world_extent; 3];
        let pos: [f32; 3] = self.world_origin.into();
        self.gpu_tex
            .as_ref()
            .map(|t| (t, [1.0, 1.0, 1.0, alpha], pos, dims))
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
