//! `EngineBridge` — a descriptor-driven flat-arena voxel renderer. A
//! generalized port of `viewer_runtime::vs::VsBridge`: instead of hard-coded
//! mana-band roles + weapon VFX, it is driven by a [`RenderDescriptor`] and a
//! per-frame `&[AgentView]` snapshot.
//!
//! - Arena floor sized from `arena_radius` (origin-centered).
//! - A palette assigning one material index per `AgentVisual.color` and per
//!   `VfxSpec.color`.
//! - `refresh` clears last frame's sparse cells, paints each alive agent the
//!   first `AgentVisual` whose `when` field-range contains the agent's column
//!   value, then paints VFX: `Ring` around the camera-followed agent on each
//!   `tick % period == 0`, `BeamToNearest` to the nearest agent matching the
//!   target range.
//!
//! The CPU grid-painting (the descriptor logic) is factored into
//! [`EngineBridge::paint`] which writes into any [`PaintGrid`], so it is
//! unit-testable headlessly. The GPU upload sits behind the same seam as the
//! original `VsBridge` (`refresh` re-uploads the texture after painting).

use engine_play_api::{
    AgentView, CameraSpec, FieldRange, RenderDescriptor, VfxKind, VfxSpec,
};

/// Reserved palette slot for the arena floor (grey). Agent + VFX colors are
/// assigned slots starting at [`PAL_DYNAMIC_START`].
pub const PAL_AIR: u8 = 0;
pub const PAL_FLOOR: u8 = 1;
/// First palette index assigned to descriptor colors (agents then VFX).
pub const PAL_DYNAMIC_START: u8 = 8;

/// Vertical (renderer Y-up) cell the floor occupies; agents/VFX splat one cell
/// above so they sit visibly on top.
const FLOOR_VERT: u32 = 0;
const SPLAT_VERT: u32 = 1;

/// Read a named column off an `AgentView`. Unknown names → 0.0 (matches the
/// contract's "missing columns default to 0").
pub fn view_column(a: &AgentView, field: &str) -> f32 {
    match field {
        "hp" => a.hp,
        "mana" => a.mana,
        "move_speed" => a.move_speed,
        "creature_type" => a.creature_type as f32,
        "x" => a.pos[0],
        "y" => a.pos[1],
        "z" => a.pos[2],
        _ => 0.0,
    }
}

/// `[lo, hi)` membership test for a `FieldRange` against an agent column.
pub fn in_range(a: &AgentView, r: &FieldRange) -> bool {
    let v = view_column(a, &r.field);
    v >= r.lo && v < r.hi
}

/// A sink for painted cells. Implemented by the real `VoxelGrid` (GPU path) and
/// by a recording grid in tests, so the descriptor painting logic is shared and
/// headlessly verifiable.
pub trait PaintGrid {
    /// Set cell `(x, y, z)` to material `mat`.
    fn set(&mut self, x: u32, y: u32, z: u32, mat: u8);
}

/// A test/headless `PaintGrid` that records the final material at each touched
/// cell (last-writer-wins, matching `VoxelGrid::set`).
#[derive(Default, Clone, Debug)]
pub struct Painted {
    pub cells: std::collections::HashMap<(u32, u32, u32), u8>,
}

impl Painted {
    pub fn new() -> Self {
        Self::default()
    }
    /// Material at a cell, if painted.
    pub fn at(&self, x: u32, y: u32, z: u32) -> Option<u8> {
        self.cells.get(&(x, y, z)).copied()
    }
    /// Count of cells currently set to `mat`.
    pub fn count_of(&self, mat: u8) -> usize {
        self.cells.values().filter(|&&m| m == mat).count()
    }
}

impl PaintGrid for Painted {
    fn set(&mut self, x: u32, y: u32, z: u32, mat: u8) {
        self.cells.insert((x, y, z), mat);
    }
}

/// Grid geometry derived from `arena_radius`. The grid is origin-centered: a
/// world position is shifted by `+half` on each planar axis to land in
/// `[0, dim)`. The sim's XY plane maps to renderer XZ (floor); vertical is Y-up.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Planar grid size (cells) on both X and Z.
    pub dim: u32,
    /// Vertical headroom (cells); floor at 0, splats at 1.
    pub dim_y: u32,
    /// World→grid offset (= dim/2).
    pub half: f32,
}

impl Geometry {
    /// Size a grid that comfortably contains a `[-radius, +radius]` arena with a
    /// margin so edge spawns map in-grid. Rounded up to an even cell count.
    pub fn from_radius(arena_radius: f32) -> Self {
        let margin = 8.0;
        let span = (arena_radius.max(1.0) + margin) * 2.0;
        let dim = (span.ceil() as u32).max(2);
        let dim = dim + (dim & 1); // even, so origin sits on a cell boundary
        Self {
            dim,
            dim_y: 4,
            half: dim as f32 / 2.0,
        }
    }

    /// World (origin-centered, XY plane) → voxel cell at `vert`. `None` if the
    /// planar cell falls outside the grid.
    pub fn world_to_voxel(&self, pos: [f32; 3], vert: u32) -> Option<(u32, u32, u32)> {
        let cx = pos[0] + self.half;
        let cz = pos[1] + self.half;
        if cx < 0.0 || cz < 0.0 {
            return None;
        }
        let x = cx.floor() as u32;
        let z = cz.floor() as u32;
        if x >= self.dim || z >= self.dim || vert >= self.dim_y {
            return None;
        }
        Some((x, vert, z))
    }
}

/// The palette: a material index per agent visual + per VFX, all RGB. Slot
/// assignment is deterministic (agents in descriptor order, then VFX).
pub struct Palette {
    /// Material index for `agents[i]`.
    agent_mat: Vec<u8>,
    /// Material index for `vfx[i]`.
    vfx_mat: Vec<u8>,
    /// Full RGBA palette for GPU upload.
    rgba: [[u8; 4]; 256],
}

impl Palette {
    fn build(desc: &RenderDescriptor) -> Self {
        // voxel_engine's MaterialPalette → RGBA, mirroring VsBridge.
        use voxel_engine::voxel::material::{MaterialPalette, MaterialType, PaletteEntry};
        fn entry(c: [u8; 3]) -> PaletteEntry {
            PaletteEntry {
                r: c[0],
                g: c[1],
                b: c[2],
                roughness: 200,
                emissive: 0,
                material_type: MaterialType::Stone,
            }
        }
        let mut p = MaterialPalette::new();
        p.set(PAL_FLOOR, entry([110, 110, 110]));

        let mut next = PAL_DYNAMIC_START;
        let mut agent_mat = Vec::with_capacity(desc.agents.len());
        for a in &desc.agents {
            p.set(next, entry(a.color));
            agent_mat.push(next);
            next = next.wrapping_add(1);
        }
        let mut vfx_mat = Vec::with_capacity(desc.vfx.len());
        for v in &desc.vfx {
            p.set(next, entry(v.color));
            vfx_mat.push(next);
            next = next.wrapping_add(1);
        }
        Self {
            agent_mat,
            vfx_mat,
            rgba: p.to_rgba(),
        }
    }
}

/// Descriptor-driven voxel renderer. Holds the parsed descriptor, derived grid
/// geometry, and the palette. The CPU painting is in [`EngineBridge::paint`];
/// the GPU resources live in [`bridge_gpu::GpuResources`] (compiled only when
/// the bridge is constructed with a `VulkanContext`).
pub struct EngineBridge {
    desc: RenderDescriptor,
    geom: Geometry,
    palette: Palette,
    gpu: Option<bridge_gpu::GpuResources>,
}

impl EngineBridge {
    /// Construct a headless bridge (no GPU): geometry + palette only. Used by
    /// tests and by the headless `update()` path.
    pub fn new_headless(desc: RenderDescriptor) -> Self {
        let geom = Geometry::from_radius(desc.arena_radius);
        let palette = Palette::build(&desc);
        Self {
            desc,
            geom,
            palette,
            gpu: None,
        }
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }
    pub fn descriptor(&self) -> &RenderDescriptor {
        &self.desc
    }

    /// The agent the camera follows: the first agent in the `CameraSpec::Follow`
    /// range (Observer → none). Used for camera centering + Ring VFX origin.
    pub fn followed<'a>(&self, agents: &'a [AgentView]) -> Option<&'a AgentView> {
        match &self.desc.camera {
            CameraSpec::Follow(range) => {
                agents.iter().find(|a| a.alive && in_range(a, range))
            }
            CameraSpec::Observer => None,
        }
    }

    /// Pick the material for an agent: the first `AgentVisual` whose `when`
    /// range contains the agent's column value. `None` if no visual matches.
    fn agent_material(&self, a: &AgentView) -> Option<u8> {
        self.desc
            .agents
            .iter()
            .position(|v| in_range(a, &v.when))
            .map(|i| self.palette.agent_mat[i])
    }

    /// Paint one frame into `grid` (CPU-only, no GPU): clears `last_cells` from
    /// the previous frame, paints alive agents, then VFX. Returns the cells
    /// painted this frame (so the caller can clear them next frame). The floor
    /// is painted once at construction by the GPU path; here `paint` only
    /// touches the sparse agent/VFX layer at `SPLAT_VERT`.
    pub fn paint(
        &self,
        grid: &mut dyn PaintGrid,
        agents: &[AgentView],
        tick: u64,
        last_cells: &[(u32, u32, u32)],
    ) -> Vec<(u32, u32, u32)> {
        // Clear previous frame's sparse cells back to floor/air.
        for &(x, vert, z) in last_cells {
            let restore = if vert == FLOOR_VERT { PAL_FLOOR } else { PAL_AIR };
            grid.set(x, vert, z, restore);
        }

        let mut painted: Vec<(u32, u32, u32)> = Vec::new();

        // Agents: one splat per alive agent at the first matching visual.
        for a in agents.iter().filter(|a| a.alive) {
            let Some(mat) = self.agent_material(a) else {
                continue;
            };
            if let Some(cell) = self.geom.world_to_voxel(a.pos, SPLAT_VERT) {
                grid.set(cell.0, cell.1, cell.2, mat);
                painted.push(cell);
            }
        }

        // VFX, keyed off the followed agent (the player).
        if let Some(follow) = self.followed(agents) {
            let origin = follow.pos;
            for (i, vfx) in self.desc.vfx.iter().enumerate() {
                let mat = self.palette.vfx_mat[i];
                self.paint_vfx(grid, vfx, mat, origin, agents, tick, &mut painted);
            }
        }

        painted
    }

    /// Paint a single VFX spec if its period fires this tick.
    fn paint_vfx(
        &self,
        grid: &mut dyn PaintGrid,
        vfx: &VfxSpec,
        mat: u8,
        origin: [f32; 3],
        agents: &[AgentView],
        tick: u64,
        painted: &mut Vec<(u32, u32, u32)>,
    ) {
        let period = vfx.period.max(1) as u64;
        let fires = tick % period == 0;
        match &vfx.kind {
            VfxKind::Ring => {
                if fires {
                    self.paint_ring(grid, origin, vfx.radius, 36, mat, painted);
                }
            }
            VfxKind::BeamToNearest { target } => {
                if !fires {
                    return;
                }
                // Nearest agent matching `target` (planar distance).
                let mut best: Option<(f32, [f32; 3])> = None;
                for e in agents.iter().filter(|a| a.alive && in_range(a, target)) {
                    let (dx, dy) = (e.pos[0] - origin[0], e.pos[1] - origin[1]);
                    let d2 = dx * dx + dy * dy;
                    if best.map_or(true, |(bd, _)| d2 < bd) {
                        best = Some((d2, e.pos));
                    }
                }
                if let Some((_, epos)) = best {
                    self.paint_beam(grid, origin, epos, mat, painted);
                }
            }
        }
    }

    /// Paint a flat ring of `mat` voxels at `radius` (world units) around
    /// `center`, at `SPLAT_VERT`.
    fn paint_ring(
        &self,
        grid: &mut dyn PaintGrid,
        center: [f32; 3],
        radius: f32,
        segments: u32,
        mat: u8,
        painted: &mut Vec<(u32, u32, u32)>,
    ) {
        for i in 0..segments {
            let ang = std::f32::consts::TAU * (i as f32) / (segments as f32);
            let wx = center[0] + radius * ang.cos();
            let wy = center[1] + radius * ang.sin();
            if let Some(cell) = self.geom.world_to_voxel([wx, wy, 0.0], SPLAT_VERT) {
                grid.set(cell.0, cell.1, cell.2, mat);
                painted.push(cell);
            }
        }
    }

    /// Paint a straight beam of `mat` voxels from `from` to `to` at `SPLAT_VERT`.
    fn paint_beam(
        &self,
        grid: &mut dyn PaintGrid,
        from: [f32; 3],
        to: [f32; 3],
        mat: u8,
        painted: &mut Vec<(u32, u32, u32)>,
    ) {
        let steps = 48;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let wx = from[0] + (to[0] - from[0]) * t;
            let wy = from[1] + (to[1] - from[1]) * t;
            if let Some(cell) = self.geom.world_to_voxel([wx, wy, 0.0], SPLAT_VERT) {
                grid.set(cell.0, cell.1, cell.2, mat);
                painted.push(cell);
            }
        }
    }
}

/// GPU-backed half of the bridge: the CPU `VoxelGrid` + uploaded texture +
/// allocator. Split into a submodule so the descriptor logic above stays
/// GPU-free and testable. The methods here mirror `VsBridge`'s GPU seam
/// (`new`, per-frame `refresh` re-upload, `render_object`, `destroy`).
pub mod bridge_gpu {
    use super::*;
    use anyhow::Result;
    use voxel_engine::vulkan::allocator::VulkanAllocator;
    use voxel_engine::vulkan::instance::VulkanContext;
    use voxel_engine::vulkan::voxel_gpu::{upload_grid_to_gpu, GpuVoxelTexture};
    use voxel_engine::voxel::grid::VoxelGrid;

    /// World-space cell size (1 voxel == 1 world unit), matching `VsBridge`.
    pub const CELL_SIZE: f32 = 1.0;

    /// The real GPU grid is a valid paint sink, so `EngineBridge::paint`'s
    /// descriptor logic is shared verbatim between the headless and GPU paths.
    impl PaintGrid for VoxelGrid {
        fn set(&mut self, x: u32, y: u32, z: u32, mat: u8) {
            VoxelGrid::set(self, x, y, z, mat);
        }
    }

    /// GPU resources owned alongside an `EngineBridge`.
    pub struct GpuResources {
        cpu_grid: VoxelGrid,
        last_cells: Vec<(u32, u32, u32)>,
        gpu_tex: Option<GpuVoxelTexture>,
        alloc: VulkanAllocator,
    }

    impl EngineBridge {
        /// Construct a GPU-backed bridge: build the flat floor, upload it.
        pub fn new(ctx: &VulkanContext, desc: RenderDescriptor) -> Result<Self> {
            let mut bridge = EngineBridge::new_headless(desc);
            let g = bridge.geom;
            let mut cpu_grid = VoxelGrid::new(g.dim, g.dim_y, g.dim);
            for x in 0..g.dim {
                for z in 0..g.dim {
                    cpu_grid.set(x, FLOOR_VERT, z, PAL_FLOOR);
                }
            }
            let mut alloc = VulkanAllocator::new(ctx)?;
            let gpu_tex = upload_grid_to_gpu(ctx, &mut alloc, &cpu_grid, &bridge.palette.rgba)?;
            bridge.gpu = Some(GpuResources {
                cpu_grid,
                last_cells: Vec::with_capacity(512),
                gpu_tex: Some(gpu_tex),
                alloc,
            });
            Ok(bridge)
        }

        /// Per-frame GPU refresh: paint the sparse layer via the shared
        /// [`EngineBridge::paint`], then re-upload the texture (mirrors
        /// `VsBridge::refresh`'s destroy-and-reupload).
        pub fn refresh(
            &mut self,
            ctx: &VulkanContext,
            agents: &[AgentView],
            tick: u64,
        ) -> Result<()> {
            // Take the GPU state out so `&self` painting can borrow the rest.
            let mut gpu = self.gpu.take().expect("refresh called on headless bridge");
            let last = std::mem::take(&mut gpu.last_cells);
            let painted = self.paint(&mut gpu.cpu_grid, agents, tick, &last);
            gpu.last_cells = painted;

            if let Some(old) = gpu.gpu_tex.take() {
                old.destroy(ctx, &mut gpu.alloc);
            }
            let new_tex =
                upload_grid_to_gpu(ctx, &mut gpu.alloc, &gpu.cpu_grid, &self.palette.rgba)?;
            gpu.gpu_tex = Some(new_tex);
            self.gpu = Some(gpu);
            Ok(())
        }

        /// Single render-object tuple for `VoxelRenderer::render_frame_gpu`.
        pub fn render_object(
            &self,
        ) -> Option<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])> {
            let g = self.geom;
            let dims = [
                g.dim as f32 * CELL_SIZE,
                g.dim_y as f32 * CELL_SIZE,
                g.dim as f32 * CELL_SIZE,
            ];
            let pos = [0.0f32, 0.0, 0.0];
            self.gpu
                .as_ref()
                .and_then(|r| r.gpu_tex.as_ref())
                .map(|t| (t, [1.0, 1.0, 1.0, 0.5], pos, dims))
        }

        /// Explicitly drop GPU resources (winit's run_app doesn't drop on exit).
        pub fn destroy(&mut self, ctx: &VulkanContext) {
            if let Some(mut gpu) = self.gpu.take() {
                if let Some(tex) = gpu.gpu_tex.take() {
                    tex.destroy(ctx, &mut gpu.alloc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MOCK_RENDER;

    fn sample_desc() -> RenderDescriptor {
        RenderDescriptor::from_json(MOCK_RENDER).unwrap()
    }

    #[test]
    fn geometry_centers_origin() {
        let g = Geometry::from_radius(42.0);
        // origin maps to the central cell.
        let (x, _, z) = g.world_to_voxel([0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!((x, z), (g.dim / 2, g.dim / 2));
        // far outside is clipped.
        assert_eq!(g.world_to_voxel([1000.0, 0.0, 0.0], 1), None);
    }

    #[test]
    fn bridge_paints_from_descriptor() {
        let bridge = EngineBridge::new_headless(sample_desc());
        let g = bridge.geometry();

        // A player (mana 1.0 → cyan band) at origin + an enemy (mana 2.0).
        let player = AgentView {
            pos: [0.0, 0.0, 0.0],
            alive: true,
            hp: 100.0,
            mana: 1.0,
            move_speed: 0.4,
            creature_type: 0,
        };
        let enemy = AgentView {
            pos: [8.0, 0.0, 0.0],
            alive: true,
            hp: 30.0,
            mana: 2.0,
            move_speed: 0.5,
            creature_type: 1,
        };
        let agents = [player, enemy];

        // Tick 1: no nova ring (period 40), no beam (period 12) — agents only.
        let mut grid = Painted::new();
        let painted = bridge.paint(&mut grid, &agents, 1, &[]);
        let player_cell = g.world_to_voxel(player.pos, SPLAT_VERT).unwrap();
        // Player got the cyan (first agent visual = palette slot PAL_DYNAMIC_START).
        assert_eq!(grid.at(player_cell.0, player_cell.1, player_cell.2), Some(PAL_DYNAMIC_START));
        // Enemy got the second agent visual.
        let enemy_cell = g.world_to_voxel(enemy.pos, SPLAT_VERT).unwrap();
        assert_eq!(
            grid.at(enemy_cell.0, enemy_cell.1, enemy_cell.2),
            Some(PAL_DYNAMIC_START + 1)
        );
        assert!(painted.contains(&player_cell));

        // Tick 40: nova Ring fires around the player. The ring material is the
        // first VFX slot (after the 2 agent slots).
        let nova_mat = PAL_DYNAMIC_START + 2;
        let mut grid40 = Painted::new();
        let _ = bridge.paint(&mut grid40, &agents, 40, &[]);
        assert!(
            grid40.count_of(nova_mat) > 0,
            "expected a nova ring at tick 40; painted materials: {:?}",
            grid40.cells.values().collect::<std::collections::HashSet<_>>()
        );

        // Tick 12: bolt BeamToNearest fires player→enemy (beam material = 2nd VFX slot).
        let bolt_mat = PAL_DYNAMIC_START + 3;
        let mut grid12 = Painted::new();
        let _ = bridge.paint(&mut grid12, &agents, 12, &[]);
        assert!(
            grid12.count_of(bolt_mat) > 0,
            "expected a bolt beam at tick 12"
        );
    }

    #[test]
    fn paint_clears_last_frame_cells() {
        let bridge = EngineBridge::new_headless(sample_desc());
        let mut grid = Painted::new();
        // Pretend a stale cell from last frame at a splat position.
        let stale = (5, SPLAT_VERT, 5);
        grid.set(stale.0, stale.1, stale.2, 99);
        let _ = bridge.paint(&mut grid, &[], 1, &[stale]);
        // Cleared back to air (splat vert).
        assert_eq!(grid.at(stale.0, stale.1, stale.2), Some(PAL_AIR));
    }

    #[test]
    fn followed_picks_player_band() {
        let bridge = EngineBridge::new_headless(sample_desc());
        let player = AgentView {
            pos: [1.0, 2.0, 0.0],
            alive: true,
            hp: 100.0,
            mana: 1.0,
            move_speed: 0.4,
            creature_type: 0,
        };
        let enemy = AgentView { mana: 2.0, ..player };
        let agents = [enemy, player];
        let f = bridge.followed(&agents).unwrap();
        assert_eq!(f.mana, 1.0);
    }
}
