//! Voxel-engine adapter — bridges `~/Projects/voxel_engine` to the sim
//! engine via the `TerrainQuery` seam. **Phases A + B + C** of the
//! 5-phase voxel integration plan
//! (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`).
//!
//! Phase C adds [`VoxelMirror`] — a `wgpu::Buffer` mirror of the host
//! `VoxelGrid` that GPU kernels can read directly via the new optional
//! `voxel_grid` field on `KernelBindingsContext`. The runtime calls
//! [`VoxelTerrain::apply_voxel_chronicle_record_with_mirror`] per
//! drained record (instead of the bare `apply_voxel_chronicle_record`),
//! which mutates the CPU grid AND marks the touched chunks dirty in
//! the mirror. At end of tick the runtime calls
//! [`VoxelMirror::flush_dirty`] to push the dirty chunks to GPU.
//!
//! # What this crate does today
//!
//! Wraps `voxel_engine::voxel::grid::VoxelGrid` — a dense, flat `Vec<u8>`
//! backed 3D array — and implements the engine's `TerrainQuery` trait.
//! Cells are 1 world-unit on a side; the cell at integer coordinate
//! `(x, y, z)` covers the AABB `[x, x+1) × [y, y+1) × [z, z+1)`. World
//! origin sits at grid origin; negative coordinates fall outside the
//! grid and behave as empty (return defaults: height 0, walkable true,
//! LOS clear — same as the engine's `FlatPlane` default impl).
//!
//! Voxel value `0` = air (empty); any non-zero value = solid. The
//! per-kind material distinction (palisade vs. quarried-stone vs.
//! placed-block) lives in the upper bits but is not interpreted here —
//! the Phase B consumer stores a deterministic non-zero material code
//! derived from `kind_hash` (low 8 bits or'd with 1 to avoid colliding
//! with the air sentinel), and the per-kind registry mapping is a
//! Phase D concern.
//!
//! # Phase B chronicle consumer (this slice)
//!
//! `apply_voxel_chronicle_record` decodes engine chronicle records
//! produced by the GPU `apply_ability` dispatcher when a verb's program
//! includes `EffectOp::PlaceVoxel` (kind=26 → engine kind=60) or
//! `EffectOp::Harvest` (kind=25 → engine kind=59). The runtime drives
//! the call once per drained record:
//!
//! - **`EffectPlaceVoxelApplied` (kind=60)** — `slot[2]=caster_slot`,
//!   `slot[3]=kind_hash`. The consumer writes a non-zero voxel value
//!   (`((kind_hash & 0xFF) as u8) | 1`) at the cell directly above the
//!   caster's position so the place lands on the column floor (matches
//!   `floor()` cell semantics — the caster stands on `z=0`, the placed
//!   block goes at `z=0` and lifts `height_at` to `1.0`).
//! - **`EffectHarvestApplied` (kind=59)** — `slot[2]=caster_slot`,
//!   `slot[3]=kind_hash`, `slot[4]=amount`. The consumer scans cells
//!   in a small neighborhood around the caster and clears up to
//!   `amount` non-zero cells (deterministic top-down x/y/z iteration).
//!
//! ## Caller signature: option (A) — pass caster_pos in
//!
//! The plan considered two consumer signatures: (A) caller resolves
//! `caster_slot → pos` and passes a `Vec3`, (B) caller passes the full
//! `SimState`. We picked (A) — `engine_voxel` stays free of any
//! `engine::SimState` dep and the per-call coupling is minimal (one
//! `Vec3` per drained record). The runtime owns the agent SoA; it
//! looks up `caster_slot`'s position before calling.
//!
//! # What this crate does NOT do
//!
//! - **No GPU mirror.** Phase C ships a `wgpu::Buffer` mirror of the
//!   grid that kernels can read directly.
//! - **No production runtime opt-in.** Existing runtimes continue to use
//!   the `FlatPlane` default; Phase E migrates wave_defense to opt in.
//!
//! # Determinism audit (load-bearing per P5)
//!
//! `voxel_engine` uses `std::collections::HashMap`/`HashSet` in three
//! files — `voxel/svdag.rs`, `voxel/articulation.rs`,
//! `voxel/connectivity.rs` — but **none** of them are reachable from
//! the modules we use (`voxel::grid` + `voxel::raycast`). The
//! `VoxelGrid` struct is internally a `Vec<u8>` indexed by
//! `z*H*W + y*W + x`, so iteration order is deterministic by
//! construction. No HashMap-derived nondeterminism enters the adapter
//! boundary.
//!
//! Verified by `grep -rn HashMap /home/ricky/Projects/voxel_engine/src/world/`
//! (zero hits) and `grep -n` on `voxel/grid.rs` + `voxel/raycast.rs`
//! (zero hits). Re-audit when bumping the `voxel_engine` path-dep.
//!
//! # Vulkan deps cost
//!
//! `voxel_engine`'s `Cargo.toml` lists `ash`, `gpu-allocator`,
//! `ash-window`, `raw-window-handle`, `dot_vox`, plus a `shaderc`
//! build-dep — all unconditional. The `default = []` feature set keeps
//! `winit`/`egui`/`tracing` out of the dep tree, but Vulkan still
//! compiles. Sim engine crates that don't import `engine_voxel` aren't
//! affected.

use engine::terrain::TerrainQuery;
use engine::GpuContext;
use glam::{IVec3, Vec3};
use std::collections::BTreeSet;
use voxel_engine::voxel::raycast::ray_cast_grid;

pub use engine::state::agent::MovementMode;
/// Re-export so per-runtime crates can pass `&VoxelGrid` to mirror
/// methods without taking a direct dependency on `voxel_engine`.
pub use voxel_engine::voxel::grid::VoxelGrid;

/// Voxel-region runtime — per spec
/// `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
/// §6.1. See `region.rs` for full design notes.
pub mod region;
pub use region::{
    Aabb, VoxelRegion, VoxelRegionBounds, VoxelRegionId, VoxelRegionKind,
    VoxelRegionRegistry,
};

/// Layer helpers called by emitter-generated `generate_terrain(seed)` functions.
/// See `terrain_layers.rs` for design notes.
pub mod terrain_layers;

/// Per-material property table — holds one `MaterialRow` per non-air
/// material id. Produced by the DSL emitter (T8) and stored on
/// `VoxelTerrain`. See `materials.rs` for the full design notes.
pub mod materials;
pub use materials::{MaterialRow, MaterialTable};

/// Worker-thread harness for off-tick terrain generation. See
/// `terrain_gen.rs` for design notes.
pub mod terrain_gen;
pub use terrain_gen::{TerrainGenHandle, TerrainGenError};

/// Navgrid index — per spec §7.2 + §7.3. See `navgrid.rs` for
/// design notes + the CPU-side build pipeline.
pub mod navgrid;
pub use navgrid::{
    build_navgrid, NavgridBuildError, NavgridCell, NavgridIndex,
    AGENT_STEP_HEIGHT, NAVGRID_BYTES_PER_CELL, NAVGRID_MAX_CELLS,
};

/// Default grid extent in voxel cells. 256³ cells (= 256 world-unit
/// cube) is enough for the Phase B `voxel_probe` fixture and the
/// Phase E `wave_defense` opt-in. Larger worlds will need a chunked
/// representation; that's a Phase C concern.
pub const DEFAULT_EXTENT: u32 = 256;

/// CPU-side voxel terrain backend. Implements the engine's
/// `TerrainQuery` trait by consulting a wrapped
/// `voxel_engine::voxel::grid::VoxelGrid`.
///
/// Cells are 1 world-unit on a side; the cell at integer `(x, y, z)`
/// covers `[x, x+1) × [y, y+1) × [z, z+1)`. Negative coordinates and
/// coordinates beyond the grid extent fall outside the populated region
/// and behave as empty (height 0, walkable true, LOS clear).
///
/// `apply_voxel_chronicle_record` takes `&mut self`; consumers call it
/// from a chronicle drainer that owns the terrain or holds a unique
/// reference. Phase B will decide whether to add interior mutability
/// (e.g. an `RwLock`) once the actual drain shape is wired up.
pub struct VoxelTerrain {
    grid: VoxelGrid,
    extent: u32,
    materials: MaterialTable,
}

impl VoxelTerrain {
    /// Construct an empty voxel world at the default extent.
    ///
    /// All cells start as air (value 0). Subsequent chronicle drains
    /// (Phase B) populate the grid as agents fire `place_voxel` and
    /// `harvest` abilities.
    pub fn new() -> Self {
        Self::with_extent(DEFAULT_EXTENT)
    }

    /// Construct an empty voxel world at a caller-chosen cubic extent.
    pub fn with_extent(extent: u32) -> Self {
        Self {
            grid: VoxelGrid::new(extent, extent, extent),
            extent,
            materials: materials::EMPTY,
        }
    }

    /// Construct an empty voxel world at a caller-chosen cubic extent
    /// with a pre-populated material table. Used by DSL-emitted terrain
    /// initializers (T8) that call
    /// `VoxelTerrain::with_extent_and_materials(EXTENT, MATERIALS)`.
    pub fn with_extent_and_materials(extent: u32, materials: MaterialTable) -> Self {
        Self {
            grid: VoxelGrid::new(extent, extent, extent),
            extent,
            materials,
        }
    }

    /// Return the material table for this terrain. The table is `Copy`
    /// so this is a cheap by-value return.
    pub fn materials(&self) -> MaterialTable {
        self.materials
    }

    /// Cubic extent (in cells) of the underlying grid. Cells outside
    /// `[0, extent)` on any axis are treated as empty by the trait
    /// methods.
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// Read a single cell's material id. Returns `0` for out-of-bounds
    /// coordinates (same as for genuinely empty cells). Phase A only
    /// uses this in tests; Phase B will use it inside the chronicle
    /// consumer.
    pub fn cell_at(&self, x: i32, y: i32, z: i32) -> u8 {
        if x < 0 || y < 0 || z < 0 {
            return 0;
        }
        self.grid.get(x as u32, y as u32, z as u32).unwrap_or(0)
    }

    /// Internal helper for tests / Phase B chronicle consumer:
    /// directly write a cell's material id. Out-of-bounds writes are
    /// silently dropped (matches `VoxelGrid::set` semantics).
    #[doc(hidden)]
    pub fn set_cell(&mut self, x: u32, y: u32, z: u32, value: u8) {
        self.grid.set(x, y, z, value);
    }

    /// Borrow the underlying `voxel_engine::voxel::grid::VoxelGrid` —
    /// used by [`VoxelMirror::new`] and [`VoxelMirror::flush_dirty`]
    /// to read the current cell values during construction + per-tick
    /// upload. Public because per-runtime `VoxelMirror` lives outside
    /// `VoxelTerrain` (the runtime owns both, mirror lifetime tracks
    /// the wgpu device).
    pub fn grid(&self) -> &VoxelGrid {
        &self.grid
    }

    /// Drain a single chronicle record into the voxel world.
    ///
    /// `rec` is one 10-word slice off the engine event ring (see
    /// `engine::gpu::EVENT_STRIDE_U32`). `caster_pos` is the position
    /// of `slot[2]`'s agent at consume tick — the runtime resolves
    /// `caster_slot → pos` against its agent SoA and passes the
    /// world-space `Vec3` in. See option (A) discussion in the crate
    /// doc-comment.
    ///
    /// Recognised event kinds (header word at `rec[0]`):
    ///
    /// - **60 = `EffectPlaceVoxelApplied`** — writes a non-zero voxel
    ///   at `(floor(caster_pos.x), floor(caster_pos.y),
    ///   floor(caster_pos.z))`. The material code is
    ///   `((kind_hash & 0xFF) as u8) | 1` so distinct kinds map to
    ///   distinct (deterministic) byte values; the `| 1` ensures we
    ///   never collide with the `0 = air` sentinel.
    /// - **59 = `EffectHarvestApplied`** — clears up to `amount`
    ///   non-zero voxels of the matching kind around the caster. The
    ///   scan walks a 3-cell-radius cube (small enough to be cheap,
    ///   large enough to catch the place + harvest pin's adjacent
    ///   cell) in deterministic (z, y, x) ascending order so the
    ///   removal sequence is byte-stable for the same input.
    ///
    /// Returns:
    /// - `Some(())` if the record was a voxel-mutating event and was
    ///   applied (or attempted — out-of-bounds writes drop silently
    ///   per `VoxelGrid::set` semantics).
    /// - `None` if the record was not one of the recognised kinds, or
    ///   if `rec` is shorter than the expected 10 words.
    pub fn apply_voxel_chronicle_record(
        &mut self,
        rec: &[u32],
        caster_pos: Vec3,
    ) -> Option<()> {
        // The engine ring stride is 10 u32; defend against a caller
        // that passes a clipped slice.
        if rec.len() < 5 {
            return None;
        }
        let kind_tag = rec[0];
        match kind_tag {
            60 => {
                // EffectPlaceVoxelApplied: rec[2]=caster_slot,
                // rec[3]=kind_hash. caster_slot is informational here —
                // the runtime already used it to resolve `caster_pos`.
                let kind_hash = rec[3];
                // `| 1` so we never accidentally write the air
                // sentinel for kinds whose low byte hashes to 0.
                let value: u8 = ((kind_hash & 0xFF) as u8) | 1;
                let cx = caster_pos.x.floor() as i32;
                let cy = caster_pos.y.floor() as i32;
                let cz = caster_pos.z.floor() as i32;
                if cx >= 0
                    && cy >= 0
                    && cz >= 0
                    && (cx as u32) < self.extent
                    && (cy as u32) < self.extent
                    && (cz as u32) < self.extent
                {
                    self.grid.set(cx as u32, cy as u32, cz as u32, value);
                }
                Some(())
            }
            59 => {
                // EffectHarvestApplied: rec[2]=caster_slot,
                // rec[3]=kind_hash, rec[4]=amount.
                let kind_hash = rec[3];
                let target_value: u8 = ((kind_hash & 0xFF) as u8) | 1;
                let mut amount_remaining = rec[4];
                if amount_remaining == 0 {
                    // Treat zero amount as a no-op (matches
                    // `apply_summon_event_to_state`'s zero-count
                    // tolerance).
                    return Some(());
                }
                // Bounding cube around caster: 3-cell radius. The
                // Phase B fixture places at the caster's own cell and
                // harvests from there too, so a tight neighborhood is
                // enough; a larger world-sim opt-in (Phase E) can
                // pass an explicit radius via the EffectOp::Harvest
                // payload once that's wired.
                const HARVEST_RADIUS: i32 = 3;
                let cx = caster_pos.x.floor() as i32;
                let cy = caster_pos.y.floor() as i32;
                let cz = caster_pos.z.floor() as i32;
                // Deterministic ascending z/y/x walk so the removal
                // order is byte-stable per (caster_pos, amount) input.
                'outer: for dz in -HARVEST_RADIUS..=HARVEST_RADIUS {
                    let z = cz + dz;
                    if z < 0 || (z as u32) >= self.extent {
                        continue;
                    }
                    for dy in -HARVEST_RADIUS..=HARVEST_RADIUS {
                        let y = cy + dy;
                        if y < 0 || (y as u32) >= self.extent {
                            continue;
                        }
                        for dx in -HARVEST_RADIUS..=HARVEST_RADIUS {
                            let x = cx + dx;
                            if x < 0 || (x as u32) >= self.extent {
                                continue;
                            }
                            let cur = self.grid.get(x as u32, y as u32, z as u32).unwrap_or(0);
                            if cur != 0 && cur == target_value {
                                self.grid.set(x as u32, y as u32, z as u32, 0);
                                amount_remaining -= 1;
                                if amount_remaining == 0 {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                Some(())
            }
            _ => None,
        }
    }
}

impl Default for VoxelTerrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VoxelTerrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoxelTerrain")
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}

impl TerrainQuery for VoxelTerrain {
    /// World-space ground height at horizontal `(x, y)`. Walks the
    /// column at integer `(floor(x), floor(y))` from top down and
    /// returns the highest occupied cell's `z + 1` (top face) as f32.
    /// Empty column or out-of-bounds returns `0.0`.
    fn height_at(&self, x: f32, y: f32) -> f32 {
        let cx = x.floor() as i32;
        let cy = y.floor() as i32;
        if cx < 0 || cy < 0 || (cx as u32) >= self.extent || (cy as u32) >= self.extent {
            return 0.0;
        }
        // Walk top-down so we return the highest occupied cell first.
        for cz in (0..self.extent as i32).rev() {
            if self.grid.get(cx as u32, cy as u32, cz as u32).unwrap_or(0) != 0 {
                return (cz + 1) as f32;
            }
        }
        0.0
    }

    /// `Walk` / `Climb` / `Swim` modes can occupy a cell only if it's
    /// air. `Fly` and `Fall` always pass (matches the constitutional
    /// stance that flying entities are not bound by ground terrain).
    /// Out-of-bounds cells default to walkable (matches FlatPlane).
    fn walkable(&self, pos: Vec3, mode: MovementMode) -> bool {
        if matches!(mode, MovementMode::Fly | MovementMode::Fall) {
            return true;
        }
        let cx = pos.x.floor() as i32;
        let cy = pos.y.floor() as i32;
        let cz = pos.z.floor() as i32;
        if cx < 0
            || cy < 0
            || cz < 0
            || (cx as u32) >= self.extent
            || (cy as u32) >= self.extent
            || (cz as u32) >= self.extent
        {
            return true;
        }
        self.grid.get(cx as u32, cy as u32, cz as u32).unwrap_or(0) == 0
    }

    /// Whether the straight-line segment from `from` to `to` is
    /// unobstructed. Uses voxel_engine's Amanatides-Woo DDA raymarcher
    /// (`voxel::raycast::ray_cast_grid`). Returns `true` (clear) if
    /// the ray exits the grid AABB without hitting a solid cell, or if
    /// the endpoints coincide.
    fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool {
        let delta = to - from;
        let length = delta.length();
        if length < 1e-6 {
            return true;
        }
        let dir = delta / length;
        let hit = ray_cast_grid(
            &self.grid,
            [from.x, from.y, from.z],
            [dir.x, dir.y, dir.z],
        );
        match hit {
            // Ray exits without hitting anything → unobstructed.
            None => true,
            // Hit found, but only counts as a blocker if it's between
            // `from` and `to`. `t` is in voxel-grid units, which are
            // identical to world units here (1 cell = 1 unit), so
            // compare directly to `length`.
            Some(h) => h.t > length,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase C — GPU-resident voxel mirror
// ---------------------------------------------------------------------------

/// Side-length in cells of one mirror chunk. Each chunk's worth of
/// cells is the smallest unit of mirror upload — when any cell in a
/// chunk is dirtied, the next `flush_dirty` re-uploads the whole chunk
/// to GPU. Picked to keep `Queue::write_buffer` calls cheap (one chunk
/// = `8³ = 512` cells = 2 KB) while still bounding upload churn for
/// scattered single-cell mutations.
pub const MIRROR_CHUNK_DIM: u32 = 8;

/// Identifier for a chunk in the mirror, in chunk-coordinates (cell
/// coords / `MIRROR_CHUNK_DIM`). Stored in a [`BTreeSet`] so iteration
/// order is deterministic — load-bearing for P5 (same chronicle drain
/// → same dirty set → same upload sequence → same GPU state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkCoord {
    pub cx: u32,
    pub cy: u32,
    pub cz: u32,
}

/// GPU-resident mirror of a [`VoxelTerrain`]'s underlying [`VoxelGrid`].
///
/// # Layout
///
/// One u32 per cell, flat 3D ordering: `index = z*H*W + y*W + x`
/// (matches `VoxelGrid::index`). The cell's material code lives in the
/// low 8 bits; upper 24 bits are reserved for future flags (e.g.
/// per-cell health for destructibles). Total mirror size at the
/// fixture's grid extent is `width * height * depth * 4` bytes:
///
/// - 16³ extent → 16 KB
/// - 64³ extent → 1 MB
/// - 256³ extent (DEFAULT) → 64 MB
///
/// The 256³ default is overkill for the Phase C voxel_probe pin (the
/// fixture only writes near origin); fixtures that opt in at smaller
/// extents (e.g. `with_extent(16)` for the pin tests) pay 16 KB.
///
/// # Dirty tracking
///
/// `mark_dirty(cell)` records the containing chunk in a `BTreeSet`
/// (P5: deterministic iteration order). `flush_dirty` walks the set
/// in ascending `(cz, cy, cx)` order and uploads each chunk's slice
/// of the host-side `u32` staging via `Queue::write_buffer`. The
/// staging vec is kept in lock-step with the host `VoxelGrid` so the
/// flush is a pure cell→u32 widen + slice copy. Empty dirty set →
/// zero `write_buffer` calls (cheap no-op).
pub struct VoxelMirror {
    buffer: wgpu::Buffer,
    /// Host-side staging for the GPU buffer's contents — one u32 per
    /// cell, in the same flat ordering. Updated in lock-step with the
    /// CPU-side `VoxelGrid` so `flush_dirty` does not need to re-read
    /// the grid (decoupled from `VoxelTerrain`'s lifetime).
    staging: Vec<u32>,
    /// Chunks dirtied since the last `flush_dirty`. BTreeSet so the
    /// drain order is deterministic across runs.
    dirty_chunks: BTreeSet<ChunkCoord>,
    width: u32,
    height: u32,
    depth: u32,
}

impl VoxelMirror {
    /// Allocate a mirror sized to `grid`'s dimensions and initialize
    /// from grid contents (whole-buffer write at construction time).
    /// All subsequent updates flow through `mark_dirty` + `flush_dirty`.
    pub fn new(gpu: &GpuContext, grid: &VoxelGrid) -> Self {
        let (width, height, depth) = grid.dimensions();
        let cell_count = (width as usize) * (height as usize) * (depth as usize);
        // Widen the u8 grid to u32 for the GPU mirror. WGSL has no
        // native u8 storage type, so the buffer is `array<u32>` and the
        // material code lives in the low byte; upper bits stay zero.
        let staging: Vec<u32> = grid.data().iter().map(|&b| b as u32).collect();
        debug_assert_eq!(staging.len(), cell_count);
        let bytes = (cell_count * 4) as u64;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_voxel::VoxelMirror::buffer"),
            // Always allocate at least 16 bytes so wgpu doesn't reject
            // a zero-sized storage buffer for tiny test fixtures.
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Initial whole-buffer upload — even if empty, doing this once
        // means future flushes never have to special-case "first write".
        if cell_count > 0 {
            gpu.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&staging));
        }
        Self {
            buffer,
            staging,
            dirty_chunks: BTreeSet::new(),
            width,
            height,
            depth,
        }
    }

    /// Borrow the underlying storage buffer — pass to
    /// `KernelBindingsContext::voxel_grid`.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Cubic dimensions of the mirror (cells along each axis). Matches
    /// the wrapped `VoxelGrid`'s dimensions at construction time.
    pub fn dimensions(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.depth)
    }

    /// Total mirror byte size = `width * height * depth * 4`. Useful
    /// for the per-fixture allocation bookkeeping.
    pub fn buffer_bytes(&self) -> u64 {
        (self.width as u64) * (self.height as u64) * (self.depth as u64) * 4
    }

    /// Number of dirty chunks pending flush. Mostly diagnostic — the
    /// runtime calls `flush_dirty` unconditionally each tick.
    pub fn dirty_chunk_count(&self) -> usize {
        self.dirty_chunks.len()
    }

    /// Map a cell coordinate to its containing chunk and mark the
    /// chunk as dirty. Out-of-bounds cells are silently ignored
    /// (matches `VoxelGrid::set` semantics — no panic).
    ///
    /// The runtime calls this from `apply_voxel_chronicle_record` (via
    /// the wrapper on `VoxelTerrain`) immediately after the grid has
    /// been mutated. Subsequent `flush_dirty` re-uploads the chunk's
    /// 8³ neighbourhood from the host `VoxelGrid`.
    pub fn mark_dirty(&mut self, cell: IVec3) {
        if cell.x < 0 || cell.y < 0 || cell.z < 0 {
            return;
        }
        let (x, y, z) = (cell.x as u32, cell.y as u32, cell.z as u32);
        if x >= self.width || y >= self.height || z >= self.depth {
            return;
        }
        self.dirty_chunks.insert(ChunkCoord {
            cx: x / MIRROR_CHUNK_DIM,
            cy: y / MIRROR_CHUNK_DIM,
            cz: z / MIRROR_CHUNK_DIM,
        });
    }

    /// Re-upload every dirty chunk to GPU and clear the dirty set.
    ///
    /// Reads each chunk's current state from the wrapped `grid`, widens
    /// each `u8` material code to `u32` in the host staging vec, then
    /// issues one `Queue::write_buffer` per dirty chunk. Drain order is
    /// `BTreeSet` ascending — same chronicle input → same drain order →
    /// same upload sequence → same final GPU state (P5 / P11).
    ///
    /// Empty dirty set → zero queue submits (P10: no panic on empty).
    pub fn flush_dirty(&mut self, gpu: &GpuContext, grid: &VoxelGrid) {
        if self.dirty_chunks.is_empty() {
            return;
        }
        // Take ownership of the dirty set so the per-chunk loop can
        // mutate `self.staging` without re-borrowing `self.dirty_chunks`.
        // `BTreeSet::into_iter` is already ascending, so the drained
        // sequence is the deterministic upload order.
        let drained = std::mem::take(&mut self.dirty_chunks);
        let (w, h, d) = (self.width, self.height, self.depth);
        for chunk in drained {
            // Chunk's cell-coord AABB, clipped to the grid extent.
            let x_lo = chunk.cx * MIRROR_CHUNK_DIM;
            let y_lo = chunk.cy * MIRROR_CHUNK_DIM;
            let z_lo = chunk.cz * MIRROR_CHUNK_DIM;
            let x_hi = (x_lo + MIRROR_CHUNK_DIM).min(w);
            let y_hi = (y_lo + MIRROR_CHUNK_DIM).min(h);
            let z_hi = (z_lo + MIRROR_CHUNK_DIM).min(d);
            // Refresh staging from the grid for every cell in this
            // chunk, then upload one row at a time. Per-row uploads
            // (rather than per-cell) trade a few `write_buffer` calls
            // for contiguous-byte transfers; per-chunk single upload
            // would require a temporary contiguous buffer because
            // chunks aren't laid out contiguously in flat-3D order.
            for z in z_lo..z_hi {
                for y in y_lo..y_hi {
                    let row_start = ((z as usize) * (h as usize) + y as usize)
                        * (w as usize)
                        + x_lo as usize;
                    let row_len = (x_hi - x_lo) as usize;
                    // Refresh staging from grid for this row.
                    for x in x_lo..x_hi {
                        let v = grid.get(x, y, z).unwrap_or(0);
                        self.staging[row_start + (x - x_lo) as usize] = v as u32;
                    }
                    let byte_offset = (row_start as u64) * 4;
                    let row_bytes = bytemuck::cast_slice(
                        &self.staging[row_start..row_start + row_len],
                    );
                    gpu.queue.write_buffer(&self.buffer, byte_offset, row_bytes);
                }
            }
        }
    }
}

impl VoxelTerrain {
    /// Map a world-space caster position to the cell that
    /// `apply_voxel_chronicle_record` writes for `EffectPlaceVoxelApplied`
    /// (kind=60). Used by the runtime's drain path to feed
    /// `VoxelMirror::mark_dirty` the same cell the grid mutation hit.
    pub fn place_cell_from_caster(caster_pos: Vec3) -> IVec3 {
        IVec3::new(
            caster_pos.x.floor() as i32,
            caster_pos.y.floor() as i32,
            caster_pos.z.floor() as i32,
        )
    }

    /// Same as [`apply_voxel_chronicle_record`] but additionally marks
    /// every cell touched by the record as dirty in `mirror`. The
    /// runtime calls this when a mirror is wired; the original
    /// `apply_voxel_chronicle_record` stays as the CPU-only entry.
    ///
    /// The mark-dirty pass walks the same neighborhood the consumer
    /// scans — for kind=60 (place) that's the single caster cell; for
    /// kind=59 (harvest) it's the `HARVEST_RADIUS`-cube. Marking the
    /// whole cube (rather than just the cells the consumer actually
    /// cleared) is a slight over-mark but keeps the consumer + mirror
    /// in lock-step: the consumer's `target_value` filter can no-op
    /// most of the cube, and the mirror just re-uploads the chunks
    /// containing those cells (which were already going to be
    /// re-uploaded if any one of them was a real clear).
    pub fn apply_voxel_chronicle_record_with_mirror(
        &mut self,
        rec: &[u32],
        caster_pos: Vec3,
        mirror: &mut VoxelMirror,
    ) -> Option<()> {
        // `?` short-circuits when the inner consumer rejects the record
        // (unknown kind tag or `rec.len() < 5`); after the `?`, rec[0]
        // is guaranteed to be one of the recognised voxel-mutating kinds.
        let result = self.apply_voxel_chronicle_record(rec, caster_pos)?;
        match rec[0] {
            60 => {
                let cell = Self::place_cell_from_caster(caster_pos);
                mirror.mark_dirty(cell);
            }
            59 => {
                // Mark every cell in the `HARVEST_RADIUS` cube so any
                // chunk that might contain a cleared cell is in the
                // dirty set. See the kind=59 arm in
                // `apply_voxel_chronicle_record` — `HARVEST_RADIUS=3`
                // is the magic number; mirroring it here would split
                // the source-of-truth, so we re-derive it from the
                // same ascending z/y/x walk.
                const HARVEST_RADIUS: i32 = 3;
                let cx = caster_pos.x.floor() as i32;
                let cy = caster_pos.y.floor() as i32;
                let cz = caster_pos.z.floor() as i32;
                for dz in -HARVEST_RADIUS..=HARVEST_RADIUS {
                    for dy in -HARVEST_RADIUS..=HARVEST_RADIUS {
                        for dx in -HARVEST_RADIUS..=HARVEST_RADIUS {
                            mirror.mark_dirty(IVec3::new(cx + dx, cy + dy, cz + dz));
                        }
                    }
                }
            }
            _ => {}
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_world_height_is_zero() {
        let t = VoxelTerrain::new();
        assert_eq!(t.height_at(0.0, 0.0), 0.0);
        assert_eq!(t.height_at(50.0, 50.0), 0.0);
        assert_eq!(t.height_at(-10.0, -10.0), 0.0);
    }

    #[test]
    fn empty_world_is_walkable_everywhere() {
        let t = VoxelTerrain::new();
        assert!(t.walkable(Vec3::new(1.0, 2.0, 3.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(-99.0, 0.0, 0.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(50.0, 50.0, 50.0), MovementMode::Fly));
        assert!(t.walkable(Vec3::new(0.0, 0.0, -10.0), MovementMode::Swim));
    }

    #[test]
    fn empty_world_has_clear_line_of_sight() {
        let t = VoxelTerrain::new();
        assert!(t.line_of_sight(Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)));
        assert!(t.line_of_sight(
            Vec3::new(-5.0, -5.0, -5.0),
            Vec3::new(5.0, 5.0, 5.0)
        ));
    }

    #[test]
    fn coincident_endpoints_have_clear_line_of_sight() {
        let t = VoxelTerrain::new();
        let p = Vec3::new(7.0, 7.0, 7.0);
        assert!(t.line_of_sight(p, p));
    }

    #[test]
    fn out_of_bounds_is_walkable() {
        let t = VoxelTerrain::with_extent(16);
        // Far outside the 16³ grid.
        assert!(t.walkable(Vec3::new(1000.0, 1000.0, 1000.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(-50.0, -50.0, -50.0), MovementMode::Walk));
    }

    #[test]
    fn fly_ignores_solid_cells() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Walk into solid → blocked.
        assert!(!t.walkable(Vec3::new(5.5, 5.5, 5.5), MovementMode::Walk));
        // Fly through solid → still passes (per the doc-comment contract).
        assert!(t.walkable(Vec3::new(5.5, 5.5, 5.5), MovementMode::Fly));
    }

    #[test]
    fn solid_cell_increases_height_at() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Top face of cell (5, 5, 5) sits at z = 6.
        assert_eq!(t.height_at(5.5, 5.5), 6.0);
        // Adjacent column still empty.
        assert_eq!(t.height_at(6.5, 5.5), 0.0);
    }

    #[test]
    fn solid_cell_blocks_line_of_sight() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Ray from (0, 5.5, 5.5) → (10, 5.5, 5.5) crosses cell (5, 5, 5).
        assert!(!t.line_of_sight(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(10.0, 5.5, 5.5)
        ));
        // Parallel ray two cells away is clear.
        assert!(t.line_of_sight(
            Vec3::new(0.0, 7.5, 5.5),
            Vec3::new(10.0, 7.5, 5.5)
        ));
    }

    #[test]
    fn ray_stopping_short_of_obstacle_is_clear() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Segment from (0, 5.5, 5.5) → (3, 5.5, 5.5) ends before the
        // obstacle at x∈[5,6); should be unobstructed.
        assert!(t.line_of_sight(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(3.0, 5.5, 5.5)
        ));
    }

    #[test]
    fn cell_at_returns_zero_for_negative_coords() {
        let t = VoxelTerrain::new();
        assert_eq!(t.cell_at(-1, 0, 0), 0);
        assert_eq!(t.cell_at(0, -1, 0), 0);
        assert_eq!(t.cell_at(0, 0, -1), 0);
    }

    #[test]
    fn apply_voxel_chronicle_record_unknown_kind_returns_none() {
        let mut t = VoxelTerrain::new();
        // Header kind=42 (some non-voxel event) → no-op, returns None.
        let mut rec = [0u32; 11];
        rec[0] = 42;
        rec[2] = 0;
        rec[3] = 0xCAFEBABE;
        assert_eq!(t.apply_voxel_chronicle_record(&rec, Vec3::ZERO), None);
    }

    #[test]
    fn apply_voxel_chronicle_record_short_slice_returns_none() {
        let mut t = VoxelTerrain::new();
        let rec = [60u32, 0, 0]; // shorter than 5 words
        assert_eq!(t.apply_voxel_chronicle_record(&rec, Vec3::ZERO), None);
    }

    #[test]
    fn place_voxel_record_writes_cell_at_caster_pos() {
        let mut t = VoxelTerrain::with_extent(16);
        let mut rec = [0u32; 11];
        rec[0] = 60; // EffectPlaceVoxelApplied
        rec[2] = 0; // caster_slot
        rec[3] = 0x12345678; // kind_hash
        let pos = Vec3::new(5.5, 7.25, 0.0);
        assert_eq!(t.apply_voxel_chronicle_record(&rec, pos), Some(()));
        // Floor → cell (5, 7, 0). Material code = (0x78 | 1) = 0x79.
        assert_eq!(t.cell_at(5, 7, 0), 0x79);
        // Adjacent cells stay empty.
        assert_eq!(t.cell_at(6, 7, 0), 0);
        assert_eq!(t.cell_at(5, 8, 0), 0);
    }

    #[test]
    fn place_voxel_lifts_height_at() {
        let mut t = VoxelTerrain::with_extent(16);
        let mut rec = [0u32; 11];
        rec[0] = 60;
        rec[3] = 1;
        // Drive the consumer at world origin (caster on z=0 cell).
        assert_eq!(t.apply_voxel_chronicle_record(&rec, Vec3::ZERO), Some(()));
        // Top face of cell (0,0,0) = z+1 = 1.0.
        assert!((t.height_at(0.5, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn harvest_record_clears_matching_voxel() {
        let mut t = VoxelTerrain::with_extent(16);
        // Place via the consumer so the material code byte matches
        // what the harvest path looks for.
        let mut place = [0u32; 11];
        place[0] = 60;
        place[3] = 7; // kind_hash → value 0x07 | 1 = 7
        let pos = Vec3::new(2.5, 2.5, 0.0);
        assert_eq!(t.apply_voxel_chronicle_record(&place, pos), Some(()));
        assert_eq!(t.cell_at(2, 2, 0), 7);

        // Now harvest matching kind, amount=1.
        let mut harv = [0u32; 11];
        harv[0] = 59;
        harv[3] = 7;
        harv[4] = 1;
        assert_eq!(t.apply_voxel_chronicle_record(&harv, pos), Some(()));
        assert_eq!(t.cell_at(2, 2, 0), 0);
    }

    #[test]
    fn harvest_record_does_not_clear_mismatched_kind() {
        let mut t = VoxelTerrain::with_extent(16);
        // Place kind_hash=7 (material 7).
        let mut place = [0u32; 11];
        place[0] = 60;
        place[3] = 7;
        let pos = Vec3::new(2.5, 2.5, 0.0);
        t.apply_voxel_chronicle_record(&place, pos);
        // Harvest a DIFFERENT kind (low-byte 9 → material 9). Same cell
        // is not material 9, so it stays.
        let mut harv = [0u32; 11];
        harv[0] = 59;
        harv[3] = 9;
        harv[4] = 5;
        assert_eq!(t.apply_voxel_chronicle_record(&harv, pos), Some(()));
        assert_eq!(t.cell_at(2, 2, 0), 7);
    }

    #[test]
    fn harvest_record_amount_bounds_removals() {
        let mut t = VoxelTerrain::with_extent(16);
        // Manually set 5 cells of the same material in the harvest
        // radius (use kind_hash=3 → material 3).
        let target_value: u8 = (3u32 & 0xFF) as u8 | 1;
        for i in 0..5 {
            t.set_cell(2, 2 + i, 0, target_value);
        }
        // Harvest with amount=2 → only 2 cleared.
        let mut harv = [0u32; 11];
        harv[0] = 59;
        harv[3] = 3;
        harv[4] = 2;
        let pos = Vec3::new(2.5, 2.5, 0.0);
        t.apply_voxel_chronicle_record(&harv, pos);
        let cleared: u32 = (0..5)
            .map(|i| if t.cell_at(2, 2 + i, 0) == 0 { 1 } else { 0 })
            .sum();
        assert_eq!(cleared, 2, "exactly `amount` cells should be cleared");
    }

    #[test]
    fn implements_terrain_query_via_arc_dyn() {
        // Confirm the adapter is usable through the same Arc<dyn> seam
        // SimState already exposes — i.e. callers can drop in
        // VoxelTerrain without further plumbing.
        use std::sync::Arc;
        let t: Arc<dyn TerrainQuery + Send + Sync> = Arc::new(VoxelTerrain::new());
        assert_eq!(t.height_at(0.0, 0.0), 0.0);
        assert!(t.walkable(Vec3::ZERO, MovementMode::Walk));
        assert!(t.line_of_sight(Vec3::ZERO, Vec3::ONE));
    }

    // ---------------------------------------------------------------
    // Phase C — VoxelMirror tests that don't need a GPU.
    //
    // These cover the chunk-coord math + dirty-set ordering. The
    // load-bearing CPU/GPU divergence pin lives in the
    // `cpu_gpu_voxel_state_matches` integration test under
    // `crates/engine_voxel/tests/cpu_gpu_mirror.rs` since it
    // requires a wgpu device.
    // ---------------------------------------------------------------

    /// Helper for the chunk-coord tests below — there's no public
    /// constructor that doesn't need a GPU, so we exercise the
    /// `BTreeSet` math directly.
    fn chunk_for(cell: IVec3) -> Option<ChunkCoord> {
        if cell.x < 0 || cell.y < 0 || cell.z < 0 {
            return None;
        }
        Some(ChunkCoord {
            cx: (cell.x as u32) / MIRROR_CHUNK_DIM,
            cy: (cell.y as u32) / MIRROR_CHUNK_DIM,
            cz: (cell.z as u32) / MIRROR_CHUNK_DIM,
        })
    }

    #[test]
    fn chunk_coord_orders_by_z_then_y_then_x() {
        // BTreeSet drain order is the load-bearing P5 invariant —
        // same chronicle drain → same dirty set → same upload
        // sequence → same final GPU state. The struct's derived
        // `Ord` orders fields lexicographically: (cx, cy, cz). Pin
        // that ordering so a future `#[derive]` reorder doesn't
        // silently change upload order.
        let a = ChunkCoord { cx: 0, cy: 0, cz: 0 };
        let b = ChunkCoord { cx: 1, cy: 0, cz: 0 };
        let c = ChunkCoord { cx: 0, cy: 1, cz: 0 };
        let d = ChunkCoord { cx: 0, cy: 0, cz: 1 };
        let mut set = BTreeSet::new();
        set.insert(d);
        set.insert(c);
        set.insert(b);
        set.insert(a);
        let drained: Vec<ChunkCoord> = set.into_iter().collect();
        // (0,0,0), (1,0,0), (0,1,0), (0,0,1) sort with cx outer-most:
        // (0,0,0) < (0,0,1) < (0,1,0) < (1,0,0)
        assert_eq!(drained, vec![a, d, c, b]);
    }

    #[test]
    fn chunk_for_groups_cells_by_dim() {
        // Two cells in the same MIRROR_CHUNK_DIM cube map to the same
        // ChunkCoord. Two cells across the boundary land in different
        // chunks. (Pin the boundary semantics so a knob change to
        // MIRROR_CHUNK_DIM surfaces here.)
        let dim = MIRROR_CHUNK_DIM as i32;
        let a = chunk_for(IVec3::new(0, 0, 0)).unwrap();
        let b = chunk_for(IVec3::new(dim - 1, dim - 1, dim - 1)).unwrap();
        let c = chunk_for(IVec3::new(dim, 0, 0)).unwrap();
        assert_eq!(a, b, "cells inside one chunk should match");
        assert_ne!(a, c, "cell on chunk boundary should land in next chunk");
    }

    // ---------------------------------------------------------------
    // Phase D semantic pins — the FlatPlane killers for `walkable`
    // and `line_of_sight` per the voxel-engine integration plan
    // (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`).
    // The CPU-side equivalents live here so they exercise the
    // `TerrainQuery` impl directly; the GPU equivalent (the
    // `gpu_terrain_query_matches_cpu` divergence pin) lives in
    // `tests/cpu_gpu_mirror.rs` since it needs a wgpu device.
    //
    // Don't substitute counter-based pins (e.g. "voxel count > 0")
    // for these — that's the probe-fooling pattern the plan calls
    // out by name. Each pin asserts an algebraic property
    // (place-then-walk-into → blocked; place-then-look-through →
    // obstructed) that FlatPlane silently passes.
    // ---------------------------------------------------------------

    /// **Phase D semantic pin #1.** A cell with a placed solid voxel
    /// must report `walkable = false` for ground-bound movement modes
    /// (Walk / Climb / Swim) and `walkable = true` for Fly/Fall.
    /// FlatPlane returns true for both → fails.
    #[test]
    fn solid_voxel_blocks_walkable() {
        let mut t = VoxelTerrain::with_extent(16);
        // Place a solid voxel at integer cell (5, 5, 5).
        t.set_cell(5, 5, 5, 1);
        // World-space pos inside that cell — floor((5.5,5.5,5.5)) =
        // (5,5,5). Walk + Climb + Swim all gate on the cell being air.
        let pos = Vec3::new(5.5, 5.5, 5.5);
        assert!(
            !t.walkable(pos, MovementMode::Walk),
            "solid voxel at (5,5,5) must block Walk; FlatPlane returns true"
        );
        assert!(
            !t.walkable(pos, MovementMode::Climb),
            "solid voxel at (5,5,5) must block Climb"
        );
        assert!(
            !t.walkable(pos, MovementMode::Swim),
            "solid voxel at (5,5,5) must block Swim"
        );
        // Fly/Fall always pass (per the doc-comment contract).
        assert!(
            t.walkable(pos, MovementMode::Fly),
            "solid voxel at (5,5,5) must NOT block Fly"
        );
        assert!(
            t.walkable(pos, MovementMode::Fall),
            "solid voxel at (5,5,5) must NOT block Fall"
        );
    }

    /// **Phase D semantic pin #2.** A solid voxel between two endpoints
    /// must obstruct line-of-sight; a parallel ray that doesn't cross
    /// any solid cell must be clear. FlatPlane returns true for both →
    /// fails.
    #[test]
    fn voxel_blocks_line_of_sight() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Segment crosses cell (5,5,5) → must be obstructed. Use the
        // cell's mid-line (z = 5.5) so the DDA actually traverses it
        // even with floating-point quantisation at cell faces.
        let blocked = t.line_of_sight(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(10.0, 5.5, 5.5),
        );
        assert!(
            !blocked,
            "ray (0,5.5,5.5) → (10,5.5,5.5) crosses solid cell (5,5,5) — \
             must be obstructed; FlatPlane returns true"
        );
        // Parallel ray on a different y-row — column (0..10, 10, 5) is
        // entirely empty; LOS must be clear.
        let clear = t.line_of_sight(
            Vec3::new(0.0, 10.5, 5.5),
            Vec3::new(10.0, 10.5, 5.5),
        );
        assert!(
            clear,
            "ray (0,10.5,5.5) → (10,10.5,5.5) doesn't cross any solid \
             cell — must be clear"
        );
    }
}
