//! Per-fixture runtime for `assets/sim/boids.sim`.
//!
//! Compiler-emitted: every per-kernel module + `dispatch::*` helper +
//! `KernelId` / `BufferRef` enum lives in `OUT_DIR/generated.rs`,
//! pulled in via [`include!`]. Hand-written: only the
//! [`BoidsState`] orchestration shell — buffer allocation, initial
//! data upload, dispatch wiring, and on-demand position readback.
//!
//! ## State shape
//!
//! [`BoidsState`] owns:
//!
//! - One [`engine::GpuContext`] (instance / adapter / device / queue).
//! - Per-field GPU storage buffers (`pos_buf`, `vel_buf`) sized to
//!   `agent_count * sizeof(Vec3Padded)`. WGSL's `array<vec3<f32>>`
//!   uses 16-byte stride — the [`Vec3Padded`] interop type matches.
//! - One uniform buffer (`cfg_buf`) holding [`PhysicsMoveBoidCfg`]
//!   (the agent-cap loop bound the dispatch reads from).
//! - One COPY_DST | MAP_READ staging buffer (`pos_staging`) used as
//!   the readback target for [`positions`]. Sized identically to
//!   `pos_buf`. Allocated once at construction; reused every
//!   readback.
//! - A host-side `pos_cache: Vec<Vec3>` (the slice [`positions`]
//!   returns) and a `dirty: bool` so consecutive [`positions`] calls
//!   without intervening [`step`]s skip the readback.
//! - The compiler-emitted [`generated::dispatch::KernelCache`], which
//!   lazy-init's each kernel's pipeline + BGL on first dispatch.
//!
//! ## Step body
//!
//! Encodes one `physics_MoveBoid` dispatch (the only kernel boids
//! actually needs — the other emitted kernels are wolf-sim plumbing
//! that this fixture ignores), submits, marks `dirty`, increments
//! the tick. No readback in the hot path.
//!
//! ## Position readback
//!
//! [`positions`] is on-demand: when `dirty` is set, it encodes a
//! `pos_buf → pos_staging` copy, submits, blocks on map_async, and
//! decodes the mapped bytes into `pos_cache`. When the application
//! polls per-tick, this trips once per tick; when it polls less
//! often, the readback cost amortises. Explicit `wgpu::PollType::Wait`
//! drives the device synchronously so the host call path stays sync.
//!
//! ## What the trait surface looks like to callers
//!
//! Identical to a CPU-only runtime — `make_sim()` returns a
//! `Box<dyn CompiledSim>`, the application crate sees only the
//! trait, switching to a future fixture's runtime is a one-line
//! Cargo.toml package alias change.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

// Compiler-emitted modules pulled in at crate root: per-kernel
// pipelines + bindings, the `dispatch::*` lazy-init helpers, and the
// `KernelId` / `BufferRef` enums. Lives at the top level (not inside
// a wrapper mod) because the emitted `dispatch` module references
// per-kernel modules via absolute `crate::physics_MoveBoid::*`
// paths. Each `pub mod` the include splices in carries its own outer
// `#[allow(non_snake_case, ...)]` (set by build.rs), so no
// crate-level allow is needed here.
include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Host-side representation of a WGSL `vec3<f32>` array element.
/// WGSL specifies 16-byte stride / 16-byte alignment for `vec3<f32>`
/// in storage arrays — packing as four `f32`s (with the fourth left
/// as zero padding) keeps host writes / reads byte-compatible with
/// shader accesses on either end.
#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Padded {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z, _pad: 0.0 }
    }
}

impl From<Vec3Padded> for Vec3 {
    fn from(p: Vec3Padded) -> Self {
        Vec3::new(p.x, p.y, p.z)
    }
}

/// Per-fixture state for the boids simulation.
pub struct BoidsState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,
    /// Sorted-by-cell agent ids — `array<u32>` sized `agent_cap`
    /// (one slot per agent total, no per-cell cap). After the
    /// three-phase counting sort populates this, cell `c`'s agent
    /// ids occupy `[spatial_grid_starts[c] .. spatial_grid_starts[c + 1])`.
    /// Replaces the bounded-list `cell * MAX + slot` layout — no
    /// per-cell capacity, no silently-dropped agents.
    spatial_grid_cells: wgpu::Buffer,
    /// Per-cell atomic counters — `array<atomic<u32>>` sized
    /// `num_cells`. Triple-duty across the counting-sort phases:
    /// (1) phase-1 counts, (2) phase-2 reset to zero, (3) phase-3
    /// per-cell write cursors that re-accumulate to the cell's
    /// total count. Diagnostic kernel reads it post-scatter to
    /// surface per-tick density stats.
    spatial_grid_offsets: wgpu::Buffer,
    /// Per-cell start offsets after the prefix scan — `array<u32>`
    /// sized `num_cells + 1`. `starts[c]` is cell `c`'s position in
    /// `spatial_grid_cells`; `starts[c + 1] - starts[c]` is the
    /// cell's count. The trailing `starts[num_cells]` equals
    /// `agent_cap` (every agent ended up somewhere). Written
    /// cooperatively by the parallel scan (phases 2a + 2c); read by
    /// phase 3 (BuildHashScatter) + the tiled-MoveBoid cooperative
    /// load + walk.
    spatial_grid_starts: wgpu::Buffer,
    /// Per-workgroup-chunk total used by the parallel prefix scan —
    /// `array<u32>` sized `ceil(num_cells / SCAN_CHUNK_SIZE)` (~42
    /// entries for boids' 22³=10 648 grid at 256-cell chunks).
    /// Phase 2a writes each chunk's total; phase 2b serial-scans it
    /// in place into an exclusive prefix; phase 2c reads the
    /// per-chunk base.
    spatial_chunk_sums: wgpu::Buffer,
    /// Pre-allocated zero buffer used as the COPY_SRC for the per-
    /// tick offsets clear. Sized to match `spatial_grid_offsets`.
    /// Allocated once at construction; cheaper than rebuilding a
    /// `vec![0u32; num_cells]` host array every tick.
    spatial_offsets_zero: wgpu::Buffer,

    cache: dispatch::KernelCache,

    pos_cache: Vec<Vec3>,
    dirty: bool,

    /// Per-kernel GPU-side wall time captured via `wgpu::QuerySet`
    /// timestamp queries. `None` when the adapter doesn't support
    /// `Features::TIMESTAMP_QUERY` (typical for some software
    /// fallbacks); the sim still runs, just without per-kernel
    /// attribution. See [`BoidsState::metrics`] for the public
    /// surface.
    timing: Option<TimingState>,

    /// Hand-written diagnostic kernel + buffers that scan
    /// `spatial_grid_offsets` after BuildHash and surface the
    /// per-tick spatial-grid health (max occupancy, dropped agent
    /// count, non-empty cell count). Always allocated — the GPU cost
    /// is one ~10k-thread dispatch per tick (microseconds at boids'
    /// grid size); the host cost is one extra readback when
    /// `metrics()` is called. Lives on the runtime side rather than
    /// in the compiler emit because the BoidsState already owns the
    /// spatial buffers and pipeline cache, and the diagnostic body
    /// is small + fixture-agnostic; future fixtures with the same
    /// spatial-grid shape can share this path verbatim.
    diag: SpatialDiagState,

    /// Last-completed metrics snapshot — populated by `metrics()`
    /// after a readback completes. Surfaces zeros until at least one
    /// `step()` + `metrics()` pair has run; thereafter it holds the
    /// most recent successfully-resolved tick's GPU times.
    last_metrics: BoidsMetrics,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Owned wgpu state for per-kernel timestamp queries.
///
/// Slot layout (5 timestamps per tick):
///   0: before offsets-clear copy
///   1: after offsets-clear copy   → clear_ns = (1) - (0)
///   2: after BuildHash dispatch   → build_hash_ns = (2) - (1)
///   3: after MoveBoid dispatch    → move_boid_ns = (3) - (2)
///   4: end-of-tick sentinel       → total_ns = (4) - (0)
///
/// Each `step()` writes timestamps 0..=4 into `query_set`,
/// `resolve_query_set`s them into `resolve_buf`, and copies that
/// onto `readback_buf` (MAP_READ). Readback is *lazy*: `metrics()`
/// maps `readback_buf` and decodes when the caller asks. Map_async
/// runs on the same queue as the dispatches, so by the time the
/// caller awaits it the prior tick's GPU work has flushed.
struct TimingState {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    /// `Queue::get_timestamp_period()` once at construction. Multiply
    /// by raw u64 deltas to get nanoseconds.
    period_ns: f32,
    /// True after a `step()` has written timestamps; flips back to
    /// false once `metrics()` consumes them. Prevents double-readback
    /// on consecutive `metrics()` calls without an intervening
    /// `step()`.
    has_pending: bool,
}

/// Per-kernel GPU wall time for the most recently completed tick,
/// plus structural counters. Returned by [`BoidsState::metrics`].
/// All times are GPU-side (resolved from `wgpu::QuerySet` timestamps,
/// not host wall clock); they exclude submit / map_async wait /
/// driver overhead. A zero `*_ns` value means "no measurement
/// available yet" — either timestamp_query isn't supported on this
/// adapter, or `metrics()` ran before the first `step()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoidsMetrics {
    /// GPU time spent in the offsets-buffer zero-clear copy.
    pub clear_ns: u64,
    /// GPU time spent in the spatial-hash BuildHash kernel.
    pub build_hash_ns: u64,
    /// GPU time spent in the physics MoveBoid kernel.
    pub move_boid_ns: u64,
    /// GPU time from clear-start to MoveBoid-end.
    pub total_ns: u64,
    /// True when `Features::TIMESTAMP_QUERY` is available on the
    /// adapter. When false, every `*_ns` field stays zero.
    pub timestamp_supported: bool,
    /// Tick number the times were captured at. Returned alongside the
    /// times so callers can detect "metrics from a stale tick".
    pub tick: u64,

    // ---- Spatial-grid health (from the diagnostic kernel) ----
    /// Largest single-cell agent count observed this tick. Includes
    /// "wanted" counts past `MAX_PER_CELL` — the BuildHash kernel
    /// keeps incrementing the offsets atomic past the cap even
    /// though the slot writes get gated, so this surfaces the true
    /// peak density rather than the clamped one.
    pub max_per_cell_seen: u32,
    /// Total agents the BuildHash kernel tried to slot but couldn't,
    /// summed across all cells (= `Σ max(0, count - MAX_PER_CELL)`).
    /// Non-zero means the spatial walks missed agents this tick;
    /// the simulation under-reports neighbors for the affected cells.
    pub dropped_agents: u32,
    /// Number of cells with at least one agent this tick. Useful as
    /// a denominator for "average cells touched" + as a sanity check
    /// (`dropped_agents > 0` implies high density in *some* cells,
    /// not necessarily all).
    pub nonempty_cells: u32,
}

/// Hand-written diagnostic-kernel state. One pipeline + bind-group
/// layout, one stats buffer (`array<atomic<u32>, 4>`), one
/// pre-allocated zero-init buffer for the per-tick clear, one
/// MAP_READ staging buffer.
///
/// Stats slot layout (matches the WGSL kernel below):
///   [0] max occupancy seen this tick (via `atomicMax`)
///   [1] sum of `max(0, count - MAX_PER_CELL)` across all cells
///   [2] count of non-empty cells
///   [3] reserved (future histogram bucket)
struct SpatialDiagState {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cfg_buf: wgpu::Buffer,
    stats_buf: wgpu::Buffer,
    stats_zero: wgpu::Buffer,
    stats_staging: wgpu::Buffer,
    has_pending: bool,
}

/// `cs_spatial_diag` cfg uniform (16 B padded). Mirrors
/// `SpatialDiagCfg` in the inline WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpatialDiagCfg {
    num_cells: u32,
    max_per_cell: u32,
    _pad: [u32; 2],
}

/// WGSL for the diagnostic kernel. Per-cell dispatch (one thread per
/// cell, bounded by `cfg.num_cells`). Each non-empty cell atomic-
/// updates the four stats slots.
///
/// Why hand-written: the compiler emit doesn't have a "scan a
/// SpatialStorage buffer and write a small reduction" surface, and
/// adding one for a single diagnostic dispatch isn't worth the
/// emit-path scope. The WGSL stays small + fixture-agnostic so
/// every spatial-grid-using fixture can reuse this same kernel.
const SPATIAL_DIAG_WGSL: &str = "
struct SpatialDiagCfg { num_cells: u32, max_per_cell: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read_write> spatial_grid_offsets: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> diag_stats: array<atomic<u32>, 4>;
@group(0) @binding(2) var<uniform> diag_cfg: SpatialDiagCfg;

@compute @workgroup_size(64)
fn cs_spatial_diag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= diag_cfg.num_cells) { return; }
    let count = atomicLoad(&spatial_grid_offsets[cell]);
    if (count == 0u) { return; }
    atomicMax(&diag_stats[0], count);
    if (count > diag_cfg.max_per_cell) {
        atomicAdd(&diag_stats[1], count - diag_cfg.max_per_cell);
    }
    atomicAdd(&diag_stats[2], 1u);
}
";

impl BoidsState {
    /// Construct an N-boid simulation with deterministic initial
    /// positions + velocities derived from `seed` via engine's keyed
    /// PCG RNG (P5: `per_agent_u32(seed, agent_id, tick=0, purpose)`).
    /// Allocates GPU storage / staging buffers sized for `agent_count`
    /// and uploads the initial host data.
    ///
    /// Auto-derives the initial-spread half-extent from `agent_count` so
    /// the average spatial-grid density stays around one agent per cell
    /// (well below `MAX_PER_CELL`). See [`Self::new_with_spread`] for
    /// the override path used when a fixture wants a specific cube size
    /// (e.g. visual flocking demos that need a tight starting cluster).
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::new_with_spread(seed, agent_count, None)
    }

    /// Construct with an explicit `spread` override (`Some(half_extent)`)
    /// or fall back to density-derived auto-spread when `None`.
    ///
    /// **Auto-spread formula** (active when `spread = None`):
    /// ```text
    /// spread = clamp(cbrt(agent_count) * (CELL_SIZE / 2),
    ///                CELL_SIZE / 2,
    ///                WORLD_HALF_EXTENT)
    /// ```
    /// Rationale: the swarm fills a cube whose side length is
    /// `cbrt(N) * CELL_SIZE`, i.e. `cbrt(N)` cells per axis, giving a
    /// uniform-distribution average density of ~1 agent per cell. For
    /// counts that would overflow the world (`cbrt(N) * CELL_SIZE / 2 >
    /// WORLD_HALF_EXTENT`, i.e. roughly `N > grid_dim^3 ≈ 10k` at the
    /// current 128-unit world), spread saturates at `WORLD_HALF_EXTENT`
    /// — agents fill the entire world and density rises with N. The
    /// proper fix at >10k is to grow the world (deferred); this
    /// constructor at least keeps small-N caps comfortable and
    /// matches the spatial-grid `MAX_PER_CELL` budget at boids' usual
    /// scale.
    ///
    /// Lower bound `CELL_SIZE / 2` (= 3.0) prevents the pathological
    /// `N ≤ 1` collapse to zero spread.
    pub fn new_with_spread(seed: u64, agent_count: u32, spread: Option<f32>) -> Self {
        let n = agent_count as usize;
        let spread = spread.unwrap_or_else(|| Self::auto_spread(agent_count));
        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boids_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boids_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cfg = physics_MoveBoid::PhysicsMoveBoidCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boids_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Spatial-grid buffers (real counting sort layout):
        //   - spatial_grid_cells: sized `agent_cap` (every agent gets
        //     a unique slot in the sorted output; no per-cell cap)
        //   - spatial_grid_offsets: sized `num_cells` (atomic counts;
        //     reused as scatter cursor across the three phases)
        //   - spatial_grid_starts: sized `num_cells + 1` (prefix-scan
        //     output; cell c's agent ids live in
        //     `cells[starts[c]..starts[c + 1]]`)
        //
        // Constants live in `dsl_compiler::cg::emit::spatial` so the
        // host-side allocation can never drift from what the kernels
        // were compiled against (cell size, world bounds, grid_dim,
        // num_cells — all the same WGSL `const` declarations the
        // build.rs emit produces).
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (agent_count as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Parallel prefix-scan carry buffer. Sized to one u32 per
        // scan chunk (`PER_SCAN_CHUNK_WORKGROUP_X` cells per chunk).
        // The scan sequence (Local → Carry → Add) writes/reads every
        // slot before any consumer touches it, so the buffer doesn't
        // need a per-tick clear — the runtime allocation can stay
        // uninitialized.
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = (sp::num_cells()).div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Per-tick zero-clear source for offsets. `write_buffer`
        // would also work but allocating a `Vec<u32; num_cells>`
        // every tick churns the host heap; a pre-built zero buffer
        // + copy_buffer_to_buffer is one allocation up front + one
        // GPU-side copy per tick.
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boids_runtime::spatial_offsets_zero"),
            contents: &zeros,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Spatial-grid diagnostic kernel — small fixed-overhead
        // pipeline that scans `spatial_grid_offsets` after each
        // BuildHash dispatch and writes 4 `u32` stats. See the
        // `SPATIAL_DIAG_WGSL` constant + `SpatialDiagState` doc
        // for the slot layout. Always built; cost is one ~num_cells
        // / 64 workgroup dispatch per tick (≈170 workgroups for
        // boids' 22³=10 648 cells) which timestamps put in the
        // microsecond range — well below noise on the bench.
        let diag = {
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("boids_runtime::spatial_diag::wgsl"),
                source: wgpu::ShaderSource::Wgsl(SPATIAL_DIAG_WGSL.into()),
            });
            let bgl = gpu
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("boids_runtime::spatial_diag::bgl"),
                    entries: &[
                        // spatial_grid_offsets (atomic; ReadWrite required for atomic access
                        // even though the diag kernel only reads/atomicLoads)
                        engine::gpu::bgl_storage(0, false),
                        // diag_stats (atomic)
                        engine::gpu::bgl_storage(1, false),
                        // diag_cfg (uniform)
                        engine::gpu::bgl_uniform(2),
                    ],
                });
            let pl = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("boids_runtime::spatial_diag::pl"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });
            let pipeline = gpu
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("boids_runtime::spatial_diag::pipeline"),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some("cs_spatial_diag"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let diag_cfg = SpatialDiagCfg {
                num_cells: dsl_compiler::cg::emit::spatial::num_cells(),
                max_per_cell: dsl_compiler::cg::emit::spatial::MAX_PER_CELL,
                _pad: [0; 2],
            };
            let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("boids_runtime::spatial_diag::cfg"),
                contents: bytemuck::bytes_of(&diag_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let stats_size = 4 * 4u64; // 4 × u32
            let stats_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("boids_runtime::spatial_diag::stats"),
                size: stats_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_zero =
                gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("boids_runtime::spatial_diag::stats_zero"),
                    contents: bytemuck::cast_slice(&[0u32; 4]),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            let stats_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("boids_runtime::spatial_diag::stats_staging"),
                size: stats_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            SpatialDiagState {
                pipeline,
                bgl,
                cfg_buf,
                stats_buf,
                stats_zero,
                stats_staging,
                has_pending: false,
            }
        };

        // Timestamp instrumentation — best-effort. Falls back to None
        // when the adapter doesn't expose `TIMESTAMP_QUERY`. Five
        // timestamps per tick (see TimingState slot layout); the
        // resolve buffer is sized for u64 timestamps (8 bytes each).
        let timing = if gpu.supports_timestamp_query() {
            const SLOTS: u32 = 5;
            let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("boids_runtime::timing"),
                ty: wgpu::QueryType::Timestamp,
                count: SLOTS,
            });
            let resolve_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("boids_runtime::timing_resolve"),
                size: (SLOTS as u64) * 8,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            });
            let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("boids_runtime::timing_readback"),
                size: (SLOTS as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let period_ns = gpu.queue.get_timestamp_period();
            Some(TimingState {
                query_set,
                resolve_buf,
                readback_buf,
                period_ns,
                has_pending: false,
            })
        } else {
            None
        };

        Self {
            gpu,
            pos_buf,
            vel_buf,
            cfg_buf,
            pos_staging,
            spatial_grid_cells,
            spatial_grid_offsets,
            spatial_grid_starts,
            spatial_chunk_sums,
            spatial_offsets_zero,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            timing,
            diag,
            last_metrics: BoidsMetrics::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-kernel GPU-side wall times for the most recently completed
    /// tick. On `Features::TIMESTAMP_QUERY`-capable adapters this
    /// drives a `device.poll(Wait)` to synchronise with the GPU
    /// before decoding the timestamp buffer; the readback runs
    /// once and the result caches in `last_metrics` until the next
    /// `step()` overwrites it. On adapters without timestamp
    /// support, returns a sentinel where every `*_ns` is zero and
    /// `timestamp_supported = false`.
    pub fn metrics(&mut self) -> BoidsMetrics {
        // Both readbacks (timestamps + spatial-grid stats) share a
        // single `device.poll(Wait)` so we only stall the host once.
        // Either source can be absent: timestamps are gated on the
        // adapter feature; the diag staging may be missing if no
        // step has run yet (`has_pending = false`).
        let want_timing = self
            .timing
            .as_ref()
            .map(|t| t.has_pending)
            .unwrap_or(false);
        let want_diag = self.diag.has_pending;
        if !want_timing && !want_diag {
            return self.last_metrics;
        }
        if want_timing {
            let slice = self.timing.as_ref().unwrap().readback_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                res.expect("timing_readback map_async failed");
            });
        }
        if want_diag {
            let slice = self.diag.stats_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                res.expect("spatial_diag stats_staging map_async failed");
            });
        }
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during metrics readback");

        // Decode + unmap each source independently. Carry forward
        // any unread fields from `last_metrics` so a partial readback
        // doesn't zero out the other source.
        let mut next = self.last_metrics;
        next.tick = self.tick;
        if want_timing {
            let timing = self.timing.as_mut().unwrap();
            let raw: [u64; 5] = {
                let view = timing.readback_buf.slice(..).get_mapped_range();
                let words: &[u64] = bytemuck::cast_slice(&view);
                [words[0], words[1], words[2], words[3], words[4]]
            };
            timing.readback_buf.unmap();
            timing.has_pending = false;
            let period = timing.period_ns as f64;
            let ns = |a: u64, b: u64| ((b.saturating_sub(a)) as f64 * period) as u64;
            next.clear_ns = ns(raw[0], raw[1]);
            next.build_hash_ns = ns(raw[1], raw[2]);
            next.move_boid_ns = ns(raw[2], raw[3]);
            next.total_ns = ns(raw[0], raw[4]);
            next.timestamp_supported = true;
        }
        if want_diag {
            let stats: [u32; 4] = {
                let view = self.diag.stats_staging.slice(..).get_mapped_range();
                let words: &[u32] = bytemuck::cast_slice(&view);
                [words[0], words[1], words[2], words[3]]
            };
            self.diag.stats_staging.unmap();
            self.diag.has_pending = false;
            next.max_per_cell_seen = stats[0];
            next.dropped_agents = stats[1];
            next.nonempty_cells = stats[2];
            // stats[3] reserved for future histogram bucket
        }
        self.last_metrics = next;
        self.last_metrics
    }

    /// Read-only view of the seed used at construction. Kept for
    /// snapshot/replay debugging; not exposed through [`CompiledSim`].
    #[allow(dead_code)]
    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    /// Density-aware initial-spread half-extent, derived from
    /// `agent_count` so the spatial-grid average occupancy stays near
    /// one agent per cell. See [`Self::new_with_spread`] for the
    /// formula's docstring + rationale. Exposed (`pub`) so the bench
    /// harness can print the chosen spread per cap.
    pub fn auto_spread(agent_count: u32) -> f32 {
        use dsl_compiler::cg::emit::spatial as sp;
        let n = agent_count.max(1) as f32;
        let target = n.cbrt() * (sp::CELL_SIZE * 0.5);
        target.clamp(sp::CELL_SIZE * 0.5, sp::WORLD_HALF_EXTENT)
    }

    /// Encode a `pos_buf → pos_staging` copy, submit, block on
    /// map_async, decode the mapped bytes into `pos_cache`, unmap.
    /// Sync end-to-end via `wgpu::PollType::Wait`.
    fn readback_positions(&mut self) {
        let n = self.agent_count as usize;
        let bytes = (n * std::mem::size_of::<Vec3Padded>()) as u64;

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("boids_runtime::pos_readback"),
            });
        encoder.copy_buffer_to_buffer(&self.pos_buf, 0, &self.pos_staging, 0, bytes);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.pos_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("pos_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during pos readback");

        {
            let view = slice.get_mapped_range();
            let padded: &[Vec3Padded] = bytemuck::cast_slice(&view);
            self.pos_cache.clear();
            self.pos_cache
                .extend(padded.iter().copied().map(Vec3::from));
        }
        self.pos_staging.unmap();
    }
}

impl CompiledSim for BoidsState {
    fn step(&mut self) {
        // One encoder per tick chains:
        //   1. Clear `spatial_grid_offsets` to all-zero (copy from
        //      the pre-built zero buffer). Required precondition for
        //      the bounded counting-sort BuildHash kernel.
        //   2. Dispatch `spatial_build_hash` — populates the per-cell
        //      slot list from this tick's positions.
        //   3. Dispatch `physics_MoveBoid` — reads the populated grid
        //      via ForEachNeighbor walks, writes new pos/vel.
        // wgpu enforces the 1→2→3 order via command-buffer sequencing
        // (no cross-dispatch synchronization needed for storage-buffer
        // R/W within the same encoder submission — the device
        // serialises adjacent compute passes against one another).
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("boids_runtime::step"),
                });

        // Per-kernel timestamp queries — interleaved with the
        // dispatches via `encoder.write_timestamp` (the
        // pass-external form, available under bare
        // `Features::TIMESTAMP_QUERY` without
        // `TIMESTAMP_QUERY_INSIDE_PASSES`). Slot layout matches the
        // `TimingState` docstring: 0=before-clear, 1=after-clear /
        // before-BuildHash, 2=after-BuildHash / before-MoveBoid,
        // 3=after-MoveBoid, 4=end-of-tick sentinel. When the adapter
        // doesn't support timestamps, `self.timing` is None and the
        // dispatches run unwrapped.
        if let Some(t) = self.timing.as_ref() {
            encoder.write_timestamp(&t.query_set, 0);
        }

        // (1) Clear spatial_grid_offsets — copy_buffer_to_buffer is
        //     cheaper than rebuilding a host-side zero array each tick.
        let offsets_size =
            dsl_compiler::cg::emit::spatial::offsets_bytes();
        encoder.copy_buffer_to_buffer(
            &self.spatial_offsets_zero,
            0,
            &self.spatial_grid_offsets,
            0,
            offsets_size,
        );
        if let Some(t) = self.timing.as_ref() {
            encoder.write_timestamp(&t.query_set, 1);
        }

        // (2) Five-kernel real counting sort:
        //     2a. BuildHashCount      — per-agent atomicAdd into offsets
        //     2b. BuildHashScanLocal  — workgroup-local Hillis-Steele
        //                               scan over each 256-cell chunk;
        //                               writes per-chunk inclusive
        //                               prefix to starts and chunk
        //                               total to chunk_sums
        //     2c. BuildHashScanCarry  — single-thread exclusive scan
        //                               over chunk_sums in place
        //     2d. BuildHashScanAdd    — adds chunk_sums[chunk_id] to
        //                               every starts entry in the
        //                               chunk; resets offsets to zero
        //                               so phase 3 can reuse it as a
        //                               write cursor
        //     2e. BuildHashScatter    — per-agent atomicAdd on offsets
        //                               (now write cursor) → write
        //                               agent_id into
        //                               cells[starts[cell] + local_slot]
        //
        // wgpu's command-buffer ordering serialises the dispatches
        // against one another (same encoder + adjacent compute
        // passes). After the scatter, `cells` holds every agent id
        // grouped by cell, with no per-cell capacity cap.
        let count_bindings = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_count(
            &mut self.cache,
            &count_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_local_bindings =
            spatial_build_hash_scan_local::SpatialBuildHashScanLocalBindings {
                spatial_grid_offsets: &self.spatial_grid_offsets,
                spatial_grid_starts: &self.spatial_grid_starts,
                spatial_chunk_sums: &self.spatial_chunk_sums,
                cfg: &self.cfg_buf,
            };
        dispatch::dispatch_spatial_build_hash_scan_local(
            &mut self.cache,
            &scan_local_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_carry_bindings =
            spatial_build_hash_scan_carry::SpatialBuildHashScanCarryBindings {
                spatial_chunk_sums: &self.spatial_chunk_sums,
                cfg: &self.cfg_buf,
            };
        dispatch::dispatch_spatial_build_hash_scan_carry(
            &mut self.cache,
            &scan_carry_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_add_bindings =
            spatial_build_hash_scan_add::SpatialBuildHashScanAddBindings {
                spatial_grid_offsets: &self.spatial_grid_offsets,
                spatial_grid_starts: &self.spatial_grid_starts,
                spatial_chunk_sums: &self.spatial_chunk_sums,
                cfg: &self.cfg_buf,
            };
        dispatch::dispatch_spatial_build_hash_scan_add(
            &mut self.cache,
            &scan_add_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scatter_bindings = spatial_build_hash_scatter::SpatialBuildHashScatterBindings {
            agent_pos: &self.pos_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache,
            &scatter_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        if let Some(t) = self.timing.as_ref() {
            encoder.write_timestamp(&t.query_set, 2);
        }

        // (2.5) Spatial-grid diagnostic dispatch — clears the stats
        // buffer and runs the per-cell scan that surfaces
        // max-occupancy / dropped-agents / nonempty-cells. Lives
        // BETWEEN BuildHash and MoveBoid so it sees the just-built
        // grid; dropped from timestamp accounting (the diag kernel
        // is a measurement instrument, not a sim cost).
        encoder.copy_buffer_to_buffer(
            &self.diag.stats_zero,
            0,
            &self.diag.stats_buf,
            0,
            4 * 4u64,
        );
        let diag_bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("boids_runtime::spatial_diag::bg"),
            layout: &self.diag.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.diag.stats_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.diag.cfg_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("boids_runtime::spatial_diag::pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.diag.pipeline);
            pass.set_bind_group(0, &diag_bg, &[]);
            let num_cells = dsl_compiler::cg::emit::spatial::num_cells();
            let workgroups = num_cells.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        // Copy stats → staging so metrics() can read without a fresh dispatch.
        encoder.copy_buffer_to_buffer(
            &self.diag.stats_buf,
            0,
            &self.diag.stats_staging,
            0,
            4 * 4u64,
        );

        // (3) Physics MoveBoid dispatch — reads agent_pos / agent_vel
        //     + the spatial grid (cells, offsets, starts), writes new
        //     positions/velocities.
        let bindings = physics_MoveBoid::PhysicsMoveBoidBindings {
            agent_pos: &self.pos_buf,
            agent_vel: &self.vel_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_moveboid(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        if let Some(t) = self.timing.as_ref() {
            encoder.write_timestamp(&t.query_set, 3);
            encoder.write_timestamp(&t.query_set, 4);
            // Resolve the 5 raw u64 timestamps into `resolve_buf`,
            // then copy onto the MAP_READ-able readback buffer so a
            // later `metrics()` call can decode without re-resolving.
            encoder.resolve_query_set(&t.query_set, 0..5, &t.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(
                &t.resolve_buf,
                0,
                &t.readback_buf,
                0,
                (5u64) * 8,
            );
        }
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        if let Some(t) = self.timing.as_mut() {
            t.has_pending = true;
        }
        self.diag.has_pending = true;
        self.dirty = true;
        self.tick += 1;
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn positions(&mut self) -> &[Vec3] {
        if self.dirty {
            self.readback_positions();
            self.dirty = false;
        }
        &self.pos_cache
    }
}

/// Construct a boxed [`CompiledSim`] for the application crate.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(BoidsState::new(seed, agent_count))
}

/// Map a `u32` RNG draw into `[-1.0, 1.0]`. PCG outputs are uniform
/// over `u32::MIN..=u32::MAX`; the centred remap keeps the math
/// branch-free and stable across host/GPU.
fn normalise(raw: u32) -> f32 {
    let half = (u32::MAX / 2) as f32;
    (raw as f32 - half) / half
}
