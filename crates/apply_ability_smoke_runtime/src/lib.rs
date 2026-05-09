//! Per-fixture runtime for `assets/sim/apply_ability_smoke.sim` —
//! task #133 (CPU↔GPU apply_program parity test, P3).
//!
//! ## What this exercises
//!
//! End-to-end pipe of the apply_ability dispatcher kernel:
//!
//! 1. Build a single-program AbilityRegistry on the host
//!    (`AbilityProgram::new_single_target` with one `Damage(30.0)`
//!    EffectOp), pack via `PackedAbilityRegistry::pack`, upload via
//!    `PackedAbilityRegistryGpu::upload`.
//! 2. Allocate per-agent SoA buffers (`agent_alive`, `agent_level`),
//!    seed every slot with `alive=1` and `level=1` (so each agent
//!    dispatches AbilityId(1)).
//! 3. Allocate the event_ring + event_tail buffers and the cfg uniform
//!    that the kernel binds (see the build.rs-emitted bindings
//!    struct for the exact 8-binding shape).
//! 4. Encode + dispatch one tick of `physics_DispatchAbility`. The
//!    kernel's per-agent body reads `agent_level[agent_id]` as the
//!    AbilityId, walks the registry's effect slots, and emits one
//!    chronicle record per chronicle-bearing EffectOp into
//!    `event_ring`. With our single-Damage program, every alive agent
//!    emits exactly 1 record (kind=26 EffectDamageApplied).
//! 5. Read back event_ring + event_tail and compare every record
//!    byte-for-byte against the CPU oracle in
//!    `dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record`.
//!
//! ## Important: caster/target slots are 0-based agent_ids, not 1-based AgentIds
//!
//! The dispatcher kernel uses `gid.x` (the workgroup's per-thread
//! global invocation index, 0-based) as both `caster_slot` and
//! `target_slot` for the implicit-target rule (no `by ... target ...`
//! clause). So with 2 agents, the records will have caster_id=0,1 and
//! target_id=0,1 (matching the agent SoA slot index, NOT the host
//! AgentId raw value). The CPU oracle takes caster_id + target_id as
//! plain u32s, so the parity comparison passes whichever convention
//! the runtime adopts — we just need to feed the SAME values to both
//! sides.
//!
//! ## GPU adapter availability
//!
//! Construction touches the GPU (`GpuContext::new_blocking`).
//! On a host without a wgpu-compatible adapter the constructor
//! panics; the parity test detects this and skips with an
//! explanatory message rather than failing.

use engine::ability::registry::AbilityRegistry;
use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate, PackedAbilityRegistry,
};
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};
use engine::GpuContext;
use wgpu::util::DeviceExt;

/// Per-agent stat snapshot for the smoke runtime. Mirrors the 8-field
/// `engine::ability::program::CasterStats` shape — used by the parity
/// sweep test to seed both the GPU's per-stat agent SoA buffers AND
/// the CPU oracle's `CasterStats` snapshot from the same source.
#[derive(Copy, Clone, Debug, Default)]
pub struct PerAgentStats {
    pub attack_damage: f32,
    pub ability_power: f32,
    pub max_hp:        f32,
    pub hp:            f32,
    pub armor:         f32,
    pub magic_resist:  f32,
    pub move_speed:    f32,
    pub mana:          f32,
}

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-record stride in u32 words — matches
/// `dsl_compiler::cpu_chronicle_reference::CHRONICLE_RECORD_STRIDE_U32`
/// and the engine's `EVENT_STRIDE_U32`. Pinned at 10 (header 2 +
/// payload 8).
pub const CHRONICLE_STRIDE_U32: u32 = 10;

/// Default ring slot capacity — one slot per chronicle record. The
/// smoke fixture only needs a few records per tick, so 256 slots is
/// generous (and far below the 65 536 the production helper allocates).
const RING_SLOTS: u32 = 256;

// Spatial-grid sizing constants. Mirror
// `dsl_compiler::cg::emit::spatial::{CELL_SIZE, WORLD_HALF_EXTENT,
// num_cells}` — the WGSL prelude bakes these into `SPATIAL_*` consts;
// the runtime allocations below have to match in size and grid
// topology or the dispatcher's `cell_index` returns out-of-bounds
// indices and reads garbage. Today the smoke runtime uses CPU-side
// pre-population (not a real BuildHash dispatch), so the only
// constraints are:
//   - `spatial_grid_starts.len() == num_cells + 1`
//   - `spatial_grid_cells.len() >= n_agents`
// Both buffers are populated per-construct from caller-supplied
// agent positions (or default `(0,0,0)` for every agent — every slot
// lands in the world-origin cell, see `try_new_with_registry`).
const SPATIAL_CELL_SIZE_F: f32 = 6.0;
const SPATIAL_WORLD_HALF_EXTENT_F: f32 = 64.0;
const SPATIAL_NUM_CELLS_PER_AXIS: u32 = 22; // ceil(128 / 6)
const SPATIAL_NUM_CELLS: u32 =
    SPATIAL_NUM_CELLS_PER_AXIS * SPATIAL_NUM_CELLS_PER_AXIS * SPATIAL_NUM_CELLS_PER_AXIS;

/// Mirror of the WGSL `pos_to_cell(p: vec3<f32>) -> u32` helper from
/// `cg::emit::spatial::compose_spatial_prelude`. Same clamp semantics
/// (out-of-extent positions snap to the boundary cell).
fn host_pos_to_cell(x: f32, y: f32, z: f32) -> u32 {
    let max_idx = (SPATIAL_NUM_CELLS_PER_AXIS - 1) as i32;
    let cx = ((x + SPATIAL_WORLD_HALF_EXTENT_F) / SPATIAL_CELL_SIZE_F).max(0.0) as i32;
    let cy = ((y + SPATIAL_WORLD_HALF_EXTENT_F) / SPATIAL_CELL_SIZE_F).max(0.0) as i32;
    let cz = ((z + SPATIAL_WORLD_HALF_EXTENT_F) / SPATIAL_CELL_SIZE_F).max(0.0) as i32;
    let cx = cx.clamp(0, max_idx) as u32;
    let cy = cy.clamp(0, max_idx) as u32;
    let cz = cz.clamp(0, max_idx) as u32;
    (cz * SPATIAL_NUM_CELLS_PER_AXIS + cy) * SPATIAL_NUM_CELLS_PER_AXIS + cx
}

/// Per-fixture state for the apply_ability dispatcher smoke test.
/// Owns:
///   - The wgpu context.
///   - Per-agent SoA buffers (`agent_alive`, `agent_level`).
///   - Event-ring + tail buffers (atomic u32 storage).
///   - The packed-registry GPU buffers (uploaded once at
///     construction; immutable thereafter).
///   - Per-kernel cfg uniform.
///   - Pipeline cache.
///
/// `n_agents` is captured from the constructor for the dispatch
/// `agent_cap`. The constructor seeds `alive[*]=1` and `level[*]=1`
/// so every agent dispatches AbilityId(1).
pub struct ApplyAbilitySmokeState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,
    // #121 AOE Path B: agent position SoA. Read by the dispatcher's
    // AOE walk at both `target_slot` (cast center) and per spatial-
    // grid candidate. `vec3<f32>` per slot — 12 bytes (WGSL pads the
    // type to 16 bytes via the storage-buffer alignment rules, so the
    // host buffer is sized 16 bytes per slot).
    agent_pos_buf: wgpu::Buffer,
    // Wave 1.5#4 GPU wire-up: per-stat agent SoA columns the dispatcher
    // reads at `caster_slot` for the `scale_bonus = Σ percent * stat`
    // computation. Initialized to zero — the smoke fixture's program
    // (Damage 30 with no scaling slots) writes zero scale_bonus
    // regardless. Real fixtures (Bleed-style +5% MaxHp) populate these.
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // #182 AOE Path B non-degenerate-direction pin: per-agent
    // `engaged_with` SoA column. Read by the third physics rule
    // (`DispatchAbilityToOther`) at `target = agents.engaged_with(self)`
    // — the dispatcher kernel emits
    //   `let target_slot: u32 = u32(agent_engaged_with[agent_id]);`
    // so the cast's target slot is decoupled from `caster_slot`. With a
    // non-self target, Cone / Line / Wall (and any other direction-bearing
    // shape) form a non-degenerate `apex → target_pos` axis and the GPU
    // walk's spatial filter actually gates candidates instead of
    // collapsing through the `dir_len_sq < 1e-6 → no-op` branch.
    //
    // Encoding: raw u32 holding the 0-based slot index (matches
    // `target_chaser_runtime`'s `engaged_with_init` convention — slot 0
    // points at slot 0, slot 1 points at slot 0, etc.; the kernel reads
    // `agent_engaged_with[caster] as u32` directly without offset). The
    // None-sentinel (`0xFFFFFFFFu`) is the caller's responsibility for
    // any agent the kernel will dispatch from — the smoke fixture only
    // marks the caster alive, so the engaged_with value of the others is
    // dead-store and the kernel's `where (self.alive)` gate prevents the
    // sentinel from feeding into a pos read.
    agent_engaged_with_buf: wgpu::Buffer,

    // #121 AOE Path B: spatial-grid state read by the dispatcher's
    // 27-cell walk. The smoke runtime pre-populates these on the host
    // (no real BuildHash kernel runs in this fixture's schedule); the
    // default layout puts every alive agent in cell 0 (caller can
    // override via `set_agent_positions_in_cell_0`). `grid_starts`
    // holds the inclusive prefix-sum offsets used by the WGSL walk:
    //   spatial_grid_starts[c]   = first slot index in `grid_cells`
    //                              that holds an agent in cell `c`
    //   spatial_grid_starts[c+1] = one past the last slot
    spatial_grid_cells_buf: wgpu::Buffer,
    spatial_grid_starts_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU (only the 3 columns the
    // dispatcher binds are read; the upload helper allocates all
    // columns regardless — wasted bytes are bounded by registry size). --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Event ring + tail (shared infrastructure via `EventRing` so the
    //    compiler-emitted `Bindings::from_context_with_extras` constructor
    //    can resolve `event_ring` / `event_tail` through the standard
    //    `KernelBindingsContext`). --
    event_ring: EventRing,
    /// Staging buffer for `read_event_ring` host readback. Sized
    /// to the standard `EventRing` ring capacity.
    event_ring_staging: wgpu::Buffer,
    /// Staging buffer for `read_event_tail` host readback (4 bytes).
    event_tail_staging: wgpu::Buffer,

    // -- Cfg uniform --
    physics_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
}

impl ApplyAbilitySmokeState {
    /// Construct a smoke runtime with `n_agents` slots, an
    /// AbilityRegistry holding ONE program at AbilityId(1) (a single
    /// `Damage(30.0)` EffectOp), and `agent_level[*] = 1` so each
    /// alive agent dispatches AbilityId(1) when stepped.
    ///
    /// The constructor blocks on `GpuContext::new_blocking()`, so
    /// callers without a GPU adapter will receive a panic at this
    /// point. Tests that need to skip in that case should use
    /// `try_new` (see below).
    pub fn new(n_agents: u32) -> Self {
        Self::try_new(n_agents).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — returns `None` when no compatible wgpu
    /// adapter is available on the host. Lets the parity test in this
    /// crate degrade to a skip-with-message instead of a panic.
    pub fn try_new(n_agents: u32) -> Option<Self> {
        let program = AbilityProgram::new_single_target(
            /*range*/ 5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let mut builder = AbilityRegistryBuilder::new();
        let id = builder.register(program);
        debug_assert_eq!(
            id,
            AbilityId::new(1).unwrap(),
            "first registered program must land at AbilityId(1)"
        );
        let registry = builder.build();
        Self::try_new_with_registry(
            n_agents,
            &registry,
            /*per_agent_levels*/ &vec![1u32; n_agents as usize],
            /*per_agent_stats*/  &vec![PerAgentStats::default(); n_agents as usize],
        )
    }

    /// Build the smoke fixture against a caller-supplied
    /// `AbilityRegistry`, with explicit per-agent `agent_level`
    /// (= AbilityId.raw_u32 to dispatch) and per-agent stat snapshots.
    ///
    /// Used by the parity sweep test (`tests/parity_apply_program_sweep.rs`)
    /// to upload a 10-program registry once, then arrange `agent_level[i]`
    /// to point at one of the registered ability slots per agent. The
    /// dispatcher reads `agent_level[caster_slot]` to pick the AbilityId,
    /// so varying `level` per agent dispatches a different program per
    /// SoA slot in a single tick — the natural matrix shape for sweep
    /// testing every modifier × variant combination.
    ///
    /// Stats land in the per-stat agent SoA columns the dispatcher's
    /// `scale_bonus = Σ percent * agent_stat[caster_slot]` switch reads;
    /// pass `&vec![PerAgentStats::default(); n]` for non-scaling fixtures.
    /// Returns `None` when no wgpu adapter is available on the host.
    pub fn try_new_with_registry(
        n_agents:         u32,
        registry:         &AbilityRegistry,
        per_agent_levels: &[u32],
        per_agent_stats:  &[PerAgentStats],
    ) -> Option<Self> {
        assert_eq!(
            per_agent_levels.len(),
            n_agents as usize,
            "per_agent_levels length must match n_agents",
        );
        assert_eq!(
            per_agent_stats.len(),
            n_agents as usize,
            "per_agent_stats length must match n_agents",
        );
        let gpu = GpuContext::new_blocking().ok()?;
        let packed = PackedAbilityRegistry::pack(registry);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "apply_ability_smoke");

        // -- Agent SoA: alive=1 for every slot; per-agent level + stats
        //    from the caller. The kernel's per-agent body reads
        //    `agent_alive[agent_id]` as the where-clause gate and
        //    `agent_level[caster_slot]` as the AbilityId — varying level
        //    per agent dispatches a different ability per SoA slot in
        //    one tick (the matrix shape the parity sweep test relies on).
        let alive_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let agent_level_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::agent_level"),
                contents: bytemuck::cast_slice(per_agent_levels),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // #121 AOE: agent_pos SoA. WGSL-side `array<vec3<f32>>` reads
        // through 16-byte-stride alignment (vec3 in storage buffers
        // pads to vec4); allocate `n_agents * 16` bytes and zero-init.
        // Default position (0,0,0) lands every agent in cell 0 of the
        // spatial grid, matching the default `spatial_grid_*` layout
        // below. Callers wanting non-zero positions overwrite via
        // `set_agent_positions(&[Vec3])`.
        let agent_pos_init = vec![0u32; (n_agents as usize) * 4];
        let agent_pos_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::agent_pos"),
                contents: bytemuck::cast_slice(&agent_pos_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // #121 AOE: spatial-grid state. The smoke fixture pre-populates
        // these on the host instead of running a real BuildHash kernel
        // (the dispatcher's needs are bounded — 27-cell walk around the
        // cast center — so a hand-rolled layout suffices for testing).
        //
        // Default agent positions are all `(0,0,0)`, which maps to one
        // specific cell via the WGSL `pos_to_cell` helper (mirrored on
        // the host as `host_pos_to_cell`). Initialise the grid with
        // every alive slot landing in that cell:
        //   grid_cells = [0, 1, 2, …, n_agents - 1] in slots 0..n_agents
        //   grid_starts[c] = 0           for c <= origin_cell
        //   grid_starts[c] = n_agents    for c >  origin_cell
        // (inclusive-prefix layout — `grid_starts[c+1] - grid_starts[c]`
        // gives the count in cell `c`).
        //
        // Callers that override positions via `set_agent_positions`
        // also rebuild the grid via `set_spatial_grid_for_positions`
        // — without that, the spatial walk reads stale offsets.
        let origin_cell = host_pos_to_cell(0.0, 0.0, 0.0);
        let mut grid_cells_init: Vec<u32> = (0..n_agents).collect();
        // Pad to at least 1 entry so wgpu accepts the zero-size buffer
        // case (n_agents == 0 isn't supported but defensive).
        if grid_cells_init.is_empty() {
            grid_cells_init.push(0);
        }
        let spatial_grid_cells_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::spatial_grid_cells"),
                contents: bytemuck::cast_slice(&grid_cells_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // grid_starts has `num_cells + 1` entries (inclusive prefix
        // counts). With every alive agent sitting in `origin_cell`:
        //   grid_starts[c]   = 0          for c <= origin_cell
        //   grid_starts[c]   = n_agents   for c >  origin_cell
        // Cell `origin_cell` has `n_agents` entries (slots 0..n_agents
        // in `grid_cells`); every other cell is empty.
        let mut grid_starts_init: Vec<u32> =
            Vec::with_capacity((SPATIAL_NUM_CELLS as usize) + 1);
        for c in 0..=SPATIAL_NUM_CELLS {
            grid_starts_init.push(if c <= origin_cell { 0 } else { n_agents });
        }
        let spatial_grid_starts_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::spatial_grid_starts"),
                contents: bytemuck::cast_slice(&grid_starts_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Wave 1.5#4 GPU scaling: per-stat columns for the dispatcher's
        // `agent_stat()` switch. The parity sweep seeds these from
        // `per_agent_stats[i].<field>` so the GPU and the CPU oracle
        // (passing `CasterStats { … }` to apply_program) read the same
        // f32 values.
        let mk_stat_col = |label: &str, extract: fn(&PerAgentStats) -> f32| {
            let col: Vec<f32> = per_agent_stats.iter().map(extract).collect();
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&col),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf =
            mk_stat_col("apply_ability_smoke_runtime::agent_attack_damage", |s| s.attack_damage);
        let agent_ability_power_buf =
            mk_stat_col("apply_ability_smoke_runtime::agent_ability_power", |s| s.ability_power);
        let agent_max_hp_buf        =
            mk_stat_col("apply_ability_smoke_runtime::agent_max_hp",        |s| s.max_hp);
        let agent_hp_buf            =
            mk_stat_col("apply_ability_smoke_runtime::agent_hp",            |s| s.hp);
        let agent_armor_buf         =
            mk_stat_col("apply_ability_smoke_runtime::agent_armor",         |s| s.armor);
        let agent_magic_resist_buf  =
            mk_stat_col("apply_ability_smoke_runtime::agent_magic_resist",  |s| s.magic_resist);
        let agent_move_speed_buf    =
            mk_stat_col("apply_ability_smoke_runtime::agent_move_speed",    |s| s.move_speed);
        let agent_mana_buf          =
            mk_stat_col("apply_ability_smoke_runtime::agent_mana",          |s| s.mana);
        // #182: per-agent `engaged_with` column. Default-init to the
        // caster's own slot for every agent (raw u32 = slot index). The
        // explicit-target physics rule reads
        // `agent_engaged_with[caster]` to drive `target_slot`; callers
        // override the caster's entry via `set_agent_engaged_with` to
        // dispatch at a non-self target. Default-self matches the
        // existing degenerate-cone test fixture's behavior so that
        // dispatching the new kernel without overrides reproduces the
        // self-cast (degenerate) outcome — useful as the default
        // behavioral floor; non-degenerate tests must opt in.
        let engaged_with_init: Vec<u32> = (0..n_agents).collect();
        let agent_engaged_with_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::agent_engaged_with"),
                contents: bytemuck::cast_slice(&engaged_with_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // -- Event ring + tail. Standard `EventRing` so the compiler-
        //    emitted `Bindings::from_context_with_extras` constructor
        //    can resolve the `event_ring` / `event_tail` bindings via
        //    `KernelBindingsContext::event_ring`.
        let event_ring = EventRing::new(&gpu, "apply_ability_smoke_runtime");
        // Staging buffer sized for the smoke fixture's worst-case readback
        // (a few records per tick — the per-tick smoke records are bounded
        // by `RING_SLOTS`). `read_event_ring` only copies the actually-
        // requested record count, not the full 1M-slot ring.
        let staging_bytes = (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let event_ring_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_smoke_runtime::event_ring_staging"),
            size: staging_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_smoke_runtime::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniform. Compiler-emitted struct shape is
        //    `{ agent_cap, tick, seed, _pad }` (4 × u32). We seed it
        //    once at construction; `step()` overwrites tick before
        //    every dispatch.
        let cfg_init = physics_DispatchAbility::PhysicsDispatchAbilityCfg {
            agent_cap: n_agents,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_level_buf,
            agent_pos_buf,
            agent_attack_damage_buf,
            agent_ability_power_buf,
            agent_max_hp_buf,
            agent_hp_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            agent_mana_buf,
            agent_engaged_with_buf,
            spatial_grid_cells_buf,
            spatial_grid_starts_buf,
            registry_gpu,
            event_ring,
            event_ring_staging,
            event_tail_staging,
            physics_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
        })
    }

    /// Overwrite per-agent alive flags. Used by the AOE parity sweep
    /// to dispatch from a single caster while keeping the other agents
    /// in the SoA as quiescent AOE targets (their `agent_alive == 0`
    /// gates them out of the per-agent dispatch loop's `where
    /// (self.alive)` clause but they still appear in the spatial grid
    /// for the caster's AOE walk).
    pub fn set_agent_alive(&self, alive: &[u32]) {
        assert_eq!(
            alive.len(),
            self.n_agents as usize,
            "alive slice must have one entry per agent (got {} for {} agents)",
            alive.len(),
            self.n_agents,
        );
        self.gpu.queue.write_buffer(
            &self.agent_alive_buf,
            0,
            bytemuck::cast_slice(alive),
        );
    }

    /// Overwrite per-agent world positions in the `agent_pos` SoA AND
    /// rebuild the spatial grid (`grid_cells` + `grid_starts`) to
    /// match. The dispatcher's AOE walk reads `grid_starts[cell..+1]`
    /// for each cell in the 27-neighborhood; without rebuilding, the
    /// walk reads stale offsets from the constructor's "all in
    /// origin cell" layout and the in-circle set is wrong.
    ///
    /// Used by the AOE behavioral pin (`tests/aoe_chronicle_pin.rs`)
    /// and the AOE parity sweep (`parity_apply_program_sweep.rs`) to
    /// set up a row of agents at known (x, 0, 0) coordinates so the
    /// dispatcher's spatial walk produces the expected per-target
    /// chronicle records. Each entry writes 16 bytes to the agent_pos
    /// buffer (vec3 padded to vec4 for storage-buffer alignment) at
    /// byte offset `i * 16`.
    pub fn set_agent_positions(&self, positions: &[[f32; 3]]) {
        assert_eq!(
            positions.len(),
            self.n_agents as usize,
            "positions slice must have one entry per agent (got {} for {} agents)",
            positions.len(),
            self.n_agents,
        );
        // Pack into vec4-padded layout (storage-buffer alignment for
        // `array<vec3<f32>>`).
        let mut padded: Vec<f32> = Vec::with_capacity(positions.len() * 4);
        for &[x, y, z] in positions {
            padded.push(x);
            padded.push(y);
            padded.push(z);
            padded.push(0.0);
        }
        self.gpu.queue.write_buffer(
            &self.agent_pos_buf,
            0,
            bytemuck::cast_slice(&padded),
        );

        // Rebuild the spatial grid so each agent lands in the cell its
        // new position maps to. Bucket agents by cell, then write a
        // counting-sort layout:
        //   grid_cells[grid_starts[c] + k] = AgentId of the k-th agent
        //                                    in cell c (0-based)
        //   grid_starts[c]   = inclusive start of cell c
        //   grid_starts[c+1] = inclusive end of cell c
        //
        // Agents within a cell are stored in AgentId-ascending order
        // (the host loop iterates in slot order and pushes monotonically
        // into the bucket vec) — matches `state.spatial().within_radius`'s
        // sort-ascending contract on the CPU oracle side.
        let mut buckets: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for (slot, &[x, y, z]) in positions.iter().enumerate() {
            let cell = host_pos_to_cell(x, y, z);
            buckets.entry(cell).or_default().push(slot as u32);
        }
        // Bucket counts (per cell, 0 elsewhere). BTreeMap → walk in
        // ascending cell order to flatten into grid_cells.
        let mut grid_cells: Vec<u32> = Vec::with_capacity(self.n_agents as usize);
        for (_cell, ids) in &buckets {
            grid_cells.extend(ids.iter().copied());
        }
        let total = grid_cells.len() as u32;
        // Pad grid_cells to at least 1 entry (defensive — wgpu rejects zero-size).
        if grid_cells.is_empty() {
            grid_cells.push(0);
        }
        // grid_starts[c] = inclusive prefix sum = number of agents in
        // cells [0..c). The dispatcher's walk reads `grid_starts[cell]`
        // and `grid_starts[cell + 1]` to bracket cell `cell`'s agent
        // slots in `grid_cells`.
        let mut counts: Vec<u32> = vec![0u32; SPATIAL_NUM_CELLS as usize];
        for (&cell, ids) in &buckets {
            counts[cell as usize] = ids.len() as u32;
        }
        let mut grid_starts: Vec<u32> =
            Vec::with_capacity((SPATIAL_NUM_CELLS as usize) + 1);
        let mut running: u32 = 0;
        grid_starts.push(0);
        for &c in &counts {
            running += c;
            grid_starts.push(running);
        }
        debug_assert_eq!(running, total);

        self.gpu.queue.write_buffer(
            &self.spatial_grid_cells_buf,
            0,
            bytemuck::cast_slice(&grid_cells),
        );
        self.gpu.queue.write_buffer(
            &self.spatial_grid_starts_buf,
            0,
            bytemuck::cast_slice(&grid_starts),
        );
    }

    /// Overwrite per-agent `engaged_with` slots. Used by the
    /// explicit-target AOE pins (`tests/aoe_chronicle_pin.rs::aoe_*_
    /// non_degenerate_*`) to seed the caster's engagement target so the
    /// `DispatchAbilityToOther` kernel computes a non-self `target_slot
    /// = agent_engaged_with[caster_slot]`. With a real non-self target,
    /// the apex→target_pos direction is non-degenerate and the Cone /
    /// Line / Wall walks gate candidates by their actual angular /
    /// corridor predicate instead of short-circuiting through the
    /// `dir_len_sq < 1e-6 → no-op` branch.
    ///
    /// Encoding: raw u32 = 0-based slot index. Mirror of
    /// `target_chaser_runtime`'s `engaged_with_init` convention — the
    /// kernel reads `agent_engaged_with[caster_slot]` and feeds it
    /// directly into `agent_pos[target_slot]` / spatial walks. Caller
    /// MUST keep the value in `[0, n_agents)` for any caster slot the
    /// kernel will run on (the `where (self.alive)` gate prevents reads
    /// for dead agents — set their engaged_with however you like, it's
    /// dead-store).
    pub fn set_agent_engaged_with(&self, engaged_with: &[u32]) {
        assert_eq!(
            engaged_with.len(),
            self.n_agents as usize,
            "engaged_with slice must have one entry per agent (got {} for {} agents)",
            engaged_with.len(),
            self.n_agents,
        );
        self.gpu.queue.write_buffer(
            &self.agent_engaged_with_buf,
            0,
            bytemuck::cast_slice(engaged_with),
        );
    }

    /// Encode + dispatch one tick of `physics_DispatchAbility`.
    /// The encoder also clears `event_tail` to zero before the
    /// dispatch (so producers atomicAdd from 0).
    pub fn step(&mut self, tick: u32) {
        self.step_with_seed(tick, 0);
    }

    /// Encode + dispatch one tick with an explicit `seed` low-32-bit
    /// value. Mirrors the cfg uniform's `seed: u32` field — feeds
    /// the chance gate's `per_agent_u32_with_extra` draw inside the
    /// dispatcher kernel, so callers can pin the host's
    /// `world_seed as u32` and the GPU's `cfg.seed` to identical
    /// values for cross-backend parity (P11). Used by the
    /// `parity_apply_program_sweep` test.
    pub fn step_with_seed(&mut self, tick: u32, seed: u32) {
        let cfg = physics_DispatchAbility::PhysicsDispatchAbilityCfg {
            agent_cap: self.n_agents,
            tick,
            seed,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::step"),
                });

        // Clear event_tail to 0 so producer atomicAdd starts from slot 0.
        self.event_ring.clear_tail_in(&mut encoder);

        let agent_buffers = AgentBuffers {
            alive_buf: Some(&self.agent_alive_buf),
            level_buf: Some(&self.agent_level_buf),
            pos_buf: Some(&self.agent_pos_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            max_hp_buf: Some(&self.agent_max_hp_buf),
            hp_buf: Some(&self.agent_hp_buf),
            armor_buf: Some(&self.agent_armor_buf),
            magic_resist_buf: Some(&self.agent_magic_resist_buf),
            move_speed_buf: Some(&self.agent_move_speed_buf),
            mana_buf: Some(&self.agent_mana_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
            voxel_grid: None,
        };
        let extras = physics_DispatchAbility::PhysicsDispatchAbilityExtras {
            spatial_grid_cells:  &self.spatial_grid_cells_buf,
            spatial_grid_starts: &self.spatial_grid_starts_buf,
            cfg: &self.physics_cfg_buf,
        };
        let bindings =
            physics_DispatchAbility::PhysicsDispatchAbilityBindings::from_context_with_extras(
                &ctx, &extras,
            );
        dispatch::dispatch_physics_dispatchability(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.n_agents,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
    }

    /// Encode + dispatch one tick of `physics_DispatchAbilityToOther`
    /// — the third physics rule whose `target` operand reads
    /// `agents.engaged_with(self)`. The dispatcher kernel emits
    /// `target_slot = u32(agent_engaged_with[caster_slot])`, decoupling
    /// the target slot from the caster slot. With a non-self target
    /// seeded via `set_agent_engaged_with`, AOE shapes that gate on
    /// apex→target direction (Cone / Line / Wall) hit a non-degenerate
    /// branch; the GPU walk then filters candidates by their actual
    /// angular / corridor predicate instead of short-circuiting.
    ///
    /// Mirrors `step_with_seed` for the ring-clear + cfg uniform write,
    /// just routes through the new kernel's dispatch fn (which has the
    /// extra `agent_engaged_with` binding). Same `cfg.seed` shape →
    /// chance gates draw bit-for-bit identically across the two
    /// kernels (the seed flows through `per_agent_u32_with_extra`
    /// keyed on `caster_slot`, not on the kernel id).
    pub fn step_explicit_target_with_seed(&mut self, tick: u32, seed: u32) {
        let cfg = physics_DispatchAbilityToOther::PhysicsDispatchAbilityToOtherCfg {
            agent_cap: self.n_agents,
            tick,
            seed,
            _pad: 0,
        };
        // `physics_cfg_buf` is shared across kernels (same 16-byte
        // `{agent_cap, tick, seed, _pad}` shape) — overwrite it with
        // this kernel's cfg before recording.
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::step_explicit_target"),
                });

        self.event_ring.clear_tail_in(&mut encoder);

        let agent_buffers = AgentBuffers {
            alive_buf: Some(&self.agent_alive_buf),
            level_buf: Some(&self.agent_level_buf),
            pos_buf: Some(&self.agent_pos_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            max_hp_buf: Some(&self.agent_max_hp_buf),
            hp_buf: Some(&self.agent_hp_buf),
            armor_buf: Some(&self.agent_armor_buf),
            magic_resist_buf: Some(&self.agent_magic_resist_buf),
            move_speed_buf: Some(&self.agent_move_speed_buf),
            mana_buf: Some(&self.agent_mana_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
            voxel_grid: None,
        };
        let extras =
            physics_DispatchAbilityToOther::PhysicsDispatchAbilityToOtherExtras {
                agent_engaged_with: &self.agent_engaged_with_buf,
                spatial_grid_cells:  &self.spatial_grid_cells_buf,
                spatial_grid_starts: &self.spatial_grid_starts_buf,
                cfg: &self.physics_cfg_buf,
            };
        let bindings =
            physics_DispatchAbilityToOther::PhysicsDispatchAbilityToOtherBindings::from_context_with_extras(
                &ctx, &extras,
            );
        dispatch::dispatch_physics_dispatchabilitytoother(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.n_agents,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
    }

    /// Convenience wrapper for `step_explicit_target_with_seed(tick, 0)`.
    pub fn step_explicit_target(&mut self, tick: u32) {
        self.step_explicit_target_with_seed(tick, 0);
    }

    /// Block on the GPU and read back `event_tail`. Returns the
    /// number of chronicle records written by the most-recent
    /// `step()`.
    pub fn read_event_tail(&self) -> u32 {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::read_event_tail"),
                });
        encoder.copy_buffer_to_buffer(self.event_ring.tail(), 0, &self.event_tail_staging, 0, 4);
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = self.event_tail_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("event_tail_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during event_tail readback");

        let value = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            words[0]
        };
        self.event_tail_staging.unmap();
        value
    }

    /// Block on the GPU and read back the first `n_records` records
    /// from `event_ring`. Each record is 10 u32 words.
    pub fn read_event_ring(&self, n_records: u32) -> Vec<[u32; 10]> {
        // Copy only the bytes covering `n_records` (capped at the
        // staging buffer size) — the host then walks the staging slice
        // word-by-word.
        let want_bytes =
            (n_records as u64).max(1) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let staging_cap_bytes =
            (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let copy_bytes = want_bytes.min(staging_cap_bytes);
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::read_event_ring"),
                });
        encoder.copy_buffer_to_buffer(
            self.event_ring.ring(),
            0,
            &self.event_ring_staging,
            0,
            copy_bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = self.event_ring_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("event_ring_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during event_ring readback");

        let records = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            // Pull the first `n_records` records out of the flat slice.
            let mut out = Vec::with_capacity(n_records as usize);
            for r in 0..(n_records as usize) {
                let base = r * (CHRONICLE_STRIDE_U32 as usize);
                let mut rec = [0u32; 10];
                rec.copy_from_slice(&words[base..base + 10]);
                out.push(rec);
            }
            out
        };
        self.event_ring_staging.unmap();
        records
    }

    pub fn n_agents(&self) -> u32 {
        self.n_agents
    }
}

/// Sort chronicle records by `(target_slot, kind)` so the host-side
/// comparison is order-stable. The dispatcher uses `atomicAdd` to
/// claim slots, so the records' relative order in the ring depends
/// on workgroup scheduling — comparing as a sorted set sidesteps
/// that nondeterminism for the smoke parity check.
#[cfg(test)]
fn canonicalize(records: &mut [[u32; 10]]) {
    records.sort_by_key(|r| (r[3], r[0]));
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record;
    use engine::ability::apply::apply_program;
    use engine::ability::program::{AbilityProgram, CasterStats, EffectOp, Gate};
    use engine::ids::AgentId;

    /// End-to-end CPU↔GPU parity test (#133). Closes the loop the
    /// `cpu_chronicle_pipeline` test set leaves open: the GPU side
    /// finally has a runtime crate driving the dispatcher kernel,
    /// so the comparison can run.
    ///
    /// **Skip path.** When `GpuContext::new_blocking` returns Err
    /// (no compatible wgpu adapter on the host — common on CI without
    /// a software-rendering fallback), the test prints a skip message
    /// and returns Ok. The build itself still validated the kernel
    /// emit + the binding hookup at compile time, so the skip is
    /// noisy-but-safe.
    #[test]
    fn gpu_chronicle_records_match_cpu_oracle_for_2_agents_1_damage_effect() {
        let n_agents: u32 = 2;
        let tick: u32 = 0;

        let mut state = match ApplyAbilitySmokeState::try_new(n_agents) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[apply_ability_smoke parity] skipping: no wgpu adapter \
                     available on this host. The build itself still validates \
                     the kernel emit (apply_ability_smoke.sim → WGSL) and the \
                     8-binding hookup at compile time."
                );
                return;
            }
        };

        // 1. Run one tick of the dispatcher. With agent_level[*]=1 +
        //    one Damage(30.0) op at AbilityId(1) + one chronicle-bearing
        //    EffectOp per program slot, every alive agent emits
        //    EXACTLY one record (kind=26 EffectDamageApplied).
        state.step(tick);

        // 2. Read back the tail count + records.
        let tail = state.read_event_tail();
        assert_eq!(
            tail, n_agents,
            "expected one chronicle record per alive agent (n={n_agents}); \
             got tail={tail}"
        );
        let mut gpu_records = state.read_event_ring(tail);
        canonicalize(&mut gpu_records);

        // 3. Build the CPU oracle: same program, same caster/target
        //    for each agent. The dispatcher writes `agent_id` (= 0-based
        //    SoA slot) into both caster_slot and target_slot for the
        //    implicit-target rule, so the oracle uses the slot index
        //    as both (NOT the 1-based AgentId raw value).
        let program = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let mut cpu_records: Vec<[u32; 10]> = Vec::with_capacity(n_agents as usize);
        for slot in 0..n_agents {
            // apply_program takes 1-based AgentIds — `AgentId::new` rejects
            // zero — but the chronicle record we're comparing against uses
            // the 0-based slot the GPU writes. apply_program emits one
            // ApplyEvent per chronicle-bearing EffectOp; we feed it (1+slot)
            // for the AgentId nichefulness, then pass `slot` as caster_id
            // and target_id into the chronicle reference (which the GPU
            // dispatcher writes 0-based via gid.x).
            let aid = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            let events = apply_program(
                &program,
                /*caster*/ aid,
                /*target*/ aid,
                tick as u64,
                /*world_seed*/ 0,
                &CasterStats::default(),
                &CasterStats::default(),
            );
            for ev in events {
                if let Some(rec) = apply_event_to_chronicle_record(
                    ev,
                    tick,
                    /*caster_id*/ slot,
                    /*target_id*/ slot,
                ) {
                    cpu_records.push(rec);
                }
            }
        }
        canonicalize(&mut cpu_records);

        // 4. Byte-for-byte parity assert.
        assert_eq!(
            gpu_records.len(),
            cpu_records.len(),
            "record count mismatch: gpu={} cpu={}",
            gpu_records.len(),
            cpu_records.len(),
        );
        for (i, (g, c)) in gpu_records.iter().zip(cpu_records.iter()).enumerate() {
            assert_eq!(
                g, c,
                "record {i}: GPU={g:?} CPU={c:?} (kind={:?}, tick={:?}, \
                 caster={:?}, target={:?}, payload[4]={:?})",
                g[0], g[1], g[2], g[3], g[4],
            );
        }
    }
}
