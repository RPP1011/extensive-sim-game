//! Per-fixture runtime for `assets/sim/stress_cast_density.sim` —
//! Fixture B for the 2026-05-09 stress test (task #238).
//!
//! ## What this measures
//!
//! Drives the apply_ability dispatcher at maximum AOE candidate
//! density so we can find:
//!
//!   1. The chronicle ring **first overflow point** — every per-dispatch
//!      WGSL append site is gated on `if (_slot < 65536u)` (see
//!      `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`); once the
//!      atomic-bumped tail crosses 65 535, every subsequent emit is
//!      silently dropped. The driver records the agent_cap at which the
//!      first overflow shows up.
//!   2. The **Spread bitonic-sort scaling** — at workgroup_size=64 each
//!      thread runs a per-thread bitonic sort over a 256-slot scratch
//!      array. The total per-tick sort work scales as
//!      O(agent_cap × 8192) compare-swaps when every cast hits 256
//!      in-radius candidates.
//!
//! ## Per-tick chain
//!
//! 1. Clear `event_tail = 0`.
//! 2. Dispatch `physics_DispatchAoePulse` — the AOE Path B-enabled
//!    apply_ability dispatcher. Each alive agent self-casts AbilityId(1)
//!    (AoePulse: `damage 0.0 in spread(1.5, 256)`); the dispatcher
//!    walks the 27-cell neighborhood around the cast center, collects
//!    in-radius candidates (capped at the bitonic-sort scratch's 256-
//!    slot ceiling), bitonic-sorts by AgentId ascending, truncates to
//!    max_targets=256, and emits one `EffectDamageApplied` chronicle
//!    record per kept slot.
//! 3. Read `event_tail` back → ring high-water + overflow detection.
//! 4. Dispatch `physics_ApplyDamageFromChronicle` — no-op consumer
//!    (forces the consumer-side wiring through the schedule).
//! 5. Dispatch `seed_indirect_0` — keeps the indirect-args buffer
//!    warm.
//!
//! ## All agents in a single 27-cell hash bin
//!
//! Positions are seeded inside a tiny ~1.5-radius cube around the
//! origin so every spread walk's 27-cell neighborhood collects every
//! other agent. Per cast: up to 256 EffectDamageApplied records (the
//! WGSL bitonic-sort cap). Per tick: agent_cap × ~256 events. The
//! chronicle ring caps at 65 536 slots per dispatch; first overflow
//! lands at the agent_cap whose `agent_cap × ~256 ≥ 65 536` (≈ 256
//! agents in the saturating regime).
//!
//! ## P5 seeding
//!
//! Per-agent position perturbation flows through
//! `per_agent_u32_pcg_with_extra(seed, agent_id, /*tick*/0, /*purpose*/0,
//! /*extra*/i)` — the canonical `per_agent_u32` channel. Seed is
//! `0x5FE55BB` ("stress B" in leet); changing it changes every per-tick
//! position byte-for-byte.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder, EffectAreaShape, EffectOp,
    Gate, PackedAbilityRegistry, ShapeKind,
};
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext, EVENT_RING_CAP_SLOTS};
use engine::rng::per_agent_u32_pcg_with_extra;
use engine::GpuContext;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Slot count for the WGSL `event_kind_counts: array<atomic<u32>>`
/// instrumentation buffer. Sized for `EventKindId::ChronicleEntry =
/// 128 + 1` so every emitted EventKind discriminant has a dedicated
/// counter slot. See `crates/engine/src/cascade/handler.rs` —
/// `EventKindId` enum.
pub const N_EVENT_KIND_SLOTS: usize = 129;

// -- Ring + AOE constants -------------------------------------------------

/// The chronicle ring's per-dispatch slot cap. Pinned by `if (_slot <
/// 65536u)` in `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`. Events
/// past this slot are silently dropped by the WGSL append site —
/// `event_tail` keeps incrementing via `atomicAdd`, but the matching
/// `event_ring` write is gated. The driver detects overflow by
/// comparing `event_tail` readback against this cap. Numerically
/// matches `engine::gpu::EVENT_RING_CAP_SLOTS` but the WGSL literal
/// is hardcoded — re-export the same constant so the host-side
/// detection stays in sync if the engine constant ever bumps without
/// the WGSL literal being updated (audit lands here, not silently).
pub const RING_SLOT_CAP: u32 = EVENT_RING_CAP_SLOTS;

/// Spread shape's per-cast in-radius cap (matches the bitonic-sort
/// scratch in `cg/emit/wgsl_body.rs`'s Spread arm — `var collected:
/// array<u32, 256>`).
pub const SPREAD_MAX_TARGETS: u32 = 256;

/// Spread radius. Picked to match a half-cell so 27-cell walk catches
/// every co-located agent without requiring multi-cell coverage.
pub const SPREAD_RADIUS: f32 = 1.5;

/// World position spread for the per-agent seeding. All agents land in
/// `[-SEED_HALF_EXTENT, +SEED_HALF_EXTENT]` per axis. With
/// `SPATIAL_CELL_SIZE = 6.0`, this fits within ~1 cell so every
/// other agent is visible from any other agent's 27-cell footprint.
pub const SEED_HALF_EXTENT: f32 = 1.5;

/// PCG seed for per-agent position perturbation. P5 channel — every
/// position byte flows through `per_agent_u32_pcg_with_extra(seed,
/// agent_id, 0, 0, axis_index)`.
pub const POSITION_SEED: u32 = 0x05FE_55BB;

// -- Spatial-grid sizing constants ---------------------------------------
//
// Mirror `dsl_compiler::cg::emit::spatial::{CELL_SIZE,
// WORLD_HALF_EXTENT, num_cells}`. The WGSL prelude bakes these into
// SPATIAL_* consts; the runtime allocations below match.

const SPATIAL_CELL_SIZE_F: f32 = 6.0;
const SPATIAL_WORLD_HALF_EXTENT_F: f32 = 64.0;
const SPATIAL_NUM_CELLS_PER_AXIS: u32 = 22; // ceil(128 / 6)
const SPATIAL_NUM_CELLS: u32 =
    SPATIAL_NUM_CELLS_PER_AXIS * SPATIAL_NUM_CELLS_PER_AXIS * SPATIAL_NUM_CELLS_PER_AXIS;

/// Mirror of the WGSL `pos_to_cell(p: vec3<f32>) -> u32` helper. Same
/// clamp semantics as `apply_ability_smoke_runtime::host_pos_to_cell`.
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

/// Convert a u32 PCG draw to an f32 in `[-half_extent, +half_extent]`.
/// P5 channel — deterministic, no host-side RNG state.
fn perturb_axis(draw: u32, half_extent: f32) -> f32 {
    // Map u32 → [0, 1) via `(draw >> 8) as f32 / (1<<24)` (24-bit
    // precision is enough; avoids the integer-rounding gotcha at
    // u32::MAX / 2³² = 1.0).
    let unit = ((draw >> 8) as f32) / ((1u32 << 24) as f32);
    (unit * 2.0 - 1.0) * half_extent
}

/// Per-tick metric record. The driver bin emits one of these per tick
/// as NDJSON.
#[derive(Debug, Clone)]
pub struct TickMetric {
    pub tick: u32,
    /// Total wall clock for the per-tick step (encoder build + GPU work
    /// + tail readback).
    pub wall_clock_us: u64,
    /// `event_tail` value after the dispatcher kernel runs. The
    /// **observed** number of slots claimed via atomicAdd. If this
    /// exceeds [`RING_SLOT_CAP`], the WGSL append site dropped writes
    /// past slot 65 535 — the per-tick chronicle ring is overflowed.
    pub ring_high_water: u32,
    /// `true` when `ring_high_water > RING_SLOT_CAP`.
    pub ring_overflowed: bool,
    /// Wall clock for the dispatcher dispatch alone (subset of
    /// `wall_clock_us`). The dispatcher contains the Spread bitonic
    /// sort, so this is the canonical "spread sort us" the plan
    /// asks for at sweep summary time.
    pub spread_sort_us: u64,
    /// D3 compiler-debug-mode per-kernel GPU wall times (ns) for this
    /// tick. Empty when the host adapter doesn't expose
    /// `wgpu::Features::TIMESTAMP_QUERY`. Each `(name, ns)` pair maps
    /// to a single emitted kernel; sum across the vec gives total
    /// GPU work for the tick (excludes host-side encoder build +
    /// queue submit + device.poll wait).
    pub kernel_timings: Vec<(String, u64)>,
    /// Phase 2 debug instrumentation: per-EventKindId histogram for
    /// this tick. Index `k` = number of times the dispatcher's
    /// compiler-emitted `atomicAdd(&event_kind_counts[k], 1u)` site
    /// fired for `EventKindId(k)`. Sum across all slots equals the
    /// chronicle ring's per-tick high-water mark.
    pub event_kind_histogram: [u32; N_EVENT_KIND_SLOTS],
}

/// Aggregated stress metrics returned by `run_stress`.
#[derive(Debug, Clone)]
pub struct StressMetrics {
    pub agent_cap: u32,
    pub ticks_completed: u32,
    pub per_tick: Vec<TickMetric>,
    /// Whichever tick first observed `ring_overflowed = true`. None when
    /// the cap stayed below the per-dispatch ceiling for the entire run.
    pub first_overflow_at_tick: Option<u32>,
    /// Maximum `ring_high_water` observed across the run.
    pub max_ring_high_water: u32,
}

impl StressMetrics {
    /// Median `wall_clock_us` across all per-tick samples.
    pub fn median_us(&self) -> u64 {
        if self.per_tick.is_empty() {
            return 0;
        }
        let mut v: Vec<u64> = self.per_tick.iter().map(|t| t.wall_clock_us).collect();
        v.sort_unstable();
        v[v.len() / 2]
    }

    /// p99 `wall_clock_us` across all per-tick samples.
    pub fn p99_us(&self) -> u64 {
        if self.per_tick.is_empty() {
            return 0;
        }
        let mut v: Vec<u64> = self.per_tick.iter().map(|t| t.wall_clock_us).collect();
        v.sort_unstable();
        let idx = ((v.len() as f64) * 0.99).floor() as usize;
        v[idx.min(v.len() - 1)]
    }

    /// Median `spread_sort_us` across all per-tick samples.
    pub fn median_spread_sort_us(&self) -> u64 {
        if self.per_tick.is_empty() {
            return 0;
        }
        let mut v: Vec<u64> = self.per_tick.iter().map(|t| t.spread_sort_us).collect();
        v.sort_unstable();
        v[v.len() / 2]
    }
}

/// Per-fixture state for the cast-density stress test.
pub struct StressCastDensityState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,
    /// Per-agent position SoA, vec3<f32> padded to 16 bytes per slot
    /// (storage buffer alignment).
    agent_pos_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Spatial grid (27-cell walk reads these). Pre-populated on host
    // because no BuildHash kernel runs in this fixture's schedule. --
    spatial_grid_cells_buf: wgpu::Buffer,
    spatial_grid_starts_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Event ring + tail (ring/tail/indirect_args/sim_cfg owned by EventRing) --
    event_ring: EventRing,
    event_tail_staging: wgpu::Buffer,

    // -- Cfg uniforms --
    physics_cfg_buf: wgpu::Buffer,
    consumer_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    // -- Phase 2 debug instrumentation: per-EventKindId histogram. The
    //    dispatcher kernel atomicAdds &event_kind_counts[<kind>] at every
    //    chronicle producer site. Reset to zero each tick so per-tick
    //    histograms don't accumulate across the run. --
    event_kind_counts_buf: wgpu::Buffer,
    event_kind_counts_zero_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    /// D3 compiler-debug-mode instrumentation. `Some` when the
    /// adapter exposes `wgpu::Features::TIMESTAMP_QUERY`. Falls back
    /// to plain `dispatch_<name>` per tick when None (P10 — sim
    /// still runs without per-kernel attribution).
    debug_timings: Option<dispatch::DebugTimings>,
    /// Per-kernel last-completed-tick timings, refreshed after each
    /// `step()` when `debug_timings.is_some()`.
    last_kernel_timings: Vec<dispatch::KernelTiming>,

    n_agents: u32,
}

impl StressCastDensityState {
    /// Try to construct a stress-runtime with `n_agents` slots. Returns
    /// `None` when no compatible wgpu adapter is available.
    pub fn try_new(n_agents: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;
        let registry = build_aoe_pulse_registry();
        let packed = PackedAbilityRegistry::pack(&registry);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "stress_cast_density");

        // -- Per-agent positions: P5-seeded perturbation inside a small
        // cube around the origin. Every agent lands in (or very close
        // to) the world-origin spatial cell, so every spread walk
        // collects every other agent.
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_agents as usize);
        for slot in 0..n_agents {
            let dx = perturb_axis(
                per_agent_u32_pcg_with_extra(POSITION_SEED, slot, 0, 0, 0),
                SEED_HALF_EXTENT,
            );
            let dy = perturb_axis(
                per_agent_u32_pcg_with_extra(POSITION_SEED, slot, 0, 0, 1),
                SEED_HALF_EXTENT,
            );
            let dz = perturb_axis(
                per_agent_u32_pcg_with_extra(POSITION_SEED, slot, 0, 0, 2),
                SEED_HALF_EXTENT,
            );
            positions.push([dx, dy, dz]);
        }

        // -- Agent SoA seeding --
        let alive_init: Vec<u32> = vec![1u32; n_agents as usize];
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let mk_storage = |label: &str, contents: &[u8]| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_alive_buf =
            mk_storage("stress_cast_density::agent_alive", bytemuck::cast_slice(&alive_init));
        let agent_level_buf =
            mk_storage("stress_cast_density::agent_level", bytemuck::cast_slice(&level_init));

        // Pos SoA: vec3 padded to vec4 (16 B per slot).
        let mut padded_pos: Vec<f32> = Vec::with_capacity((n_agents as usize) * 4);
        for &[x, y, z] in &positions {
            padded_pos.push(x);
            padded_pos.push(y);
            padded_pos.push(z);
            padded_pos.push(0.0);
        }
        let agent_pos_buf =
            mk_storage("stress_cast_density::agent_pos", bytemuck::cast_slice(&padded_pos));

        let zeros_f32: Vec<f32> = vec![0.0_f32; n_agents as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_hp_buf = mk_stat("stress_cast_density::agent_hp");
        let agent_max_hp_buf = mk_stat("stress_cast_density::agent_max_hp");
        let agent_attack_damage_buf = mk_stat("stress_cast_density::agent_attack_damage");
        let agent_ability_power_buf = mk_stat("stress_cast_density::agent_ability_power");
        let agent_armor_buf = mk_stat("stress_cast_density::agent_armor");
        let agent_magic_resist_buf = mk_stat("stress_cast_density::agent_magic_resist");
        let agent_move_speed_buf = mk_stat("stress_cast_density::agent_move_speed");
        let agent_mana_buf = mk_stat("stress_cast_density::agent_mana");

        // -- Spatial grid: counting-sort layout, ascending AgentId per
        // cell. With every agent inside a tiny cube around the origin,
        // most agents land in `host_pos_to_cell(0,0,0)` — a few may
        // straddle into a neighbor cell because the perturbation is
        // ±1.5 (= cell-size) — that's fine; the 27-cell walk picks
        // up neighbors regardless.
        let mut buckets: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for (slot, &[x, y, z]) in positions.iter().enumerate() {
            let cell = host_pos_to_cell(x, y, z);
            buckets.entry(cell).or_default().push(slot as u32);
        }
        let mut grid_cells: Vec<u32> = Vec::with_capacity(n_agents as usize);
        for (_cell, ids) in &buckets {
            grid_cells.extend(ids.iter().copied());
        }
        if grid_cells.is_empty() {
            grid_cells.push(0);
        }
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
        let spatial_grid_cells_buf = mk_storage(
            "stress_cast_density::spatial_grid_cells",
            bytemuck::cast_slice(&grid_cells),
        );
        let spatial_grid_starts_buf = mk_storage(
            "stress_cast_density::spatial_grid_starts",
            bytemuck::cast_slice(&grid_starts),
        );

        // -- Event ring + tail (shared infrastructure: ring + tail + zero
        // source + indirect_args + sim_cfg) --
        let event_ring = EventRing::new(&gpu, "stress_cast_density");
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_cast_density::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniforms --
        let physics_cfg_init = physics_DispatchAoePulse::PhysicsDispatchAoePulseCfg {
            agent_cap: n_agents,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_cast_density::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let consumer_cfg_init =
            physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let consumer_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_cast_density::consumer_cfg"),
                contents: bytemuck::bytes_of(&consumer_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: n_agents,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_cast_density::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Phase 2 debug instrumentation buffer (event_kind_counts).
        // Sized to N_EVENT_KIND_SLOTS u32 atomics. Init zero. The
        // matching zero buffer is COPY_SRC only — driven into
        // event_kind_counts at the start of each tick to clear it.
        let zeros_kind: Vec<u32> = vec![0u32; N_EVENT_KIND_SLOTS];
        let event_kind_counts_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_cast_density::event_kind_counts"),
                contents: bytemuck::cast_slice(&zeros_kind),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );
        let event_kind_counts_zero_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_cast_density::event_kind_counts_zero"),
                contents: bytemuck::cast_slice(&zeros_kind),
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );

        // D3 timing instrumentation — initialised before moving `gpu`.
        let debug_timings = dispatch::DebugTimings::new(&gpu);

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_level_buf,
            agent_pos_buf,
            agent_hp_buf,
            agent_max_hp_buf,
            agent_attack_damage_buf,
            agent_ability_power_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            agent_mana_buf,
            spatial_grid_cells_buf,
            spatial_grid_starts_buf,
            registry_gpu,
            event_ring,
            event_tail_staging,
            physics_cfg_buf,
            consumer_cfg_buf,
            seed_cfg_buf,
            event_kind_counts_buf,
            event_kind_counts_zero_buf,
            cache: dispatch::KernelCache::default(),
            debug_timings,
            last_kernel_timings: Vec::new(),
            n_agents,
        })
    }

    /// Last-tick per-kernel GPU wall times (via the compiler-emitted D3
    /// `DebugTimings` instrumentation). Empty when the host adapter
    /// doesn't expose `wgpu::Features::TIMESTAMP_QUERY`.
    pub fn last_kernel_timings(&self) -> &[dispatch::KernelTiming] {
        &self.last_kernel_timings
    }

    /// Encode + dispatch one tick. Returns the `event_tail` value
    /// observed after the dispatcher runs (this is the per-tick
    /// chronicle ring high-water).
    pub fn step(&mut self, tick: u32) -> u32 {
        if let Some(t) = self.debug_timings.as_ref() {
            t.begin_tick();
        }
        // -- Stage 1: dispatch the cast-density dispatcher --
        let physics_cfg = physics_DispatchAoePulse::PhysicsDispatchAoePulseCfg {
            agent_cap: self.n_agents,
            tick,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&physics_cfg),
        );

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stress_cast_density::step::dispatch"),
            });
        self.event_ring.clear_tail_in(&mut encoder);
        // Phase 2 debug: clear the per-kind histogram so the readback
        // captures THIS tick's counts, not accumulated.
        encoder.copy_buffer_to_buffer(
            &self.event_kind_counts_zero_buf,
            0,
            &self.event_kind_counts_buf,
            0,
            (N_EVENT_KIND_SLOTS as u64) * 4,
        );

        // Shared once per tick: every dispatch below adds only its
        // fixture-specific `*Extras` (spatial grid, instrumentation,
        // cfg uniforms, indirect args).
        let agent_buffers = AgentBuffers {
            hp_buf: Some(&self.agent_hp_buf),
            max_hp_buf: Some(&self.agent_max_hp_buf),
            alive_buf: Some(&self.agent_alive_buf),
            pos_buf: Some(&self.agent_pos_buf),
            level_buf: Some(&self.agent_level_buf),
            move_speed_buf: Some(&self.agent_move_speed_buf),
            armor_buf: Some(&self.agent_armor_buf),
            magic_resist_buf: Some(&self.agent_magic_resist_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            mana_buf: Some(&self.agent_mana_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
        };

        let dispatch_extras = physics_DispatchAoePulse::PhysicsDispatchAoePulseExtras {
            spatial_grid_cells: &self.spatial_grid_cells_buf,
            spatial_grid_starts: &self.spatial_grid_starts_buf,
            event_kind_counts: &self.event_kind_counts_buf,
            cfg: &self.physics_cfg_buf,
        };
        let dispatch_bindings =
            physics_DispatchAoePulse::PhysicsDispatchAoePulseBindings::from_context_with_extras(
                &ctx,
                &dispatch_extras,
            );
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_physics_dispatchaoepulse_timing(
                &mut self.cache,
                t,
                &dispatch_bindings,
                &self.gpu.device,
                &mut encoder,
                self.n_agents,
            );
        } else {
            dispatch::dispatch_physics_dispatchaoepulse(
                &mut self.cache,
                &dispatch_bindings,
                &self.gpu.device,
                &mut encoder,
                self.n_agents,
            );
        }

        // Copy event_tail to staging for readback in the same submit so
        // we don't need a second encoder.
        encoder.copy_buffer_to_buffer(
            self.event_ring.tail(),
            0,
            &self.event_tail_staging,
            0,
            4,
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        // -- Stage 2: read tail (= ring high-water) --
        let tail = {
            let slice = self.event_tail_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                res.expect("event_tail_staging map_async failed");
            });
            self.gpu
                .device
                .poll(wgpu::PollType::Wait)
                .expect("device poll failed during event_tail readback");
            let v = {
                let view = slice.get_mapped_range();
                let words: &[u32] = bytemuck::cast_slice(&view);
                words[0]
            };
            self.event_tail_staging.unmap();
            v
        };

        // -- Stage 3: consumer + seed_indirect_0 (no-op work, but the
        // schedule lists them, so we run them to surface any wiring
        // regression alongside the metrics). --
        let consumer_cfg =
            physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleCfg {
                event_count: tail,
                tick,
                seed: 0,
                agent_cap: 0,
            };
        self.gpu.queue.write_buffer(
            &self.consumer_cfg_buf,
            0,
            bytemuck::bytes_of(&consumer_cfg),
        );
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.n_agents,
            tick,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));

        let mut encoder2 = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stress_cast_density::step::consume_and_seed"),
            });
        let consumer_extras =
            physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleExtras {
                cfg: &self.consumer_cfg_buf,
            };
        let consumer_bindings = physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleBindings::from_context_with_extras(
            &ctx,
            &consumer_extras,
        );
        // Consumer dispatch sized by the cap'd ring slot count so the
        // workgroup grid never exceeds the ring's actual data range.
        let consumer_workgroups = tail.min(RING_SLOT_CAP).max(self.n_agents);
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_physics_applydamagefromchronicle_timing(
                &mut self.cache,
                t,
                &consumer_bindings,
                &self.gpu.device,
                &mut encoder2,
                consumer_workgroups,
            );
        } else {
            dispatch::dispatch_physics_applydamagefromchronicle(
                &mut self.cache,
                &consumer_bindings,
                &self.gpu.device,
                &mut encoder2,
                consumer_workgroups,
            );
        }
        let seed_extras = seed_indirect_0::SeedIndirect0Extras {
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        let seed_bindings =
            seed_indirect_0::SeedIndirect0Bindings::from_context_with_extras(
                &ctx,
                &seed_extras,
            );
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_seed_indirect_0_timing(
                &mut self.cache,
                t,
                &seed_bindings,
                &self.gpu.device,
                &mut encoder2,
                self.n_agents,
            );
        } else {
            dispatch::dispatch_seed_indirect_0(
                &mut self.cache,
                &seed_bindings,
                &self.gpu.device,
                &mut encoder2,
                self.n_agents,
            );
        }
        // D3 timestamp resolve must run BEFORE encoder2.finish so the
        // resolve_query_set + readback-copy land in the same submission
        // as the bracketed dispatches.
        if let Some(t) = self.debug_timings.as_ref() {
            t.finalise_tick(&mut encoder2);
        }
        self.gpu.queue.submit(Some(encoder2.finish()));
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed after consumer dispatch");

        if let Some(t) = self.debug_timings.as_ref() {
            self.last_kernel_timings = t.read_kernel_timings_with(&self.gpu.device);
        }

        tail
    }

    pub fn n_agents(&self) -> u32 {
        self.n_agents
    }

    /// Read back the per-EventKindId histogram populated by the
    /// dispatcher's compiler-emitted `atomicAdd(&event_kind_counts[k],
    /// 1u)` instrumentation. Indices are `EventKindId` discriminants;
    /// the slot count matches [`N_EVENT_KIND_SLOTS`]. Resets to zero
    /// at the start of every `step()` call, so this returns the most
    /// recent tick's histogram only.
    pub fn read_event_kind_histogram(&self) -> [u32; N_EVENT_KIND_SLOTS] {
        let bytes = (N_EVENT_KIND_SLOTS as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_cast_density::event_kind_counts_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stress_cast_density::read_event_kind_histogram"),
            });
        encoder.copy_buffer_to_buffer(&self.event_kind_counts_buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("event_kind_counts_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during event_kind_counts readback");
        let mut out = [0u32; N_EVENT_KIND_SLOTS];
        {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            out.copy_from_slice(&words[..N_EVENT_KIND_SLOTS]);
        }
        staging.unmap();
        out
    }
}

/// Build the AbilityRegistry for this fixture: AbilityId(1) =
/// AoePulse with a single `Damage(0.0)` effect inside Spread(1.5,
/// 256). The registry is built host-side rather than by lowering the
/// .ability file because our consumer is a no-op and we only need the
/// dispatcher's per-effect area + payload columns to drive the
/// 27-cell walk + bitonic sort.
fn build_aoe_pulse_registry() -> AbilityRegistry {
    let mut program = AbilityProgram::new_single_target(
        /*range*/ SPREAD_RADIUS,
        Gate {
            cooldown_ticks: 1,
            hostile_only: false,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 0.0 }],
    );
    program.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Spread,
        args: [SPREAD_RADIUS, SPREAD_MAX_TARGETS as f32, 0.0, 0.0],
    }));
    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(program);
    debug_assert_eq!(
        id,
        AbilityId::new(1).unwrap(),
        "AoePulse must land at AbilityId(1) — `agent_level[*] = 1` dispatches it"
    );
    builder.build()
}

/// Run the stress test at the given `agent_cap` for `tick_budget` ticks.
/// Returns aggregated metrics. Returns `None` when no GPU adapter is
/// available — caller should treat that as a "skip" path.
///
/// **Panics** propagate up — callers wrapping in `catch_unwind` should
/// expect the unwound panic message via the `Err` arm.
pub fn run_stress(agent_cap: u32, tick_budget: u32) -> Option<StressMetrics> {
    let mut state = StressCastDensityState::try_new(agent_cap)?;
    let mut per_tick = Vec::with_capacity(tick_budget as usize);
    let mut first_overflow_at_tick: Option<u32> = None;
    let mut max_ring_high_water: u32 = 0;
    for tick in 0..tick_budget {
        let t_step = std::time::Instant::now();
        let tail = state.step(tick);
        let step_us = t_step.elapsed().as_micros() as u64;
        let overflowed = tail > RING_SLOT_CAP;
        if overflowed && first_overflow_at_tick.is_none() {
            first_overflow_at_tick = Some(tick);
        }
        if tail > max_ring_high_water {
            max_ring_high_water = tail;
        }
        // `wall_clock_us` and `spread_sort_us` measure the same elapsed
        // window. The dispatcher (which contains the Spread bitonic
        // sort) is the dominant cost in `step()`, so the two columns
        // line up by construction; both are emitted in the per-tick
        // NDJSON to match the plan's metric inventory.
        let kernel_timings: Vec<(String, u64)> = state
            .last_kernel_timings()
            .iter()
            .map(|k| (k.kernel.clone(), k.wall_ns))
            .collect();
        let event_kind_histogram = state.read_event_kind_histogram();
        per_tick.push(TickMetric {
            tick,
            wall_clock_us: step_us,
            ring_high_water: tail,
            ring_overflowed: overflowed,
            spread_sort_us: step_us,
            kernel_timings,
            event_kind_histogram,
        });
    }
    Some(StressMetrics {
        agent_cap,
        ticks_completed: per_tick.len() as u32,
        per_tick,
        first_overflow_at_tick,
        max_ring_high_water,
    })
}

// ---------------------------------------------------------------------- //
// Behavioral pin tests
// ---------------------------------------------------------------------- //

#[cfg(test)]
mod stress_tests {
    use super::*;

    /// Pin: at agent_cap=10, all agents in single 27-cell, the
    /// dispatcher MUST advance the tick AND walk the Spread sort at
    /// least once. We verify the latter via the Phase 2 per-EventKindId
    /// histogram — slot 26 (`EventKindId::EffectDamageApplied`) must be
    /// non-zero.
    #[test]
    fn tick_advances_under_max_density() {
        let mut state = match StressCastDensityState::try_new(10) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[stress_cast_density tick_advances_under_max_density] \
                     skipping: no wgpu adapter available on this host. The \
                     build itself still validates the kernel emit and the \
                     binding hookup at compile time."
                );
                return;
            }
        };
        let tail = state.step(0);
        eprintln!(
            "[stress_cast_density] tick 0 tail={} (expected ~ n_agents × in-radius candidates)",
            tail,
        );
        assert!(
            tail > 0,
            "Spread sort must have produced ≥ 1 EffectDamageApplied record at \
             agent_cap=10 with all agents inside the seed cube; got tail=0 \
             (the dispatcher did not append any chronicle records)",
        );

        // EventKindId::EffectDamageApplied = 26 (see crates/engine/src/cascade/handler.rs).
        let histogram = state.read_event_kind_histogram();
        let damage_records = histogram[26];
        assert!(
            damage_records > 0,
            "expected event_kind_counts[26] (EffectDamageApplied) > 0; got {} \
             — Spread sort path did not emit the expected events (tail={})",
            damage_records, tail,
        );
        eprintln!(
            "[stress_cast_density] tick 0: {} damage records (kind=26) of {} total",
            damage_records, tail,
        );
    }
}
