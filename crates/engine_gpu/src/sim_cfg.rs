//! Shared GPU-resident sim state — tick, world seed, world-scalar
//! fields, and cache-invalidation generation counters. Bound as a
//! `storage` buffer (not uniform) because the tick field is atomically
//! incremented by the seed-indirect kernel.
//!
//! WGSL-side struct layout must match this Rust definition byte-for-
//! byte. The `sim_cfg_layout` regression test fences field offsets.
//!
//! See `docs/superpowers/specs/2026-04-22-gpu-sim-state-design.md`.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};
use engine::state::SimState;

/// Phase D (follow-up subsystem 1) — GPU-resident sim-wide state.
/// Replaces per-kernel cfg uniforms for the world-scalar fields.
///
/// ## Field-path notes
///
/// The combat + movement scalars mirror `state.config.combat.*` and
/// `state.config.movement.*`. A few planned fields don't live in
/// `state.config` today, so `SimCfg::from_state` populates them from
/// the canonical constants / per-agent defaults used elsewhere:
///
///   * `move_speed_mult` — per-agent field on `SimState`, no sim-wide
///     config. Seeded to `1.0` (the spawn default).
///   * `kin_radius` — designer-tunable via `state.config.combat.kin_radius`
///     (promoted from the retired `engine_gpu::cascade::DEFAULT_KIN_RADIUS`
///     const on 2026-04-22). SimCfg just mirrors the Config field so GPU
///     kernels can read it via the shared uniform.
///   * `cascade_max_iterations` — const in
///     `engine_gpu::cascade::MAX_CASCADE_ITERATIONS`; duplicated as `8`.
///
/// Later tasks that wire SimCfg into kernels will reconcile these
/// with the canonical constants (e.g. by importing them here) — Task
/// 2.1 is layout-only, so we avoid the coupling.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct SimCfg {
    /// Current sim tick. Atomically incremented at end of each tick
    /// by the seed-indirect kernel (Task 2.3).
    pub tick:                          u32,
    /// World RNG seed low u32 (CPU `state.seed & 0xFFFF_FFFF`).
    pub world_seed_lo:                 u32,
    /// World RNG seed high u32.
    pub world_seed_hi:                 u32,
    /// Padding to 16-byte alignment for the f32 block.
    pub _pad0:                         u32,

    /// World-scalar combat config fields — mirrors of
    /// `state.config.combat.*` and `state.config.movement.*`.
    pub engagement_range:              f32,
    pub attack_damage:                 f32,
    pub attack_range:                  f32,
    pub move_speed:                    f32,
    pub move_speed_mult:               f32,
    pub kin_radius:                    f32,

    /// Cascade's max iteration count (same u32 as per-kernel cfg today).
    pub cascade_max_iterations:        u32,

    /// Cache-invalidation counters. Incremented on the CPU side when
    /// the `CascadeRegistry` or `PackedAbilityRegistry` changes shape.
    /// GPU-side caches key on these for redundant-upload elision.
    pub rules_registry_generation:     u32,
    pub abilities_registry_generation: u32,

    /// Reserved headroom (~12 bytes) for sim-wide u32 fields that
    /// subsystems (2)/(3) will land here rather than adding new
    /// bindings. Zero-initialise; consumers ignore until claimed.
    pub _reserved:                     [u32; 3],
}

impl SimCfg {
    /// Populate from `SimState` at batch-entry time. Reads world seed,
    /// config scalars, and current tick. Called by `ensure_resident_init`.
    pub fn from_state(state: &SimState) -> Self {
        Self {
            tick:                          state.tick,
            world_seed_lo:                 (state.seed & 0xFFFF_FFFF) as u32,
            world_seed_hi:                 (state.seed >> 32) as u32,
            _pad0:                         0,
            engagement_range:              state.config.combat.engagement_range,
            attack_damage:                 state.config.combat.attack_damage,
            attack_range:                  state.config.combat.attack_range,
            move_speed:                    state.config.movement.move_speed_mps,
            // No sim-wide move-speed multiplier today — per-agent only.
            // Seed to the spawn default (1.0).
            move_speed_mult:               1.0,
            // Designer-tunable via `config combat.kin_radius` in
            // `assets/sim/config.sim` (promoted from a hardcoded const
            // 2026-04-23).
            kin_radius:                    state.config.combat.kin_radius,
            // Mirrors `engine_gpu::cascade::MAX_CASCADE_ITERATIONS`.
            cascade_max_iterations:        8,
            rules_registry_generation:     0,
            abilities_registry_generation: 0,
            _reserved:                     [0; 3],
        }
    }
}

/// Allocate the GPU-side SimCfg buffer, sized for one instance. Usage
/// flags: STORAGE (kernel reads + atomic tick writes), COPY_SRC
/// (snapshot readback), COPY_DST (host upload of initial values).
pub fn create_sim_cfg_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("engine_gpu::sim_cfg"),
        size:               std::mem::size_of::<SimCfg>() as u64,
        usage:              wgpu::BufferUsages::STORAGE
                         | wgpu::BufferUsages::COPY_SRC
                         | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Upload a fresh `SimCfg` snapshot to the device buffer via
/// `queue.write_buffer`. Called once per batch at
/// `ensure_resident_init` entry to seed world scalars + tick.
pub fn upload_sim_cfg(queue: &wgpu::Queue, buf: &wgpu::Buffer, cfg: &SimCfg) {
    queue.write_buffer(buf, 0, bytemuck::bytes_of(cfg));
}
