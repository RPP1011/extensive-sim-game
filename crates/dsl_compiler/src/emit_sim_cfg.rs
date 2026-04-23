//! Shared WGSL emission for the GPU-resident `SimCfg` struct.
//!
//! `SimCfg` carries world-scalar sim state (tick, world seed, combat /
//! movement radii that are sim-wide rather than subsystem-local, and
//! cache-invalidation generation counters) in a single storage buffer
//! that every GPU kernel binds. See
//! `docs/superpowers/plans/2026-04-22-gpu-sim-state.md` for the design
//! rationale.
//!
//! Layout must match `engine_gpu::sim_cfg::SimCfg` byte-for-byte; the
//! `sim_cfg_layout` regression test in `engine_gpu` fences drift. This
//! module is the single source of truth for the WGSL struct — every
//! kernel emitter that wants the shared buffer calls
//! [`emit_sim_cfg_struct_wgsl`] instead of open-coding the struct.

use std::fmt::Write;

/// Emit the shared `SimCfg` struct declaration + binding into `out`.
/// Layout mirrors `engine_gpu::sim_cfg::SimCfg` exactly.
///
/// `binding` parameterises the `@binding(...)` attribute so the caller
/// can place the buffer past its own kernel-local bindings. The struct
/// is `var<storage, read>` (not `uniform`) because the seed-indirect
/// kernel atomically increments `tick` on this same buffer.
pub fn emit_sim_cfg_struct_wgsl(out: &mut String, binding: u32) {
    writeln!(out, "struct SimCfg {{").unwrap();
    writeln!(out, "    tick:                          u32,").unwrap();
    writeln!(out, "    world_seed_lo:                 u32,").unwrap();
    writeln!(out, "    world_seed_hi:                 u32,").unwrap();
    writeln!(out, "    _sim_cfg_pad0:                 u32,").unwrap();
    writeln!(out, "    engagement_range:              f32,").unwrap();
    writeln!(out, "    attack_damage:                 f32,").unwrap();
    writeln!(out, "    attack_range:                  f32,").unwrap();
    writeln!(out, "    move_speed:                    f32,").unwrap();
    writeln!(out, "    move_speed_mult:               f32,").unwrap();
    writeln!(out, "    kin_radius:                    f32,").unwrap();
    writeln!(out, "    cascade_max_iterations:        u32,").unwrap();
    writeln!(out, "    rules_registry_generation:     u32,").unwrap();
    writeln!(out, "    abilities_registry_generation: u32,").unwrap();
    writeln!(out, "    _sim_cfg_reserved0:            u32,").unwrap();
    writeln!(out, "    _sim_cfg_reserved1:            u32,").unwrap();
    writeln!(out, "    _sim_cfg_reserved2:            u32,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(
        out,
        "@group(0) @binding({binding}) var<storage, read> sim_cfg: SimCfg;"
    )
    .unwrap();
    writeln!(out).unwrap();
}
