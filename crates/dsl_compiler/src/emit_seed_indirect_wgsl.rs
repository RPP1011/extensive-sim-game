//! WGSL emitter for `SeedIndirectKernel` — seeds the cascade's
//! indirect-args buffer for the next iteration based on the apply-path
//! event ring's tail count.
//!
//! Recovered from the pre-T16 hand-written `SEED_INDIRECT_WGSL`
//! constant (`crates/engine_gpu/src/cascade_resident.rs:275` at
//! commit `4474566c~1`) and adapted to the post-T16 raw-u32 binding
//! layout.
//!
//! ## Adaptations vs. pre-T16
//!
//! - Pre-T16 declared `apply_tail: array<atomic<u32>>` and used
//!   `atomicLoad`. New BGL declares `apply_tail: array<u32>` (read).
//!   Replaced with raw read.
//! - Pre-T16 had a `num_events` output binding; new BGL drops it
//!   (consumers read directly from the source tail).
//! - Pre-T16 used `atomicAdd(&sim_cfg.tick, 1u)` to bump the GPU-side
//!   tick. New BGL declares `sim_cfg: array<u32>` (non-atomic) and the
//!   CPU forward inside `step_batch` is still authoritative for tick.
//!   The atomic bump is dropped here; will return when the tick field
//!   moves to `atomic<u32>` in the BGL or the CPU forward retires.
//!
//! ## Workgroup
//!
//! Single 1-thread dispatch. Computes `wg = ceil(n / 64), capped at
//! cap_wg` and writes `indirect_args[0..3] = (wg, 1, 1)`.

const WORKGROUP_LANE_COUNT: u32 = 64;

/// Emit the body of `engine_gpu_rules/src/seed_indirect.wgsl`.
pub fn emit_seed_indirect_wgsl() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read>       apply_tail:    array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> indirect_args: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> sim_cfg:       array<u32>;\n\
struct SeedIndirectCfg {{ iter_idx: u32, _pad0: u32, _pad1: u32, _pad2: u32 }};\n\
@group(0) @binding(3) var<uniform>             cfg:           SeedIndirectCfg;\n\
\n\
const WG_LANES: u32 = {WORKGROUP_LANE_COUNT}u;\n\
// cap_wg = max workgroups for cascade per-iter dispatch. Until the\n\
// cfg uniform exposes `cap_wg` directly, derive from a fixed ceiling\n\
// equal to ceil(MAX_AGENTS / WG_LANES). 200 000 / 64 ≈ 3 125, round\n\
// up to a power-of-2 for headroom.\n\
const CAP_WG: u32 = 4096u;\n\
\n\
@compute @workgroup_size(1)\n\
fn cs_seed_indirect(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let n = apply_tail[0];\n\
    let req = (n + WG_LANES - 1u) / WG_LANES;\n\
    var wg = req;\n\
    if (wg > CAP_WG) {{ wg = CAP_WG; }}\n\
    indirect_args[0] = wg;\n\
    indirect_args[1] = 1u;\n\
    indirect_args[2] = 1u;\n\
    // sim_cfg.tick atomic bump retired post-T16 (see module docs).\n\
    // CPU forward inside step_batch is authoritative for tick state\n\
    // until the BGL slot promotes to atomic<u32>.\n\
    let _iter = cfg.iter_idx;\n\
    let _sc0 = sim_cfg[0];\n\
}}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_indirect_naga_parses() {
        let src = emit_seed_indirect_wgsl();
        naga::front::wgsl::parse_str(&src)
            .map_err(|e| {
                format!("--- WGSL ---\n{src}\n--- naga error ---\n{}", e.emit_to_string(&src))
            })
            .expect("emit_seed_indirect_wgsl should parse cleanly");
    }
}
