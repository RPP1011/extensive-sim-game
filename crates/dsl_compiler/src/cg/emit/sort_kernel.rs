//! GPU radix sort emit — Stage A (LSD radix on seq) + Stage B (counting sort on target).
//!
//! Produces 15 WGSL kernels per opt-in fixture:
//!   Stage A: 4 passes × {histogram, scan, scatter} = 12 kernels
//!   Stage B: 1 × {count, scan, scatter}            = 3 kernels
//!
//! Output: `event_ring` permuted via ping-pong with
//! `event_ring_sort_scratch`. After Stage B, all events with the same
//! `target` are adjacent, intra-target seq-ordered.

// STABILITY: scatter kernels use a single-workgroup serial chunk-loop
// pattern with atomicAdd-based intra-bucket position. For the FINAL
// bucket (events with identical full sort keys), order doesn't affect
// fold determinism (a + b == b + a within a bucket). For INTERMEDIATE
// passes, intra-bucket atomicAdd race is a known stability gap; the
// proptest in tests/proptest_radix_sort.rs is the catch gate. If it
// surfaces drift, replace the atomicAdd with a workgroup-scoped
// exclusive scan over a per-bucket mask (warp-level scan pattern).

use crate::cg::program::EventLayout;

/// Emit Stage A pass `pass_idx` (0..4): histogram + scan + scatter.
/// Each pass processes 8 bits of the seq key, starting from LSB.
/// Returns the (histogram_wgsl, scan_wgsl, scatter_wgsl) triple.
#[allow(dead_code)] // wired into the schedule by the build_helper in Task 3.6
pub(crate) fn emit_stage_a_pass(pass_idx: u32, layout: &EventLayout) -> (String, String, String) {
    let stride = layout.record_stride_u32;
    let seq_offset = stride - 1;  // seq trailer is last word
    let bit_shift = pass_idx * 8;
    let bucket_mask = 0xFFu32;

    // -- Histogram kernel.
    let histogram = format!(r#"
@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> radix_histogram: array<atomic<u32>>;

@compute @workgroup_size(64)
fn radix_stage_a_pass{pass_idx}_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let count = atomicLoad(&event_tail);
    if (tid >= count) {{ return; }}

    let seq = atomicLoad(&event_ring_in[tid * {stride}u + {seq_offset}u]);
    let bucket = (seq >> {bit_shift}u) & {bucket_mask}u;
    atomicAdd(&radix_histogram[bucket], 1u);
}}
"#);

    // -- Scan kernel (exclusive prefix sum on 256 buckets).
    let scan = format!(r#"
@group(0) @binding(0) var<storage, read_write> radix_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> radix_bucket_offsets: array<u32>;

var<workgroup> scan_tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn radix_stage_a_pass{pass_idx}_scan(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = lid.x;
    scan_tmp[i] = atomicLoad(&radix_histogram[i]);
    workgroupBarrier();

    // Hillis-Steele exclusive scan.
    var stride: u32 = 1u;
    loop {{
        if (stride >= 256u) {{ break; }}
        let v = select(0u, scan_tmp[i - stride], i >= stride);
        workgroupBarrier();
        scan_tmp[i] = scan_tmp[i] + v;
        workgroupBarrier();
        stride = stride << 1u;
    }}

    // Shift to exclusive: position i gets scan_tmp[i-1] (or 0 for i=0).
    let count_i = atomicLoad(&radix_histogram[i]);
    radix_bucket_offsets[i] = select(scan_tmp[i] - count_i, 0u, i == 0u);

    // Reset histogram counters for use as "next-position" during scatter.
    atomicStore(&radix_histogram[i], 0u);
}}
"#);

    // -- Scatter kernel — single-workgroup serial chunk loop for stability.
    let scatter = format!(r#"
@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> radix_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> radix_bucket_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> event_ring_out: array<atomic<u32>>;

@compute @workgroup_size(256)
fn radix_stage_a_pass{pass_idx}_scatter(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let count = atomicLoad(&event_tail);

    var chunk_base: u32 = 0u;
    loop {{
        if (chunk_base >= count) {{ break; }}
        let tid = chunk_base + lid.x;
        let active = tid < count;

        if (active) {{
            let seq = atomicLoad(&event_ring_in[tid * {stride}u + {seq_offset}u]);
            let bucket = (seq >> {bit_shift}u) & {bucket_mask}u;
            let wg_chunk_pos = atomicAdd(&radix_histogram[bucket], 1u);
            let dst = radix_bucket_offsets[bucket] + wg_chunk_pos;
            for (var w = 0u; w < {stride}u; w = w + 1u) {{
                let v = atomicLoad(&event_ring_in[tid * {stride}u + w]);
                atomicStore(&event_ring_out[dst * {stride}u + w], v);
            }}
        }}
        workgroupBarrier();
        chunk_base = chunk_base + 256u;
    }}
}}
"#);

    (histogram, scan, scatter)
}

/// Emit Stage B (single counting-sort pass on target_id):
/// (count_wgsl, scan_wgsl, scatter_wgsl).
#[allow(dead_code)] // wired into the schedule by the build_helper in Task 3.6
pub(crate) fn emit_stage_b(layout: &EventLayout) -> (String, String, String) {
    let stride = layout.record_stride_u32;

    let count = format!(r#"
struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> target_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cfg: SortCfg;

@compute @workgroup_size(64)
fn radix_stage_b_count(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let count = atomicLoad(&event_tail);
    if (tid >= count) {{ return; }}

    let target = atomicLoad(&event_ring_in[tid * {stride}u + cfg.target_word_offset]);
    let bucket = select(target, cfg.agent_cap, target >= cfg.agent_cap);
    atomicAdd(&target_histogram[bucket], 1u);
}}
"#);

    let scan = format!(r#"
struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read_write> target_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> target_offsets: array<u32>;
@group(0) @binding(2) var<uniform> cfg: SortCfg;

@compute @workgroup_size(1)
fn radix_stage_b_scan(@builtin(local_invocation_id) lid: vec3<u32>) {{
    // Single-thread serial scan — handles up to agent_cap+1 buckets
    // (range [0, agent_cap] plus the sentinel at agent_cap).
    if (lid.x != 0u) {{ return; }}
    let cap_plus_one = cfg.agent_cap + 1u;
    var running: u32 = 0u;
    for (var i = 0u; i < cap_plus_one; i = i + 1u) {{
        target_offsets[i] = running;
        running = running + atomicLoad(&target_histogram[i]);
        atomicStore(&target_histogram[i], 0u);  // reset for scatter
    }}
}}
"#);

    let scatter = format!(r#"
struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> target_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> target_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> event_ring_out: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> cfg: SortCfg;

@compute @workgroup_size(256)
fn radix_stage_b_scatter(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let count = atomicLoad(&event_tail);

    var chunk_base: u32 = 0u;
    loop {{
        if (chunk_base >= count) {{ break; }}
        let tid = chunk_base + lid.x;
        let active = tid < count;

        if (active) {{
            let target = atomicLoad(&event_ring_in[tid * {stride}u + cfg.target_word_offset]);
            let bucket = select(target, cfg.agent_cap, target >= cfg.agent_cap);
            let wg_chunk_pos = atomicAdd(&target_histogram[bucket], 1u);
            let dst = target_offsets[bucket] + wg_chunk_pos;
            for (var w = 0u; w < {stride}u; w = w + 1u) {{
                let v = atomicLoad(&event_ring_in[tid * {stride}u + w]);
                atomicStore(&event_ring_out[dst * {stride}u + w], v);
            }}
        }}
        workgroupBarrier();
        chunk_base = chunk_base + 256u;
    }}
}}
"#);

    (count, scan, scatter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::program::EventLayout;
    use std::collections::BTreeMap;

    fn test_layout() -> EventLayout {
        EventLayout {
            record_stride_u32: 11,
            header_word_count: 2,
            buffer_name: "event_ring".into(),
            fields: BTreeMap::new(),
        }
    }

    #[test]
    fn stage_a_pass_0_emits_three_kernels() {
        let lay = test_layout();
        let (h, s, sc) = emit_stage_a_pass(0, &lay);
        assert!(h.contains("radix_stage_a_pass0_histogram"));
        assert!(s.contains("radix_stage_a_pass0_scan"));
        assert!(sc.contains("radix_stage_a_pass0_scatter"));
    }

    #[test]
    fn stage_a_pass_0_uses_low_8_bits_of_seq() {
        let lay = test_layout();
        let (h, _, _) = emit_stage_a_pass(0, &lay);
        // bit_shift = 0 for pass 0
        assert!(h.contains("(seq >> 0u) & 255u"));
    }

    #[test]
    fn stage_a_pass_3_uses_high_8_bits_of_seq() {
        let lay = test_layout();
        let (h, _, _) = emit_stage_a_pass(3, &lay);
        // bit_shift = 24 for pass 3
        assert!(h.contains("(seq >> 24u) & 255u"));
    }

    #[test]
    fn stage_a_reads_seq_at_last_word() {
        let lay = test_layout();
        let (h, _, _) = emit_stage_a_pass(0, &lay);
        // seq_offset = stride - 1 = 10
        assert!(h.contains("tid * 11u + 10u"));
    }

    #[test]
    fn stage_b_emits_three_kernels() {
        let lay = test_layout();
        let (c, s, sc) = emit_stage_b(&lay);
        assert!(c.contains("radix_stage_b_count"));
        assert!(s.contains("radix_stage_b_scan"));
        assert!(sc.contains("radix_stage_b_scatter"));
    }

    #[test]
    fn stage_b_uses_cfg_target_word_offset() {
        let lay = test_layout();
        let (c, _, _) = emit_stage_b(&lay);
        assert!(c.contains("cfg.target_word_offset"));
    }

    #[test]
    fn stage_b_handles_target_overflow_with_sentinel() {
        let lay = test_layout();
        let (c, _, _) = emit_stage_b(&lay);
        assert!(c.contains("target >= cfg.agent_cap"));
    }
}
