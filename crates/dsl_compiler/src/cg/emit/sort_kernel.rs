//! GPU event-ring sort emit — ONE single-workgroup dispatch.
//!
//! Sorts the prior-tick chronicle ring by `(target, seq)` so f32 view
//! folds accumulate in a canonical order. The permutation is exactly
//! the one the historical 15-kernel pipeline produced (4 LSD radix
//! passes on the 32-bit `seq` trailer, then a stable counting sort on
//! the clamped `target` word): every pass here is a stable 8-bit LSD
//! radix pass, and the target passes run over the SAME clamped key
//! (`min(target, agent_cap)`), so the final order is "by clamped
//! target, then by seq, then by input order" in both designs.
//!
//! Why one dispatch: the ring is empty on most ticks, so the sort's
//! cost was pure launch overhead — 15 dispatches (five of them a
//! million threads wide, over the ring CAPACITY rather than its tail)
//! plus a 46 MB copy of the whole ring back from the scratch buffer,
//! every tick. This kernel is one 256-thread workgroup that walks
//! `tail` records, ping-pongs ring ↔ scratch across the passes with
//! workgroup + storage barriers, and copies only the live records
//! back when the pass count is odd.
//!
//! Determinism: the only atomics are per-bucket COUNTS (commutative);
//! every record's destination is `bucket_cursor + rank`, where `rank`
//! is the number of earlier records in the same 256-record block with
//! the same key — computed by an ordered scan, never by atomics.
//!
//! Workgroup memory: 4 × 256 × 4 B = 4 KiB, under every adapter's
//! floor, so no limit negotiation is needed.

use crate::cg::program::EventLayout;

/// Entry-point / kernel name of the single sort dispatch.
pub const SORT_KERNEL_NAME: &str = "event_ring_sort";

/// Emit the complete WGSL module for the single-dispatch sort.
///
/// Bindings:
/// * 0 — `event_ring` (storage, read_write): sorted in place.
/// * 1 — `event_tail` (storage, read): live record count.
/// * 2 — `scratch` (storage, read_write): ping-pong buffer, ≥ ring size.
/// * 3 — `cfg` (uniform): `{ target_word_offset, agent_cap, _, _ }`.
pub(crate) fn emit_single_dispatch_sort(layout: &EventLayout, ring_cap_slots: u32) -> String {
    let stride = layout.record_stride_u32;
    let seq_offset = stride - 1; // seq trailer is the last word
    format!(
        r#"// GENERATED — single-dispatch stable radix sort of the chronicle ring
// by (clamped target, seq). See dsl_compiler::cg::emit::sort_kernel.
struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read_write> ring: array<u32>;
@group(0) @binding(1) var<storage, read> event_tail: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(3) var<uniform> cfg: SortCfg;

const STRIDE: u32 = {stride}u;
const SEQ_OFFSET: u32 = {seq_offset}u;
const WG: u32 = 256u;
const RING_CAP: u32 = {ring_cap_slots}u;

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> cursor: array<u32, 256>;
var<workgroup> blk_key: array<u32, 256>;
var<workgroup> blk_cnt: array<atomic<u32>, 256>;

fn load_word(from_ring: bool, i: u32) -> u32 {{
    if (from_ring) {{ return ring[i]; }}
    return scratch[i];
}}

fn store_word(to_ring: bool, i: u32, v: u32) {{
    if (to_ring) {{ ring[i] = v; }} else {{ scratch[i] = v; }}
}}

// 8-bit key of record `r` for pass `p`: passes 0..3 walk the seq
// trailer LSB-first; passes 4.. walk the CLAMPED target LSB-first.
fn key_of(from_ring: bool, r: u32, p: u32) -> u32 {{
    if (p < 4u) {{
        let seq = load_word(from_ring, r * STRIDE + SEQ_OFFSET);
        return (seq >> (p * 8u)) & 0xFFu;
    }}
    let tgt = load_word(from_ring, r * STRIDE + cfg.target_word_offset);
    let clamped = select(tgt, cfg.agent_cap, tgt >= cfg.agent_cap);
    return (clamped >> ((p - 4u) * 8u)) & 0xFFu;
}}

@compute @workgroup_size(256)
fn {name}(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let t = lid.x;
    let n = min(event_tail[0], RING_CAP);
    // Enough 8-bit passes to cover every clamped target value
    // (0 ..= agent_cap). agent_cap = 0 needs none: every key is 0.
    let tbits = 32u - countLeadingZeros(cfg.agent_cap);
    let tpasses = (tbits + 7u) / 8u;
    let npasses = 4u + tpasses;
    var from_ring = true;
    for (var p = 0u; p < npasses; p = p + 1u) {{
        // 1. Bucket histogram over the live records.
        atomicStore(&hist[t], 0u);
        workgroupBarrier();
        for (var r = t; r < n; r = r + WG) {{
            atomicAdd(&hist[key_of(from_ring, r, p)], 1u);
        }}
        workgroupBarrier();
        // 2. Exclusive prefix sum → per-bucket write cursor.
        if (t == 0u) {{
            var running = 0u;
            for (var b = 0u; b < 256u; b = b + 1u) {{
                cursor[b] = running;
                running = running + atomicLoad(&hist[b]);
            }}
        }}
        workgroupBarrier();
        // 3. Stable scatter, 256 records per block. A record's rank
        //    inside its bucket is the count of EARLIER records in the
        //    block with the same key (ordered scan, deterministic).
        for (var base = 0u; base < n; base = base + WG) {{
            let r = base + t;
            let valid = r < n;
            var k = 0xFFFFFFFFu;
            if (valid) {{ k = key_of(from_ring, r, p); }}
            blk_key[t] = k;
            atomicStore(&blk_cnt[t], 0u);
            workgroupBarrier();
            if (valid) {{
                var rank = 0u;
                for (var u = 0u; u < t; u = u + 1u) {{
                    if (blk_key[u] == k) {{ rank = rank + 1u; }}
                }}
                let dst = cursor[k] + rank;
                for (var w = 0u; w < STRIDE; w = w + 1u) {{
                    store_word(!from_ring, dst * STRIDE + w, load_word(from_ring, r * STRIDE + w));
                }}
                atomicAdd(&blk_cnt[k], 1u);
            }}
            workgroupBarrier();
            cursor[t] = cursor[t] + atomicLoad(&blk_cnt[t]);
            workgroupBarrier();
        }}
        storageBarrier();
        workgroupBarrier();
        from_ring = !from_ring;
    }}
    // An odd pass count leaves the sorted records in `scratch`; fold
    // consumers read the canonical ring, so copy the LIVE records back.
    if (!from_ring) {{
        let words = n * STRIDE;
        for (var i = t; i < words; i = i + WG) {{
            ring[i] = scratch[i];
        }}
    }}
}}
"#,
        name = SORT_KERNEL_NAME,
    )
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
    fn emits_single_entry_point_named_event_ring_sort() {
        let src = emit_single_dispatch_sort(&test_layout(), 1_048_576);
        assert_eq!(src.matches("@compute").count(), 1, "exactly one entry point");
        assert!(src.contains("fn event_ring_sort("));
        assert!(src.contains("@workgroup_size(256)"));
    }

    #[test]
    fn reads_seq_at_last_word_and_target_via_cfg_offset() {
        let src = emit_single_dispatch_sort(&test_layout(), 1_048_576);
        assert!(src.contains("const STRIDE: u32 = 11u;"));
        assert!(src.contains("const SEQ_OFFSET: u32 = 10u;"));
        assert!(src.contains("cfg.target_word_offset"));
    }

    #[test]
    fn clamps_target_overflow_to_sentinel_bucket() {
        let src = emit_single_dispatch_sort(&test_layout(), 1_048_576);
        assert!(src.contains("select(tgt, cfg.agent_cap, tgt >= cfg.agent_cap)"));
    }

    #[test]
    fn bounds_live_count_by_ring_capacity() {
        let src = emit_single_dispatch_sort(&test_layout(), 4096);
        assert!(src.contains("const RING_CAP: u32 = 4096u;"));
        assert!(src.contains("min(event_tail[0], RING_CAP)"));
    }

    #[test]
    fn copies_back_only_when_pass_count_is_odd() {
        let src = emit_single_dispatch_sort(&test_layout(), 1_048_576);
        assert!(src.contains("if (!from_ring) {"));
        assert!(src.contains("ring[i] = scratch[i];"));
    }
}
