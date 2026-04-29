//! WGSL emitter for `AppendEventsKernel` — copies events from the
//! per-tick apply-path ring (`source_ring`) to the per-batch
//! accumulator ring (`batch_ring`), advancing the latter's atomic
//! tail.
//!
//! Recovered from the pre-T16 hand-written `APPEND_EVENTS_WGSL`
//! constant (`crates/engine_gpu/src/cascade_resident.rs:552` at
//! commit `4474566c~1`) and adapted to the post-T16 raw-u32 binding
//! layout.
//!
//! ## Adaptations vs. pre-T16
//!
//! - Pre-T16 named bindings `apply_tail`, `apply_records`,
//!   `batch_tail`, `batch_records`. New BGL renames to `source_ring`,
//!   `source_tail`, `batch_ring`, `batch_tail` (semantically identical).
//! - Pre-T16's `apply_tail` was `array<atomic<u32>>` with `atomicLoad`;
//!   new BGL declares `source_tail: array<u32>` (read). Replaced with
//!   raw read.
//! - `batch_tail` remains `atomic<u32>` (matches pre-T16 semantics —
//!   we still need atomic bump to claim a destination slot).
//!
//! ## Layout contract
//!
//! Each event record is 10 u32s (kind + tick + payload[8]). Mirrors
//! `engine::event::PAYLOAD_WORDS + 2` and the runtime prelude's
//! `RECORD_U32_STRIDE`. Every thread copies one record.

const RECORD_U32_STRIDE: u32 = 10;

/// Emit the body of `engine_gpu_rules/src/append_events.wgsl`.
pub fn emit_append_events_wgsl() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read>       source_ring: array<u32>;\n\
@group(0) @binding(1) var<storage, read>       source_tail: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> batch_ring:  array<u32>;\n\
@group(0) @binding(3) var<storage, read_write> batch_tail:  atomic<u32>;\n\
struct AppendEventsCfg {{ source_capacity: u32, batch_capacity: u32, _pad0: u32, _pad1: u32 }};\n\
@group(0) @binding(4) var<uniform>             cfg:         AppendEventsCfg;\n\
\n\
const RECORD_U32_STRIDE: u32 = {RECORD_U32_STRIDE}u;\n\
\n\
@compute @workgroup_size(64)\n\
fn cs_append_events(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let n = source_tail[0];\n\
    let i = gid.x;\n\
    if (i >= n) {{ return; }}\n\
    let dst = atomicAdd(&batch_tail, 1u);\n\
    if (dst >= cfg.batch_capacity) {{ return; }}\n\
    let src_off = i * RECORD_U32_STRIDE;\n\
    let dst_off = dst * RECORD_U32_STRIDE;\n\
    for (var w: u32 = 0u; w < RECORD_U32_STRIDE; w = w + 1u) {{\n\
        batch_ring[dst_off + w] = source_ring[src_off + w];\n\
    }}\n\
}}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_events_naga_parses() {
        let src = emit_append_events_wgsl();
        naga::front::wgsl::parse_str(&src)
            .map_err(|e| {
                format!("--- WGSL ---\n{src}\n--- naga error ---\n{}", e.emit_to_string(&src))
            })
            .expect("emit_append_events_wgsl should parse cleanly");
    }
}
