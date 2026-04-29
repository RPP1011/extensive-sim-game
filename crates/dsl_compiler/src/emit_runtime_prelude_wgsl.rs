//! Runtime prelude WGSL — shared helpers every prelude-dependent
//! kernel `include_str!`s after its binding declarations.
//!
//! Recovered from the pre-T16 hand-written `EVENT_RING_WGSL` constant
//! (`crates/engine_gpu/src/event_ring.rs:839` at commit `4474566c~1`)
//! and adapted to the post-T16 raw-u32 binding layout.
//!
//! ## Layout contract
//!
//! Consumer kernels MUST declare the following bindings BEFORE
//! `include_str!`ing this prelude:
//!
//! - `event_ring_records: array<u32>` (read_write)
//! - `event_ring_tail: atomic<u32>` (read_write)
//!
//! Each `EventRecord` occupies `RECORD_U32_STRIDE` consecutive `u32`
//! slots in `event_ring_records`. Layout:
//!
//! ```text
//! [0]      kind
//! [1]      tick
//! [2..10]  payload[0..8]
//! ```
//!
//! `RECORD_U32_STRIDE = 2 + PAYLOAD_WORDS = 10`. Matches the host-side
//! `EventRecord` Pod struct in `engine_gpu_rules`.
//!
//! ## What this module emits
//!
//! Two functions are exposed:
//!
//! - [`emit_runtime_prelude_wgsl`] — emits `gpu_emit_event` plus the
//!   per-kind helpers (`gpu_emit_agent_moved`, etc.). The kernel
//!   includes this verbatim.
//! - [`emit_runtime_prelude_consts`] — emits the host-substituted
//!   capacity / payload-words consts. The kernel includes this BEFORE
//!   the prelude body so the consts are in scope.

/// Number of u32 words per `EventRecord` (kind + tick + payload[8]).
/// Mirror of `engine::event::PAYLOAD_WORDS + 2`.
pub const RECORD_U32_STRIDE: u32 = 10;

/// The `EVENT_RING_PAYLOAD_WORDS` host-substituted const value.
/// Mirror of `engine::event::PAYLOAD_WORDS`.
pub const PAYLOAD_WORDS: u32 = 8;

/// Emit the prelude consts. Caller substitutes the per-kernel
/// `event_ring_capacity` (in records, not u32 words). Sits at the top
/// of every prelude-dependent shader, before the kernel's binding
/// declarations.
pub fn emit_runtime_prelude_consts(event_ring_capacity: u32) -> String {
    format!(
        "// Runtime-prelude consts (Stream B prelude module).\n\
const EVENT_RING_CAP: u32 = {event_ring_capacity}u;\n\
const EVENT_RING_PAYLOAD_WORDS: u32 = {PAYLOAD_WORDS}u;\n\
const RECORD_U32_STRIDE: u32 = {RECORD_U32_STRIDE}u;\n"
    )
}

/// Emit the prelude body — `gpu_emit_event` + per-kind helpers. Sits
/// AFTER the kernel's `event_ring_records` + `event_ring_tail`
/// binding declarations.
pub fn emit_runtime_prelude_wgsl() -> String {
    String::from(
        "// Runtime-prelude body (Stream B prelude module).\n\
//\n\
// Every per-kind helper packs typed args into the fixed 10-word\n\
// EventRecord layout the CPU drain expects:\n\
//   [0]      kind\n\
//   [1]      tick\n\
//   [2..10]  payload[0..8]\n\
//\n\
// Returns the slot index the record landed in, or 0xFFFFFFFFu on\n\
// overflow. Silent drop on full ring — the CPU drain sees tail > cap\n\
// and flips the overflow flag. We deliberately don't retry / spin\n\
// because that would serialise emitters.\n\
\n\
fn gpu_emit_event(kind: u32, tick: u32,\n\
                  p0: u32, p1: u32, p2: u32, p3: u32,\n\
                  p4: u32, p5: u32, p6: u32, p7: u32) -> u32 {\n\
    let idx = atomicAdd(&event_ring_tail, 1u);\n\
    if (idx >= EVENT_RING_CAP) {\n\
        return 0xFFFFFFFFu;\n\
    }\n\
    let base = idx * RECORD_U32_STRIDE;\n\
    event_ring_records[base + 0u] = kind;\n\
    event_ring_records[base + 1u] = tick;\n\
    event_ring_records[base + 2u] = p0;\n\
    event_ring_records[base + 3u] = p1;\n\
    event_ring_records[base + 4u] = p2;\n\
    event_ring_records[base + 5u] = p3;\n\
    event_ring_records[base + 6u] = p4;\n\
    event_ring_records[base + 7u] = p5;\n\
    event_ring_records[base + 8u] = p6;\n\
    event_ring_records[base + 9u] = p7;\n\
    return idx;\n\
}\n\
\n\
// Per-kind encoders. Argument names match the Event::<Variant> field\n\
// names; the helper packs them into the payload layout the CPU drain\n\
// expects. WGSL reserves `target` so helpers use `target_id` etc.\n\
\n\
fn gpu_emit_agent_moved(actor: u32, tick: u32,\n\
                        fx: f32, fy: f32, fz: f32,\n\
                        lx: f32, ly: f32, lz: f32) -> u32 {\n\
    return gpu_emit_event(0u, tick, actor,\n\
                          bitcast<u32>(fx), bitcast<u32>(fy), bitcast<u32>(fz),\n\
                          bitcast<u32>(lx), bitcast<u32>(ly), bitcast<u32>(lz),\n\
                          0u);\n\
}\n\
\n\
fn gpu_emit_agent_attacked(actor: u32, target_id: u32, damage: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(1u, tick, actor, target_id, bitcast<u32>(damage),\n\
                          0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_agent_died(agent_id: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(2u, tick, agent_id, 0u, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_agent_fled(agent_id: u32, tick: u32,\n\
                       fx: f32, fy: f32, fz: f32,\n\
                       tx: f32, ty: f32, tz: f32) -> u32 {\n\
    return gpu_emit_event(3u, tick, agent_id,\n\
                          bitcast<u32>(fx), bitcast<u32>(fy), bitcast<u32>(fz),\n\
                          bitcast<u32>(tx), bitcast<u32>(ty), bitcast<u32>(tz),\n\
                          0u);\n\
}\n\
\n\
fn gpu_emit_agent_ate(agent_id: u32, delta: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(4u, tick, agent_id, bitcast<u32>(delta),\n\
                          0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_agent_drank(agent_id: u32, delta: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(5u, tick, agent_id, bitcast<u32>(delta),\n\
                          0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_agent_rested(agent_id: u32, delta: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(6u, tick, agent_id, bitcast<u32>(delta),\n\
                          0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_agent_cast(actor: u32, ability: u32, target_id: u32, depth: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(7u, tick, actor, ability, target_id, depth,\n\
                          0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_opportunity_attack(actor: u32, target_id: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(25u, tick, actor, target_id, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_effect_damage(actor: u32, target_id: u32, amount: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(26u, tick, actor, target_id, bitcast<u32>(amount),\n\
                          0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_effect_heal(actor: u32, target_id: u32, amount: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(27u, tick, actor, target_id, bitcast<u32>(amount),\n\
                          0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_effect_shield(actor: u32, target_id: u32, amount: f32, tick: u32) -> u32 {\n\
    return gpu_emit_event(28u, tick, actor, target_id, bitcast<u32>(amount),\n\
                          0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_effect_stun(actor: u32, target_id: u32, expires_at_tick: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(29u, tick, actor, target_id, expires_at_tick,\n\
                          0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_cast_depth_exceeded(actor: u32, ability: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(33u, tick, actor, ability, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_engagement_committed(actor: u32, target_id: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(34u, tick, actor, target_id, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_engagement_broken(actor: u32, former_target: u32, reason: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(35u, tick, actor, former_target, reason,\n\
                          0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_fear_spread(observer: u32, dead_kin: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(36u, tick, observer, dead_kin, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_pack_assist(observer: u32, target_id: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(37u, tick, observer, target_id, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n\
\n\
fn gpu_emit_rally_call(observer: u32, wounded_kin: u32, tick: u32) -> u32 {\n\
    return gpu_emit_event(38u, tick, observer, wounded_kin, 0u, 0u, 0u, 0u, 0u, 0u);\n\
}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic mini-shader that declares the prelude's
    /// expected bindings + a no-op entry point that calls `gpu_emit_event`.
    /// Asserts the combined WGSL parses via naga.
    fn synthetic_test_shader() -> String {
        let mut out = String::new();
        out.push_str(&emit_runtime_prelude_consts(4096));
        out.push_str("@group(0) @binding(0) var<storage, read_write> event_ring_records: array<u32>;\n");
        out.push_str("@group(0) @binding(1) var<storage, read_write> event_ring_tail: atomic<u32>;\n");
        out.push_str(&emit_runtime_prelude_wgsl());
        out.push_str(
            "\n@compute @workgroup_size(1)\n\
fn cs_test(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let _idx = gpu_emit_agent_moved(1u, 0u, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);\n\
}\n",
        );
        out
    }

    #[test]
    fn prelude_naga_parses_with_synthetic_bindings() {
        let src = synthetic_test_shader();
        naga::front::wgsl::parse_str(&src)
            .map_err(|e| format!("{src}\n\n--- naga error ---\n{}", e.emit_to_string(&src)))
            .expect("prelude + synthetic bindings should parse");
    }

    #[test]
    fn prelude_emits_per_kind_helpers() {
        let src = emit_runtime_prelude_wgsl();
        for helper in &[
            "gpu_emit_event",
            "gpu_emit_agent_moved",
            "gpu_emit_agent_attacked",
            "gpu_emit_agent_died",
            "gpu_emit_agent_fled",
            "gpu_emit_effect_damage",
            "gpu_emit_engagement_committed",
        ] {
            assert!(src.contains(helper), "expected {helper} in prelude");
        }
    }

    #[test]
    fn record_stride_matches_host_layout() {
        // Host: kind + tick + PAYLOAD_WORDS u32 fields per record.
        assert_eq!(RECORD_U32_STRIDE, 2 + PAYLOAD_WORDS);
        // Mirror of engine::event::PAYLOAD_WORDS = 8.
        assert_eq!(PAYLOAD_WORDS, 8);
    }
}
