//! Per-physics-rule invocation counter — research-mode attribution for
//! the resident cascade's GPU cost (2026-04-24).
//!
//! # Design
//!
//! Gated on `ENGINE_GPU_CASCADE_RULE_COUNT=1`. When enabled, the
//! resident physics shader allocates an extra storage buffer at
//! BGL slot 23 holding one `atomic<u32>` per (non-`@cpu_only`) physics
//! rule. Each rule body prepends `atomicAdd(&per_rule_counter[idx], 1u)`
//! after its kind guard (see
//! [`dsl_compiler::emit_physics_wgsl_with_counter`]). At end of
//! `step_batch` we readback the buffer and report per-rule invocation
//! counts.
//!
//! Combined with the existing per-iter GPU timestamps (`cascade iter N`
//! marks), this gives us:
//!   * which rule fires most often → hot suspect
//!   * whether any rule fires unexpectedly often → attribution signal
//!
//! NOT a µs-per-rule number — rules vary in per-invocation cost.
//! Follow-up work (Option B in the research plan) splits the fused
//! dispatcher to get true per-rule µs if a rule looks suspicious.
//!
//! # Revert
//!
//! Production code is unchanged when the env var is unset. The buffer
//! is not allocated, BGL slot 23 is absent from the layout, and the
//! shader emits identical byte output to pre-instrumentation.

#![cfg(feature = "gpu")]

use std::sync::atomic::{AtomicU8, Ordering};

/// Binding slot on the resident physics BGL for the per-rule counter
/// storage buffer. Slot 22 is the alive bitmap; 17 / 18-19 / 20-21 are
/// gold / standing / memory. Slot 23 is the next free slot.
pub const PER_RULE_COUNTER_BINDING: u32 = 23;

/// Opt-in: read `ENGINE_GPU_CASCADE_RULE_COUNT=1` from the environment
/// exactly once per process and cache the result. Production paths do
/// NOT set this var, so behaviour is unchanged.
///
/// Returns `true` iff instrumentation should be compiled in + allocated
/// + bound + read back.
pub fn rule_count_enabled() -> bool {
    static CACHED: AtomicU8 = AtomicU8::new(2); // 0 = off, 1 = on, 2 = uninit
    match CACHED.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("ENGINE_GPU_CASCADE_RULE_COUNT")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false);
            CACHED.store(if on { 1 } else { 0 }, Ordering::Relaxed);
            on
        }
    }
}

/// Byte size of the counter storage buffer for `num_rules` rules. One
/// `atomic<u32>` per rule. Clamped to at least 4 B so the allocator
/// never sees a zero-sized buffer (empty-rule-list edge case).
#[inline]
pub fn counter_bytes(num_rules: u32) -> u64 {
    ((num_rules as u64) * 4).max(4)
}

/// Create a counter storage buffer sized for `num_rules` counters. Zero-
/// initialised by wgpu. Usage: `STORAGE | COPY_SRC | COPY_DST` so the
/// cascade driver can `encoder.clear_buffer` at top-of-batch and the
/// host can `copy_buffer_to_buffer` into a readback staging buffer.
pub fn create_counter_buffer(device: &wgpu::Device, num_rules: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::per_rule_counter"),
        size: counter_bytes(num_rules),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create a readback staging buffer matching `counter_bytes(num_rules)`.
pub fn create_counter_staging(device: &wgpu::Device, num_rules: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::per_rule_counter::staging"),
        size: counter_bytes(num_rules),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Map the staging buffer and decode `num_rules` u32 counts. Blocks on
/// the GPU fence via the caller-supplied device poll. Returns one u32
/// per rule in the order established by the ordered rule-name list the
/// shader was built against.
pub fn read_counters(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    num_rules: u32,
) -> Vec<u32> {
    let bytes = counter_bytes(num_rules);
    let slice = staging.slice(0..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap().expect("per_rule_counter map_async");
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity(num_rules as usize);
    for i in 0..num_rules as usize {
        let base = i * 4;
        let w = u32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]);
        out.push(w);
    }
    drop(data);
    staging.unmap();
    out
}
