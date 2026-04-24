//! Research instrumentation — per-view read counters for the scoring kernel.
//!
//! Enabled via `ENGINE_GPU_SCORING_VIEW_COUNT=1`. When set, the scoring
//! WGSL emitter injects a `atomicAdd(&view_read_counter[view_idx], 1u)`
//! at the top of every per-view read helper (see
//! `dsl_compiler::emit_scoring_wgsl::scoring_view_count_enabled`). The
//! engine_gpu side extends the scoring BGL with a slot-24 `array<atomic<u32>>`
//! binding and owns a single storage buffer that is:
//!
//!   * zeroed at the top of every `step_batch` call,
//!   * read back after the batch completes,
//!   * printed by the perf test under `--nocapture`.
//!
//! The buffer is sized to one u32 per non-Lazy view in the scoring
//! kernel's view_specs (alphabetical by `view_name`, matching
//! `dsl_compiler::emit_scoring_wgsl::scoring_view_binding_order`).
//!
//! Production paths (env var unset) allocate nothing, bind nothing, and
//! the shader emits no counter code.
//!
//! **Binding slot**: 24 — after alive_bitmap at slot 22 and the
//! reserved slot 23 for task #96's cascade rule counter. Matches
//! `dsl_compiler::emit_scoring_wgsl::VIEW_READ_COUNTER_BINDING`.

#![cfg(feature = "gpu")]

use dsl_compiler::emit_scoring_wgsl::{
    scoring_view_count_enabled, view_read_counter_slot_count, view_read_counter_view_names,
    VIEW_READ_COUNTER_BINDING,
};
use dsl_compiler::emit_view_wgsl::ViewStorageSpec;

/// Re-export so callers that already import via `crate::view_read_counter`
/// don't need a second import path.
pub const BINDING: u32 = VIEW_READ_COUNTER_BINDING;

/// Cached — identical lookup semantics to
/// [`dsl_compiler::emit_scoring_wgsl::scoring_view_count_enabled`]. Kept
/// re-exported here so engine_gpu callers have one import path.
pub fn enabled() -> bool {
    scoring_view_count_enabled()
}

/// Per-view slot count for the counter buffer. Matches the WGSL emitter's
/// `view_read_counter_slot_count`. Always at least 1 so the buffer
/// allocation never sees a zero size.
pub fn slot_count(specs: &[ViewStorageSpec]) -> u32 {
    view_read_counter_slot_count(specs).max(1)
}

/// Byte size of the counter storage buffer for the given view specs.
pub fn buffer_bytes(specs: &[ViewStorageSpec]) -> u64 {
    slot_count(specs) as u64 * 4
}

/// Create the counter storage buffer. Zero-initialised by wgpu —
/// `step_batch` re-zeros via `clear_buffer` at the top of each batch.
pub fn create_buffer(device: &wgpu::Device, specs: &[ViewStorageSpec]) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::view_read_counter"),
        size: buffer_bytes(specs),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create the readback buffer (MAP_READ + COPY_DST). Separate from the
/// storage buffer so the scoring kernel can write atomically on the GPU
/// while the host reads a mappable copy.
pub fn create_readback(device: &wgpu::Device, specs: &[ViewStorageSpec]) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::view_read_counter::readback"),
        size: buffer_bytes(specs),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Per-view counter labels, in the same order as the counter buffer slots.
/// Used by the report path in `chronicle_batch_perf_n100k` so each row of
/// the readback dump gets the matching view name.
pub fn view_names(specs: &[ViewStorageSpec]) -> Vec<String> {
    view_read_counter_view_names(specs)
}
