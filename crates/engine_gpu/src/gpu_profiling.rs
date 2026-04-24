//! Perf Stage A — GPU-resident timestamp instrumentation.
//!
//! Owns a `wgpu::QuerySet` + resolve/readback buffer pair and exposes
//! `mark(encoder, label)` + `finish(encoder)` helpers that plumb
//! `wgpu::QueryType::Timestamp` writes into a caller-supplied command
//! encoder. After `queue.submit + device.poll(Wait)` the caller reads
//! back the raw u64 timestamps and converts deltas to microseconds via
//! `Queue::get_timestamp_period()`.
//!
//! ## Fallback
//!
//! If the adapter does not advertise `Features::TIMESTAMP_QUERY` +
//! `Features::TIMESTAMP_QUERY_INSIDE_ENCODERS` the profiler is a no-op:
//! `mark` silently drops the label (keeping the wall-clock-only perf
//! path unchanged) and `read_phase_us` returns an empty vector. Every
//! call site pays at most one branch on the `enabled` flag.
//!
//! ## Usage pattern inside `step_batch`
//!
//! ```ignore
//! profiler.begin_frame();
//! profiler.mark(&mut encoder, "fused_unpack_begin");
//! // ... encode fused_unpack kernel ...
//! profiler.mark(&mut encoder, "fused_unpack_end");
//! // ... more phases ...
//! profiler.finish_frame(&mut encoder);
//! queue.submit([encoder.finish()]);
//! device.poll(wgpu::PollType::Wait);
//! let phases = profiler.read_phase_us(&device, &queue);
//! ```

#![cfg(feature = "gpu")]

/// Max number of timestamps a single batch call can record. Sized for
/// 8 cascade iters × 2 stamps/iter + 6 phase bookends × 2 stamps = 28,
/// rounded up to the nearest multiple of `QUERY_RESOLVE_BUFFER_ALIGNMENT`
/// with headroom.
const MAX_TIMESTAMPS: u32 = 64;

/// Size in bytes of one u64 timestamp slot — matches `wgpu::QUERY_SIZE`.
const TIMESTAMP_BYTES: u64 = 8;

/// GPU-resident profiler. Wraps a `wgpu::QuerySet` + resolve buffer +
/// mappable readback buffer; owns a label table so the caller can map
/// raw query indices back to phase names when reading back.
///
/// Cheap to keep around: ~1 KB of device memory total (64 × 8 B query
/// set + 64 × 8 B storage resolve + 64 × 8 B readback).
pub struct GpuProfiler {
    /// Adapter+device actually advertise `TIMESTAMP_QUERY` +
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS`. `false` means every method
    /// is a cheap no-op and `read_phase_us` returns an empty Vec.
    enabled: bool,
    /// Query set of `MAX_TIMESTAMPS` timestamp slots. `None` when the
    /// adapter lacks the feature.
    query_set: Option<wgpu::QuerySet>,
    /// Storage buffer (`QUERY_RESOLVE`) that `resolve_query_set` writes
    /// raw u64 timestamp ticks into.
    resolve_buf: Option<wgpu::Buffer>,
    /// Mappable readback of `resolve_buf` that the host maps to read
    /// the u64 values after `queue.submit + device.poll(Wait)`.
    readback_buf: Option<wgpu::Buffer>,
    /// Labels accumulated for the current frame. `labels[i]` is the
    /// name passed to the `mark()` call that wrote slot `i`. Cleared
    /// by `begin_frame`.
    labels: Vec<&'static str>,
    /// Next free query index — incremented by each `mark()` call,
    /// reset to 0 by `begin_frame`. Capped at `MAX_TIMESTAMPS`;
    /// further `mark()` calls silently drop (logs once).
    next_index: u32,
    /// `Queue::get_timestamp_period()` — nanoseconds per timestamp
    /// tick. Cached because the queue handle isn't threaded into
    /// `read_phase_us` via the encoder.
    timestamp_period_ns: f32,
}

/// One resolved phase from a frame's timestamp readback.
#[derive(Clone, Debug)]
pub struct PhaseSample {
    pub label: &'static str,
    /// Raw timestamp tick read from the query set (u64 counter, units
    /// are device-specific — multiply by `timestamp_period_ns` to get
    /// nanoseconds).
    pub raw_tick: u64,
}

impl GpuProfiler {
    /// Build a profiler against a device that was created with
    /// `TIMESTAMP_QUERY` + `TIMESTAMP_QUERY_INSIDE_ENCODERS` enabled.
    /// Pass `enabled=false` to get a pure no-op profiler on adapters
    /// that don't support timestamp queries.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) -> Self {
        if !enabled {
            return Self {
                enabled: false,
                query_set: None,
                resolve_buf: None,
                readback_buf: None,
                labels: Vec::new(),
                next_index: 0,
                timestamp_period_ns: 0.0,
            };
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("engine_gpu::gpu_profiling::query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_TIMESTAMPS,
        });

        let resolve_bytes = MAX_TIMESTAMPS as u64 * TIMESTAMP_BYTES;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::gpu_profiling::resolve_buf"),
            size: resolve_bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::gpu_profiling::readback_buf"),
            size: resolve_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            enabled: true,
            query_set: Some(query_set),
            resolve_buf: Some(resolve_buf),
            readback_buf: Some(readback_buf),
            labels: Vec::with_capacity(MAX_TIMESTAMPS as usize),
            next_index: 0,
            timestamp_period_ns: queue.get_timestamp_period(),
        }
    }

    /// True iff timestamp queries are wired up. Callers use this to
    /// skip eprintln sections cleanly when running on a software
    /// adapter.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Zero the frame-scoped state — label table + next_index cursor.
    /// Call at the top of each `step_batch` invocation.
    pub fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }
        self.labels.clear();
        self.next_index = 0;
    }

    /// Emit a `write_timestamp` into the encoder at the current
    /// recording position. `label` is stored verbatim (must be
    /// `'static`) so the readback can surface it without allocating.
    ///
    /// No-op when the profiler is disabled or the query set is full.
    pub fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        if !self.enabled {
            return;
        }
        if self.next_index >= MAX_TIMESTAMPS {
            // Ran out of slots — silently drop. The test harness logs
            // a warning via `read_phase_us` if this ever fires.
            return;
        }
        let query_set = self
            .query_set
            .as_ref()
            .expect("query_set present when enabled");
        encoder.write_timestamp(query_set, self.next_index);
        self.labels.push(label);
        self.next_index += 1;
    }

    /// Encode the resolve + copy-to-readback commands for the frame's
    /// recorded timestamps. Must be the LAST calls into the encoder
    /// before `finish()` so every `mark()` above lands in the readback.
    pub fn finish_frame(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.next_index == 0 {
            return;
        }
        let query_set = self.query_set.as_ref().expect("query_set");
        let resolve_buf = self.resolve_buf.as_ref().expect("resolve_buf");
        let readback_buf = self.readback_buf.as_ref().expect("readback_buf");

        encoder.resolve_query_set(query_set, 0..self.next_index, resolve_buf, 0);
        let copy_bytes = self.next_index as u64 * TIMESTAMP_BYTES;
        encoder.copy_buffer_to_buffer(resolve_buf, 0, readback_buf, 0, copy_bytes);
    }

    /// Map the readback buffer and convert the resolved u64 ticks into
    /// per-phase microseconds. Returns a vec of `(label, delta_us)`
    /// pairs where each entry is the time between consecutive `mark`
    /// calls — entry `i` covers the interval from `mark` `i` to
    /// `mark` `i+1`, with `label = labels[i]` (the OPENING mark's
    /// label, so bracket your phases as
    /// `mark(begin); ...phase...; mark(end)` and expect one entry per
    /// PAIR, labelled with the "begin" name).
    ///
    /// Must be called AFTER `queue.submit + device.poll(Wait)` so the
    /// readback buffer is populated.
    pub fn read_phase_us(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Vec<(&'static str, u64)> {
        if !self.enabled || self.next_index < 2 {
            return Vec::new();
        }

        let readback_buf = self.readback_buf.as_ref().expect("readback_buf");
        let byte_len = self.next_index as u64 * TIMESTAMP_BYTES;
        let slice = readback_buf.slice(..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let map_result = match rx.recv() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        if map_result.is_err() {
            return Vec::new();
        }
        let data = slice.get_mapped_range();
        let casted: &[u64] = bytemuck::cast_slice(&data);
        let ticks: Vec<u64> = casted[..self.next_index as usize].to_vec();
        drop(data);
        readback_buf.unmap();

        let mut out = Vec::with_capacity(self.labels.len().saturating_sub(1));
        for i in 0..(ticks.len().saturating_sub(1)) {
            let delta_ticks = ticks[i + 1].saturating_sub(ticks[i]);
            let ns = (delta_ticks as f64) * (self.timestamp_period_ns as f64);
            let us = (ns / 1000.0).round() as u64;
            let label = self.labels.get(i).copied().unwrap_or("<anon>");
            out.push((label, us));
        }
        out
    }
}

/// Probe the adapter for the two timestamp features the profiler
/// needs. Used by `GpuBackend::new_async` to decide whether to request
/// the features on the device and to wire the profiler into the
/// resident path.
///
/// Returns `true` only when both `TIMESTAMP_QUERY` and
/// `TIMESTAMP_QUERY_INSIDE_ENCODERS` are advertised — the profiler
/// places `write_timestamp` calls outside compute passes, which is the
/// inside-encoders case.
pub fn adapter_supports_timestamps(features: wgpu::Features) -> bool {
    features.contains(wgpu::Features::TIMESTAMP_QUERY)
        && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
}

/// Static per-cascade-iteration begin labels for indices 0..7. Used
/// inside the cascade driver so profiler labels remain `&'static str`
/// without heap allocation. Matches `MAX_CASCADE_ITERATIONS = 8`.
pub const CASCADE_ITER_BEGIN_LABELS: [&str; 8] = [
    "cascade iter 0",
    "cascade iter 1",
    "cascade iter 2",
    "cascade iter 3",
    "cascade iter 4",
    "cascade iter 5",
    "cascade iter 6",
    "cascade iter 7",
];

/// Matching "end" labels for each cascade iteration. Each phase entry
/// in `read_phase_us` is `(begin_label, delta_us)` — the "end" labels
/// are only used as the final sentinel at the close of iter N so the
/// last interval resolves cleanly.
pub const CASCADE_ITER_END_LABELS: [&str; 8] = [
    "cascade iter 0 end",
    "cascade iter 1 end",
    "cascade iter 2 end",
    "cascade iter 3 end",
    "cascade iter 4 end",
    "cascade iter 5 end",
    "cascade iter 6 end",
    "cascade iter 7 end",
];

