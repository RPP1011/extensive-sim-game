//! Task #79 SP-1/SP-2 — GPU side storage for `@symmetric_pair_topk` views.
//!
//! Companion to the CPU `Standing` struct at
//! `crates/engine/src/generated/views/standing.rs` and the WGSL fold
//! kernel source emitted by
//! `crates/dsl_compiler/src/emit_view_wgsl.rs::emit_symmetric_pair_topk_fold_wgsl`.
//!
//! # Binding layout (must stay in lockstep with the resident physics BGL)
//!
//!   * slot 18 — `standing_records_buf` : `array<StandingEdge>`, flat,
//!     indexed as `owner_slot * K + slot_idx`. Sized `agent_cap * K *
//!     sizeof(StandingEdge) = agent_cap * K * 16` bytes.
//!   * slot 19 — `standing_counts_buf`  : `array<atomic<u32>>`, one
//!     `u32` per owner slot. Sized `agent_cap * 4` bytes.
//!
//! Slot 17 is `gold_buf` (Task 3.4). Slots 20+ are reserved for
//! Subsystem 3 (ability-eval GPU port). Keeping standing at 18/19
//! leaves room below for any tightly-coupled cold-state additions
//! without renumbering.
//!
//! # Byte layout: `StandingEdgeGpu` (16 bytes)
//!
//! Matches the WGSL struct emitted by `emit_symmetric_pair_topk_fold_wgsl`:
//!
//! ```text
//!   offset 0: other        : u32 (raw AgentId of higher-id endpoint; 0 = empty slot)
//!   offset 4: value        : i32 (clamped to [-1000, 1000])
//!   offset 8: anchor_tick  : u32 (tick of last write)
//!   offset 12: _pad        : u32 (WGSL 16-byte alignment)
//! ```
//!
//! The CPU `StandingEdge` struct is 12 bytes; GPU padding to 16 B
//! matches WGSL's struct layout rules (multiples of the largest member
//! alignment, which for any `u32`/`i32` member array element is 4 B
//! natural but is rounded up to 16 B because the array itself is bound
//! as a storage buffer — see the WGSL spec and the kernel's
//! `struct StandingEdge { ...; _pad: u32 }` declaration).
//!
//! # Canonicalisation rule
//!
//! Per the WGSL fold kernel (`emit_view_wgsl.rs:1273-1292`):
//!
//!   - The fold canonicalises `(a, b) -> (min, max)`.
//!   - `owner = min(a, b)`; the edge is stored only on the lower-id
//!     endpoint's row. The readback parser mirrors by storing only at
//!     `owner` and relying on `Standing::get(a, b)` to canonicalise the
//!     query pair (CPU side already does this — the struct is ignorant
//!     of where the edge actually lives).
//!
//! # Concurrency caveats (per the emitted kernel)
//!
//!   - Update-in-place (`slots[i].value += delta`) is non-atomic. Safe
//!     for the current physics cascade because each
//!     `EffectStandingDelta` event is processed by exactly one thread
//!     per iteration and the same canonical pair is rarely co-fired
//!     within one iteration (would require two rules both emitting
//!     standing deltas for the same pair in the same tick).
//!   - Reserve-slot races serialise through `atomicAdd` on
//!     `standing_counts_buf`. Losers fall through to the evict arm.
//!   - Evict scan+write is non-atomic; concurrent evictions on the
//!     same row can lose updates. Audit in a future phase if real
//!     usage becomes racy.
//!
//! # Symmetry on readback
//!
//! The GPU fold writes to the OWNER (canonical-min) row only. On
//! readback we rebuild `Standing.slots` the same way; `Standing::get`
//! and `adjust` already canonicalise input pairs so symmetric reads
//! work without special handling.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};

use engine_rules::views::standing::{Standing, StandingEdge};
use engine::ids::AgentId;

/// Slot count per owner — the `K` from `symmetric_pair_topk(K=8)` on
/// the standing view.
pub const STANDING_K: u32 = 8;

/// GPU-side edge struct. 16 bytes to match the WGSL struct layout the
/// fold kernel emits. CPU `StandingEdge` is 12 bytes; we add a trailing
/// `_pad` on upload / strip it on readback.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct StandingEdgeGpu {
    pub other: u32,
    pub value: i32,
    pub anchor_tick: u32,
    pub _pad: u32,
}

/// Resident-path storage for the `standing` view. Two buffers owned by
/// `ResidentPathContext`:
///
///   * `records_buf` — flat `[StandingEdgeGpu; agent_cap * K]`.
///   * `counts_buf`  — one `atomic<u32>` per owner slot.
///
/// Allocated on first `ensure_resident_init` call + on agent_cap grow;
/// bound into the resident physics BGL at slots 18 / 19; read back in
/// `snapshot()` and merged into `state.views.standing`.
pub struct ViewStorageSymmetricPair {
    pub records_buf: wgpu::Buffer,
    pub counts_buf: wgpu::Buffer,
    pub agent_cap: u32,
    pub k: u32,
}

impl ViewStorageSymmetricPair {
    /// Byte size of the records buffer for `agent_cap * k` slots.
    pub fn records_bytes(agent_cap: u32, k: u32) -> u64 {
        (agent_cap as u64)
            .saturating_mul(k as u64)
            .saturating_mul(std::mem::size_of::<StandingEdgeGpu>() as u64)
            .max(std::mem::size_of::<StandingEdgeGpu>() as u64)
    }

    /// Byte size of the counts buffer for `agent_cap` owners.
    pub fn counts_bytes(agent_cap: u32) -> u64 {
        (agent_cap as u64).saturating_mul(4).max(4)
    }

    /// Allocate storage for `agent_cap` owners with `K = k` slots
    /// each. Buffers are zero-initialised by wgpu (no
    /// `mapped_at_creation`); the caller's `upload_from_cpu` populates
    /// from a CPU `Standing` struct.
    pub fn new(device: &wgpu::Device, agent_cap: u32, k: u32) -> Self {
        let records_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::view_storage_symmetric_pair::records"),
            size: Self::records_bytes(agent_cap, k),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::view_storage_symmetric_pair::counts"),
            size: Self::counts_bytes(agent_cap),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            records_buf,
            counts_buf,
            agent_cap,
            k,
        }
    }

    /// Serialise CPU `Standing` into the flat GPU arrays.
    ///
    /// Walk every owner slot `0..agent_cap`; for each owner that has
    /// a row in `cpu.slots`, pack its 8 `StandingEdge`s into contiguous
    /// GPU slots and write the population count. Unused owners (ids
    /// beyond `cpu.len()`) get zero-initialised records + count = 0.
    ///
    /// The per-row population count is `N` where the first `N` entries
    /// of the CPU row have `other != 0`. The CPU-side `adjust` keeps
    /// the row compacted only in insertion order — evict writes
    /// in-place, so the populated subset isn't necessarily a prefix.
    /// We pack it to a prefix here to match the GPU kernel's
    /// `scan_len = min(count, K)` contract.
    pub fn upload_from_cpu(&self, queue: &wgpu::Queue, cpu: &Standing) {
        let k = self.k as usize;
        let cap = self.agent_cap as usize;

        let mut records = vec![StandingEdgeGpu::default(); cap * k];
        let mut counts: Vec<u32> = vec![0; cap];

        for owner_slot in 0..cap {
            // CPU row: owner = owner_slot + 1 (1-based). We scan all
            // higher-id agents up to `cap` as potential `other` partners
            // because `Standing.slots` is private — the public surface
            // is `get(owner, other)` which returns the stored value on
            // the canonical row. The GPU kernel owns canonicalisation
            // identically; only pairs with `other > owner` live on this
            // row.
            let row_opt = get_cpu_row(cpu, owner_slot, cap);
            if let Some(row) = row_opt {
                let mut write_idx = 0usize;
                for edge in row.iter() {
                    if edge.other == 0 {
                        continue;
                    }
                    if write_idx >= k {
                        break;
                    }
                    let dst = owner_slot * k + write_idx;
                    records[dst] = StandingEdgeGpu {
                        other: edge.other,
                        value: edge.value,
                        anchor_tick: edge.anchor_tick,
                        _pad: 0,
                    };
                    write_idx += 1;
                }
                counts[owner_slot] = write_idx as u32;
            }
        }

        if !records.is_empty() {
            queue.write_buffer(&self.records_buf, 0, bytemuck::cast_slice(&records));
        }
        if !counts.is_empty() {
            queue.write_buffer(&self.counts_buf, 0, bytemuck::cast_slice(&counts));
        }
    }

    /// Read the GPU storage back and rebuild the CPU `Standing`
    /// struct. Blocking — allocates a throwaway staging buffer per
    /// call. Fine for `snapshot()` (Option (a) per the plan; mirrors
    /// the `gold_buf` readback pattern).
    pub fn readback_into_cpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        out: &mut Standing,
    ) -> Result<(), String> {
        let cap = self.agent_cap as usize;
        let k = self.k as usize;
        if cap == 0 || k == 0 {
            *out = Standing::new();
            return Ok(());
        }

        let records_bytes = Self::records_bytes(self.agent_cap, self.k) as usize;
        let counts_bytes = Self::counts_bytes(self.agent_cap) as usize;

        let records: Vec<StandingEdgeGpu> = crate::gpu_util::readback::readback_typed(
            device,
            queue,
            &self.records_buf,
            records_bytes,
        )
        .map_err(|e| format!("standing records readback: {e}"))?;
        let counts: Vec<u32> = crate::gpu_util::readback::readback_typed(
            device,
            queue,
            &self.counts_buf,
            counts_bytes,
        )
        .map_err(|e| format!("standing counts readback: {e}"))?;

        // Rebuild. Clear the CPU struct and repopulate by invoking
        // `adjust` with the correct canonical pairs + absolute values.
        // The GPU kernel already did the find-or-evict; the GPU slot
        // values ARE the final values, so we bypass `adjust`'s delta
        // semantics by constructing the struct through the row-level
        // reset helper below.
        //
        // To stay consistent with CPU `Standing`'s `adjust`, we use
        // its public API: `adjust(owner, other, delta=value, tick)`
        // from an empty baseline. `owner` here is 1-based (AgentId
        // raw), `other_id` is the stored `edge.other`.
        *out = Standing::new();
        for owner_slot in 0..cap {
            let count = counts.get(owner_slot).copied().unwrap_or(0).min(self.k) as usize;
            if count == 0 {
                continue;
            }
            // Owner AgentId is `owner_slot + 1` (1-based).
            let owner_raw = (owner_slot as u32).saturating_add(1);
            let Some(owner_id) = AgentId::new(owner_raw) else {
                continue;
            };
            let row_base = owner_slot * k;
            for slot_i in 0..count {
                let edge = records[row_base + slot_i];
                if edge.other == 0 {
                    continue;
                }
                let Some(other_id) = AgentId::new(edge.other) else {
                    continue;
                };
                // `adjust` canonicalises (a, b) → (min, max). We pass
                // (owner_id, other_id); `owner_id < other_id` is the
                // canonical invariant the GPU kernel maintained (min
                // is the owner), so `adjust` lands the delta exactly
                // on slot `(owner_slot, other_id)`.
                //
                // Apply as a delta from 0 baseline; clamp is a no-op
                // because the GPU kernel already clamped. anchor_tick
                // flows through.
                out.adjust(owner_id, other_id, edge.value, edge.anchor_tick);
            }
        }
        Ok(())
    }
}

/// Return a slice of `StandingEdge` for the given owner slot by
/// scanning every potential `other` partner up to `other_cap` (1-based
/// id max). Returns None if the owner slot is beyond `cpu.len()`.
///
/// Works around the private `slots` field on `Standing` by going
/// through the public `get` accessor — one edge at a time. O(N²) on
/// upload, which only runs on first resident-init / agent_cap grow.
/// If this becomes hot we can add a public bulk accessor upstream.
fn get_cpu_row(
    cpu: &Standing,
    owner_slot: usize,
    other_cap: usize,
) -> Option<[StandingEdge; Standing::K]> {
    if owner_slot >= cpu.len() {
        return None;
    }
    // Owner AgentId is 1-based: owner_slot + 1.
    let owner_raw = (owner_slot as u32).saturating_add(1);
    let Some(owner_id) = AgentId::new(owner_raw) else {
        return Some([StandingEdge::default(); Standing::K]);
    };

    let mut row: [StandingEdge; Standing::K] = [StandingEdge::default(); Standing::K];
    let mut write_idx = 0usize;
    // The CPU owner is `min(a, b)` canonically — only pairs with
    // `other > owner` live on this row. Iterate every potential
    // `other` up to `other_cap` (1-based, inclusive).
    let max_other = other_cap as u32;
    for other_raw in (owner_raw + 1)..=max_other {
        let Some(other_id) = AgentId::new(other_raw) else {
            continue;
        };
        let v = cpu.get(owner_id, other_id);
        if v == 0 {
            continue;
        }
        if write_idx >= Standing::K {
            break;
        }
        // We don't know the anchor_tick of the CPU-side edge without
        // a bulk accessor; use 0 here. This is fine because the GPU
        // kernel only uses anchor_tick for decay (not implemented
        // yet) and overwrites it on any mutation.
        row[write_idx] = StandingEdge {
            other: other_raw,
            value: v,
            anchor_tick: 0,
        };
        write_idx += 1;
    }
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue) {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("adapter");
            let adapter_limits = adapter.limits();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("view_storage_symmetric_pair::test::device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("device");
            (device, queue)
        })
    }

    #[test]
    fn standing_edge_gpu_is_16_bytes() {
        assert_eq!(std::mem::size_of::<StandingEdgeGpu>(), 16);
        assert_eq!(std::mem::align_of::<StandingEdgeGpu>(), 4);
    }

    #[test]
    fn new_allocates_records_and_counts() {
        let (device, _queue) = gpu_device_queue();
        let cap = 16u32;
        let storage = ViewStorageSymmetricPair::new(&device, cap, STANDING_K);
        assert_eq!(storage.agent_cap, cap);
        assert_eq!(storage.k, STANDING_K);
        assert_eq!(
            storage.records_buf.size(),
            (cap as u64) * (STANDING_K as u64) * 16,
        );
        assert_eq!(storage.counts_buf.size(), (cap as u64) * 4);
    }

    #[test]
    fn new_handles_zero_cap() {
        let (device, _queue) = gpu_device_queue();
        // Zero cap must still produce a valid (nonzero-size) buffer
        // because wgpu rejects zero-size storage buffers.
        let storage = ViewStorageSymmetricPair::new(&device, 0, STANDING_K);
        assert!(storage.records_buf.size() >= 16);
        assert!(storage.counts_buf.size() >= 4);
    }

    /// Round-trip: seed a CPU `Standing` with a handful of pairs, upload
    /// to GPU storage, read back, assert every (a,b) lookup returns the
    /// original value.
    #[test]
    fn upload_readback_round_trip_single_pair() {
        let (device, queue) = gpu_device_queue();
        let cap = 16u32;
        let storage = ViewStorageSymmetricPair::new(&device, cap, STANDING_K);

        let mut cpu = Standing::new();
        let a = AgentId::new(1).unwrap();
        let b = AgentId::new(5).unwrap();
        cpu.adjust(a, b, 42, 10);

        storage.upload_from_cpu(&queue, &cpu);

        let mut out = Standing::new();
        storage
            .readback_into_cpu(&device, &queue, &mut out)
            .expect("readback");

        assert_eq!(out.get(a, b), 42, "round-trip must preserve value");
        assert_eq!(
            out.get(b, a),
            42,
            "symmetry: get(b,a) == get(a,b) after readback",
        );
    }

    #[test]
    fn upload_readback_round_trip_multiple_pairs() {
        let (device, queue) = gpu_device_queue();
        let cap = 32u32;
        let storage = ViewStorageSymmetricPair::new(&device, cap, STANDING_K);

        let mut cpu = Standing::new();
        let pairs: &[(u32, u32, i32)] = &[
            (1, 2, 50),
            (1, 3, -75),
            (2, 5, 100),
            (3, 7, -200),
            (4, 8, 250),
        ];
        for &(a_raw, b_raw, v) in pairs {
            let a = AgentId::new(a_raw).unwrap();
            let b = AgentId::new(b_raw).unwrap();
            cpu.adjust(a, b, v, 0);
        }

        storage.upload_from_cpu(&queue, &cpu);

        let mut out = Standing::new();
        storage
            .readback_into_cpu(&device, &queue, &mut out)
            .expect("readback");

        for &(a_raw, b_raw, v) in pairs {
            let a = AgentId::new(a_raw).unwrap();
            let b = AgentId::new(b_raw).unwrap();
            assert_eq!(
                out.get(a, b),
                v,
                "pair ({a_raw}, {b_raw}) value mismatch after round-trip",
            );
            assert_eq!(out.get(b, a), v, "symmetry for ({a_raw}, {b_raw})");
        }
    }

    #[test]
    fn empty_cpu_round_trips_empty() {
        let (device, queue) = gpu_device_queue();
        let storage = ViewStorageSymmetricPair::new(&device, 8, STANDING_K);
        let cpu = Standing::new();
        storage.upload_from_cpu(&queue, &cpu);

        let mut out = Standing::new();
        storage
            .readback_into_cpu(&device, &queue, &mut out)
            .expect("readback");

        // No owners have rows → get returns 0 for every pair.
        let a = AgentId::new(1).unwrap();
        let b = AgentId::new(2).unwrap();
        assert_eq!(out.get(a, b), 0);
    }
}
