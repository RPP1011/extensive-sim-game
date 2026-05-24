//! Shared per-fixture event-ring + view-storage runtime helpers.
//!
//! Three fixture runtimes (predator_prey, particle_collision,
//! crowd_navigation) carry structurally identical event-ring +
//! view-storage allocation + per-tick dispatch chain plumbing —
//! ~150 lines × 3 of mechanical repetition. This module factors
//! the common shape into two reusable structs so adding a fourth
//! event-emitting fixture is `EventRing::new()` + one
//! `ViewStorage::new()` per fold view + a few binding pass-throughs.
//!
//! # What lives here
//!
//! - [`EventRing`]: per-fixture buffers (`event_ring`, `event_tail`,
//!   `event_tail_zero`, `indirect_args_0`, `sim_cfg`). Construct
//!   once at runtime init; `clear_tail_in()` drains the tail at
//!   the start of each `step()` so producers atomicAdd from 0.
//! - [`ViewStorage`]: per-view buffers (`primary` + optional
//!   `anchor`/`ids` + staging) + a host-side `Vec<f32>` cache + a
//!   `readback()` method that encodes the staging copy + map_async
//!   + cast back into the cache.
//!
//! # What stays in the per-fixture runtime
//!
//! - The actual dispatch order — fixtures pick which kernels run,
//!   which view storage each fold writes, which cfg uniforms get
//!   updated per tick. The helpers here own buffers, not policy.
//! - Per-fold cfg buffer ownership (the cfg struct shape is
//!   compiler-emitted per-view, so wrapping it isn't worth the
//!   layer of indirection).

use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

/// Default slot capacity of the per-tick event ring. Mirrors the
/// `if (_slot < 1048576u)` gate the WGSL emit writes per producer
/// site (47 sites in `cg/emit/wgsl_body.rs`). 1 048 576 slots ×
/// 11 u32/slot × 4 bytes = 44 MB per fixture.
///
/// Bumped 65 536 → 1 048 576 (16×) on 2026-05-09 (Task #239) after
/// stress fixtures (`crates/stress_cast_density_runtime/`) showed the
/// previous cap saturated at agent_cap ≈ 480-500 in max-density AOE
/// fixtures — silently dropping ~99.6% of casts at agent_cap=64 000.
/// Real VRAM measurement: heaviest fixture used ~56 MB delta (0.23%
/// of 24 GB), so 40 MB ring is trivial.
///
/// If this cap saturates again, the next bump candidates are: 16M
/// slots (640 MB, 1% of 24 GB VRAM) or per-event-kind ring fanout.
pub const EVENT_RING_CAP_SLOTS: u32 = 1_048_576;

/// u32 words per event record (2 header + 8 payload + 1 seq trailer).
/// Matches `populate_event_kinds` in the CG lowering driver. The 11th
/// word (index 10) is the per-record sequence number written by the CG
/// sort-then-fold pass. Future per-kind ring fanout would surface a
/// per-kind override.
pub const EVENT_STRIDE_U32: u32 = 11;

/// Per-fixture event-ring infrastructure: ring + tail + the
/// pre-built tail-clear source + indirect-args output + a
/// placeholder sim_cfg buffer (folds bind it but the current
/// view bodies don't read fields from it).
///
/// Producers (PerAgent / PerEvent kernels with `emit <Event>`
/// bodies) atomicAdd against [`Self::tail`] to acquire a write
/// slot, then atomicStore the tag/tick/payload into [`Self::ring`].
/// `seed_indirect_0` reads [`Self::tail`] to populate
/// [`Self::indirect_args_0`] for the future
/// `dispatch_workgroups_indirect` wire-up.
pub struct EventRing {
    ring: wgpu::Buffer,
    tail: wgpu::Buffer,
    tail_zero: wgpu::Buffer,
    indirect_args_0: wgpu::Buffer,
    sim_cfg: wgpu::Buffer,
    /// Host-side estimate of the per-tick tail position. Reset to 0
    /// by [`Self::clear_tail_in`]; bumped by [`Self::note_emits`]
    /// after each producer dispatch with the upper bound that
    /// producer could append. Read by [`Self::tail_value`] so a
    /// chronicle / fold dispatch can size its `event_count` to cover
    /// every prior emit in the same tick — without hard-coding
    /// `agent_count * N_producers` constants that silently drop
    /// events when a new producer is added. See
    /// `docs/superpowers/notes/2026-05-04-diplomacy_probe.md` Gap #2.
    tail_estimate: u32,
}

impl EventRing {
    /// Allocate the event-ring infrastructure. `label` becomes a
    /// prefix for every owned buffer's debug label
    /// (`"<label>::event_ring"`, `"<label>::event_tail"`, …).
    pub fn new(gpu: &GpuContext, label: &str) -> Self {
        let event_ring_bytes =
            (EVENT_RING_CAP_SLOTS as u64) * (EVENT_STRIDE_U32 as u64) * 4;
        let ring = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::event_ring")),
            size: event_ring_bytes,
            // COPY_SRC added 2026-05-09 (Phase B voxel integration,
            // task #251) so per-fixture host-side chronicle drains can
            // `copy_buffer_to_buffer(ring, …)` into their staging
            // buffers without rolling a parallel ring like wave_defense.
            // wave_defense's hand-rolled buffer pre-dates this and
            // stays as-is for now (Phase E may converge them).
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tail = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::event_tail")),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tail_zero = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}::event_tail_zero")),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );
        let indirect_args_0 = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::indirect_args_0")),
            size: 12, // 3 × u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_cfg = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}::sim_cfg")),
                contents: &[0u8; 16],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        );
        Self {
            ring,
            tail,
            tail_zero,
            indirect_args_0,
            sim_cfg,
            tail_estimate: 0,
        }
    }

    /// The producer-side ring (`array<atomic<u32>>` on producer
    /// kernels, `array<u32>` on consumer kernels — same physical
    /// buffer, different binding type per kernel).
    pub fn ring(&self) -> &wgpu::Buffer {
        &self.ring
    }

    /// The single-element atomic counter (`array<atomic<u32>, 1>`)
    /// producers atomicAdd against to acquire a write slot.
    pub fn tail(&self) -> &wgpu::Buffer {
        &self.tail
    }

    /// Indirect-args output of `seed_indirect_0` (3 × u32 holding
    /// `(workgroup_x, 1, 1)`). Plumbed into the consumer fold's
    /// future `dispatch_workgroups_indirect` call.
    pub fn indirect_args_0(&self) -> &wgpu::Buffer {
        &self.indirect_args_0
    }

    /// Placeholder sim_cfg buffer required by every fold kernel's
    /// binding layout (slot 5). Today the runtime doesn't write
    /// any fields the folds read, so 16 bytes of zero suffices.
    pub fn sim_cfg(&self) -> &wgpu::Buffer {
        &self.sim_cfg
    }

    /// Encode a per-tick `tail = 0` clear into the supplied
    /// command encoder. Call at the start of every `step()` before
    /// any producer kernel writes events.
    ///
    /// Also resets the host-side [`Self::tail_value`] estimate to 0
    /// so the per-tick chain rebuilds the estimate from scratch via
    /// [`Self::note_emits`] calls.
    pub fn clear_tail_in(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.tail_zero, 0, &self.tail, 0, 4);
        self.tail_estimate = 0;
    }

    /// Encode a per-tick header-word clear for the first `max_slots`
    /// ring entries. Call at the start of every `step()` (before any
    /// producer dispatch) when the producer's per-tick emit count is
    /// **variable** — i.e. less than the upper bound the consumer's
    /// `event_count` cfg field reports. Without this, slots that
    /// stale events from previous ticks left at `kind=1u` get
    /// re-folded by the next consumer dispatch, double-counting them.
    ///
    /// The clear writes a `0u` (`kind == 0u`, the sentinel "no event"
    /// tag) to header word `[0]` of each slot in `0..max_slots`,
    /// overwriting whatever stale tag the previous tick left there.
    /// `max_slots` must be the upper bound on this tick's producer
    /// emit count (typically `agent_count` for one-emit-per-agent
    /// rules); slots beyond that are unaffected (the producer never
    /// reaches them, so they keep whatever tag they last held — which
    /// the consumer's `event_count` upper bound never lets it walk
    /// past).
    ///
    /// **When you don't need this:** producers that emit exactly
    /// `agent_count` events per tick (every alive agent fires
    /// unconditionally — e.g. cooldown_probe's `CheckAndCast` rule)
    /// always overwrite the same slot range, so the consumer never
    /// reads stale headers. Only stochastic / conditional producers
    /// need the per-tick clear.
    ///
    /// See `docs/superpowers/notes/2026-05-04-stochastic_probe.md`
    /// (Gap #2 follow-up) for the discovery.
    pub fn clear_ring_headers_in(
        &self,
        gpu: &crate::gpu::GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        max_slots: u32,
    ) {
        if max_slots == 0 {
            return;
        }
        // Lazy-init a zero scratch buffer sized for the first
        // `max_slots * stride` u32 words. Writing through `queue`
        // would also work but requires a separate submission boundary;
        // a buffer-to-buffer copy fuses cleanly into the per-tick
        // encoder. Allocating per-call is cheap (the buffer is small)
        // but if profiling shows overhead, hoist into `Self`.
        let needed_bytes =
            (max_slots as u64) * (EVENT_STRIDE_U32 as u64) * 4;
        let zeros = vec![0u8; needed_bytes as usize];
        let scratch = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("EventRing::clear_ring_headers_in::scratch"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );
        encoder.copy_buffer_to_buffer(&scratch, 0, &self.ring, 0, needed_bytes);
    }

    /// Bump the host-side per-tick tail estimate by `n` slots after a
    /// producer dispatch. `n` must be the upper bound on slots that
    /// producer could have appended to the ring this tick — typically
    /// the dispatch's `agent_count` (one ActionSelected per scoring
    /// thread) or `agent_count * N` for a producer that emits N
    /// events per agent. Using an upper bound is safe because the
    /// per-handler tag filter inside every consumer body skips
    /// non-matching slots.
    ///
    /// This is the host-side mirror of the GPU `event_tail` counter:
    /// the runtime bumps the estimate as it queues producer kernels;
    /// downstream chronicle / fold dispatches read [`Self::tail_value`]
    /// for their `event_count` cfg field. The estimate is never read
    /// back from the GPU buffer (that would force a host-GPU sync
    /// every tick), so it stays accurate only as long as the runtime
    /// faithfully calls `note_emits` for every producer; missing a
    /// call surfaces as missed events in the consumer.
    ///
    /// See `docs/superpowers/notes/2026-05-04-diplomacy_probe.md` Gap #2.
    pub fn note_emits(&mut self, n: u32) {
        self.tail_estimate = self.tail_estimate.saturating_add(n);
    }

    /// Current host-side per-tick tail estimate. Returns the sum of
    /// every [`Self::note_emits`] call since the last
    /// [`Self::clear_tail_in`]. Use as the `event_count` cfg field on
    /// a consumer (chronicle / fold) dispatch that runs AFTER one or
    /// more producers in the same tick. The cooperating per-handler
    /// tag filter inside the consumer body still rejects
    /// non-matching slots, so passing the upper bound is correct.
    ///
    /// See `docs/superpowers/notes/2026-05-04-diplomacy_probe.md` Gap #2
    /// for the multi-stage cascade pattern this accessor enables.
    pub fn tail_value(&self) -> u32 {
        self.tail_estimate
    }

    /// Force the host-side tail estimate back to 0. Used by host-driven
    /// chronicle injection paths that bypass the per-step
    /// `clear_tail_in()` flow (e.g. `tom_probe_runtime::observe()` queues
    /// a synthetic record OUTSIDE the per-tick step encoder).
    pub fn reset_tail_estimate(&mut self) {
        self.tail_estimate = 0;
    }

    /// Append one 10-word chronicle record into the ring at the
    /// current host-side tail estimate slot, then bump the tail (both
    /// the GPU `event_tail` counter and the host-side estimate) by 1.
    ///
    /// Used by host-driven runtime APIs that need to inject synthetic
    /// chronicle events from CPU (without going through a producer
    /// kernel). Today's only caller is `tom_probe_runtime`'s
    /// `observe()` / `scry()` / `reveal()` methods, which retired their
    /// hand-written WGSL consumers in favour of .sim-authored
    /// `on EffectObserveApplied { ... }` / `EffectScryApplied` /
    /// `EffectRevealApplied` consumer rules — the rules read from this
    /// ring, so the runtime needs to write into it.
    ///
    /// The 10-word record layout mirrors the GPU dispatcher's
    /// `atomicStore` writes (see
    /// `crates/dsl_compiler/src/cpu_chronicle_reference.rs`):
    ///   * `[0]` = kind tag (e.g. `64` for EffectObserveApplied)
    ///   * `[1]` = tick
    ///   * `[2..]` = per-variant payload words
    ///
    /// `record` carries the full 10 words; the caller is responsible
    /// for setting kind+tick+payload into `record[0..]` per the engine
    /// `EventKindId` schema.
    ///
    /// Both writes are queued via `wgpu::Queue::write_buffer`; the
    /// next encoder submission will see the bumped tail and the new
    /// record. If the caller is mid-encoder (between
    /// `clear_tail_in()` and a consumer dispatch), the writes still
    /// land in submission order — wgpu sequences `write_buffer` calls
    /// before the encoder's commands at submit time.
    ///
    /// **Out-of-band tail assumption.** This helper writes the new
    /// tail directly via `write_buffer` (not atomicAdd from a
    /// producer kernel). It assumes no producer kernel is concurrently
    /// running — the caller drains the ring (`clear_tail_in`), queues
    /// the synthetic record, then dispatches the consumer. Mixing
    /// host-side appends with kernel-side producers in the same tick
    /// is not supported.
    pub fn append_chronicle_record(
        &mut self,
        queue: &wgpu::Queue,
        record: &[u32; EVENT_STRIDE_U32 as usize],
    ) {
        let slot_idx = self.tail_estimate;
        // Slot beyond ring capacity? Saturate (drop the record).
        // Today's fixtures stay well under 65k; this is just defensive.
        if slot_idx >= EVENT_RING_CAP_SLOTS {
            return;
        }
        let byte_offset =
            (slot_idx as u64) * (EVENT_STRIDE_U32 as u64) * 4;
        queue.write_buffer(
            &self.ring,
            byte_offset,
            bytemuck::cast_slice(record),
        );
        let next_tail = slot_idx + 1;
        queue.write_buffer(&self.tail, 0, bytemuck::bytes_of(&next_tail));
        self.tail_estimate = next_tail;
    }
}

/// Per-view fold-storage buffers: `primary` (the accumulator the
/// fold body RMWs), optional `anchor` (set when the view carries
/// `@decay`), optional `ids` (set when the view's storage hint is
/// per-entity-top-K), a staging buffer for host readback, and a
/// host-side cache.
///
/// The fold bindings struct exposes anchor / ids as
/// `Option<&'a Buffer>`; pass [`Self::anchor`] / [`Self::ids`]
/// directly to it.
pub struct ViewStorage {
    primary: wgpu::Buffer,
    anchor: Option<wgpu::Buffer>,
    ids: Option<wgpu::Buffer>,
    staging: wgpu::Buffer,
    cache: Vec<f32>,
    dirty: bool,
    /// Number of f32 slots the view stores. `staging` is sized to
    /// `slot_count * 4` bytes; `cache` is `Vec<f32>` of length
    /// `slot_count`.
    slot_count: u32,
}

impl ViewStorage {
    /// Allocate per-view storage. `slot_count` is the number of
    /// f32 slots (= agent_count for per-agent views, agent_count^2
    /// for pair-keyed views). `has_anchor` and `has_ids` toggle
    /// the optional auxiliary buffers per the view's `@decay` /
    /// storage-hint config.
    pub fn new(
        gpu: &GpuContext,
        label: &str,
        slot_count: u32,
        has_anchor: bool,
        has_ids: bool,
    ) -> Self {
        // 16-byte minimum so empty/small views don't crash the
        // BGL validator (wgpu requires >0-sized buffer bindings).
        let bytes = ((slot_count as u64) * 4).max(16);
        let mk = |suffix: &str| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}::{suffix}")),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let primary = mk("primary");
        let anchor = if has_anchor { Some(mk("anchor")) } else { None };
        let ids = if has_ids { Some(mk("ids")) } else { None };
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}::staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            primary,
            anchor,
            ids,
            staging,
            cache: vec![0.0; slot_count as usize],
            dirty: false,
            slot_count,
        }
    }

    pub fn primary(&self) -> &wgpu::Buffer {
        &self.primary
    }
    pub fn anchor(&self) -> Option<&wgpu::Buffer> {
        self.anchor.as_ref()
    }
    pub fn ids(&self) -> Option<&wgpu::Buffer> {
        self.ids.as_ref()
    }

    /// Mark the host-side cache stale. Call once per `step()`
    /// after the fold dispatch has been encoded so the next
    /// `readback()` triggers a fresh staging copy + map.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Force a copy of `primary → staging`, map it, and refresh
    /// the host-side `Vec<f32>` cache. Idempotent until the next
    /// `mark_dirty()` so consecutive calls without an intervening
    /// step share the same readback.
    pub fn readback(&mut self, gpu: &GpuContext) -> &[f32] {
        if self.dirty {
            let mut encoder = gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("ViewStorage::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.primary,
                0,
                &self.staging,
                0,
                (self.slot_count as u64) * 4,
            );
            gpu.queue.submit(Some(encoder.finish()));
            let slice = self.staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let bytes = slice.get_mapped_range();
            let counts: &[f32] = bytemuck::cast_slice(&bytes);
            self.cache.clear();
            self.cache.extend_from_slice(counts);
            drop(bytes);
            self.staging.unmap();
            self.dirty = false;
        }
        &self.cache
    }
}
