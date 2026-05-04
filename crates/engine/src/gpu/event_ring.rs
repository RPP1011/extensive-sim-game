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

/// Default slot capacity of the per-tick event ring. Mirrors
/// `DEFAULT_EVENT_RING_CAP_SLOTS` in the WGSL emit body
/// (`cg/emit/wgsl_body.rs::lower_emit_to_wgsl`). 65 536 slots ×
/// 10 u32/slot × 4 bytes = 2.5 MB per fixture — comfortable
/// margin for any per-tick producer cap a smoke fixture would
/// realistically configure.
pub const EVENT_RING_CAP_SLOTS: u32 = 65_536;

/// u32 words per event record (2 header + 8 payload). Matches
/// `populate_event_kinds` in the CG lowering driver. Future
/// per-kind ring fanout would surface a per-kind override.
pub const EVENT_STRIDE_U32: u32 = 10;

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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
