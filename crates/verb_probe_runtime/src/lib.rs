//! Per-fixture runtime for `assets/sim/verb_fire_probe.sim` — the
//! smallest end-to-end probe of the verb cascade.
//!
//! Runtime structure mirrors `predator_prey_runtime`: compiler-emitted
//! kernels live in `OUT_DIR/generated.rs` (pulled in via `include!`),
//! the hand-written `VerbProbeState` owns the GPU context, per-field
//! storage, the kernel cache, the event ring, and the per-tick
//! dispatch chain.
//!
//! ## Per-tick chain (orchestrated)
//!
//! Closes Gap #4 from `2026-05-04-verb-fire-probe.md` — the verb
//! cascade has THREE rounds within a tick that the runtime now
//! dispatches in order:
//!
//! 1. clear event_tail = 0
//! 2. (Round 1, mask) `mask_verb_Pray` — sets bitmap bit when
//!    `self.alive` (always true)
//! 3. (Round 1, scoring) `scoring` — argmax picks verb_Pray
//!    (action_id=0); each agent slot atomically appends an
//!    `ActionSelected{actor=agent, action_id=0, target=NO_TARGET}`
//!    event onto the ring (kind tag = 2u)
//! 4. (Round 2, chronicle) `physics_verb_chronicle_Pray` — reads
//!    each ActionSelected slot, gates on `action_id == 0`, and emits
//!    `PrayCompleted{prayer: actor, faith_delta: config.faith_step}`
//!    (kind tag = 1u). Dispatched with a generous workgroup count
//!    (`agent_count` threads, rounded to a 64-thread workgroup); the
//!    ring is sentinel-pre-stamped at construction (every `+3`
//!    `action_id` slot = `0xFFFFFFFFu`) so workgroup-rounding threads
//!    that read past the actual scoring tail find a non-zero
//!    `action_id` and skip emission. From tick 1 onward, those slots
//!    hold prior-tick `PrayCompleted` payloads where `+3` =
//!    `bitcast<u32>(faith_delta)` ≈ `0x3f800000` — also non-zero, so
//!    no spurious emission.
//! 5. seed_indirect_0 — reads tail count to compute fold workgroup
//!    count (kept for parity with the original schedule; not
//!    consumed by today's direct dispatches)
//! 6. (Round 3, fold) `fold_faith` — reads PrayCompleted events
//!    (tag=1, the kind filter landed in `cb24fd69` for view-fold
//!    bodies). Dispatched with `event_count = agent_count * 2` to
//!    cover both ActionSelected + PrayCompleted slots; the per-handler
//!    tag check ignores ActionSelected.
//! 7. kick_snapshot — flag the snapshot slot.
//!
//! Observable: with `faith_step = 1.0`, every alive slot's faith
//! value should equal the tick count (e.g. `100.0` after 100 ticks).

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage, EVENT_RING_CAP_SLOTS, EVENT_STRIDE_U32};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Padded {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z, _pad: 0.0 }
    }
}

/// Per-fixture state for the verb-fire probe. Carries one Devotee SoA
/// (pos, vel, alive), the event ring, the faith view storage, the
/// scoring_output buffer, the per-mask bitmap, and per-kernel cfg
/// uniforms. The `agent_alive` field is needed by the `mask_verb_Pray`
/// kernel; `pos` / `vel` are kept so the per-agent SoA shape matches
/// the boids/pp pattern (the probe doesn't actually use them, but
/// the compiler's `pack_agents` codepath references them and we
/// keep them parallel for symmetry / to avoid divergence from the
/// reference runtime shape).
pub struct VerbProbeState {
    gpu: GpuContext,

    // -- Agent SoA (read by mask_verb_Pray, scored over by scoring) --
    /// 1 = alive, 0 = dead. Initialised to all-1 so every slot's mask
    /// predicate (`self.alive`) evaluates true.
    agent_alive_buf: wgpu::Buffer,

    // -- Mask bitmap (one bit per agent in u32 words) --
    /// `mask_0_bitmap`: u32 array of length `ceil(agent_count / 32)`.
    /// `mask_verb_Pray` atomicOrs into the appropriate word/bit per
    /// matching agent slot.
    mask_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    /// Layout: `[best_action, best_target, bitcast<u32>(best_utility),
    /// 0]` per agent slot. Sized at `agent_count * 4`.
    scoring_output_buf: wgpu::Buffer,

    // -- Event ring + per-view storage helpers --
    event_ring: EventRing,
    faith: ViewStorage,
    faith_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms (each is repr(C) {agent_cap, tick, _pad}) --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,
    #[allow(dead_code)]
    snapshot_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl VerbProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — only `alive` is read by any compiled kernel
        // (mask_verb_Pray). Sized to agent_count u32 slots.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verb_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Mask bitmap — `ceil(agent_count / 32)` u32 words. Cleared
        // each tick before mask_verb_Pray runs so stale bits from
        // last tick don't leak.
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verb_probe_runtime::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A pre-zeroed source buffer for per-tick clears via
        // copy_buffer_to_buffer (no separate clear kernel emitted).
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verb_probe_runtime::mask_0_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 × u32 per agent
        let scoring_output_bytes = (agent_count as u64) * 4 * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verb_probe_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Event-ring + faith view storage (per-agent f32, no anchor,
        // no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "verb_probe_runtime");
        let faith = ViewStorage::new(
            &gpu,
            "verb_probe_runtime::faith",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // Sentinel pre-stamp of the event ring's `+3` slot in every
        // record. The compiled `physics_verb_chronicle_Pray` kernel
        // has neither an `event_count` bound nor a kind-tag check
        // (the cb24fd69 fix only added that to view-fold bodies, not
        // physics-rule bodies). When we direct-dispatch the chronicle
        // with `agent_count` threads rounded up to a 64-thread
        // workgroup, the rounding threads (e.g. 32..63 at
        // agent_count=32) read past the actual scoring tail. If those
        // slots were zero-initialised, their `action_id` (= u32 at
        // `event_idx*10 + 3`) would be `0u`, the chronicle's
        // `if (local_1 == 0u)` predicate would match, and they'd
        // emit spurious `PrayCompleted` events with `actor = 0` —
        // skewing `faith[0]` on tick 0. Stamping `0xFFFFFFFFu` into
        // every slot's `+3` word makes the predicate fail for any
        // unused/stale slot. From tick 1 onward, those slots hold
        // prior-tick PrayCompleted payloads where `+3` is
        // `bitcast<u32>(faith_delta)` ≈ `0x3f800000` — also non-zero,
        // so the fix continues to hold past tick 0. Total upload:
        // 65536 × 4 bytes = 256 KB, one-time.
        {
            let cap = EVENT_RING_CAP_SLOTS as usize;
            let stride = EVENT_STRIDE_U32 as usize;
            let mut sentinel = vec![0u32; cap * stride];
            let mut i = 0usize;
            while i < cap {
                sentinel[i * stride + 3] = 0xFFFF_FFFFu32;
                i += 1;
            }
            gpu.queue.write_buffer(
                event_ring.ring(),
                0,
                bytemuck::cast_slice(&sentinel),
            );
        }

        // Per-kernel cfg uniforms. Each kernel's struct shape is
        // `{ agent_cap: u32, tick: u32, _pad: [u32; 2] }` (compatible
        // across mask/scoring/seed/snapshot — they all share the same
        // layout per the compiler emit).
        let cfg_init = mask_verb_Pray::MaskVerbPrayCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let mk_cfg = |label: &str| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mask_cfg_buf = mk_cfg("verb_probe_runtime::mask_cfg");
        let scoring_cfg_buf = mk_cfg("verb_probe_runtime::scoring_cfg");
        let chronicle_cfg_buf = mk_cfg("verb_probe_runtime::chronicle_cfg");
        let seed_cfg_buf = mk_cfg("verb_probe_runtime::seed_cfg");
        let snapshot_cfg_buf = mk_cfg("verb_probe_runtime::snapshot_cfg");

        let faith_cfg_init = fold_faith::FoldFaithCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let faith_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verb_probe_runtime::faith_cfg"),
                contents: bytemuck::bytes_of(&faith_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_alive_buf,
            mask_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            event_ring,
            faith,
            faith_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_cfg_buf,
            seed_cfg_buf,
            snapshot_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-prayer faith accumulator (one f32 per agent slot).
    pub fn faith(&mut self) -> &[f32] {
        self.faith.readback(&self.gpu)
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl CompiledSim for VerbProbeState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("verb_probe_runtime::step"),
            },
        );

        // (1) Per-tick clear of event_tail. Producers atomicAdd
        // against it to acquire ring slots.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1b) Per-tick clear of the mask bitmap so stale bits from
        // last tick don't leak into the next mask evaluation.
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_bitmap_buf,
            0,
            mask_bytes.max(4),
        );

        // (2) mask_verb_Pray — reads agent_alive, atomicOrs into
        // mask_0_bitmap.
        let mask_cfg = mask_verb_Pray::MaskVerbPrayCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = mask_verb_Pray::MaskVerbPrayBindings {
            agent_alive: &self.agent_alive_buf,
            mask_0_bitmap: &self.mask_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_mask_verb_pray(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) scoring — picks verb_Pray each tick, atomically appends
        // an ActionSelected event per agent slot.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            mask_0_bitmap: &self.mask_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) physics_verb_chronicle_Pray — Round 2 of the cascade.
        // Reads each ActionSelected slot the scoring kernel just
        // emitted, gates on `action_id == 0` (Pray's id), and emits
        // a `PrayCompleted{ prayer: actor, faith_delta:
        // config.faith_step }` event (kind tag = 1u).
        //
        // Dispatch threads = `agent_count` (matches the scoring
        // emit count: one ActionSelected per alive slot). The
        // workgroup-rounding (next multiple of 64) gives extra
        // threads that read past the actual scoring tail; the
        // sentinel pre-stamp at construction guarantees those slots'
        // `+3` (action_id) word is non-zero, so the chronicle's
        // predicate fails and no spurious emissions land.
        let chronicle_cfg = physics_verb_chronicle_Pray::PhysicsVerbChroniclePrayCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_bindings =
            physics_verb_chronicle_Pray::PhysicsVerbChroniclePrayBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_pray(
            &mut self.cache,
            &chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5) seed_indirect_0 — reads event_tail, populates
        // indirect_args_0 with workgroup count.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf,
            0,
            bytemuck::bytes_of(&seed_cfg),
        );
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6) fold_faith — Round 3. Reads PrayCompleted events
        // (kind tag = 1u, filtered by the per-handler tag check
        // landed in cb24fd69) and RMWs view_storage_primary.
        // event_count is sized generously to cover both event kinds
        // (ActionSelected from Round 1 + PrayCompleted from Round 2):
        // total ~= 2 × agent_count slots/tick. The in-kernel tag
        // check ignores the ActionSelected slots.
        let event_count_estimate = self.agent_count * 2;
        let faith_cfg = fold_faith::FoldFaithCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.faith_cfg_buf,
            0,
            bytemuck::bytes_of(&faith_cfg),
        );
        let faith_bindings = fold_faith::FoldFaithBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.faith.primary(),
            view_storage_anchor: self.faith.anchor(),
            view_storage_ids: self.faith.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.faith_cfg_buf,
        };
        dispatch::dispatch_fold_faith(
            &mut self.cache,
            &faith_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (7) kick_snapshot — flags the snapshot slot.
        // Skipped — no kick_snapshot binding allocator and the
        // `snapshot_kick` slot is a host-side artefact for non-probe
        // pipelines. Same skip pattern as predator_prey_runtime.

        self.gpu.queue.submit(Some(encoder.finish()));
        self.faith.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice. The
        // `CompiledSim` trait demands the method but the probe doesn't
        // need positional readback.
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(VerbProbeState::new(seed, agent_count))
}
