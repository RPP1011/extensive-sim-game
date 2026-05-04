//! Per-fixture runtime for `assets/sim/tom_probe.sim` — Theory-of-Mind
//! end-to-end probe (post-fix shape).
//!
//! ## What lights up
//!
//! - The Knower SoA (`agent_alive`).
//! - The `BeliefAcquired` event ring + per-tick tail clear.
//! - The `physics_WhatIBelieve` kernel (per-Knower; emits one
//!   `BeliefAcquired { observer: self, subject: self, fact_bit: 1 }`
//!   per tick into the ring).
//! - The `fold_beliefs` kernel (per-event; OR's `fact_bit` into
//!   `view_storage_primary[observer * agent_cap + subject]` via
//!   WGSL native `atomicOr` — no CAS retry, P11 trivial).
//! - Per-(observer, subject) belief storage: a single `u32` buffer
//!   sized `agent_cap × agent_cap`, allocated locally (NOT through
//!   `engine::gpu::ViewStorage` because that helper's host-side cache
//!   is `Vec<f32>`; the bit pattern would be re-bitcast on every
//!   readback). The buffer is the same `array<atomic<u32>>` BGL
//!   shape the fold kernel expects, so binding parity holds.
//!
//! ## Expected outcome (FULL FIRE)
//!
//! After N ticks at agent_count = N: `beliefs(i, i) = 1u` for every
//! alive Knower; every off-diagonal `beliefs(i, j != i) = 0u`. The
//! tom_probe_app driver asserts both halves and reports OUTCOME (a)
//! FULL FIRE on success.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::EventRing;

/// Per-fixture state for the ToM probe. Carries the Knower SoA
/// (`agent_alive`), the `BeliefAcquired` event ring, the `beliefs`
/// per-(observer, subject) `u32` storage, and the cfg uniforms for
/// the producer (`physics_WhatIBelieve`) + consumer (`fold_beliefs`)
/// kernels.
pub struct TomProbeState {
    gpu: GpuContext,

    // -- Agent SoA (read by physics_WhatIBelieve to gate `self.alive`) --
    /// 1 = alive, 0 = dead. Initialised all-1 so every Knower fires
    /// the producer rule each tick.
    agent_alive_buf: wgpu::Buffer,

    // -- Per-(observer, subject) belief storage --
    //
    // `pair_map`-keyed: `agent_cap × agent_cap × u32`. The fold body
    // indexes `view_storage_primary[observer * cfg.second_key_pop +
    // subject]`. We allocate this locally (instead of via
    // `engine::gpu::ViewStorage`) so the host-side readback can
    // surface a `&[u32]` directly without an f32 bitcast round-trip.
    beliefs_primary: wgpu::Buffer,
    beliefs_staging: wgpu::Buffer,
    beliefs_cache: Vec<u32>,
    beliefs_dirty: bool,

    // -- Event ring + per-kernel cfg uniforms --
    event_ring: EventRing,
    physics_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl TomProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Knower SoA — `agent_alive` is the only field the producer
        // rule reads (`where (self.alive)`). Every slot starts alive
        // so every tick fires.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Per-(observer, subject) `beliefs` storage. `pair_map` →
        // size = `agent_cap × agent_cap × u32`. The wgpu BGL is the
        // same `array<atomic<u32>>` shape the fold kernel expects;
        // we expose `STORAGE | COPY_SRC | COPY_DST` so the
        // per-readback `copy_buffer_to_buffer → staging map` dance
        // works.
        let belief_slot_count = (agent_count as u64) * (agent_count as u64);
        let belief_bytes = (belief_slot_count * 4).max(16);
        let beliefs_primary = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tom_probe_runtime::beliefs_primary"),
            size: belief_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let beliefs_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tom_probe_runtime::beliefs_staging"),
            size: belief_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let event_ring = EventRing::new(&gpu, "tom_probe_runtime");

        let physics_cfg_init =
            physics_WhatIBelieve::PhysicsWhatIBelieveCfg {
                agent_cap: agent_count,
                tick: 0,
                _pad: [0, 0],
            };
        let physics_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::physics_WhatIBelieve_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let fold_cfg_init = fold_beliefs::FoldBeliefsCfg {
            event_count: 0,
            tick: 0,
            // `beliefs(observer: Agent, subject: Agent)` — both keys
            // are Agent, so second_key_pop = agent_count. The fold
            // body composes `view_storage_primary[k1 *
            // second_key_pop + k2]` so this MUST equal agent_count
            // for the diagonal to land at index `i * N + i`.
            second_key_pop: agent_count,
            _pad: 0,
        };
        let fold_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::fold_beliefs_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_alive_buf,
            beliefs_primary,
            beliefs_staging,
            beliefs_cache: vec![0u32; belief_slot_count as usize],
            beliefs_dirty: false,
            event_ring,
            physics_cfg_buf,
            fold_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-(observer, subject) belief bitset, flattened row-major:
    /// slot `[observer * agent_count + subject]` holds the OR-folded
    /// fact bits the observer believes about the subject.
    /// Length = `agent_count × agent_count`.
    pub fn beliefs(&mut self) -> &[u32] {
        if self.beliefs_dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs::readback"),
                },
            );
            let bytes = (self.beliefs_cache.len() as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &self.beliefs_primary,
                0,
                &self.beliefs_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.beliefs_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_staging.unmap();
            self.beliefs_dirty = false;
        }
        &self.beliefs_cache
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

impl CompiledSim for TomProbeState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("tom_probe_runtime::step"),
            },
        );

        // (1) Per-tick clear of event_tail. The producer rule
        // atomicAdd's against it during physics_WhatIBelieve to
        // acquire write slots; the count accumulates over the tick
        // and the fold kernel reads it via cfg.event_count. Clearing
        // here guarantees a fresh per-tick slot count even though
        // event slots from prior ticks linger in the ring (the fold
        // kernel's `event_idx >= cfg.event_count` early-return
        // filters stale slots).
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) physics_WhatIBelieve — per-Knower; emits one
        // `BeliefAcquired { observer: self, subject: self, fact_bit:
        // 1 }` per tick when `self.alive`.
        let physics_cfg = physics_WhatIBelieve::PhysicsWhatIBelieveCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0, 0],
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&physics_cfg),
        );
        let physics_bindings =
            physics_WhatIBelieve::PhysicsWhatIBelieveBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_alive: &self.agent_alive_buf,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_physics_whatibelieve(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) fold_beliefs — per-event; OR's `fact_bit` into
        // `beliefs_primary[observer * agent_cap + subject]` via
        // `atomicOr`. We size event_count = agent_count: every alive
        // Knower emits exactly one event per tick, so this is the
        // exact upper bound (no skip / no over-dispatch). The
        // kernel's `event_idx >= cfg.event_count` early-return
        // filters anything beyond the producer's per-tick batch even
        // if leftover slots from prior ticks remain in the ring.
        let event_count_estimate = self.agent_count;
        let fold_cfg = fold_beliefs::FoldBeliefsCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.fold_cfg_buf,
            0,
            bytemuck::bytes_of(&fold_cfg),
        );
        let fold_bindings = fold_beliefs::FoldBeliefsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.beliefs_primary,
            // No `@decay` and no top-K → no anchor / no ids; the
            // generated `record()` body falls back to primary via
            // `unwrap_or(primary_buf)` per `kernel.rs`'s slot-aliasing
            // convention.
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_beliefs(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.beliefs_dirty = true;
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice. Same shape
        // as verb_probe_runtime (which has the same comment).
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TomProbeState::new(seed, agent_count))
}
