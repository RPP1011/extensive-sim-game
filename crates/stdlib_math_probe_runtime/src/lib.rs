//! Per-fixture runtime for `assets/sim/stdlib_math_probe.sim` —
//! discovery probe stress-testing under-exercised stdlib math + RNG
//! conversion surfaces. Mirrors `stochastic_probe_runtime` shape
//! (single physics rule + single fold + no @decay).
//!
//! Per-tick chain: `clear_tail` + `clear_ring_headers_in` →
//! `physics_SampleAndBucket` (math stdlib chain + emit Sampled with
//! bucket = `rng.action() % 4u`) → `seed_indirect_0` (parity) →
//! `fold_sampled_count` (per-handler tag-filter on Sampled, atomic
//! RMW into per-agent f32 slot).
//!
//! With AGENT_COUNT=32, TICKS=100, log_amount=1.0: per-slot
//! `sampled_count[N] = 100` (unconditional emit). Surface coverage,
//! tier breakdown, and the five-gap punch list (Gaps #A-#E) live in
//! `docs/superpowers/notes/2026-05-04-stdlib_math_probe.md`.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the stdlib math probe. Owns:
///   - Agent SoA (alive only — pos/vel are declaration-only here)
///   - Event ring + per-view storage (sampled_count: f32 per slot)
///   - Per-kernel cfg uniforms
pub struct StdlibMathProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. All-1 init so every slot's `self.alive`
    /// gate evaluates true.
    agent_alive_buf: wgpu::Buffer,
    /// Per-agent position (vec3<f32>, 16-byte aligned per WGSL std430
    /// for vec3 in arrays). Read by the physics body's
    /// `planar_distance(self.pos, self.pos)` and
    /// `z_separation(self.pos, self.pos)` calls (Gap #B close,
    /// 2026-05-04). Initialised to all-zero — the spatial calls
    /// evaluate to 0.0 but exercise the WGSL emit + prelude wiring.
    agent_pos_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-agent emit count (f32). Fed by Sampled (kind tag = 1u).
    /// After T ticks: `sampled_count[N] = T`.
    sampled_count: ViewStorage,
    sampled_count_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    physics_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl StdlibMathProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — `alive` is read by physics_SampleAndBucket (BGL
        // slot 2). Initialised to all-1 so every slot fires its gate.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stdlib_math_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Per-agent position SoA — 4 × f32 per slot (vec3 + pad for
        // std430 16-byte alignment). All-zero init: the probe's
        // spatial calls are `planar_distance(self.pos, self.pos)` and
        // `z_separation(self.pos, self.pos)` — both evaluate to 0
        // regardless of input, so the buffer's contents don't shift
        // the observable. The buffer's job is to satisfy the BGL
        // binding the physics kernel now declares (Gap #B close).
        let pos_init: Vec<f32> = vec![0.0_f32; (agent_count as usize) * 4];
        let agent_pos_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stdlib_math_probe_runtime::agent_pos"),
                contents: bytemuck::cast_slice(&pos_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Event ring + sampled_count view storage (per-agent f32, no
        // anchor, no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "stdlib_math_probe_runtime");
        let sampled_count = ViewStorage::new(
            &gpu,
            "stdlib_math_probe_runtime::sampled_count",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // Per-kernel cfg uniforms. `seed: u32` is the low 32 bits of
        // the runtime's u64 world seed — the GPU primitive
        // `per_agent_u32(seed, agent_id, tick, purpose_id)` keys on
        // u32 (matches `RNG_WGSL_PRELUDE` in dsl_compiler).
        let seed_lo = seed as u32;
        let physics_cfg_init = physics_SampleAndBucket::PhysicsSampleAndBucketCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: seed_lo,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stdlib_math_probe_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: seed_lo,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stdlib_math_probe_runtime::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let sampled_count_cfg_init = fold_sampled_count::FoldSampledCountCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let sampled_count_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stdlib_math_probe_runtime::sampled_count_cfg"),
                contents: bytemuck::bytes_of(&sampled_count_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            agent_pos_buf,
            event_ring,
            sampled_count,
            sampled_count_cfg_buf,
            physics_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-agent emit count (one f32 per slot). After T ticks:
    /// `sampled_count[N] = T` (unconditional emit on Tick).
    pub fn sampled_count(&mut self) -> &[f32] {
        self.sampled_count.readback(&self.gpu)
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

impl CompiledSim for StdlibMathProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("stdlib_math_probe_runtime::step"),
                });

        // (1) Per-tick clear of event_tail + ring headers (same
        // shape as stochastic_probe — the producer emits once per
        // alive agent, the consumer dispatches over agent_count, so
        // unused upper-tail slots need their `kind` words cleared
        // each tick).
        self.event_ring.clear_tail_in(&mut encoder);
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, self.agent_count);

        // (2) physics_SampleAndBucket — reads agent_alive. For each
        // alive agent: runs the math/rng `let` chain, computes the
        // bucket via `rng.action() % 4u`, and emits Sampled{ agent,
        // bucket, amount=1.0 } into the event ring (kind tag = 1u).
        let physics_cfg = physics_SampleAndBucket::PhysicsSampleAndBucketCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: self.seed as u32,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_bindings = physics_SampleAndBucket::PhysicsSampleAndBucketBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_physics_sampleandbucket(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        // Producer upper bound: at most one Sampled per agent per
        // tick. Records the bound on the host-side ring estimator so
        // downstream consumers can size `event_count` from
        // `tail_value()`.
        self.event_ring.note_emits(self.agent_count);

        // (3) seed_indirect_0 — populates indirect_args_0 from
        // event_tail (parity with the stochastic_probe pattern).
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: self.seed as u32,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
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

        // (4) fold_sampled_count — per-handler tag-filter on Sampled
        // (kind = 1u), atomic RMW into per-agent primary view
        // storage.
        let event_count_estimate = self.event_ring.tail_value();
        let sampled_count_cfg = fold_sampled_count::FoldSampledCountCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.sampled_count_cfg_buf,
            0,
            bytemuck::bytes_of(&sampled_count_cfg),
        );
        let sampled_count_bindings = fold_sampled_count::FoldSampledCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.sampled_count.primary(),
            view_storage_anchor: self.sampled_count.anchor(),
            view_storage_ids: self.sampled_count.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.sampled_count_cfg_buf,
        };
        dispatch::dispatch_fold_sampled_count(
            &mut self.cache,
            &sampled_count_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // kick_snapshot intentionally skipped — host-side artefact,
        // same skip pattern as stochastic_probe / cooldown_probe.

        self.gpu.queue.submit(Some(encoder.finish()));
        self.sampled_count.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice.
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(StdlibMathProbeState::new(seed, agent_count))
}
