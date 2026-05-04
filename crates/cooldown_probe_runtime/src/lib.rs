//! Per-fixture runtime for `assets/sim/cooldown_probe.sim` — the
//! follow-up probe to abilities_probe (Gap #4 in
//! `docs/superpowers/notes/2026-05-04-abilities_probe.md`).
//!
//! ## What this exercises
//!
//! Two surfaces wired through to the runtime end-to-end:
//!
//! 1. **`agents.cooldown_next_ready_tick(self)`** — per-agent SoA
//!    field read via the same `agents.<field>(<expr>)` lowering arm
//!    that target_chaser exercises with `agents.pos(...)`. The field
//!    binding lands as `agent_cooldown_next_ready_tick` in the
//!    physics kernel's BGL slot 3. The runtime allocates the
//!    per-agent u32 SoA, initialises it to a STAGGERED pattern
//!    (`ready_at[N] = N`), and observes the per-slot fire counts.
//! 2. **`world.tick`** read in physics body, paired with
//!    `if (world.tick >= ready_at)` gating an `emit`. Tests that the
//!    tick preamble local (`let tick = cfg.tick;`) reaches the
//!    physics-body scope and the IrStmt::If lowering routes cleanly.
//!
//! ## Per-tick chain
//!
//! 1. `clear_tail` — event_tail = 0 so `atomicAdd` slots restart.
//! 2. `physics_CheckAndCast` — reads `agent_alive` + `agent_cooldown_
//!    next_ready_tick`; for each alive agent emits `ActivationLogged`
//!    when `tick >= ready_at`.
//! 3. `seed_indirect_0` — populates indirect-args buffer from the
//!    tail count (kept for parity).
//! 4. `fold_activations` — per-handler tag-filter on
//!    `ActivationLogged` (kind = 1u), atomic RMW into the per-caster
//!    activations view storage.
//!
//! ## Observable
//!
//! With AGENT_COUNT=32, TICKS=100, init `ready_at[N] = N`:
//!
//!   activations[N] = max(0, TICKS - N) = 100 - N
//!
//! See `docs/superpowers/notes/2026-05-04-cooldown_probe.md` for
//! the discovery report.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the cooldown probe. Owns:
///   - Agent SoA (alive + cooldown_next_ready_tick — two buffers)
///   - Event ring + per-view storage (activations: f32 per slot)
///   - Per-kernel cfg uniforms
pub struct CooldownProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. All-1 init so every slot's `self.alive`
    /// gate evaluates true.
    agent_alive_buf: wgpu::Buffer,
    /// `cooldown_next_ready_tick` SoA — one u32 per agent. Initialised
    /// to `ready_at[N] = N` (staggered) so per-slot fire count is
    /// `max(0, TICKS - N)`.
    agent_cooldown_next_ready_tick_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-caster activation count (f32). Fed by ActivationLogged
    /// (kind tag = 1u). After T ticks: `activations[N] = max(0, T - N)`.
    activations: ViewStorage,
    activations_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    physics_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl CooldownProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — `alive` is read by physics_CheckAndCast (BGL
        // slot 2). Initialised to all-1 so every slot fires its gate.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cooldown_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // `cooldown_next_ready_tick` SoA — read by physics_CheckAndCast
        // (BGL slot 3). Staggered init: `ready_at[N] = N`. With
        // TICKS=100, AGENT_COUNT=32, slot 0 fires 100 times (always
        // ready), slot 31 fires 69 times.
        let cooldown_init: Vec<u32> = (0..agent_count).collect();
        let agent_cooldown_next_ready_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cooldown_probe_runtime::agent_cooldown_next_ready_tick"),
                contents: bytemuck::cast_slice(&cooldown_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Event ring + activations view storage (per-agent f32, no
        // anchor, no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "cooldown_probe_runtime");
        let activations = ViewStorage::new(
            &gpu,
            "cooldown_probe_runtime::activations",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // Per-kernel cfg uniforms.
        let physics_cfg_init = physics_CheckAndCast::PhysicsCheckAndCastCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("cooldown_probe_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("cooldown_probe_runtime::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let activations_cfg_init = fold_activations::FoldActivationsCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let activations_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("cooldown_probe_runtime::activations_cfg"),
                contents: bytemuck::bytes_of(&activations_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            agent_cooldown_next_ready_tick_buf,
            event_ring,
            activations,
            activations_cfg_buf,
            physics_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-caster activation count (one f32 per slot). After T ticks:
    /// `activations[N] = max(0, T - N)` under the staggered init pattern.
    pub fn activations(&mut self) -> &[f32] {
        self.activations.readback(&self.gpu)
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

impl CompiledSim for CooldownProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cooldown_probe_runtime::step"),
                });

        // (1) Per-tick clear of event_tail.
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) physics_CheckAndCast — reads agent_alive + agent_cooldown_
        // next_ready_tick. For each alive agent: if `tick >= ready_at`,
        // emits ActivationLogged{ caster=self, activated=1.0 } into the
        // event ring (kind tag = 1u).
        let physics_cfg = physics_CheckAndCast::PhysicsCheckAndCastCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_bindings = physics_CheckAndCast::PhysicsCheckAndCastBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_alive: &self.agent_alive_buf,
            agent_cooldown_next_ready_tick: &self.agent_cooldown_next_ready_tick_buf,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_physics_checkandcast(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) seed_indirect_0 — populates indirect_args_0 from
        // event_tail (kept for parity with the verb_probe pattern;
        // direct dispatch in (4) doesn't actually consume the args).
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
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

        // (4) fold_activations — per-handler tag-filter on
        // ActivationLogged (kind = 1u), atomic RMW into per-caster
        // primary view storage. Sized at agent_count (one slot per
        // alive agent emits at most one event per tick).
        let event_count_estimate = self.agent_count;
        let activations_cfg = fold_activations::FoldActivationsCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.activations_cfg_buf,
            0,
            bytemuck::bytes_of(&activations_cfg),
        );
        let activations_bindings = fold_activations::FoldActivationsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.activations.primary(),
            view_storage_anchor: self.activations.anchor(),
            view_storage_ids: self.activations.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.activations_cfg_buf,
        };
        dispatch::dispatch_fold_activations(
            &mut self.cache,
            &activations_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // kick_snapshot intentionally skipped — host-side artefact,
        // same skip pattern as verb_probe_runtime.

        self.gpu.queue.submit(Some(encoder.finish()));
        self.activations.mark_dirty();
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
    Box::new(CooldownProbeState::new(seed, agent_count))
}
