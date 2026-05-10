//! Per-fixture runtime for `assets/sim/firebolt_interrupt_probe.sim` —
//! Plan G G2.5 behavioural pin.
//!
//! ## What this exercises
//!
//! Variant of `firebolt_probe_runtime` that proves the
//! interrupt-detection path closes the loop:
//!   cast → busy stamped → mid-cast damage event →
//!   InterruptCastOnDamage clears busy → ResolveBusy doesn't fire at
//!   the would-be resolve tick.
//!
//! Adds two new physics rules on top of the firebolt_probe baseline:
//!
//!   * `physics_InjectDamage` (per_agent) — at the configured tick,
//!     emits a Damaged event on every alive agent (simulates an
//!     external attack mid-cast).
//!   * `physics_InterruptCastOnDamage` (post) — chronicle consumer;
//!     on any EffectDamageApplied whose target is mid-cast
//!     (`busy_until_tick > 0`), clears the busy SoA columns.
//!
//! ## Per-tick chain
//!
//! Three encoder submissions, with TWO `event_tail` readbacks between
//! them so the post-pass and final-pass consumer kernels see the
//! correct per-tick `event_count`:
//!
//!   Stage 1 (encoder A) — per_agent emitters:
//!     1. Clear `event_tail`.
//!     2. `physics_DispatchFirebolt` — writes `EffectCastBeginApplied`
//!        records when `busy_until_tick == 0`.
//!     3. `physics_InjectDamage` — writes `EffectDamageApplied`
//!        records on every alive agent when
//!        `world.tick == config.interrupt.inject_at_tick`.
//!     4. `seed_indirect_0` — plumbing kernel for indirect-args parity.
//!     5. Copy `event_tail` to staging.
//!   --- readback `event_tail` → `post_event_count` ---
//!
//!   Stage 2 (encoder B) — post-pass consumers + ResolveBusy:
//!     6. `physics_RecordCastBegin` — chronicle consumer for
//!        `EffectCastBeginApplied`; stamps the three busy SoA fields.
//!     7. `physics_InterruptCastOnDamage` — chronicle consumer for
//!        `EffectDamageApplied`; clears the busy SoA fields when the
//!        target is mid-cast.
//!     8. `physics_ResolveBusy` — fires `EffectDamageApplied` when
//!        `world.tick >= busy_until_tick > 0`. Runs AFTER
//!        InterruptCastOnDamage in this fixture so an interrupt at
//!        tick T can prevent a resolve at the same tick.
//!     9. Copy `event_tail` to staging again (resolve may have
//!        emitted new damage records on top of the inject events).
//!   --- readback `event_tail` → `damage_event_count` ---
//!
//!   Stage 3 (encoder C) — apply hp deltas:
//!    10. `physics_ApplyChronicleDamage` — decrements `agent_hp` by
//!        the chronicle's `amount` field for each `EffectDamageApplied`
//!        record (filters by kind tag, so feeding it the full count
//!        is safe even though the ring also holds CastBegin records).
//!
//! ## Behavioural pin (`cast_interrupted_by_external_damage`)
//!
//! With `n_agents = 2`, `initial_hp = 100.0`,
//! `inject_at_tick = 1`, `inject_amount = 10.0`:
//!
//!   * Tick 0: cast begins. busy_until_tick = 3. hp = [100, 100].
//!   * Tick 1: InjectDamage emits 10 dmg → InterruptCastOnDamage
//!     clears busy → ApplyChronicleDamage drops hp.
//!     hp = [90, 90].
//!   * Tick 2: dispatcher re-casts (busy == 0). New busy_until_tick = 5.
//!     No damage. hp = [90, 90].
//!   * Tick 3: original cast WOULD have resolved here (firebolt_probe
//!     baseline drops hp to 75 at this tick). With interrupt fired,
//!     busy_until_tick = 5 (from the second cast), so ResolveBusy
//!     predicate fails (3 < 5). hp stays at [90, 90] — the diff
//!     vs the baseline (90 not 75) is the behavioural signal.
//!
//! ## GPU adapter availability
//!
//! Construction touches the GPU (`GpuContext::new_blocking`). On hosts
//! without a wgpu-compatible adapter the constructor returns `None`;
//! the closed-loop test detects this and skips with an explanatory
//! message rather than failing.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate, PackedAbilityRegistry,
};
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};
use engine::GpuContext;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-fixture state for the firebolt-interrupt-probe closed-loop demo.
/// Owns the wgpu context, agent SoA, busy SoA, event ring, packed
/// registry, and the per-kernel cfg uniforms.
pub struct FireboltInterruptProbeState {
    gpu: GpuContext,

    // -- Agent SoA (standard columns the dispatcher binds) --
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_hp_staging: wgpu::Buffer,
    // Per-stat scaling columns (all-zero — Firebolt's CastBegin op has
    // no stat scaling, so `scale_bonus = 0.0` regardless).
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Per-agent busy SoA (Plan G G2.7). The interrupt rule writes
    //    busy_until_tick + busy_with_ability_id (busy_started_at_tick
    //    isn't cleared by interrupt today — matches the .sim).
    agent_busy_until_tick_buf: wgpu::Buffer,
    agent_busy_with_ability_id_buf: wgpu::Buffer,
    agent_busy_started_at_tick_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Event ring + tail + tail readback --
    event_ring: EventRing,
    /// Staging buffer for `event_tail` host readback (4 bytes).
    event_tail_staging: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    dispatch_cfg_buf: wgpu::Buffer,
    inject_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,
    record_cfg_buf: wgpu::Buffer,
    interrupt_cfg_buf: wgpu::Buffer,
    resolve_cfg_buf: wgpu::Buffer,
    damage_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
    /// Initial hp for every agent (host-only — used by tests for
    /// expected-state computation).
    pub initial_hp: f32,
    /// Current tick — incremented at the end of every `step()` call.
    tick: u32,
    /// Plan G tunable cfg — per-tick value the InterruptCastOnDamage
    /// kernel reads as `cfg.config_interrupt_mask`. Mirrors the .sim's
    /// `config.interrupt.mask` `@runtime` field. Defaults to 15
    /// (standard interrupt mask = Damage|Stun|CasterDied|TargetDied).
    /// Tests override via [`Self::set_interrupt_mask`] to exercise
    /// `interrupts: standard - { damage }` semantics.
    interrupt_mask: u32,
}

impl FireboltInterruptProbeState {
    /// Construct a closed-loop runtime with `n_agents` slots. Builds
    /// an AbilityRegistry holding ONE program at `AbilityId(1)` — the
    /// same deferred-cast Firebolt as `firebolt_probe_runtime`.
    ///
    /// Panics if no wgpu adapter is available — call `try_new` for
    /// the fallible variant.
    pub fn new(n_agents: u32, initial_hp: f32) -> Self {
        Self::try_new(n_agents, initial_hp).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — returns `None` when no compatible wgpu
    /// adapter is available on the host. Lets the closed-loop test
    /// degrade to a skip-with-message instead of a panic.
    pub fn try_new(n_agents: u32, initial_hp: f32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        // -- Build the registry: one Firebolt at AbilityId(1).
        //    Same construction as firebolt_probe_runtime — single
        //    EffectOp::CastBegin op (`duration_ticks = 3`).
        let mut program = AbilityProgram::new_single_target(
            /*range*/ 5.0,
            Gate {
                cooldown_ticks:  10,
                hostile_only:    false,
                line_of_sight:   false,
            },
            std::iter::empty(),
        );
        program.effects.push(EffectOp::CastBegin {
            ability_id:     0,
            duration_ticks: 3,
            target_slot:    0,
            target_x_q8:    0,
            target_y_q8:    0,
        });

        let mut builder = AbilityRegistryBuilder::new();
        let id = builder.register(program);
        debug_assert_eq!(
            id,
            AbilityId::new(1).unwrap(),
            "first registered program must land at AbilityId(1)",
        );
        let registry = builder.build();
        let packed = PackedAbilityRegistry::pack(&registry);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "firebolt_interrupt_probe");

        // -- Agent SoA: alive=1, level=1, hp=initial_hp.
        let alive_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("firebolt_interrupt_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_level_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("firebolt_interrupt_probe::agent_level"),
                contents: bytemuck::cast_slice(&level_init),
                usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let hp_init: Vec<f32> = vec![initial_hp; n_agents as usize];
        let agent_hp_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("firebolt_interrupt_probe::agent_hp"),
                contents: bytemuck::cast_slice(&hp_init),
                usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let agent_hp_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("firebolt_interrupt_probe::agent_hp_staging"),
            size:               (n_agents as u64) * 4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Per-stat scaling columns (all-zero — see firebolt_probe).
        let zeros_f32: Vec<f32> = vec![0.0_f32; n_agents as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("firebolt_interrupt_probe::agent_attack_damage");
        let agent_ability_power_buf = mk_stat("firebolt_interrupt_probe::agent_ability_power");
        let agent_max_hp_buf        = mk_stat("firebolt_interrupt_probe::agent_max_hp");
        let agent_armor_buf         = mk_stat("firebolt_interrupt_probe::agent_armor");
        let agent_magic_resist_buf  = mk_stat("firebolt_interrupt_probe::agent_magic_resist");
        let agent_move_speed_buf    = mk_stat("firebolt_interrupt_probe::agent_move_speed");
        let agent_mana_buf          = mk_stat("firebolt_interrupt_probe::agent_mana");

        // -- Per-agent busy SoA. All zero at start so every agent's
        //    `where (busy_until_tick == 0)` gate fires on tick 0.
        let zeros_u32: Vec<u32> = vec![0u32; n_agents as usize];
        let mk_busy = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytemuck::cast_slice(&zeros_u32),
                usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let agent_busy_until_tick_buf      = mk_busy("firebolt_interrupt_probe::agent_busy_until_tick");
        let agent_busy_with_ability_id_buf = mk_busy("firebolt_interrupt_probe::agent_busy_with_ability_id");
        let agent_busy_started_at_tick_buf = mk_busy("firebolt_interrupt_probe::agent_busy_started_at_tick");

        // -- Event ring + tail + tail readback staging.
        let event_ring = EventRing::new(&gpu, "firebolt_interrupt_probe");
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("firebolt_interrupt_probe::event_tail_staging"),
            size:               4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniforms. Per-rule struct shapes match the compiler
        //    emit:
        //      per_agent emitters (DispatchFirebolt, InjectDamage,
        //        ResolveBusy, seed_indirect_0):
        //          (agent_cap, tick, seed, _pad)
        //      post consumers (RecordCastBegin, InterruptCastOnDamage,
        //        ApplyChronicleDamage):
        //          (event_count, tick, seed, agent_cap)
        //    Initial values get overwritten in step() before each
        //    dispatch.
        let mk_uniform = |label: &str, bytes: &[u8]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytes,
                usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let dispatch_cfg_init = physics_DispatchFirebolt::PhysicsDispatchFireboltCfg {
            agent_cap: n_agents, tick: 0, seed: 0, _pad: 0,
        };
        let dispatch_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::dispatch_cfg",
            bytemuck::bytes_of(&dispatch_cfg_init),
        );
        let inject_cfg_init = physics_InjectDamage::PhysicsInjectDamageCfg {
            agent_cap: n_agents, tick: 0, seed: 0, _pad: 0,
        };
        let inject_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::inject_cfg",
            bytemuck::bytes_of(&inject_cfg_init),
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: n_agents, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::seed_cfg",
            bytemuck::bytes_of(&seed_cfg_init),
        );
        let record_cfg_init = physics_RecordCastBegin::PhysicsRecordCastBeginCfg {
            event_count: 0, tick: 0, seed: 0, agent_cap: n_agents,
        };
        let record_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::record_cfg",
            bytemuck::bytes_of(&record_cfg_init),
        );
        // Plan G tunable cfg — `config.interrupt.mask` is `@runtime`,
        // so the kernel's Cfg struct now carries `config_interrupt_mask:
        // u32`. Initial value mirrors the .sim default (15 = standard
        // interrupt mask) so callers that don't override the mask
        // (`set_interrupt_mask`) keep the prior behaviour. Tests that
        // exercise the `standard - { damage }` semantics overwrite this
        // via `set_interrupt_mask(14)` after construction.
        let interrupt_cfg_init = physics_InterruptCastOnDamage::PhysicsInterruptCastOnDamageCfg {
            event_count: 0, tick: 0, seed: 0, agent_cap: n_agents,
            config_interrupt_mask: 15,
        };
        let interrupt_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::interrupt_cfg",
            bytemuck::bytes_of(&interrupt_cfg_init),
        );
        let resolve_cfg_init = physics_ResolveBusy::PhysicsResolveBusyCfg {
            agent_cap: n_agents, tick: 0, seed: 0, _pad: 0,
        };
        let resolve_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::resolve_cfg",
            bytemuck::bytes_of(&resolve_cfg_init),
        );
        let damage_cfg_init = physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageCfg {
            event_count: 0, tick: 0, seed: 0, agent_cap: n_agents,
        };
        let damage_cfg_buf = mk_uniform(
            "firebolt_interrupt_probe::damage_cfg",
            bytemuck::bytes_of(&damage_cfg_init),
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_level_buf,
            agent_hp_buf,
            agent_hp_staging,
            agent_attack_damage_buf,
            agent_ability_power_buf,
            agent_max_hp_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            agent_mana_buf,
            agent_busy_until_tick_buf,
            agent_busy_with_ability_id_buf,
            agent_busy_started_at_tick_buf,
            registry_gpu,
            event_ring,
            event_tail_staging,
            dispatch_cfg_buf,
            inject_cfg_buf,
            seed_cfg_buf,
            record_cfg_buf,
            interrupt_cfg_buf,
            resolve_cfg_buf,
            damage_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
            initial_hp,
            tick: 0,
            // Default = 15 (standard interrupt mask). Mirrors the .sim
            // default; `set_interrupt_mask` overrides per test.
            interrupt_mask: 15,
        })
    }

    /// Plan G tunable cfg — override the per-tick interrupt mask the
    /// `InterruptCastOnDamage` kernel reads from
    /// `cfg.config_interrupt_mask`. Bit layout matches
    /// `engine::ability::interrupt::InterruptKind`:
    ///   bit 0 = Damage, bit 1 = Stun, bit 2 = CasterDied,
    ///   bit 3 = TargetDied, bit 4 = Movement.
    /// `set_interrupt_mask(14)` clears bit 0 (= `standard - { damage }`)
    /// — proves the mask gate suppresses damage-driven interrupts.
    /// The new value applies on the next `step()` call (the cfg buffer
    /// is rewritten at the start of every step).
    pub fn set_interrupt_mask(&mut self, mask: u32) {
        self.interrupt_mask = mask;
    }

    /// Block on the GPU and read `event_tail` back into a host u32.
    /// Used between encoder stages to populate the consumer kernels'
    /// `event_count` cfg field.
    fn read_event_tail(&self) -> u32 {
        let slice = self.event_tail_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("event_tail_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during event_tail readback");
        let v = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            words[0]
        };
        self.event_tail_staging.unmap();
        v
    }

    /// Encode + dispatch one tick of the closed-loop pipeline.
    ///
    /// See module docs for the full per-tick chain. Three encoder
    /// submissions, two intermediate `event_tail` readbacks.
    pub fn step(&mut self) {
        let tick = self.tick;

        // Refresh per-tick cfg uniforms (tick + agent_cap). The
        // event_count fields on the consumer cfgs are filled in
        // between encoder submissions once we read tail back.
        let dispatch_cfg = physics_DispatchFirebolt::PhysicsDispatchFireboltCfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.dispatch_cfg_buf, 0, bytemuck::bytes_of(&dispatch_cfg));
        let inject_cfg = physics_InjectDamage::PhysicsInjectDamageCfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.inject_cfg_buf, 0, bytemuck::bytes_of(&inject_cfg));
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let resolve_cfg = physics_ResolveBusy::PhysicsResolveBusyCfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.resolve_cfg_buf, 0, bytemuck::bytes_of(&resolve_cfg));

        // -- Stage 1: clear tail + per_agent emitters
        //    (DispatchFirebolt + InjectDamage) + seed_indirect_0.
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("firebolt_interrupt_probe::step::per_agent_emit"),
            },
        );
        self.event_ring.clear_tail_in(&mut encoder);

        let agent_buffers = AgentBuffers {
            alive_buf:         Some(&self.agent_alive_buf),
            level_buf:         Some(&self.agent_level_buf),
            hp_buf:            Some(&self.agent_hp_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            max_hp_buf:        Some(&self.agent_max_hp_buf),
            armor_buf:         Some(&self.agent_armor_buf),
            magic_resist_buf:  Some(&self.agent_magic_resist_buf),
            move_speed_buf:    Some(&self.agent_move_speed_buf),
            mana_buf:          Some(&self.agent_mana_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state:      &agent_buffers,
            event_ring: &self.event_ring,
            registry:   &self.registry_gpu,
            voxel_grid: None,
        };

        let dispatch_extras = physics_DispatchFirebolt::PhysicsDispatchFireboltExtras {
            agent_busy_until_tick: &self.agent_busy_until_tick_buf,
            cfg:                   &self.dispatch_cfg_buf,
        };
        let dispatch_bindings =
            physics_DispatchFirebolt::PhysicsDispatchFireboltBindings::from_context_with_extras(
                &ctx, &dispatch_extras,
            );
        dispatch::dispatch_physics_dispatchfirebolt(
            &mut self.cache, &dispatch_bindings, &self.gpu.device, &mut encoder, self.n_agents,
        );

        let inject_extras = physics_InjectDamage::PhysicsInjectDamageExtras {
            cfg: &self.inject_cfg_buf,
        };
        let inject_bindings =
            physics_InjectDamage::PhysicsInjectDamageBindings::from_context_with_extras(
                &ctx, &inject_extras,
            );
        dispatch::dispatch_physics_injectdamage(
            &mut self.cache, &inject_bindings, &self.gpu.device, &mut encoder, self.n_agents,
        );

        let seed_extras = seed_indirect_0::SeedIndirect0Extras {
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg:             &self.seed_cfg_buf,
        };
        let seed_bindings =
            seed_indirect_0::SeedIndirect0Bindings::from_context_with_extras(&ctx, &seed_extras);
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache, &seed_bindings, &self.gpu.device, &mut encoder, self.n_agents,
        );

        encoder.copy_buffer_to_buffer(self.event_ring.tail(), 0, &self.event_tail_staging, 0, 4);
        self.gpu.queue.submit(Some(encoder.finish()));

        let post_event_count = self.read_event_tail();

        // -- Stage 2: post-pass consumers + ResolveBusy.
        //    Order matters:
        //      RecordCastBegin (stamps busy from this tick's CastBegins)
        //      InterruptCastOnDamage (clears busy from inject-damage events)
        //      ResolveBusy (per_agent emitter for the deferred-resolve damage)
        //    InterruptCastOnDamage runs BEFORE ResolveBusy so a
        //    same-tick interrupt prevents a same-tick resolve.
        let record_cfg = physics_RecordCastBegin::PhysicsRecordCastBeginCfg {
            event_count: post_event_count,
            tick,
            seed:        0,
            agent_cap:   self.n_agents,
        };
        self.gpu.queue.write_buffer(&self.record_cfg_buf, 0, bytemuck::bytes_of(&record_cfg));
        let interrupt_cfg = physics_InterruptCastOnDamage::PhysicsInterruptCastOnDamageCfg {
            event_count: post_event_count,
            tick,
            seed:        0,
            agent_cap:   self.n_agents,
            // Plan G tunable cfg — sourced from
            // `Self::interrupt_mask` (default 15; tests override via
            // `set_interrupt_mask`). Determines whether
            // EffectDamageApplied events clear the busy SoA on a
            // mid-cast target.
            config_interrupt_mask: self.interrupt_mask,
        };
        self.gpu.queue.write_buffer(
            &self.interrupt_cfg_buf, 0, bytemuck::bytes_of(&interrupt_cfg),
        );

        let mut encoder2 = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("firebolt_interrupt_probe::step::post_and_resolve"),
            },
        );

        let agent_buffers = AgentBuffers {
            alive_buf:         Some(&self.agent_alive_buf),
            level_buf:         Some(&self.agent_level_buf),
            hp_buf:            Some(&self.agent_hp_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            max_hp_buf:        Some(&self.agent_max_hp_buf),
            armor_buf:         Some(&self.agent_armor_buf),
            magic_resist_buf:  Some(&self.agent_magic_resist_buf),
            move_speed_buf:    Some(&self.agent_move_speed_buf),
            mana_buf:          Some(&self.agent_mana_buf),
            ..Default::default()
        };
        let ctx2 = KernelBindingsContext {
            state:      &agent_buffers,
            event_ring: &self.event_ring,
            registry:   &self.registry_gpu,
            voxel_grid: None,
        };

        let record_extras = physics_RecordCastBegin::PhysicsRecordCastBeginExtras {
            agent_busy_until_tick:      &self.agent_busy_until_tick_buf,
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            agent_busy_started_at_tick: &self.agent_busy_started_at_tick_buf,
            cfg:                        &self.record_cfg_buf,
        };
        let record_bindings =
            physics_RecordCastBegin::PhysicsRecordCastBeginBindings::from_context_with_extras(
                &ctx2, &record_extras,
            );
        // The post-pass kernels dispatch `(agent_cap+63)/64` workgroups;
        // pass enough threads to cover at least one record per agent.
        let post_dispatch_count = self.n_agents.max(post_event_count);
        dispatch::dispatch_physics_recordcastbegin(
            &mut self.cache, &record_bindings, &self.gpu.device, &mut encoder2, post_dispatch_count,
        );

        let interrupt_extras = physics_InterruptCastOnDamage::PhysicsInterruptCastOnDamageExtras {
            agent_busy_until_tick:      &self.agent_busy_until_tick_buf,
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            cfg:                        &self.interrupt_cfg_buf,
        };
        let interrupt_bindings =
            physics_InterruptCastOnDamage::PhysicsInterruptCastOnDamageBindings::from_context_with_extras(
                &ctx2, &interrupt_extras,
            );
        dispatch::dispatch_physics_interruptcastondamage(
            &mut self.cache,
            &interrupt_bindings,
            &self.gpu.device,
            &mut encoder2,
            post_dispatch_count,
        );

        let resolve_extras = physics_ResolveBusy::PhysicsResolveBusyExtras {
            agent_busy_until_tick:      &self.agent_busy_until_tick_buf,
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            cfg:                        &self.resolve_cfg_buf,
        };
        let resolve_bindings =
            physics_ResolveBusy::PhysicsResolveBusyBindings::from_context_with_extras(
                &ctx2, &resolve_extras,
            );
        dispatch::dispatch_physics_resolvebusy(
            &mut self.cache, &resolve_bindings, &self.gpu.device, &mut encoder2, self.n_agents,
        );

        encoder2.copy_buffer_to_buffer(self.event_ring.tail(), 0, &self.event_tail_staging, 0, 4);
        self.gpu.queue.submit(Some(encoder2.finish()));

        let damage_event_count = self.read_event_tail();

        // -- Stage 3: ApplyChronicleDamage. The tail now holds inject
        //    damage records + resolve damage records; the consumer's
        //    PerEvent kernel filters by kind tag, so feeding it the
        //    full count is safe.
        let damage_cfg = physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageCfg {
            event_count: damage_event_count,
            tick,
            seed:        0,
            agent_cap:   self.n_agents,
        };
        self.gpu.queue.write_buffer(&self.damage_cfg_buf, 0, bytemuck::bytes_of(&damage_cfg));

        let mut encoder3 = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("firebolt_interrupt_probe::step::apply_damage"),
            },
        );

        let agent_buffers = AgentBuffers {
            alive_buf:         Some(&self.agent_alive_buf),
            level_buf:         Some(&self.agent_level_buf),
            hp_buf:            Some(&self.agent_hp_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            max_hp_buf:        Some(&self.agent_max_hp_buf),
            armor_buf:         Some(&self.agent_armor_buf),
            magic_resist_buf:  Some(&self.agent_magic_resist_buf),
            move_speed_buf:    Some(&self.agent_move_speed_buf),
            mana_buf:          Some(&self.agent_mana_buf),
            ..Default::default()
        };
        let ctx3 = KernelBindingsContext {
            state:      &agent_buffers,
            event_ring: &self.event_ring,
            registry:   &self.registry_gpu,
            voxel_grid: None,
        };

        let damage_extras = physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageExtras {
            cfg: &self.damage_cfg_buf,
        };
        let damage_bindings =
            physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageBindings::from_context_with_extras(
                &ctx3, &damage_extras,
            );
        dispatch::dispatch_physics_applychronicledamage(
            &mut self.cache,
            &damage_bindings,
            &self.gpu.device,
            &mut encoder3,
            self.n_agents.max(damage_event_count),
        );

        self.gpu.queue.submit(Some(encoder3.finish()));

        self.tick += 1;
    }

    /// Block on the GPU and read back the per-agent hp array.
    pub fn read_agent_hp(&self) -> Vec<f32> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("firebolt_interrupt_probe::read_agent_hp"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.agent_hp_buf, 0, &self.agent_hp_staging, 0, (self.n_agents as u64) * 4,
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = self.agent_hp_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("agent_hp_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during agent_hp readback");
        let out = {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            floats.to_vec()
        };
        self.agent_hp_staging.unmap();
        out
    }

    /// Block on the GPU and read back the per-agent `busy_until_tick`
    /// SoA. Used by tests to confirm the cast lifecycle (cast → busy →
    /// interrupt → re-cast) is observable from the host.
    pub fn read_busy_until_tick(&self) -> Vec<u32> {
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("firebolt_interrupt_probe::busy_until_tick_staging"),
            size:               (self.n_agents as u64) * 4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("firebolt_interrupt_probe::read_busy_until_tick"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.agent_busy_until_tick_buf,
            0,
            &staging,
            0,
            (self.n_agents as u64) * 4,
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("busy_until_tick_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during busy_until_tick readback");
        let out = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            words.to_vec()
        };
        staging.unmap();
        out
    }

    pub fn n_agents(&self) -> u32 {
        self.n_agents
    }

    pub fn tick(&self) -> u32 {
        self.tick
    }
}

#[cfg(test)]
mod closed_loop_tests {
    use super::*;

    /// **Plan G G2.5 behavioural pin.** Demonstrates the
    /// interrupt-detection path closes the loop on real wgpu hardware.
    ///
    /// With the .sim's default config (`inject_at_tick = 1`,
    /// `inject_amount = 10.0`, `firebolt damage = 25.0`,
    /// `cast_duration = 3 ticks`):
    ///
    ///   - Tick 0: cast begins. busy_until_tick = 3. hp = [100, 100].
    ///   - Tick 1: InjectDamage emits 10 dmg →
    ///     InterruptCastOnDamage clears busy → ApplyChronicleDamage
    ///     drops hp. hp = [90, 90].
    ///   - Tick 2: dispatcher re-casts. busy_until_tick = 5.
    ///     hp = [90, 90].
    ///   - Tick 3: ResolveBusy predicate fails (3 < 5). hp stays at
    ///     [90, 90]. **This is the critical assertion — without the
    ///     interrupt path the firebolt_probe baseline drops to 75.**
    ///
    /// **Skip path.** When `GpuContext::new_blocking` returns Err
    /// (no compatible wgpu adapter on the host), the test prints a
    /// skip message and returns. The build itself still validated
    /// kernel emit + binding hookup at compile time.
    #[test]
    fn cast_interrupted_by_external_damage() {
        let n_agents:   u32 = 2;
        let initial_hp: f32 = 100.0;
        let inject_amount: f32 = 10.0;

        let mut state = match FireboltInterruptProbeState::try_new(n_agents, initial_hp) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[firebolt_interrupt_probe closed-loop] skipping: no wgpu \
                     adapter available on this host. The build itself still \
                     validates the kernel emit (firebolt_interrupt_probe.sim \
                     → 11 WGSL kernels including physics_InjectDamage and \
                     physics_InterruptCastOnDamage) and binding hookup at \
                     compile time.",
                );
                return;
            }
        };

        // -- Tick 0: cast begins. RecordCastBegin stamps
        //    busy_until_tick = world.tick + duration = 0 + 3 = 3.
        //    No InjectDamage (tick != 1). hp stays at 100.
        state.step();
        let hp = state.read_agent_hp();
        assert_eq!(hp.len(), n_agents as usize);
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - initial_hp).abs() < 1e-4,
                "tick 0: agent {i} hp = {h}, expected {initial_hp} (cast just started)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(b, 3, "tick 0: agent {i} busy_until_tick = {b}, expected 3");
        }

        // -- Tick 1: InjectDamage fires (tick == inject_at_tick).
        //    InterruptCastOnDamage sees busy_until_tick=3 > 0 →
        //    clears busy. ApplyChronicleDamage decrements hp by 10.
        //    hp = [90, 90]; busy_until_tick = [0, 0].
        state.step();
        let hp = state.read_agent_hp();
        let expected_hp_t1 = initial_hp - inject_amount;
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - expected_hp_t1).abs() < 1e-4,
                "tick 1: agent {i} hp = {h}, expected {expected_hp_t1} (injected dmg landed)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(
                b, 0,
                "tick 1: agent {i} busy_until_tick = {b}, expected 0 (interrupt cleared busy)",
            );
        }

        // -- Tick 2: dispatcher sees busy == 0 → fires NEW cast.
        //    RecordCastBegin stamps busy_until_tick = 2 + 3 = 5.
        //    No InjectDamage (tick != 1). No resolve (busy_until_tick=5
        //    > tick=2). hp stays at 90.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - expected_hp_t1).abs() < 1e-4,
                "tick 2: agent {i} hp = {h}, expected {expected_hp_t1} (re-cast started, no dmg)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(
                b, 5,
                "tick 2: agent {i} busy_until_tick = {b}, expected 5 (second cast stamped)",
            );
        }

        // -- Tick 3: **CRITICAL ASSERTION.** Without interrupt the
        //    original cast WOULD fire damage 25 here (firebolt_probe
        //    baseline gets hp=75 at tick 3). With interrupt, the
        //    busy_until_tick is 5 (from the second cast at tick 2),
        //    so ResolveBusy's `world.tick >= busy_until_tick` predicate
        //    fails (3 < 5). No resolve. hp stays at 90.
        //
        //    The diff vs firebolt_probe (90 not 75) IS the test value
        //    — it proves InterruptCastOnDamage cleared the original
        //    cast's busy state at tick 1 before its tick-3 resolve.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - expected_hp_t1).abs() < 1e-4,
                "tick 3: agent {i} hp = {h}, expected {expected_hp_t1} \
                 (interrupt prevented original cast's resolve — \
                 baseline would be {})",
                initial_hp - 25.0,
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(
                b, 5,
                "tick 3: agent {i} busy_until_tick = {b}, expected 5 (second cast still busy)",
            );
        }
    }

    /// **Plan G tunable cfg behavioural pin.** Same fixture as
    /// `cast_interrupted_by_external_damage` but with the interrupt
    /// mask narrowed to `14` (= 0b01110 = `standard - { damage }` —
    /// bit 0 cleared). The mask gate in
    /// `physics_InterruptCastOnDamage`'s WGSL body is
    /// `(cfg.config_interrupt_mask % 2u) == 1u`; with the bit cleared
    /// the gate evaluates `false`, so EffectDamageApplied events
    /// DON'T clear the busy SoA. The original cast's tick-3 resolve
    /// fires normally.
    ///
    /// Per-tick chain (with `mask = 14`):
    ///   * Tick 0: cast begins. busy_until_tick = 3. hp = [100, 100].
    ///   * Tick 1: InjectDamage emits 10 dmg. InterruptCastOnDamage
    ///     sees busy=3>0 BUT `(mask % 2) != 1` → no-op.
    ///     ApplyChronicleDamage drops hp by 10. hp = [90, 90].
    ///   * Tick 2: dispatcher sees busy=3 still > 0 → does NOT cast.
    ///     No new busy stamp. hp = [90, 90].
    ///   * Tick 3: ResolveBusy fires (`world.tick >= busy_until_tick`
    ///     === `3 >= 3`). Emits EffectDamageApplied{amount=25}.
    ///     ApplyChronicleDamage drops hp by 25. hp = [65, 65].
    ///
    /// **The hp=65 assertion at tick 3 is the critical signal.** Diff
    /// vs `cast_interrupted_by_external_damage` (which sees hp=90 at
    /// tick 3) is exactly 25 hp = the firebolt resolve damage that
    /// was suppressed when the mask included Damage and that lands
    /// here when it doesn't.
    #[test]
    fn cast_with_mask_excluding_damage_resolves_normally() {
        let n_agents:   u32 = 2;
        let initial_hp: f32 = 100.0;
        let inject_amount:   f32 = 10.0;
        let firebolt_damage: f32 = 25.0;

        let mut state = match FireboltInterruptProbeState::try_new(n_agents, initial_hp) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[firebolt_interrupt_probe @runtime mask] skipping: no \
                     wgpu adapter available on this host. The build itself \
                     still validates the kernel emit (firebolt_interrupt_probe.sim \
                     → 11 WGSL kernels including \
                     physics_InterruptCastOnDamage with cfg.config_interrupt_mask) \
                     and binding hookup at compile time.",
                );
                return;
            }
        };

        // Override the interrupt mask BEFORE any step. 14 = 0b01110 =
        // standard with bit 0 (Damage) cleared. The next step()'s cfg
        // refresh writes this into the InterruptCastOnDamage uniform.
        state.set_interrupt_mask(14);

        // -- Tick 0: cast begins. busy_until_tick = 3. hp = 100.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - initial_hp).abs() < 1e-4,
                "tick 0: agent {i} hp = {h}, expected {initial_hp} (cast just started)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(b, 3, "tick 0: agent {i} busy_until_tick = {b}, expected 3");
        }

        // -- Tick 1: InjectDamage fires. InterruptCastOnDamage sees
        //    busy=3>0 BUT mask gate fails (14 % 2 == 0) → no-op.
        //    ApplyChronicleDamage drops hp by 10. busy STAYS at 3.
        //    hp = [90, 90]; busy_until_tick = [3, 3].
        state.step();
        let hp = state.read_agent_hp();
        let expected_hp_t1 = initial_hp - inject_amount; // 90.0
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - expected_hp_t1).abs() < 1e-4,
                "tick 1: agent {i} hp = {h}, expected {expected_hp_t1} \
                 (injected dmg landed; mask gate suppressed interrupt)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(
                b, 3,
                "tick 1: agent {i} busy_until_tick = {b}, expected 3 \
                 (mask=14 suppressed damage-driven interrupt; original \
                 cast still busy)",
            );
        }

        // -- Tick 2: dispatcher sees busy=3 still > 0 → does NOT
        //    re-cast. No InjectDamage. hp stays at 90; busy stays at 3.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - expected_hp_t1).abs() < 1e-4,
                "tick 2: agent {i} hp = {h}, expected {expected_hp_t1} \
                 (busy still set; no new cast, no damage)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(
                b, 3,
                "tick 2: agent {i} busy_until_tick = {b}, expected 3 \
                 (original cast still busy)",
            );
        }

        // -- Tick 3: **CRITICAL ASSERTION.** ResolveBusy predicate
        //    fires (`world.tick >= busy_until_tick` === `3 >= 3`).
        //    Emits EffectDamageApplied{amount=25}. The same chronicle
        //    consumer that JUST cleared the busy state in
        //    ResolveBusy's body also feeds InterruptCastOnDamage —
        //    but mask=14 keeps that no-op. ApplyChronicleDamage runs
        //    against the resolve event, dropping hp by 25.
        //    hp = [65, 65]. The diff vs the mask=15 baseline (90) is
        //    exactly 25 = the firebolt resolve damage that didn't get
        //    suppressed.
        state.step();
        let hp = state.read_agent_hp();
        let expected_hp_t3 = expected_hp_t1 - firebolt_damage; // 65.0
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - expected_hp_t3).abs() < 1e-4,
                "tick 3: agent {i} hp = {h}, expected {expected_hp_t3} \
                 (mask=14 suppressed interrupt → original cast resolved \
                 → 25 firebolt damage landed on top of the 10 inject \
                 damage). Mask=15 baseline would be 90.",
            );
        }
    }
}
