//! Per-fixture runtime for `assets/sim/firebolt_probe.sim` — Plan G G4.
//!
//! ## What this exercises
//!
//! Closes the cast → busy → resolve → damage loop end-to-end on real
//! wgpu hardware. The fixture has 2 agents that each cast a deferred
//! Firebolt at themselves. The Firebolt program is registered at
//! `AbilityId(1)` with a single `EffectOp::CastBegin` op
//! (`duration_ticks = 3`); the busy-resolution rule fires the
//! hardcoded 25.0 damage from `config.firebolt.damage` on resolve.
//!
//! ## Per-tick chain
//!
//! 1. `event_tail` cleared to 0 (so the dispatcher's atomic counter
//!    starts at 0 each tick).
//! 2. `physics_DispatchFirebolt` (per-agent) — when `busy_until_tick
//!    == 0`, writes `EffectCastBeginApplied` chronicle records into
//!    `event_ring`.
//! 3. `seed_indirect_0` — fills `indirect_args_0` from the post-dispatch
//!    tail (kept for parity).
//! 4. Read tail back to host so the consumer kernels can be parameterised
//!    with the right `event_count`.
//! 5. `physics_RecordCastBegin` (post) — chronicle consumer for
//!    `EffectCastBeginApplied`; stamps the three busy SoA fields the
//!    fixture writes (`busy_until_tick`, `busy_with_ability_id`,
//!    `busy_started_at_tick` — see the `EffectCastBeginApplied`
//!    declaration in the .sim for why `busy_target_slot` is left
//!    unwritten today).
//! 6. `physics_ResolveBusy` (per-agent) — fires `EffectDamageApplied`
//!    when `world.tick >= busy_until_tick > 0`, clears busy state.
//! 7. Read tail again (the resolve stage adds new damage records on
//!    top of whatever the dispatch stage wrote this tick).
//! 8. `physics_ApplyChronicleDamage` (post) — decrements `agent_hp` by
//!    the chronicle's `amount` field for each `EffectDamageApplied`
//!    record.
//!
//! ## Behavioural pin
//!
//! With `n_agents = 2`, `initial_hp = 100.0`, `firebolt damage = 25`,
//! `cast duration = 3 ticks`:
//!
//! - Tick 0: cast begins (busy_until_tick = 3). hp unchanged.
//! - Tick 1, 2: still casting. hp unchanged.
//! - Tick 3: `world.tick >= busy_until_tick`, ResolveBusy fires →
//!   ApplyChronicleDamage decrements hp by 25 → hp = [75, 75].
//! - Tick 4: agent now idle (busy cleared); dispatcher kicks off a
//!   new cast (busy_until_tick = 7).
//! - Tick 7: second cast resolves → hp = [50, 50].
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
// Plan E-A5 pilot — runtime_core.rs MUST be included at crate root
// because it references `dispatch::KernelCache`, `schedule::SCHEDULE` etc.
// at module-relative paths.
include!(concat!(env!("OUT_DIR"), "/runtime_core.rs"));

/// Per-fixture state for the firebolt-probe closed-loop demo. Owns:
///   - The wgpu context.
///   - Standard agent SoA (alive, hp, level + the per-stat scaling
///     buffers the apply_ability dispatcher binds).
///   - The four per-agent busy SoA columns Plan G G2.7 added
///     (busy_until_tick, busy_with_ability_id, busy_started_at_tick,
///     busy_target_slot — all u32).
///   - Event-ring + tail buffers.
///   - Staging buffers for `event_tail` and `agent_hp` host readback.
///   - Per-kernel cfg uniforms.
///   - The packed-registry GPU buffers (uploaded once at construction).
pub struct FireboltProbeState {
    gpu: GpuContext,

    // -- Agent SoA (standard columns the dispatcher binds) --
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_hp_staging: wgpu::Buffer,
    // Per-stat scaling columns. The Firebolt program is just CastBegin
    // (no stat scaling on the cast op itself), so these stay all-zero
    // — `scale_bonus = 0.0` regardless. Wired so the dispatcher's
    // `from_context_with_extras` can resolve every standard column it
    // binds.
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Per-agent busy SoA (Plan G G2.7) — fixture extras. The MVP
    //    .sim doesn't write busy_target_slot (the chronicle's `target`
    //    is an AgentId; the DSL doesn't yet auto-coerce AgentId → u32
    //    on the rhs of a u32 SoA setter), so we don't allocate that
    //    column here either. Add it back when interrupts (G2.5) or
    //    threats-zone projection (G3) need it.
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
    seed_cfg_buf: wgpu::Buffer,
    record_cfg_buf: wgpu::Buffer,
    resolve_cfg_buf: wgpu::Buffer,
    damage_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
    /// Initial hp for every agent (host-only — used by tests for
    /// expected-state computation).
    pub initial_hp: f32,
    /// Current tick — incremented at the end of every `step()` call.
    tick: u32,
}

impl FireboltProbeState {
    /// Construct a closed-loop runtime with `n_agents` slots. Builds an
    /// AbilityRegistry holding ONE program at `AbilityId(1)` — a
    /// deferred-cast Firebolt that lowers a `cast { duration: 3t }` ability
    /// (single `EffectOp::CastBegin { duration_ticks: 3 }`).
    ///
    /// Panics if no wgpu adapter is available — call `try_new` for the
    /// fallible variant.
    pub fn new(n_agents: u32, initial_hp: f32) -> Self {
        Self::try_new(n_agents, initial_hp).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — returns `None` when no compatible wgpu
    /// adapter is available on the host. Lets the closed-loop test
    /// degrade to a skip-with-message instead of a panic.
    pub fn try_new(n_agents: u32, initial_hp: f32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        // -- Build the registry: one Firebolt at AbilityId(1).
        //
        // Construct the AbilityProgram directly via the engine API
        // rather than parsing a .ability file (matches what the
        // .ability lowering would produce — see
        // crates/dsl_compiler/src/ability_lower.rs:909-960 for the
        // CastBegin lowering arm). The placeholder fields
        // (`ability_id`, `target_slot`, `target_x_q8`, `target_y_q8`)
        // mirror the lowering — the apply path overrides them at
        // dispatch.
        let mut program = AbilityProgram::new_single_target(
            /*range*/ 5.0,
            Gate {
                cooldown_ticks:  10,
                hostile_only:    false,
                line_of_sight:   false,
            },
            // Empty — the CastBegin op pushed below is the entire
            // ability program for this fixture.
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
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "firebolt_probe");

        // -- Agent SoA: alive=1, level=1, hp=initial_hp.
        let alive_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("firebolt_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_level_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("firebolt_probe::agent_level"),
                contents: bytemuck::cast_slice(&level_init),
                usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let hp_init: Vec<f32> = vec![initial_hp; n_agents as usize];
        let agent_hp_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("firebolt_probe::agent_hp"),
                contents: bytemuck::cast_slice(&hp_init),
                usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let agent_hp_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("firebolt_probe::agent_hp_staging"),
            size:               (n_agents as u64) * 4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Per-stat scaling columns (all-zero — the CastBegin op has
        //    no stat scaling, so `scale_bonus = 0.0` regardless).
        let zeros_f32: Vec<f32> = vec![0.0_f32; n_agents as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("firebolt_probe::agent_attack_damage");
        let agent_ability_power_buf = mk_stat("firebolt_probe::agent_ability_power");
        let agent_max_hp_buf        = mk_stat("firebolt_probe::agent_max_hp");
        let agent_armor_buf         = mk_stat("firebolt_probe::agent_armor");
        let agent_magic_resist_buf  = mk_stat("firebolt_probe::agent_magic_resist");
        let agent_move_speed_buf    = mk_stat("firebolt_probe::agent_move_speed");
        let agent_mana_buf          = mk_stat("firebolt_probe::agent_mana");

        // -- Per-agent busy SoA (Plan G G2.7). All zero at start so
        //    every agent's `where (busy_until_tick == 0)` gate fires
        //    the dispatcher on tick 0.
        let zeros_u32: Vec<u32> = vec![0u32; n_agents as usize];
        let mk_busy = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytemuck::cast_slice(&zeros_u32),
                // COPY_SRC so `read_busy_until_tick` (and any future
                // host-side observers of the busy SoA) can stage the
                // buffer back to the host.
                usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let agent_busy_until_tick_buf      = mk_busy("firebolt_probe::agent_busy_until_tick");
        let agent_busy_with_ability_id_buf = mk_busy("firebolt_probe::agent_busy_with_ability_id");
        let agent_busy_started_at_tick_buf = mk_busy("firebolt_probe::agent_busy_started_at_tick");

        // -- Event ring + tail + tail readback staging.
        let event_ring = EventRing::new(&gpu, "firebolt_probe");
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("firebolt_probe::event_tail_staging"),
            size:               4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniforms. Per-rule struct shapes match the compiler
        //    emit: dispatch + resolve + seed have `(agent_cap, tick,
        //    seed, _pad)`, record + damage have `(event_count, tick,
        //    seed, agent_cap)`. Initial values get overwritten in
        //    step() before each dispatch.
        let mk_uniform = |label: &str, bytes: &[u8]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(label),
                contents: bytes,
                usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let dispatch_cfg_init = physics_DispatchFirebolt::PhysicsDispatchFireboltCfg {
            agent_cap: n_agents,
            tick:      0,
            seed:      0,
            _pad:      0,
        };
        let dispatch_cfg_buf =
            mk_uniform("firebolt_probe::dispatch_cfg", bytemuck::bytes_of(&dispatch_cfg_init));
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: n_agents,
            tick:      0,
            seed:      0,
            _pad:      0,
        };
        let seed_cfg_buf =
            mk_uniform("firebolt_probe::seed_cfg", bytemuck::bytes_of(&seed_cfg_init));
        let record_cfg_init = physics_RecordCastBegin::PhysicsRecordCastBeginCfg {
            event_count: 0,
            tick:        0,
            seed:        0,
            agent_cap:   n_agents,
        };
        let record_cfg_buf =
            mk_uniform("firebolt_probe::record_cfg", bytemuck::bytes_of(&record_cfg_init));
        let resolve_cfg_init = physics_ResolveBusy::PhysicsResolveBusyCfg {
            agent_cap: n_agents,
            tick:      0,
            seed:      0,
            _pad:      0,
        };
        let resolve_cfg_buf =
            mk_uniform("firebolt_probe::resolve_cfg", bytemuck::bytes_of(&resolve_cfg_init));
        let damage_cfg_init = physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageCfg {
            event_count: 0,
            tick:        0,
            seed:        0,
            agent_cap:   n_agents,
        };
        let damage_cfg_buf =
            mk_uniform("firebolt_probe::damage_cfg", bytemuck::bytes_of(&damage_cfg_init));

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
            seed_cfg_buf,
            record_cfg_buf,
            resolve_cfg_buf,
            damage_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
            initial_hp,
            tick: 0,
        })
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

    /// Encode + dispatch one tick of the closed-loop pipeline. The
    /// pipeline splits across THREE encoder submissions because two
    /// `event_tail` readbacks sit between them:
    ///
    ///   Stage 1 (encoder A):
    ///     1. Clear `event_tail`.
    ///     2. Dispatch `physics_DispatchFirebolt` — writes
    ///        `EffectCastBeginApplied` records into `event_ring` for
    ///        every alive agent with `busy_until_tick == 0`.
    ///     3. Dispatch `seed_indirect_0` (parity).
    ///     4. Copy `event_tail` to staging.
    ///   --- readback `event_tail` → `record_event_count` ---
    ///   Stage 2 (encoder B):
    ///     5. Dispatch `physics_RecordCastBegin` — chronicle consumer
    ///        for `EffectCastBeginApplied`; stamps the four busy SoA
    ///        fields. The `record_cfg.event_count` is set from the
    ///        post-dispatch tail.
    ///     6. Dispatch `physics_ResolveBusy` — fires
    ///        `EffectDamageApplied` when busy elapses, clears busy
    ///        state.
    ///     7. Copy `event_tail` to staging again (it now holds the
    ///        post-resolve count: dispatch records + any damage
    ///        records emitted by ResolveBusy).
    ///   --- readback `event_tail` → `damage_event_count` ---
    ///   Stage 3 (encoder C):
    ///     8. Dispatch `physics_ApplyChronicleDamage` — decrements
    ///        `agent_hp` by the chronicle's `amount` field for each
    ///        damage record.
    pub fn step(&mut self) {
        let tick = self.tick;

        // Refresh per-tick cfg uniforms (tick + agent_cap).
        let dispatch_cfg = physics_DispatchFirebolt::PhysicsDispatchFireboltCfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.dispatch_cfg_buf, 0, bytemuck::bytes_of(&dispatch_cfg));
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let resolve_cfg = physics_ResolveBusy::PhysicsResolveBusyCfg {
            agent_cap: self.n_agents, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.resolve_cfg_buf, 0, bytemuck::bytes_of(&resolve_cfg));

        // -- Stage 1: clear tail + run dispatcher + seed_indirect_0.
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("firebolt_probe::step::dispatch") },
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

        let record_event_count = self.read_event_tail();

        // -- Stage 2: RecordCastBegin (consumer for EffectCastBeginApplied)
        //    + ResolveBusy (per-agent emitter for EffectDamageApplied).
        let record_cfg = physics_RecordCastBegin::PhysicsRecordCastBeginCfg {
            event_count: record_event_count,
            tick,
            seed:        0,
            agent_cap:   self.n_agents,
        };
        self.gpu.queue.write_buffer(&self.record_cfg_buf, 0, bytemuck::bytes_of(&record_cfg));

        let mut encoder2 = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("firebolt_probe::step::record_resolve") },
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
        let ctx = KernelBindingsContext {
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
                &ctx, &record_extras,
            );
        // RecordCastBegin's `record()` dispatches `(agent_cap+63)/64`
        // workgroups; pass enough to cover at least one record per agent.
        dispatch::dispatch_physics_recordcastbegin(
            &mut self.cache,
            &record_bindings,
            &self.gpu.device,
            &mut encoder2,
            self.n_agents.max(record_event_count),
        );

        let resolve_extras = physics_ResolveBusy::PhysicsResolveBusyExtras {
            agent_busy_until_tick:      &self.agent_busy_until_tick_buf,
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            cfg:                        &self.resolve_cfg_buf,
        };
        let resolve_bindings =
            physics_ResolveBusy::PhysicsResolveBusyBindings::from_context_with_extras(
                &ctx, &resolve_extras,
            );
        dispatch::dispatch_physics_resolvebusy(
            &mut self.cache, &resolve_bindings, &self.gpu.device, &mut encoder2, self.n_agents,
        );

        encoder2.copy_buffer_to_buffer(self.event_ring.tail(), 0, &self.event_tail_staging, 0, 4);
        self.gpu.queue.submit(Some(encoder2.finish()));

        let damage_event_count = self.read_event_tail();

        // -- Stage 3: ApplyChronicleDamage (consumer for
        //    EffectDamageApplied). The tail now holds dispatch records +
        //    resolve records; the consumer's PerEvent kernel filters by
        //    kind tag, so feeding it the full count is safe.
        let damage_cfg = physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageCfg {
            event_count: damage_event_count,
            tick,
            seed:        0,
            agent_cap:   self.n_agents,
        };
        self.gpu.queue.write_buffer(&self.damage_cfg_buf, 0, bytemuck::bytes_of(&damage_cfg));

        let mut encoder3 = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("firebolt_probe::step::damage") },
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
        let ctx = KernelBindingsContext {
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
                &ctx, &damage_extras,
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
            &wgpu::CommandEncoderDescriptor { label: Some("firebolt_probe::read_agent_hp") },
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
    /// resolve → idle) is observable from the host.
    pub fn read_busy_until_tick(&self) -> Vec<u32> {
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("firebolt_probe::busy_until_tick_staging"),
            size:               (self.n_agents as u64) * 4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("firebolt_probe::read_busy_until_tick"),
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

    /// **Plan G G4 behavioural pin.** Demonstrates the cast → busy →
    /// resolve → damage loop on real wgpu hardware:
    ///
    ///   - Tick 0: cast begins. agent_hp unchanged at 100.
    ///   - Tick 1, 2: still casting. agent_hp unchanged.
    ///   - Tick 3: busy elapses → ResolveBusy fires → ApplyChronicleDamage
    ///     decrements agent_hp by 25 → hp = 75. Busy state cleared.
    ///   - Tick 4: dispatcher kicks off second cast (busy = 0 again).
    ///   - Tick 7: second cast resolves → hp = 50.
    ///
    /// **Skip path.** When `GpuContext::new_blocking` returns Err
    /// (no compatible wgpu adapter on the host — common on CI without
    /// software-rendering fallback), the test prints a skip message
    /// and returns Ok. The build itself still validated kernel emit +
    /// binding hookup at compile time, so the skip is noisy-but-safe.
    #[test]
    fn firebolt_resolves_after_three_ticks() {
        let n_agents:   u32 = 2;
        let initial_hp: f32 = 100.0;
        let damage:     f32 = 25.0;

        let mut state = match FireboltProbeState::try_new(n_agents, initial_hp) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[firebolt_probe closed-loop] skipping: no wgpu adapter \
                     available on this host. The build itself still validates \
                     the kernel emit (firebolt_probe.sim → 9 WGSL kernels) and \
                     the binding hookup at compile time.",
                );
                return;
            }
        };

        // -- Tick 0: cast begins. The dispatcher writes
        //    EffectCastBeginApplied; RecordCastBegin stamps
        //    busy_until_tick = world.tick + duration = 0 + 3 = 3.
        //    ResolveBusy's predicate is `world.tick >=
        //    busy_until_tick > 0`; at tick=0 with busy_until_tick=3,
        //    the predicate fails and no damage is emitted. hp stays at 100.
        state.step();
        let hp = state.read_agent_hp();
        assert_eq!(hp.len(), n_agents as usize);
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - initial_hp).abs() < 1e-4,
                "tick 0: agent {i} hp = {h}, expected {initial_hp} (still casting)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(b, 3, "tick 0: agent {i} busy_until_tick = {b}, expected 3");
        }

        // -- Tick 1: still casting (busy_until_tick = 3, world.tick = 1).
        //    Dispatcher's `where (busy_until_tick == 0)` gate fails, so
        //    no new cast. ResolveBusy's `world.tick >= busy_until_tick`
        //    fails. hp unchanged.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - initial_hp).abs() < 1e-4,
                "tick 1: agent {i} hp = {h}, expected {initial_hp} (still casting)",
            );
        }

        // -- Tick 2: still casting (world.tick = 2 < busy_until_tick = 3).
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            assert!(
                (h - initial_hp).abs() < 1e-4,
                "tick 2: agent {i} hp = {h}, expected {initial_hp} (still casting)",
            );
        }

        // -- Tick 3: cast resolves (world.tick = 3 == busy_until_tick = 3).
        //    ResolveBusy fires per agent → emits EffectDamageApplied →
        //    ApplyChronicleDamage decrements hp by 25 → hp = 75.
        //    ResolveBusy also clears busy_until_tick to 0.
        //    The dispatcher fires for EVERY alive agent in this same
        //    tick (the dispatcher runs FIRST per the schedule, before
        //    ResolveBusy clears busy state — so on tick 3 the
        //    dispatcher's `busy_until_tick == 0` gate STILL fails since
        //    busy_until_tick is still 3 at dispatcher time. The cast
        //    re-fires only at tick 4.).
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            let expected = initial_hp - damage;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 3: agent {i} hp = {h}, expected {expected} (cast resolved)",
            );
        }
        // Busy cleared to 0 by ResolveBusy.
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(
                b, 0,
                "tick 3: agent {i} busy_until_tick = {b}, expected 0 (busy cleared)",
            );
        }

        // -- Tick 4: dispatcher sees busy == 0, kicks off SECOND cast.
        //    busy_until_tick gets stamped to world.tick + 3 = 4 + 3 = 7.
        //    No resolve (busy_until_tick = 7 > tick = 4). hp stays at 75.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            let expected = initial_hp - damage;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 4: agent {i} hp = {h}, expected {expected} (second cast just started)",
            );
        }
        let busy = state.read_busy_until_tick();
        for (i, &b) in busy.iter().enumerate() {
            assert_eq!(b, 7, "tick 4: agent {i} busy_until_tick = {b}, expected 7");
        }

        // -- Ticks 5, 6: still casting second cast.
        state.step();
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            let expected = initial_hp - damage;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 6: agent {i} hp = {h}, expected {expected} (still casting #2)",
            );
        }

        // -- Tick 7: second cast resolves → hp = 75 - 25 = 50.
        state.step();
        let hp = state.read_agent_hp();
        for (i, &h) in hp.iter().enumerate() {
            let expected = initial_hp - 2.0 * damage;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 7: agent {i} hp = {h}, expected {expected} (second cast resolved)",
            );
        }
    }
}

// Plan E-A5 — pilot integration. Verify the compiler-emitted
// `GeneratedRuntime` (Plan E-A2/A3) compiles + initializes end-to-end
// on the firebolt_probe fixture without disturbing the existing
// hand-written FireboltProbeState. Doesn't replace anything yet —
// just proves the generator output IS buildable + GPU-initializable
// against the same .sim. Failure mode signals what A4 step() emit
// must add to make a full migration possible.
#[cfg(test)]
mod a5_pilot_generator_smoke {
    use crate::{GeneratedRuntime, FIXTURE_NAME, KERNEL_COUNT};

    #[test]
    fn generated_runtime_try_new_initializes_against_firebolt_probe_kernels() {
        let r = match GeneratedRuntime::try_new(0xCAFE, 4) {
            Some(s) => s,
            None => {
                eprintln!("[a5_pilot] skipping: no wgpu adapter on host.");
                return;
            }
        };
        assert_eq!(r.agent_count, 4);
        assert_eq!(r.tick, 0);
        assert_eq!(r.seed, 0xCAFE);
        assert_eq!(FIXTURE_NAME, "firebolt_probe");
        assert!(KERNEL_COUNT > 0);
        eprintln!(
            "[a5_pilot] GeneratedRuntime initialized OK; FIXTURE_NAME={FIXTURE_NAME} KERNEL_COUNT={KERNEL_COUNT}"
        );

        // A4 — step() is currently a no-op SCHEDULE walk. Verify it
        // runs (validates encoder + AgentBuffers + KernelBindingsContext
        // build) without panicking. tick should advance from 0 to 1.
        let mut r = r;
        assert_eq!(r.tick, 0);
        r.step();
        assert_eq!(r.tick, 1, "step() must advance tick");
        r.step();
        assert_eq!(r.tick, 2);
        eprintln!("[a5_pilot] step() ran 2 ticks; tick now = {}", r.tick);
    }
}
