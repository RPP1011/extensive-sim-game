//! Per-fixture runtime for `assets/sim/apply_ability_chronicle_consumer.sim`
//! — closed-loop demo for task #138.
//!
//! ## What this exercises
//!
//! End-to-end pipe of BOTH halves of the chronicle pipeline through wgpu:
//!
//! 1. `physics_DispatchAbility` — apply_ability dispatcher writes
//!    `EffectDamageApplied` chronicle records into `event_ring` (same
//!    kernel as `apply_ability_smoke_runtime`).
//! 2. `seed_indirect_0` — plumbing kernel that reads `event_tail` and
//!    fills `indirect_args_0` with `(wg, 1, 1)` for the consumer's
//!    indirect dispatch path. We don't actually use indirect dispatch
//!    in the consumer's `record()` (it dispatches directly with
//!    `(agent_cap+63)/64` workgroups), but we run seed_indirect_0
//!    anyway because the schedule lists it and a future runtime may
//!    swap to the indirect path.
//! 3. `physics_ApplyChronicleDamage` — PerEvent consumer reads the
//!    records and decrements `agent_hp[target]` via `agents.set_hp`.
//!
//! ## Closed-loop kind tag (engine-aliased; no post-processing)
//!
//! The dispatcher writes `EffectDamageApplied` records with the engine's
//! hardcoded `EventKindId::EffectDamageApplied = 26` in the header word
//! (see `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`'s
//! `event_kind_id_for_effect_kind`).
//!
//! The compiler aliases known engine event names to their hardcoded
//! discriminants — see
//! `dsl_ast::engine_events::engine_event_kind_id_for_name`. The
//! resolver populates `EventIR::engine_kind_id` and both
//! `populate_event_kinds` (driver) and `resolve_event_ref` (driver)
//! mirror that assignment, so the consumer's PerEvent kernel emits
//! `if (kind == 26u)` directly — matching the dispatcher's hardcoded
//! write tag. Pre-fix this crate's `build.rs` had to sed-rewrite
//! `== 1u` to `== 26u` to close the loop; that workaround is gone.
//!
//! `agent_hp[i]` decrements by the chronicle's amount per tick,
//! demonstrating the closed loop on real GPU hardware.
//!
//! ## GPU adapter availability
//!
//! Construction touches the GPU (`GpuContext::new_blocking`). On hosts
//! without a wgpu-compatible adapter the constructor returns `None`;
//! the closed-loop test in this crate detects this and skips with an
//! explanatory message rather than failing.

use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate, PackedAbilityRegistry,
};
use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::GpuContext;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-record stride in u32 words — matches
/// `dsl_compiler::cpu_chronicle_reference::CHRONICLE_RECORD_STRIDE_U32`
/// and the engine's `EVENT_STRIDE_U32`. Pinned at 10 (header 2 +
/// payload 8).
pub const CHRONICLE_STRIDE_U32: u32 = 10;

/// Default ring slot capacity. We only need a few records per tick for
/// the smoke loop (one per agent), so 256 slots is plenty.
const RING_SLOTS: u32 = 256;

/// Per-fixture state for the chronicle-consumer closed-loop demo.
/// Owns:
///   - The wgpu context.
///   - Per-agent SoA: `agent_alive`, `agent_level`, `agent_hp`.
///   - Event-ring + tail buffers (atomic u32 storage).
///   - Indirect-args buffer for the seed_indirect_0 plumbing kernel.
///   - The packed-registry GPU buffers (uploaded once at construction;
///     immutable thereafter).
///   - Per-kernel cfg uniforms (DispatchAbility, ApplyChronicleDamage,
///     SeedIndirect0).
///   - Pipeline cache.
///
/// `n_agents` is captured from the constructor for the dispatch
/// `agent_cap`. The constructor seeds `alive[*]=1`, `level[*]=1`,
/// `hp[*]=100.0` so every agent dispatches AbilityId(1) + has hp to
/// drain.
pub struct ApplyAbilityChronicleConsumerState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_hp_staging: wgpu::Buffer,
    // Wave 1.5#4 GPU wire-up: per-stat agent SoA columns the dispatcher
    // reads at `caster_slot` for `scale_bonus` computation. The closed-
    // loop fixture's program is `Damage(30.0)` (no scaling slots), so
    // these stay all-zero and `scale_bonus = 0.0` regardless. Wired so
    // the binding generator's struct field-set matches the dispatcher's
    // recorded reads. (`agent_hp` above is shared with the chronicle
    // consumer's `agents.set_hp` write — the dispatcher reads it via
    // `caster_slot`, the consumer writes it via `target_id`; same
    // binding, distinct access patterns.)
    agent_attack_damage_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Event ring + tail --
    event_ring_buf: wgpu::Buffer,
    event_tail_buf: wgpu::Buffer,
    event_tail_staging: wgpu::Buffer,
    /// Pre-built zero buffer for per-tick `event_tail = 0` clears.
    event_tail_zero: wgpu::Buffer,

    // -- Indirect args (for seed_indirect_0) --
    indirect_args_buf: wgpu::Buffer,

    // -- Cfg uniforms --
    physics_cfg_buf: wgpu::Buffer,
    consumer_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
    /// Initial hp for every agent (host-only — used by tests for
    /// expected-state computation).
    pub initial_hp: f32,
}

impl ApplyAbilityChronicleConsumerState {
    /// Construct a closed-loop runtime with `n_agents` slots. Builds
    /// an AbilityRegistry holding ONE program at AbilityId(1) (a
    /// single `Damage(30.0)` EffectOp), and seeds `agent_level[*] = 1`
    /// + `agent_alive[*] = 1` + `agent_hp[*] = initial_hp`.
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
        // -- Build the registry: one Damage(30.0) ability at AbilityId(1).
        let program = AbilityProgram::new_single_target(
            /*range*/ 5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        Self::try_new_with_program(n_agents, initial_hp, program, None)
    }

    /// Fallible constructor that takes a custom `AbilityProgram` and
    /// optional initial-hp override per agent. Used by the
    /// when-predicate behavioral pin to register a Damage(50) ability
    /// gated on `when target.hp < 20`, then seed agents with mixed
    /// hp values to verify the predicate gates the chronicle write
    /// per-agent.
    pub fn try_new_with_program(
        n_agents:        u32,
        initial_hp:      f32,
        program:         AbilityProgram,
        per_agent_hp:    Option<&[f32]>,
    ) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;
        let mut builder = AbilityRegistryBuilder::new();
        let id = builder.register(program);
        debug_assert_eq!(
            id,
            AbilityId::new(1).unwrap(),
            "first registered program must land at AbilityId(1)"
        );
        let registry = builder.build();
        let packed = PackedAbilityRegistry::pack(&registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed,
            &gpu,
            "apply_ability_chronicle_consumer",
        );

        // -- Agent SoA: alive=1, level=1, hp=initial_hp.
        let alive_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_level_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::agent_level"),
                contents: bytemuck::cast_slice(&level_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Per-agent HP override lets the predicate behavioral pin
        // seed agents with mixed values (e.g. [10.0, 50.0]) so the
        // dispatcher's `when target.hp < 20` predicate fires
        // selectively. Default = uniform `initial_hp` for the
        // baseline closed-loop test.
        let hp_init: Vec<f32> = match per_agent_hp {
            Some(slice) => {
                assert_eq!(
                    slice.len(),
                    n_agents as usize,
                    "per_agent_hp length mismatch with n_agents",
                );
                slice.to_vec()
            }
            None => vec![initial_hp; n_agents as usize],
        };
        let agent_hp_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::agent_hp"),
                contents: bytemuck::cast_slice(&hp_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let agent_hp_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_chronicle_consumer::agent_hp_staging"),
            size: (n_agents as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Wave 1.5#4 GPU scaling: per-stat columns for the dispatcher's
        // `agent_stat()` switch. All zero — closed-loop program has no
        // scaling slots so `scale_bonus = 0.0`.
        let zeros_f32: Vec<f32> = vec![0.0_f32; n_agents as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("apply_ability_chronicle_consumer::agent_attack_damage");
        let agent_max_hp_buf        = mk_stat("apply_ability_chronicle_consumer::agent_max_hp");
        let agent_armor_buf         = mk_stat("apply_ability_chronicle_consumer::agent_armor");
        let agent_magic_resist_buf  = mk_stat("apply_ability_chronicle_consumer::agent_magic_resist");
        let agent_move_speed_buf    = mk_stat("apply_ability_chronicle_consumer::agent_move_speed");
        let agent_mana_buf          = mk_stat("apply_ability_chronicle_consumer::agent_mana");

        // -- Event ring + tail. Both atomic-typed for the producer's
        //    atomicAdd / atomicStore.
        let ring_bytes = (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let event_ring_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_chronicle_consumer::event_ring"),
            size: ring_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_chronicle_consumer::event_tail"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::event_tail_zero"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_chronicle_consumer::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Indirect args buffer (3 u32s: x,y,z workgroup counts).
        //    seed_indirect_0 writes into it; we never read it on the
        //    host (the consumer dispatches directly anyway).
        let indirect_args_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_chronicle_consumer::indirect_args_0"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // -- Cfg uniforms. Three different cfg structs across the
        //    three kernels we drive each tick. Values are seeded once
        //    at construction; `step()` overwrites tick + event_count
        //    before each dispatch.
        let physics_cfg_init = physics_DispatchAbility::PhysicsDispatchAbilityCfg {
            agent_cap: n_agents,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let consumer_cfg_init =
            physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                _pad0: 0,
            };
        let consumer_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::consumer_cfg"),
                contents: bytemuck::bytes_of(&consumer_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: n_agents,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_chronicle_consumer::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_level_buf,
            agent_hp_buf,
            agent_hp_staging,
            agent_attack_damage_buf,
            agent_max_hp_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            agent_mana_buf,
            registry_gpu,
            event_ring_buf,
            event_tail_buf,
            event_tail_staging,
            event_tail_zero,
            indirect_args_buf,
            physics_cfg_buf,
            consumer_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
            initial_hp,
        })
    }

    /// Encode + dispatch one tick of the closed-loop pipeline:
    ///
    ///   1. Clear `event_tail` to 0.
    ///   2. Dispatch `physics_DispatchAbility` — writes
    ///      `EffectDamageApplied` records into `event_ring` (one per
    ///      alive agent).
    ///   3. Dispatch `seed_indirect_0` — fills `indirect_args_0` with
    ///      the workgroup count derived from `event_tail`.
    ///   4. Read tail back to host so we can populate the consumer's
    ///      `event_count` cfg field. (We use direct dispatch in the
    ///      consumer rather than indirect, but both need to know the
    ///      record count — direct via `cfg.event_count`, indirect
    ///      via `indirect_args_0`.)
    ///   5. Dispatch `physics_ApplyChronicleDamage` — reads each
    ///      record, decrements `agent_hp[target_slot]` by the
    ///      chronicle's `amount` field.
    pub fn step(&mut self, tick: u32) {
        // -- Stage 1: clear tail + run dispatcher + seed_indirect.
        let physics_cfg = physics_DispatchAbility::PhysicsDispatchAbilityCfg {
            agent_cap: self.n_agents,
            tick,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.n_agents,
            tick,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_chronicle_consumer::step::dispatch"),
                });

        // 1. Clear event_tail to 0 so the dispatcher's atomicAdd starts at 0.
        encoder.copy_buffer_to_buffer(&self.event_tail_zero, 0, &self.event_tail_buf, 0, 4);

        // 2. Dispatch the apply_ability dispatcher.
        let dispatch_bindings =
            physics_DispatchAbility::PhysicsDispatchAbilityBindings {
                event_ring: &self.event_ring_buf,
                event_tail: &self.event_tail_buf,
                agent_alive: &self.agent_alive_buf,
                agent_level: &self.agent_level_buf,
                ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
                ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
                ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
                ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
                ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
                ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
                ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
                ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
                ability_registry_when_pred_binder:  &self.registry_gpu.when_pred_binder,
                ability_registry_when_pred_field:   &self.registry_gpu.when_pred_field,
                ability_registry_when_pred_op:      &self.registry_gpu.when_pred_op,
                ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
                agent_attack_damage: &self.agent_attack_damage_buf,
                agent_max_hp:        &self.agent_max_hp_buf,
                agent_hp:            &self.agent_hp_buf,
                agent_armor:         &self.agent_armor_buf,
                agent_magic_resist:  &self.agent_magic_resist_buf,
                agent_move_speed:    &self.agent_move_speed_buf,
                agent_mana:          &self.agent_mana_buf,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_physics_dispatchability(
            &mut self.cache,
            &dispatch_bindings,
            &self.gpu.device,
            &mut encoder,
            self.n_agents,
        );

        // 3. Dispatch the seed_indirect_0 plumbing kernel (fills
        //    indirect_args_0 — unused in our direct-dispatch path,
        //    but the schedule lists it and a future runtime may use it).
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            indirect_args_0: &self.indirect_args_buf,
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.n_agents,
        );

        // 4. Copy tail to staging so we can read it back below.
        encoder.copy_buffer_to_buffer(&self.event_tail_buf, 0, &self.event_tail_staging, 0, 4);

        self.gpu.queue.submit(Some(encoder.finish()));

        // -- Stage 2: read tail to populate consumer's event_count cfg.
        let tail = {
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
        };

        // -- Stage 3: dispatch the consumer kernel with event_count = tail.
        let consumer_cfg =
            physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageCfg {
                event_count: tail,
                tick,
                seed: 0,
                _pad0: 0,
            };
        self.gpu
            .queue
            .write_buffer(&self.consumer_cfg_buf, 0, bytemuck::bytes_of(&consumer_cfg));

        let mut encoder2 =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_chronicle_consumer::step::consume"),
                });
        let consumer_bindings =
            physics_ApplyChronicleDamage::PhysicsApplyChronicleDamageBindings {
                event_ring: &self.event_ring_buf,
                event_tail: &self.event_tail_buf,
                agent_hp: &self.agent_hp_buf,
                cfg: &self.consumer_cfg_buf,
            };
        // Dispatch enough workgroups to cover `tail` records. The
        // kernel's record() helper uses `(agent_cap+63)/64` workgroups,
        // so passing `n_agents` works as long as event_count <= n_agents
        // (one record per agent in our self-cast configuration). For a
        // future case with more events than agents, we'd need to pass
        // `tail` instead.
        dispatch::dispatch_physics_applychronicledamage(
            &mut self.cache,
            &consumer_bindings,
            &self.gpu.device,
            &mut encoder2,
            self.n_agents.max(tail),
        );
        self.gpu.queue.submit(Some(encoder2.finish()));
    }

    /// Block on the GPU and read back the per-agent hp array.
    pub fn read_agent_hp(&self) -> Vec<f32> {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_chronicle_consumer::read_agent_hp"),
                });
        encoder.copy_buffer_to_buffer(
            &self.agent_hp_buf,
            0,
            &self.agent_hp_staging,
            0,
            (self.n_agents as u64) * 4,
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

    pub fn n_agents(&self) -> u32 {
        self.n_agents
    }
}

#[cfg(test)]
mod closed_loop_tests {
    use super::*;
    use engine::ability::program::{
        EffectPredicate, EffectPredicateBinder, EffectPredicateOp, EffectWhenCondition,
        ScalingStatRef,
    };

    /// **Closed-loop pin for task #138.** Demonstrates the full
    /// chronicle pipeline on real wgpu hardware:
    ///
    ///   - Dispatcher writes `EffectDamageApplied` records into
    ///     `event_ring` (one per alive agent, kind=26 in the engine's
    ///     hardcoded EventKindId space).
    ///   - Consumer reads them and decrements `agent_hp[target]` by
    ///     the chronicle's `amount` field.
    ///
    /// With `n_agents=2`, `Damage(30.0)`, `initial_hp=100.0`:
    ///   - After tick 0: hp[*] = 70.0
    ///   - After tick 1: hp[*] = 40.0
    ///   - After tick 2: hp[*] = 10.0
    ///
    /// **Skip path.** When `GpuContext::new_blocking` returns Err
    /// (no compatible wgpu adapter on the host — common on CI
    /// without software-rendering fallback), the test prints a skip
    /// message and returns Ok. The build itself still validated
    /// kernel emit + binding hookup at compile time, so the skip is
    /// noisy-but-safe.
    #[test]
    fn agent_hp_decrements_per_tick_via_chronicle_consumer() {
        let n_agents: u32 = 2;
        let initial_hp: f32 = 100.0;
        let damage_per_tick: f32 = 30.0;

        let mut state = match ApplyAbilityChronicleConsumerState::try_new(
            n_agents,
            initial_hp,
        ) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[apply_ability_chronicle_consumer closed-loop] skipping: \
                     no wgpu adapter available on this host. The build itself \
                     still validates the kernel emit (apply_ability_chronicle_consumer.sim \
                     → 7 WGSL kernels) and the binding hookup at compile time."
                );
                return;
            }
        };

        // Tick 0: hp = 100 - 30 = 70 for each agent.
        state.step(0);
        let hp0 = state.read_agent_hp();
        assert_eq!(
            hp0.len(),
            n_agents as usize,
            "agent_hp readback length mismatch"
        );
        for (i, &h) in hp0.iter().enumerate() {
            let expected = initial_hp - damage_per_tick;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 0: agent {i} hp = {h}, expected {expected}"
            );
        }

        // Tick 1: hp = 70 - 30 = 40.
        state.step(1);
        let hp1 = state.read_agent_hp();
        for (i, &h) in hp1.iter().enumerate() {
            let expected = initial_hp - 2.0 * damage_per_tick;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 1: agent {i} hp = {h}, expected {expected}"
            );
        }

        // Tick 2: hp = 40 - 30 = 10.
        state.step(2);
        let hp2 = state.read_agent_hp();
        for (i, &h) in hp2.iter().enumerate() {
            let expected = initial_hp - 3.0 * damage_per_tick;
            assert!(
                (h - expected).abs() < 1e-4,
                "tick 2: agent {i} hp = {h}, expected {expected}"
            );
        }
    }

    /// **Wave 1.5#7 GPU eval — behavioral pin.** Builds a registry with
    /// one Damage(50) ability gated on `when target.hp < 20`. Pre-seeds
    /// 2 agents at hp=[10, 50]. The dispatcher targets each agent with
    /// itself (single-agent self-cast in this fixture). Asserts:
    ///
    ///   - agent 0 (hp=10): predicate passes → chronicle write fires
    ///     → consumer applies Damage(50) → hp = 10 - 50 = -40.
    ///   - agent 1 (hp=50): predicate fails → no chronicle write →
    ///     hp stays at 50 (untouched).
    ///
    /// Without the GPU when-predicate gate (Wave 1.5#7 GPU eval), BOTH
    /// agents would receive the chronicle and lose 50 hp — so the
    /// expected hp[1] = 50.0 is the load-bearing assertion.
    #[test]
    fn agent_hp_decrements_only_when_predicate_passes() {
        let n_agents: u32 = 2;
        let initial_per_agent_hp = [10.0_f32, 50.0_f32];

        // Build Damage(50) ability with when target.hp < 20 predicate.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        prog.when_per_effect.push(Some(EffectWhenCondition {
            when_cond:     "target.hp < 20".to_string(),
            else_cond:     None,
            when_compiled: Some(EffectPredicate {
                binder:  EffectPredicateBinder::Target,
                field:   ScalingStatRef::Hp.discriminant(),
                op:      EffectPredicateOp::Lt,
                literal: 20.0,
            }),
        }));

        let mut state = match ApplyAbilityChronicleConsumerState::try_new_with_program(
            n_agents,
            /*initial_hp (unused — overridden below)*/ 0.0,
            prog,
            Some(&initial_per_agent_hp),
        ) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[apply_ability_chronicle_consumer when-predicate] skipping: \
                     no wgpu adapter available on this host."
                );
                return;
            }
        };

        // Tick 0: dispatcher runs once per agent (self-cast). Predicate
        // gates chronicle write per-agent.
        state.step(0);
        let hp = state.read_agent_hp();
        assert_eq!(hp.len(), n_agents as usize);
        // Agent 0: hp=10 < 20 → predicate passes → Damage(50) → hp = -40.
        assert!(
            (hp[0] - (10.0 - 50.0)).abs() < 1e-4,
            "agent 0 (initial hp=10) must receive Damage(50) since \
             predicate `target.hp < 20` passes; saw hp[0]={}",
            hp[0],
        );
        // Agent 1: hp=50 NOT < 20 → predicate fails → NO chronicle →
        // hp stays at 50.0. THIS IS THE LOAD-BEARING ASSERTION:
        // without the GPU predicate gate, hp[1] would also drop to 0.
        assert!(
            (hp[1] - 50.0).abs() < 1e-4,
            "agent 1 (initial hp=50) must NOT receive Damage because \
             predicate `target.hp < 20` fails; saw hp[1]={} (50.0 expected; \
             a value of 0.0 means the GPU predicate gate did NOT fire)",
            hp[1],
        );
    }
}
