//! Per-fixture runtime for
//! `assets/sim/apply_ability_verb_chronicle_consumer.sim` — STRUCTURAL
//! TEMPLATE for task #138 (Wire one duel_abilities verb to apply_ability).
//!
//! ## What this exercises
//!
//! End-to-end pipe of the **verb-body apply_ability dispatcher** + a
//! **chronicle consumer** through wgpu, in a single fused kernel:
//!
//!   - The verb expander synthesises `physics_verb_chronicle_Cast` from
//!     the `Cast` verb body (`apply_ability ... by self target self`).
//!   - The `ApplyChronicleDamage` PerEvent rule reads
//!     `EffectDamageApplied` records (kind=26) and decrements
//!     `agent_hp[target]` via `agents.set_hp(...)`.
//!
//! Both rules are PerEvent-shape so the scheduler **fuses** them into a
//! single kernel named
//! `physics_ApplyChronicleDamage_and_verb_chronicle_Cast`. The kernel
//! body has TWO arms per event slot: op#1 filters on `kind == 26` and
//! decrements hp; op#2 filters on `kind == 2` (ActionSelected) and runs
//! the apply_ability effect loop, appending fresh damage records.
//!
//! ## Per-tick dispatch sequence
//!
//! Because the dispatcher and the consumer share the same fused kernel
//! over event_ring, we run the kernel **twice** per tick to close the
//! loop:
//!
//!   1. Clear `event_tail` to 0. Seed `event_ring[0..n_agents]` with
//!      synthetic `ActionSelected` records (kind=2 — see EVENT_KIND_*
//!      constant below for why 2 not 1). Set `event_tail = n_agents`.
//!      Dispatch with `cfg.event_count = n_agents`. Op#2 fires per
//!      seeded slot, runs the apply_ability dispatcher, appends one
//!      `EffectDamageApplied` (kind=26) per agent. `event_tail` is now
//!      `2 * n_agents`. Op#1 sees no kind=26 records yet (the dispatcher
//!      writes them mid-iteration; same workgroup invocation can't see
//!      its own appends).
//!   2. Read `event_tail` back. Dispatch with `cfg.event_count = tail`
//!      (= `2 * n_agents`). Now slots `[n_agents..2*n_agents]` carry
//!      kind=26 records — op#1 fires per record and decrements hp. Op#2
//!      fires AGAIN on the original `[0..n_agents]` ActionSelected slots
//!      and appends MORE damage records, but they get garbage-collected
//!      on the next tick's clear.
//!
//! After tick 0 with `n_agents=2`, `Damage(30.0)`, `initial_hp=100.0`:
//!   - hp[*] = 70.0
//!
//! ## EVENT_KIND_ACTION_SELECTED = 2 (NOT 1)
//!
//! The `apply_ability_verb_smoke` fixture uses kind=1 because its .sim
//! source declares only `Tick` (idx 0), making `ActionSelected` land at
//! the next .sim-local index = 1. THIS fixture also declares
//! `EffectDamageApplied` (idx 1) before ActionSelected, bumping
//! ActionSelected to idx 2. The fused kernel's op#2 filter is
//! `if (kind == 2u)` — verifiable at the WGSL level if you re-read
//! `physics_ApplyChronicleDamage_and_verb_chronicle_Cast.wgsl`.
//!
//! `EffectDamageApplied`'s kind, by contrast, is engine-aliased to 26
//! via `dsl_ast::engine_events::engine_event_kind_id_for_name` —
//! independent of the .sim's declaration order. The compile-pin in
//! `crates/dsl_compiler/tests/apply_ability_smoke.rs::
//! apply_ability_verb_chronicle_consumer_compiles_with_tolerated_p6`
//! asserts that pinning.
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
use engine::GpuContext;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-record stride in u32 words — matches
/// `dsl_compiler::cpu_chronicle_reference::CHRONICLE_RECORD_STRIDE_U32`
/// and the engine's `EVENT_STRIDE_U32`. Pinned at 10 (header 2 + payload 8).
pub const CHRONICLE_STRIDE_U32: u32 = 10;

/// Default ring slot capacity — generous for the smoke fixture (we
/// emit n_agents seeded ActionSelected + n_agents damage on dispatch 1
/// + n_agents MORE damage on dispatch 2, then garbage-collect on the
/// next tick).
const RING_SLOTS: u32 = 256;

/// EventKindId for `ActionSelected` in this fixture's .sim-local
/// declaration order. The fixture declares:
///   - `event Tick { }`               → idx 0
///   - `event EffectDamageApplied {}` → idx 1
///   - `ActionSelected`               → idx 2 (implicit / synthesised)
///
/// The fused kernel's op#2 filter is `if (kind == 2u)`. Pre-fix the
/// `apply_ability_verb_smoke` fixture (no `EffectDamageApplied` decl)
/// landed ActionSelected at idx 1; this fixture's value differs because
/// the user added the engine event explicitly to satisfy the consumer
/// rule's pattern match.
const EVENT_KIND_ACTION_SELECTED: u32 = 2;

/// Action-id byte the verb expander assigns to the `Cast` verb (it's
/// the first — and only — verb in the `scoring { Cast = 1.0 }` block,
/// so it lands at scoring-row 0; the dispatcher gates on
/// `payload[3] == 0u` for the Cast action).
const ACTION_ID_CAST: u32 = 0;

/// Per-fixture state for the verb-body apply_ability dispatcher +
/// chronicle consumer closed-loop demo. Owns:
///   - The wgpu context.
///   - Per-agent SoA: `agent_level`, `agent_hp` (no `agent_alive` —
///     the fused kernel binds `agent_level` only on the dispatcher half,
///     and the gating predicate `self.alive` is upstream in
///     `mask_verb_Cast` which we BYPASS).
///   - Event-ring + tail buffers (atomic u32 storage).
///   - The packed-registry GPU buffers.
///   - One cfg uniform for the fused kernel.
///   - Pipeline cache.
///
/// `n_agents` is captured for both the seeded ActionSelected count
/// and the per-tick dispatch sizing.
pub struct ApplyAbilityVerbChronicleConsumerState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_level_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_hp_staging: wgpu::Buffer,
    // Wave 1.5#4 GPU wire-up: per-stat columns for the dispatcher's
    // `scale_bonus` switch. All zero — verb-chronicle-consumer's
    // program (Damage 30) has no scaling slots.
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

    // -- Cfg uniform --
    physics_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
    /// Initial hp for every agent (host-only — used by tests for
    /// expected-state computation).
    pub initial_hp: f32,
}

impl ApplyAbilityVerbChronicleConsumerState {
    /// Construct a closed-loop runtime with `n_agents` slots. Builds
    /// an AbilityRegistry holding ONE program at AbilityId(1) (a
    /// single `Damage(30.0)` EffectOp), and seeds `agent_level[*] = 1`
    /// + `agent_hp[*] = initial_hp`.
    ///
    /// Panics if no wgpu adapter is available — call `try_new` for
    /// the fallible variant.
    pub fn new(n_agents: u32, initial_hp: f32) -> Self {
        Self::try_new(n_agents, initial_hp).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — returns `None` when no compatible wgpu
    /// adapter is available on the host.
    pub fn try_new(n_agents: u32, initial_hp: f32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        // -- Build the registry: one Damage(30.0) ability at AbilityId(1).
        let program = AbilityProgram::new_single_target(
            /*range*/ 5.0,
            Gate {
                cooldown_ticks: 10,
                hostile_only: false,
                line_of_sight: false,
            },
            [EffectOp::Damage { amount: 30.0 }],
        );
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
            "apply_ability_verb_chronicle_consumer",
        );

        // -- Agent SoA: level=1 + hp=initial_hp.
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_level_buf =
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(
                        "apply_ability_verb_chronicle_consumer::agent_level",
                    ),
                    contents: bytemuck::cast_slice(&level_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
        let hp_init: Vec<f32> = vec![initial_hp; n_agents as usize];
        let agent_hp_buf =
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(
                        "apply_ability_verb_chronicle_consumer::agent_hp",
                    ),
                    contents: bytemuck::cast_slice(&hp_init),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });
        let agent_hp_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(
                "apply_ability_verb_chronicle_consumer::agent_hp_staging",
            ),
            size: (n_agents as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Wave 1.5#4 GPU scaling: per-stat columns (zeroed).
        let zeros_f32: Vec<f32> = vec![0.0_f32; n_agents as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("apply_ability_verb_chronicle_consumer::agent_attack_damage");
        let agent_max_hp_buf        = mk_stat("apply_ability_verb_chronicle_consumer::agent_max_hp");
        let agent_armor_buf         = mk_stat("apply_ability_verb_chronicle_consumer::agent_armor");
        let agent_magic_resist_buf  = mk_stat("apply_ability_verb_chronicle_consumer::agent_magic_resist");
        let agent_move_speed_buf    = mk_stat("apply_ability_verb_chronicle_consumer::agent_move_speed");
        let agent_mana_buf          = mk_stat("apply_ability_verb_chronicle_consumer::agent_mana");

        // -- Event ring + tail (atomic-typed u32 storage).
        let ring_bytes = (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let event_ring_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(
                "apply_ability_verb_chronicle_consumer::event_ring",
            ),
            size: ring_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(
                "apply_ability_verb_chronicle_consumer::event_tail",
            ),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_zero = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(
                    "apply_ability_verb_chronicle_consumer::event_tail_zero",
                ),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(
                "apply_ability_verb_chronicle_consumer::event_tail_staging",
            ),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniform for the fused kernel.
        let cfg_init = physics_ApplyChronicleDamage_and_verb_chronicle_Cast::PhysicsApplyChronicleDamageAndVerbChronicleCastCfg {
            event_count: n_agents,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let physics_cfg_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(
                    "apply_ability_verb_chronicle_consumer::physics_cfg",
                ),
                contents: bytemuck::bytes_of(&cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Some(Self {
            gpu,
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
            physics_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
            initial_hp,
        })
    }

    /// Encode + dispatch one tick of the closed-loop pipeline. See
    /// crate-level docs for the full rationale; in short:
    ///
    ///   1. Clear `event_tail = 0`. Seed ring[0..n_agents] with
    ///      ActionSelected records. Set tail = n_agents. Dispatch
    ///      fused kernel with event_count = n_agents → op#2 emits
    ///      damage records to [n_agents..2*n_agents]. Op#1 no-ops
    ///      (no kind==26 records visible in this dispatch).
    ///   2. Read tail back. Dispatch fused kernel again with
    ///      event_count = tail → op#1 decrements hp on the damage
    ///      records. (Op#2 also re-fires on the seeded ActionSelected
    ///      slots; extra records get GC'd at next tick's clear.)
    pub fn step(&mut self, tick: u32) {
        // ---- Stage 1: clear tail, seed ring, dispatch (op#2 emits damage).
        let cfg_stage1 = physics_ApplyChronicleDamage_and_verb_chronicle_Cast::PhysicsApplyChronicleDamageAndVerbChronicleCastCfg {
            event_count: self.n_agents,
            tick,
            seed: 0,
            _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&cfg_stage1),
        );

        // Seed event_ring[0..n_agents] with synthetic ActionSelected
        // records. Layout per record (10 u32 words):
        //   [0] kind         = ActionSelected (=2 for THIS fixture, see
        //                      EVENT_KIND_ACTION_SELECTED docs)
        //   [1] tick         = current tick
        //   [2] caster/actor = agent_id (0-based slot; used by dispatcher
        //                      as caster_slot AND target_slot for
        //                      `by self target self`)
        //   [3] action_id    = Cast (=0) — gates the dispatcher
        //   [4] target       = 0xFFFFFFFF (no explicit target; dispatcher
        //                      uses payload[2] = caster for both)
        //   [5..10] padding  = 0
        let mut seeded =
            vec![0u32; (self.n_agents as usize) * (CHRONICLE_STRIDE_U32 as usize)];
        for agent_id in 0..self.n_agents {
            let base = (agent_id as usize) * (CHRONICLE_STRIDE_U32 as usize);
            seeded[base + 0] = EVENT_KIND_ACTION_SELECTED;
            seeded[base + 1] = tick;
            seeded[base + 2] = agent_id;
            seeded[base + 3] = ACTION_ID_CAST;
            seeded[base + 4] = 0xFFFF_FFFF;
        }
        self.gpu
            .queue
            .write_buffer(&self.event_ring_buf, 0, bytemuck::cast_slice(&seeded));

        let mut encoder1 =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(
                        "apply_ability_verb_chronicle_consumer::step::stage1",
                    ),
                });

        // Clear event_tail to 0 first, then write n_agents below.
        encoder1.copy_buffer_to_buffer(
            &self.event_tail_zero,
            0,
            &self.event_tail_buf,
            0,
            4,
        );
        // event_tail = n_agents — dispatcher's atomicAdd starts allocating
        // slots from n_agents upward, leaving the seeded ActionSelected
        // entries intact. (Submit + write_buffer between submit calls is
        // also valid; we use copy_buffer_to_buffer earlier and then write
        // n_agents directly.)
        self.gpu.queue.submit(Some(encoder1.finish()));
        self.gpu.queue.write_buffer(
            &self.event_tail_buf,
            0,
            bytemuck::bytes_of(&self.n_agents),
        );

        // Now dispatch the fused kernel. event_count = n_agents.
        let mut encoder2 =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(
                        "apply_ability_verb_chronicle_consumer::step::stage1_dispatch",
                    ),
                });
        let bindings = physics_ApplyChronicleDamage_and_verb_chronicle_Cast::PhysicsApplyChronicleDamageAndVerbChronicleCastBindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            agent_hp: &self.agent_hp_buf,
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
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_mana:          &self.agent_mana_buf,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_physics_applychronicledamage_and_verb_chronicle_cast(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder2,
            self.n_agents,
        );
        // Copy tail to staging in the same submit.
        encoder2.copy_buffer_to_buffer(
            &self.event_tail_buf,
            0,
            &self.event_tail_staging,
            0,
            4,
        );
        self.gpu.queue.submit(Some(encoder2.finish()));

        // ---- Stage 2: read tail, dispatch again with event_count=tail.
        let tail = {
            let slice = self.event_tail_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                res.expect("event_tail_staging map_async failed");
            });
            self.gpu
                .device
                .poll(wgpu::PollType::Wait)
                .expect("device poll failed during stage1 event_tail readback");
            let v = {
                let view = slice.get_mapped_range();
                let words: &[u32] = bytemuck::cast_slice(&view);
                words[0]
            };
            self.event_tail_staging.unmap();
            v
        };
        debug_assert_eq!(
            tail,
            2 * self.n_agents,
            "stage1 should leave tail at 2*n_agents (n_agents seeded + \
             n_agents emitted Damage); got {tail}"
        );

        // Bump cfg.event_count = tail; tick stays the same.
        let cfg_stage2 = physics_ApplyChronicleDamage_and_verb_chronicle_Cast::PhysicsApplyChronicleDamageAndVerbChronicleCastCfg {
            event_count: tail,
            tick,
            seed: 0,
            _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&cfg_stage2),
        );

        let mut encoder3 =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(
                        "apply_ability_verb_chronicle_consumer::step::stage2_dispatch",
                    ),
                });
        // Re-bind (Bindings borrows are short-lived; ok to rebuild).
        let bindings2 = physics_ApplyChronicleDamage_and_verb_chronicle_Cast::PhysicsApplyChronicleDamageAndVerbChronicleCastBindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            agent_hp: &self.agent_hp_buf,
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
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_mana:          &self.agent_mana_buf,
            cfg: &self.physics_cfg_buf,
        };
        // Workgroup count covers `tail` invocations (one per slot).
        dispatch::dispatch_physics_applychronicledamage_and_verb_chronicle_cast(
            &mut self.cache,
            &bindings2,
            &self.gpu.device,
            &mut encoder3,
            tail,
        );
        self.gpu.queue.submit(Some(encoder3.finish()));
    }

    /// Block on the GPU and read back the per-agent hp array.
    pub fn read_agent_hp(&self) -> Vec<f32> {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(
                        "apply_ability_verb_chronicle_consumer::read_agent_hp",
                    ),
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

    /// **Closed-loop pin: structural template for task #138.**
    /// Demonstrates the verb-body apply_ability dispatcher + chronicle
    /// consumer chain on real wgpu hardware:
    ///
    ///   - Verb-body dispatcher (op#2 of the fused kernel) writes
    ///     `EffectDamageApplied` records into `event_ring` (kind=26 in
    ///     the engine's hardcoded EventKindId space).
    ///   - Consumer (op#1 of the fused kernel) reads them and decrements
    ///     `agent_hp[target]` by the chronicle's `amount` field.
    ///
    /// With `n_agents=2`, `Damage(30.0)`, `initial_hp=100.0`:
    ///   - After tick 0: hp[*] = 70.0
    ///
    /// **Skip path.** When `GpuContext::new_blocking` returns Err
    /// (no compatible wgpu adapter on the host), the test prints a
    /// skip message and returns Ok. The build itself still validated
    /// kernel emit + binding hookup at compile time, so the skip is
    /// noisy-but-safe.
    #[test]
    fn agent_hp_decrements_30_per_tick_via_verb_body_then_chronicle_consumer() {
        let n_agents: u32 = 2;
        let initial_hp: f32 = 100.0;
        let damage_per_tick: f32 = 30.0;

        let mut state = match ApplyAbilityVerbChronicleConsumerState::try_new(
            n_agents,
            initial_hp,
        ) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[apply_ability_verb_chronicle_consumer closed-loop] \
                     skipping: no wgpu adapter available on this host. The \
                     build itself still validates the kernel emit \
                     (apply_ability_verb_chronicle_consumer.sim → 8 WGSL \
                     kernels) and the binding hookup at compile time."
                );
                return;
            }
        };

        // ONE tick: hp = 100 - 30 = 70 for each agent.
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
    }
}
