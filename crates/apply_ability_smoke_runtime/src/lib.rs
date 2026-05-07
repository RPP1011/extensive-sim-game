//! Per-fixture runtime for `assets/sim/apply_ability_smoke.sim` —
//! task #133 (CPU↔GPU apply_program parity test, P3).
//!
//! ## What this exercises
//!
//! End-to-end pipe of the apply_ability dispatcher kernel:
//!
//! 1. Build a single-program AbilityRegistry on the host
//!    (`AbilityProgram::new_single_target` with one `Damage(30.0)`
//!    EffectOp), pack via `PackedAbilityRegistry::pack`, upload via
//!    `PackedAbilityRegistryGpu::upload`.
//! 2. Allocate per-agent SoA buffers (`agent_alive`, `agent_level`),
//!    seed every slot with `alive=1` and `level=1` (so each agent
//!    dispatches AbilityId(1)).
//! 3. Allocate the event_ring + event_tail buffers and the cfg uniform
//!    that the kernel binds (see the build.rs-emitted bindings
//!    struct for the exact 8-binding shape).
//! 4. Encode + dispatch one tick of `physics_DispatchAbility`. The
//!    kernel's per-agent body reads `agent_level[agent_id]` as the
//!    AbilityId, walks the registry's effect slots, and emits one
//!    chronicle record per chronicle-bearing EffectOp into
//!    `event_ring`. With our single-Damage program, every alive agent
//!    emits exactly 1 record (kind=26 EffectDamageApplied).
//! 5. Read back event_ring + event_tail and compare every record
//!    byte-for-byte against the CPU oracle in
//!    `dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record`.
//!
//! ## Important: caster/target slots are 0-based agent_ids, not 1-based AgentIds
//!
//! The dispatcher kernel uses `gid.x` (the workgroup's per-thread
//! global invocation index, 0-based) as both `caster_slot` and
//! `target_slot` for the implicit-target rule (no `by ... target ...`
//! clause). So with 2 agents, the records will have caster_id=0,1 and
//! target_id=0,1 (matching the agent SoA slot index, NOT the host
//! AgentId raw value). The CPU oracle takes caster_id + target_id as
//! plain u32s, so the parity comparison passes whichever convention
//! the runtime adopts — we just need to feed the SAME values to both
//! sides.
//!
//! ## GPU adapter availability
//!
//! Construction touches the GPU (`GpuContext::new_blocking`).
//! On a host without a wgpu-compatible adapter the constructor
//! panics; the parity test detects this and skips with an
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

/// Default ring slot capacity — one slot per chronicle record. The
/// smoke fixture only needs a few records per tick, so 256 slots is
/// generous (and far below the 65 536 the production helper allocates).
const RING_SLOTS: u32 = 256;

/// Per-fixture state for the apply_ability dispatcher smoke test.
/// Owns:
///   - The wgpu context.
///   - Per-agent SoA buffers (`agent_alive`, `agent_level`).
///   - Event-ring + tail buffers (atomic u32 storage).
///   - The packed-registry GPU buffers (uploaded once at
///     construction; immutable thereafter).
///   - Per-kernel cfg uniform.
///   - Pipeline cache.
///
/// `n_agents` is captured from the constructor for the dispatch
/// `agent_cap`. The constructor seeds `alive[*]=1` and `level[*]=1`
/// so every agent dispatches AbilityId(1).
pub struct ApplyAbilitySmokeState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,
    // Wave 1.5#4 GPU wire-up: per-stat agent SoA columns the dispatcher
    // reads at `caster_slot` for the `scale_bonus = Σ percent * stat`
    // computation. Initialized to zero — the smoke fixture's program
    // (Damage 30 with no scaling slots) writes zero scale_bonus
    // regardless. Real fixtures (Bleed-style +5% MaxHp) populate these.
    agent_attack_damage_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU (only the 3 columns the
    // dispatcher binds are read; the upload helper allocates all
    // columns regardless — wasted bytes are bounded by registry size). --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Event ring + tail --
    event_ring_buf: wgpu::Buffer,
    event_tail_buf: wgpu::Buffer,
    event_ring_staging: wgpu::Buffer,
    event_tail_staging: wgpu::Buffer,
    /// Pre-built zero buffer for per-tick `event_tail = 0` clears.
    event_tail_zero: wgpu::Buffer,

    // -- Cfg uniform --
    physics_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
}

impl ApplyAbilitySmokeState {
    /// Construct a smoke runtime with `n_agents` slots, an
    /// AbilityRegistry holding ONE program at AbilityId(1) (a single
    /// `Damage(30.0)` EffectOp), and `agent_level[*] = 1` so each
    /// alive agent dispatches AbilityId(1) when stepped.
    ///
    /// The constructor blocks on `GpuContext::new_blocking()`, so
    /// callers without a GPU adapter will receive a panic at this
    /// point. Tests that need to skip in that case should use
    /// `try_new` (see below).
    pub fn new(n_agents: u32) -> Self {
        Self::try_new(n_agents).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — returns `None` when no compatible wgpu
    /// adapter is available on the host. Lets the parity test in this
    /// crate degrade to a skip-with-message instead of a panic.
    pub fn try_new(n_agents: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        // -- Build the registry: one Damage(30.0) ability registered at
        //    AbilityId(1). This matches the canonical CPU oracle test
        //    in `cpu_chronicle_pipeline::single_damage_ability_produces_one_chronicle_record`.
        let program = AbilityProgram::new_single_target(
            /*range*/ 5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
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
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "apply_ability_smoke");

        // -- Agent SoA: alive=1, level=1 for every slot. The kernel's
        //    per-agent body reads `agent_alive[agent_id]` as the
        //    where-clause gate and `agent_level[agent_id]` as the
        //    AbilityId.
        let alive_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_level_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::agent_level"),
                contents: bytemuck::cast_slice(&level_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Wave 1.5#4 GPU scaling: per-stat columns for the dispatcher's
        // `agent_stat()` switch. All zero — the smoke program has no
        // scaling slots so `scale_bonus = 0.0` regardless. The
        // `bytemuck::cast_slice(&vec![0.0_f32; ...])` shape mirrors
        // `agent_alive`/`agent_level`.
        let zeros_f32: Vec<f32> = vec![0.0_f32; n_agents as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("apply_ability_smoke_runtime::agent_attack_damage");
        let agent_max_hp_buf        = mk_stat("apply_ability_smoke_runtime::agent_max_hp");
        let agent_hp_buf            = mk_stat("apply_ability_smoke_runtime::agent_hp");
        let agent_armor_buf         = mk_stat("apply_ability_smoke_runtime::agent_armor");
        let agent_magic_resist_buf  = mk_stat("apply_ability_smoke_runtime::agent_magic_resist");
        let agent_move_speed_buf    = mk_stat("apply_ability_smoke_runtime::agent_move_speed");
        let agent_mana_buf          = mk_stat("apply_ability_smoke_runtime::agent_mana");

        // -- Event ring + tail. The kernel binds both as
        //    `array<atomic<u32>>` so we tag them STORAGE (the dispatcher
        //    writes via atomicAdd / atomicStore). COPY_SRC needed for
        //    readback into the staging buffer.
        let ring_bytes = (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let event_ring_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_smoke_runtime::event_ring"),
            size: ring_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_smoke_runtime::event_tail"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::event_tail_zero"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Staging buffers for host readback.
        let event_ring_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_smoke_runtime::event_ring_staging"),
            size: ring_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_smoke_runtime::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniform. Compiler-emitted struct shape is
        //    `{ agent_cap, tick, seed, _pad }` (4 × u32). We seed it
        //    once at construction; `step()` overwrites tick before
        //    every dispatch.
        let cfg_init = physics_DispatchAbility::PhysicsDispatchAbilityCfg {
            agent_cap: n_agents,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("apply_ability_smoke_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_level_buf,
            agent_attack_damage_buf,
            agent_max_hp_buf,
            agent_hp_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            agent_mana_buf,
            registry_gpu,
            event_ring_buf,
            event_tail_buf,
            event_ring_staging,
            event_tail_staging,
            event_tail_zero,
            physics_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
        })
    }

    /// Encode + dispatch one tick of `physics_DispatchAbility`.
    /// The encoder also clears `event_tail` to zero before the
    /// dispatch (so producers atomicAdd from 0).
    pub fn step(&mut self, tick: u32) {
        let cfg = physics_DispatchAbility::PhysicsDispatchAbilityCfg {
            agent_cap: self.n_agents,
            tick,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::step"),
                });

        // Clear event_tail to 0 so producer atomicAdd starts from slot 0.
        encoder.copy_buffer_to_buffer(&self.event_tail_zero, 0, &self.event_tail_buf, 0, 4);

        let bindings = physics_DispatchAbility::PhysicsDispatchAbilityBindings {
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
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.n_agents,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
    }

    /// Block on the GPU and read back `event_tail`. Returns the
    /// number of chronicle records written by the most-recent
    /// `step()`.
    pub fn read_event_tail(&self) -> u32 {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::read_event_tail"),
                });
        encoder.copy_buffer_to_buffer(&self.event_tail_buf, 0, &self.event_tail_staging, 0, 4);
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = self.event_tail_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("event_tail_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during event_tail readback");

        let value = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            words[0]
        };
        self.event_tail_staging.unmap();
        value
    }

    /// Block on the GPU and read back the first `n_records` records
    /// from `event_ring`. Each record is 10 u32 words.
    pub fn read_event_ring(&self, n_records: u32) -> Vec<[u32; 10]> {
        // Always copy back the full ring buffer; the host filters by
        // n_records below. (Partial mapping in wgpu requires aligned
        // offsets — easier to just blit the full ring for the smoke
        // test.)
        let total_bytes = (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply_ability_smoke_runtime::read_event_ring"),
                });
        encoder.copy_buffer_to_buffer(
            &self.event_ring_buf,
            0,
            &self.event_ring_staging,
            0,
            total_bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = self.event_ring_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("event_ring_staging map_async failed");
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during event_ring readback");

        let records = {
            let view = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            // Pull the first `n_records` records out of the flat slice.
            let mut out = Vec::with_capacity(n_records as usize);
            for r in 0..(n_records as usize) {
                let base = r * (CHRONICLE_STRIDE_U32 as usize);
                let mut rec = [0u32; 10];
                rec.copy_from_slice(&words[base..base + 10]);
                out.push(rec);
            }
            out
        };
        self.event_ring_staging.unmap();
        records
    }

    pub fn n_agents(&self) -> u32 {
        self.n_agents
    }
}

/// Sort chronicle records by `(target_slot, kind)` so the host-side
/// comparison is order-stable. The dispatcher uses `atomicAdd` to
/// claim slots, so the records' relative order in the ring depends
/// on workgroup scheduling — comparing as a sorted set sidesteps
/// that nondeterminism for the smoke parity check.
#[cfg(test)]
fn canonicalize(records: &mut [[u32; 10]]) {
    records.sort_by_key(|r| (r[3], r[0]));
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record;
    use engine::ability::apply::apply_program;
    use engine::ability::program::{AbilityProgram, CasterStats, EffectOp, Gate};
    use engine::ids::AgentId;

    /// End-to-end CPU↔GPU parity test (#133). Closes the loop the
    /// `cpu_chronicle_pipeline` test set leaves open: the GPU side
    /// finally has a runtime crate driving the dispatcher kernel,
    /// so the comparison can run.
    ///
    /// **Skip path.** When `GpuContext::new_blocking` returns Err
    /// (no compatible wgpu adapter on the host — common on CI without
    /// a software-rendering fallback), the test prints a skip message
    /// and returns Ok. The build itself still validated the kernel
    /// emit + the binding hookup at compile time, so the skip is
    /// noisy-but-safe.
    #[test]
    fn gpu_chronicle_records_match_cpu_oracle_for_2_agents_1_damage_effect() {
        let n_agents: u32 = 2;
        let tick: u32 = 0;

        let mut state = match ApplyAbilitySmokeState::try_new(n_agents) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[apply_ability_smoke parity] skipping: no wgpu adapter \
                     available on this host. The build itself still validates \
                     the kernel emit (apply_ability_smoke.sim → WGSL) and the \
                     8-binding hookup at compile time."
                );
                return;
            }
        };

        // 1. Run one tick of the dispatcher. With agent_level[*]=1 +
        //    one Damage(30.0) op at AbilityId(1) + one chronicle-bearing
        //    EffectOp per program slot, every alive agent emits
        //    EXACTLY one record (kind=26 EffectDamageApplied).
        state.step(tick);

        // 2. Read back the tail count + records.
        let tail = state.read_event_tail();
        assert_eq!(
            tail, n_agents,
            "expected one chronicle record per alive agent (n={n_agents}); \
             got tail={tail}"
        );
        let mut gpu_records = state.read_event_ring(tail);
        canonicalize(&mut gpu_records);

        // 3. Build the CPU oracle: same program, same caster/target
        //    for each agent. The dispatcher writes `agent_id` (= 0-based
        //    SoA slot) into both caster_slot and target_slot for the
        //    implicit-target rule, so the oracle uses the slot index
        //    as both (NOT the 1-based AgentId raw value).
        let program = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let mut cpu_records: Vec<[u32; 10]> = Vec::with_capacity(n_agents as usize);
        for slot in 0..n_agents {
            // apply_program takes 1-based AgentIds — `AgentId::new` rejects
            // zero — but the chronicle record we're comparing against uses
            // the 0-based slot the GPU writes. apply_program emits one
            // ApplyEvent per chronicle-bearing EffectOp; we feed it (1+slot)
            // for the AgentId nichefulness, then pass `slot` as caster_id
            // and target_id into the chronicle reference (which the GPU
            // dispatcher writes 0-based via gid.x).
            let aid = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            let events = apply_program(
                &program,
                /*caster*/ aid,
                /*target*/ aid,
                tick as u64,
                /*world_seed*/ 0,
                &CasterStats::default(),
                &CasterStats::default(),
            );
            for ev in events {
                if let Some(rec) = apply_event_to_chronicle_record(
                    ev,
                    tick,
                    /*caster_id*/ slot,
                    /*target_id*/ slot,
                ) {
                    cpu_records.push(rec);
                }
            }
        }
        canonicalize(&mut cpu_records);

        // 4. Byte-for-byte parity assert.
        assert_eq!(
            gpu_records.len(),
            cpu_records.len(),
            "record count mismatch: gpu={} cpu={}",
            gpu_records.len(),
            cpu_records.len(),
        );
        for (i, (g, c)) in gpu_records.iter().zip(cpu_records.iter()).enumerate() {
            assert_eq!(
                g, c,
                "record {i}: GPU={g:?} CPU={c:?} (kind={:?}, tick={:?}, \
                 caster={:?}, target={:?}, payload[4]={:?})",
                g[0], g[1], g[2], g[3], g[4],
            );
        }
    }
}
