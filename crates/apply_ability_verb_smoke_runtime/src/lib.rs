//! Per-fixture runtime for `assets/sim/apply_ability_verb_smoke.sim` —
//! verb-body counterpart to `apply_ability_smoke_runtime` (which pins
//! the physics-scope `apply_ability` surface). Closes the GPU coverage
//! gap on the verb-body side.
//!
//! ## What this exercises
//!
//! End-to-end pipe of the **verb-body** apply_ability dispatcher
//! (`physics_verb_chronicle_Cast` — synthesised by the verb expander
//! wrapping the verb body into a physics handler whose kernel name is
//! `verb_chronicle_<VerbName>`):
//!
//! 1. Build a single-program AbilityRegistry on the host
//!    (`AbilityProgram::new_single_target` with one `Damage(30.0)`
//!    EffectOp), pack via `PackedAbilityRegistry::pack`, upload via
//!    `PackedAbilityRegistryGpu::upload`.
//! 2. Allocate per-agent SoA buffers (`agent_level` only — the
//!    verb-chronicle kernel doesn't bind `agent_alive`, that gate is
//!    upstream in the `mask_verb_Cast` kernel which we BYPASS), seed
//!    every slot with `level=1` (so each agent dispatches AbilityId(1)).
//! 3. Allocate the event_ring + event_tail buffers and the cfg uniform
//!    that the kernel binds (see the build.rs-emitted bindings struct
//!    for the exact 7-binding shape).
//! 4. **Pre-seed event_ring** with synthetic ActionSelected records
//!    (kind=1, payload[2]=agent_id, payload[3]=0=Cast action). In
//!    production the upstream `scoring` kernel writes these by argmaxing
//!    over the verb's `score 1.0`. We bypass scoring (and its mask_verb
//!    predecessor) so the chronicle kernel can be dispatched standalone
//!    — exactly mirroring what `apply_ability_smoke_runtime` does for
//!    the physics-scope path (which has no scoring/mask preamble).
//! 5. Encode + dispatch one tick of `physics_verb_chronicle_Cast`. The
//!    PerEvent-shape kernel iterates `event_ring[event_idx]` for
//!    `event_count` slots, gates on `kind==1 && payload[3]==0`, then
//!    walks the registry's effect slots for AbilityId(level[agent_id])
//!    and emits one chronicle record per chronicle-bearing EffectOp. With
//!    our single-Damage program, every seeded ActionSelected emits
//!    exactly 1 EffectDamageApplied record (kind=26).
//! 6. Read back event_ring + event_tail and compare the appended damage
//!    records (skipping the 2 seeded ActionSelected entries) byte-for-byte
//!    against the CPU oracle in
//!    `dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record`.
//!
//! ## Why not run the full mask + scoring + chronicle chain?
//!
//! The task brief asked for a standalone dispatch of the chronicle
//! kernel, paralleling the physics-scope runtime which has no upstream
//! preamble. The verb-body path naturally has a `mask → scoring →
//! chronicle` pipeline (verb mask predicate → score argmax → apply),
//! but the chronicle kernel's *contract* is "given event_ring entries
//! of kind=ActionSelected with the verb's action_id, dispatch
//! apply_ability". That contract can be tested directly by hand-writing
//! event_ring — same abstraction boundary the physics-scope test sits
//! at (where the test hand-seeds `agent_level` instead of running an
//! upstream classification kernel that would have written it).
//!
//! ## Important: caster/target slots are 0-based agent_ids, not 1-based AgentIds
//!
//! The dispatcher kernel uses `local_0 = event_ring[event_idx*10+2]`
//! (which we seed = `agent_id`, the 0-based SoA slot) as both
//! `caster_slot` and `target_slot` for the implicit-target rule (the
//! verb fixture uses `by self target self`). So with 2 agents, the
//! emitted records will have caster_id=0,1 and target_id=0,1 (matching
//! the agent SoA slot index, NOT the host AgentId raw value). The CPU
//! oracle takes caster_id + target_id as plain u32s, so the parity
//! comparison passes whichever convention the runtime adopts — we just
//! need to feed the SAME values to both sides.
//!
//! ## GPU adapter availability
//!
//! Construction touches the GPU (`GpuContext::new_blocking`).
//! On a host without a wgpu-compatible adapter the constructor
//! panics; the parity test detects this and skips with an
//! explanatory message rather than failing.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate, PackedAbilityRegistry,
};
use engine::GpuContext;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-record stride in u32 words — matches
/// `dsl_compiler::cpu_chronicle_reference::CHRONICLE_RECORD_STRIDE_U32`
/// and the engine's `EVENT_STRIDE_U32`. Pinned at 10 (header 2 +
/// payload 8).
pub const CHRONICLE_STRIDE_U32: u32 = 10;

/// Default ring slot capacity — one slot per chronicle record. The
/// smoke fixture only needs a few records per tick (n_agents seeded
/// ActionSelected + n_agents emitted Damage = 2*n_agents), so 256 slots
/// is generous (and far below the 65 536 the production helper allocates).
const RING_SLOTS: u32 = 256;

/// EventKindId for `ActionSelected` (the synthetic event the scoring
/// kernel writes in production, and which we hand-seed here). The
/// chronicle kernel gates on `event_ring[event_idx*10+0] == 1u`, so
/// this must match `EventKindId::ActionSelected as u32`.
const EVENT_KIND_ACTION_SELECTED: u32 = 1;

/// Action-id byte the verb expander assigns to the `Cast` verb (it's
/// the first — and only — verb in the `scoring { Cast = 1.0 }` block,
/// so it lands at scoring-row 0; the chronicle kernel gates on
/// `payload[3] == 0u` for the Cast action).
const ACTION_ID_CAST: u32 = 0;

/// Per-fixture state for the verb-body apply_ability dispatcher smoke
/// test. Owns:
///   - The wgpu context.
///   - Per-agent SoA buffers (just `agent_level` — the verb-chronicle
///     kernel doesn't bind `agent_alive`).
///   - Event-ring + tail buffers (atomic u32 storage).
///   - The packed-registry GPU buffers (uploaded once at
///     construction; immutable thereafter).
///   - Per-kernel cfg uniform.
///   - Pipeline cache.
///
/// `n_agents` is captured from the constructor for the dispatch
/// `event_count` (one ActionSelected event per agent gets seeded). The
/// constructor seeds `level[*]=1` so every agent dispatches AbilityId(1).
pub struct ApplyAbilityVerbSmokeState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_level_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU (only the 3 columns the
    // dispatcher binds are read; the upload helper allocates all
    // columns regardless — wasted bytes are bounded by registry size). --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Event ring + tail --
    event_ring_buf: wgpu::Buffer,
    event_tail_buf: wgpu::Buffer,
    event_ring_staging: wgpu::Buffer,
    event_tail_staging: wgpu::Buffer,

    // -- Cfg uniform --
    physics_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    n_agents: u32,
}

impl ApplyAbilityVerbSmokeState {
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
        //    AbilityId(1). Same shape as the physics-scope runtime — the
        //    dispatcher's effect-loop is identical between the two
        //    surfaces; only the gating preamble differs.
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
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "apply_ability_verb_smoke");

        // -- Agent SoA: level=1 for every slot. The kernel reads
        //    `agent_level[target_expr_7]` (with `target_expr_7 = local_0
        //    = event_ring payload[2] = agent_id`) as the AbilityId.
        let level_init: Vec<u32> = vec![1u32; n_agents as usize];
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apply_ability_verb_smoke_runtime::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // -- Event ring + tail. The kernel binds both as
        //    `array<atomic<u32>>` so we tag them STORAGE (the dispatcher
        //    reads via atomicLoad and writes via atomicAdd / atomicStore).
        //    COPY_SRC needed for readback into the staging buffer;
        //    COPY_DST needed to seed the synthetic ActionSelected
        //    records before each dispatch.
        let ring_bytes = (RING_SLOTS as u64) * (CHRONICLE_STRIDE_U32 as u64) * 4;
        let event_ring_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_verb_smoke_runtime::event_ring"),
            size: ring_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_verb_smoke_runtime::event_tail"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffers for host readback.
        let event_ring_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_verb_smoke_runtime::event_ring_staging"),
            size: ring_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apply_ability_verb_smoke_runtime::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // -- Cfg uniform. Compiler-emitted struct shape is
        //    `{ event_count, tick, seed, _pad0 }` (4 × u32 — note that
        //    the verb-chronicle kernel uses `event_count` instead of
        //    the physics-scope kernel's `agent_cap`, since it dispatches
        //    over event_ring slots rather than per agent). We seed it
        //    once at construction; `step()` overwrites tick + event_count
        //    before every dispatch.
        let cfg_init = physics_verb_chronicle_Cast::PhysicsVerbChronicleCastCfg {
            event_count: n_agents,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apply_ability_verb_smoke_runtime::physics_cfg"),
            contents: bytemuck::bytes_of(&cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Some(Self {
            gpu,
            agent_level_buf,
            registry_gpu,
            event_ring_buf,
            event_tail_buf,
            event_ring_staging,
            event_tail_staging,
            physics_cfg_buf,
            cache: dispatch::KernelCache::default(),
            n_agents,
        })
    }

    /// Encode + dispatch one tick of `physics_verb_chronicle_Cast`.
    /// The encoder also:
    ///   - writes `event_tail = n_agents` (so producer atomicAdd starts
    ///     appending damage records AFTER the seeded ActionSelected
    ///     entries, just as the production scoring kernel would leave it),
    ///   - seeds slots [0..n_agents] of `event_ring` with synthetic
    ///     ActionSelected records (kind=1, tick, agent_id, action=Cast=0,
    ///     target=0xFFFFFFFF, then 5 padding zeros to fill the 10-word
    ///     stride).
    pub fn step(&mut self, tick: u32) {
        let cfg = physics_verb_chronicle_Cast::PhysicsVerbChronicleCastCfg {
            event_count: self.n_agents,
            tick,
            seed: 0,
            _pad0: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&cfg));

        // -- Seed event_ring with synthetic ActionSelected records, one
        //    per agent. Layout per record (10 u32 words, matching the
        //    chronicle stride):
        //      [0] kind         = ActionSelected (1)
        //      [1] tick         = current tick
        //      [2] caster/actor = agent_id (0-based slot)
        //      [3] action_id    = Cast (0) — gates the chronicle dispatch
        //      [4] target       = 0xFFFFFFFF (no explicit target;
        //                         the Cast verb is `by self target self`
        //                         so the chronicle reads agent_id from
        //                         payload[2] for both caster + target)
        //      [5..10] padding  = 0
        let mut seeded = vec![0u32; (self.n_agents as usize) * (CHRONICLE_STRIDE_U32 as usize)];
        for agent_id in 0..self.n_agents {
            let base = (agent_id as usize) * (CHRONICLE_STRIDE_U32 as usize);
            seeded[base + 0] = EVENT_KIND_ACTION_SELECTED;
            seeded[base + 1] = tick;
            seeded[base + 2] = agent_id;
            seeded[base + 3] = ACTION_ID_CAST;
            seeded[base + 4] = 0xFFFF_FFFF;
            // [5..10] already zero from vec init.
        }
        self.gpu
            .queue
            .write_buffer(&self.event_ring_buf, 0, bytemuck::cast_slice(&seeded));

        // event_tail = n_agents — chronicle's atomicAdd starts allocating
        // slots from `n_agents` upward, leaving the seeded records intact.
        self.gpu.queue.write_buffer(
            &self.event_tail_buf,
            0,
            bytemuck::bytes_of(&self.n_agents),
        );

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apply_ability_verb_smoke_runtime::step"),
            });

        let bindings = physics_verb_chronicle_Cast::PhysicsVerbChronicleCastBindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            agent_level: &self.agent_level_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            cfg: &self.physics_cfg_buf,
        };
        // The dispatch helper takes `agent_cap` but uses it solely for
        // the workgroup count `(N + 63) / 64`. For the PerEvent shape
        // here we pass `event_count` (= n_agents), so the kernel
        // dispatches one workgroup per seeded ActionSelected event.
        dispatch::dispatch_physics_verb_chronicle_cast(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.n_agents,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
    }

    /// Block on the GPU and read back `event_tail`. Returns the total
    /// count of records in the ring (= n_agents seeded ActionSelected +
    /// the chronicle records the kernel appended on top).
    pub fn read_event_tail(&self) -> u32 {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apply_ability_verb_smoke_runtime::read_event_tail"),
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
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apply_ability_verb_smoke_runtime::read_event_ring"),
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

    /// End-to-end CPU↔GPU parity test for the **verb-body**
    /// apply_ability dispatcher. Mirror of the physics-scope test in
    /// `apply_ability_smoke_runtime`: same one-program registry, same
    /// 2-agent setup, same byte-for-byte chronicle comparison, but
    /// driving the `physics_verb_chronicle_Cast` kernel that the verb
    /// expander synthesises.
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

        let mut state = match ApplyAbilityVerbSmokeState::try_new(n_agents) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[apply_ability_verb_smoke parity] skipping: no wgpu adapter \
                     available on this host. The build itself still validates \
                     the kernel emit (apply_ability_verb_smoke.sim → WGSL) and the \
                     7-binding hookup at compile time."
                );
                return;
            }
        };

        // 1. Run one tick of the dispatcher. The seeded event_ring
        //    contains 2 ActionSelected records (one per agent, action=
        //    Cast=0). With `agent_level[*]=1` + one Damage(30.0) op at
        //    AbilityId(1) + one chronicle-bearing EffectOp per program
        //    slot, every seeded ActionSelected emits EXACTLY one Damage
        //    record (kind=26 EffectDamageApplied).
        state.step(tick);

        // 2. Read back the tail count + records.
        //    Expected: tail = n_agents (seeded ActionSelected) +
        //                   n_agents (emitted Damage records) = 2 * n_agents.
        let tail = state.read_event_tail();
        let expected_tail = 2 * n_agents;
        assert_eq!(
            tail, expected_tail,
            "expected {expected_tail} records (n_agents seeded ActionSelected \
             + n_agents Damage); got tail={tail}"
        );

        // Read back ALL records, then split into seeded prefix +
        // appended damage suffix. The damage suffix is what we compare
        // against the CPU oracle.
        let all_records = state.read_event_ring(tail);
        // Sanity: prefix records are the synthetic ActionSelected entries
        // we wrote (kind=1).
        for (i, r) in all_records.iter().take(n_agents as usize).enumerate() {
            assert_eq!(
                r[0], EVENT_KIND_ACTION_SELECTED,
                "seeded record {i} should be kind=ActionSelected (1); got kind={}",
                r[0]
            );
        }
        let mut gpu_damage_records: Vec<[u32; 10]> =
            all_records.into_iter().skip(n_agents as usize).collect();
        canonicalize(&mut gpu_damage_records);

        // 3. Build the CPU oracle: same program, same caster/target
        //    for each agent. The dispatcher writes `agent_id` (= 0-based
        //    SoA slot) into both caster_slot and target_slot for the
        //    `by self target self` rule, so the oracle uses the slot
        //    index as both (NOT the 1-based AgentId raw value).
        let program = AbilityProgram::new_single_target(
            5.0,
            Gate {
                cooldown_ticks: 10,
                hostile_only: false,
                line_of_sight: false,
            },
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
            // dispatcher writes 0-based via the seeded payload[2]).
            let aid = AgentId::new(slot + 1).expect("slot+1 is non-zero by construction");
            let events = apply_program(
                &program,
                /*caster*/ aid,
                /*target*/ aid,
                tick as u64,
                /*world_seed*/ 0,
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
            gpu_damage_records.len(),
            cpu_records.len(),
            "record count mismatch: gpu={} cpu={}",
            gpu_damage_records.len(),
            cpu_records.len(),
        );
        for (i, (g, c)) in gpu_damage_records.iter().zip(cpu_records.iter()).enumerate() {
            assert_eq!(
                g, c,
                "record {i}: GPU={g:?} CPU={c:?} (kind={:?}, tick={:?}, \
                 caster={:?}, target={:?}, payload[4]={:?})",
                g[0], g[1], g[2], g[3], g[4],
            );
        }
    }
}
