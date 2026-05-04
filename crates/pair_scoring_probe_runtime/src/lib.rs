//! Per-fixture runtime for `assets/sim/pair_scoring_probe.sim` —
//! discovery probe of PAIR-FIELD SCORING (spec §8.3).
//!
//! ## Outcome (b) NO FIRE — gap chain confirmed at compile time
//!
//! The fixture's verb `Heal(self, target: Agent)` triggers a synthesised
//! mask whose head shape is `IrActionHeadShape::Positional([("target",
//! _, AgentId)])`. `lower_mask` rejects positional heads without a
//! `from` clause (`LoweringError::UnsupportedMaskHeadShape`, Task 2.6
//! still open). The verb syntax has no `from` clause to thread, so the
//! gap is structural — see Gap #1 in the discovery doc.
//!
//! Consequence: the partial CG program excludes the mask + scoring
//! kernels. The PartialEmit set this runtime drives is:
//!
//!   - `physics_verb_chronicle_Heal` (would consume `ActionSelected`)
//!   - `fold_received` (would consume `Healed` events from the
//!     chronicle's emit body)
//!   - `seed_indirect_0`, `pack_agents`, `unpack_agents`,
//!     `upload_sim_cfg`, `kick_snapshot` (plumbing)
//!
//! With the scoring kernel missing, NO `ActionSelected` events ever
//! land in the ring → the chronicle's `if (action_id == 0u)` gate is
//! never reached → no `Healed` events ever land → `received[N]` stays
//! at 0.0 every slot. That's the OUTCOME (b) NO-FIRE observable the
//! sim_app reports.
//!
//! See `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md` for
//! the full gap chain.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the pair-scoring probe. Owns the agent SoA,
/// the event ring, the view storage, and a chronicle / fold cfg
/// uniform pair. The mask + scoring kernels never compiled, so there
/// are no per-mask bitmaps and no scoring-output buffer.
pub struct PairScoringProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. Initialised all-1 so any future mask
    /// predicate (`self.alive`) would fire — today none of the
    /// emitted kernels actually read it.
    #[allow(dead_code)]
    agent_alive_buf: wgpu::Buffer,
    /// `cooldown_next_ready_tick` SoA — the score expression in the
    /// fixture references this via `agents.cooldown_next_ready_tick(target)`.
    /// Initialised to a STAGGERED pattern (`ready_at[N] = N * 10`) so
    /// IF the scoring kernel had compiled, slot 0 would absorb all
    /// healing. Not actually read by any compiled kernel (the score
    /// expression failed to lower → no kernel references it).
    #[allow(dead_code)]
    agent_cooldown_next_ready_tick_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-target healing-received accumulator (f32). Fed by `Healed`
    /// (kind tag = 1u). With no scoring kernel, no ActionSelected
    /// events land in the ring → chronicle never emits → fold sees
    /// nothing → every slot stays at 0.0.
    received: ViewStorage,
    received_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    chronicle_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl PairScoringProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — `alive` is unused by emitted kernels (no mask
        // compiled) but kept for parity with sibling probes. Initialised
        // all-1.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Cooldown SoA — staggered init `ready_at[N] = N * 10` so that
        // IF the scoring kernel had lowered, the per-pair argmax would
        // pick slot 0 (lowest cooldown → highest inverted score). Not
        // actually read by any compiled kernel today.
        let cooldown_init: Vec<u32> = (0..agent_count).map(|n| n * 10).collect();
        let agent_cooldown_next_ready_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::agent_cooldown_next_ready_tick"),
                contents: bytemuck::cast_slice(&cooldown_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Event ring + received view storage (per-agent f32, no anchor,
        // no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "pair_scoring_probe_runtime");
        let received = ViewStorage::new(
            &gpu,
            "pair_scoring_probe_runtime::received",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // Chronicle cfg uniform — sized for the verb chronicle even
        // though it's never going to find an ActionSelected slot.
        let chronicle_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                _pad0: 0,
            };
        let chronicle_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::chronicle_cfg"),
                contents: bytemuck::bytes_of(&chronicle_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let received_cfg_init = fold_received::FoldReceivedCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let received_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("pair_scoring_probe_runtime::received_cfg"),
                contents: bytemuck::bytes_of(&received_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            agent_cooldown_next_ready_tick_buf,
            event_ring,
            received,
            received_cfg_buf,
            chronicle_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-target healing-received accumulator (one f32 per slot).
    /// Under outcome (b) NO FIRE — every slot stays at 0.0 because no
    /// scoring kernel emits ActionSelected events. If a future commit
    /// closes Gap #1 (mask positional head + Task 2.6 from-clause
    /// routing) AND Gap #3 (verb-injected scoring entry uses Positional
    /// head shape) AND Gap #4 (ScoringArgmax dispatch grows N×N pair
    /// shape), this view should converge to:
    ///
    ///   received[0] = TICKS × (AGENT_COUNT - 1) × heal.amount
    ///               = 100 × 7 × 5.0 = 3500.0
    ///   received[N] = 0.0 for N > 0
    pub fn received(&mut self) -> &[f32] {
        self.received.readback(&self.gpu)
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

impl CompiledSim for PairScoringProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pair_scoring_probe_runtime::step"),
                });

        // (1) Per-tick clear of event_tail.
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) Chronicle dispatch — runs even though no ActionSelected
        // ever lands (event_count = agent_count is generous; the
        // per-handler tag check rejects whatever's in the ring). Useful
        // because it exercises the same dispatch shape downstream of
        // the (missing) scoring kernel and surfaces any binding-shape
        // regressions in the chronicle emit path.
        let chronicle_cfg =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                _pad0: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_bindings =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_heal(
            &mut self.cache,
            &chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) seed_indirect_0 — populates indirect_args_0 from
        // event_tail (kept for parity).
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
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

        // (4) fold_received — RMW per `Healed` event (kind = 1u). The
        // chronicle never emitted any, so the per-handler tag check
        // rejects every slot it sees and `received[N]` stays at 0.0
        // for every slot. Sized at agent_count * 2 to cover any
        // background event ring contents.
        let event_count_estimate = self.agent_count * 2;
        let received_cfg = fold_received::FoldReceivedCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.received_cfg_buf,
            0,
            bytemuck::bytes_of(&received_cfg),
        );
        let received_bindings = fold_received::FoldReceivedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.received.primary(),
            view_storage_anchor: self.received.anchor(),
            view_storage_ids: self.received.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.received_cfg_buf,
        };
        dispatch::dispatch_fold_received(
            &mut self.cache,
            &received_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.received.mark_dirty();
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
    Box::new(PairScoringProbeState::new(seed, agent_count))
}
