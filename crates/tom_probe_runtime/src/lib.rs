//! Per-fixture runtime for `assets/sim/tom_probe.sim` — discovery probe
//! of the Theory-of-Mind belief-read path.
//!
//! NOTE: this crate exists primarily to surface the gap chain
//! documented in `docs/superpowers/notes/2026-05-04-tom-probe.md`.
//! Both physics rules in the .sim fixture (`CheckBelief` exercising
//! `BeliefsAccessor` and `CheckBeliefBit` exercising
//! `theory_of_mind.believes_knows()`) are dropped at CG-lower time
//! today. The compiler still emits the `fact_witnesses` view-fold +
//! the admin kernels, so this runtime stands up:
//!
//!   - the Knower SoA (pos, vel, alive)
//!   - the LearnedFact event ring (which stays empty per tick)
//!   - the fact_witnesses view storage (per-Knower f32, no @decay)
//!
//! and dispatches the fold each tick. With no LearnedFact events
//! ever written, fact_witnesses[i] = 0.0 for all i across all ticks
//! — the OUTCOME (b) NO FIRE signal that the gap chain produces.
//!
//! When the BeliefsAccessor + believes_knows lowering gaps close in
//! follow-up work, this runtime needs ONE more thing the verb_probe
//! runtime didn't: a per-(observer, target) belief storage SoA. The
//! shape of that SoA is the OPEN DESIGN QUESTION surfaced by this
//! probe (see the discovery doc, "Gap #3" — runtime infrastructure).

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Padded {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z, _pad: 0.0 }
    }
}

/// Per-fixture state for the ToM probe. Carries the Knower SoA
/// (`agent_alive`), the LearnedFact event ring, and the
/// `fact_witnesses` view storage. Today the event ring stays empty
/// every tick because both producer physics rules drop out at lower
/// time.
pub struct TomProbeState {
    gpu: GpuContext,

    // -- Agent SoA (would be read by the producer rules if they
    //    hadn't dropped out at lower time) --
    /// 1 = alive, 0 = dead. Initialised all-1 so the (currently-
    /// dropped) `self.alive` predicate would evaluate true.
    #[allow(dead_code)]
    agent_alive_buf: wgpu::Buffer,

    // -- Per-(observer, target) belief storage placeholder --
    //
    // RUNTIME GAP #3 (see discovery doc): the lowered probe would
    // need a per-(observer, target) f32 confidence buffer here, plus
    // bind-group plumbing through the producer kernel. Today that
    // shape is undecided (per-agent ring vs flattened pair grid vs
    // BoundedMap mirror). Field stub kept so the design surface is
    // visible in the runtime even though no buffer is allocated.

    // -- Event ring + per-view storage helpers --
    event_ring: EventRing,
    fact_witnesses: ViewStorage,
    fact_witnesses_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl TomProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Knower SoA — only `alive` would be read by the (dropped)
        // producer kernels. Allocate it for shape parity with
        // verb_probe and so that when the lowering gap closes the
        // binding is already in place.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Event-ring + fact_witnesses view storage (per-agent f32, no
        // anchor, no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "tom_probe_runtime");
        let fact_witnesses = ViewStorage::new(
            &gpu,
            "tom_probe_runtime::fact_witnesses",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // The fold kernel's cfg uniform — same {event_count, tick,
        // second_key_pop, _pad} layout that other view-folds use. We
        // size event_count = 0 each tick because no producer rule
        // emits today; if the gaps close, this becomes the ring
        // tail count.
        let fact_witnesses_cfg_init = fold_fact_witnesses::FoldFactWitnessesCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let fact_witnesses_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::fact_witnesses_cfg"),
                contents: bytemuck::bytes_of(&fact_witnesses_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_alive_buf,
            event_ring,
            fact_witnesses,
            fact_witnesses_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-observer fact_witnesses accumulator (one f32 per Knower
    /// slot). Stays at 0.0 today because both producer rules dropped
    /// out at lower time.
    pub fn fact_witnesses(&mut self) -> &[f32] {
        self.fact_witnesses.readback(&self.gpu)
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

        // (1) Per-tick clear of event_tail. With no producer rule the
        // ring stays empty, but we clear for shape parity.
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) fold_fact_witnesses — reads LearnedFact events and
        // accumulates into fact_witnesses[observer]. With no producer
        // emitting LearnedFact today, the kernel's tag-check guard
        // (event_ring[..0u] == LearnedFact_kind_id) finds zero
        // matching slots and the storage stays at 0.0.
        //
        // event_count is sized to agent_count so that IF a producer
        // ever lands and writes LearnedFact slots into the (currently
        // pre-zeroed) ring, the fold would see them. With event_count
        // = 0 today (the safe lower bound), the kernel early-returns
        // on every thread and the ring is left untouched.
        let event_count_estimate: u32 = 0;
        let fact_witnesses_cfg = fold_fact_witnesses::FoldFactWitnessesCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.fact_witnesses_cfg_buf,
            0,
            bytemuck::bytes_of(&fact_witnesses_cfg),
        );
        let fact_witnesses_bindings = fold_fact_witnesses::FoldFactWitnessesBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.fact_witnesses.primary(),
            view_storage_anchor: self.fact_witnesses.anchor(),
            view_storage_ids: self.fact_witnesses.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fact_witnesses_cfg_buf,
        };
        dispatch::dispatch_fold_fact_witnesses(
            &mut self.cache,
            &fact_witnesses_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.fact_witnesses.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice. Same shape as
        // verb_probe_runtime (which has the same comment).
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TomProbeState::new(seed, agent_count))
}
