//! Per-fixture runtime for `assets/sim/stochastic_probe.sim` —
//! discovery probe for the `rng.*` namespace in a real per-agent
//! physics body. Mirrors `cooldown_probe_runtime` shape (single
//! physics rule + single fold + no @decay), but the physics body
//! reads from the deterministic per-agent RNG primitive instead of
//! a per-agent SoA slot.
//!
//! ## What this exercises
//!
//! Three surfaces wired through (in principle) end-to-end:
//!
//! 1. **`rng.action()`** — the only nullary `rng.*` purpose that
//!    lowers cleanly today (per
//!    `crates/dsl_compiler/src/cg/lower/expr.rs:2532-2544`). Returns
//!    u32. The lowered WGSL emits the literal call shape
//!    `per_agent_u32(seed, agent_id, tick, "action")` per
//!    `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:937-947`.
//!
//! 2. **Determinism (P5)** — same seed must yield byte-identical
//!    activations across two independent runs. The runtime
//!    constructs two `StochasticProbeState`s with the same seed,
//!    runs each TICKS times, and the harness compares the
//!    activations buffers element-wise.
//!
//! 3. **IfStmt gating on a u32 comparison** — same `IfStmt` shape
//!    cooldown_probe exercises with `tick >= ready_at`, but the
//!    LHS is RNG-derived rather than a SoA read.
//!
//! ## Per-tick chain
//!
//! 1. `clear_tail` — event_tail = 0 so `atomicAdd` slots restart.
//! 2. `physics_MaybeFire` — reads `agent_alive`; for each alive
//!    agent draws `rng.action()`, gates `if (draw % 100) < 30`,
//!    emits `Activated { agent: self, amount: 1.0 }`.
//! 3. `seed_indirect_0` — populates indirect-args buffer from the
//!    tail count (parity with the cooldown_probe pattern; the
//!    direct fold dispatch in (4) doesn't actually consume the args).
//! 4. `fold_activations` — per-handler tag-filter on `Activated`
//!    (kind = 1u), atomic RMW into per-agent activations storage.
//!
//! Following the diplomacy_probe Gap #2 close (commit `16905527`),
//! the cascade pattern uses `EventRing::note_emits` after the
//! producer dispatch and `EventRing::tail_value()` to size the
//! consumer dispatch's `event_count` cfg field. With one producer
//! per tick this is functionally equivalent to passing
//! `agent_count`, but it demonstrates the same cascade shape future
//! multi-stage probes will use.
//!
//! ## Observable
//!
//! With AGENT_COUNT=32, TICKS=1000, threshold=30%, log_amount=1.0:
//!
//!   per-slot activations[N] ≈ TICKS × 0.30 = 300
//!
//! ## Predicted outcome
//!
//! **(b) GAP — naga validation rejects the emitted WGSL**. The
//! `physics_MaybeFire.wgsl` body references three free identifiers
//! the WGSL emit pipeline doesn't bind:
//!
//!   - `seed`          — not in the kernel preamble (only
//!                       `agent_id`, `tick` are bound by
//!                       `thread_indexing_preamble`).
//!   - `per_agent_u32` — host Rust function; no WGSL prelude shim.
//!   - `"action"`      — string literal; WGSL does not support
//!                       string literals.
//!
//! See `docs/superpowers/notes/2026-05-04-stochastic_probe.md` for
//! the discovery report + gap punch list.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the stochastic probe. Owns:
///   - Agent SoA (alive only — pos/vel are declaration-only here)
///   - Event ring + per-view storage (activations: f32 per slot)
///   - Per-kernel cfg uniforms
pub struct StochasticProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. All-1 init so every slot's `self.alive`
    /// gate evaluates true.
    agent_alive_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-agent activation count (f32). Fed by Activated (kind tag
    /// = 1u). After T ticks: `activations[N] ≈ T × P(fire) = T × 0.30`.
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

impl StochasticProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — `alive` is read by physics_MaybeFire (BGL slot
        // 2). Initialised to all-1 so every slot fires its gate.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stochastic_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Event ring + activations view storage (per-agent f32, no
        // anchor, no ids — view declares no @decay).
        let event_ring = EventRing::new(&gpu, "stochastic_probe_runtime");
        let activations = ViewStorage::new(
            &gpu,
            "stochastic_probe_runtime::activations",
            agent_count,
            false, // no @decay anchor
            false, // no top-K ids
        );

        // Per-kernel cfg uniforms. `seed: u32` is the low 32 bits of
        // the runtime's u64 world seed — the GPU primitive
        // `per_agent_u32(seed, agent_id, tick, purpose_id)` keys on
        // u32 (matches `RNG_WGSL_PRELUDE` in dsl_compiler).
        let seed_lo = seed as u32;
        let physics_cfg_init = physics_MaybeFire::PhysicsMaybeFireCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: seed_lo,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stochastic_probe_runtime::physics_cfg"),
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
                label: Some("stochastic_probe_runtime::seed_cfg"),
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
                label: Some("stochastic_probe_runtime::activations_cfg"),
                contents: bytemuck::bytes_of(&activations_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
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

    /// Per-agent activation count (one f32 per slot). After T ticks:
    /// `activations[N] ≈ T × 0.30` under uniform RNG draws.
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

impl CompiledSim for StochasticProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("stochastic_probe_runtime::step"),
                });

        // (1) Per-tick clear of event_tail. Resets the host-side
        // `tail_value()` estimate to 0 so `note_emits` accumulates
        // cleanly within the tick. Also clear the ring header words
        // for the first `agent_count` slots so stale `kind=1u` tags
        // from previous ticks don't get re-folded — necessary because
        // this fixture's producer emits a STOCHASTIC count (≈30% of
        // agents per tick) but the consumer dispatches over the
        // upper bound (`agent_count`); without clearing, slots past
        // the actual producer tail still hold prior-tick tags. See
        // `EventRing::clear_ring_headers_in` doc.
        self.event_ring.clear_tail_in(&mut encoder);
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, self.agent_count);

        // (2) physics_MaybeFire — reads agent_alive. For each alive
        // agent: draws rng.action(), gates `(draw % 100) < 30`,
        // emits Activated{ agent=self, amount=1.0 } into the event
        // ring (kind tag = 1u).
        let physics_cfg = physics_MaybeFire::PhysicsMaybeFireCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: self.seed as u32,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_bindings = physics_MaybeFire::PhysicsMaybeFireBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_alive: &self.agent_alive_buf,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_physics_maybefire(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        // Producer upper bound: at most one Activated per agent per
        // tick. Records the bound on the host-side ring estimator so
        // downstream consumers can size `event_count` from
        // `tail_value()` (Gap #2 cascade pattern, commit 16905527).
        self.event_ring.note_emits(self.agent_count);

        // (3) seed_indirect_0 — populates indirect_args_0 from
        // event_tail (parity with cooldown_probe / verb_probe).
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

        // (4) fold_activations — per-handler tag-filter on Activated
        // (kind = 1u), atomic RMW into per-agent primary view
        // storage. `event_count` sourced from
        // `event_ring.tail_value()` — the host-tracked per-tick
        // upper bound (one Activated per agent per tick = agent_count
        // here, but pulling through `tail_value()` exercises the
        // multi-stage cascade accessor pattern future probes will
        // use).
        let event_count_estimate = self.event_ring.tail_value();
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
        // same skip pattern as cooldown_probe / verb_probe.

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
    Box::new(StochasticProbeState::new(seed, agent_count))
}
