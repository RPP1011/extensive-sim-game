//! Decay-enabled threats view fixture (sibling of
//! `threats_view_probe_runtime`). Adds `@decay(rate=0.9, per=tick)`
//! to the threats view; the per-tick `decay_threats` kernel runs
//! before the `fold_threats` kernel each tick.
//!
//! ## Behavioural pin
//!
//! With `AGENT_COUNT=4`, `cast_begin_tick=0`:
//!
//!   Tick 0: storage starts at 0. Decay multiplies 0 * 0.9 = 0.
//!     MarkCasterBusy stamps every agent busy. fold_threats adds
//!     +4 per observer. threats[obs] = 4.0.
//!   Tick K (busy preserved): storage_K = storage_{K-1} * 0.9 + 4.
//!     Steady-state: 4 / (1 - 0.9) = 40.
//!
//! The decay pin verifies:
//!   1. After many ticks of constant +4 with decay, the value is
//!      bounded near the steady-state cap (40), not growing
//!      unboundedly like the no-decay sibling.
//!   2. After the busy state clears (caster_clear_tick), the
//!      decay-only ticks drive the value back toward 0.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext, ViewStorage};
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

pub struct ThreatsWithDecayProbeState {
    gpu: GpuContext,
    agent_alive_buf: wgpu::Buffer,
    /// `busy_with_ability_id` SoA — written by MarkCasterBusy (when
    /// the test harness wants busy stamped) and consumed by
    /// fold_threats's busy-filter early-exit.
    agent_busy_with_ability_id_buf: wgpu::Buffer,

    threats: ViewStorage,

    event_ring: EventRing,

    physics_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,
    decay_cfg_buf: wgpu::Buffer,

    registry_gpu: PackedAbilityRegistryGpu,
    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    #[allow(dead_code)]
    seed: u64,
}

impl ThreatsWithDecayProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::try_new(seed, agent_count).expect("init wgpu adapter + device")
    }

    pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threats_with_decay_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let busy_init: Vec<u32> = vec![0u32; agent_count as usize];
        let agent_busy_with_ability_id_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threats_with_decay_probe::agent_busy_with_ability_id"),
                contents: bytemuck::cast_slice(&busy_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let event_ring = EventRing::new(&gpu, "threats_with_decay_probe");
        let threats = ViewStorage::new(
            &gpu,
            "threats_with_decay_probe::threats",
            agent_count,
            false, // no anchor
            false, // no ids
        );

        let physics_cfg_init = physics_MarkCasterBusy::PhysicsMarkCasterBusyCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_with_decay_probe::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let fold_cfg_init = fold_threats::FoldThreatsCfg {
            event_count: agent_count,
            tick: 0, second_key_pop: 1, _pad: 0,
        };
        let fold_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_with_decay_probe::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Decay cfg — single-key view, so slot_count == agent_cap.
        let decay_cfg_init = decay_threats::DecayThreatsCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_with_decay_probe::decay_cfg"),
                contents: bytemuck::bytes_of(&decay_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &PackedAbilityRegistry::pack(&engine::ability::AbilityRegistry::new()),
            &gpu,
            "threats_with_decay_probe",
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_busy_with_ability_id_buf,
            threats,
            event_ring,
            physics_cfg_buf,
            fold_cfg_buf,
            decay_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        })
    }

    pub fn read_threats(&mut self) -> &[f32] {
        self.threats.readback(&self.gpu)
    }

    /// Test hook: zero out the busy SoA so subsequent ticks see no
    /// busy candidates → fold adds 0 → only decay drives the view.
    pub fn clear_busy(&mut self) {
        let zeros: Vec<u32> = vec![0u32; self.agent_count as usize];
        self.gpu.queue.write_buffer(
            &self.agent_busy_with_ability_id_buf,
            0,
            bytemuck::cast_slice(&zeros),
        );
    }
}

impl CompiledSim for ThreatsWithDecayProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threats_with_decay_probe::step"),
            });
        self.event_ring.clear_tail_in(&mut encoder);

        let agent_buffers = AgentBuffers {
            alive_buf: Some(&self.agent_alive_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
            voxel_grid: None,
        };

        // (1) decay_threats — anchor multiplication. Runs FIRST so
        // the per-tick fold sees the decayed value; if no busy
        // sources exist this tick, the fold contributes 0 and the
        // view drifts toward 0.
        let decay_cfg = decay_threats::DecayThreatsCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.decay_cfg_buf, 0, bytemuck::bytes_of(&decay_cfg));
        let decay_bindings = decay_threats::DecayThreatsBindings {
            view_storage_primary: self.threats.primary(),
            cfg: &self.decay_cfg_buf,
        };
        dispatch::dispatch_decay_threats(
            &mut self.cache,
            &decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) MarkCasterBusy — sets busy SoA at cast_begin_tick=0
        // on every alive agent.
        let physics_cfg = physics_MarkCasterBusy::PhysicsMarkCasterBusyCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_extras = physics_MarkCasterBusy::PhysicsMarkCasterBusyExtras {
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            cfg: &self.physics_cfg_buf,
        };
        let physics_bindings =
            physics_MarkCasterBusy::PhysicsMarkCasterBusyBindings::from_context_with_extras(
                &ctx, &physics_extras,
            );
        dispatch::dispatch_physics_markcasterbusy(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) fold_threats — adds +1.0 per (observer, busy source)
        // pair after decay has scaled the prior tick's value.
        let fold_cfg = fold_threats::FoldThreatsCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.fold_cfg_buf, 0, bytemuck::bytes_of(&fold_cfg));
        let fold_bindings = fold_threats::FoldThreatsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.threats.primary(),
            view_storage_anchor: self.threats.anchor(),
            view_storage_ids: self.threats.ids(),
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_threats(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.threats.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(ThreatsWithDecayProbeState::new(seed, agent_count))
}

#[cfg(test)]
mod decay_lifecycle_tests {
    use super::*;

    /// Decay annotation pin: the view value is bounded near the
    /// steady-state cap (4 / (1 - 0.9) = 40), not growing
    /// unboundedly. After busy clears the value decays toward 0.
    #[test]
    fn threats_decay_bounds_value_and_falls_to_zero() {
        const N: u32 = 4;
        let mut state = match ThreatsWithDecayProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[threats_with_decay_probe] skipping: no wgpu adapter on host. \
                     Build still validated emit + bindings at compile time."
                );
                return;
            }
        };

        // Tick 0 — decay: 0 * 0.9 = 0; busy stamped; fold adds +4.
        // threats[obs] = 4.0 every observer.
        state.step();
        let r = state.read_threats();
        for (obs, &v) in r.iter().enumerate() {
            assert!(
                (v - 4.0).abs() < 1e-3,
                "tick 0: observer {obs} threats = {v} (expected 4.0)"
            );
        }

        // Run 30 more busy ticks. Steady-state cap is 40; before
        // reaching it the value monotonically increases, but stays
        // strictly below 40 / (1 - 0.9^k_max).
        for _ in 0..30 {
            state.step();
        }
        let r = state.read_threats();
        for (obs, &v) in r.iter().enumerate() {
            assert!(
                v < 40.5,
                "after 31 busy ticks: observer {obs} threats = {v} should be < 40.5 (steady-state cap = 40)"
            );
            assert!(
                v > 30.0,
                "after 31 busy ticks: observer {obs} threats = {v} should be > 30 (close to cap)"
            );
        }

        // Now clear busy. Subsequent ticks: fold contributes 0, so
        // only decay runs. After 30 decay-only ticks, value should
        // be < 1.0 (0.9^30 ≈ 0.042; 40 * 0.042 ≈ 1.68; we use a
        // looser bound to absorb prior overshoot).
        state.clear_busy();
        for _ in 0..30 {
            state.step();
        }
        let r = state.read_threats();
        for (obs, &v) in r.iter().enumerate() {
            assert!(
                v < 2.0,
                "after 30 decay-only ticks: observer {obs} threats = {v} should be < 2.0 (decayed from ~40)"
            );
            assert!(v >= 0.0, "decay must not drive value below 0; got {v}");
        }
    }

    /// **Short-vs-long-term threat effectiveness.** Closes the
    /// "stresstest short and long term threats" half of the
    /// recurring prompt. Quantifies how `@decay` lets the threat
    /// system distinguish a one-shot spike from a sustained
    /// pressure — the property AI scoring needs so a single old
    /// threat doesn't keep an agent in panic mode forever.
    ///
    /// Scenario A (sustained / long-term): busy stays at 1 every
    /// tick. The view climbs toward the steady-state cap of
    /// `N / (1 - rate)` and stays there. Mock host-side scoring
    /// (Flee threshold = 0.5, mirroring `dodger_probe.sim`'s Idle
    /// baseline) reads each tick → expect ~100% Flee.
    ///
    /// Scenario B (spike / short-term): busy=1 at tick 0, then
    /// cleared. Tick 0 fold spikes the view to N. Subsequent ticks
    /// run decay only — view drops geometrically toward 0. The
    /// per-tick "would Flee" answer flips from yes to no when the
    /// view crosses 0.5. With N=4 and rate=0.9, that crossing is
    /// at tick log_{0.9}(0.5/4) = ~19.7 → tick 20.
    ///
    /// The effectiveness metric IS this transition: a real AI's
    /// behavior tracks the threat's recency, not just its presence.
    /// Without decay, both scenarios would be indistinguishable —
    /// the spike would persist as a static value forever.
    #[test]
    fn short_vs_long_term_threat_effectiveness_via_decay() {
        const N: u32 = 4;
        const TICKS: usize = 32;
        const FLEE_THRESHOLD: f32 = 0.5;

        // Helper: per-tick "would the dodger pick Flee" given the view value.
        // Mirrors dodger_probe.sim's `Flee score (threats.intensity_at(self))`
        // vs `Idle score 0.5` argmax.
        fn would_flee(threats: f32) -> bool {
            threats > FLEE_THRESHOLD
        }

        // --- Scenario A: sustained busy (long-term threat).
        let mut state_a = match ThreatsWithDecayProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!("[short-vs-long] skipping: no wgpu adapter on host.");
                return;
            }
        };
        let mut a_per_tick: Vec<f32> = Vec::with_capacity(TICKS);
        let mut a_flee_per_tick: Vec<bool> = Vec::with_capacity(TICKS);
        for _ in 0..TICKS {
            state_a.step();
            // Read observer 0 — every observer is symmetric here.
            let v = state_a.read_threats()[0];
            a_per_tick.push(v);
            a_flee_per_tick.push(would_flee(v));
        }

        // --- Scenario B: spike + decay (short-term threat).
        let mut state_b =
            ThreatsWithDecayProbeState::try_new(0xCAFE, N).expect("second wgpu state");
        // Tick 0 — busy=1 (preloaded), fold runs, view = 4.0.
        state_b.step();
        // Now clear busy. From here, fold contributes 0 every tick;
        // only decay runs.
        state_b.clear_busy();
        let mut b_per_tick: Vec<f32> = Vec::with_capacity(TICKS);
        let mut b_flee_per_tick: Vec<bool> = Vec::with_capacity(TICKS);
        b_per_tick.push(state_b.read_threats()[0]);
        b_flee_per_tick.push(would_flee(b_per_tick[0]));
        for _ in 1..TICKS {
            state_b.step();
            let v = state_b.read_threats()[0];
            b_per_tick.push(v);
            b_flee_per_tick.push(would_flee(v));
        }

        let a_flees: usize = a_flee_per_tick.iter().filter(|x| **x).count();
        let b_flees: usize = b_flee_per_tick.iter().filter(|x| **x).count();
        let a_pct = (a_flees as f64 / TICKS as f64) * 100.0;
        let b_pct = (b_flees as f64 / TICKS as f64) * 100.0;
        let b_first_no_flee = b_flee_per_tick.iter().position(|x| !x);

        eprintln!(
            "[short-vs-long] N={N} TICKS={TICKS} threshold={FLEE_THRESHOLD}"
        );
        eprintln!(
            "[short-vs-long] LONG-TERM (sustained busy):  {a_flees}/{TICKS} ticks would-Flee ({a_pct:.1}%)"
        );
        eprintln!(
            "[short-vs-long] SHORT-TERM (one spike+decay): {b_flees}/{TICKS} ticks would-Flee ({b_pct:.1}%)"
        );
        eprintln!(
            "[short-vs-long]   spike crosses below threshold at tick {:?}",
            b_first_no_flee
        );
        eprintln!(
            "[short-vs-long]   per-tick view A (long): {a_per_tick:.2?}"
        );
        eprintln!(
            "[short-vs-long]   per-tick view B (short): {b_per_tick:.2?}"
        );

        // Pin: long-term sustains Flee 100% of ticks.
        assert_eq!(
            a_flees, TICKS,
            "LONG-TERM threat must sustain would-Flee every tick; got {a_flees}/{TICKS}"
        );
        // Pin: short-term spike eventually crosses below threshold.
        // At rate=0.9, view = 4 * 0.9^t. 4 * 0.9^t < 0.5 → t > 19.7.
        // So tick ~20 is the crossover.
        let crossover = b_first_no_flee
            .expect("SHORT-TERM threat must eventually decay below threshold");
        assert!(
            (15..=25).contains(&crossover),
            "decay crossover should be near tick 20 (4 * 0.9^t < 0.5 at t≈19.7); got {crossover}"
        );
        // Pin: behavioural delta — long-term keeps fleeing more than short-term.
        let delta_pp = a_pct - b_pct;
        assert!(
            delta_pp >= 30.0,
            "long-term threat should sustain Flee much more than short-term spike; got delta {delta_pp:.1}pp"
        );
    }
}
