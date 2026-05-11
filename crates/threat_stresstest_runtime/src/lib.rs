//! Plan G G3h — per_entity_ring_probe_runtime sibling.
//!
//! Drives `assets/sim/threat_stresstest.sim` end-to-end on real
//! wgpu hardware. The .sim composes G3d's PerAgentEventScan dispatch
//! with G3a's view fold storage:
//!
//!   * physics MarkAllBusy — at `cast_begin_tick`, every alive
//!     agent stamps `busy_with_ability_id = cast_ability_id`. The
//!     where-clause guards on `world.tick == cast_begin_tick`, so
//!     this fires exactly once per agent per simulation.
//!   * fold_threats — PerAgentEventScan over (observer, source_candidate).
//!     The kernel preamble busy-filters: if
//!     `agent_busy_with_ability_id[source_candidate] == 0u`, return.
//!     Then `self += 1.0` increments observer's threats[] by 1 per
//!     busy candidate.
//!
//! ## Behavioural pin
//!
//! With `AGENT_COUNT=4`, `cast_begin_tick=0`:
//!
//!   Tick 0: every agent runs MarkAllBusy (each becomes busy with
//!     ability 7). Then fold_threats dispatches 4×4 = 16 (observer,
//!     candidate) pairs. All 4 candidates pass the busy filter (every
//!     agent is busy at tick 0), so each observer gets 4 increments.
//!     threats[obs] = 4.0 for every obs.
//!   Tick 1: MarkAllBusy's where-clause fails (`world.tick != 0`),
//!     so no busy state changes. The agents stay busy. fold_threats
//!     fires again, adding 4.0 per observer. threats = [8.0; 4].
//!
//! The non-zero count proves the dispatch + busy-filter + view
//! storage all wired correctly. A more realistic threats view would
//! decay (`@decay`) so the count reflects "current" threats, not
//! historical sum.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext, ViewStorage};
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
include!(concat!(env!("OUT_DIR"), "/runtime_core.rs"));

pub struct ThreatStresstestState {
    gpu: GpuContext,
    agent_alive_buf: wgpu::Buffer,
    /// `busy_with_ability_id` SoA — written by MarkAllBusy; read
    /// by the threats fold's busy-filter early-exit.
    agent_busy_with_ability_id_buf: wgpu::Buffer,

    /// Per-observer scalar f32 view (the `threats` view's primary
    /// storage). Sized `agent_count * 4` bytes, zero-init.
    threats: ViewStorage,

    event_ring: EventRing,

    physics_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,

    registry_gpu: PackedAbilityRegistryGpu,
    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    #[allow(dead_code)]
    seed: u64,
}

impl ThreatStresstestState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::try_new(seed, agent_count).expect("init wgpu adapter + device")
    }

    pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threat_stresstest::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // busy_with_ability_id starts at 0 (no agents busy). MarkAllBusy
        // stamps cast_ability_id at cast_begin_tick=0 on every alive
        // agent, switching the busy bits on for the rest of the sim.
        let busy_init: Vec<u32> = vec![0u32; agent_count as usize];
        let agent_busy_with_ability_id_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threat_stresstest::agent_busy_with_ability_id"),
                contents: bytemuck::cast_slice(&busy_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let event_ring = EventRing::new(&gpu, "threat_stresstest");
        let threats = ViewStorage::new(
            &gpu,
            "threat_stresstest::threats",
            agent_count,
            false, // no anchor
            false, // no ids
        );

        let physics_cfg_init = physics_MarkAllBusy::PhysicsMarkAllBusyCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threat_stresstest::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // PerAgentEventScan reuses the ViewFold cfg's `event_count`
        // field as agent_cap (per the cfg-shape gotcha documented in
        // `cg/emit/kernel.rs::build_view_fold_per_agent_event_scan_body`).
        // Runtime sets it to agent_count.
        let fold_cfg_init = fold_threats::FoldThreatsCfg {
            event_count: agent_count,
            tick: 0, second_key_pop: 1, _pad: 0,
        };
        let fold_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threat_stresstest::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &PackedAbilityRegistry::pack(&engine::ability::AbilityRegistry::new()),
            &gpu,
            "threat_stresstest",
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_busy_with_ability_id_buf,
            threats,
            event_ring,
            physics_cfg_buf,
            fold_cfg_buf,
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
}

impl CompiledSim for ThreatStresstestState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threat_stresstest::step"),
            });
        self.event_ring.clear_tail_in(&mut encoder);

        // Note: agent_busy_with_ability_id is NOT in AgentBuffers's
        // canonical SoA field set today — it's threaded through the
        // per-kernel `extras` struct instead (both physics_MarkAllBusy
        // and fold_threats expose it as an extras field). Future
        // iteration could promote it to AgentBuffers if more fixtures
        // need it from ctx.state.
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

        // (1) MarkAllBusy — sets agent_busy_with_ability_id at
        // cast_begin_tick=0 on every alive agent.
        let physics_cfg = physics_MarkAllBusy::PhysicsMarkAllBusyCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_extras = physics_MarkAllBusy::PhysicsMarkAllBusyExtras {
            agent_busy_with_ability_id: &self.agent_busy_with_ability_id_buf,
            cfg: &self.physics_cfg_buf,
        };
        let physics_bindings =
            physics_MarkAllBusy::PhysicsMarkAllBusyBindings::from_context_with_extras(
                &ctx, &physics_extras,
            );
        dispatch::dispatch_physics_markallbusy(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) fold_threats — PerAgentEventScan over (observer,
        // source_candidate) pairs. Busy-filter early-exits non-busy
        // sources; remaining pairs increment observer's view by 1.0.
        let fold_cfg = fold_threats::FoldThreatsCfg {
            event_count: self.agent_count, // PerAgentEventScan reuses as agent_cap
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
    Box::new(ThreatStresstestState::new(seed, agent_count))
}

#[cfg(test)]
mod threats_lifecycle_tests {
    use super::*;

    /// Plan G G3h — load-bearing GPU pin. Composes the full G3
    /// stack on real wgpu:
    ///
    ///   * G3d PerAgentEventScan dispatch (2-D `(obs, src)` thread layout)
    ///   * busy-filter early-exit (G3d kernel preamble)
    ///   * G3a view fold storage (per-agent f32 primary slot)
    ///   * `agents.set_busy_with_ability_id(...)` write (G2.7 setter)
    ///
    /// At tick 0, every agent stamps `busy_with_ability_id = 7`.
    /// fold_threats then sees 4 busy candidates × 4 observers = 16
    /// pairs all passing the busy-filter, each contributing +1.0.
    /// Result: every observer's `threats[obs] == 4.0`.
    ///
    /// At tick 1, busy state is preserved (MarkAllBusy's
    /// where-clause fails for tick != 0), fold runs again, +4.0 per
    /// observer. Result: `threats == [8.0, 8.0, 8.0, 8.0]`.
    ///
    /// The non-zero count proves the entire dispatch + filter +
    /// storage chain works.
    #[test]
    fn threats_count_increments_per_busy_source() {
        const N: u32 = 4;
        let mut state = match ThreatStresstestState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[threat_stresstest] skipping: no wgpu adapter on host. \
                     Build still validated emit + bindings at compile time."
                );
                return;
            }
        };

        // Tick 0 — MarkAllBusy fires (every agent becomes busy);
        // fold_threats then runs with all 4 sources busy.
        state.step();
        let r = state.read_threats();
        assert_eq!(r.len(), N as usize);
        for (obs, &count) in r.iter().enumerate() {
            assert!((count - 4.0).abs() < 1e-3,
                "tick 0: observer {obs} threats count = {count} (expected 4.0 = N busy candidates)");
        }

        // Tick 1 — busy state preserved (MarkAllBusy gates on tick == 0,
        // so it doesn't re-write but doesn't clear either). Fold adds
        // another +4.0 per observer.
        state.step();
        let r = state.read_threats();
        for (obs, &count) in r.iter().enumerate() {
            assert!((count - 8.0).abs() < 1e-3,
                "tick 1: observer {obs} threats count = {count} (expected 8.0 = 2 ticks × 4 busy)");
        }
    }

    /// Plan H — at-scale stresstest. 256 agents = 65,536
    /// (observer, source) pairs per tick. Worst case (every source
    /// busy → no busy-filter savings) so every pair runs the
    /// CAS-add body. Drives 32 ticks and reports per-tick wall
    /// clock + final threats[0] = 256 × 32 = 8192 (every observer
    /// sees every busy candidate every tick).
    #[test]
    fn stresstest_256_agents_32_ticks_perf_report() {
        const N: u32 = 256;
        const TICKS: u32 = 32;
        let mut state = match ThreatStresstestState::try_new(0xBEEF, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[threat_stresstest] skipping perf pin: no wgpu adapter on host."
                );
                return;
            }
        };

        // Warmup tick — first dispatch incurs pipeline compile cost;
        // measure steady-state from tick 1 onward.
        let warmup_start = std::time::Instant::now();
        state.step();
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
        let _ = state.read_threats(); // force GPU sync

        // Steady-state: 32 ticks at 256² = 65,536 pairs each.
        let mut tick_ms_samples: Vec<f64> = Vec::with_capacity(TICKS as usize);
        let stress_start = std::time::Instant::now();
        for _ in 0..TICKS {
            let t = std::time::Instant::now();
            state.step();
            tick_ms_samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let _ = state.read_threats(); // force GPU sync of last batch
        let stress_total_ms = stress_start.elapsed().as_secs_f64() * 1000.0;

        let r = state.read_threats();
        // After warmup (tick 0) + TICKS more: each tick adds N (256) per observer.
        // total ticks where fold ran = 1 + TICKS = 33.
        let expected = (N as f64) * (1.0 + TICKS as f64);
        let observed = r[0] as f64;
        eprintln!(
            "[threat_stresstest perf] N={N} TICKS={TICKS} → {pairs}/tick pairs",
            pairs = N * N,
        );
        eprintln!("[threat_stresstest perf] warmup tick: {warmup_ms:.3} ms (incl pipeline compile)");
        eprintln!(
            "[threat_stresstest perf] steady-state: {TICKS} ticks in {stress_total_ms:.3} ms ({:.3} ms/tick avg)",
            stress_total_ms / TICKS as f64,
        );
        let mut sorted = tick_ms_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95) / 100];
        let p_max = sorted[sorted.len() - 1];
        eprintln!(
            "[threat_stresstest perf] per-tick: p50={p50:.3} ms, p95={p95:.3} ms, max={p_max:.3} ms"
        );
        eprintln!(
            "[threat_stresstest perf] threats[0] = {observed} (expected {expected} = N × (1+TICKS))"
        );

        // Correctness pin: per-observer count matches expectation
        // (proves the dispatch + body fired the right number of times).
        for (obs, &count) in r.iter().enumerate().take(8) {
            assert!((count as f64 - expected).abs() < 1.0,
                "observer {obs} count = {count}, expected {expected}");
        }
    }
}


// Plan E-A6 sweep — generator validation smoke test for threat_stresstest_runtime.
#[cfg(test)]
mod a6_sweep_smoke {
    use crate::{GeneratedRuntime, FIXTURE_NAME, KERNEL_COUNT};

    #[test]
    fn generated_runtime_works_for_this_fixture() {
        let mut r = match GeneratedRuntime::try_new(0xCAFE, 4) {
            Some(s) => s,
            None => {
                eprintln!("[a6_sweep] skipping: no wgpu adapter on host.");
                return;
            }
        };
        assert!(KERNEL_COUNT > 0);
        r.step();
        r.step();
        assert_eq!(r.tick, 2);
        eprintln!("[a6_sweep] {}: KERNEL_COUNT={}, ran 2 ticks", FIXTURE_NAME, KERNEL_COUNT);
    }
}
