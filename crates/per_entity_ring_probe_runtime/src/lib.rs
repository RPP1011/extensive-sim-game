//! Plan G G3a-step-3 — per_entity_ring_probe_runtime.
//!
//! Drives `assets/sim/per_entity_ring_probe.sim` end-to-end on real
//! wgpu hardware. The fixture's `recent_damages` view uses the new
//! PerEntityRing storage hint (K=4); G3a-step-2 (PR #91) added the
//! WGSL emit for ring-append. This runtime supplies the GPU buffers
//! that the emit expects:
//!
//!   * `view_storage_primary`: `agent_count * K` u32 slots
//!     (= the ring storage; each cell holds one f32 amount bitcast).
//!   * `view_storage_anchor`: `agent_count` atomic u32 slots
//!     (= the cursors counter; atomicAdd hands out monotonic
//!     indices, ring_idx = `target * K + (cursor % K)`).
//!
//! Both buffers are zero-initialized so the cursor starts at 0 and
//! ring slots show 0 for unwritten cells.
//!
//! ## Behavioural pin
//!
//! With AGENT_COUNT=2, K=4, .sim's InjectDamage emitting
//! `Damaged{target=self, amount=10 + tick * 10}` per tick:
//!
//!   Tick 0: cursor++=0 ⇒ slot 0 ← 10. ring[0] = [10, 0, 0, 0]
//!   Tick 1: cursor++=1 ⇒ slot 1 ← 20. ring[0] = [10, 20, 0, 0]
//!   Tick 2: cursor++=2 ⇒ slot 2 ← 30. ring[0] = [10, 20, 30, 0]
//!   Tick 3: cursor++=3 ⇒ slot 3 ← 40. ring[0] = [10, 20, 30, 40]
//!   Tick 4: cursor++=4 ⇒ slot 0 ← 50 (wrap). ring[0] = [50, 20, 30, 40]
//!
//! Agent 1 has the same pattern (each agent damages itself per the
//! .sim's `InjectDamage` rule).
//!
//! The wrap at tick 4 is the load-bearing assertion — proves the
//! `% K` modulo in the ring-append WGSL works correctly.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
include!(concat!(env!("OUT_DIR"), "/runtime_core.rs"));

const RING_K: u32 = 4;

pub struct PerEntityRingProbeState {
    gpu: GpuContext,
    agent_alive_buf: wgpu::Buffer,

    /// Ring storage — `agent_count * K` u32 slots, zero-init.
    /// Each cell holds the bitcast f32 amount written by the
    /// ring-append emit. WGSL declares this as
    /// `array<atomic<u32>>` (atomicStore at the K-modulo'd index).
    view_storage_primary_buf: wgpu::Buffer,
    /// Cursors — `agent_count` atomic u32 slots, zero-init.
    /// WGSL declares this as `array<atomic<u32>>` (atomicAdd to
    /// allocate the next ring index).
    view_storage_anchor_buf: wgpu::Buffer,
    /// Staging buffer for the primary readback (`agent_count * K *
    /// 4` bytes).
    view_storage_primary_staging: wgpu::Buffer,
    /// Staging buffer for the cursors readback (`agent_count * 4`
    /// bytes).
    view_storage_anchor_staging: wgpu::Buffer,

    event_ring: EventRing,

    physics_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,

    registry_gpu: PackedAbilityRegistryGpu,
    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    #[allow(dead_code)]
    seed: u64,
}

impl PerEntityRingProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::try_new(seed, agent_count).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — `None` when no compatible wgpu adapter
    /// is on the host. Lets the closed-loop test degrade to a
    /// skip-with-message instead of a panic (mirrors
    /// `firebolt_probe_runtime::try_new`).
    pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        // Agent SoA — `alive` is read by physics_InjectDamage's
        // `where (self.alive)` gate. All-1.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("per_entity_ring_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Ring storage. agent_count * K u32 slots, zero-init.
        // wgpu requires nonzero-sized storage buffer bindings, so
        // floor at 16 bytes for tiny tests.
        let primary_bytes = ((agent_count as u64) * (RING_K as u64) * 4).max(16);
        let primary_init: Vec<u32> = vec![0u32; (primary_bytes / 4) as usize];
        let view_storage_primary_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("per_entity_ring_probe::view_storage_primary"),
                contents: bytemuck::cast_slice(&primary_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Cursors counter. agent_count atomic u32 slots, zero-init.
        let anchor_bytes = ((agent_count as u64) * 4).max(16);
        let anchor_init: Vec<u32> = vec![0u32; (anchor_bytes / 4) as usize];
        let view_storage_anchor_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("per_entity_ring_probe::view_storage_anchor"),
                contents: bytemuck::cast_slice(&anchor_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let view_storage_primary_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("per_entity_ring_probe::view_storage_primary_staging"),
            size: primary_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let view_storage_anchor_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("per_entity_ring_probe::view_storage_anchor_staging"),
            size: anchor_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let event_ring = EventRing::new(&gpu, "per_entity_ring_probe");

        let physics_cfg_init = physics_InjectDamage::PhysicsInjectDamageCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("per_entity_ring_probe::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("per_entity_ring_probe::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let fold_cfg_init = fold_recent_damages::FoldRecentDamagesCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let fold_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("per_entity_ring_probe::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &PackedAbilityRegistry::pack(&engine::ability::AbilityRegistry::new()),
            &gpu,
            "per_entity_ring_probe",
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            view_storage_primary_buf,
            view_storage_anchor_buf,
            view_storage_primary_staging,
            view_storage_anchor_staging,
            event_ring,
            physics_cfg_buf,
            seed_cfg_buf,
            fold_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        })
    }

    /// Read the ring storage as `agent_count * K` f32 values
    /// (bitcast from the underlying u32). Layout:
    /// `ring[agent * K + slot]`.
    pub fn read_ring(&self) -> Vec<f32> {
        let primary_bytes =
            ((self.agent_count as u64) * (RING_K as u64) * 4).max(16);
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("per_entity_ring_probe::read_ring"),
            });
        encoder.copy_buffer_to_buffer(
            &self.view_storage_primary_buf,
            0,
            &self.view_storage_primary_staging,
            0,
            primary_bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = self.view_storage_primary_staging.slice(..primary_bytes);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("primary_staging map_async failed")
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during ring readback");
        let out = {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            floats.to_vec()
        };
        self.view_storage_primary_staging.unmap();
        out
    }

    /// Read the cursor counters (one u32 per agent). Useful for
    /// asserting the expected number of ring-appends fired.
    pub fn read_cursors(&self) -> Vec<u32> {
        let anchor_bytes = ((self.agent_count as u64) * 4).max(16);
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("per_entity_ring_probe::read_cursors"),
            });
        encoder.copy_buffer_to_buffer(
            &self.view_storage_anchor_buf,
            0,
            &self.view_storage_anchor_staging,
            0,
            anchor_bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = self.view_storage_anchor_staging.slice(..anchor_bytes);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("anchor_staging map_async failed")
        });
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed during cursors readback");
        let out = {
            let view = slice.get_mapped_range();
            let ints: &[u32] = bytemuck::cast_slice(&view);
            ints.to_vec()
        };
        self.view_storage_anchor_staging.unmap();
        out
    }
}

impl CompiledSim for PerEntityRingProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("per_entity_ring_probe::step"),
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

        // (1) physics_InjectDamage — per_agent: emit Damaged{
        // source=self, target=self, amount = 10 + tick * 10 } each
        // tick.
        let physics_cfg = physics_InjectDamage::PhysicsInjectDamageCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_extras = physics_InjectDamage::PhysicsInjectDamageExtras {
            cfg: &self.physics_cfg_buf,
        };
        let physics_bindings =
            physics_InjectDamage::PhysicsInjectDamageBindings::from_context_with_extras(
                &ctx, &physics_extras,
            );
        dispatch::dispatch_physics_injectdamage(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) seed_indirect_0 — parity (the consumer dispatch is
        // direct, but the schedule lists the indirect-args seed).
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let seed_extras = seed_indirect_0::SeedIndirect0Extras {
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings::from_context_with_extras(
            &ctx, &seed_extras,
        );
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) fold_recent_damages — Plan G G3a's ring-append. Reads
        // each Damaged event's (target, amount) from the ring;
        // atomicAdd's the cursor for the target; writes amount at
        // ring slot `target * K + (cursor % K)`. The buffer
        // contents flow back via read_ring().
        //
        // event_count_estimate = agent_count: each alive agent
        // emits exactly one Damaged event per tick (per the
        // InjectDamage rule's `where (self.alive && tick >= ...)`).
        let event_count_estimate = self.agent_count;
        let fold_cfg = fold_recent_damages::FoldRecentDamagesCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.fold_cfg_buf, 0, bytemuck::bytes_of(&fold_cfg));
        let fold_bindings = fold_recent_damages::FoldRecentDamagesBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.view_storage_primary_buf,
            view_storage_anchor: Some(&self.view_storage_anchor_buf),
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_recent_damages(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
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
    Box::new(PerEntityRingProbeState::new(seed, agent_count))
}

#[cfg(test)]
mod ring_lifecycle_tests {
    use super::*;

    /// Plan G G3a-step-3 — load-bearing GPU pin. Exercises the
    /// PerEntityRing emit (PR #91) end-to-end on real wgpu hardware.
    /// The wrap at tick 4 (slot 0 overwritten with `50.0`) proves
    /// the `% K` modulo in the ring-append WGSL works.
    #[test]
    fn ring_fills_then_wraps_at_k_plus_1_events() {
        const N: u32 = 2;
        let mut state = match PerEntityRingProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[per_entity_ring_probe] skipping: no wgpu adapter on host. \
                     Build still validated emit + bindings at compile time."
                );
                return;
            }
        };

        // Tick 0 — first event lands in ring slot 0.
        state.step();
        let r = state.read_ring();
        // ring layout: [agent0_slot0, agent0_slot1, agent0_slot2,
        // agent0_slot3, agent1_slot0, agent1_slot1, …].
        assert_eq!(r.len(), (N as usize) * (RING_K as usize));
        assert!((r[0] - 10.0).abs() < 1e-3, "tick 0: agent0 slot0 should be 10.0, got {}", r[0]);
        assert!(r[1].abs() < 1e-3, "tick 0: agent0 slot1 unwritten, got {}", r[1]);
        assert!((r[4] - 10.0).abs() < 1e-3, "tick 0: agent1 slot0 should be 10.0, got {}", r[4]);

        // Tick 1 — second event in slot 1.
        state.step();
        let r = state.read_ring();
        assert!((r[0] - 10.0).abs() < 1e-3);
        assert!((r[1] - 20.0).abs() < 1e-3, "tick 1: agent0 slot1 should be 20.0, got {}", r[1]);

        // Tick 2 — slot 2.
        state.step();
        let r = state.read_ring();
        assert!((r[2] - 30.0).abs() < 1e-3, "tick 2: agent0 slot2 should be 30.0, got {}", r[2]);

        // Tick 3 — slot 3 fills the ring.
        state.step();
        let r = state.read_ring();
        assert!((r[3] - 40.0).abs() < 1e-3, "tick 3: agent0 slot3 should be 40.0, got {}", r[3]);
        // Agent 0 ring is now full: [10, 20, 30, 40].
        for (i, expected) in [10.0_f32, 20.0, 30.0, 40.0].iter().enumerate() {
            assert!((r[i] - expected).abs() < 1e-3,
                "after tick 3: agent0 slot{i} = {} (expected {expected})", r[i]);
        }

        // Tick 4 — fifth event WRAPS to slot 0 (cursor=4, 4 % 4 == 0).
        // This is the load-bearing assertion.
        state.step();
        let r = state.read_ring();
        assert!((r[0] - 50.0).abs() < 1e-3,
            "tick 4 (WRAP): agent0 slot0 should be 50.0 (overwriting 10.0), got {}", r[0]);
        assert!((r[1] - 20.0).abs() < 1e-3, "tick 4: agent0 slot1 unchanged at 20.0, got {}", r[1]);
        assert!((r[2] - 30.0).abs() < 1e-3, "tick 4: agent0 slot2 unchanged at 30.0, got {}", r[2]);
        assert!((r[3] - 40.0).abs() < 1e-3, "tick 4: agent0 slot3 unchanged at 40.0, got {}", r[3]);

        // Cursor sanity — each agent has fired 5 events.
        let cursors = state.read_cursors();
        assert_eq!(cursors[0], 5, "agent0 cursor after 5 events");
        assert_eq!(cursors[1], 5, "agent1 cursor after 5 events");
    }

    /// Plan G G3a-step-3 hardening — verify the cursor wraps cleanly
    /// after multiple full revolutions. After 8 ticks, cursor=8 and
    /// 8 % 4 == 0 again, so slot 0 receives tick-7's amount = 80.0
    /// (NOT 10.0 from tick 0 nor 50.0 from tick 4).
    ///
    /// Sequence (one event per agent per tick, amount = 10 + tick*10):
    ///   t=0: cursor=0, slot 0 ← 10
    ///   t=1: cursor=1, slot 1 ← 20
    ///   t=2: cursor=2, slot 2 ← 30
    ///   t=3: cursor=3, slot 3 ← 40
    ///   t=4: cursor=4, slot 0 ← 50 (1st wrap)
    ///   t=5: cursor=5, slot 1 ← 60
    ///   t=6: cursor=6, slot 2 ← 70
    ///   t=7: cursor=7, slot 3 ← 80
    /// → ring[0] = [50, 60, 70, 80]; cursors[0] = 8.
    ///
    /// One more tick (8) wraps again:
    ///   t=8: cursor=8, slot 0 ← 90 (2nd wrap)
    /// → ring[0] = [90, 60, 70, 80]; cursors[0] = 9.
    ///
    /// This pin guards against a regression where the modulo is
    /// computed wrong (e.g. `cursor / K` instead of `cursor % K`)
    /// — the failure pattern would be slot 0 staying at 50 / never
    /// getting tick-8's value.
    #[test]
    fn ring_wraps_cleanly_after_two_full_revolutions() {
        const N: u32 = 2;
        let mut state = match PerEntityRingProbeState::try_new(0xDEAD, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[per_entity_ring_probe] skipping multi-wrap pin: no wgpu adapter."
                );
                return;
            }
        };
        // 8 ticks → cursor=8 → ring just wrapped once (slot 0 = tick-4's 50,
        // slot 1 = tick-5's 60, slot 2 = tick-6's 70, slot 3 = tick-7's 80).
        for _ in 0..8 {
            state.step();
        }
        let r = state.read_ring();
        for (slot, expected) in [(0_usize, 50.0_f32), (1, 60.0), (2, 70.0), (3, 80.0)] {
            assert!((r[slot] - expected).abs() < 1e-3,
                "after 8 ticks (1st full wrap): agent0 slot{slot} = {} (expected {expected})",
                r[slot]);
        }
        let cursors = state.read_cursors();
        assert_eq!(cursors[0], 8, "agent0 cursor after 8 events");

        // Tick 8 → 2nd wrap, slot 0 ← 90.
        state.step();
        let r = state.read_ring();
        assert!((r[0] - 90.0).abs() < 1e-3,
            "after 9 ticks (2nd wrap): agent0 slot0 = {} (expected 90 = tick-8's amount)",
            r[0]);
        // Slots 1-3 still hold tick 5/6/7 amounts.
        assert!((r[1] - 60.0).abs() < 1e-3);
        assert!((r[2] - 70.0).abs() < 1e-3);
        assert!((r[3] - 80.0).abs() < 1e-3);
        let cursors = state.read_cursors();
        assert_eq!(cursors[0], 9, "agent0 cursor after 9 events");
    }

    /// Plan G G3a-step-3 hardening — agent rings are independent.
    /// Both agents emit Damaged{target=self} per tick (per the .sim's
    /// InjectDamage rule's `target: self`), so each maintains its
    /// OWN cursor + ring. No cross-agent interference.
    ///
    /// Pin: after 4 ticks, agent0's ring AND agent1's ring both
    /// contain [10, 20, 30, 40] (identical sequences but stored
    /// in disjoint memory regions).
    #[test]
    fn agent_rings_are_independent() {
        const N: u32 = 2;
        let mut state = match PerEntityRingProbeState::try_new(0xBEEF, N) {
            Some(s) => s,
            None => {
                eprintln!("[per_entity_ring_probe] skipping independence pin: no wgpu adapter.");
                return;
            }
        };
        for _ in 0..4 {
            state.step();
        }
        let r = state.read_ring();
        // Layout: [a0_s0, a0_s1, a0_s2, a0_s3, a1_s0, a1_s1, a1_s2, a1_s3].
        let agent0_ring = &r[0..4];
        let agent1_ring = &r[4..8];
        let expected = [10.0_f32, 20.0, 30.0, 40.0];
        for (i, e) in expected.iter().enumerate() {
            assert!((agent0_ring[i] - e).abs() < 1e-3,
                "agent0 slot{i} = {} (expected {e})", agent0_ring[i]);
            assert!((agent1_ring[i] - e).abs() < 1e-3,
                "agent1 slot{i} = {} (expected {e})", agent1_ring[i]);
        }
        // Cursor independence — both agents at exactly 4. The buffer
        // is floor-sized at 16 bytes (= 4 u32) so it may report extra
        // trailing zeros for tiny agent_counts; only the first
        // `agent_count` slots are semantically meaningful.
        let cursors = state.read_cursors();
        assert_eq!(&cursors[..N as usize], &[4, 4],
            "cursors must increment per-agent independently; got {cursors:?}");
    }
}


// Plan E-A6 sweep — generator validation smoke test for per_entity_ring_probe_runtime.
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
