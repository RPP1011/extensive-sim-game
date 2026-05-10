//! Plan G G3 final composition — threats_struct_probe_runtime.
//!
//! Drives `assets/sim/threats_struct_probe.sim` end-to-end on real wgpu
//! hardware. The .sim composes:
//!   * G3a/b/c — `@per_entity_ring(K = 4)` storage with struct-payload
//!     fold body via `self.append(...)` (multi-statement body with `let`
//!     prelude bindings).
//!   * G3d — `@dispatch(per_agent_event_scan)` per-(observer, source_candidate)
//!     iteration with busy-filter early-exit on the source.
//!
//! The fold's emit (extended in `cg/emit/kernel.rs::build_view_fold_per_agent_event_scan_body`
//! to detect a registered `ViewLayout` for the view) writes the 8-field
//! ThreatZoneCell at the observer's ring slot per qualifying source
//! candidate.
//!
//! ## Behavioural pin
//!
//! With `AGENT_COUNT=4`, `cast_begin_tick=0`:
//!
//!   Tick 0:
//!     * MarkCasterBusy fires (every alive agent's busy_with_ability_id
//!       becomes 7).
//!     * fold_threats dispatches 4×4 pairs. All 4 source candidates pass
//!       the busy-filter, so each observer's ring receives 4 cells (cursor
//!       advances 0→4).
//!     * Cells contain (in u32 packed form):
//!         zone_kind = 1 (config.probe.zone_kind)
//!         radius_q8 = 1024 (config.probe.zone_radius_q8 = 4.0 q8)
//!         expires_at_tick = world.tick + 100 = 100
//!         center/dir/source = 0 (MVP placeholders; see fixture header
//!                              for the gap-tracking on real reads).
//!
//!   Tick 1:
//!     * MarkCasterBusy where-clause fails (`world.tick != 0`); busy state
//!       unchanged.
//!     * fold_threats fires again, appending 4 more cells per observer.
//!       Cursor → 8; ring slots 0..4 are overwritten by new cells (% 4).
//!
//! The load-bearing pin: after tick 0, observer 0's first ring cell has
//! `zone_kind == 1` and `radius_q8 == 1024`, proving the struct-payload
//! ring-append fired correctly under the PerAgentEventScan dispatch.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-agent ring depth — matches `assets/sim/threats_struct_probe.sim`'s
/// `@per_entity_ring(K = 4)`.
const RING_K: u32 = 4;

/// Per-cell stride in u32 words — matches the design doc's 8-field
/// `ThreatZoneCell` (each field stored at one u32 word).
const CELL_STRIDE_U32: u32 = 8;

pub struct ThreatsStructProbeState {
    gpu: GpuContext,
    agent_alive_buf: wgpu::Buffer,
    /// `busy_with_ability_id` SoA — written by MarkCasterBusy; read by
    /// the fold's busy-filter early-exit.
    agent_busy_with_ability_id_buf: wgpu::Buffer,

    /// Per-cell ring storage. Sized `agent_count * K * CELL_STRIDE_U32`
    /// u32s, zero-init. Each cell is the 8-field ThreatZoneCell;
    /// per-cell layout is `[zone_kind, center_x_q8, center_y_q8,
    /// radius_q8, dir_x_q8, dir_y_q8, expires_at_tick, source]`.
    view_storage_primary_buf: wgpu::Buffer,
    /// Per-observer cursor counter (atomic). `agent_count` u32 slots.
    view_storage_anchor_buf: wgpu::Buffer,
    /// Staging buffer for primary readback.
    view_storage_primary_staging: wgpu::Buffer,
    /// Staging buffer for cursor readback.
    view_storage_anchor_staging: wgpu::Buffer,

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

impl ThreatsStructProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::try_new(seed, agent_count).expect("init wgpu adapter + device")
    }

    pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {
        let gpu = GpuContext::new_blocking().ok()?;

        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threats_struct_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let busy_init: Vec<u32> = vec![0u32; agent_count as usize];
        let agent_busy_with_ability_id_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threats_struct_probe::agent_busy_with_ability_id"),
                contents: bytemuck::cast_slice(&busy_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        // Per-cell ring storage: agent_count * K cells * 8 u32 fields.
        // Floor at 16 bytes for tiny tests so wgpu accepts the binding.
        let primary_bytes = ((agent_count as u64)
            * (RING_K as u64)
            * (CELL_STRIDE_U32 as u64)
            * 4)
            .max(16);
        let primary_init: Vec<u32> = vec![0u32; (primary_bytes / 4) as usize];
        let view_storage_primary_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_struct_probe::view_storage_primary"),
                contents: bytemuck::cast_slice(&primary_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Cursors: agent_count atomic u32 slots, zero-init.
        let anchor_bytes = ((agent_count as u64) * 4).max(16);
        let anchor_init: Vec<u32> = vec![0u32; (anchor_bytes / 4) as usize];
        let view_storage_anchor_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_struct_probe::view_storage_anchor"),
                contents: bytemuck::cast_slice(&anchor_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let view_storage_primary_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threats_struct_probe::view_storage_primary_staging"),
            size: primary_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let view_storage_anchor_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threats_struct_probe::view_storage_anchor_staging"),
            size: anchor_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let event_ring = EventRing::new(&gpu, "threats_struct_probe");

        let physics_cfg_init = physics_MarkCasterBusy::PhysicsMarkCasterBusyCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_struct_probe::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // PerAgentEventScan reuses the ViewFold cfg's `event_count` field
        // as agent_cap (per the cfg-shape note in
        // `build_view_fold_per_agent_event_scan_body`). Runtime sets it
        // to agent_count.
        let fold_cfg_init = fold_threats::FoldThreatsCfg {
            event_count: agent_count,
            tick: 0, second_key_pop: 1, _pad: 0,
        };
        let fold_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threats_struct_probe::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &PackedAbilityRegistry::pack(&engine::ability::AbilityRegistry::new()),
            &gpu,
            "threats_struct_probe",
        );

        Some(Self {
            gpu,
            agent_alive_buf,
            agent_busy_with_ability_id_buf,
            view_storage_primary_buf,
            view_storage_anchor_buf,
            view_storage_primary_staging,
            view_storage_anchor_staging,
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

    /// Read the per-cell ring storage as `agent_count * K * CELL_STRIDE_U32`
    /// u32 words. Layout: `cells[agent * K * stride + slot * stride + field]`.
    pub fn read_cells(&self) -> Vec<u32> {
        let primary_bytes = ((self.agent_count as u64)
            * (RING_K as u64)
            * (CELL_STRIDE_U32 as u64)
            * 4)
            .max(16);
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threats_struct_probe::read_cells"),
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
            .expect("device poll failed during cells readback");
        let out = {
            let view = slice.get_mapped_range();
            let ints: &[u32] = bytemuck::cast_slice(&view);
            ints.to_vec()
        };
        self.view_storage_primary_staging.unmap();
        out
    }

    pub fn read_cursors(&self) -> Vec<u32> {
        let anchor_bytes = ((self.agent_count as u64) * 4).max(16);
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threats_struct_probe::read_cursors"),
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

    /// Pull the 8-field cell at `(agent, slot)` as a `[u32; 8]` for
    /// readable assertions. Layout matches the design doc's
    /// ThreatZoneCell — fields in declaration order.
    pub fn cell(&self, cells: &[u32], agent: u32, slot: u32) -> [u32; 8] {
        let base = (agent * RING_K * CELL_STRIDE_U32 + slot * CELL_STRIDE_U32) as usize;
        let mut out = [0u32; 8];
        for i in 0..8 {
            out[i] = cells[base + i];
        }
        out
    }
}

impl CompiledSim for ThreatsStructProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threats_struct_probe::step"),
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

        // (1) MarkCasterBusy — sets agent_busy_with_ability_id at
        // cast_begin_tick=0 on every alive agent.
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

        // (2) fold_threats — PerAgentEventScan over (observer,
        // source_candidate). Busy-filter early-exits non-busy sources;
        // qualifying pairs ring-append a struct cell at observer's slot.
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
            view_storage_primary: &self.view_storage_primary_buf,
            view_storage_anchor: Some(&self.view_storage_anchor_buf),
            view_storage_ids: None,
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
    Box::new(ThreatsStructProbeState::new(seed, agent_count))
}

#[cfg(test)]
mod threats_struct_lifecycle_tests {
    use super::*;

    /// Per-cell field indices into the 8-u32 ThreatZoneCell. Matches
    /// `assets/sim/threats_struct_probe.sim`'s `self.append(...)` field
    /// order which the lowering / emit preserve verbatim.
    const F_ZONE_KIND: usize = 0;
    const F_CENTER_X: usize = 1;
    const F_CENTER_Y: usize = 2;
    const F_RADIUS_Q8: usize = 3;
    const F_DIR_X: usize = 4;
    const F_DIR_Y: usize = 5;
    const F_EXPIRES_AT_TICK: usize = 6;
    const F_SOURCE: usize = 7;

    /// Plan G G3 final composition — load-bearing GPU pin.
    ///
    /// Asserts the struct-payload PerAgentEventScan fold runs end-to-end
    /// on real wgpu and the per-(observer, source_candidate) cells carry
    /// the expected 8-field metadata. After tick 0:
    ///   * Every observer's cursor advanced by N (= 4 busy candidates).
    ///   * Each cell's zone_kind == 1, radius_q8 == 1024,
    ///     expires_at_tick == 100 (tick 0 + 100 duration).
    ///   * Center / dir / source are MVP placeholder zeros (real reads
    ///     surface in follow-up gaps b/c/d documented in fixture
    ///     header).
    #[test]
    fn struct_cells_populate_per_busy_pair() {
        const N: u32 = 4;
        let mut state = match ThreatsStructProbeState::try_new(0xCAFE, N) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[threats_struct_probe] skipping: no wgpu adapter on host. \
                     Build still validated emit + bindings at compile time."
                );
                return;
            }
        };

        // Tick 0 — MarkCasterBusy fires (every agent becomes busy);
        // fold_threats then runs with all 4 sources busy and writes
        // 4 cells per observer.
        state.step();

        let cursors = state.read_cursors();
        for (obs, &c) in cursors.iter().take(N as usize).enumerate() {
            assert_eq!(
                c, N,
                "tick 0: observer {obs} cursor = {c} (expected {N} = N busy candidates)",
            );
        }

        let cells = state.read_cells();
        // Sanity on overall buffer length.
        let expected_len = (N * RING_K * CELL_STRIDE_U32) as usize;
        assert_eq!(cells.len(), expected_len,
            "per-cell buffer length mismatch: got {} u32s, expected {}", cells.len(), expected_len);

        // Inspect every observer's slot 0 — must carry the expected
        // ThreatZoneCell metadata.
        for obs in 0..N {
            let c = state.cell(&cells, obs, 0);
            eprintln!("[threats_struct] observer {obs} slot 0: {c:?}");
            assert_eq!(c[F_ZONE_KIND], 1,
                "observer {obs}: zone_kind = {} (expected 1 = config.probe.zone_kind)", c[F_ZONE_KIND]);
            assert_eq!(c[F_CENTER_X], 0, "observer {obs}: center_x_q8 = {} (MVP placeholder = 0)", c[F_CENTER_X]);
            assert_eq!(c[F_CENTER_Y], 0, "observer {obs}: center_y_q8 = {} (MVP placeholder = 0)", c[F_CENTER_Y]);
            assert_eq!(c[F_RADIUS_Q8], 1024,
                "observer {obs}: radius_q8 = {} (expected 1024 = config.probe.zone_radius_q8)", c[F_RADIUS_Q8]);
            assert_eq!(c[F_DIR_X], 0, "observer {obs}: dir_x_q8 placeholder");
            assert_eq!(c[F_DIR_Y], 0, "observer {obs}: dir_y_q8 placeholder");
            assert_eq!(c[F_EXPIRES_AT_TICK], 100,
                "observer {obs}: expires_at_tick = {} (expected 100 = tick 0 + duration 100)",
                c[F_EXPIRES_AT_TICK]);
            assert_eq!(c[F_SOURCE], 0,
                "observer {obs}: source = {} (MVP placeholder = 0; real read needs source_candidate binding)", c[F_SOURCE]);
        }

        // Tick 1 — busy state preserved, fold runs again, ring wraps:
        // cursor 4..7 land at slots (4..7) % 4 = 0..3 ⇒ all slots
        // overwritten. expires_at_tick now = world.tick(1) + 100 = 101.
        state.step();
        let cursors = state.read_cursors();
        for (obs, &c) in cursors.iter().take(N as usize).enumerate() {
            assert_eq!(
                c,
                N * 2,
                "tick 1: observer {obs} cursor = {c} (expected {} after 2 ticks)",
                N * 2,
            );
        }
        let cells = state.read_cells();
        for obs in 0..N {
            let c = state.cell(&cells, obs, 0);
            assert_eq!(c[F_EXPIRES_AT_TICK], 101,
                "tick 1: observer {obs} slot 0 expires_at_tick = {} (expected 101 = tick 1 + 100; ring wrapped)",
                c[F_EXPIRES_AT_TICK]);
            assert_eq!(c[F_RADIUS_Q8], 1024,
                "tick 1: observer {obs} radius_q8 unchanged at 1024");
        }
    }
}
