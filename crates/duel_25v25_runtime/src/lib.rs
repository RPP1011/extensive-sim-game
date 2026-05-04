//! Per-fixture runtime for `assets/sim/duel_25v25.sim` — first
//! SCALE-UP fixture (25 Red vs 25 Blue squad scuffle). Bridges
//! `particle_collision_runtime`'s spatial-grid scaffolding with
//! `duel_1v1_runtime`'s chronicle damage path.
//!
//! Per-tick chain (mirrors particle_collision_runtime + duel_1v1):
//!
//!   1. clear_tail + clear ring headers + spatial offsets clear
//!   2. spatial_build_hash (5 phases): count → scan_local → scan_carry
//!      → scan_add → scatter
//!   3. ScanAndStrike — body-form spatial walk emits Damaged per
//!      neighbour in range whose creature_type opposes ours
//!      (gated every 2 ticks by per-handler `where`).
//!   4. ApplyDamage — chronicle physics reads Damaged, writes
//!      target HP, sets alive=false on HP<=0, emits Defeated.
//!   5. seed_indirect_0 (keeps args buffer warm)
//!   6. fold_damage_dealt (per-source f32 accumulator)
//!   7. fold_defeats_received (per-target f32 count)
//!
//! ## Init layout
//!
//! 50 agent slots: even slots (0, 2, …, 48) → RedCombatant
//! (creature_type=0), odd slots (1, 3, …, 49) → BlueCombatant
//! (creature_type=1). 25 of each. Position split across x=0:
//! - Red: x ∈ [-2.0, 0.0), y=z=0.0
//! - Blue: x ∈ [0.0, 2.0], y=z=0.0
//!
//! At spatial_radius=1.5 (matches @spatial annotation), most
//! Combatants near the seam find at least one enemy neighbour
//! per tick.
//!
//! Initial HP = 50.0 (lower than 1v1's 100.0 so the battle ends
//! faster). Strike damage = 5.0; with cooldown 2 ticks and ~1-3
//! enemy neighbours per Combatant in the contested zone, an
//! agent typically takes ~5-15 dmg per active tick.

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

/// Per-fixture state for the 25v25 duel.
pub struct Duel25v25State {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_pos_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_creature_type_buf: wgpu::Buffer,

    // -- Spatial grid --
    spatial_grid_cells: wgpu::Buffer,
    spatial_grid_offsets: wgpu::Buffer,
    spatial_grid_starts: wgpu::Buffer,
    spatial_chunk_sums: wgpu::Buffer,
    spatial_offsets_zero: wgpu::Buffer,

    // -- Event ring + view storage --
    event_ring: EventRing,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,
    defeats_received: ViewStorage,
    defeats_received_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    scan_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl Duel25v25State {
    /// Construct a 50-agent battlefield (25 Red + 25 Blue) with the
    /// position layout described in the crate docs.
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert_eq!(
            agent_count, 50,
            "duel_25v25 expects exactly 50 agents (25 Red + 25 Blue); got {agent_count}",
        );
        let n = agent_count as usize;

        // Position layout: split at x=0. Red on left (x in [-2,0)),
        // Blue on right (x in [0,2]). Slight y-jitter per slot so
        // multiple Combatants on the same x get distributed cells.
        // The spatial-grid cell size lives in dsl_compiler::cg::emit
        // ::spatial; for the default radius we don't need to read it,
        // a unit-spread layout is sufficient.
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut hp_init: Vec<f32> = Vec::with_capacity(n);
        let mut alive_init: Vec<u32> = Vec::with_capacity(n);
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);

        // 25 Red (even slots) + 25 Blue (odd slots). Lay them out in
        // a 5x5 grid each, with y∈[-2,2] and z=0. Red slot k → grid
        // (k/5, k%5) at x in [-2, -0.4]; Blue slot k → similar at
        // x in [0.4, 2.0].
        for slot in 0..agent_count {
            let is_red = slot % 2 == 0;
            let team_index = slot / 2; // 0..25
            let row = (team_index / 5) as f32; // 0..5
            let col = (team_index % 5) as f32; // 0..5
            let y = (row - 2.0) * 0.8; // -1.6..1.6
            let z = (col - 2.0) * 0.8; // -1.6..1.6
            let x = if is_red {
                -2.0 + (col * 0.4) // -2.0..-0.4
            } else {
                0.4 + (col * 0.4) // 0.4..2.0
            };
            pos_padded.push(Vec3::new(x, y, z).into());
            hp_init.push(50.0_f32);
            alive_init.push(1u32);
            creature_init.push(if is_red { 0u32 } else { 1u32 });
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let agent_pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_25v25_runtime::agent_pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_25v25_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_25v25_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("duel_25v25_runtime::agent_creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Spatial-grid buffers (mirror particle_collision_runtime) ----
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (agent_count as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_25v25_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_25v25_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_25v25_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = sp::num_cells().div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_25v25_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("duel_25v25_runtime::spatial_offsets_zero"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Event ring + view storage ----
        let event_ring = EventRing::new(&gpu, "duel_25v25_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "duel_25v25_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );
        let defeats_received = ViewStorage::new(
            &gpu,
            "duel_25v25_runtime::defeats_received",
            agent_count,
            false,
            false,
        );

        // ---- Per-kernel cfg uniforms ----
        let scan_cfg = physics_ScanAndStrike::PhysicsScanAndStrikeCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scan_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_25v25_runtime::scan_cfg"),
            contents: bytemuck::bytes_of(&scan_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_25v25_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_25v25_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let damage_dealt_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("duel_25v25_runtime::damage_dealt_cfg"),
                contents: bytemuck::bytes_of(&damage_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let defeats_cfg = fold_defeats_received::FoldDefeatsReceivedCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let defeats_received_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("duel_25v25_runtime::defeats_received_cfg"),
                contents: bytemuck::bytes_of(&defeats_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_pos_buf,
            agent_hp_buf,
            agent_alive_buf,
            agent_creature_type_buf,
            spatial_grid_cells,
            spatial_grid_offsets,
            spatial_grid_starts,
            spatial_chunk_sums,
            spatial_offsets_zero,
            event_ring,
            damage_dealt,
            damage_dealt_cfg_buf,
            defeats_received,
            defeats_received_cfg_buf,
            scan_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-source damage_dealt readback (one f32 per agent slot).
    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    /// Per-target defeats_received readback (one f32 per agent slot).
    pub fn defeats_received(&mut self) -> &[f32] {
        self.defeats_received.readback(&self.gpu)
    }

    /// Per-agent HP readback (allocates a staging buffer + maps).
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    /// Per-agent alive readback (1 = alive, 0 = dead).
    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    /// Per-agent creature_type readback (0 = Red, 1 = Blue).
    pub fn read_creature_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_creature_type_buf, "creature_type")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("duel_25v25_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("duel_25v25_runtime::read_f32"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_u32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("duel_25v25_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("duel_25v25_runtime::read_u32"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
}

impl CompiledSim for Duel25v25State {
    fn step(&mut self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("duel_25v25_runtime::step"),
            });

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // ScanAndStrike per agent emits up to ~27 cells × MAX_PER_CELL
        // candidates; ApplyDamage may fan-out one Defeated per Damaged.
        // Bound headers clear above the worst-case slots produced per
        // tick.
        use dsl_compiler::cg::emit::spatial as sp;
        let max_neighbour_emits = self
            .agent_count
            .saturating_mul(sp::MAX_PER_CELL)
            .saturating_mul(27);
        let max_slots_per_tick = max_neighbour_emits.saturating_mul(2).min(65536);
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, max_slots_per_tick);
        let offsets_size = sp::offsets_bytes();
        encoder.copy_buffer_to_buffer(
            &self.spatial_offsets_zero,
            0,
            &self.spatial_grid_offsets,
            0,
            offsets_size,
        );

        // (2) Spatial-hash counting sort (5 phases). Mirrors
        // particle_collision_runtime; required input for ScanAndStrike's
        // body-form spatial walk.
        let scan_cfg = physics_ScanAndStrike::PhysicsScanAndStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.scan_cfg_buf, 0, bytemuck::bytes_of(&scan_cfg));

        let count_b = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.scan_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_count(
            &mut self.cache,
            &count_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_local_b = spatial_build_hash_scan_local::SpatialBuildHashScanLocalBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.scan_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_local(
            &mut self.cache,
            &scan_local_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_carry_b = spatial_build_hash_scan_carry::SpatialBuildHashScanCarryBindings {
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.scan_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_carry(
            &mut self.cache,
            &scan_carry_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_add_b = spatial_build_hash_scan_add::SpatialBuildHashScanAddBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.scan_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_add(
            &mut self.cache,
            &scan_add_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scatter_b = spatial_build_hash_scatter::SpatialBuildHashScatterBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.scan_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache,
            &scatter_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) ScanAndStrike — body-form spatial walk emits Damaged.
        let scan_bindings = physics_ScanAndStrike::PhysicsScanAndStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.scan_cfg_buf,
        };
        dispatch::dispatch_physics_scanandstrike(
            &mut self.cache,
            &scan_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) ApplyDamage — chronicle physics, PerEvent over Damaged.
        // Reads Damaged, writes agent_hp + agent_alive, may emit
        // Defeated. event_count is the upper bound on Damaged events
        // produced per tick (one per ScanAndStrike emit). Over-
        // provision is safe — the kernel's per-handler tag check
        // ignores foreign kinds.
        let event_count_estimate = max_neighbour_emits.min(65536);
        let apply_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            _pad0: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg));
        let apply_bindings = physics_ApplyDamage::PhysicsApplyDamageBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage(
            &mut self.cache,
            &apply_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (5) seed_indirect_0 — keeps args buffer warm.
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

        // (6) fold_damage_dealt — RMW per Damaged event.
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_dealt_cfg_buf,
            0,
            bytemuck::bytes_of(&damage_cfg),
        );
        let damage_bindings = fold_damage_dealt::FoldDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_dealt.primary(),
            view_storage_anchor: self.damage_dealt.anchor(),
            view_storage_ids: self.damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_damage_dealt(
            &mut self.cache,
            &damage_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (7) fold_defeats_received — RMW per Defeated event.
        let defeats_cfg = fold_defeats_received::FoldDefeatsReceivedCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.defeats_received_cfg_buf,
            0,
            bytemuck::bytes_of(&defeats_cfg),
        );
        let defeats_bindings = fold_defeats_received::FoldDefeatsReceivedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.defeats_received.primary(),
            view_storage_anchor: self.defeats_received.anchor(),
            view_storage_ids: self.defeats_received.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.defeats_received_cfg_buf,
        };
        dispatch::dispatch_fold_defeats_received(
            &mut self.cache,
            &defeats_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.defeats_received.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(Duel25v25State::new(seed, agent_count))
}
