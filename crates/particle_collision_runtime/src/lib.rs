//! Per-fixture runtime for `assets/sim/particle_collision_min.sim`
//! (the Stage-0 staged fixture for the design target at
//! `particle_collision.sim`). Mirrors `predator_prey_runtime`'s
//! shape — pos+vel storage, agent_cap uniform cfg, on-demand
//! position readback. Stage 0 has a single MoveParticle integrator;
//! later stages copy the per-pair Collision detection + view-fold
//! impulse accumulator from the design target.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
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

impl From<Vec3Padded> for Vec3 {
    fn from(p: Vec3Padded) -> Self {
        Vec3::new(p.x, p.y, p.z)
    }
}

pub struct ParticleCollisionState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    /// Per-fixture event-ring infrastructure (event_ring +
    /// event_tail + tail-zero source + indirect_args + sim_cfg
    /// placeholder). Shared via [`engine::gpu::EventRing`].
    event_ring: EventRing,
    /// Per-particle collision-count accumulator. The
    /// `fold_collision_count` kernel RMWs
    /// `view_storage_primary[a_id]` AND `…[b_id]` by 1.0 per
    /// Collision event; readback via [`Self::collision_counts`].
    collision_count: ViewStorage,
    collision_count_cfg_buf: wgpu::Buffer,

    // ---- Spatial-grid (slice 2b body-form spatial query) -------------
    /// Sorted-by-cell agent ids — `array<u32>` sized `agent_cap`. After
    /// the five-phase counting sort populates this, cell `c`'s agent
    /// ids occupy `[spatial_grid_starts[c] .. spatial_grid_starts[c +
    /// 1])`. Mirrors the boids spatial layout — same WGSL constants
    /// from `dsl_compiler::cg::emit::spatial`, no per-fixture knobs.
    spatial_grid_cells: wgpu::Buffer,
    /// Per-cell atomic counters. Triple-duty across counting sort.
    spatial_grid_offsets: wgpu::Buffer,
    /// Per-cell start offsets after the prefix scan — `array<u32>`
    /// sized `num_cells + 1`. Read by the body-form spatial walk emit
    /// (`spatial_grid_starts[_cell]`/`[+1]`) to bound each cell's
    /// candidate slice.
    spatial_grid_starts: wgpu::Buffer,
    /// Per-workgroup-chunk total used by the parallel prefix scan.
    spatial_chunk_sums: wgpu::Buffer,
    /// Pre-allocated zero buffer used as the COPY_SRC for the per-tick
    /// offsets clear.
    spatial_offsets_zero: wgpu::Buffer,

    cache: dispatch::KernelCache,
    pos_cache: Vec<Vec3>,
    dirty: bool,
    tick: u64,
    agent_count: u32,
    seed: u64,
}

fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl ParticleCollisionState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let spread = (agent_count as f32).cbrt().max(1.0);

        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id =
                AgentId::new(slot + 1).expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"pc_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"pc_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"pc_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"pc_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"pc_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"pc_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_collision_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_collision_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cfg = physics_MoveParticle::PhysicsMoveParticleCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_collision_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_collision_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Shared event-ring + per-view storage helpers ----
        let event_ring = EventRing::new(&gpu, "particle_collision_runtime");
        let collision_count = ViewStorage::new(
            &gpu,
            "particle_collision_runtime::collision_count",
            agent_count,
            false,
            false,
        );
        let cc_cfg = fold_collision_count::FoldCollisionCountCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let collision_count_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("particle_collision_runtime::collision_count_cfg"),
                contents: bytemuck::bytes_of(&cc_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // ---- Spatial-grid buffers (slice 2b body-form spatial query) -
        // Constants live in `dsl_compiler::cg::emit::spatial` so the
        // host-side allocation matches the WGSL `const` declarations
        // the build.rs emit produces. Same layout boids_runtime uses:
        // bounded counting sort, no per-cell cap on `cells`,
        // `starts[c..c+1]` slices the cell's agent-id range.
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (agent_count as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_collision_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_collision_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_collision_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = sp::num_cells().div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_collision_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle_collision_runtime::spatial_offsets_zero"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        Self {
            gpu,
            pos_buf,
            vel_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            collision_count,
            collision_count_cfg_buf,
            spatial_grid_cells,
            spatial_grid_offsets,
            spatial_grid_starts,
            spatial_chunk_sums,
            spatial_offsets_zero,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-particle collision-count accumulator readback.
    ///
    /// Stage 1 (slice 2b): the body-form spatial walk in
    /// `MoveParticle` emits one `Collision { a: self, b: other,
    /// impulse }` event per (self, other) candidate slot in the
    /// 27-cell neighbourhood. Each event hits BOTH the
    /// `on Collision { a: agent }` and `on Collision { b: agent }`
    /// handlers, incrementing `collision_count[a]` AND
    /// `collision_count[b]` by 1.0. With the auto-spread the
    /// initial particles are dispersed so the per-cell density is
    /// ~1; the per-tick total is therefore approximately
    /// `2 * (sum over self of cells_in_neighbourhood)` which
    /// scales with N (linear in agent count) rather than N²
    /// (full N² walk) or N (the prior placeholder per-self emit).
    /// See pc_app.rs for the analytical derivation pinned at the
    /// fixture's seeded layout.
    pub fn collision_counts(&mut self) -> &[f32] {
        self.collision_count.readback(&self.gpu)
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("particle_collision_runtime::positions::copy"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.pos_buf,
                0,
                &self.pos_staging,
                0,
                (self.agent_count as u64) * std::mem::size_of::<Vec3Padded>() as u64,
            );
            self.gpu.queue.submit(Some(encoder.finish()));

            let slice = self.pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");

            let bytes = slice.get_mapped_range();
            let padded: &[Vec3Padded] = bytemuck::cast_slice(&bytes);
            for (cache, p) in self.pos_cache.iter_mut().zip(padded.iter()) {
                *cache = (*p).into();
            }
            drop(bytes);
            self.pos_staging.unmap();
            self.dirty = false;
        }
        &self.pos_cache
    }
}

impl CompiledSim for ParticleCollisionState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_collision_runtime::step"),
            });

        // Clear event_tail + spatial offsets before producers run.
        self.event_ring.clear_tail_in(&mut encoder);
        let offsets_size = dsl_compiler::cg::emit::spatial::offsets_bytes();
        encoder.copy_buffer_to_buffer(
            &self.spatial_offsets_zero,
            0,
            &self.spatial_grid_offsets,
            0,
            offsets_size,
        );

        // (1) Spatial-hash counting sort (5 phases). Required input
        // for the body-form spatial walk emitted into MoveParticle.
        // Same layout as boids — see boids_runtime::step for the
        // fixture-agnostic narrative.
        let count_b = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.cfg_buf,
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
            cfg: &self.cfg_buf,
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
            cfg: &self.cfg_buf,
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
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_add(
            &mut self.cache,
            &scan_add_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scatter_b = spatial_build_hash_scatter::SpatialBuildHashScatterBindings {
            agent_pos: &self.pos_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache,
            &scatter_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) MoveParticle — body-form spatial walk emits one Collision
        // event per (self, other) candidate slot in the 27-cell
        // neighbourhood. `agent_pos` is read-write here because the
        // physics rule both reads it (for the walk's self-cell
        // computation) and writes the integrated position before the
        // walk.
        let bindings = physics_MoveParticle::PhysicsMoveParticleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_vel: &self.vel_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_moveparticle(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // event_count = static upper bound covering the per-pair
        // emit. Each particle's body-form walk visits up to
        // (27 cells × MAX_PER_CELL) candidates; emitting
        // `agent_count * MAX_PER_CELL * 27` covers the worst case.
        // The kernel's `if (event_idx >= cfg.event_count) return;`
        // gate skips empty event-ring slots beyond the actual tail.
        // The DEFAULT_EVENT_RING_CAP_SLOTS is 65536 so as long as
        // the upper bound stays below that the ring never overflows.
        use dsl_compiler::cg::emit::spatial as sp;
        let event_count = std::cmp::min(
            self.agent_count.saturating_mul(sp::MAX_PER_CELL).saturating_mul(27),
            65536,
        );
        let cc_cfg = fold_collision_count::FoldCollisionCountCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.collision_count_cfg_buf,
            0,
            bytemuck::bytes_of(&cc_cfg),
        );

        // (4) fold_collision_count — RMW view_storage_primary by
        // 1.0 per Collision event whose `a` OR `b` matches the
        // slot.
        let cc_bindings = fold_collision_count::FoldCollisionCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.collision_count.primary(),
            view_storage_anchor: self.collision_count.anchor(),
            view_storage_ids: self.collision_count.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.collision_count_cfg_buf,
        };
        dispatch::dispatch_fold_collision_count(
            &mut self.cache,
            &cc_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.collision_count.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        self.read_positions()
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(ParticleCollisionState::new(seed, agent_count))
}
