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
            _pad: [0; 2],
        };
        let collision_count_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("particle_collision_runtime::collision_count_cfg"),
                contents: bytemuck::bytes_of(&cc_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            collision_count,
            collision_count_cfg_buf,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-particle collision-count accumulator readback. Each
    /// MoveParticle emit fires `Collision { a: self, b: self,
    /// impulse }` so each Particle slot increments its own
    /// collision_count by 2 per tick (the view's two `on Collision
    /// { a/b: agent } { self += 1.0 }` handlers BOTH match when
    /// a == b == self).
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

        // Clear event_tail before producers run.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1) MoveParticle — emits one Collision event per particle
        // per tick.
        let bindings = physics_MoveParticle::PhysicsMoveParticleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_moveparticle(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) seed_indirect_0 — keeps indirect-args buffer warm.
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

        // event_count = agent_count (one Collision per particle).
        let cc_cfg = fold_collision_count::FoldCollisionCountCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.collision_count_cfg_buf,
            0,
            bytemuck::bytes_of(&cc_cfg),
        );

        // (3) fold_collision_count — RMW view_storage_primary by
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
            self.agent_count,
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
