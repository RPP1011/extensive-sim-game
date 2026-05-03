//! Per-fixture runtime for `assets/sim/crowd_navigation_min.sim`
//! (Stage 0 staged fixture for the design target at
//! `crowd_navigation.sim`). Mirrors the
//! `predator_prey_runtime` / `particle_collision_runtime` shape:
//! pos+vel storage, agent_cap uniform cfg, on-demand position
//! readback. Stage 0 has a single MoveWalker integrator; later
//! stages copy the per-walker mode-state-machine view, group
//! memberships + leader-driven goal updates, and the utility-
//! backend scoring rows from the design target.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

const EVENT_RING_CAP_SLOTS: u32 = 65_536;
const EVENT_STRIDE_U32: u32 = 10;

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

pub struct CrowdNavigationState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    // ---- Event-ring + fold dispatch state (mirrors pp/pc_runtime) ----
    event_ring_buf: wgpu::Buffer,
    event_tail_buf: wgpu::Buffer,
    event_tail_zero: wgpu::Buffer,
    indirect_args_0_buf: wgpu::Buffer,
    sim_cfg_buf: wgpu::Buffer,
    stuck_count_primary: wgpu::Buffer,
    stuck_count_staging: wgpu::Buffer,
    stuck_count_cfg_buf: wgpu::Buffer,
    stuck_count_cache: Vec<f32>,
    stuck_count_dirty: bool,

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

impl CrowdNavigationState {
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
                normalise(per_agent_u32(seed, agent_id, 0, b"cn_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"cn_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"cn_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"cn_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"cn_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"cn_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("crowd_navigation_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("crowd_navigation_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cfg = physics_MoveWalker::PhysicsMoveWalkerCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("crowd_navigation_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd_navigation_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Event ring infrastructure (mirrors pp/pc_runtime) ----
        let event_ring_bytes =
            (EVENT_RING_CAP_SLOTS as u64) * (EVENT_STRIDE_U32 as u64) * 4;
        let event_ring_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd_navigation_runtime::event_ring"),
            size: event_ring_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let event_tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd_navigation_runtime::event_tail"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let event_tail_zero = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("crowd_navigation_runtime::event_tail_zero"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );
        let indirect_args_0_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd_navigation_runtime::indirect_args_0"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("crowd_navigation_runtime::sim_cfg"),
                contents: &[0u8; 16],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        );
        let view_storage_bytes = (n as u64) * 4;
        let stuck_count_primary = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd_navigation_runtime::stuck_count_primary"),
            size: view_storage_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stuck_count_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd_navigation_runtime::stuck_count_staging"),
            size: view_storage_bytes.max(16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let sc_cfg = fold_stuck_count::FoldStuckCountCfg {
            event_count: 0,
            tick: 0,
            _pad: [0; 2],
        };
        let stuck_count_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("crowd_navigation_runtime::stuck_count_cfg"),
                contents: bytemuck::bytes_of(&sc_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            cfg_buf,
            pos_staging,
            event_ring_buf,
            event_tail_buf,
            event_tail_zero,
            indirect_args_0_buf,
            sim_cfg_buf,
            stuck_count_primary,
            stuck_count_staging,
            stuck_count_cfg_buf,
            stuck_count_cache: vec![0.0; n],
            stuck_count_dirty: false,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
        }
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
                    label: Some("crowd_navigation_runtime::positions::copy"),
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

    /// Per-walker stuck-event count readback. Each MoveWalker emit
    /// fires `Stuck { walker: self, ticks: 1 }` per tick, so each
    /// walker slot increments stuck_count by 1 per tick.
    pub fn stuck_counts(&mut self) -> &[f32] {
        if self.stuck_count_dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("crowd_navigation_runtime::stuck_counts::copy"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.stuck_count_primary,
                0,
                &self.stuck_count_staging,
                0,
                (self.agent_count as u64) * 4,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.stuck_count_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let bytes = slice.get_mapped_range();
            let counts: &[f32] = bytemuck::cast_slice(&bytes);
            self.stuck_count_cache.clear();
            self.stuck_count_cache.extend_from_slice(counts);
            drop(bytes);
            self.stuck_count_staging.unmap();
            self.stuck_count_dirty = false;
        }
        &self.stuck_count_cache
    }
}

impl CompiledSim for CrowdNavigationState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("crowd_navigation_runtime::step"),
            });

        // Clear event_tail before producers run.
        encoder.copy_buffer_to_buffer(
            &self.event_tail_zero,
            0,
            &self.event_tail_buf,
            0,
            4,
        );

        // (1) MoveWalker — emits Stuck { walker: self, ticks: 1 }
        // each tick. event_ring + event_tail bindings synthesized
        // by the binding-gen for any EventRing access.
        let bindings = physics_MoveWalker::PhysicsMoveWalkerBindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            agent_pos: &self.pos_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_movewalker(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            indirect_args_0: &self.indirect_args_0_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // event_count = agent_count (every walker emits exactly one
        // Stuck per tick).
        let sc_cfg = fold_stuck_count::FoldStuckCountCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.stuck_count_cfg_buf,
            0,
            bytemuck::bytes_of(&sc_cfg),
        );

        // (3) fold_stuck_count — RMW view_storage_primary by 1.0
        // per Stuck event whose `walker` matches the slot.
        let sc_bindings = fold_stuck_count::FoldStuckCountBindings {
            event_ring: &self.event_ring_buf,
            event_tail: &self.event_tail_buf,
            view_storage_primary: &self.stuck_count_primary,
            view_storage_anchor: None, // stuck_count has no @decay
            view_storage_ids: None,
            sim_cfg: &self.sim_cfg_buf,
            cfg: &self.stuck_count_cfg_buf,
        };
        dispatch::dispatch_fold_stuck_count(
            &mut self.cache,
            &sc_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.stuck_count_dirty = true;
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
    Box::new(CrowdNavigationState::new(seed, agent_count))
}
