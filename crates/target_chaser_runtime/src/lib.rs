//! Per-fixture runtime for `assets/sim/target_chaser.sim` —
//! stress fixture for slice 1 of stdlib-into-CG-IR.
//!
//! Each agent stores a `target` AgentId in the `engaged_with`
//! SoA slot (initialized to `(slot+1) mod cap` — a deterministic
//! ring of "follow next"). The compiler-emitted physics kernel
//! reads `agents.pos(self.engaged_with)` via slice 1's stmt-scope
//! `let target_expr_<N>` hoist, so each agent steers toward the
//! position of the agent it points at. Over time the chasers form
//! a ring orbit (each pulled toward the next), the test that the
//! cross-agent target read actually moves bytes through the SoA
//! at runtime.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

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

pub struct TargetChaserState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-agent `alive` u32 (1 = alive, 0 = dead). Required by the
    /// physics rule's `where (self.alive)` guard.
    alive_buf: wgpu::Buffer,
    /// Per-agent target slot — `array<u32>` of length `agent_cap`.
    /// Each slot holds the AgentId of the agent this slot chases.
    /// Initialized to `(slot + 1) mod cap` so chasers form a ring.
    /// This is the `engaged_with` field on the canonical Agent SoA;
    /// the kernel binds it as `agent_engaged_with` and reads through
    /// it via slice 1's `let target_expr_<N>: u32 = agent_engaged_with[
    /// agent_id];` hoist.
    engaged_with_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

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

impl TargetChaserState {
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
                normalise(per_agent_u32(seed, agent_id, 0, b"tc_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"tc_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"tc_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"tc_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"tc_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"tc_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }

        // Hub-and-spokes topology: slot 0 targets itself (so it
        // stays put — pull-toward-self is the zero vector and
        // damping decays its initial velocity); every other slot
        // targets slot 0. With this layout the system converges
        // cleanly: 31 damped chasers pull toward a fixed point.
        // A cyclic ring (slot i → (i+1) mod cap) instead produces
        // tangential pursuit and unbounded centripetal motion —
        // less useful as a slice-1 verification observable, since
        // "did the chase work?" gets confounded with "did the
        // dynamics stabilize?". Stored as raw u32 (matching how
        // the WGSL binding sees it: `agent_engaged_with: array<u32>`).
        let engaged_with_init: Vec<u32> = vec![0u32; n];
        // Per-agent alive flag — every agent starts alive.
        let alive_init: Vec<u32> = vec![1u32; n];

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target_chaser_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target_chaser_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target_chaser_runtime::alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let engaged_with_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("target_chaser_runtime::engaged_with"),
                contents: bytemuck::cast_slice(&engaged_with_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let cfg = physics_ChaseTarget::PhysicsChaseTargetCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target_chaser_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("target_chaser_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            gpu,
            pos_buf,
            vel_buf,
            alive_buf,
            engaged_with_buf,
            cfg_buf,
            pos_staging,
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
                    label: Some("target_chaser_runtime::positions::copy"),
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

impl CompiledSim for TargetChaserState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("target_chaser_runtime::step"),
            });

        let bindings = physics_ChaseTarget::PhysicsChaseTargetBindings {
            agent_pos: &self.pos_buf,
            agent_alive: &self.alive_buf,
            agent_engaged_with: &self.engaged_with_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_chasetarget(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
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
    Box::new(TargetChaserState::new(seed, agent_count))
}
