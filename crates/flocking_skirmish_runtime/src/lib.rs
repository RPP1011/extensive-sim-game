//! Per-fixture runtime for `assets/sim/flocking_skirmish.sim` —
//! THIRTEENTH real fixture and the first to bring boids dynamics into
//! a combat context.
//!
//! See `crates/predator_prey_runtime/src/lib.rs` for the
//! template this follows. The shape is essentially identical:
//!   - One Agent SoA shared between two creature types (Red=0, Blue=1)
//!   - Per-creature physics dispatch each tick (MoveRed + MoveBlue),
//!     gated on `creature_type` so each thread only commits writes
//!     for matching agents
//!
//! What differs from predator_prey_min:
//!   - Adds an `agent_hp_buf` (each agent carries hp; combat damage
//!     decrements it via `agents.set_hp` from inside the per-agent
//!     body)
//!   - No emit / event-ring path needed — combat is realised via the
//!     direct hp write in the per-agent body, not via a Damaged-event
//!     chronicle (the duel_1v1 path)
//!   - Per-team alive readback: count slots with hp > 0 per team

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// 16-byte WGSL `vec3<f32>` interop type. Same shape duplicated across
/// every per-fixture runtime crate (no inter-fixture coupling beyond
/// `engine` + `dsl_compiler`).
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

/// Per-fixture state for the flocking_skirmish simulation.
pub struct FlockingSkirmishState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-agent HP. Initialised to `INITIAL_HP` for every slot. Each
    /// per-agent body computes `damage_taken = sum(...enemies in
    /// attack_radius)` and writes `set_hp(self, hp - damage)`. An
    /// agent with `hp <= 0` is considered dead by the host-side
    /// readback (no in-DSL alive flag — keeps the fixture lean).
    hp_buf: wgpu::Buffer,
    /// Per-agent `creature_type` discriminant — `array<u32>` of length
    /// `agent_count`. Each slot holds the EntityRef declaration-order
    /// index for that agent (Red = 0, Blue = 1 — matching the order of
    /// `entity Red` / `entity Blue` in `flocking_skirmish.sim`).
    creature_type_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    /// Number of agents initialised as Red (creature_type = 0). Slots
    /// `[0..red_count)`. Set at construction; immutable thereafter.
    red_count: u32,
    /// Number of agents initialised as Blue (creature_type = 1). Slots
    /// `[red_count..red_count + blue_count)`.
    blue_count: u32,

    cache: dispatch::KernelCache,

    pos_cache: Vec<Vec3>,
    dirty: bool,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

const INITIAL_HP: f32 = 100.0;

/// Map a u32 hash output into a uniform `[-1, 1)` float — same pattern
/// other runtimes use to derive deterministic initial state from the
/// keyed PCG RNG (P5).
fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl FlockingSkirmishState {
    /// Construct a flocking_skirmish simulation with `red_count` Red
    /// agents (creature_type = 0) followed by `blue_count` Blue
    /// agents (creature_type = 1) sharing one Agent SoA. Total agent
    /// count is `red_count + blue_count`.
    ///
    /// Initial positions: Red flock clustered around (-INIT_OFFSET,
    /// 0, 0) and Blue clustered around (+INIT_OFFSET, 0, 0). This
    /// gives the two flocks a clean "approach from opposite sides"
    /// opening so the first ~30 ticks are dominated by team flocking
    /// (no combat) and combat onsets when the flocks meet near the
    /// origin around tick 40-60. Initial velocities are tiny random
    /// nudges so the flocks aren't perfectly stationary at t=0.
    pub fn new(seed: u64, red_count: u32, blue_count: u32) -> Self {
        let agent_count = red_count + blue_count;
        let n = agent_count as usize;

        // Red flock origin and Blue flock origin. Distance between
        // them is 2*INIT_OFFSET. Spread per flock chosen so the
        // average inter-agent distance starts comfortably above
        // `separation_radius` (1.5) — otherwise the initial separation
        // burst dominates everything. A spread of 10 units in 3D for
        // ~100 agents gives an average nearest-neighbour distance of
        // ~3 units.
        const INIT_OFFSET: f32 = 15.0;
        const FLOCK_SPREAD: f32 = 10.0;
        // Stronger initial X-axis bias toward the enemy centre so the
        // flocks actually march at each other (the per-agent inertia
        // is 0.85; a tiny initial velocity dies in ~50 ticks before
        // the inter-flock cohesion can pull them together).
        const VEL_NUDGE: f32 = 0.05;
        const APPROACH_BIAS: f32 = 1.5;

        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);

        for slot in 0..agent_count {
            let agent_id = AgentId::new(slot + 1).expect("slot+1 non-zero");
            // Slots [0..red_count) are Red, slots [red_count..) are Blue.
            let is_red = slot < red_count;
            let center_x = if is_red { -INIT_OFFSET } else { INIT_OFFSET };
            let p = Vec3::new(
                center_x
                    + normalise(per_agent_u32(seed, agent_id, 0, b"fs_pos_x"))
                        * FLOCK_SPREAD,
                normalise(per_agent_u32(seed, agent_id, 0, b"fs_pos_y")) * FLOCK_SPREAD,
                normalise(per_agent_u32(seed, agent_id, 0, b"fs_pos_z")) * FLOCK_SPREAD,
            );
            // Initial velocity: random nudge plus a strong bias toward
            // the enemy centre. Sets up a "two armies marching toward
            // each other" opening; the tightness signal then comes
            // from within-flock cohesion drowning out the inter-flock
            // bias once they're close enough to fight.
            let bias_x = if is_red { APPROACH_BIAS } else { -APPROACH_BIAS };
            let v = Vec3::new(
                bias_x
                    + normalise(per_agent_u32(seed, agent_id, 0, b"fs_vel_x"))
                        * VEL_NUDGE,
                normalise(per_agent_u32(seed, agent_id, 0, b"fs_vel_y")) * VEL_NUDGE,
                normalise(per_agent_u32(seed, agent_id, 0, b"fs_vel_z")) * VEL_NUDGE,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
            creature_init.push(if is_red { 0u32 } else { 1u32 });
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flocking_skirmish_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flocking_skirmish_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let hp_init: Vec<f32> = vec![INITIAL_HP; n];
        let hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flocking_skirmish_runtime::hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flocking_skirmish_runtime::creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Both MoveRed and MoveBlue share the same cfg shape (agent_cap
        // + tick + seed + pad) — reusing one buffer for both is safe.
        let cfg = physics_MoveRed::PhysicsMoveRedCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flocking_skirmish_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flocking_skirmish_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            gpu,
            pos_buf,
            vel_buf,
            hp_buf,
            creature_type_buf,
            cfg_buf,
            pos_staging,
            red_count,
            blue_count,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn red_count(&self) -> u32 { self.red_count }
    pub fn blue_count(&self) -> u32 { self.blue_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
    pub fn agent_count(&self) -> u32 { self.agent_count }

    /// Per-agent HP readback. Length = `agent_count`. Slots
    /// `[0..red_count)` are Red, slots `[red_count..)` are Blue. An
    /// agent is "alive" when `hp > 0.0`.
    pub fn read_hp(&self) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flocking_skirmish_runtime::hp_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("flocking_skirmish_runtime::read_hp") },
        );
        encoder.copy_buffer_to_buffer(&self.hp_buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("flocking_skirmish_runtime::positions::copy"),
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

impl CompiledSim for FlockingSkirmishState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("flocking_skirmish_runtime::step"),
            },
        );

        // Update the per-tick `tick` field in the cfg uniform — the
        // PerAgent kernels read `cfg.tick` for any deterministic RNG
        // calls (none in this fixture today, but the slot is there).
        let cfg = physics_MoveRed::PhysicsMoveRedCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.cfg_buf, 0, bytemuck::bytes_of(&cfg));

        // (1) MoveRed — gated on creature_type == 0 inside the kernel.
        // Reads agent_pos, agent_vel, agent_hp, agent_creature_type;
        // writes pos, vel, hp.
        let red_bindings = physics_MoveRed::PhysicsMoveRedBindings {
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            agent_hp: &self.hp_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_movered(
            &mut self.cache,
            &red_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) MoveBlue — gated on creature_type == 1. Same shape as
        // MoveRed; same buffer set.
        let blue_bindings = physics_MoveBlue::PhysicsMoveBlueBindings {
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            agent_hp: &self.hp_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_moveblue(
            &mut self.cache,
            &blue_bindings,
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

/// Build a boxed `CompiledSim` so the `sim_app` runner can switch
/// between fixture runtimes via a one-line constructor swap. Default
/// composition: 100 Red + 100 Blue (200 agents).
pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(FlockingSkirmishState::new(seed, 100, 100))
}
