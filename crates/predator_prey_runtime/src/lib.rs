//! Per-fixture runtime for `assets/sim/predator_prey_min.sim` (the
//! Stage 0 skeleton tracking the design target at
//! `assets/sim/predator_prey.sim`).
//!
//! Mirrors the [`boids_runtime`] crate's shape — compiler-emitted
//! kernels + dispatch + schedule live in `OUT_DIR/generated.rs`,
//! pulled in via `include!`. The hand-written orchestration shell
//! ([`PredatorPreyState`]) owns the GPU context, per-field storage
//! buffers, the kernel cache, and the host-side position cache.
//!
//! ## Stage 0 scope
//!
//! - One entity SoA: `pos_buf: array<vec3<f32>>` + `vel_buf` per slot.
//!   Both `Hare` and `Wolf` agent types share the same SoA layout (the
//!   compiler today doesn't distinguish them at the buffer level —
//!   that comes with PP Stage 1's per-handler where-clause split).
//! - One per-tick dispatch: `physics_MoveAlive` (the trivial integrator
//!   in `predator_prey_min.sim`).
//! - Position readback on demand for visualisation / smoke tests.
//!
//! Deferred to later PP stages:
//!   - Spatial-grid buffers (no spatial query in the Stage 0 fixture)
//!   - Per-kernel GPU timestamps + diagnostic kernel
//!   - Event rings + view-fold storage (Stages 2, 5)
//!   - Per-creature-type init (Stage 1 splits the SoA-vs-id semantics)

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// 16-byte WGSL `vec3<f32>` interop type. Same shape as the one in
/// `boids_runtime`; duplicated here rather than re-exported to keep
/// each fixture-runtime crate self-contained (no inter-fixture
/// coupling beyond `engine` + `dsl_compiler`).
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

/// Per-fixture state for the predator_prey simulation.
pub struct PredatorPreyState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-agent `creature_type` discriminant — `array<u32>` of
    /// length `agent_count`. Each slot holds the EntityRef
    /// declaration-order index for that agent (Hare=0, Wolf=1 for
    /// pp_min). The compiler's per-handler `where (self.creature_type
    /// == <Entity>)` lowering compares this against the
    /// EntityRef-derived literal at the WGSL layer; the runtime is
    /// responsible for keeping the per-slot value in sync with the
    /// declaration order.
    creature_type_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    /// Hare and Wolf shapes share a single agent_cap uniform. The
    /// emitted PhysicsMoveHareCfg / PhysicsMoveWolfCfg structs are
    /// byte-identical (same fields, same padding), so one buffer is
    /// safe to bind to both kernels.

    /// Per-creature counts the smoke fixtures introspect to verify
    /// the where-guarded dispatch routed correctly. Same agent_count
    /// total — split 50/50 between Hare (slot index even → Hare) and
    /// Wolf (odd → Wolf) at init time.
    hare_count: u32,
    wolf_count: u32,

    cache: dispatch::KernelCache,

    pos_cache: Vec<Vec3>,
    dirty: bool,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Map a u32 hash output into a uniform `[-1, 1)` float — same pattern
/// boids_runtime uses to derive deterministic initial positions /
/// velocities from the keyed PCG RNG (P5).
fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl PredatorPreyState {
    /// Construct an N-agent simulation with deterministic initial
    /// positions + velocities derived from `seed` via engine's keyed
    /// PCG (P5: `per_agent_u32(seed, agent_id, tick=0, purpose)`).
    ///
    /// Stage 0 doesn't distinguish Hare from Wolf at init time —
    /// every slot gets the same uniform-cube spawn. PP Stage 1 will
    /// thread per-creature-type init alongside the where-clause
    /// physics split (e.g. wolves cluster, hares scatter).
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        // Pick a spread that gives ~1 agent per cubic unit at the
        // default population — small enough for visual smoke tests,
        // doesn't matter for Stage 0 since there's no spatial query.
        let spread = (agent_count as f32).cbrt().max(1.0);

        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"pp_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"pp_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"pp_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"pp_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"pp_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"pp_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Per-creature-type discriminants (matches EntityRef order
        // in `predator_prey_min.sim`: Hare = 0, Wolf = 1). 50/50
        // split — even slots are Hare, odd slots are Wolf. The
        // compiler emits `if (creature_type == 0u) { … }` for the
        // MoveHare kernel and `if (creature_type == 1u) { … }` for
        // MoveWolf, so each per-agent thread runs both kernels but
        // commits writes for exactly one.
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);
        let mut hare_count = 0u32;
        let mut wolf_count = 0u32;
        for slot in 0..agent_count {
            let disc = if slot % 2 == 0 {
                hare_count += 1;
                0u32 // Hare
            } else {
                wolf_count += 1;
                1u32 // Wolf
            };
            creature_init.push(disc);
        }
        let creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_runtime::creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let cfg = physics_MoveHare::PhysicsMoveHareCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            gpu,
            pos_buf,
            vel_buf,
            creature_type_buf,
            hare_count,
            wolf_count,
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

    /// Current simulation tick. Increments at the end of each `step()`.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Number of agents initialised as Hare (creature_type = 0).
    pub fn hare_count(&self) -> u32 {
        self.hare_count
    }

    /// Number of agents initialised as Wolf (creature_type = 1).
    pub fn wolf_count(&self) -> u32 {
        self.wolf_count
    }

    /// Seed used to derive initial state. Stable across the sim's
    /// lifetime; used by replay / regression fixtures.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("predator_prey_runtime::positions::copy"),
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

impl CompiledSim for PredatorPreyState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_runtime::step"),
            },
        );

        // Two PerAgent dispatches per tick, sequenced in op-id order
        // (Hare → Wolf). Each kernel runs over every slot but the
        // where-guarded If wrap from `cg/lower/physics.rs` early-
        // returns for non-matching creature types, so each agent
        // commits exactly one set_pos write per tick.
        let hare_bindings = physics_MoveHare::PhysicsMoveHareBindings {
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_movehare(
            &mut self.cache,
            &hare_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let wolf_bindings = physics_MoveWolf::PhysicsMoveWolfBindings {
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_movewolf(
            &mut self.cache,
            &wolf_bindings,
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

    /// On-demand readback. When `dirty` is set, encode a
    /// `pos_buf → pos_staging` copy, submit, await the map, and
    /// decode the mapped bytes into `pos_cache`. Consecutive calls
    /// without an intervening `step()` skip the readback.
    fn positions(&mut self) -> &[Vec3] {
        self.read_positions()
    }
}

/// Build a boxed `CompiledSim` so the `sim_app` runner can switch
/// between fixture runtimes via a one-line constructor swap. Mirrors
/// `boids_runtime::make_sim`.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(PredatorPreyState::new(seed, agent_count))
}
