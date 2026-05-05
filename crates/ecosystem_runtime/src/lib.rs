//! Per-fixture runtime for `assets/sim/ecosystem_cascade.sim` —
//! the three-tier event-cascade fixture (Plant ← Herbivore ← Carnivore).
//!
//! Mirrors the [`predator_prey_runtime`] / [`swarm_storm_runtime`]
//! shape: compiler-emitted kernels + dispatch + schedule live in
//! `OUT_DIR/generated.rs`, pulled in via `include!`. The hand-written
//! orchestration shell ([`EcosystemState`]) owns the GPU context,
//! per-field SoA buffers, the kernel cache, the host-side position
//! cache, the shared event-ring, and the three @decay-anchored
//! view-storage instances.
//!
//! ## What's new vs the precedents
//!
//! - **Three entity types** via `where (self.creature_type == X)`
//!   discriminator (Plant=0, Herbivore=1, Carnivore=2 — declaration
//!   order in the .sim file). The compiler emits three separate
//!   `physics_Move<Tier>` kernels each with its own where-clause guard;
//!   each per-agent thread runs all three but commits writes for
//!   exactly one.
//! - **Two distinct event kinds** (PlantEaten, HerbivoreEaten) — the
//!   compiler tags emits with kind=1 (PlantEaten from MoveHerbivore)
//!   and kind=2 (HerbivoreEaten from MoveCarnivore). The schedule
//!   synthesizer emitted a SINGLE shared event ring (one
//!   `EventRing::new()` call here, both physics emitters atomicAdd
//!   against the same `event_tail`).
//! - **Three @decay views** (recent_browse rate=0.92, predator_pressure
//!   rate=0.95, plant_pressure rate=0.90) — each carries a per-tick
//!   anchor multiply (B2 lowering) that runs before the per-event
//!   fold lands. All three views share the same event_ring; the fold
//!   kernels currently DO NOT discriminate by event tag (they read
//!   `event_ring[idx*10+3]` blindly), so each view accumulates from
//!   BOTH PlantEaten and HerbivoreEaten emits — see the GAP note at
//!   the top of `ecosystem_app.rs` for details.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// 16-byte WGSL `vec3<f32>` interop. Same shape as the predator_prey /
/// swarm_storm runtimes use; duplicated here to keep each fixture-
/// runtime crate self-contained (no inter-fixture coupling beyond
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

/// Per-fixture state for the ecosystem_cascade simulation.
pub struct EcosystemState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-agent `creature_type` discriminant — `array<u32>` of length
    /// `agent_count`. Each slot holds the EntityRef declaration-order
    /// index for that agent (Plant=0, Herbivore=1, Carnivore=2). The
    /// per-handler `where (self.creature_type == <Entity>)` lowering
    /// compares this against the literal at the WGSL layer; the runtime
    /// keeps the per-slot value in sync with declaration order.
    creature_type_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    /// Single shared event-ring for both PlantEaten + HerbivoreEaten.
    /// The schedule synthesizer emitted ONE shared ring (both physics
    /// emitters' Bindings expect the same `event_ring` + `event_tail`
    /// slot, both fold kernels read from the same buffer; the kind
    /// tag lives in `event_ring[slot*10+0]` but the fold kernels
    /// currently don't filter on it).
    event_ring: EventRing,

    /// recent_browse(by: Agent) -> f32. `@decay(rate=0.92)`. Per the
    /// .sim spec this view is "Per-Herbivore browse activity" but
    /// since the fold kernel doesn't filter by event kind, every
    /// emit (PlantEaten + HerbivoreEaten) currently lands here too.
    recent_browse: ViewStorage,
    recent_browse_cfg_buf: wgpu::Buffer,
    recent_browse_decay_cfg_buf: wgpu::Buffer,

    /// predator_pressure(by: Agent) -> f32. `@decay(rate=0.95)`. Same
    /// caveat — fold reads `event_ring[idx*10+3]` (the `by` field of
    /// either kind) without tag discrimination.
    predator_pressure: ViewStorage,
    predator_pressure_cfg_buf: wgpu::Buffer,
    predator_pressure_decay_cfg_buf: wgpu::Buffer,

    /// plant_pressure(plant: Agent) -> f32. `@decay(rate=0.90)`. The
    /// view binds to the `plant` field of PlantEaten which (per the
    /// .sim placeholder) equals `self`, so plant slots stay zero.
    plant_pressure: ViewStorage,
    plant_pressure_cfg_buf: wgpu::Buffer,
    plant_pressure_decay_cfg_buf: wgpu::Buffer,

    /// Per-creature-type init counts. The smoke fixture introspects
    /// these to verify the where-guarded dispatch routed correctly.
    plant_count: u32,
    herbivore_count: u32,
    carnivore_count: u32,

    cache: dispatch::KernelCache,
    pos_cache: Vec<Vec3>,
    dirty: bool,
    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Map a u32 hash output into a uniform `[-1, 1)` float — same pattern
/// the other runtimes use for keyed-PCG-derived initial state.
fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl EcosystemState {
    /// Construct an N-agent simulation with deterministic initial
    /// positions + velocities derived from `seed` via the engine's
    /// keyed PCG (P5: `per_agent_u32(seed, agent_id, tick=0, purpose)`).
    ///
    /// Slots are split into roughly equal thirds in declaration order:
    /// the first `agent_count/3` slots are Plants, the next third are
    /// Herbivores, the remainder are Carnivores. The tier order
    /// matches the `entity` declaration order in `ecosystem_cascade.sim`
    /// so the where-clause discriminants (`creature_type == 0u/1u/2u`)
    /// fire on the right slots.
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let spread = (agent_count as f32).cbrt().max(1.0);

        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"eco_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"eco_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"eco_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"eco_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"eco_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"eco_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }

        // Three-way split. With agent_count = 64 → 21 plants + 21
        // herbivores + 22 carnivores; the "+1 herbivore/+2 carnivore"
        // remainder lives in the last two slots so the prefix-summed
        // bands stay contiguous (debuggable).
        let third = agent_count / 3;
        let plant_count = third;
        let herbivore_count = third;
        let carnivore_count = agent_count - 2 * third;
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let disc = if slot < plant_count {
                0u32 // Plant
            } else if slot < plant_count + herbivore_count {
                1u32 // Herbivore
            } else {
                2u32 // Carnivore
            };
            creature_init.push(disc);
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ecosystem_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ecosystem_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        // All three physics_Move<Tier>Cfg structs are byte-identical
        // (agent_cap + tick + 8 bytes pad), so one cfg buffer is safe
        // to bind to all three kernels — same trick predator_prey uses
        // for MoveHare/MoveWolf.
        let cfg = physics_MovePlant::PhysicsMovePlantCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ecosystem_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ecosystem_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Shared event-ring (single ring for both PlantEaten +
        // HerbivoreEaten — what the schedule synthesizer expects) ----
        let event_ring = EventRing::new(&gpu, "ecosystem_runtime");

        // ---- Three @decay views, each with anchor (no ids) ----
        let recent_browse = ViewStorage::new(
            &gpu,
            "ecosystem_runtime::recent_browse",
            agent_count,
            true,  // has_anchor (carries @decay)
            false, // no top-K storage hint
        );
        let predator_pressure = ViewStorage::new(
            &gpu,
            "ecosystem_runtime::predator_pressure",
            agent_count,
            true,
            false,
        );
        let plant_pressure = ViewStorage::new(
            &gpu,
            "ecosystem_runtime::plant_pressure",
            agent_count,
            true,
            false,
        );

        let recent_browse_cfg = fold_recent_browse::FoldRecentBrowseCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let recent_browse_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::recent_browse_cfg"),
                contents: bytemuck::bytes_of(&recent_browse_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let predator_pressure_cfg = fold_predator_pressure::FoldPredatorPressureCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let predator_pressure_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::predator_pressure_cfg"),
                contents: bytemuck::bytes_of(&predator_pressure_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let plant_pressure_cfg = fold_plant_pressure::FoldPlantPressureCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let plant_pressure_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::plant_pressure_cfg"),
                contents: bytemuck::bytes_of(&plant_pressure_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let recent_browse_decay_cfg = decay_recent_browse::DecayRecentBrowseCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let recent_browse_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::recent_browse_decay_cfg"),
                contents: bytemuck::bytes_of(&recent_browse_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let predator_pressure_decay_cfg =
            decay_predator_pressure::DecayPredatorPressureCfg {
                agent_cap: agent_count,
                tick: 0,
                slot_count: agent_count,
            _pad: 0,
            };
        let predator_pressure_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::predator_pressure_decay_cfg"),
                contents: bytemuck::bytes_of(&predator_pressure_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let plant_pressure_decay_cfg = decay_plant_pressure::DecayPlantPressureCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let plant_pressure_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("ecosystem_runtime::plant_pressure_decay_cfg"),
                contents: bytemuck::bytes_of(&plant_pressure_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            creature_type_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            recent_browse,
            recent_browse_cfg_buf,
            recent_browse_decay_cfg_buf,
            predator_pressure,
            predator_pressure_cfg_buf,
            predator_pressure_decay_cfg_buf,
            plant_pressure,
            plant_pressure_cfg_buf,
            plant_pressure_decay_cfg_buf,
            plant_count,
            herbivore_count,
            carnivore_count,
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
    pub fn plant_count(&self) -> u32 {
        self.plant_count
    }
    pub fn herbivore_count(&self) -> u32 {
        self.herbivore_count
    }
    pub fn carnivore_count(&self) -> u32 {
        self.carnivore_count
    }

    /// Returns the per-slot `creature_type` discriminant (Plant=0,
    /// Herbivore=1, Carnivore=2). Useful for per-tier sample logging.
    pub fn creature_types(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.agent_count as usize);
        let third = self.agent_count / 3;
        for slot in 0..self.agent_count {
            let disc = if slot < third {
                0
            } else if slot < 2 * third {
                1
            } else {
                2
            };
            out.push(disc);
        }
        out
    }

    pub fn recent_browses(&mut self) -> &[f32] {
        self.recent_browse.readback(&self.gpu)
    }

    pub fn predator_pressures(&mut self) -> &[f32] {
        self.predator_pressure.readback(&self.gpu)
    }

    pub fn plant_pressures(&mut self) -> &[f32] {
        self.plant_pressure.readback(&self.gpu)
    }

    /// One-shot GPU readback of the per-slot `creature_type`
    /// discriminant (Plant=0, Herbivore=1, Carnivore=2 — declaration
    /// order in `ecosystem_cascade.sim`). Mirrors the
    /// `predator_prey_runtime::read_creature_types` shape: ad-hoc
    /// staging buffer + map_async + poll/wait. Allocates a fresh
    /// staging buffer per call — `snapshot()` only fires at viz cadence
    /// (~10 Hz) so the cost is acceptable.
    ///
    /// Distinct from the host-only `creature_types()` accessor above,
    /// which synthesises the discriminants from declaration-order
    /// without touching the GPU. The viz path goes through this so it
    /// reflects the real SoA contents (defence-in-depth against any
    /// future kernel that mutates `creature_type`).
    fn read_creature_types_gpu(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ecosystem_runtime::creature_type_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("ecosystem_runtime::creature_type::copy"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.creature_type_buf, 0, &staging, 0, bytes,
        );
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

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("ecosystem_runtime::positions::copy"),
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

impl CompiledSim for EcosystemState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("ecosystem_runtime::step"),
            },
        );

        // Per-tick clear of the shared event_tail. Both physics
        // emitters atomicAdd against this counter; the count
        // accumulates over the tick and gets read back to size the
        // fold dispatch.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1) MovePlant — no event_ring/event_tail bindings (Plants
        // don't emit). Reads creature_type, pos, vel; writes pos.
        let plant_bindings = physics_MovePlant::PhysicsMovePlantBindings {
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_moveplant(
            &mut self.cache,
            &plant_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) MoveHerbivore — emits `PlantEaten { plant: self, by: self,
        // pos }` (kind tag = 1u in the WGSL emit body). Bindings include
        // event_ring + event_tail.
        let herb_bindings = physics_MoveHerbivore::PhysicsMoveHerbivoreBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_moveherbivore(
            &mut self.cache,
            &herb_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) MoveCarnivore — emits `HerbivoreEaten { prey: self, by:
        // self, pos }` (kind tag = 2u). Same shared ring / tail.
        let carn_bindings = physics_MoveCarnivore::PhysicsMoveCarnivoreBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_creature_type: &self.creature_type_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_movecarnivore(
            &mut self.cache,
            &carn_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) seed_indirect_0 — keeps the indirect-args buffer warm
        // for the eventual dispatch_workgroups_indirect wire-up.
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

        // Per-tick event count = herbivore + carnivore (each emits
        // exactly one event per tick within its where-clause guard).
        // The folds early-return past `cfg.event_count` so an
        // upper-bound estimate is safe; agent_count is the cheap
        // upper bound we use throughout the precedent runtimes.
        let event_count_estimate = self.herbivore_count + self.carnivore_count;

        // (5a) decay_recent_browse — anchor multiply BEFORE the fold.
        let rb_decay_cfg = decay_recent_browse::DecayRecentBrowseCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.recent_browse_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&rb_decay_cfg),
        );
        let rb_decay_bindings = decay_recent_browse::DecayRecentBrowseBindings {
            view_storage_primary: self.recent_browse.primary(),
            cfg: &self.recent_browse_decay_cfg_buf,
        };
        dispatch::dispatch_decay_recent_browse(
            &mut self.cache,
            &rb_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5b) fold_recent_browse — RMW primary[event[3]] += 1.0 per
        // event. NOTE: kernel does NOT filter by event tag — see GAP
        // note in the .sim file's view docs.
        let rb_cfg = fold_recent_browse::FoldRecentBrowseCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.recent_browse_cfg_buf,
            0,
            bytemuck::bytes_of(&rb_cfg),
        );
        let rb_bindings = fold_recent_browse::FoldRecentBrowseBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.recent_browse.primary(),
            view_storage_anchor: self.recent_browse.anchor(),
            view_storage_ids: self.recent_browse.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.recent_browse_cfg_buf,
        };
        // Pass event_count as the dispatch sizing argument (the helper
        // derives workgroup_x = ceil(arg / 64); the kernel uses gid.x
        // as event_idx and early-returns past cfg.event_count).
        dispatch::dispatch_fold_recent_browse(
            &mut self.cache,
            &rb_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        // (6a/b) decay + fold predator_pressure (rate=0.95).
        let pp_decay_cfg = decay_predator_pressure::DecayPredatorPressureCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.predator_pressure_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&pp_decay_cfg),
        );
        let pp_decay_bindings = decay_predator_pressure::DecayPredatorPressureBindings {
            view_storage_primary: self.predator_pressure.primary(),
            cfg: &self.predator_pressure_decay_cfg_buf,
        };
        dispatch::dispatch_decay_predator_pressure(
            &mut self.cache,
            &pp_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let pp_cfg = fold_predator_pressure::FoldPredatorPressureCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.predator_pressure_cfg_buf,
            0,
            bytemuck::bytes_of(&pp_cfg),
        );
        let pp_bindings = fold_predator_pressure::FoldPredatorPressureBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.predator_pressure.primary(),
            view_storage_anchor: self.predator_pressure.anchor(),
            view_storage_ids: self.predator_pressure.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.predator_pressure_cfg_buf,
        };
        dispatch::dispatch_fold_predator_pressure(
            &mut self.cache,
            &pp_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        // (7a/b) decay + fold plant_pressure (rate=0.90).
        let plp_decay_cfg = decay_plant_pressure::DecayPlantPressureCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.plant_pressure_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&plp_decay_cfg),
        );
        let plp_decay_bindings = decay_plant_pressure::DecayPlantPressureBindings {
            view_storage_primary: self.plant_pressure.primary(),
            cfg: &self.plant_pressure_decay_cfg_buf,
        };
        dispatch::dispatch_decay_plant_pressure(
            &mut self.cache,
            &plp_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let plp_cfg = fold_plant_pressure::FoldPlantPressureCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.plant_pressure_cfg_buf,
            0,
            bytemuck::bytes_of(&plp_cfg),
        );
        let plp_bindings = fold_plant_pressure::FoldPlantPressureBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.plant_pressure.primary(),
            view_storage_anchor: self.plant_pressure.anchor(),
            view_storage_ids: self.plant_pressure.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.plant_pressure_cfg_buf,
        };
        dispatch::dispatch_fold_plant_pressure(
            &mut self.cache,
            &plp_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.recent_browse.mark_dirty();
        self.predator_pressure.mark_dirty();
        self.plant_pressure.mark_dirty();
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

    /// Snapshot per-agent state for the universal `viz_app` renderer.
    ///
    /// `ecosystem_cascade.sim` declares `pos: vec3` on all three
    /// entities AND the per-tier physics rules write to it through
    /// `agents.set_pos`, so positions come from a real GPU readback
    /// of `pos_buf` (no synthesised layout).
    ///
    /// `creature_types` encoding (6 entries, indexed by
    /// `tier * 2 + dead_bit`):
    ///
    /// |  i | tier      | state |
    /// |----|-----------|-------|
    /// |  0 | Plant     | alive |
    /// |  1 | Plant     | dead  |
    /// |  2 | Herbivore | alive |
    /// |  3 | Herbivore | dead  |
    /// |  4 | Carnivore | alive |
    /// |  5 | Carnivore | dead  |
    ///
    /// `tier` derives from a REAL SoA read of `creature_type_buf`:
    /// the .sim's entity-declaration order pins Plant=0 / Herbivore=1
    /// / Carnivore=2.
    ///
    /// `alive`: ecosystem_cascade has no Eaten→SoA writeback path
    /// (no `agent_alive_buf`) — the views accumulate Eaten events
    /// but no field flips a kill bit. Following `predator_prey_runtime`
    /// Stage 0's approach, every slot reports alive (1) and the dead
    /// variants (1/3/5 in the encoding above) currently never fire.
    /// When a future per-pair body-iter resolves real strikes and
    /// emits an ApplyDefeat handler, swap this for a `read_alive()`
    /// GPU readback.
    ///
    /// Initial-state safe: GPU buffers are populated by
    /// `create_buffer_init` at construction, so calling `snapshot()`
    /// before any `step()` returns the deterministic spawn cube with
    /// the contiguous-band tier assignment from `new()`.
    fn snapshot(&mut self) -> AgentSnapshot {
        // Position readback first — drives `dirty` flush via the
        // existing pos_staging path.
        let positions: Vec<Vec3> = self.read_positions().to_vec();
        let tiers: Vec<u32> = self.read_creature_types_gpu();
        let n = self.agent_count as usize;
        // No Eaten→SoA writeback today — every slot is alive.
        let alive: Vec<u32> = vec![1u32; n];
        let creature_types: Vec<u32> = (0..n)
            .map(|i| {
                let tier = tiers[i] & 0b11; // 0 plant, 1 herb, 2 carn
                let dead_bit = if alive[i] == 0 { 1 } else { 0 };
                tier * 2 + dead_bit
            })
            .collect();
        AgentSnapshot { positions, creature_types, alive }
    }

    /// 6 entries matching the `snapshot.creature_types` encoding:
    /// `[plant_alive, plant_dead, herb_alive, herb_dead, carn_alive,
    /// carn_dead]`. Plant=green `p`, Herbivore=amber `h`, Carnivore=red
    /// `C`. All dead variants share a grey tombstone × so the renderer
    /// has a uniform "removed" indicator.
    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![
            VizGlyph::new('p', 47),         // 0: plant alive (green)
            VizGlyph::new('\u{00D7}', 240), // 1: plant dead (grey ×)
            VizGlyph::new('h', 215),        // 2: herbivore alive (amber)
            VizGlyph::new('\u{00D7}', 240), // 3: herbivore dead (grey ×)
            VizGlyph::new('C', 196),        // 4: carnivore alive (red)
            VizGlyph::new('\u{00D7}', 240), // 5: carnivore dead (grey ×)
        ]
    }

    /// Default zoom around the deterministic spawn cube. `new()`
    /// derives per-axis spreads from `cbrt(agent_count)` (so a 64-agent
    /// fixture spans roughly `[-4, 4]` per axis); ±5 keeps every spawn
    /// on screen with breathing room for the per-tier integrators
    /// (plant_speed=0.05, herbivore_speed=0.4, carnivore_speed=0.55,
    /// all bounded by config). The renderer auto-scales if positions
    /// wander outside.
    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        let span = (self.agent_count as f32).cbrt().max(1.0) + 1.0;
        Some((Vec3::new(-span, -span, 0.0), Vec3::new(span, span, 0.0)))
    }
}

/// Build a boxed `CompiledSim` so the `sim_app` runner can switch
/// between fixture runtimes via a one-line constructor swap. Mirrors
/// `boids_runtime::make_sim`.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(EcosystemState::new(seed, agent_count))
}

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// Snapshot before any tick must report initial state: every agent
    /// alive, deterministic three-tier creature_type assignment from
    /// the contiguous-band split in `new()`, and positions inside the
    /// spawn cube derived from the `cbrt(agent_count)` spread. Guards
    /// the construction-only readback path so `viz_app` can render
    /// frame 0 with content instead of a blank arena.
    #[test]
    fn snapshot_after_construction_returns_initial_state() {
        const N: u32 = 9; // 3 plants + 3 herbivores + 3 carnivores
        let mut state = EcosystemState::new(0xCAFE_F00D, N);
        let snap = state.snapshot();

        assert_eq!(snap.positions.len(), N as usize, "positions length");
        assert_eq!(
            snap.creature_types.len(),
            N as usize,
            "creature_types length",
        );
        assert_eq!(snap.alive.len(), N as usize, "alive length");

        // No Eaten→SoA writeback path today — every slot must be alive.
        let alive_total: u32 = snap.alive.iter().sum();
        assert_eq!(
            alive_total, N,
            "every slot must be alive at construction (no Eaten→SoA \
             path); got alive={:?}",
            snap.alive,
        );

        // Three-way contiguous-band split per `new()`: third = N/3 = 3,
        // so slots [0..3) = Plant (encoded 0), [3..6) = Herbivore
        // (encoded 2), [6..9) = Carnivore (encoded 4). Encoded value
        // = tier*2 because every dead_bit = 0.
        let third = N / 3;
        for (i, &ct) in snap.creature_types.iter().enumerate() {
            let i_u = i as u32;
            let expected_tier = if i_u < third {
                0u32
            } else if i_u < 2 * third {
                1u32
            } else {
                2u32
            };
            let expected = expected_tier * 2;
            assert_eq!(
                ct, expected,
                "slot {i}: tier band must match contiguous split \
                 (expected tier={expected_tier}, encoded={expected}); \
                 got {ct}",
            );
        }

        // Tier counts must match the runtime's own accounting.
        let plant_n = snap.creature_types.iter().filter(|&&c| c == 0).count();
        let herb_n = snap.creature_types.iter().filter(|&&c| c == 2).count();
        let carn_n = snap.creature_types.iter().filter(|&&c| c == 4).count();
        assert_eq!(plant_n as u32, state.plant_count(), "plant count");
        assert_eq!(herb_n as u32, state.herbivore_count(), "herbivore count");
        assert_eq!(carn_n as u32, state.carnivore_count(), "carnivore count");

        // Positions must lie inside the deterministic spawn cube.
        let spread = (N as f32).cbrt().max(1.0);
        for (i, p) in snap.positions.iter().enumerate() {
            assert!(
                p.x.abs() <= spread + 0.001
                    && p.y.abs() <= spread + 0.001
                    && p.z.abs() <= spread + 0.001,
                "slot {i} position {p:?} outside spawn cube ±{spread}",
            );
        }

        // Glyph table must cover every encoded value the snapshot can
        // produce (6 entries, max index = 5).
        let glyphs = state.glyph_table();
        assert_eq!(glyphs.len(), 6, "glyph_table must have 6 entries");
        for (i, &ct) in snap.creature_types.iter().enumerate() {
            assert!(
                (ct as usize) < glyphs.len(),
                "slot {i}: creature_type {ct} out of glyph_table range",
            );
        }
    }

    /// After ticking the simulation forward, every per-tier integrator
    /// advances its band by `vel * speed` per step, so at least one
    /// slot's position must have changed from the initial spawn. Proves
    /// the snapshot reflects live GPU state, not a cached construction-
    /// time copy.
    #[test]
    fn snapshot_after_tick_reflects_state_change() {
        const N: u32 = 9;
        let mut state = EcosystemState::new(0xCAFE_F00D, N);
        let initial = state.snapshot().positions.clone();

        for _ in 0..50 {
            state.step();
        }

        let snap = state.snapshot();
        assert_eq!(snap.positions.len(), N as usize);

        let any_moved = initial
            .iter()
            .zip(snap.positions.iter())
            .any(|(a, b)| {
                (a.x - b.x).abs() > 1e-6
                    || (a.y - b.y).abs() > 1e-6
                    || (a.z - b.z).abs() > 1e-6
            });
        assert!(
            any_moved,
            "after 50 ticks, expected per-tier integrator to drift at \
             least one slot; saw all positions identical",
        );
    }
}
