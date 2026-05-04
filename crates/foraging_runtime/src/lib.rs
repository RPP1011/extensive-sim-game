//! Per-fixture runtime for `assets/sim/foraging_colony.sim` —
//! the ant-foraging fixture exercising the per-tick Drop emit + a
//! single `pheromone_deposits(ant: Agent) -> f32` view with
//! `@decay(rate=0.88)`. Mirrors the
//! [`crowd_navigation_runtime`] / [`bartering_runtime`] shape:
//! compiler-emitted kernels + dispatch + schedule live in
//! `OUT_DIR/generated.rs`, pulled in via `include!`. The hand-written
//! orchestration shell ([`ForagingState`]) owns the GPU context,
//! per-field SoA buffers, the kernel cache, the host-side position
//! cache, the shared event-ring, and the single decay-anchored view
//! storage.
//!
//! ## Fixture-specific shape vs the precedents
//!
//! - **Single Agent SoA** (`Ant : Agent`): `pos`, `vel`, `alive`. The
//!   `physics WanderAndDrop` rule reads + writes `pos` and reads
//!   `vel` + `alive` (where-clause guard). One Drop event is emitted
//!   per alive ant per tick (kind tag 0u in the WGSL emit body).
//! - **Single shared event ring** (Drop is the only emitted event).
//! - **Single @decay-anchored view** (`pheromone_deposits`,
//!   rate=0.88). Per-tick anchor multiply runs BEFORE the per-event
//!   fold. Each Drop emit increments `view_storage_primary[ant_slot]`
//!   by 1.0; with `decay = 0.88` the per-slot steady-state is
//!   `1 / (1 - 0.88) ≈ 8.33`.
//!
//! ## What the .sim file gates behind GAPs (NOT exercised here)
//!
//! - `entity Food : Item` + `entity Colony : Group` declarations
//!   parse + resolve today (per the bartering coverage probe), but
//!   neither has SoA storage allocated here — no field on either is
//!   read by the active rule body, and the would-be `pheromone_trail
//!   (ant: Agent, food: Food)` view (`storage = pair_map`) plus the
//!   `colony_intake(colony: Group)` view stay commented out in the
//!   .sim file behind `// GAP:` markers. Slices 2 + 3 of the
//!   implementation task probed those paths and surfaced the gaps —
//!   see the .sim file for the surfaced compiler-side blockers.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture Food-entity population. Sets the `pair_map`
/// `pheromone_trail` view's second-axis size: storage allocates
/// `agent_count × FOOD_COUNT` slots and the fold body composes
/// `view_storage_primary[ant * FOOD_COUNT + carried]`. Closes the
/// "Item-population-aware pair_map sizing" gap (commit `cd93cb04`
/// slice-2 wrap-up): before this fixture, the runtime hard-coded
/// `second_key_pop = agent_count`, treating the Food key as if it
/// were Agent-keyed and over-allocating storage to `agent_count²`.
///
/// The constant lives runtime-side (rather than as a `@cap(N)`
/// annotation in the .sim) because Item population is a per-fixture
/// allocation choice — the compiler stays population-agnostic and
/// the runtime supplies the count via the `second_key_pop` /
/// `slot_count` cfg fields the emit already consults. A future
/// fixture wanting a different food count just edits this constant
/// (and the matching `foraging_app` assertion).
pub const FOOD_COUNT: u32 = 16;

/// 16-byte WGSL `vec3<f32>` interop. Same shape as the sibling
/// runtimes use; duplicated here to keep each fixture-runtime crate
/// self-contained (no inter-fixture coupling beyond `engine` +
/// `dsl_compiler`).
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

/// Per-fixture state for the foraging_colony simulation.
pub struct ForagingState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-ant alive flag. The `where (self.alive)` guard on the
    /// WanderAndDrop handler reads this slot; ants with `alive == 0`
    /// neither integrate nor emit a Drop (so the view's per-slot
    /// steady-state would diverge from the analytical 8.33 if any
    /// slot were toggled off — left at 1u everywhere here for the
    /// uniform-population observable).
    alive_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    /// Single shared event-ring (Drop is the only emitted event).
    event_ring: EventRing,

    /// `pheromone_deposits(ant: Agent) -> f32` storage. `@decay`
    /// (rate=0.88) → has_anchor=true. No top-K storage hint → ids
    /// stays absent.
    pheromone_deposits: ViewStorage,
    pheromone_deposits_cfg_buf: wgpu::Buffer,
    pheromone_deposits_decay_cfg_buf: wgpu::Buffer,

    /// `colony_intake(colony: Group) -> f32` storage. `@decay`
    /// (rate=0.95) → has_anchor=true. SLICE 3 PROBE — first
    /// Group-targeted view anywhere. The Deposited event is never
    /// emitted by any active rule, so the per-tick fold runs with
    /// event_count=0 (no-op) and the per-tick decay multiplies the
    /// already-zero buffer. Wiring it through proves the
    /// Group-keyed view's compile + dispatch path works end-to-end
    /// even before a Deposited emitter lands. Note the .sim file's
    /// GAP: the Group-keyed view emit currently sizes storage as
    /// `agent_cap` slots (same gap as `pair_map` — no
    /// Group-population-aware sizing), so the readback would be
    /// agent-shaped today; that's fine for a zero-fold no-op.
    colony_intake: ViewStorage,
    colony_intake_cfg_buf: wgpu::Buffer,
    colony_intake_decay_cfg_buf: wgpu::Buffer,

    /// `pheromone_trail(ant: Agent, food: Food) -> f32` storage.
    /// `@decay(rate=0.88)` → has_anchor=true. Item-population-aware
    /// pair_map sizing (closes the `cd93cb04` slice-2 wrap-up gap):
    /// the view is keyed on `(Agent, Food)` and the fixture supplies
    /// the per-Food population via [`FOOD_COUNT`], so storage
    /// allocates `agent_count × FOOD_COUNT` slots (NOT
    /// `agent_count²`) and `second_key_pop == FOOD_COUNT`. The
    /// emit-side index composes
    /// `view_storage_primary[ant * FOOD_COUNT + carried]`.
    ///
    /// `Drop.carried` is still typed as `AgentId` in the event
    /// schema (full Item-SoA lower hasn't arrived for
    /// `self.engaged_with`-style reads), but the .sim hard-codes
    /// `carried: 0` so every event lands on food slot 0 of its
    /// ant's row. Steady state: per-`(ant, 0)` ≈ `1 / (1 - 0.88)
    /// ≈ 8.33`; every other `(ant, k)` slot stays at 0 (no producer
    /// targets them); total ≈ `agent_count × 8.33`. Once
    /// `carried` swaps to a real `ItemId`, the per-(ant, food)
    /// distribution emerges naturally from whatever target the
    /// physics rule selects.
    pheromone_trail: ViewStorage,
    pheromone_trail_cfg_buf: wgpu::Buffer,
    pheromone_trail_decay_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,
    pos_cache: Vec<Vec3>,
    dirty: bool,
    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Map a u32 hash output into a uniform `[-1, 1)` float — same
/// pattern the sibling runtimes use for keyed-PCG-derived initial
/// state.
fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl ForagingState {
    /// Construct an N-ant simulation with deterministic initial
    /// positions + velocities derived from `seed` via the engine's
    /// keyed PCG (P5: `per_agent_u32(seed, agent_id, tick=0,
    /// purpose)`). All slots start alive (alive=1u everywhere) so
    /// every slot emits one Drop per tick — the analytical
    /// observable in `foraging_app` keys on this uniformity.
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
                normalise(per_agent_u32(seed, agent_id, 0, b"forage_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"forage_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"forage_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"forage_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"forage_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"forage_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }
        let alive_init: Vec<u32> = vec![1u32; n];

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_runtime::alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cfg = physics_WanderAndDrop::PhysicsWanderAndDropCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foraging_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Shared event-ring + single @decay view storage ----
        let event_ring = EventRing::new(&gpu, "foraging_runtime");
        let pheromone_deposits = ViewStorage::new(
            &gpu,
            "foraging_runtime::pheromone_deposits",
            agent_count,
            true,  // has_anchor (carries @decay)
            false, // no top-K storage hint
        );

        let pd_cfg = fold_pheromone_deposits::FoldPheromoneDepositsCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let pheromone_deposits_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("foraging_runtime::pheromone_deposits_cfg"),
                contents: bytemuck::bytes_of(&pd_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let pd_decay_cfg = decay_pheromone_deposits::DecayPheromoneDepositsCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let pheromone_deposits_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("foraging_runtime::pheromone_deposits_decay_cfg"),
                contents: bytemuck::bytes_of(&pd_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // SLICE 3 — Group-keyed colony_intake view storage. Sized
        // to `agent_count` because the emit currently treats
        // Group-keying identically to Agent-keying (no
        // Group-population-aware sizing — see .sim file GAP). With
        // no Deposited emitter the buffer stays zero throughout.
        let colony_intake = ViewStorage::new(
            &gpu,
            "foraging_runtime::colony_intake",
            agent_count,
            true,  // has_anchor (carries @decay rate=0.95)
            false, // no top-K storage hint
        );
        let ci_cfg = fold_colony_intake::FoldColonyIntakeCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let colony_intake_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("foraging_runtime::colony_intake_cfg"),
                contents: bytemuck::bytes_of(&ci_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let ci_decay_cfg = decay_colony_intake::DecayColonyIntakeCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let colony_intake_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("foraging_runtime::colony_intake_decay_cfg"),
                contents: bytemuck::bytes_of(&ci_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Item-population-aware pair_map pheromone_trail storage.
        // Sized `agent_count × FOOD_COUNT` — the second axis is the
        // Food entity's per-fixture population, decoupled from the
        // Agent count. Closes the slice-2 wrap-up gap that previously
        // forced `agent_count²` over-allocation. The fold body
        // composes `view_storage_primary[ant * second_key_pop +
        // carried]` with `second_key_pop == FOOD_COUNT`; with
        // `Drop { carried: 0 }` from WanderAndDrop, only the
        // `(ant, 0)` column accumulates (≈8.33/slot) while every
        // `(ant, k)` for k != 0 stays at 0.
        let trail_slots = agent_count * FOOD_COUNT;
        let pheromone_trail = ViewStorage::new(
            &gpu,
            "foraging_runtime::pheromone_trail",
            trail_slots,
            true,  // has_anchor (carries @decay rate=0.88)
            false, // no top-K storage hint
        );
        let pt_cfg = fold_pheromone_trail::FoldPheromoneTrailCfg {
            event_count: 0,
            tick: 0,
            // pair_map: second key sized to the Food entity's
            // population (FOOD_COUNT), independent of agent_count —
            // the runtime supplies this constant rather than
            // hard-coding `agent_count`.
            second_key_pop: FOOD_COUNT,
            _pad: 0,
        };
        let pheromone_trail_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("foraging_runtime::pheromone_trail_cfg"),
                contents: bytemuck::bytes_of(&pt_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let pt_decay_cfg = decay_pheromone_trail::DecayPheromoneTrailCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: trail_slots, // pair_map: agent_cap × FOOD_COUNT
            _pad: 0,
        };
        let pheromone_trail_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("foraging_runtime::pheromone_trail_decay_cfg"),
                contents: bytemuck::bytes_of(&pt_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            alive_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            pheromone_deposits,
            pheromone_deposits_cfg_buf,
            pheromone_deposits_decay_cfg_buf,
            colony_intake,
            colony_intake_cfg_buf,
            colony_intake_decay_cfg_buf,
            pheromone_trail,
            pheromone_trail_cfg_buf,
            pheromone_trail_decay_cfg_buf,
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

    /// Per-ant pheromone deposit accumulator. Each tick one Drop
    /// event lands on each alive ant's slot; @decay multiplies the
    /// running total by 0.88 BEFORE the fold lands. Steady state
    /// per slot ≈ `1 / (1 - 0.88)` = 8.33.
    pub fn pheromone_deposits(&mut self) -> &[f32] {
        self.pheromone_deposits.readback(&self.gpu)
    }

    /// Group-keyed colony intake (SLICE 3 PROBE). With no
    /// Deposited emitter the buffer stays at the all-zeros initial
    /// state regardless of tick count; readback proves the
    /// Group-keyed view's compile + dispatch path runs cleanly
    /// end-to-end.
    pub fn colony_intakes(&mut self) -> &[f32] {
        self.colony_intake.readback(&self.gpu)
    }

    /// Per-(ant, food) pheromone_trail accumulator, flattened in
    /// row-major order: slot `[ant * FOOD_COUNT + food]` holds the
    /// decayed Drop count from `Drop { ant, carried = food }`.
    /// Length = `agent_count × FOOD_COUNT`. With WanderAndDrop
    /// emitting `Drop { ant: self, carried: 0 }` only the `(ant, 0)`
    /// column accumulates; each cell there converges to ~8.33
    /// (`@decay(rate = 0.88)` steady state). Every other
    /// `(ant, k)` slot stays at exactly 0.
    pub fn pheromone_trail(&mut self) -> &[f32] {
        self.pheromone_trail.readback(&self.gpu)
    }

    /// Number of agents (ants) in the simulation. Useful as the
    /// stride for [`Self::pheromone_trail`] readback indexing.
    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("foraging_runtime::positions::copy"),
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

impl CompiledSim for ForagingState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foraging_runtime::step"),
            });

        // Per-tick clear of the shared event_tail. The single Drop
        // emitter atomicAdds against this counter; the count gets
        // read back to size the fold dispatch.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1) WanderAndDrop — emits 1 Drop event per alive ant per
        // tick (`emit Drop { ant: self, carried: self, pos: new_pos }`).
        let bindings = physics_WanderAndDrop::PhysicsWanderAndDropBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_alive: &self.alive_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_wanderanddrop(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) seed_indirect_0 — keeps the indirect-args buffer warm
        // for the eventual `dispatch_workgroups_indirect` wire-up
        // (siblings dispatch this even though they don't consume the
        // indirect path yet — keeping the parity).
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

        // Per-tick event count = agent_count (every alive ant emits
        // exactly one Drop per tick; all ants stay alive in this
        // runtime).
        let event_count = self.agent_count;

        // (3a) decay_pheromone_deposits — anchor multiply BEFORE the
        // fold. `view_storage_primary[slot] *= 0.88`.
        let pd_decay_cfg = decay_pheromone_deposits::DecayPheromoneDepositsCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.pheromone_deposits_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&pd_decay_cfg),
        );
        let pd_decay_bindings =
            decay_pheromone_deposits::DecayPheromoneDepositsBindings {
                view_storage_primary: self.pheromone_deposits.primary(),
                cfg: &self.pheromone_deposits_decay_cfg_buf,
            };
        dispatch::dispatch_decay_pheromone_deposits(
            &mut self.cache,
            &pd_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3b) fold_pheromone_deposits — RMW
        // `view_storage_primary[event[3]] += 1.0` per Drop event.
        let pd_cfg = fold_pheromone_deposits::FoldPheromoneDepositsCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.pheromone_deposits_cfg_buf,
            0,
            bytemuck::bytes_of(&pd_cfg),
        );
        let pd_bindings = fold_pheromone_deposits::FoldPheromoneDepositsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.pheromone_deposits.primary(),
            view_storage_anchor: self.pheromone_deposits.anchor(),
            view_storage_ids: self.pheromone_deposits.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.pheromone_deposits_cfg_buf,
        };
        dispatch::dispatch_fold_pheromone_deposits(
            &mut self.cache,
            &pd_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (4a) decay_colony_intake — anchor multiply BEFORE the
        // (no-op) fold. With Deposited never emitted this multiplies
        // an all-zeros buffer; the dispatch is here to prove the
        // Group-keyed view's decay path runs end-to-end.
        let ci_decay_cfg = decay_colony_intake::DecayColonyIntakeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.colony_intake_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&ci_decay_cfg),
        );
        let ci_decay_bindings = decay_colony_intake::DecayColonyIntakeBindings {
            view_storage_primary: self.colony_intake.primary(),
            cfg: &self.colony_intake_decay_cfg_buf,
        };
        dispatch::dispatch_decay_colony_intake(
            &mut self.cache,
            &ci_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4b) fold_colony_intake — sized at event_count=0 (no
        // Deposited emitter wired). Every thread early-returns at
        // `event_idx >= cfg.event_count`. Wiring it through proves
        // the Group-keyed view's fold-dispatch path is healthy.
        let ci_cfg = fold_colony_intake::FoldColonyIntakeCfg {
            event_count: 0,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.colony_intake_cfg_buf,
            0,
            bytemuck::bytes_of(&ci_cfg),
        );
        let ci_bindings = fold_colony_intake::FoldColonyIntakeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.colony_intake.primary(),
            view_storage_anchor: self.colony_intake.anchor(),
            view_storage_ids: self.colony_intake.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.colony_intake_cfg_buf,
        };
        dispatch::dispatch_fold_colony_intake(
            &mut self.cache,
            &ci_bindings,
            &self.gpu.device,
            &mut encoder,
            1u32, // event_count=0 → 1 workgroup × 64 threads, all early-return
        );

        // (5a) decay_pheromone_trail — pair_map decay multiplies
        // every (ant, carried) slot. Dispatch covers `slot_count`
        // (= agent_cap × FOOD_COUNT) so the anchor reaches every
        // ant×food pair, not just the slot the per-tick emit
        // touches.
        let pt_decay_cfg = decay_pheromone_trail::DecayPheromoneTrailCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count * FOOD_COUNT,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.pheromone_trail_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&pt_decay_cfg),
        );
        let pt_decay_bindings = decay_pheromone_trail::DecayPheromoneTrailBindings {
            view_storage_primary: self.pheromone_trail.primary(),
            cfg: &self.pheromone_trail_decay_cfg_buf,
        };
        dispatch::dispatch_decay_pheromone_trail(
            &mut self.cache,
            &pt_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count * FOOD_COUNT,
        );

        // (5b) fold_pheromone_trail — RMW
        // `view_storage_primary[ant * FOOD_COUNT + carried] += 1.0`
        // per Drop event.
        let pt_cfg = fold_pheromone_trail::FoldPheromoneTrailCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: FOOD_COUNT,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.pheromone_trail_cfg_buf,
            0,
            bytemuck::bytes_of(&pt_cfg),
        );
        let pt_bindings = fold_pheromone_trail::FoldPheromoneTrailBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.pheromone_trail.primary(),
            view_storage_anchor: self.pheromone_trail.anchor(),
            view_storage_ids: self.pheromone_trail.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.pheromone_trail_cfg_buf,
        };
        dispatch::dispatch_fold_pheromone_trail(
            &mut self.cache,
            &pt_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.pheromone_deposits.mark_dirty();
        self.colony_intake.mark_dirty();
        self.pheromone_trail.mark_dirty();
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
/// between fixture runtimes via a one-line constructor swap. Mirrors
/// `boids_runtime::make_sim`.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(ForagingState::new(seed, agent_count))
}
