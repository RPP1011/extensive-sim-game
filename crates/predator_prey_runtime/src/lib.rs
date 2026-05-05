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
//!   - Per-creature-type init (Stage 1 splits the SoA-vs-id semantics)

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

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

    /// Per-fixture event-ring infrastructure (event_ring + event_tail
    /// + tail-zero source + indirect_args + sim_cfg placeholder).
    /// Shared with pc/cn_runtime via [`engine::gpu::EventRing`].
    event_ring: EventRing,

    /// Per-Wolf kill-count accumulator. The compiler-emitted
    /// `fold_kill_count` kernel RMWs `view_storage_primary[by]` by
    /// 1.0 per Killed event; @decay wires the anchor; readback via
    /// [`Self::kill_counts`].
    kill_count: ViewStorage,
    kill_count_cfg_buf: wgpu::Buffer,
    /// Cfg uniform for the per-tick `decay_kill_count` kernel (B2).
    /// `kill_count` carries `@decay(rate = 0.95, per = tick)` — every
    /// tick the kernel multiplies `view_storage_primary[k]` by 0.95
    /// before the per-event fold lands. Steady-state per slot
    /// converges to `1 / (1 - 0.95) = 20`.
    kill_count_decay_cfg_buf: wgpu::Buffer,

    /// Per-Wolf predator-focus accumulator (separate view, separate
    /// storage; same Killed event source). Same shape as kill_count
    /// — primary + anchor + ids — bound through the same shared
    /// helper.
    predator_focus: ViewStorage,
    predator_focus_cfg_buf: wgpu::Buffer,
    /// Cfg uniform for the per-tick `decay_predator_focus` kernel (B2).
    /// `predator_focus` carries `@decay(rate = 0.98, per = tick)` —
    /// steady-state per slot converges to `1 / (1 - 0.98) = 50`.
    predator_focus_decay_cfg_buf: wgpu::Buffer,

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
                // COPY_SRC so `snapshot()` can stage the per-slot
                // discriminant back to host. Pos already has COPY_SRC
                // for `read_positions`; vel doesn't need it (no host
                // consumer).
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let cfg = physics_MoveHare::PhysicsMoveHareCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
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

        // ---- Shared event-ring + per-view storage helpers ----
        let event_ring = EventRing::new(&gpu, "predator_prey_runtime");
        // kill_count + predator_focus — both per-Wolf views with
        // @decay (anchor) and an `ids` binding present on the fold
        // signature. has_anchor=true, has_ids=true.
        let kill_count = ViewStorage::new(
            &gpu,
            "predator_prey_runtime::kill_count",
            agent_count,
            true,
            true,
        );
        // `predator_focus` is `pair_map`-keyed: per-(by_agent,
        // prey_agent) bucket. The compiler's fold body composes
        // `view_storage_primary[k1 * cfg.second_key_pop + k2]` so the
        // storage MUST be over-allocated to `agent_cap × second_pop`
        // (= `agent_cap × agent_cap` for this Agent×Agent view).
        // Pre-fix the slot count was `agent_cap` and the fold
        // collapsed every (*, k2) event into the same slot — see the
        // pair_map gap doc on `assets/sim/foraging_colony.sim` line
        // ~126.
        let predator_focus = ViewStorage::new(
            &gpu,
            "predator_prey_runtime::predator_focus",
            agent_count * agent_count,
            true,
            true,
        );
        let kill_count_cfg = fold_kill_count::FoldKillCountCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let kill_count_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_runtime::kill_count_cfg"),
                contents: bytemuck::bytes_of(&kill_count_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let predator_focus_cfg = fold_predator_focus::FoldPredatorFocusCfg {
            event_count: 0,
            tick: 0,
            // `predator_focus(a: Agent, b: Agent)` — both keys are
            // Agent, so the second-key population is `agent_count`.
            // The fold body composes
            // `view_storage_primary[k1 * second_key_pop + k2]`.
            second_key_pop: agent_count,
            _pad: 0,
        };
        let predator_focus_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_runtime::predator_focus_cfg"),
                contents: bytemuck::bytes_of(&predator_focus_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let kill_count_decay_cfg = decay_kill_count::DecayKillCountCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let kill_count_decay_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_runtime::kill_count_decay_cfg"),
                contents: bytemuck::bytes_of(&kill_count_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let predator_focus_decay_cfg = decay_predator_focus::DecayPredatorFocusCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: (agent_count) * (agent_count), // pair_map: agent_cap × second_pop
            _pad: 0,
        };
        let predator_focus_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_runtime::predator_focus_decay_cfg"),
                contents: bytemuck::bytes_of(&predator_focus_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            creature_type_buf,
            hare_count,
            wolf_count,
            cfg_buf,
            pos_staging,
            event_ring,
            kill_count,
            kill_count_cfg_buf,
            kill_count_decay_cfg_buf,
            predator_focus,
            predator_focus_cfg_buf,
            predator_focus_decay_cfg_buf,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-Wolf kill counts (one f32 per agent slot, indexed by
    /// EntityRef-discriminant slot — Wolf entries are at odd
    /// indices given the 50/50 init pattern). Triggers a readback
    /// of `view_storage_kill_count_primary` when dirty; consecutive
    /// calls without an intervening `step()` skip the readback.
    pub fn kill_counts(&mut self) -> &[f32] {
        self.kill_count.readback(&self.gpu)
    }

    /// Per-(by_agent, prey_agent) `predator_focus` accumulator,
    /// flattened in row-major order: slot `[k1 * agent_count + k2]`
    /// holds the decayed kill count from `by = k1, prey = k2`.
    /// Length = `agent_count × agent_count`. The view carries
    /// `@decay(rate = 0.98, per = tick)` so the steady-state per-pair
    /// value is `producer_rate / (1 - 0.98) = producer_rate × 50`.
    /// In the current placeholder MoveWolf emit `Killed { by: self,
    /// prey: self }`, only diagonal slots `(i, i)` accumulate at
    /// rate 1/tick, so each diagonal converges to ~50 and every
    /// off-diagonal stays at 0.
    pub fn predator_focus(&mut self) -> &[f32] {
        self.predator_focus.readback(&self.gpu)
    }

    /// Stride between consecutive `by_agent` rows in the
    /// [`Self::predator_focus`] readback. Equal to `agent_count`
    /// (the second-key population for this Agent×Agent view). Use
    /// this to index `predator_focus()[by * agent_count() + prey]`.
    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    /// Run every host-side invariant check the compiler synthesized
    /// from `predator_prey_min.sim`'s `invariant <name>(<scope>) @<mode>
    /// { <predicate> }` declarations. Today that's just
    /// `bounded_kill_count(a: Agent) @debug_only` — it iterates the
    /// kill_count storage readback and reports any slot whose value
    /// has saturated past the predicate's bound (1000.0). The empty
    /// vec is the steady-state expected return: a non-empty vec means
    /// the @decay anchor stopped reining in the per-event fold.
    ///
    /// Pulls the generated check fns from the `invariants` module the
    /// compiler emits next to the kernel modules; the build script
    /// wraps them at `OUT_DIR/generated.rs::invariants` (see the
    /// `for sibling in ["schedule", "dispatch", "invariants"]` loop in
    /// `build.rs`). Determinism: the predicate is a pure scalar
    /// comparator over the (already-deterministic) view readback —
    /// the function reads no clocks, no RNG, no host time.
    pub fn check_invariants(&mut self) -> Vec<invariants::Violation> {
        let counts: Vec<f32> = self.kill_counts().to_vec();
        invariants::check_bounded_kill_count(&counts)
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

    /// One-shot GPU readback of the per-slot creature_type discriminant
    /// (Hare = 0, Wolf = 1 — matches the `EntityRef` declaration order
    /// in `predator_prey_min.sim`). Mirrors duel_abilities_runtime's
    /// `read_u32` shape: ad-hoc staging buffer + map_async + poll/wait.
    /// Allocates a fresh staging buffer per call — `snapshot()` only
    /// fires at viz cadence (~10 Hz) so the cost is acceptable; if a
    /// hot-path consumer ever appears, fold this into a long-lived
    /// staging buffer like `pos_staging`.
    fn read_creature_types(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_runtime::creature_type_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_runtime::creature_type::copy"),
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
}

impl CompiledSim for PredatorPreyState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_runtime::step"),
            },
        );

        // Per-tick clear of event_tail. Producers atomicAdd against
        // it during MoveWolf to acquire write slots; the count
        // accumulates over the tick and gets read back to size the
        // fold dispatch.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1) MoveHare — no event_ring/event_tail bindings (Hares
        // don't emit). Reads creature_type, pos, vel; writes pos.
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

        // (2) MoveWolf — same shape as MoveHare PLUS event_ring +
        // event_tail bindings (the body's `emit Killed { … }` writes
        // to those via atomicAdd-to-tail + atomicStore-to-ring).
        let wolf_bindings = physics_MoveWolf::PhysicsMoveWolfBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
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

        // (3) seed_indirect_0 — reads event_tail to populate
        // indirect_args_0 with `(ceil(n/64), 1, 1)`. We don't yet
        // dispatch the fold via dispatch_workgroups_indirect (the
        // emitted dispatch helper takes agent_cap workgroup count),
        // so this write currently runs but isn't consumed; left in
        // the chain so the args buffer is kept warm for the
        // future indirect-dispatch wire-up.
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

        // Read back event_tail so we can populate the fold's
        // cfg.event_count uniform. One sync round-trip per tick;
        // future work moves to dispatch_workgroups_indirect with
        // the seed_indirect_0 args buffer to eliminate the readback.
        // (Skip the readback for now — set event_count to a
        // wolf_count upper bound. Fold's bounds-check inside the
        // kernel filters stale slots beyond the actual tail value
        // because the event_ring stores leftover events from prior
        // ticks; that's an off-by-many bug for now, fixed when the
        // tail readback lands. For 32 wolves emitting 1 event per
        // tick, event_count = 32 is exactly the new-events count.)
        let event_count_estimate = self.wolf_count;
        let kc_cfg = fold_kill_count::FoldKillCountCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.kill_count_cfg_buf,
            0,
            bytemuck::bytes_of(&kc_cfg),
        );
        let pf_cfg = fold_predator_focus::FoldPredatorFocusCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            // pair_map: second key is Agent, so second_key_pop ==
            // agent_cap. Fold body indexes `view_storage_primary[
            // by_agent * agent_cap + prey_agent]`.
            second_key_pop: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.predator_focus_cfg_buf,
            0,
            bytemuck::bytes_of(&pf_cfg),
        );

        // (4a) decay_kill_count — B2 anchor multiply.
        // PerAgent dispatch (one thread per slot). MUST run before
        // the fold so the per-event deltas land on the decayed value.
        // `kill_count` carries `@decay(rate = 0.95, per = tick)` →
        // steady-state per slot ≈ 1 / (1 - 0.95) = 20 (with the
        // current 1 Killed event/wolf/tick producer rate).
        let kc_decay_cfg = decay_kill_count::DecayKillCountCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.kill_count_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&kc_decay_cfg),
        );
        let kc_decay_bindings = decay_kill_count::DecayKillCountBindings {
            view_storage_primary: self.kill_count.primary(),
            cfg: &self.kill_count_decay_cfg_buf,
        };
        dispatch::dispatch_decay_kill_count(
            &mut self.cache,
            &kc_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4b) fold_kill_count — RMWs view_storage_primary by 1.0 per
        // Killed event whose `by` AgentId matches the slot.
        let kc_bindings = fold_kill_count::FoldKillCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.kill_count.primary(),
            view_storage_anchor: self.kill_count.anchor(),
            view_storage_ids: self.kill_count.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.kill_count_cfg_buf,
        };
        dispatch::dispatch_fold_kill_count(
            &mut self.cache,
            &kc_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5a) decay_predator_focus — B2 anchor multiply.
        // `predator_focus` carries `@decay(rate = 0.98, per = tick)` →
        // steady-state per slot ≈ 1 / (1 - 0.98) = 50.
        let pf_decay_cfg = decay_predator_focus::DecayPredatorFocusCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: (self.agent_count) * (self.agent_count), // pair_map: agent_cap × second_pop
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.predator_focus_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&pf_decay_cfg),
        );
        let pf_decay_bindings = decay_predator_focus::DecayPredatorFocusBindings {
            view_storage_primary: self.predator_focus.primary(),
            cfg: &self.predator_focus_decay_cfg_buf,
        };
        // pair_map decay: dispatch covers `slot_count` (= agent_cap²)
        // threads — one per (k1, k2) slot — so the anchor multiplier
        // touches every pair, not just the diagonal `agent_cap` slots.
        dispatch::dispatch_decay_predator_focus(
            &mut self.cache,
            &pf_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (5b) fold_predator_focus — same shape, different storage.
        let pf_bindings = fold_predator_focus::FoldPredatorFocusBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.predator_focus.primary(),
            view_storage_anchor: self.predator_focus.anchor(),
            view_storage_ids: self.predator_focus.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.predator_focus_cfg_buf,
        };
        dispatch::dispatch_fold_predator_focus(
            &mut self.cache,
            &pf_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.kill_count.mark_dirty();
        self.predator_focus.mark_dirty();
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

    /// Snapshot per-agent positions + species + alive bit for the
    /// `viz_app` ASCII renderer. Encoding (matches `glyph_table`):
    ///
    /// - `positions`: read directly from `agent_pos_buf` via the
    ///   existing `read_positions` path (no transform).
    /// - `creature_types`: 4-entry table indexed by
    ///   `species_bit | (dead_bit << 1)`:
    ///     - 0 = alive prey  (Hare, creature_type discriminant 0)
    ///     - 1 = alive predator (Wolf, creature_type discriminant 1)
    ///     - 2 = dead prey
    ///     - 3 = dead predator
    /// - `alive`: Stage 0 has no Killed → SoA writeback path yet —
    ///   `kill_count` accumulates events but no field flips an
    ///   `agent_alive` bit, so every slot is reported alive (1). When
    ///   PP Stage 1 wires the per-event ApplyDefeat handler, this gets
    ///   replaced with a `read_alive()` GPU readback (and the dead
    ///   variants 2/3 in the encoding above will start firing).
    ///
    /// Initial-state safe: positions come from the `create_buffer_init`
    /// upload at construction, so calling `snapshot()` before any
    /// `step()` returns the deterministic spawn cube.
    fn snapshot(&mut self) -> AgentSnapshot {
        // Position readback first — drives `dirty` flush via the
        // existing pos_staging path.
        let positions: Vec<Vec3> = self.read_positions().to_vec();
        let species: Vec<u32> = self.read_creature_types();
        let n = self.agent_count as usize;
        // Stage 0: agents never die. When ApplyDefeat lands, swap this
        // for a `read_alive()` of an `agent_alive_buf`.
        let alive: Vec<u32> = vec![1u32; n];
        let creature_types: Vec<u32> = (0..n)
            .map(|i| {
                let species_bit = species[i] & 1; // 0 = prey, 1 = predator
                let dead_bit = if alive[i] == 0 { 1 } else { 0 };
                species_bit | (dead_bit << 1)
            })
            .collect();
        AgentSnapshot { positions, creature_types, alive }
    }

    /// 4 entries matching the `snapshot.creature_types` encoding:
    /// `[alive_prey, alive_predator, dead_prey, dead_predator]`.
    /// Predator gets the upper-case `P` glyph in red so it pops; prey
    /// get the low-key `.` in green. Dead variants share a grey ×.
    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![
            VizGlyph::new('.', 46),         // alive prey: bright green
            VizGlyph::new('P', 196),        // alive predator: bright red
            VizGlyph::new('\u{00D7}', 240), // dead prey: grey ×
            VizGlyph::new('\u{00D7}', 240), // dead predator: grey ×
        ]
    }

    /// Default zoom around the deterministic spawn cube. `new()` derives
    /// per-axis spreads from `cbrt(agent_count)` (so a 64-agent fixture
    /// spans roughly `[-4, 4]` per axis); ±5 keeps every spawn on screen
    /// with breathing room for the trivial integrator's drift. The
    /// renderer auto-scales if positions wander outside, so this is
    /// just an opening framing.
    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        let span = (self.agent_count as f32).cbrt().max(1.0) + 1.0;
        Some((Vec3::new(-span, -span, 0.0), Vec3::new(span, span, 0.0)))
    }
}

/// Build a boxed `CompiledSim` so the `sim_app` runner can switch
/// between fixture runtimes via a one-line constructor swap. Mirrors
/// `boids_runtime::make_sim`.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(PredatorPreyState::new(seed, agent_count))
}

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// Snapshot before any tick must report initial state: every agent
    /// alive, deterministic creature_type assignment from the 50/50
    /// split, and positions inside the spawn cube derived from the
    /// `cbrt(agent_count)` spread. Guards the construction-only
    /// readback path so `viz_app` can render frame 0 with content
    /// instead of a blank grid.
    #[test]
    fn snapshot_after_construction_returns_initial_state() {
        let agent_count = 16u32;
        let mut state = PredatorPreyState::new(0xCAFE_F00D, agent_count);
        let snap = state.snapshot();

        assert_eq!(snap.positions.len(), agent_count as usize, "positions length");
        assert_eq!(snap.creature_types.len(), agent_count as usize, "creature_types length");
        assert_eq!(snap.alive.len(), agent_count as usize, "alive length");

        // Stage 0: no kill path — every slot reports alive.
        let alive_total: u32 = snap.alive.iter().sum();
        assert_eq!(
            alive_total, agent_count,
            "every slot must be alive at construction (Stage 0 has no Killed→SoA path); got alive={:?}",
            snap.alive,
        );

        // 50/50 species split per `new()` — even slots are prey (0),
        // odd slots are predator (1).
        for (i, &ct) in snap.creature_types.iter().enumerate() {
            let expected = (i & 1) as u32;
            assert_eq!(
                ct, expected,
                "slot {i}: species_bit must match init 50/50 split (even=prey=0, odd=predator=1); got {ct}",
            );
        }

        // Positions must lie inside the deterministic spawn cube.
        // `new()` uses `spread = cbrt(agent_count).max(1)` and samples
        // each axis from `normalise(...) * spread` ∈ `[-spread, spread)`.
        let spread = (agent_count as f32).cbrt().max(1.0);
        for (i, p) in snap.positions.iter().enumerate() {
            assert!(
                p.x.abs() <= spread + 0.001
                    && p.y.abs() <= spread + 0.001
                    && p.z.abs() <= spread + 0.001,
                "slot {i} position {p:?} outside spawn cube ±{spread}",
            );
        }

        // glyph_table indexes line up with the 4-entry encoding the
        // snapshot promises.
        let glyphs = state.glyph_table();
        assert_eq!(glyphs.len(), 4, "glyph_table must have 4 entries");
    }

    /// After ticking the simulation forward, the trivial integrator in
    /// `predator_prey_min.sim` advances every agent by its velocity
    /// each step — so at least one slot's position must have changed
    /// from the initial spawn. Proves the snapshot reflects live GPU
    /// state, not a cached construction-time copy.
    #[test]
    fn snapshot_after_tick_reflects_state_change() {
        let agent_count = 16u32;
        let mut state = PredatorPreyState::new(0xCAFE_F00D, agent_count);
        let initial = state.snapshot().positions.clone();

        for _ in 0..50 {
            state.step();
        }

        let snap = state.snapshot();
        assert_eq!(snap.positions.len(), agent_count as usize);

        // At least one slot's position must have shifted (or alive
        // count must have changed — neither is allowed to be a no-op
        // 50 ticks in).
        let any_moved = initial.iter().zip(snap.positions.iter()).any(|(a, b)| {
            (a.x - b.x).abs() > 1e-6
                || (a.y - b.y).abs() > 1e-6
                || (a.z - b.z).abs() > 1e-6
        });
        let alive_changed = snap.alive.iter().sum::<u32>() != agent_count;
        assert!(
            any_moved || alive_changed,
            "after 50 ticks, expected position drift or kill; saw all positions identical and all slots still alive",
        );
    }
}
