//! Per-fixture runtime for `assets/sim/voxel_probe.sim` — Phase B of
//! the voxel-engine integration plan
//! (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`).
//!
//! ## What this exercises
//!
//! End-to-end pipe of the GPU `apply_ability` dispatcher's voxel-event
//! producers into the host-side `engine_voxel::VoxelTerrain` adapter:
//!
//!   1. PlaceTestVoxel verb fires at tick=0 → dispatcher writes one
//!      EffectPlaceVoxelApplied chronicle record (kind=60).
//!   2. HarvestTestVoxel verb fires at tick % 10 == 5 → dispatcher
//!      writes one EffectHarvestApplied chronicle record (kind=59).
//!   3. After each `step()` the runtime drains the event ring and calls
//!      `VoxelTerrain::apply_voxel_chronicle_record(rec, caster_pos)`
//!      per voxel-relevant record.
//!
//! ## Caller signature: option (A) — runtime resolves caster_pos
//!
//! Per the plan's design discussion, the chronicle consumer takes a
//! `Vec3` for the caster's position rather than a borrow on the full
//! `SimState`. The runtime owns the agent SoA and resolves
//! `caster_slot → pos` via `state.agent_pos(...)` before calling. This
//! keeps `engine_voxel` free of any `engine::SimState` dep and matches
//! the wave_defense `drain_summon_records` precedent (host-side
//! resolution before mutating world state).
//!
//! ## Per-tick chain
//!
//! Mirrors `spy_network_runtime`'s shape at N=1:
//!
//!   1. Per-tick clears (event_tail, ring headers, mask bitmaps,
//!      scoring_output).
//!   2. fused_mask_verb_HarvestTestVoxel — PerAgent, writes mask_0
//!      (HarvestTestVoxel) + mask_1 (PlaceTestVoxel). Verb gates
//!      keyed off `world.tick == 0` (Place) and `world.tick > 0 &&
//!      world.tick % 10 == 5` (Harvest); at most one mask fires per
//!      tick by design.
//!   3. scoring — PerAgent argmax over the two rows; emits ActionSelected.
//!   4. physics_verb_chronicle_HarvestTestVoxel — gates action_id==0u,
//!      walks Harvest's program, writes EffectHarvestApplied (kind=59).
//!   5. physics_verb_chronicle_PlaceTestVoxel — gates action_id==1u,
//!      walks Place's program, writes EffectPlaceVoxelApplied (kind=60).
//!   6. seed_indirect_0 — keeps the indirect_args buffer warm.
//!
//! After GPU work completes, the runtime drains the event ring on the
//! host (one tail readback + one ring readback per step) and applies
//! the voxel records to `state.terrain`.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::{AbilityId, PackedAbilityRegistry};
use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext, EVENT_RING_CAP_SLOTS, EVENT_STRIDE_U32};
use engine::state::SimState;
use engine::terrain::TerrainQuery;
use engine::GpuContext;
use engine_voxel::{VoxelMirror, VoxelTerrain};
use glam::Vec3;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// AbilityRegistry slot order — pinned by the `binding_check`-shape
/// asserts in `assert_ability_registry_matches_sim_constants`.
/// Alphabetised by file name (HarvestTestVoxel.ability,
/// PlaceTestVoxel.ability), so:
///   HarvestTestVoxel → 1
///   PlaceTestVoxel   → 2
pub const HARVEST_TEST_VOXEL_ABILITY_ID: u32 = 1;
pub const PLACE_TEST_VOXEL_ABILITY_ID: u32 = 2;

/// EventKindId of the voxel chronicle kinds we drain — matches
/// `crates/engine/src/cascade/handler.rs::EventKindId`.
pub const KIND_EFFECT_HARVEST_APPLIED: u32 = 59;
pub const KIND_EFFECT_PLACE_VOXEL_APPLIED: u32 = 60;

/// Per-fixture state for the voxel_probe simulation.
pub struct VoxelProbeState {
    gpu: GpuContext,

    /// Engine-side state — its `terrain: Arc<dyn TerrainQuery>` field
    /// is wired to a `VoxelTerrain` so future passes that consult
    /// terrain through the engine seam see the voxel backend. The
    /// drain path mutates `self.terrain` (the owned mutable copy
    /// below); Phase C/E will unify these once interior mutability
    /// or a GPU-mirror surface lands.
    #[allow(dead_code)]
    state: SimState,
    terrain: VoxelTerrain,
    /// Phase C — GPU-resident mirror of `terrain.grid()`. Updated
    /// per-tick via `mark_dirty` (during chronicle drain) followed by
    /// a single `flush_dirty` at end of tick. The buffer is wired
    /// into `KernelBindingsContext::voxel_grid` so future kernels
    /// (Phase D's terrain query lowerings) can read voxel state
    /// without a CPU↔GPU round-trip.
    mirror: VoxelMirror,
    /// Wall-clock cost of the most recent `flush_dirty` call (in
    /// nanoseconds). Exposed via [`Self::last_flush_ns`] so the
    /// behavioral pin tests + Phase E benchmarks can baseline the
    /// per-tick upload overhead at the fixture's typical mutation
    /// rate. Zero before the first tick.
    last_flush_ns: u128,

    // -- Agent SoA (GPU-side) --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask + scoring buffers --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring (with COPY_SRC for host drain) --
    event_ring: EventRing,
    /// Single-shot ring/tail copies live in dedicated COPY_SRC buffers
    /// so the host drain doesn't fight the producer's COPY_DST usage.
    /// Mirrors wave_defense's `event_ring_buf` + `event_tail_buf`
    /// staging pattern.
    event_ring_staging: wgpu::Buffer,
    event_tail_staging: wgpu::Buffer,
    /// Number of ring slots we read back per drain call. The fixture
    /// emits at most 1 record per tick (one of two verbs fires); 64
    /// slots is two orders of magnitude more headroom than needed.
    event_ring_readback_slots: u32,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_harvest_cfg_buf: wgpu::Buffer,
    chronicle_place_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry on GPU --
    registry_gpu: PackedAbilityRegistryGpu,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,

    /// Cached caster position for the lone agent at slot 0. The
    /// fixture seeds it at world origin and never moves it, so we
    /// stash it once at construction rather than re-reading per drain.
    caster_pos: Vec3,
}

impl VoxelProbeState {
    /// Construct a voxel_probe runtime with one agent at world origin.
    /// `seed` plumbs through to the cfg uniforms (no stochastic verbs
    /// in this fixture, so it's informational only).
    pub fn new(seed: u64) -> Self {
        Self::try_new(seed).expect("init wgpu adapter + device")
    }

    /// Fallible constructor — returns `None` when no compatible wgpu
    /// adapter is on the host. Lets the behavioral pin tests degrade
    /// to a skip-with-message instead of a panic.
    pub fn try_new(seed: u64) -> Option<Self> {
        // Asserts run before GPU init so drift surfaces early.
        assert_ability_registry_matches_sim_constants();

        let gpu = GpuContext::new_blocking().ok()?;

        // Build the AbilityRegistry from the .ability corpus and
        // upload it to GPU storage buffers.
        let built = build_voxel_probe_registry();
        let packed = PackedAbilityRegistry::pack(&built);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "voxel_probe_runtime");

        // Single-agent SoA. agent_count=1; the SoA still binds the
        // standard stat columns (BGL contract).
        let agent_count: u32 = 1;
        let n = agent_count as usize;

        // Opt the engine state into the voxel backend (Arc seam — no
        // `new_with_terrain` constructor exists, the existing pattern
        // stays simple). Drain path mutates `terrain` directly; the
        // engine-side Arc is a second instance held for future passes
        // that walk terrain through the SimState seam.
        //
        // Extent picked at 16³ for Phase C: the GPU mirror buffer is
        // `16 * 16 * 16 * 4 = 16 KiB`, two orders of magnitude smaller
        // than the 64 MiB the 256³ default would allocate. The fixture
        // only writes near the world origin, so 16³ is plenty of room
        // and keeps the mirror cheap on hosts without dedicated VRAM.
        const PROBE_EXTENT: u32 = 16;
        let mut state = SimState::new(agent_count, seed);
        let terrain = VoxelTerrain::with_extent(PROBE_EXTENT);
        state.terrain = Arc::new(VoxelTerrain::with_extent(PROBE_EXTENT));
        // Allocate the GPU mirror once at construction time; subsequent
        // ticks only push dirty chunks via `flush_dirty`.
        let mirror = VoxelMirror::new(&gpu, terrain.grid());

        // Spawn the lone agent at world origin so `caster_pos` is
        // pin-load-bearing.
        let agent_id = state
            .spawn_agent(engine::state::agent::AgentSpawn {
                pos: Vec3::ZERO,
                ..Default::default()
            })
            .expect("AgentSlotPool with cap=1 has room for one agent");
        let caster_pos = state
            .agent_pos(agent_id)
            .expect("freshly-spawned agent has a position");

        let hp_init = vec![100.0_f32; n];
        let alive_init = vec![1u32; n];
        let mana_init = vec![0.0_f32; n];
        let max_hp_init = vec![100.0_f32; n];
        let zeros_f32 = vec![0.0_f32; n];

        let mk_storage = |label: &str, contents: &[u8]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let agent_hp_buf =
            mk_storage("voxel_probe::agent_hp", bytemuck::cast_slice(&hp_init));
        let agent_alive_buf =
            mk_storage("voxel_probe::agent_alive", bytemuck::cast_slice(&alive_init));
        let agent_mana_buf =
            mk_storage("voxel_probe::agent_mana", bytemuck::cast_slice(&mana_init));
        let agent_max_hp_buf =
            mk_storage("voxel_probe::agent_max_hp", bytemuck::cast_slice(&max_hp_init));
        let agent_attack_damage_buf = mk_storage(
            "voxel_probe::agent_attack_damage",
            bytemuck::cast_slice(&zeros_f32),
        );
        let agent_ability_power_buf = mk_storage(
            "voxel_probe::agent_ability_power",
            bytemuck::cast_slice(&zeros_f32),
        );
        let agent_armor_buf = mk_storage(
            "voxel_probe::agent_armor",
            bytemuck::cast_slice(&zeros_f32),
        );
        let agent_magic_resist_buf = mk_storage(
            "voxel_probe::agent_magic_resist",
            bytemuck::cast_slice(&zeros_f32),
        );
        let agent_move_speed_buf = mk_storage(
            "voxel_probe::agent_move_speed",
            bytemuck::cast_slice(&zeros_f32),
        );

        // Mask bitmaps — 2 verbs (HarvestTestVoxel=0, PlaceTestVoxel=1).
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mk_mask = |label: &str| -> wgpu::Buffer {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: mask_bitmap_bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mask_0_bitmap_buf = mk_mask("voxel_probe::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("voxel_probe::mask_1_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 u32 per agent (matches scoring kernel
        // contract).
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> =
            vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Event ring + per-tick host-drain staging buffers. Cap the
        // readback at 64 slots — Phase B emits at most 1 record per
        // tick (one of two verbs fires); 64 is two orders of magnitude
        // more headroom than needed.
        let event_ring = EventRing::new(&gpu, "voxel_probe");
        let event_ring_readback_slots: u32 = 64;
        let stage_ring_bytes =
            (event_ring_readback_slots as u64) * (EVENT_STRIDE_U32 as u64) * 4;
        let event_ring_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::event_ring_staging"),
            size: stage_ring_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let event_tail_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Per-kernel cfg uniforms. Init values plumb agent_cap; per-tick
        // step() overwrites tick before each dispatch.
        let mask_cfg_init =
            fused_mask_verb_PlaceTestVoxel::FusedMaskVerbPlaceTestVoxelCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0,
                _pad: 0,
            };
        let mask_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::mask_cfg"),
                contents: bytemuck::bytes_of(&mask_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let chronicle_harvest_cfg_init =
            physics_verb_chronicle_HarvestTestVoxel::PhysicsVerbChronicleHarvestTestVoxelCfg {
                event_count: agent_count,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let chronicle_harvest_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::chronicle_harvest_cfg"),
                contents: bytemuck::bytes_of(&chronicle_harvest_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let chronicle_place_cfg_init =
            physics_verb_chronicle_PlaceTestVoxel::PhysicsVerbChroniclePlaceTestVoxelCfg {
                event_count: agent_count,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let chronicle_place_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::chronicle_place_cfg"),
                contents: bytemuck::bytes_of(&chronicle_place_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::seed_cfg"),
                contents: bytemuck::bytes_of(&seed_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Some(Self {
            gpu,
            state,
            terrain,
            mirror,
            last_flush_ns: 0,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_max_hp_buf,
            agent_attack_damage_buf,
            agent_ability_power_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            event_ring_staging,
            event_tail_staging,
            event_ring_readback_slots,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_harvest_cfg_buf,
            chronicle_place_cfg_buf,
            seed_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
            caster_pos,
        })
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Borrow the host-side voxel terrain — the one the chronicle
    /// drain mutates. Behavioral pins query height_at through this.
    pub fn terrain(&self) -> &VoxelTerrain {
        &self.terrain
    }

    /// Convenience accessor — `self.terrain().height_at(x, y)`.
    pub fn height_at(&self, x: f32, y: f32) -> f32 {
        self.terrain.height_at(x, y)
    }

    /// Drive one tick of the GPU pipeline + host-side voxel drain.
    ///
    /// Per-tick chain:
    ///   1. GPU dispatches (mask → scoring → chronicle dispatchers).
    ///   2. Host drain — applies chronicle records to the CPU
    ///      `VoxelTerrain` AND marks each touched chunk in `mirror`.
    ///   3. `mirror.flush_dirty(...)` — pushes dirty chunks to GPU
    ///      so the *next* tick's kernels see fresh voxel state.
    pub fn step(&mut self) {
        self.run_gpu_tick();
        let drained = self.drain_voxel_records();
        // Drained returns just for telemetry; the chronicle records
        // are applied in-place to `self.terrain` (and dirty chunks
        // marked in `self.mirror`).
        let _ = drained;
        // Flush the mirror at end of tick so the next tick's GPU
        // dispatches see the freshly-mutated voxel state.
        let t0 = Instant::now();
        self.mirror
            .flush_dirty(&self.gpu, self.terrain.grid());
        self.last_flush_ns = t0.elapsed().as_nanos();
        self.tick += 1;
    }

    /// Wall-clock cost of the most recent `flush_dirty` call, in
    /// nanoseconds. Zero before the first tick. Used by the Phase C
    /// behavioral pin tests to baseline per-tick upload overhead so
    /// Phase E can decide whether the CPU-mirror's per-mutation cost
    /// is the bottleneck (per the cross-cutting risk #1 in the plan).
    pub fn last_flush_ns(&self) -> u128 {
        self.last_flush_ns
    }

    /// Borrow the GPU-resident voxel mirror — used by the Phase C
    /// `cpu_gpu_voxel_state_matches` pin to read cells back through
    /// the GPU compute path.
    pub fn mirror(&self) -> &VoxelMirror {
        &self.mirror
    }

    /// Drive one tick of GPU dispatches end-to-end. Encodes per-tick
    /// clears + the 5 dispatches in the schedule, then submits and
    /// blocks on the device poll.
    fn run_gpu_tick(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("voxel_probe::step::dispatch"),
            },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = (self.agent_count.saturating_mul(4)).max(8);
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, max_slots_per_tick);
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [&self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf,
                0,
                buf,
                0,
                mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf,
            0,
            &self.scoring_output_buf,
            0,
            scoring_output_bytes.max(16),
        );

        let agent_buffers = AgentBuffers {
            hp_buf: Some(&self.agent_hp_buf),
            max_hp_buf: Some(&self.agent_max_hp_buf),
            alive_buf: Some(&self.agent_alive_buf),
            mana_buf: Some(&self.agent_mana_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            armor_buf: Some(&self.agent_armor_buf),
            magic_resist_buf: Some(&self.agent_magic_resist_buf),
            move_speed_buf: Some(&self.agent_move_speed_buf),
            ..Default::default()
        };
        // No kernel in this fixture binds `voxel_grid` yet; the wire-up
        // is forward-compat for Phase D, when terrain-query lowerings
        // start emitting that binding.
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
            voxel_grid: Some(self.mirror.buffer()),
        };

        // (2) Mask round (fused PerAgent — both verbs are self-cast).
        let mask_cfg =
            fused_mask_verb_PlaceTestVoxel::FusedMaskVerbPlaceTestVoxelCfg {
                agent_cap: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                _pad: 0,
            };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_extras =
            fused_mask_verb_PlaceTestVoxel::FusedMaskVerbPlaceTestVoxelExtras {
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                cfg: &self.mask_cfg_buf,
            };
        let mask_bindings =
            fused_mask_verb_PlaceTestVoxel::FusedMaskVerbPlaceTestVoxelBindings::from_context_with_extras(
                &ctx,
                &mask_extras,
            );
        dispatch::dispatch_fused_mask_verb_placetestvoxel(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) Scoring — argmax over 2 rows per agent.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_extras = scoring::ScoringExtras {
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        let scoring_bindings =
            scoring::ScoringBindings::from_context_with_extras(&ctx, &scoring_extras);
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) HarvestTestVoxel chronicle dispatcher (action_id==0u).
        let harvest_cfg =
            physics_verb_chronicle_HarvestTestVoxel::PhysicsVerbChronicleHarvestTestVoxelCfg {
                event_count: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                agent_cap: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_harvest_cfg_buf,
            0,
            bytemuck::bytes_of(&harvest_cfg),
        );
        let harvest_extras =
            physics_verb_chronicle_HarvestTestVoxel::PhysicsVerbChronicleHarvestTestVoxelExtras {
                cfg: &self.chronicle_harvest_cfg_buf,
            };
        let harvest_bindings =
            physics_verb_chronicle_HarvestTestVoxel::PhysicsVerbChronicleHarvestTestVoxelBindings::from_context_with_extras(
                &ctx,
                &harvest_extras,
            );
        dispatch::dispatch_physics_verb_chronicle_harvesttestvoxel(
            &mut self.cache,
            &harvest_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5) PlaceTestVoxel chronicle dispatcher (action_id==1u).
        let place_cfg =
            physics_verb_chronicle_PlaceTestVoxel::PhysicsVerbChroniclePlaceTestVoxelCfg {
                event_count: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                agent_cap: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_place_cfg_buf,
            0,
            bytemuck::bytes_of(&place_cfg),
        );
        let place_extras =
            physics_verb_chronicle_PlaceTestVoxel::PhysicsVerbChroniclePlaceTestVoxelExtras {
                cfg: &self.chronicle_place_cfg_buf,
            };
        let place_bindings =
            physics_verb_chronicle_PlaceTestVoxel::PhysicsVerbChroniclePlaceTestVoxelBindings::from_context_with_extras(
                &ctx,
                &place_extras,
            );
        dispatch::dispatch_physics_verb_chronicle_placetestvoxel(
            &mut self.cache,
            &place_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6) seed_indirect_0 — keeps the indirect_args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf,
            0,
            bytemuck::bytes_of(&seed_cfg),
        );
        let seed_extras = seed_indirect_0::SeedIndirect0Extras {
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        let seed_bindings =
            seed_indirect_0::SeedIndirect0Bindings::from_context_with_extras(
                &ctx,
                &seed_extras,
            );
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("poll after voxel_probe step submit");
    }

    /// Drain the event ring on the host. Reads back the tail counter +
    /// the first `event_ring_readback_slots` records, filters for
    /// EffectPlaceVoxelApplied (kind=60) + EffectHarvestApplied
    /// (kind=59), and dispatches each to
    /// `VoxelTerrain::apply_voxel_chronicle_record(rec, caster_pos)`.
    ///
    /// Returns the number of voxel-relevant records applied this tick.
    fn drain_voxel_records(&mut self) -> u32 {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("voxel_probe::drain"),
            },
        );
        encoder.copy_buffer_to_buffer(
            self.event_ring.tail(),
            0,
            &self.event_tail_staging,
            0,
            4,
        );
        let cap = self.event_ring_readback_slots.min(EVENT_RING_CAP_SLOTS);
        let stage_ring_bytes = (cap as u64) * (EVENT_STRIDE_U32 as u64) * 4;
        encoder.copy_buffer_to_buffer(
            self.event_ring.ring(),
            0,
            &self.event_ring_staging,
            0,
            stage_ring_bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        // Map tail.
        let tail_value = {
            let slice = self.event_tail_staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let _ = receiver.recv().expect("map_async tail result");
            let mapped = slice.get_mapped_range();
            let v: u32 = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
            drop(mapped);
            self.event_tail_staging.unmap();
            v
        };

        let actual_records = tail_value.min(cap) as usize;
        // Always unmap the ring staging (even on early-return) so the
        // next tick's copy_buffer_to_buffer doesn't panic on a still-
        // mapped destination.
        let records: Vec<[u32; EVENT_STRIDE_U32 as usize]> = {
            let slice = self.event_ring_staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let _ = receiver.recv().expect("map_async ring result");
            let out = if actual_records == 0 {
                Vec::new()
            } else {
                let mapped = slice.get_mapped_range();
                let words: &[u32] = bytemuck::cast_slice(&mapped);
                let mut out: Vec<[u32; EVENT_STRIDE_U32 as usize]> =
                    Vec::with_capacity(actual_records);
                for r in 0..actual_records {
                    let base = r * (EVENT_STRIDE_U32 as usize);
                    let mut rec = [0u32; EVENT_STRIDE_U32 as usize];
                    rec.copy_from_slice(&words[base..base + EVENT_STRIDE_U32 as usize]);
                    out.push(rec);
                }
                drop(mapped);
                out
            };
            self.event_ring_staging.unmap();
            out
        };

        let mut applied: u32 = 0;
        for rec in &records {
            let kind = rec[0];
            if std::env::var("VOXEL_PROBE_TRACE").is_ok() {
                eprintln!(
                    "[voxel_probe drain tick={}] kind={} caster={} \
                     payload_a={} payload_b={}",
                    self.tick, kind, rec[2], rec[3], rec[4],
                );
            }
            if kind != KIND_EFFECT_PLACE_VOXEL_APPLIED
                && kind != KIND_EFFECT_HARVEST_APPLIED
            {
                continue;
            }
            // Single-agent fixture: `caster_slot = rec[2]` is always 0
            // and the position is cached at construction. Multi-agent
            // fixtures resolve caster_slot → pos against the agent SoA
            // here.
            self.terrain.apply_voxel_chronicle_record_with_mirror(
                rec,
                self.caster_pos,
                &mut self.mirror,
            );
            applied += 1;
        }
        applied
    }
}

/// Single binding-check entry — asserts every .ability under
/// `assets/ability_test/voxel_probe/` lands at the expected
/// AbilityId slot. Mirrors `spy_network_runtime`'s
/// `assert_ability_registry_matches_sim_constants`.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_voxel_probe_built();
    let harvest_id = *built
        .names
        .get("HarvestTestVoxel")
        .expect("HarvestTestVoxel registered in name table");
    assert_eq!(
        harvest_id,
        AbilityId::new(HARVEST_TEST_VOXEL_ABILITY_ID).expect("non-zero AbilityId"),
        "HarvestTestVoxel's AbilityId drifted from slot {}",
        HARVEST_TEST_VOXEL_ABILITY_ID,
    );
    let place_id = *built
        .names
        .get("PlaceTestVoxel")
        .expect("PlaceTestVoxel registered in name table");
    assert_eq!(
        place_id,
        AbilityId::new(PLACE_TEST_VOXEL_ABILITY_ID).expect("non-zero AbilityId"),
        "PlaceTestVoxel's AbilityId drifted from slot {}",
        PLACE_TEST_VOXEL_ABILITY_ID,
    );
}

const PLACE_TEST_VOXEL_SRC: &str = include_str!(
    "../../../assets/ability_test/voxel_probe/PlaceTestVoxel.ability"
);
const HARVEST_TEST_VOXEL_SRC: &str = include_str!(
    "../../../assets/ability_test/voxel_probe/HarvestTestVoxel.ability"
);

/// Build the `BuiltRegistry` (registry + name table) for the
/// voxel_probe corpus. Used by both the runtime constructor and the
/// binding-check assert.
fn build_voxel_probe_built() -> dsl_compiler::ability_registry::BuiltRegistry {
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };
    // Source-order names list (alphabetised) — same order as
    // `build_registry`'s slot assignment. HarvestTestVoxel sorts
    // before PlaceTestVoxel.
    let files = vec![
        (
            "HarvestTestVoxel.ability".to_string(),
            parse("HarvestTestVoxel.ability", HARVEST_TEST_VOXEL_SRC),
        ),
        (
            "PlaceTestVoxel.ability".to_string(),
            parse("PlaceTestVoxel.ability", PLACE_TEST_VOXEL_SRC),
        ),
    ];
    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over voxel_probe corpus")
}

fn build_voxel_probe_registry() -> engine::ability::registry::AbilityRegistry {
    build_voxel_probe_built().registry
}

// ---------------------------------------------------------------------------
// Behavioral pin tests — the FlatPlane killers.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod voxel_pins {
    use super::*;

    /// **Phase B semantic pin #1.** After PlaceTestVoxel fires (tick=0)
    /// and before HarvestTestVoxel can run (first cadence at tick=5),
    /// `terrain.height_at(0.0, 0.0)` must be ≥ 1.0 — the placed voxel
    /// at (0,0,0) lifts the column's top face to z=1.
    ///
    /// FlatPlane returns 0.0 → fails. Counter-based pins like "place
    /// records drained > 0" are the probe-fooling pattern; this pin
    /// requires the actual VoxelGrid mutation to take effect AND
    /// height_at to walk the column correctly.
    #[test]
    fn placed_voxel_changes_height_at() {
        let mut state = match VoxelProbeState::try_new(0) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[voxel_probe placed_voxel_changes_height_at] skipping: \
                     no wgpu adapter available on this host."
                );
                return;
            }
        };
        // Drive past tick 0 (Place fires) but before tick 5 (Harvest's
        // first cadence). Pick tick=3 — Place's record has been drained
        // and Harvest hasn't been emitted yet.
        for _ in 0..3 {
            state.step();
        }
        let h = state.height_at(0.0, 0.0);
        assert!(
            h >= 1.0,
            "after PlaceTestVoxel fires (tick=0) the voxel at column \
             (0,0) should lift height_at to >= 1.0; saw {h}. \
             FlatPlane returns 0.0 — this assertion is the load-bearing \
             check that the chronicle drain wired all the way through \
             into VoxelGrid mutation.",
        );
        eprintln!("[voxel_probe] post-place height_at(0,0) = {h}");
    }

    /// **Phase B semantic pin #2.** After PlaceTestVoxel fires (tick=0)
    /// and HarvestTestVoxel fires twice (ticks 5 and 15), the placed
    /// voxel must be cleared — `terrain.height_at(0.0, 0.0)` returns to
    /// 0.0. The second harvest is a no-op (cell already clear, no kind
    /// match) but ensures the `amount=1` per-cast contract isn't
    /// over-removing.
    ///
    /// Catches "harvest emits a chronicle record but the consumer
    /// doesn't actually clear cells" — the consumer-half regression.
    #[test]
    fn harvested_voxel_disappears() {
        let mut state = match VoxelProbeState::try_new(0) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[voxel_probe harvested_voxel_disappears] skipping: \
                     no wgpu adapter available on this host."
                );
                return;
            }
        };
        // Drive 20 ticks: Place at tick 0; Harvest at ticks 5 and 15.
        for _ in 0..20 {
            state.step();
        }
        let h = state.height_at(0.0, 0.0);
        assert!(
            (h - 0.0).abs() < 1e-6,
            "after PlaceTestVoxel (tick 0) + HarvestTestVoxel (ticks \
             5, 15) the placed cell should be cleared and height_at \
             back to 0.0; saw {h}. A non-zero result means harvest's \
             chronicle record didn't translate into a VoxelGrid clear.",
        );
        eprintln!("[voxel_probe] post-harvest height_at(0,0) = {h}");
    }

    // -----------------------------------------------------------------
    // Phase C semantic pin — CPU/GPU mirror divergence catcher.
    // -----------------------------------------------------------------

    /// Tiny standalone compute kernel that reads N cells from the
    /// `voxel_grid` storage buffer and writes their `u32` values into
    /// a readback buffer. Used by `cpu_gpu_voxel_state_matches` to
    /// compare GPU-side vs. CPU-side reads at the same cells.
    ///
    /// Mirrors the layout of `engine_voxel::VoxelMirror`'s staging
    /// (`index = z*H*W + y*W + x`). Width/height/depth come from a
    /// small uniform; cell indices come from a separate input buffer.
    /// Out-of-bounds returns 0u (matches the planned WGSL `voxel_at`
    /// helper's contract — Phase D will lower terrain queries to call
    /// that helper from kernel bodies).
    const READBACK_WGSL: &str = r#"
struct DimsCfg {
    width: u32,
    height: u32,
    depth: u32,
    n: u32,
};
@group(0) @binding(0) var<storage, read> voxel_grid: array<u32>;
@group(0) @binding(1) var<storage, read> probe_cells: array<u32>;
@group(0) @binding(2) var<storage, read_write> probe_results: array<u32>;
@group(0) @binding(3) var<uniform> cfg: DimsCfg;

fn voxel_at(x: i32, y: i32, z: i32) -> u32 {
    if (x < 0 || y < 0 || z < 0) {
        return 0u;
    }
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    let uz: u32 = u32(z);
    if (ux >= cfg.width || uy >= cfg.height || uz >= cfg.depth) {
        return 0u;
    }
    let idx: u32 = uz * cfg.height * cfg.width + uy * cfg.width + ux;
    return voxel_grid[idx];
}

@compute @workgroup_size(64)
fn cs_voxel_readback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) {
        return;
    }
    let x = i32(probe_cells[i * 3u + 0u]);
    let y = i32(probe_cells[i * 3u + 1u]);
    let z = i32(probe_cells[i * 3u + 2u]);
    probe_results[i] = voxel_at(x, y, z);
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct DimsCfg {
        width: u32,
        height: u32,
        depth: u32,
        n: u32,
    }

    /// Read N cells from the GPU mirror via a one-shot compute
    /// dispatch. Returns the values the GPU sees at each (x, y, z).
    /// Used by the divergence pin — caller compares element-by-element
    /// to `terrain.cell_at(x, y, z)`.
    fn gpu_read_cells(
        state: &VoxelProbeState,
        cells: &[(i32, i32, i32)],
    ) -> Vec<u32> {
        let gpu = &state.gpu;
        let mirror = state.mirror();
        let (w, h, d) = mirror.dimensions();
        let n = cells.len() as u32;

        // Pack cells as u32 triples (positive cells only — the test
        // chooses non-negative coords; negative-bound testing happens
        // in the engine_voxel unit tests). The shader receives them as
        // i32 by cast; the buffer storage is u32.
        let mut packed: Vec<u32> = Vec::with_capacity((n as usize) * 3);
        for (x, y, z) in cells {
            packed.push(*x as u32);
            packed.push(*y as u32);
            packed.push(*z as u32);
        }
        // Pad to at least 12 bytes so wgpu doesn't reject a tiny buffer.
        while packed.len() < 4 {
            packed.push(0);
        }

        let cells_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::probe_cells"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );
        let results_bytes = ((n as u64) * 4).max(16);
        let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::probe_results"),
            size: results_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cfg = DimsCfg {
            width: w,
            height: h,
            depth: d,
            n,
        };
        let cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::probe_cfg"),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::probe_results_staging"),
            size: results_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("voxel_probe::readback_shader"),
                source: wgpu::ShaderSource::Wgsl(READBACK_WGSL.into()),
            });
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_probe::readback_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel_probe::readback_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxel_probe::readback_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("cs_voxel_readback"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel_probe::readback_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mirror.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel_probe::readback_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxel_probe::readback_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Workgroup size = 64; one thread per cell.
            let groups = (n + 63) / 64;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &staging, 0, results_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        receiver
            .recv()
            .expect("readback map result")
            .expect("readback map ok");
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let out: Vec<u32> = words[..n as usize].to_vec();
        drop(mapped);
        staging.unmap();
        out
    }

    /// **Phase C semantic pin.** Catches CPU/GPU mirror divergence —
    /// the failure mode where a chronicle record mutates the host
    /// `VoxelGrid` but the corresponding GPU buffer slot stays stale
    /// (or vice versa). Drives 30 ticks (multiple place + harvest
    /// cycles), then samples a deterministic set of cells across the
    /// 16³ extent and asserts CPU == GPU at every sampled cell.
    ///
    /// Failure modes this catches:
    ///   - `apply_voxel_chronicle_record_with_mirror` writes the grid
    ///     but forgets to call `mark_dirty`.
    ///   - `flush_dirty` skips a chunk (off-by-one in the BTreeSet
    ///     drain or chunk-bounds calculation).
    ///   - The cell-to-flat-index encoding diverges between
    ///     `VoxelGrid::index` (host) and `voxel_at` (GPU).
    ///
    /// Don't substitute "voxel_count_on_gpu > 0" — that pin passes
    /// for any non-empty mirror state regardless of which cells are
    /// populated.
    #[test]
    fn cpu_gpu_voxel_state_matches() {
        let mut state = match VoxelProbeState::try_new(0) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[voxel_probe cpu_gpu_voxel_state_matches] skipping: \
                     no wgpu adapter available on this host."
                );
                return;
            }
        };
        // 30 ticks covers 3 full Place→Harvest cycles (Place at 0,
        // Harvest at 5/15/25) so the mirror sees both writes and
        // clears. Track per-tick flush cost so the eprintln below can
        // report the max-tick cost (rather than just the last) — the
        // last tick is usually zero-work and not interesting.
        let mut max_flush_ns: u128 = 0;
        let mut total_flush_ns: u128 = 0;
        for _ in 0..30 {
            state.step();
            let f = state.last_flush_ns();
            if f > max_flush_ns {
                max_flush_ns = f;
            }
            total_flush_ns += f;
        }

        // Sample 10 cells deterministically across the 16³ extent.
        // Mix in the cells the fixture actually mutates (origin,
        // and the harvest-radius neighborhood) plus a few that
        // should always be air.
        let probes: Vec<(i32, i32, i32)> = vec![
            (0, 0, 0),    // place + harvest target
            (1, 0, 0),    // adjacent — air after fixture wraps up
            (0, 1, 0),
            (0, 0, 1),
            (3, 3, 3),   // far from probe — should be air
            (5, 5, 5),
            (15, 15, 15),// corner — should be air
            (8, 4, 2),
            (2, 8, 4),
            (4, 2, 8),
        ];

        let gpu_values = gpu_read_cells(&state, &probes);
        for (i, (x, y, z)) in probes.iter().enumerate() {
            let cpu_val = state.terrain().cell_at(*x, *y, *z) as u32;
            let gpu_val = gpu_values[i];
            assert_eq!(
                cpu_val, gpu_val,
                "CPU/GPU mirror divergence at cell ({x}, {y}, {z}): \
                 CPU={cpu_val}, GPU={gpu_val}. Either the chronicle \
                 consumer wrote to the host VoxelGrid without marking \
                 the chunk dirty, or `flush_dirty` skipped the chunk, \
                 or the cell→flat-index encoding diverges between host \
                 (z*H*W + y*W + x) and the WGSL `voxel_at` helper."
            );
        }
        eprintln!(
            "[voxel_probe] cpu_gpu_voxel_state_matches: {n} cells matched. \
             Mirror buffer = {bytes} B ({kib:.1} KiB). \
             Per-tick flush_dirty: max = {max_us:.2} us, mean = {mean_us:.2} us \
             across 30 ticks (last_tick = {last_us:.2} us).",
            n = probes.len(),
            bytes = state.mirror().buffer_bytes(),
            kib = state.mirror().buffer_bytes() as f64 / 1024.0,
            max_us = max_flush_ns as f64 / 1000.0,
            mean_us = (total_flush_ns as f64 / 30.0) / 1000.0,
            last_us = state.last_flush_ns() as f64 / 1000.0,
        );
    }

    // -----------------------------------------------------------------
    // Phase D semantic pin — gpu_terrain_query_matches_cpu.
    //
    // Catches CPU/GPU semantic divergence in the terrain-query helpers:
    // a chronicle drain mutates the host VoxelGrid AND the GPU mirror,
    // but the WGSL `voxel_line_of_sight` helper computes a different
    // answer than the CPU `TerrainQuery::line_of_sight` (different DDA
    // step order, different OOB defaults, different cell-floor
    // semantics). The pin dispatches a tiny GPU compute kernel that
    // calls the SAME helper text the compiler emits into production
    // kernels, then compares its output to the CPU oracle.
    //
    // **Implementation choice — test-inline WGSL.** The plan
    // (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`,
    // Phase D §4) explicitly allows this trade-off: compiler-emit-via-
    // fixture would require a fresh `.sim` file + `.ability` corpus +
    // matching runtime crate to compile a kernel that calls
    // `terrain.line_of_sight`, all to test what is fundamentally a
    // helper-math identity. The inline WGSL below is a verbatim copy of
    // the `voxel_line_of_sight` body the compiler emits into
    // `compose_wgsl_file` (see
    // `dsl_compiler::cg::emit::program::VOXEL_GRID_WGSL_PRELUDE`); a
    // sister text-equality pin in `dsl_compiler` keeps the helper text
    // pinned so this test's local copy can't drift undetected.
    //
    // The fixture mutates the voxel grid via the existing chronicle
    // drain (place at tick 0, harvest at tick 5) and adds a few
    // hand-set cells via `set_cell` so the LOS rays have varied
    // obstructions to traverse.
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // Phase E semantic pin — same_seed_same_voxel_world.
    //
    // The load-bearing P5 determinism pin for the voxel adapter: build
    // two fixture instances with the same seed, drive each for N
    // ticks, then hash the CPU-side VoxelGrid contents (every cell at
    // every (x, y, z) in the 16³ extent — total 4096 bytes). The
    // hashes MUST match. A failure means non-deterministic ordering
    // crept in somewhere — most likely a HashMap iteration in the
    // chronicle drain, or BTreeSet → HashSet substitution in the
    // mirror's dirty tracking.
    //
    // This is the cross-cutting risk #2 from the plan
    // (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`):
    // "voxel_engine's chunks may iterate by HashMap. Verify in Phase
    // A; replace with BTreeMap or sort keys at boundary. P5 violation
    // risk if missed. The same_seed_same_voxel_world pin in Phase E
    // catches it."
    //
    // The hash is computed by walking cells in ascending (z, y, x)
    // order — deterministic regardless of any internal storage
    // ordering. Two same-seed runs with different internal iteration
    // would still produce different VoxelGrid contents at the *same*
    // (x, y, z) and the hash would diverge.
    //
    // Don't substitute "both runs return non-zero counts" — that's
    // the probe-fooling pattern; same-seed runs that diverge in cell
    // *placement* would still pass a counter-based pin.
    // -----------------------------------------------------------------

    /// Walk the voxel grid in ascending (z, y, x) order and SHA-256
    /// the byte stream. Used by `same_seed_same_voxel_world` to
    /// compare two same-seed runs at byte granularity. The walk
    /// order is independent of the underlying storage's internal
    /// iteration — a HashMap-backed grid that produces the same
    /// cells at the same coordinates would still hash identically.
    fn hash_voxel_world(state: &VoxelProbeState) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let extent = state.terrain().extent() as i32;
        let mut hasher = Sha256::new();
        for z in 0..extent {
            for y in 0..extent {
                for x in 0..extent {
                    let v = state.terrain().cell_at(x, y, z);
                    hasher.update(&[v]);
                }
            }
        }
        hasher.finalize().into()
    }

    /// **Phase E semantic pin.** Two same-seed runs of the
    /// voxel_probe fixture must produce byte-identical VoxelGrid
    /// state after `N` ticks. Catches HashMap-derived non-determinism
    /// in the chronicle drain or mirror, and any future RNG-keyed
    /// voxel mutation that fails to use `per_agent_u32(...)` keying.
    ///
    /// The fixture's chronicle drain is purely a function of the
    /// chronicle records (deterministic per the GPU dispatcher's
    /// per-tick output) and the agent SoA (initialised
    /// deterministically per seed). The voxel grid mutation order is
    /// driven by the ring-slot ordering the dispatcher writes
    /// (deterministic per GPU spec) → applied via
    /// `apply_voxel_chronicle_record_with_mirror` in the order
    /// records arrive. As long as every step of that pipeline is
    /// deterministic, the final grid state hashes identically.
    #[test]
    fn same_seed_same_voxel_world() {
        const N: u64 = 50;
        let mut state_a = match VoxelProbeState::try_new(0xC0FFEE) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[same_seed_same_voxel_world] skipping: no wgpu \
                     adapter available on this host."
                );
                return;
            }
        };
        let mut state_b = match VoxelProbeState::try_new(0xC0FFEE) {
            Some(s) => s,
            None => return,
        };
        for _ in 0..N {
            state_a.step();
        }
        for _ in 0..N {
            state_b.step();
        }
        let hash_a = hash_voxel_world(&state_a);
        let hash_b = hash_voxel_world(&state_b);
        assert_eq!(
            hash_a, hash_b,
            "same-seed runs produced different VoxelGrid hashes after \
             {N} ticks: a={:02x?} b={:02x?}. Non-determinism crept in — \
             most likely a HashMap iteration order leak in the \
             chronicle drain, or BTreeSet → HashSet substitution in \
             VoxelMirror::dirty_chunks. Re-audit voxel_engine \
             dependencies for HashMap usage at the adapter boundary.",
            hash_a, hash_b,
        );
        eprintln!(
            "[voxel_probe] same_seed_same_voxel_world: hash={:02x?} \
             (matched across two seed=0xC0FFEE runs of {N} ticks)",
            &hash_a[..8]
        );
    }

    /// Verbatim copy of the `voxel_line_of_sight` helper body the
    /// compiler injects into kernels that call `terrain.line_of_sight`
    /// (see `dsl_compiler::cg::emit::program::VOXEL_GRID_WGSL_PRELUDE`).
    /// Wrapped in a one-shot compute kernel that reads N (from, to)
    /// pairs from a buffer and writes per-pair bool results back.
    ///
    /// Drift between this string and the compiler's helper would
    /// silently invalidate the divergence pin — the WGSL-prelude pin in
    /// `dsl_compiler::cg::emit::program::voxel_grid_wgsl_prelude_emits_voxel_at_helper`
    /// keeps the load-bearing `voxel_at` index formula + helper symbols
    /// pinned so a rename surfaces upstream rather than here.
    const LOS_PROBE_WGSL: &str = r#"
struct LosCfg {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> voxel_grid: array<u32>;
@group(0) @binding(1) var<storage, read> los_inputs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> los_results: array<u32>;
@group(0) @binding(3) var<uniform> cfg: LosCfg;

// Verbatim copy of the helper bodies emitted by the compiler. See
// `dsl_compiler::cg::emit::program::VOXEL_GRID_WGSL_PRELUDE`. Drift
// between this string and the compiler's helper would silently
// invalidate the divergence pin — keep them in lock-step.
fn voxel_grid_dim() -> u32 {
    return u32(round(pow(f32(arrayLength(&voxel_grid)), 1.0 / 3.0)));
}

fn voxel_at(x: i32, y: i32, z: i32) -> u32 {
    if (x < 0 || y < 0 || z < 0) {
        return 0u;
    }
    let dim: u32 = voxel_grid_dim();
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    let uz: u32 = u32(z);
    if (ux >= dim || uy >= dim || uz >= dim) {
        return 0u;
    }
    let idx: u32 = uz * dim * dim + uy * dim + ux;
    return voxel_grid[idx];
}

fn voxel_line_of_sight(seg_from: vec3<f32>, seg_to: vec3<f32>) -> bool {
    let delta: vec3<f32> = seg_to - seg_from;
    let length_sq: f32 = dot(delta, delta);
    if (length_sq < 1.0e-12) {
        return true;
    }
    let length: f32 = sqrt(length_sq);
    let dir: vec3<f32> = delta / length;

    var cx: i32 = i32(floor(seg_from.x));
    var cy: i32 = i32(floor(seg_from.y));
    var cz: i32 = i32(floor(seg_from.z));

    let step_x: i32 = select(select(0, -1, dir.x < 0.0), 1, dir.x > 0.0);
    let step_y: i32 = select(select(0, -1, dir.y < 0.0), 1, dir.y > 0.0);
    let step_z: i32 = select(select(0, -1, dir.z < 0.0), 1, dir.z > 0.0);

    let inv_dx: f32 = select(1.0e30, 1.0 / dir.x, abs(dir.x) > 1.0e-20);
    let inv_dy: f32 = select(1.0e30, 1.0 / dir.y, abs(dir.y) > 1.0e-20);
    let inv_dz: f32 = select(1.0e30, 1.0 / dir.z, abs(dir.z) > 1.0e-20);

    var t_max_x: f32 = 1.0e30;
    if (step_x > 0) {
        t_max_x = (f32(cx + 1) - seg_from.x) * inv_dx;
    } else if (step_x < 0) {
        t_max_x = (seg_from.x - f32(cx)) * (-inv_dx);
    }
    var t_max_y: f32 = 1.0e30;
    if (step_y > 0) {
        t_max_y = (f32(cy + 1) - seg_from.y) * inv_dy;
    } else if (step_y < 0) {
        t_max_y = (seg_from.y - f32(cy)) * (-inv_dy);
    }
    var t_max_z: f32 = 1.0e30;
    if (step_z > 0) {
        t_max_z = (f32(cz + 1) - seg_from.z) * inv_dz;
    } else if (step_z < 0) {
        t_max_z = (seg_from.z - f32(cz)) * (-inv_dz);
    }

    let t_delta_x: f32 = select(1.0e30, abs(inv_dx), step_x != 0);
    let t_delta_y: f32 = select(1.0e30, abs(inv_dy), step_y != 0);
    let t_delta_z: f32 = select(1.0e30, abs(inv_dz), step_z != 0);

    if (voxel_at(cx, cy, cz) != 0u) {
        return false;
    }

    let max_steps: u32 = 3u * voxel_grid_dim();
    var i: u32 = 0u;
    loop {
        if (i >= max_steps) {
            break;
        }
        var t_next: f32;
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
            cx = cx + step_x;
            t_next = t_max_x;
            t_max_x = t_max_x + t_delta_x;
        } else if (t_max_y <= t_max_z) {
            cy = cy + step_y;
            t_next = t_max_y;
            t_max_y = t_max_y + t_delta_y;
        } else {
            cz = cz + step_z;
            t_next = t_max_z;
            t_max_z = t_max_z + t_delta_z;
        }
        if (t_next > length) {
            return true;
        }
        if (voxel_at(cx, cy, cz) != 0u) {
            return false;
        }
        i = i + 1u;
    }
    return true;
}

@compute @workgroup_size(64)
fn cs_los_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) {
        return;
    }
    // los_inputs is laid out as pairs of vec4<f32> — even index = from
    // (xyz + pad), odd index = to.
    let from_v: vec4<f32> = los_inputs[i * 2u];
    let to_v: vec4<f32>   = los_inputs[i * 2u + 1u];
    let seg_from = vec3<f32>(from_v.x, from_v.y, from_v.z);
    let seg_to   = vec3<f32>(to_v.x,   to_v.y,   to_v.z);
    let clear: bool = voxel_line_of_sight(seg_from, seg_to);
    los_results[i] = select(0u, 1u, clear);
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct LosCfg {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }

    /// Dispatch the inline LOS-probe kernel against `state.mirror()` for
    /// `pairs.len()` (from, to) segments. Returns the per-pair bool
    /// (1 = clear, 0 = blocked) the GPU computed.
    fn gpu_line_of_sight_batch(
        state: &VoxelProbeState,
        pairs: &[(Vec3, Vec3)],
    ) -> Vec<bool> {
        let gpu = &state.gpu;
        let mirror = state.mirror();
        let n = pairs.len() as u32;

        // Pack each pair as two vec4<f32> (xyz + 0.0 pad). Even slot =
        // from, odd slot = to. Pad to at least 32 bytes (= 2 vec4) so
        // wgpu doesn't reject a tiny buffer for empty input.
        let mut packed: Vec<f32> = Vec::with_capacity((pairs.len() * 8).max(8));
        for (from, to) in pairs {
            packed.extend_from_slice(&[from.x, from.y, from.z, 0.0]);
            packed.extend_from_slice(&[to.x, to.y, to.z, 0.0]);
        }
        while packed.len() < 8 {
            packed.push(0.0);
        }

        let inputs_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::los_inputs"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );
        let results_bytes = ((n as u64) * 4).max(16);
        let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::los_results"),
            size: results_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cfg = LosCfg {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("voxel_probe::los_cfg"),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_probe::los_results_staging"),
            size: results_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("voxel_probe::los_probe_shader"),
                source: wgpu::ShaderSource::Wgsl(LOS_PROBE_WGSL.into()),
            });
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_probe::los_probe_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel_probe::los_probe_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxel_probe::los_probe_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("cs_los_probe"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel_probe::los_probe_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mirror.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel_probe::los_probe_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxel_probe::los_probe_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = (n + 63) / 64;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &staging, 0, results_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        receiver
            .recv()
            .expect("los_probe map result")
            .expect("los_probe map ok");
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let out: Vec<bool> = words[..n as usize].iter().map(|&w| w != 0).collect();
        drop(mapped);
        staging.unmap();
        out
    }

    /// **Phase D semantic pin** — `gpu_terrain_query_matches_cpu`. The
    /// load-bearing CPU/GPU divergence catcher for terrain-query
    /// helpers: place a deterministic mix of voxels (~10 cells across
    /// the 16³ extent), then sample 5-10 (from, to) segments through
    /// the populated region. For each segment, compare the GPU helper's
    /// `voxel_line_of_sight(from, to)` answer to the CPU oracle's
    /// `terrain.line_of_sight(from, to)`. CPU != GPU at any segment
    /// fails the pin.
    ///
    /// Failure modes this catches:
    ///   - GPU `voxel_at` index formula drifts from
    ///     `VoxelGrid::index = z*H*W + y*W + x`.
    ///   - DDA step direction signs disagree (CPU uses
    ///     voxel_engine's raymarcher; GPU uses the inline helper).
    ///   - OOB defaults diverge (one returns false, the other true).
    ///
    /// Don't substitute "GPU returns at least one true and one false"
    /// — that pin passes for any helper that produces mixed output
    /// regardless of cell-level correctness.
    #[test]
    fn gpu_terrain_query_matches_cpu() {
        let mut state = match VoxelProbeState::try_new(0) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[voxel_probe gpu_terrain_query_matches_cpu] skipping: \
                     no wgpu adapter available on this host."
                );
                return;
            }
        };
        // Drive a few ticks so the chronicle drain populates a couple
        // of cells (place at tick 0 lifts cell (0,0,0); harvest at
        // tick 5 clears it again).
        for _ in 0..3 {
            state.step();
        }
        // Plus deterministic hand-set cells across the 16³ extent so
        // the LOS rays have varied obstructions to traverse. Spread
        // across multiple z-rows + xy quadrants to exercise all three
        // DDA step directions.
        {
            let g = &mut state.terrain;
            // Wall-like cluster around y=5 covering x ∈ [3, 7].
            g.set_cell(3, 5, 5, 1);
            g.set_cell(4, 5, 5, 1);
            g.set_cell(5, 5, 5, 1);
            g.set_cell(6, 5, 5, 1);
            g.set_cell(7, 5, 5, 1);
            // Vertical pillar at (10, 10).
            g.set_cell(10, 10, 4, 1);
            g.set_cell(10, 10, 5, 1);
            g.set_cell(10, 10, 6, 1);
            // Stray cells at corners.
            g.set_cell(2, 12, 8, 1);
            g.set_cell(14, 2, 12, 1);
        }
        // Dirty those cells in the mirror so the GPU sees them.
        for (x, y, z) in [
            (3, 5, 5),
            (4, 5, 5),
            (5, 5, 5),
            (6, 5, 5),
            (7, 5, 5),
            (10, 10, 4),
            (10, 10, 5),
            (10, 10, 6),
            (2, 12, 8),
            (14, 2, 12),
        ] {
            state.mirror.mark_dirty(glam::IVec3::new(x, y, z));
        }
        state
            .mirror
            .flush_dirty(&state.gpu, state.terrain.grid());

        // 8 deterministic (from, to) pairs spanning blocked + clear
        // cases through the populated region. Mix axis-aligned with
        // diagonals so the DDA exercises all three step directions.
        let pairs: Vec<(Vec3, Vec3)> = vec![
            // Crosses the x-row wall at y=5.5, z=5.5 → blocked.
            (Vec3::new(0.5, 5.5, 5.5), Vec3::new(15.5, 5.5, 5.5)),
            // Parallel ray at y=8.5 — no obstruction → clear.
            (Vec3::new(0.5, 8.5, 5.5), Vec3::new(15.5, 8.5, 5.5)),
            // Through the (10,10,*) pillar at z=5 → blocked.
            (Vec3::new(0.5, 10.5, 5.5), Vec3::new(15.5, 10.5, 5.5)),
            // Vertical ray clear of any cells → clear.
            (Vec3::new(1.5, 1.5, 0.5), Vec3::new(1.5, 1.5, 15.5)),
            // Diagonal from origin to corner → may or may not hit
            // (depends on placed cells); include for parity.
            (Vec3::new(0.5, 0.5, 0.5), Vec3::new(15.5, 15.5, 15.5)),
            // Short segment within an empty region → clear.
            (Vec3::new(13.5, 13.5, 1.5), Vec3::new(13.5, 13.5, 3.5)),
            // Through the (2,12,8) stray cell along z → blocked.
            (Vec3::new(2.5, 12.5, 0.5), Vec3::new(2.5, 12.5, 15.5)),
            // Through the (14,2,12) stray cell along x → blocked.
            (Vec3::new(0.5, 2.5, 12.5), Vec3::new(15.5, 2.5, 12.5)),
        ];

        let cpu_results: Vec<bool> = pairs
            .iter()
            .map(|(from, to)| state.terrain().line_of_sight(*from, *to))
            .collect();
        let gpu_results = gpu_line_of_sight_batch(&state, &pairs);

        let mut diffs = Vec::new();
        for (i, ((from, to), (cpu, gpu))) in pairs
            .iter()
            .zip(cpu_results.iter().zip(gpu_results.iter()))
            .enumerate()
        {
            if cpu != gpu {
                diffs.push(format!(
                    "  pair[{i}]: from=({:.1},{:.1},{:.1}) to=({:.1},{:.1},{:.1}) \
                     CPU={cpu} GPU={gpu}",
                    from.x, from.y, from.z, to.x, to.y, to.z
                ));
            }
        }
        assert!(
            diffs.is_empty(),
            "CPU/GPU terrain-query divergence on {n}/{tot} segments:\n{lines}\n\
             Either the GPU `voxel_line_of_sight` helper diverges from \
             voxel_engine's `ray_cast_grid` (DDA step direction, OOB \
             defaults, cell-floor semantics) OR the GPU mirror's cell \
             contents differ from the host VoxelGrid. The \
             cpu_gpu_voxel_state_matches pin catches the latter; this \
             pin catches the former.",
            n = diffs.len(),
            tot = pairs.len(),
            lines = diffs.join("\n"),
        );
        eprintln!(
            "[voxel_probe] gpu_terrain_query_matches_cpu: {n} segments \
             agreed (CPU == GPU). cpu={cpu_results:?}",
            n = pairs.len()
        );
    }
}
