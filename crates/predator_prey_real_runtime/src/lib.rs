//! Per-fixture runtime for `assets/sim/predator_prey_real.sim` —
//! fourth REAL fixture: COMBAT + LIFECYCLE composed on TWO live
//! creature types simultaneously.
//!
//! Mirrors `foraging_real_runtime` for the spatial body-form +
//! chronicle-physics + CPU-side birth-slot scaffolding. Adds:
//!
//!   1. **Two LIVE creature types.** Wolves (creature_type=0) and
//!      Sheep (creature_type=1) both act per-tick. Wolves hunt sheep
//!      via the WolfHunt body-form spatial walk; sheep graze
//!      (uniform regrowth proxy via SheepGraze) and try to evade
//!      indirectly through population dynamics.
//!
//!   2. **Combat layered on lifecycle.** Sheep die TWO ways:
//!      ApplyKill's `set_alive(sheep, false)` on hp<=0 (combat
//!      death) AND EnergyDecay's `set_alive(self, false)` on
//!      hunger<=0 (starvation). Wolves die ONE way:
//!      EnergyDecay starvation (no predator preys on wolves).
//!
//!   3. **Two-species birth machinery (CPU-side).**
//!      - Wolves: when summed `kills_total` view crosses each
//!        integer multiple of `KILLS_PER_WOLF_BIRTH`, runtime flips
//!        next dead Wolf slot alive=1 with hunger=INITIAL_WOLF_ENERGY.
//!      - Sheep: every `SHEEP_BREED_INTERVAL` ticks, runtime counts
//!        well-fed sheep (hunger > SHEEP_BREED_THRESHOLD); flips
//!        floor(well_fed_count / 2) dead Sheep slots alive=1.
//!
//! Per-tick chain:
//!
//!   1. clear_tail + clear ring headers + clear spatial offsets
//!   2. spatial_build_hash (5 phases): count → scan_local →
//!      scan_carry → scan_add → scatter
//!   3. WolfHunt @phase(per_agent) — body-form spatial walk; emits
//!      Damaged per (wolf, sheep) neighbour pair (gated every 2 ticks)
//!   4. ApplyKill @phase(post) — chronicle: decrement sheep.hp,
//!      increment wolf.hunger + wolf.hp; emit Killed on hp<=0
//!   5. SheepGraze @phase(per_agent) — per-sheep hunger += graze_rate
//!   6. EnergyDecay @phase(per_agent) — universal hunger decrement,
//!      set_alive(self, false) on hunger<=0; emit Starved
//!   7. seed_indirect_0
//!   8. fold_kills_total + fold_sheep_killed_total + fold_starved_total
//!
//! After step(), CPU-side `process_lifecycle()` reads kills_total +
//! starved_total + alive + creature_type buffers and:
//!   - Flips dead Wolf slots back to alive=1 every KILLS_PER_WOLF_BIRTH
//!     accumulated kills.
//!   - Flips dead Sheep slots back to alive=1 every
//!     SHEEP_BREED_INTERVAL ticks for each pair of well-fed sheep.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Discriminant value the WGSL emits for the first-declared entity
/// (`Wolf`). Matches the `creature_type == Wolf` lowering — entity
/// declaration order assigns 0 to Wolf, 1 to Sheep.
pub const CT_WOLF: u32 = 0;
/// Discriminant for the second-declared entity (`Sheep`).
pub const CT_SHEEP: u32 = 1;

/// Initial wolf population. Wolves take slots [0..INITIAL_WOLVES).
pub const INITIAL_WOLVES: u32 = 30;
/// Initial sheep population. Sheep take slots
/// [WOLF_CAP..WOLF_CAP + INITIAL_SHEEP).
pub const INITIAL_SHEEP: u32 = 80;
/// Wolf slot reservation. Slots [0..WOLF_CAP) are ALWAYS Wolf
/// creature_type — even dead wolf placeholders. This keeps wolf
/// births from accidentally landing on sheep slots.
pub const WOLF_CAP: u32 = 100;
/// Sheep slot reservation. Slots [WOLF_CAP..WOLF_CAP + SHEEP_CAP)
/// are ALWAYS Sheep creature_type. Total slot capacity =
/// WOLF_CAP + SHEEP_CAP = 300.
pub const SHEEP_CAP: u32 = 200;
/// Total slot capacity. Wolves [0..WOLF_CAP) + Sheep
/// [WOLF_CAP..SLOT_CAP).
pub const SLOT_CAP: u32 = WOLF_CAP + SHEEP_CAP;

/// Initial Wolf hunger (energy). With decay_rate=1.0/tick and
/// meat_gain=4.0/Damaged, a hunting wolf comfortably tops up over
/// each successful raid. A non-hunting wolf starves in 50 ticks.
pub const INITIAL_WOLF_ENERGY: f32 = 50.0;
/// Initial Sheep hunger. With graze_rate=1.0/tick and decay_rate=
/// 1.0/tick, a non-eaten sheep stays at saturation indefinitely
/// (graze and decay net to 0 — sheep don't starve on their own).
/// Combat death is the dominant mortality path.
pub const INITIAL_SHEEP_ENERGY: f32 = 60.0;
/// Initial Sheep hp (combat health). With strike_damage=4.0, ~5
/// wolf hits drop a sheep — slow enough to spread combat across
/// multiple ticks (so the within-tick race-window doesn't pile up
/// 30 wolves all firing Killed at the same sheep on tick 1).
pub const INITIAL_SHEEP_HP: f32 = 20.0;
/// Wolf kills required to trigger one wolf birth. Tuned high enough
/// that wolves can't outbreed their food supply within 50 ticks but
/// low enough to be observable across 200+ ticks.
pub const KILLS_PER_WOLF_BIRTH: f32 = 8.0;
/// Tick cadence for the sheep-breed CPU-side check. Lower than the
/// wolf-hunt cadence (every 6 ticks) so well-fed sheep get a chance
/// to reproduce before being eaten.
pub const SHEEP_BREED_INTERVAL: u64 = 5;
/// Hunger threshold for a sheep to be considered "well-fed" and
/// eligible to contribute to breeding. With INITIAL_SHEEP_ENERGY=60,
/// well-fed sheep sit near 60-70; threshold at 40 lets most alive
/// sheep contribute most cycles, which is what we want for emergent
/// dynamics (sheep boom when wolves are scarce).
pub const SHEEP_BREED_THRESHOLD: f32 = 40.0;

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

/// Per-tick lifecycle counters. Reset between ticks; consumed by
/// the app-side trace.
#[derive(Default, Debug, Clone, Copy)]
pub struct TickLifecycle {
    pub wolf_births: u32,
    pub sheep_births: u32,
    pub wolf_starvations: u32,
    pub sheep_starvations: u32,
    pub sheep_kills: u32,
}

/// Per-fixture state for the predator/prey ecosystem.
pub struct PredatorPreyRealState {
    gpu: GpuContext,

    // -- Agent SoA (shared between Wolf + Sheep, discriminated by
    //    creature_type). All sized to SLOT_CAP. --
    agent_pos_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_hunger_buf: wgpu::Buffer,
    agent_creature_type_buf: wgpu::Buffer,

    // -- Spatial grid --
    spatial_grid_cells: wgpu::Buffer,
    spatial_grid_offsets: wgpu::Buffer,
    spatial_grid_starts: wgpu::Buffer,
    spatial_chunk_sums: wgpu::Buffer,
    spatial_offsets_zero: wgpu::Buffer,

    // -- Event ring + view storage --
    event_ring: EventRing,
    kills_total_view: ViewStorage,
    kills_total_cfg_buf: wgpu::Buffer,
    sheep_killed_total_view: ViewStorage,
    sheep_killed_total_cfg_buf: wgpu::Buffer,
    starved_total_view: ViewStorage,
    starved_total_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    wolfhunt_cfg_buf: wgpu::Buffer,
    applykill_cfg_buf: wgpu::Buffer,
    sheepgraze_cfg_buf: wgpu::Buffer,
    energydecay_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    /// Reserved for future P5 RNG use; today the layout + decisions
    /// are fully deterministic. Stored to keep the constructor
    /// signature parity with the prior fixture pattern.
    #[allow(dead_code)]
    seed: u64,

    // -- CPU-side lifecycle bookkeeping --
    /// Cumulative wolf births triggered so far. Each tick:
    /// `intended = floor(total_kills / KILLS_PER_WOLF_BIRTH)`;
    /// wolf-births-to-fire = max(0, intended - wolf_births_so_far).
    wolf_births_so_far: u32,
    /// Cumulative sheep births triggered so far.
    sheep_births_so_far: u32,
    /// Cumulative wolf starvations observed (from starved_total view
    /// partitioned by creature_type at readback).
    wolf_starvations_so_far: u32,
    /// Cumulative sheep starvations observed.
    sheep_starvations_so_far: u32,
    /// Cumulative sheep combat-kills (alive 1→0 transitions on Sheep
    /// slots that were NOT just starvations). Detected by snapshotting
    /// the alive bitmap each tick.
    sheep_kills_so_far: u32,
    /// Last tick's lifecycle delta (for the per-tick trace).
    last_lifecycle: TickLifecycle,
    /// Snapshot of alive bitmap from the END of the previous tick
    /// (after process_lifecycle's birth flips). Used to detect
    /// alive 1→0 transitions per slot per tick (combat + starvation
    /// deaths) and 0→1 transitions (births). Length = SLOT_CAP.
    prev_alive: Vec<u32>,
}

impl PredatorPreyRealState {
    /// Construct an ecosystem with INITIAL_WOLVES live wolves in
    /// slots [0..INITIAL_WOLVES), INITIAL_SHEEP live sheep in slots
    /// [WOLF_CAP..WOLF_CAP+INITIAL_SHEEP), and the rest as dead
    /// placeholder slots that births can later recycle.
    ///
    /// Slot reservation by creature_type ensures wolf births never
    /// land on sheep slots (and vice versa) — crucial because the
    /// CPU-side birth code can't change creature_type after init.
    pub fn new(seed: u64, _agent_count_hint: u32) -> Self {
        let n = SLOT_CAP as usize;

        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut hp_init: Vec<f32> = Vec::with_capacity(n);
        let mut alive_init: Vec<u32> = Vec::with_capacity(n);
        let mut hunger_init: Vec<f32> = Vec::with_capacity(n);
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);

        // Layout (deterministic positions for replay parity):
        //   - Wolves: 30 wolves spread across a SPARSE 12×8 grid
        //     step 1.6 (so each wolf-cell only contains ONE wolf and
        //     wolves hunt-range overlaps a small contiguous patch).
        //     This kills the within-tick race inflation that happens
        //     when many wolves are all in one sheep's neighbourhood:
        //     fewer overlapping wolves → fewer simultaneous Damaged
        //     events on the same sheep per tick → more progressive
        //     combat.
        //   - Sheep: 80 sheep on an 8×10 grid step 1.0, INTERLEAVED
        //     with the wolf field so each wolf has a few sheep within
        //     its 1.5 radius (not 80 sheep all-in-range like dense
        //     packing).
        //   - Z-plane stays 0 so the spatial-grid 27-cell walk
        //     surfaces neighbours.
        //
        // Total field: ~12-19 unit square. With hunt_radius=1.5 and
        // step ~1, each wolf's 27-cell window catches 1-3 sheep
        // (rather than 30+ in the prior dense layout). Combat is
        // therefore progressive: most sheep take 2-3 ticks to die,
        // and the population dynamics actually emerge.
        for slot in 0..SLOT_CAP {
            if slot < INITIAL_WOLVES {
                // Live Wolf. Sparse 6×5 grid step 2.0, scattered along
                // the x>0 half of the field at x ∈ [4.0, 14.0]. Wolves
                // start AWAY from sheep so combat is gradual: only the
                // few wolves nearest the seam can reach sheep on the
                // first hunt cycle, the rest have to wait for population
                // pressure to spread the herd into wolf territory.
                let row = (slot / 5) as f32;
                let col = (slot % 5) as f32;
                let x = 4.0 + col * 2.0;
                let y = (row - 2.5) * 2.0;
                pos_padded.push(Vec3::new(x, y, 0.0).into());
                hp_init.push(0.0); // Wolf.hp = kill credit, starts 0
                alive_init.push(1);
                hunger_init.push(INITIAL_WOLF_ENERGY);
                creature_init.push(CT_WOLF);
            } else if slot < WOLF_CAP {
                // Dead Wolf placeholder. Birth recycles these.
                pos_padded.push(Vec3::new(0.0, 0.0, 0.0).into());
                hp_init.push(0.0);
                alive_init.push(0);
                hunger_init.push(0.0);
                creature_init.push(CT_WOLF);
            } else if slot < WOLF_CAP + INITIAL_SHEEP {
                // Live Sheep. 10×8 grid step 1.0, x ∈ [-9.5, -0.5],
                // y ∈ [-3.5, 3.5] — anchored on the LEFT side of the
                // field, away from the wolf cluster on the right. The
                // ~4-unit gap means initial-tick wolves at x=4 only
                // reach sheep at x ≥ 2.5; sheep births spread the
                // population gradually and the boundary moves over time.
                let s_idx = slot - WOLF_CAP;
                let row = (s_idx / 10) as f32;
                let col = (s_idx % 10) as f32;
                let x = -9.5 + col * 1.0;
                let y = (row - 3.5) * 1.0;
                pos_padded.push(Vec3::new(x, y, 0.0).into());
                hp_init.push(INITIAL_SHEEP_HP);
                alive_init.push(1);
                hunger_init.push(INITIAL_SHEEP_ENERGY);
                creature_init.push(CT_SHEEP);
            } else {
                // Dead Sheep placeholder. Spawn births land slightly
                // scattered across the field interior.
                pos_padded.push(Vec3::new(0.0, 0.0, 0.0).into());
                hp_init.push(0.0);
                alive_init.push(0);
                hunger_init.push(0.0);
                creature_init.push(CT_SHEEP);
            }
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let agent_pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::agent_pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_hunger_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::agent_hunger"),
            contents: bytemuck::cast_slice(&hunger_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_real_runtime::agent_creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Spatial-grid buffers (mirror foraging_real_runtime) ----
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (SLOT_CAP as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_real_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_real_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_real_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = sp::num_cells().div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_real_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_real_runtime::spatial_offsets_zero"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Event ring + view storage ----
        let event_ring = EventRing::new(&gpu, "predator_prey_real_runtime");
        let kills_total_view = ViewStorage::new(
            &gpu,
            "predator_prey_real_runtime::kills_total",
            SLOT_CAP,
            false,
            false,
        );
        let sheep_killed_total_view = ViewStorage::new(
            &gpu,
            "predator_prey_real_runtime::sheep_killed_total",
            SLOT_CAP,
            false,
            false,
        );
        let starved_total_view = ViewStorage::new(
            &gpu,
            "predator_prey_real_runtime::starved_total",
            SLOT_CAP,
            false,
            false,
        );

        // ---- Per-kernel cfg uniforms ----
        let wolfhunt_cfg = physics_WolfHunt::PhysicsWolfHuntCfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let wolfhunt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::wolfhunt_cfg"),
            contents: bytemuck::bytes_of(&wolfhunt_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let applykill_cfg = physics_ApplyKill::PhysicsApplyKillCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let applykill_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::applykill_cfg"),
            contents: bytemuck::bytes_of(&applykill_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sheepgraze_cfg = physics_SheepGraze::PhysicsSheepGrazeCfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let sheepgraze_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::sheepgraze_cfg"),
            contents: bytemuck::bytes_of(&sheepgraze_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let energydecay_cfg = physics_EnergyDecay::PhysicsEnergyDecayCfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let energydecay_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("predator_prey_real_runtime::energydecay_cfg"),
                contents: bytemuck::bytes_of(&energydecay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predator_prey_real_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let make_view_cfg = |label: &str| -> wgpu::Buffer {
            let cfg = fold_kills_total::FoldKillsTotalCfg {
                event_count: 0,
                tick: 0,
                second_key_pop: 1,
                _pad: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let kills_total_cfg_buf = make_view_cfg("predator_prey_real_runtime::kills_total_cfg");
        let sheep_killed_total_cfg_buf =
            make_view_cfg("predator_prey_real_runtime::sheep_killed_total_cfg");
        let starved_total_cfg_buf =
            make_view_cfg("predator_prey_real_runtime::starved_total_cfg");

        Self {
            gpu,
            agent_pos_buf,
            agent_hp_buf,
            agent_alive_buf,
            agent_hunger_buf,
            agent_creature_type_buf,
            spatial_grid_cells,
            spatial_grid_offsets,
            spatial_grid_starts,
            spatial_chunk_sums,
            spatial_offsets_zero,
            event_ring,
            kills_total_view,
            kills_total_cfg_buf,
            sheep_killed_total_view,
            sheep_killed_total_cfg_buf,
            starved_total_view,
            starved_total_cfg_buf,
            wolfhunt_cfg_buf,
            applykill_cfg_buf,
            sheepgraze_cfg_buf,
            energydecay_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count: SLOT_CAP,
            seed,
            wolf_births_so_far: 0,
            sheep_births_so_far: 0,
            wolf_starvations_so_far: 0,
            sheep_starvations_so_far: 0,
            sheep_kills_so_far: 0,
            last_lifecycle: TickLifecycle::default(),
            prev_alive: alive_init,
        }
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }
    pub fn read_hunger(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hunger_buf, "hunger")
    }
    /// Snapshot agent world-space positions (XYZ). Used by the
    /// ASCII renderer to project world coords onto the terminal grid.
    pub fn read_positions(&self) -> Vec<Vec3> {
        self.read_pos_padded()
            .into_iter()
            .map(|p| Vec3::new(p.x, p.y, p.z))
            .collect()
    }
    pub fn read_creature_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_creature_type_buf, "creature_type")
    }

    pub fn kills_total(&mut self) -> &[f32] {
        self.kills_total_view.readback(&self.gpu)
    }
    pub fn sheep_killed_total(&mut self) -> &[f32] {
        self.sheep_killed_total_view.readback(&self.gpu)
    }
    pub fn starved_total(&mut self) -> &[f32] {
        self.starved_total_view.readback(&self.gpu)
    }

    pub fn last_lifecycle(&self) -> TickLifecycle { self.last_lifecycle }
    pub fn wolf_births_so_far(&self) -> u32 { self.wolf_births_so_far }
    pub fn sheep_births_so_far(&self) -> u32 { self.sheep_births_so_far }
    pub fn wolf_starvations_so_far(&self) -> u32 { self.wolf_starvations_so_far }
    pub fn sheep_starvations_so_far(&self) -> u32 { self.sheep_starvations_so_far }
    pub fn sheep_kills_so_far(&self) -> u32 { self.sheep_kills_so_far }
    pub fn slot_cap(&self) -> u32 { SLOT_CAP }
    pub fn wolf_cap(&self) -> u32 { WOLF_CAP }
    pub fn sheep_cap(&self) -> u32 { SHEEP_CAP }

    /// Count alive Wolf slots.
    pub fn count_alive_wolves(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_WOLF)
            .count() as u32
    }

    /// Count alive Sheep slots.
    pub fn count_alive_sheep(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_SHEEP)
            .count() as u32
    }

    /// CPU-side lifecycle processing. Called after each step(). Drives:
    ///   1. Wolf births: when summed `kills_total` view crosses each
    ///      integer multiple of `KILLS_PER_WOLF_BIRTH`, flips next dead
    ///      Wolf slot alive=1 with reset state. (Wolves reproduce by
    ///      successful hunting.)
    ///   2. Sheep births: every `SHEEP_BREED_INTERVAL` ticks, count
    ///      well-fed sheep (hunger > SHEEP_BREED_THRESHOLD); flip
    ///      floor(well_fed_count / 2) dead Sheep slots alive=1.
    ///      (Sheep reproduce by being well-fed.)
    ///   3. Starvation accounting: read `starved_total` view +
    ///      partition by creature_type to update wolf vs sheep
    ///      starvation counters.
    ///
    /// Returns the per-tick lifecycle delta.
    fn process_lifecycle(&mut self) -> TickLifecycle {
        // --- Read all GPU buffers we need once, then do all the
        //     bookkeeping CPU-side. ---
        // Death detection: take alive-bitmap delta vs prev_alive.
        // alive 1→0 = death this tick (combat OR starvation, all
        // distinguished by combining with starved_total readback per
        // slot). alive-bitmap delta avoids the within-tick race
        // inflation that affects raw `Killed` event counts (when many
        // wolves all see old_hp>0 and each emit Killed against the
        // same target — the inflation is cosmetic for the bitmap but
        // wrecks naive event-count metrics).
        //
        // The Starved event view IS race-tolerant per-slot (each slot
        // either starved this tick or didn't, no within-tick race
        // because EnergyDecay runs once per slot per tick). We use it
        // to PARTITION the death count: deaths = alive 1→0 transitions;
        // starvations = slots whose starved_total[slot] increased this
        // tick; combat-kills = deaths - starvations.
        let starved_per_slot = self.starved_total_view.readback(&self.gpu).to_vec();
        let mut alive = self.read_alive();
        let ct = self.read_creature_type();
        let mut hunger = self.read_hunger();
        let mut hp = self.read_hp();
        let mut pos = self.read_pos_padded();

        // Walk slots once to compute all per-tick deltas + cumulative
        // counters in one pass.
        let mut wolf_starved_total = 0u32;
        let mut sheep_starved_total = 0u32;
        let mut wolf_deaths_this_tick = 0u32;
        let mut sheep_deaths_this_tick = 0u32;
        for (slot, &cur) in alive.iter().enumerate() {
            let prev = self.prev_alive[slot];
            let c = ct[slot];
            // Cumulative starvation per creature type.
            let s = starved_per_slot[slot].round() as u32;
            if c == CT_WOLF {
                wolf_starved_total += s;
            } else if c == CT_SHEEP {
                sheep_starved_total += s;
            }
            // Death detection: alive 1 → 0 transition.
            if prev == 1 && cur == 0 {
                if c == CT_WOLF {
                    wolf_deaths_this_tick += 1;
                } else if c == CT_SHEEP {
                    sheep_deaths_this_tick += 1;
                }
            }
        }
        let new_wolf_starv = wolf_starved_total.saturating_sub(self.wolf_starvations_so_far);
        let new_sheep_starv = sheep_starved_total.saturating_sub(self.sheep_starvations_so_far);
        self.wolf_starvations_so_far = wolf_starved_total;
        self.sheep_starvations_so_far = sheep_starved_total;

        // Combat-only sheep deaths = total sheep deaths this tick MINUS
        // sheep starvations this tick. Cumulative combat-kill count
        // drives wolf births.
        let combat_sheep_deaths_this_tick =
            sheep_deaths_this_tick.saturating_sub(new_sheep_starv);
        let new_sheep_kills = combat_sheep_deaths_this_tick;
        self.sheep_kills_so_far = self.sheep_kills_so_far.saturating_add(new_sheep_kills);
        let total_combat_kills = self.sheep_kills_so_far;

        // wolf_deaths_this_tick currently includes both combat (none —
        // wolves aren't preyed on in this fixture) and starvation.
        // For the metric trace we just report wolf starvation; the
        // total wolf-deaths shadow stays as a sanity check.
        let _ = wolf_deaths_this_tick;

        // --- Wolf births: driven by cumulative combat sheep-kills.
        //     Each KILLS_PER_WOLF_BIRTH dead sheep spawns one new
        //     wolf in a free Wolf slot. Now race-immune because
        //     `total_combat_kills` is computed from alive-bitmap
        //     deltas, not raw Killed-event counts. ---
        let intended_wolf_births =
            ((total_combat_kills as f32) / KILLS_PER_WOLF_BIRTH).floor() as u32;
        let wolf_births_to_fire =
            intended_wolf_births.saturating_sub(self.wolf_births_so_far);
        let mut wolf_births_fired = 0u32;
        if wolf_births_to_fire > 0 {
            for slot in 0..(WOLF_CAP as usize) {
                if wolf_births_fired >= wolf_births_to_fire {
                    break;
                }
                if ct[slot] == CT_WOLF && alive[slot] == 0 {
                    alive[slot] = 1;
                    hunger[slot] = INITIAL_WOLF_ENERGY;
                    hp[slot] = 0.0;
                    // Re-place new wolf scattered across the SAME 6×5
                    // grid step 2.0 layout as the initial wolves (right
                    // half of field, x ∈ [4.0, 14.0]). New wolves spawn
                    // at the wolf cluster — they then "hunt" wherever
                    // sheep happen to be.
                    let ring_slot = (self.wolf_births_so_far + wolf_births_fired) as usize
                        % (INITIAL_WOLVES as usize);
                    let row = (ring_slot / 5) as f32;
                    let col = (ring_slot % 5) as f32;
                    let x = 4.0 + col * 2.0;
                    let y = (row - 2.5) * 2.0;
                    pos[slot] = Vec3::new(x, y, 0.0).into();
                    wolf_births_fired += 1;
                }
            }
            self.wolf_births_so_far += wolf_births_fired;
        }

        // --- Sheep births: every SHEEP_BREED_INTERVAL ticks, count
        //     well-fed sheep, birth floor(well_fed/2) new sheep. ---
        let mut sheep_births_fired = 0u32;
        if self.tick > 0 && self.tick % SHEEP_BREED_INTERVAL == 0 {
            let mut well_fed = 0u32;
            for slot in (WOLF_CAP as usize)..(SLOT_CAP as usize) {
                if alive[slot] == 1
                    && ct[slot] == CT_SHEEP
                    && hunger[slot] > SHEEP_BREED_THRESHOLD
                {
                    well_fed += 1;
                }
            }
            let intended = well_fed / 2;
            // Cap births per tick at the available dead-sheep slot
            // count to avoid spinning the slot scanner unnecessarily.
            let mut to_fire = intended;
            for slot in (WOLF_CAP as usize)..(SLOT_CAP as usize) {
                if to_fire == 0 {
                    break;
                }
                if ct[slot] == CT_SHEEP && alive[slot] == 0 {
                    alive[slot] = 1;
                    hunger[slot] = INITIAL_SHEEP_ENERGY;
                    hp[slot] = INITIAL_SHEEP_HP;
                    // Place new sheep across a wider scatter than the
                    // initial 10×8 grid — sheep births fan out across
                    // the full sheep-territory width to keep population
                    // density bounded (and let the herd push EAST into
                    // wolf territory as it grows).
                    let ring_slot = (self.sheep_births_so_far + sheep_births_fired) as usize
                        % 200; // larger than INITIAL_SHEEP so births
                               // distribute over a 20-position ring
                    let row = (ring_slot / 20) as f32;
                    let col = (ring_slot % 20) as f32;
                    // 20×10 spawn ring covering x ∈ [-9.5, 9.5],
                    // y ∈ [-3.5, 3.5]. Sheep that spawn east naturally
                    // intersect wolf territory, driving combat over
                    // time even after the initial west-side sheep
                    // get eaten.
                    let x = -9.5 + col * 1.0;
                    let y = (row - 3.5) * 1.0;
                    pos[slot] = Vec3::new(x, y, 0.0).into();
                    sheep_births_fired += 1;
                    to_fire -= 1;
                }
            }
            self.sheep_births_so_far += sheep_births_fired;
        }

        if wolf_births_fired > 0 || sheep_births_fired > 0 {
            // Push the updated buffers back to GPU.
            self.gpu.queue.write_buffer(
                &self.agent_alive_buf,
                0,
                bytemuck::cast_slice(&alive),
            );
            self.gpu.queue.write_buffer(
                &self.agent_hunger_buf,
                0,
                bytemuck::cast_slice(&hunger),
            );
            self.gpu.queue.write_buffer(
                &self.agent_hp_buf,
                0,
                bytemuck::cast_slice(&hp),
            );
            self.gpu.queue.write_buffer(
                &self.agent_pos_buf,
                0,
                bytemuck::cast_slice(&pos),
            );
        }

        // Snapshot alive bitmap AFTER births land — next tick's
        // 1→0 detection runs against this post-birth baseline. This
        // means a slot that's born this tick AND dies next tick gets
        // counted as a death next tick (correct).
        self.prev_alive = alive;

        TickLifecycle {
            wolf_births: wolf_births_fired,
            sheep_births: sheep_births_fired,
            wolf_starvations: new_wolf_starv,
            sheep_starvations: new_sheep_starv,
            sheep_kills: new_sheep_kills,
        }
    }

    fn read_pos_padded(&self) -> Vec<Vec3Padded> {
        let bytes = (SLOT_CAP as u64) * 16;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("predator_prey_real_runtime::pos_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_real_runtime::read_pos"),
            });
        encoder.copy_buffer_to_buffer(&self.agent_pos_buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<Vec3Padded> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (SLOT_CAP as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("predator_prey_real_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_real_runtime::read_f32"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_u32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<u32> {
        let bytes = (SLOT_CAP as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("predator_prey_real_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_real_runtime::read_u32"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
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

impl CompiledSim for PredatorPreyRealState {
    fn step(&mut self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("predator_prey_real_runtime::step"),
            });

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        use dsl_compiler::cg::emit::spatial as sp;
        let max_neighbour_emits = SLOT_CAP
            .saturating_mul(sp::MAX_PER_CELL)
            .saturating_mul(27);
        // Slots produced per tick: WolfHunt emits Damaged per neighbour
        // sheep (up to ~27*MAX_PER_CELL per wolf, gated by /2);
        // ApplyKill may fan-out Killed; EnergyDecay emits Starved per
        // dying agent. Bound headers above the worst case.
        let max_slots_per_tick = max_neighbour_emits.saturating_mul(2).min(65536);
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, max_slots_per_tick);
        let offsets_size = sp::offsets_bytes();
        encoder.copy_buffer_to_buffer(
            &self.spatial_offsets_zero,
            0,
            &self.spatial_grid_offsets,
            0,
            offsets_size,
        );

        // (2) Spatial-hash counting sort. Required input for WolfHunt.
        let wolfhunt_cfg = physics_WolfHunt::PhysicsWolfHuntCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.wolfhunt_cfg_buf,
            0,
            bytemuck::bytes_of(&wolfhunt_cfg),
        );

        let count_b = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.wolfhunt_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_count(
            &mut self.cache,
            &count_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );
        let scan_local_b = spatial_build_hash_scan_local::SpatialBuildHashScanLocalBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.wolfhunt_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_local(
            &mut self.cache,
            &scan_local_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );
        let scan_carry_b = spatial_build_hash_scan_carry::SpatialBuildHashScanCarryBindings {
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.wolfhunt_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_carry(
            &mut self.cache,
            &scan_carry_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );
        let scan_add_b = spatial_build_hash_scan_add::SpatialBuildHashScanAddBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.wolfhunt_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_add(
            &mut self.cache,
            &scan_add_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );
        let scatter_b = spatial_build_hash_scatter::SpatialBuildHashScatterBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.wolfhunt_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache,
            &scatter_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (3) WolfHunt — body-form spatial walk. Emits Damaged events.
        let wolfhunt_b = physics_WolfHunt::PhysicsWolfHuntBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.wolfhunt_cfg_buf,
        };
        dispatch::dispatch_physics_wolfhunt(
            &mut self.cache,
            &wolfhunt_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (4) ApplyKill — chronicle. Reads Damaged events, writes
        // sheep.hp + sheep.alive + wolf.hunger + wolf.hp.
        let event_count_estimate = max_neighbour_emits.min(65536);
        let applykill_cfg = physics_ApplyKill::PhysicsApplyKillCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.applykill_cfg_buf,
            0,
            bytemuck::bytes_of(&applykill_cfg),
        );
        let applykill_b = physics_ApplyKill::PhysicsApplyKillBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            cfg: &self.applykill_cfg_buf,
        };
        dispatch::dispatch_physics_applykill(
            &mut self.cache,
            &applykill_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (5) SheepGraze — per-sheep hunger += graze_rate.
        let sheepgraze_cfg = physics_SheepGraze::PhysicsSheepGrazeCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.sheepgraze_cfg_buf,
            0,
            bytemuck::bytes_of(&sheepgraze_cfg),
        );
        let sheepgraze_b = physics_SheepGraze::PhysicsSheepGrazeBindings {
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            cfg: &self.sheepgraze_cfg_buf,
        };
        dispatch::dispatch_physics_sheepgraze(
            &mut self.cache,
            &sheepgraze_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (6) EnergyDecay — universal per-agent hunger drain + death gate.
        let energydecay_cfg = physics_EnergyDecay::PhysicsEnergyDecayCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.energydecay_cfg_buf,
            0,
            bytemuck::bytes_of(&energydecay_cfg),
        );
        let energydecay_b = physics_EnergyDecay::PhysicsEnergyDecayBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            cfg: &self.energydecay_cfg_buf,
        };
        dispatch::dispatch_physics_energydecay(
            &mut self.cache,
            &energydecay_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (7) seed_indirect_0 — keeps args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let seed_b = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (8) Folds — same cfg shape, different storage targets.
        let mk_fold_cfg = |ec: u32, tick: u32| fold_kills_total::FoldKillsTotalCfg {
            event_count: ec,
            tick,
            second_key_pop: 1,
            _pad: 0,
        };
        let fold_cfg = mk_fold_cfg(event_count_estimate, self.tick as u32);
        for buf in [
            &self.kills_total_cfg_buf,
            &self.sheep_killed_total_cfg_buf,
            &self.starved_total_cfg_buf,
        ] {
            self.gpu
                .queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&fold_cfg));
        }

        let kills_b = fold_kills_total::FoldKillsTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.kills_total_view.primary(),
            view_storage_anchor: self.kills_total_view.anchor(),
            view_storage_ids: self.kills_total_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.kills_total_cfg_buf,
        };
        dispatch::dispatch_fold_kills_total(
            &mut self.cache,
            &kills_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );
        let sk_b = fold_sheep_killed_total::FoldSheepKilledTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.sheep_killed_total_view.primary(),
            view_storage_anchor: self.sheep_killed_total_view.anchor(),
            view_storage_ids: self.sheep_killed_total_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.sheep_killed_total_cfg_buf,
        };
        dispatch::dispatch_fold_sheep_killed_total(
            &mut self.cache,
            &sk_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );
        let starved_b = fold_starved_total::FoldStarvedTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.starved_total_view.primary(),
            view_storage_anchor: self.starved_total_view.anchor(),
            view_storage_ids: self.starved_total_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.starved_total_cfg_buf,
        };
        dispatch::dispatch_fold_starved_total(
            &mut self.cache,
            &starved_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.kills_total_view.mark_dirty();
        self.sheep_killed_total_view.mark_dirty();
        self.starved_total_view.mark_dirty();
        self.tick += 1;

        // CPU-side lifecycle post-processing: births, deaths
        // attribution, per-tick deltas.
        let lc = self.process_lifecycle();
        self.last_lifecycle = lc;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }

    fn snapshot(&mut self) -> engine::AgentSnapshot {
        engine::AgentSnapshot {
            positions: self.read_positions(),
            creature_types: self.read_creature_type(),
            alive: self.read_alive(),
        }
    }

    fn glyph_table(&self) -> Vec<engine::VizGlyph> {
        // creature_type discriminants: Wolf=0, Sheep=1.
        // ANSI 256: 196 = bright red, 47 = bright green.
        vec![
            engine::VizGlyph::new('W', 196),
            engine::VizGlyph::new('s', 47),
        ]
    }

    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        // Initial layout: wolves at x∈[4,14], y∈[-5,5];
        // sheep at x∈[-9.5,-0.5], y∈[-3.5,3.5]. Pad ~3 for births.
        Some((Vec3::new(-15.0, -8.0, 0.0), Vec3::new(20.0, 8.0, 0.0)))
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(PredatorPreyRealState::new(seed, agent_count))
}
