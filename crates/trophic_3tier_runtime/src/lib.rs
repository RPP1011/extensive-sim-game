//! Per-fixture runtime for `assets/sim/trophic_3tier.sim` — TWELFTH
//! REAL fixture: three-tier food web (Grass → Herbivore → Carnivore)
//! composed in a single shared Agent SoA.
//!
//! Architectural escalation from `predator_prey_real_runtime`:
//!
//!   1. **Three LIVE tiers.** Grass (creature_type=0), Herbivore
//!      (creature_type=1), Carnivore (creature_type=2). All three
//!      participate in per-tick physics — Grass via passive
//!      regrowth, Herbivores via grass-eating + decay, Carnivores
//!      via hunting + decay.
//!
//!   2. **Real food entity (not uniform graze proxy).** Sheep in
//!      predator_prey_real "grazed" via a uniform per-tick hunger
//!      gain; here, Herbivores must be neighbours of an alive
//!      Grass tile with hp >= eat_threshold to gain energy via
//!      ApplyEat (the same chronicle pattern foraging_real uses
//!      for Ants eating FoodPiles).
//!
//!   3. **Three-species birth machinery (CPU-side).**
//!      - Carnivores: when summed `kills_total` view crosses each
//!        integer multiple of `KILLS_PER_CARNIVORE_BIRTH`, runtime
//!        flips next dead Carnivore slot alive=1.
//!      - Herbivores: every `HERBIVORE_BREED_INTERVAL` ticks, runtime
//!        counts well-fed herbivores; flips floor(well_fed/2) dead
//!        Herbivore slots alive=1.
//!      - Grass: per-tick respawn — every `GRASS_RESPAWN_INTERVAL`
//!        ticks, runtime flips up to `GRASS_RESPAWN_PER_TICK` dead
//!        Grass slots alive=1 with hp=initial. Models seed-spread.
//!
//! Per-tick chain:
//!
//!   1. clear_tail + clear ring headers + clear spatial offsets
//!   2. spatial_build_hash (5 phases)
//!   3. physics_GrassRegrow_and_HerbivoreEat — fused: per-Grass
//!      hp += grass_regrow; per-Herbivore body-form spatial walk
//!      emits Eat per (herbivore, grass) neighbour pair (gated /3)
//!   4. physics_ApplyEat — chronicle: decrement grass.hp, set_alive
//!      false on hp<=0, increment herbivore.hunger
//!   5. physics_CarnivoreHunt — body-form spatial walk; emits
//!      Strike per (carnivore, herbivore) neighbour pair (gated /4)
//!   6. physics_ApplyStrike — chronicle: decrement herbivore.hp,
//!      set_alive false + emit Killed on hp<=0, increment
//!      carnivore.hunger + carnivore.hp (kill credit)
//!   7. physics_EnergyDecay — universal hunger drain on
//!      Herbivore + Carnivore (gated creature_type != Grass);
//!      set_alive false + emit Starved on hunger<=0
//!   8. seed_indirect_0
//!   9. fold kills_total + fold prey_killed_total + fold starved_total
//!
//! After step(), CPU-side `process_lifecycle()` reads kills_total +
//! starved_total + alive + creature_type buffers and:
//!   - Flips dead Carnivore slots alive=1 every KILLS_PER_CARNIVORE_BIRTH.
//!   - Flips dead Herbivore slots alive=1 every HERBIVORE_BREED_INTERVAL.
//!   - Flips dead Grass slots alive=1 every GRASS_RESPAWN_INTERVAL.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Discriminant for the first-declared entity (Grass).
pub const CT_GRASS: u32 = 0;
/// Discriminant for the second-declared entity (Herbivore).
pub const CT_HERBIVORE: u32 = 1;
/// Discriminant for the third-declared entity (Carnivore).
pub const CT_CARNIVORE: u32 = 2;

/// Initial Grass tile count. Slots [0..INITIAL_GRASS).
pub const INITIAL_GRASS: u32 = 200;
/// Initial Herbivore count. Slots [GRASS_CAP..GRASS_CAP + INITIAL_HERBIVORES).
pub const INITIAL_HERBIVORES: u32 = 50;
/// Initial Carnivore count. Slots [GRASS_CAP+HERB_CAP..GRASS_CAP+HERB_CAP+INITIAL_CARNIVORES).
pub const INITIAL_CARNIVORES: u32 = 10;

/// Grass slot reservation. [0..GRASS_CAP) are ALWAYS Grass.
pub const GRASS_CAP: u32 = 400;
/// Herbivore slot reservation. [GRASS_CAP..GRASS_CAP+HERB_CAP) are ALWAYS Herbivore.
pub const HERB_CAP: u32 = 150;
/// Carnivore slot reservation. [GRASS_CAP+HERB_CAP..SLOT_CAP) are ALWAYS Carnivore.
pub const CARN_CAP: u32 = 50;
/// Total slot capacity = 600.
pub const SLOT_CAP: u32 = GRASS_CAP + HERB_CAP + CARN_CAP;

/// Initial Grass density (hp). Caps at GRASS_CAP_DENSITY via CPU clamp.
pub const INITIAL_GRASS_DENSITY: f32 = 60.0;
/// Maximum Grass density per tile — CPU side clamp because the DSL
/// GrassRegrow rule has no upper-bound clamp baked in.
pub const GRASS_CAP_DENSITY: f32 = 100.0;
/// Initial Herbivore hunger (energy). With decay_rate=1.0 + plant_gain=4.0
/// per Eat, a herbivore in grass-rich territory net-gains.
pub const INITIAL_HERB_ENERGY: f32 = 50.0;
/// Initial Herbivore hp (combat health). With strike_damage=4.0,
/// ~5 carnivore hits drop a herbivore.
pub const INITIAL_HERB_HP: f32 = 20.0;
/// Initial Carnivore hunger (energy).
pub const INITIAL_CARN_ENERGY: f32 = 60.0;

/// Herbivore kills required per Carnivore birth. Tuned to allow a few
/// carnivore births over 1000 ticks but not let them outbreed prey.
pub const KILLS_PER_CARNIVORE_BIRTH: f32 = 6.0;
/// Tick cadence for the herbivore-breed CPU-side check.
pub const HERBIVORE_BREED_INTERVAL: u64 = 8;
/// Hunger threshold for a herbivore to be "well-fed" and eligible to
/// contribute to breeding.
pub const HERBIVORE_BREED_THRESHOLD: f32 = 40.0;
/// Tick cadence for the grass-respawn CPU-side check.
pub const GRASS_RESPAWN_INTERVAL: u64 = 5;
/// Maximum dead-grass slots flipped alive per respawn cycle.
pub const GRASS_RESPAWN_PER_TICK: u32 = 4;
/// Density a respawned grass tile starts at.
pub const GRASS_RESPAWN_DENSITY: f32 = 30.0;

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

/// Per-tick lifecycle counters for the trace.
#[derive(Default, Debug, Clone, Copy)]
pub struct TickLifecycle {
    pub grass_births: u32,
    pub herb_births: u32,
    pub carn_births: u32,
    pub grass_deaths: u32,
    pub herb_starvations: u32,
    pub carn_starvations: u32,
    pub herb_kills: u32,
}

pub struct Trophic3TierState {
    gpu: GpuContext,

    // -- Agent SoA (shared between Grass + Herbivore + Carnivore,
    //    discriminated by creature_type). Sized to SLOT_CAP. --
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
    prey_killed_total_view: ViewStorage,
    prey_killed_total_cfg_buf: wgpu::Buffer,
    starved_total_view: ViewStorage,
    starved_total_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    grass_eat_cfg_buf: wgpu::Buffer,
    applyeat_cfg_buf: wgpu::Buffer,
    carnhunt_cfg_buf: wgpu::Buffer,
    applystrike_cfg_buf: wgpu::Buffer,
    energydecay_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    #[allow(dead_code)]
    seed: u64,

    // -- CPU-side lifecycle bookkeeping --
    carn_births_so_far: u32,
    herb_births_so_far: u32,
    grass_births_so_far: u32,
    herb_starvations_so_far: u32,
    carn_starvations_so_far: u32,
    herb_kills_so_far: u32,
    grass_deaths_so_far: u32,
    last_lifecycle: TickLifecycle,
    /// Snapshot of alive bitmap from end-of-prev-tick (after births).
    prev_alive: Vec<u32>,
}

impl Trophic3TierState {
    pub fn new(seed: u64, _agent_count_hint: u32) -> Self {
        let n = SLOT_CAP as usize;

        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut hp_init: Vec<f32> = Vec::with_capacity(n);
        let mut alive_init: Vec<u32> = Vec::with_capacity(n);
        let mut hunger_init: Vec<f32> = Vec::with_capacity(n);
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);

        // Layout (deterministic positions for replay parity):
        //   - Grass: 200 tiles spread across a 20×10 grid step 1.0,
        //     covering the FULL field x∈[-10..10], y∈[-5..5].
        //   - Herbivores: 50 on a 10×5 grid step 1.0, anchored across
        //     x∈[-5..5] (interleaved with grass).
        //   - Carnivores: 10 in a sparse cluster on the right edge.
        //
        // Mixed positions ensure spatial-hash neighbourhoods span all
        // 3 tiers — every herbivore has grass within 1.5, every
        // carnivore has herbivores in range over time as the herd
        // diffuses.
        for slot in 0..SLOT_CAP {
            if slot < INITIAL_GRASS {
                // Live Grass tile. 20×10 grid step 1.0, covering
                // x∈[-10..9], y∈[-5..4].
                let row = (slot / 20) as f32;
                let col = (slot % 20) as f32;
                let x = -10.0 + col * 1.0;
                let y = -5.0 + row * 1.0;
                pos_padded.push(Vec3::new(x, y, 0.0).into());
                hp_init.push(INITIAL_GRASS_DENSITY); // density
                alive_init.push(1);
                hunger_init.push(0.0); // unused for grass
                creature_init.push(CT_GRASS);
            } else if slot < GRASS_CAP {
                // Dead Grass placeholder. Spawn births land scattered.
                pos_padded.push(Vec3::new(0.0, 0.0, 0.0).into());
                hp_init.push(0.0);
                alive_init.push(0);
                hunger_init.push(0.0);
                creature_init.push(CT_GRASS);
            } else if slot < GRASS_CAP + INITIAL_HERBIVORES {
                // Live Herbivore. 10×5 grid step 1.0, x∈[-5..4],
                // y∈[-2..2]. Anchored slightly leftward so the
                // first herbivores are not all right next to the
                // carnivore cluster.
                let h_idx = slot - GRASS_CAP;
                let row = (h_idx / 10) as f32;
                let col = (h_idx % 10) as f32;
                let x = -5.0 + col * 1.0;
                let y = -2.0 + row * 1.0;
                pos_padded.push(Vec3::new(x, y, 0.0).into());
                hp_init.push(INITIAL_HERB_HP);
                alive_init.push(1);
                hunger_init.push(INITIAL_HERB_ENERGY);
                creature_init.push(CT_HERBIVORE);
            } else if slot < GRASS_CAP + HERB_CAP {
                // Dead Herbivore placeholder.
                pos_padded.push(Vec3::new(0.0, 0.0, 0.0).into());
                hp_init.push(0.0);
                alive_init.push(0);
                hunger_init.push(0.0);
                creature_init.push(CT_HERBIVORE);
            } else if slot < GRASS_CAP + HERB_CAP + INITIAL_CARNIVORES {
                // Live Carnivore. Sparse cluster on the right edge of
                // the field, x∈[6..9], y∈[-2..2].
                let c_idx = slot - (GRASS_CAP + HERB_CAP);
                let row = (c_idx / 5) as f32;
                let col = (c_idx % 5) as f32;
                let x = 6.0 + col * 0.75;
                let y = (row - 0.5) * 2.0;
                pos_padded.push(Vec3::new(x, y, 0.0).into());
                hp_init.push(0.0); // kill credit, starts 0
                alive_init.push(1);
                hunger_init.push(INITIAL_CARN_ENERGY);
                creature_init.push(CT_CARNIVORE);
            } else {
                // Dead Carnivore placeholder.
                pos_padded.push(Vec3::new(0.0, 0.0, 0.0).into());
                hp_init.push(0.0);
                alive_init.push(0);
                hunger_init.push(0.0);
                creature_init.push(CT_CARNIVORE);
            }
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let agent_pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::agent_pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_hunger_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::agent_hunger"),
            contents: bytemuck::cast_slice(&hunger_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trophic_3tier_runtime::agent_creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Spatial-grid buffers (mirror predator_prey_real_runtime) ----
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (SLOT_CAP as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trophic_3tier_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trophic_3tier_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trophic_3tier_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = sp::num_cells().div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trophic_3tier_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trophic_3tier_runtime::spatial_offsets_zero"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Event ring + view storage ----
        let event_ring = EventRing::new(&gpu, "trophic_3tier_runtime");
        let kills_total_view = ViewStorage::new(
            &gpu,
            "trophic_3tier_runtime::kills_total",
            SLOT_CAP,
            false,
            false,
        );
        let prey_killed_total_view = ViewStorage::new(
            &gpu,
            "trophic_3tier_runtime::prey_killed_total",
            SLOT_CAP,
            false,
            false,
        );
        let starved_total_view = ViewStorage::new(
            &gpu,
            "trophic_3tier_runtime::starved_total",
            SLOT_CAP,
            false,
            false,
        );

        // ---- Per-kernel cfg uniforms ----
        let grass_eat_cfg = physics_GrassRegrow_and_HerbivoreEat::PhysicsGrassRegrowAndHerbivoreEatCfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let grass_eat_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::grass_eat_cfg"),
            contents: bytemuck::bytes_of(&grass_eat_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let applyeat_cfg = physics_ApplyEat::PhysicsApplyEatCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let applyeat_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::applyeat_cfg"),
            contents: bytemuck::bytes_of(&applyeat_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let carnhunt_cfg = physics_CarnivoreHunt::PhysicsCarnivoreHuntCfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let carnhunt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::carnhunt_cfg"),
            contents: bytemuck::bytes_of(&carnhunt_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let applystrike_cfg = physics_ApplyStrike::PhysicsApplyStrikeCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let applystrike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trophic_3tier_runtime::applystrike_cfg"),
            contents: bytemuck::bytes_of(&applystrike_cfg),
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
                label: Some("trophic_3tier_runtime::energydecay_cfg"),
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
            label: Some("trophic_3tier_runtime::seed_cfg"),
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
        let kills_total_cfg_buf = make_view_cfg("trophic_3tier_runtime::kills_total_cfg");
        let prey_killed_total_cfg_buf =
            make_view_cfg("trophic_3tier_runtime::prey_killed_total_cfg");
        let starved_total_cfg_buf = make_view_cfg("trophic_3tier_runtime::starved_total_cfg");

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
            prey_killed_total_view,
            prey_killed_total_cfg_buf,
            starved_total_view,
            starved_total_cfg_buf,
            grass_eat_cfg_buf,
            applyeat_cfg_buf,
            carnhunt_cfg_buf,
            applystrike_cfg_buf,
            energydecay_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count: SLOT_CAP,
            seed,
            carn_births_so_far: 0,
            herb_births_so_far: 0,
            grass_births_so_far: 0,
            herb_starvations_so_far: 0,
            carn_starvations_so_far: 0,
            herb_kills_so_far: 0,
            grass_deaths_so_far: 0,
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
    pub fn read_creature_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_creature_type_buf, "creature_type")
    }

    pub fn last_lifecycle(&self) -> TickLifecycle { self.last_lifecycle }
    pub fn carn_births_so_far(&self) -> u32 { self.carn_births_so_far }
    pub fn herb_births_so_far(&self) -> u32 { self.herb_births_so_far }
    pub fn grass_births_so_far(&self) -> u32 { self.grass_births_so_far }
    pub fn herb_starvations_so_far(&self) -> u32 { self.herb_starvations_so_far }
    pub fn carn_starvations_so_far(&self) -> u32 { self.carn_starvations_so_far }
    pub fn herb_kills_so_far(&self) -> u32 { self.herb_kills_so_far }
    pub fn grass_deaths_so_far(&self) -> u32 { self.grass_deaths_so_far }
    pub fn slot_cap(&self) -> u32 { SLOT_CAP }
    pub fn grass_cap(&self) -> u32 { GRASS_CAP }
    pub fn herb_cap(&self) -> u32 { HERB_CAP }
    pub fn carn_cap(&self) -> u32 { CARN_CAP }

    /// Count alive Grass tiles.
    pub fn count_alive_grass(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_GRASS)
            .count() as u32
    }

    /// Count alive Herbivores.
    pub fn count_alive_herbivores(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_HERBIVORE)
            .count() as u32
    }

    /// Count alive Carnivores.
    pub fn count_alive_carnivores(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_CARNIVORE)
            .count() as u32
    }

    /// Sum of energy per tier (Grass: total density; Herb/Carn: total
    /// hunger).
    pub fn tier_energy_totals(&self) -> (f32, f32, f32) {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        let hp = self.read_hp();
        let hunger = self.read_hunger();
        let mut grass = 0.0f32;
        let mut herb = 0.0f32;
        let mut carn = 0.0f32;
        for slot in 0..SLOT_CAP as usize {
            if alive[slot] != 1 {
                continue;
            }
            match ct[slot] {
                CT_GRASS => grass += hp[slot],
                CT_HERBIVORE => herb += hunger[slot],
                CT_CARNIVORE => carn += hunger[slot],
                _ => {}
            }
        }
        (grass, herb, carn)
    }

    /// CPU-side lifecycle processing. Drives:
    ///   1. Carnivore births: every KILLS_PER_CARNIVORE_BIRTH cumulative
    ///      herbivore-combat-kills, flip dead Carnivore slot alive=1.
    ///   2. Herbivore births: every HERBIVORE_BREED_INTERVAL ticks, count
    ///      well-fed herbivores, flip floor(well_fed/2) dead slots alive=1.
    ///   3. Grass respawn + density cap: every GRASS_RESPAWN_INTERVAL
    ///      ticks, flip up to GRASS_RESPAWN_PER_TICK dead Grass slots
    ///      alive=1; clamp all alive Grass hp to GRASS_CAP_DENSITY.
    ///   4. Starvation accounting: read starved_total view, partition
    ///      by creature_type to update per-tier counters.
    fn process_lifecycle(&mut self) -> TickLifecycle {
        let starved_per_slot = self.starved_total_view.readback(&self.gpu).to_vec();
        let mut alive = self.read_alive();
        let ct = self.read_creature_type();
        let mut hunger = self.read_hunger();
        let mut hp = self.read_hp();
        let mut pos = self.read_pos_padded();

        // Walk slots once for per-tier deltas + cumulative counters.
        let mut herb_starved_total = 0u32;
        let mut carn_starved_total = 0u32;
        let mut herb_deaths_this_tick = 0u32;
        let mut carn_deaths_this_tick = 0u32;
        let mut grass_deaths_this_tick = 0u32;
        for (slot, &cur) in alive.iter().enumerate() {
            let prev = self.prev_alive[slot];
            let c = ct[slot];
            let s = starved_per_slot[slot].round() as u32;
            if c == CT_HERBIVORE {
                herb_starved_total += s;
            } else if c == CT_CARNIVORE {
                carn_starved_total += s;
            }
            if prev == 1 && cur == 0 {
                match c {
                    CT_GRASS => grass_deaths_this_tick += 1,
                    CT_HERBIVORE => herb_deaths_this_tick += 1,
                    CT_CARNIVORE => carn_deaths_this_tick += 1,
                    _ => {}
                }
            }
        }
        let new_herb_starv =
            herb_starved_total.saturating_sub(self.herb_starvations_so_far);
        let new_carn_starv =
            carn_starved_total.saturating_sub(self.carn_starvations_so_far);
        self.herb_starvations_so_far = herb_starved_total;
        self.carn_starvations_so_far = carn_starved_total;

        // Combat herbivore deaths = total herb deaths - herb starvations.
        let combat_herb_deaths_this_tick =
            herb_deaths_this_tick.saturating_sub(new_herb_starv);
        let new_herb_kills = combat_herb_deaths_this_tick;
        self.herb_kills_so_far =
            self.herb_kills_so_far.saturating_add(new_herb_kills);
        let total_combat_kills = self.herb_kills_so_far;

        self.grass_deaths_so_far =
            self.grass_deaths_so_far.saturating_add(grass_deaths_this_tick);

        let _ = carn_deaths_this_tick; // tracked via starvation above

        // --- Carnivore births: driven by cumulative herbivore-combat-kills.
        let intended_carn_births =
            ((total_combat_kills as f32) / KILLS_PER_CARNIVORE_BIRTH).floor() as u32;
        let carn_births_to_fire =
            intended_carn_births.saturating_sub(self.carn_births_so_far);
        let mut carn_births_fired = 0u32;
        if carn_births_to_fire > 0 {
            let carn_start = (GRASS_CAP + HERB_CAP) as usize;
            let carn_end = SLOT_CAP as usize;
            for slot in carn_start..carn_end {
                if carn_births_fired >= carn_births_to_fire {
                    break;
                }
                if ct[slot] == CT_CARNIVORE && alive[slot] == 0 {
                    alive[slot] = 1;
                    hunger[slot] = INITIAL_CARN_ENERGY;
                    hp[slot] = 0.0;
                    // New carnivores spawn back at the carnivore cluster
                    // (right edge of the field). Same 5×N grid step 0.75.
                    let ring_slot = (self.carn_births_so_far + carn_births_fired) as usize
                        % (INITIAL_CARNIVORES as usize).max(1);
                    let row = (ring_slot / 5) as f32;
                    let col = (ring_slot % 5) as f32;
                    let x = 6.0 + col * 0.75;
                    let y = (row - 0.5) * 2.0;
                    pos[slot] = Vec3::new(x, y, 0.0).into();
                    carn_births_fired += 1;
                }
            }
            self.carn_births_so_far += carn_births_fired;
        }

        // --- Herbivore births: every HERBIVORE_BREED_INTERVAL ticks.
        let mut herb_births_fired = 0u32;
        if self.tick > 0 && self.tick % HERBIVORE_BREED_INTERVAL == 0 {
            let mut well_fed = 0u32;
            let herb_start = GRASS_CAP as usize;
            let herb_end = (GRASS_CAP + HERB_CAP) as usize;
            for slot in herb_start..herb_end {
                if alive[slot] == 1
                    && ct[slot] == CT_HERBIVORE
                    && hunger[slot] > HERBIVORE_BREED_THRESHOLD
                {
                    well_fed += 1;
                }
            }
            let intended = well_fed / 2;
            let mut to_fire = intended;
            for slot in herb_start..herb_end {
                if to_fire == 0 {
                    break;
                }
                if ct[slot] == CT_HERBIVORE && alive[slot] == 0 {
                    alive[slot] = 1;
                    hunger[slot] = INITIAL_HERB_ENERGY;
                    hp[slot] = INITIAL_HERB_HP;
                    // Place new herbivores across a wider scatter than
                    // the initial 10×5 grid — fan out into x∈[-9..9],
                    // y∈[-3..3] so the herd grows into both grass-
                    // territory AND carnivore-territory.
                    let ring_slot = (self.herb_births_so_far + herb_births_fired) as usize
                        % 100;
                    let row = (ring_slot / 10) as f32;
                    let col = (ring_slot % 10) as f32;
                    let x = -9.0 + col * 2.0;
                    let y = (row - 4.5) * 1.0;
                    pos[slot] = Vec3::new(x, y, 0.0).into();
                    herb_births_fired += 1;
                    to_fire -= 1;
                }
            }
            self.herb_births_so_far += herb_births_fired;
        }

        // --- Grass respawn + density cap.
        let mut grass_births_fired = 0u32;
        if self.tick > 0 && self.tick % GRASS_RESPAWN_INTERVAL == 0 {
            let mut to_fire = GRASS_RESPAWN_PER_TICK;
            for slot in 0..(GRASS_CAP as usize) {
                if to_fire == 0 {
                    break;
                }
                if ct[slot] == CT_GRASS && alive[slot] == 0 {
                    alive[slot] = 1;
                    hp[slot] = GRASS_RESPAWN_DENSITY;
                    // Respawn at the slot's ORIGINAL grid coordinates
                    // — slot index → grid position is deterministic.
                    let row = (slot / 20) as f32;
                    let col = (slot % 20) as f32;
                    let x = -10.0 + col * 1.0;
                    let y = -5.0 + row * 1.0;
                    pos[slot] = Vec3::new(x, y, 0.0).into();
                    grass_births_fired += 1;
                    to_fire -= 1;
                }
            }
            self.grass_births_so_far += grass_births_fired;
        }
        // Always-on density cap: keep alive Grass hp from blowing past
        // GRASS_CAP_DENSITY (no DSL-level clamp on GrassRegrow).
        let mut clamped_any = false;
        for slot in 0..(GRASS_CAP as usize) {
            if alive[slot] == 1 && ct[slot] == CT_GRASS && hp[slot] > GRASS_CAP_DENSITY {
                hp[slot] = GRASS_CAP_DENSITY;
                clamped_any = true;
            }
        }

        if carn_births_fired > 0
            || herb_births_fired > 0
            || grass_births_fired > 0
            || clamped_any
        {
            self.gpu
                .queue
                .write_buffer(&self.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
            self.gpu
                .queue
                .write_buffer(&self.agent_hunger_buf, 0, bytemuck::cast_slice(&hunger));
            self.gpu.queue.write_buffer(&self.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
            self.gpu.queue.write_buffer(&self.agent_pos_buf, 0, bytemuck::cast_slice(&pos));
        }

        // Snapshot alive AFTER births land — next tick's 1→0 detection
        // runs against this post-birth baseline.
        self.prev_alive = alive;

        TickLifecycle {
            grass_births: grass_births_fired,
            herb_births: herb_births_fired,
            carn_births: carn_births_fired,
            grass_deaths: grass_deaths_this_tick,
            herb_starvations: new_herb_starv,
            carn_starvations: new_carn_starv,
            herb_kills: new_herb_kills,
        }
    }

    fn read_pos_padded(&self) -> Vec<Vec3Padded> {
        let bytes = (SLOT_CAP as u64) * 16;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trophic_3tier_runtime::pos_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trophic_3tier_runtime::read_pos") },
        );
        encoder.copy_buffer_to_buffer(&self.agent_pos_buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
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
            label: Some(&format!("trophic_3tier_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trophic_3tier_runtime::read_f32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
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

    fn read_u32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<u32> {
        let bytes = (SLOT_CAP as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("trophic_3tier_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trophic_3tier_runtime::read_u32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }
}

impl CompiledSim for Trophic3TierState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trophic_3tier_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        use dsl_compiler::cg::emit::spatial as sp;
        let max_neighbour_emits =
            SLOT_CAP.saturating_mul(sp::MAX_PER_CELL).saturating_mul(27);
        // Bound headers above worst case: HerbivoreEat + CarnivoreHunt
        // both fan out per neighbour, plus EnergyDecay emits per dying
        // agent.
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

        // (2) Spatial-hash counting sort.
        let grass_eat_cfg = physics_GrassRegrow_and_HerbivoreEat::PhysicsGrassRegrowAndHerbivoreEatCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.grass_eat_cfg_buf,
            0,
            bytemuck::bytes_of(&grass_eat_cfg),
        );

        let count_b = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.grass_eat_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_count(
            &mut self.cache, &count_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );
        let scan_local_b = spatial_build_hash_scan_local::SpatialBuildHashScanLocalBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.grass_eat_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_local(
            &mut self.cache, &scan_local_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );
        let scan_carry_b = spatial_build_hash_scan_carry::SpatialBuildHashScanCarryBindings {
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.grass_eat_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_carry(
            &mut self.cache, &scan_carry_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );
        let scan_add_b = spatial_build_hash_scan_add::SpatialBuildHashScanAddBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.grass_eat_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_add(
            &mut self.cache, &scan_add_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );
        let scatter_b = spatial_build_hash_scatter::SpatialBuildHashScatterBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.grass_eat_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache, &scatter_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );

        // (3) GrassRegrow + HerbivoreEat (fused). Emits Eat events for
        //     herbivore→grass neighbour pairs.
        let grass_eat_b = physics_GrassRegrow_and_HerbivoreEat::PhysicsGrassRegrowAndHerbivoreEatBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.grass_eat_cfg_buf,
        };
        dispatch::dispatch_physics_grassregrow_and_herbivoreeat(
            &mut self.cache, &grass_eat_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );

        // (4) ApplyEat — chronicle. Reads Eat events.
        let event_count_estimate = max_neighbour_emits.min(65536);
        let applyeat_cfg = physics_ApplyEat::PhysicsApplyEatCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.applyeat_cfg_buf, 0, bytemuck::bytes_of(&applyeat_cfg),
        );
        let applyeat_b = physics_ApplyEat::PhysicsApplyEatBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            cfg: &self.applyeat_cfg_buf,
        };
        dispatch::dispatch_physics_applyeat(
            &mut self.cache, &applyeat_b, &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (5) CarnivoreHunt — body-form spatial walk. Emits Strike.
        let carnhunt_cfg = physics_CarnivoreHunt::PhysicsCarnivoreHuntCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.carnhunt_cfg_buf, 0, bytemuck::bytes_of(&carnhunt_cfg),
        );
        let carnhunt_b = physics_CarnivoreHunt::PhysicsCarnivoreHuntBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.carnhunt_cfg_buf,
        };
        dispatch::dispatch_physics_carnivorehunt(
            &mut self.cache, &carnhunt_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );

        // (6) ApplyStrike — chronicle. Reads Strike events.
        let applystrike_cfg = physics_ApplyStrike::PhysicsApplyStrikeCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.applystrike_cfg_buf, 0, bytemuck::bytes_of(&applystrike_cfg),
        );
        let applystrike_b = physics_ApplyStrike::PhysicsApplyStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            cfg: &self.applystrike_cfg_buf,
        };
        dispatch::dispatch_physics_applystrike(
            &mut self.cache, &applystrike_b, &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (7) EnergyDecay — universal hunger drain on Herbivore + Carnivore.
        let energydecay_cfg = physics_EnergyDecay::PhysicsEnergyDecayCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.energydecay_cfg_buf, 0, bytemuck::bytes_of(&energydecay_cfg),
        );
        let energydecay_b = physics_EnergyDecay::PhysicsEnergyDecayBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            cfg: &self.energydecay_cfg_buf,
        };
        dispatch::dispatch_physics_energydecay(
            &mut self.cache, &energydecay_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );

        // (8) seed_indirect_0.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let seed_b = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache, &seed_b, &self.gpu.device, &mut encoder, SLOT_CAP,
        );

        // (9) Folds.
        let mk_fold_cfg = |ec: u32, tick: u32| fold_kills_total::FoldKillsTotalCfg {
            event_count: ec,
            tick,
            second_key_pop: 1,
            _pad: 0,
        };
        let fold_cfg = mk_fold_cfg(event_count_estimate, self.tick as u32);
        for buf in [
            &self.kills_total_cfg_buf,
            &self.prey_killed_total_cfg_buf,
            &self.starved_total_cfg_buf,
        ] {
            self.gpu.queue.write_buffer(buf, 0, bytemuck::bytes_of(&fold_cfg));
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
            &mut self.cache, &kills_b, &self.gpu.device, &mut encoder, event_count_estimate,
        );
        let pk_b = fold_prey_killed_total::FoldPreyKilledTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.prey_killed_total_view.primary(),
            view_storage_anchor: self.prey_killed_total_view.anchor(),
            view_storage_ids: self.prey_killed_total_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.prey_killed_total_cfg_buf,
        };
        dispatch::dispatch_fold_prey_killed_total(
            &mut self.cache, &pk_b, &self.gpu.device, &mut encoder, event_count_estimate,
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
            &mut self.cache, &starved_b, &self.gpu.device, &mut encoder, event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.kills_total_view.mark_dirty();
        self.prey_killed_total_view.mark_dirty();
        self.starved_total_view.mark_dirty();
        self.tick += 1;

        let lc = self.process_lifecycle();
        self.last_lifecycle = lc;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(Trophic3TierState::new(seed, agent_count))
}
