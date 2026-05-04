//! Per-fixture runtime for `assets/sim/foraging_real.sim` — first
//! LIFECYCLE fixture (alive=false → true births + alive=true → false
//! deaths actually flipping during a run).
//!
//! Mirrors `duel_25v25_runtime` for the spatial body-form +
//! chronicle-physics scaffolding, with three architectural additions:
//!
//!   1. **Two entity types in one Agent SoA.** Slots are split between
//!      Ant (creature_type=0, the first-declared entity) and FoodPile
//!      (creature_type=1). Hungry/Hp fields carry foraging-specific
//!      semantics: Ant.hunger is energy (0..N, decays each tick),
//!      Ant.hp is stockpile credit (counts contributions to colony).
//!      FoodPile.hp is remaining quantity (decremented per Eat).
//!
//!   2. **CPU-side birth slot allocation.** The DSL handles deaths
//!      natively via `agents.set_alive(self, false)` from EnergyDecay
//!      and ApplyEat. Births (the alive=false → true direction the
//!      alive-bitmap had never seen flip before this fixture) live
//!      CPU-side: each tick, the runtime reads the `eat_count` view +
//!      the alive buffer, computes "intended births = total_eats /
//!      eat_per_birth - births_so_far", and for each intended birth
//!      finds the next dead Ant slot and writes alive=1, hunger=
//!      initial_energy, hp=0. The next tick's spatial_build_hash +
//!      AntFeed see the new ant naturally because they read
//!      agent_alive directly (not AliveBitmap).
//!
//!   3. **No alive_pack / pack_agents calls.** Same approach
//!      duel_1v1 + duel_25v25 take — the schedule emits these system
//!      kernels but they're not required for the per-tick chain to
//!      run. AliveBitmap is only WRITTEN by alive_pack; nothing in
//!      the foraging WGSL reads it (all alive checks are direct
//!      `agent_alive[i] != 0u` reads).
//!
//! Per-tick chain:
//!
//!   1. clear_tail + clear ring headers + clear spatial offsets
//!   2. spatial_build_hash (5 phases): count → scan_local →
//!      scan_carry → scan_add → scatter
//!   3. AntFeed @phase(per_agent) — body-form spatial walk; emits
//!      Eat per (ant, food) neighbour pair (gated every 4 ticks)
//!   4. ApplyEat @phase(post) — chronicle: decrement food.hp,
//!      increment ant.hunger + ant.hp; emit FoodDepleted on hp<=0
//!   5. EnergyDecay @phase(per_agent) — per-ant hunger decrement,
//!      set_alive(self, false) on hunger<=0; emit Starved
//!   6. seed_indirect_0
//!   7. fold_eat_count (per-source f32)
//!   8. fold_food_consumed (per-target f32)
//!   9. fold_starved_count (per-target f32)
//!  10. fold_depleted_count (per-target f32)
//!
//! After step(), CPU-side `process_births()` reads eat_count + alive,
//! flips dead Ant slots back to alive=1 with reset state.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Discriminant value the WGSL emits for the first-declared entity
/// (`Ant`). Matches the `creature_type == Ant` lowering — entity
/// declaration order assigns 0 to Ant, 1 to FoodPile.
pub const CT_ANT: u32 = 0;
/// Discriminant for the second-declared entity (`FoodPile`).
pub const CT_FOOD: u32 = 1;

/// Initial colony size. Ants take slots [0..INITIAL_ANTS).
pub const INITIAL_ANTS: u32 = 50;
/// Initial food pile count. FoodPiles take slots
/// [INITIAL_ANTS..INITIAL_ANTS+INITIAL_FOOD).
pub const INITIAL_FOOD: u32 = 20;
/// Total slot capacity. Reserve 130 unused Ant slots beyond the
/// initial 50 so the colony can grow up to 4× before saturating.
pub const SLOT_CAP: u32 = 200;
/// Initial Ant hunger (energy). With decay_rate=0.5/tick from the
/// .sim, a starving ant lasts 100 ticks — a fed ant (eat_gain=12,
/// gated to one Eat per 4 ticks) net-gains 10 energy per active
/// window when food is plentiful.
pub const INITIAL_ENERGY: f32 = 50.0;
/// Initial FoodPile hp (quantity). Each Eat consumes 1 unit. With
/// 20 piles × 30 = 600 food units, sustains ~120 ant-energy-cycles.
pub const INITIAL_FOOD_QTY: f32 = 30.0;
/// Eats required to trigger one birth. With ~20-50 eats/active-tick
/// and the 4-tick AntFeed cooldown, hundreds of births fire over the
/// first wave. Tuned high enough to leave meaningful pacing across
/// the run without saturating slot_cap before food matters.
pub const EAT_PER_BIRTH: f32 = 60.0;

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
    pub births_this_tick: u32,
    pub deaths_this_tick: u32,
    pub eats_this_tick: u32,
}

/// Per-fixture state for the foraging colony.
pub struct ForagingRealState {
    gpu: GpuContext,

    // -- Agent SoA (shared between Ant + FoodPile, discriminated by
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
    eat_count_view: ViewStorage,
    eat_count_cfg_buf: wgpu::Buffer,
    food_consumed_view: ViewStorage,
    food_consumed_cfg_buf: wgpu::Buffer,
    starved_count_view: ViewStorage,
    starved_count_cfg_buf: wgpu::Buffer,
    depleted_count_view: ViewStorage,
    depleted_count_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    antfeed_cfg_buf: wgpu::Buffer,
    applyeat_cfg_buf: wgpu::Buffer,
    energydecay_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    /// Reserved for future P5 RNG use; today the layout + decisions
    /// are fully deterministic. Stored to keep the constructor
    /// signature parity with the duel_25v25 fixture pattern.
    #[allow(dead_code)]
    seed: u64,

    // -- CPU-side lifecycle bookkeeping --
    /// Cumulative births triggered so far. Each tick:
    /// `intended = floor(total_eats / EAT_PER_BIRTH)`; births to
    /// fire this tick = max(0, intended - births_so_far).
    births_so_far: u32,
    /// Cumulative deaths observed (reads of starved_count + alive
    /// drops). Used for trace output + sanity checks.
    deaths_so_far: u32,
    /// Last tick's lifecycle delta (for the per-tick trace).
    last_lifecycle: TickLifecycle,
    /// Last sampled total eat_count across all ant slots.
    last_total_eats: f32,
    /// Last sampled alive count for ants (used to detect deaths).
    last_alive_ants: u32,
}

impl ForagingRealState {
    /// Construct a colony battlefield with INITIAL_ANTS live ants in
    /// slots [0..INITIAL_ANTS), INITIAL_FOOD live food piles in slots
    /// [INITIAL_ANTS..INITIAL_ANTS+INITIAL_FOOD), and the rest as
    /// dead Ant placeholder slots that births can later recycle.
    ///
    /// The `_seed` is reserved for future P5 RNG use; today the layout
    /// is fully deterministic.
    pub fn new(seed: u64, _agent_count_hint: u32) -> Self {
        let n = SLOT_CAP as usize;

        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut hp_init: Vec<f32> = Vec::with_capacity(n);
        let mut alive_init: Vec<u32> = Vec::with_capacity(n);
        let mut hunger_init: Vec<f32> = Vec::with_capacity(n);
        let mut creature_init: Vec<u32> = Vec::with_capacity(n);

        // Layout:
        //   - Slots [0..INITIAL_ANTS): live Ants on a 7×8 grid
        //     centered at home (~origin). Spread on a 0.6 step so
        //     each ant has multiple food slots within the 1.5 spatial
        //     radius.
        //   - Slots [INITIAL_ANTS..INITIAL_ANTS+INITIAL_FOOD): live
        //     FoodPiles spread across the same area to seed Eat
        //     events from tick 0.
        //   - Slots [INITIAL_ANTS+INITIAL_FOOD..SLOT_CAP): dead Ant
        //     placeholders at origin. Births will flip these alive.
        //
        // All food piles cluster among ants on the SAME plane so the
        // spatial-grid neighbour walk surfaces them naturally.
        for slot in 0..SLOT_CAP {
            if slot < INITIAL_ANTS {
                // Live Ant. 7×8 grid step 0.6 → ~4.2 × 4.8 area.
                let row = (slot / 8) as f32;
                let col = (slot % 8) as f32;
                let x = (col - 3.5) * 0.6;
                let y = (row - 3.5) * 0.6;
                pos_padded.push(Vec3::new(x, y, 0.0).into());
                hp_init.push(0.0); // Ant.hp = stockpile credit, starts 0
                alive_init.push(1);
                hunger_init.push(INITIAL_ENERGY);
                creature_init.push(CT_ANT);
            } else if slot < INITIAL_ANTS + INITIAL_FOOD {
                // Live FoodPile. Spread on a 4×5 grid step 0.8 across
                // the same physical area as the ants.
                let f_idx = slot - INITIAL_ANTS;
                let row = (f_idx / 5) as f32;
                let col = (f_idx % 5) as f32;
                let x = (col - 2.0) * 0.8;
                let y = (row - 1.5) * 0.8;
                pos_padded.push(Vec3::new(x, y, 0.5).into());
                hp_init.push(INITIAL_FOOD_QTY);
                alive_init.push(1);
                hunger_init.push(0.0); // FoodPile.hunger unused
                creature_init.push(CT_FOOD);
            } else {
                // Dead Ant placeholder. Birth recycles these slots.
                // Position lives at origin (will be overwritten by
                // birth_at()).
                pos_padded.push(Vec3::new(0.0, 0.0, 0.0).into());
                hp_init.push(0.0);
                alive_init.push(0);
                hunger_init.push(0.0);
                creature_init.push(CT_ANT);
            }
        }

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let agent_pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_real_runtime::agent_pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_real_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_real_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_hunger_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_real_runtime::agent_hunger"),
            contents: bytemuck::cast_slice(&hunger_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("foraging_real_runtime::agent_creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Spatial-grid buffers (mirror duel_25v25_runtime) ----
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (SLOT_CAP as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foraging_real_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foraging_real_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foraging_real_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = sp::num_cells().div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foraging_real_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("foraging_real_runtime::spatial_offsets_zero"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // ---- Event ring + view storage ----
        let event_ring = EventRing::new(&gpu, "foraging_real_runtime");
        let eat_count_view = ViewStorage::new(
            &gpu,
            "foraging_real_runtime::eat_count",
            SLOT_CAP,
            false,
            false,
        );
        let food_consumed_view = ViewStorage::new(
            &gpu,
            "foraging_real_runtime::food_consumed",
            SLOT_CAP,
            false,
            false,
        );
        let starved_count_view = ViewStorage::new(
            &gpu,
            "foraging_real_runtime::starved_count",
            SLOT_CAP,
            false,
            false,
        );
        let depleted_count_view = ViewStorage::new(
            &gpu,
            "foraging_real_runtime::depleted_count",
            SLOT_CAP,
            false,
            false,
        );

        // ---- Per-kernel cfg uniforms ----
        let antfeed_cfg = physics_AntFeed::PhysicsAntFeedCfg {
            agent_cap: SLOT_CAP,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let antfeed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_real_runtime::antfeed_cfg"),
            contents: bytemuck::bytes_of(&antfeed_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let applyeat_cfg = physics_ApplyEat::PhysicsApplyEatCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            _pad0: 0,
        };
        let applyeat_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foraging_real_runtime::applyeat_cfg"),
            contents: bytemuck::bytes_of(&applyeat_cfg),
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
                label: Some("foraging_real_runtime::energydecay_cfg"),
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
            label: Some("foraging_real_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let make_view_cfg = |label: &str| -> wgpu::Buffer {
            let cfg = fold_eat_count::FoldEatCountCfg {
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
        let eat_count_cfg_buf = make_view_cfg("foraging_real_runtime::eat_count_cfg");
        let food_consumed_cfg_buf = make_view_cfg("foraging_real_runtime::food_consumed_cfg");
        let starved_count_cfg_buf = make_view_cfg("foraging_real_runtime::starved_count_cfg");
        let depleted_count_cfg_buf = make_view_cfg("foraging_real_runtime::depleted_count_cfg");

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
            eat_count_view,
            eat_count_cfg_buf,
            food_consumed_view,
            food_consumed_cfg_buf,
            starved_count_view,
            starved_count_cfg_buf,
            depleted_count_view,
            depleted_count_cfg_buf,
            antfeed_cfg_buf,
            applyeat_cfg_buf,
            energydecay_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count: SLOT_CAP,
            seed,
            births_so_far: 0,
            deaths_so_far: 0,
            last_lifecycle: TickLifecycle::default(),
            last_total_eats: 0.0,
            last_alive_ants: INITIAL_ANTS,
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

    pub fn eat_count(&mut self) -> &[f32] {
        self.eat_count_view.readback(&self.gpu)
    }
    pub fn food_consumed(&mut self) -> &[f32] {
        self.food_consumed_view.readback(&self.gpu)
    }
    pub fn starved_count(&mut self) -> &[f32] {
        self.starved_count_view.readback(&self.gpu)
    }
    pub fn depleted_count(&mut self) -> &[f32] {
        self.depleted_count_view.readback(&self.gpu)
    }

    pub fn last_lifecycle(&self) -> TickLifecycle { self.last_lifecycle }
    pub fn births_so_far(&self) -> u32 { self.births_so_far }
    pub fn deaths_so_far(&self) -> u32 { self.deaths_so_far }
    pub fn slot_cap(&self) -> u32 { SLOT_CAP }

    /// Count alive Ant slots (creature_type==0 && alive==1).
    pub fn count_alive_ants(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_ANT)
            .count() as u32
    }

    /// Count alive FoodPile slots.
    pub fn count_alive_food(&self) -> u32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        alive
            .iter()
            .zip(ct.iter())
            .filter(|(a, c)| **a == 1 && **c == CT_FOOD)
            .count() as u32
    }

    /// Total food remaining (sum of FoodPile.hp across alive piles).
    pub fn total_food_remaining(&self) -> f32 {
        let alive = self.read_alive();
        let ct = self.read_creature_type();
        let hp = self.read_hp();
        alive
            .iter()
            .zip(ct.iter())
            .zip(hp.iter())
            .filter(|((a, c), _)| **a == 1 && **c == CT_FOOD)
            .map(|(_, h)| h.max(0.0))
            .sum()
    }

    /// CPU-side birth processing. Called after each step(). Reads the
    /// per-source eat_count view (cumulative, summed across all ant
    /// slots) and the alive buffer; flips dead Ant slots to alive=1
    /// up to (total_eats / EAT_PER_BIRTH - births_so_far) times.
    /// Each born ant gets reset state: alive=1, hunger=INITIAL_ENERGY,
    /// hp=0, position scattered around the home grid.
    ///
    /// Returns the number of births fired this tick.
    fn process_births(&mut self) -> u32 {
        let eat = self.eat_count_view.readback(&self.gpu).to_vec();
        let total_eats: f32 = eat.iter().sum();
        let intended_births = (total_eats / EAT_PER_BIRTH).floor() as u32;
        let to_fire = intended_births.saturating_sub(self.births_so_far);
        if to_fire == 0 {
            return 0;
        }

        let mut alive = self.read_alive();
        let ct = self.read_creature_type();
        let mut hunger = self.read_hunger();
        let mut hp = self.read_hp();
        let mut pos = self.read_pos_padded();

        let mut fired = 0u32;
        // Walk slots looking for dead Ant placeholders. Skip
        // FoodPile slots and live ants.
        for slot in 0..SLOT_CAP as usize {
            if fired >= to_fire {
                break;
            }
            if ct[slot] == CT_ANT && alive[slot] == 0 {
                alive[slot] = 1;
                hunger[slot] = INITIAL_ENERGY;
                hp[slot] = 0.0;
                // Scatter the new ant back into the active 7x8 colony
                // grid using its slot index modulo a deterministic
                // ring (P5: no thread RNG; this is positional, not
                // randomised).
                let ring_slot = (self.births_so_far + fired) as usize % (INITIAL_ANTS as usize);
                let row = (ring_slot / 8) as f32;
                let col = (ring_slot % 8) as f32;
                let x = (col - 3.5) * 0.6;
                let y = (row - 3.5) * 0.6;
                pos[slot] = Vec3::new(x, y, 0.0).into();
                fired += 1;
            }
        }

        if fired == 0 {
            return 0;
        }

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

        self.births_so_far += fired;
        // Note: `last_total_eats` is updated by step() AFTER this
        // call, so the per-tick `eats_this_tick` delta computation
        // sees the pre-process_births baseline.
        fired
    }

    fn read_pos_padded(&self) -> Vec<Vec3Padded> {
        let bytes = (SLOT_CAP as u64) * 16;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foraging_real_runtime::pos_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foraging_real_runtime::read_pos"),
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
            label: Some(&format!("foraging_real_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foraging_real_runtime::read_f32"),
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
            label: Some(&format!("foraging_real_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foraging_real_runtime::read_u32"),
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

impl CompiledSim for ForagingRealState {
    fn step(&mut self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foraging_real_runtime::step"),
            });

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        use dsl_compiler::cg::emit::spatial as sp;
        let max_neighbour_emits = SLOT_CAP
            .saturating_mul(sp::MAX_PER_CELL)
            .saturating_mul(27);
        // Slots produced per tick: AntFeed emits Eat per neighbour
        // (up to ~27*MAX_PER_CELL per ant, gated by /4 cooldown);
        // ApplyEat may fan-out FoodDepleted; EnergyDecay emits
        // Starved per dying ant. Bound headers above the worst case.
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

        // (2) Spatial-hash counting sort. Required input for AntFeed.
        let antfeed_cfg = physics_AntFeed::PhysicsAntFeedCfg {
            agent_cap: SLOT_CAP,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.antfeed_cfg_buf, 0, bytemuck::bytes_of(&antfeed_cfg));

        let count_b = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.agent_pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.antfeed_cfg_buf,
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
            cfg: &self.antfeed_cfg_buf,
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
            cfg: &self.antfeed_cfg_buf,
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
            cfg: &self.antfeed_cfg_buf,
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
            cfg: &self.antfeed_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache,
            &scatter_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (3) AntFeed — body-form spatial walk. Emits Eat events.
        // The compiler-emitted shader implicitly reads `agent_pos
        // [agent_id]` for the 27-cell window centring even when
        // the body doesn't reference `self.pos`; the binding is
        // surfaced via the Pos read added to
        // `collect_stmt_dependencies` for `ForEachNeighborBody`.
        let antfeed_b = physics_AntFeed::PhysicsAntFeedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.antfeed_cfg_buf,
        };
        dispatch::dispatch_physics_antfeed(
            &mut self.cache,
            &antfeed_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (4) ApplyEat — chronicle. Reads Eat events, writes
        // food.hp + food.alive + ant.hunger + ant.hp.
        let event_count_estimate = max_neighbour_emits.min(65536);
        let applyeat_cfg = physics_ApplyEat::PhysicsApplyEatCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            _pad0: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.applyeat_cfg_buf, 0, bytemuck::bytes_of(&applyeat_cfg));
        let applyeat_b = physics_ApplyEat::PhysicsApplyEatBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_hunger: &self.agent_hunger_buf,
            cfg: &self.applyeat_cfg_buf,
        };
        dispatch::dispatch_physics_applyeat(
            &mut self.cache,
            &applyeat_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (5) EnergyDecay — per-ant per-tick hunger drain + death gate.
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
            agent_creature_type: &self.agent_creature_type_buf,
            cfg: &self.energydecay_cfg_buf,
        };
        dispatch::dispatch_physics_energydecay(
            &mut self.cache,
            &energydecay_b,
            &self.gpu.device,
            &mut encoder,
            SLOT_CAP,
        );

        // (6) seed_indirect_0 — keeps args buffer warm.
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

        // (7-10) Folds.
        let mk_fold_cfg = |ec: u32, tick: u32| fold_eat_count::FoldEatCountCfg {
            event_count: ec,
            tick,
            second_key_pop: 1,
            _pad: 0,
        };
        let fold_cfg = mk_fold_cfg(event_count_estimate, self.tick as u32);
        for buf in [
            &self.eat_count_cfg_buf,
            &self.food_consumed_cfg_buf,
            &self.starved_count_cfg_buf,
            &self.depleted_count_cfg_buf,
        ] {
            self.gpu
                .queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&fold_cfg));
        }

        let eat_b = fold_eat_count::FoldEatCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.eat_count_view.primary(),
            view_storage_anchor: self.eat_count_view.anchor(),
            view_storage_ids: self.eat_count_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.eat_count_cfg_buf,
        };
        dispatch::dispatch_fold_eat_count(
            &mut self.cache,
            &eat_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );
        let food_b = fold_food_consumed::FoldFoodConsumedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.food_consumed_view.primary(),
            view_storage_anchor: self.food_consumed_view.anchor(),
            view_storage_ids: self.food_consumed_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.food_consumed_cfg_buf,
        };
        dispatch::dispatch_fold_food_consumed(
            &mut self.cache,
            &food_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );
        let starved_b = fold_starved_count::FoldStarvedCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.starved_count_view.primary(),
            view_storage_anchor: self.starved_count_view.anchor(),
            view_storage_ids: self.starved_count_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.starved_count_cfg_buf,
        };
        dispatch::dispatch_fold_starved_count(
            &mut self.cache,
            &starved_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );
        let depleted_b = fold_depleted_count::FoldDepletedCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.depleted_count_view.primary(),
            view_storage_anchor: self.depleted_count_view.anchor(),
            view_storage_ids: self.depleted_count_view.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.depleted_count_cfg_buf,
        };
        dispatch::dispatch_fold_depleted_count(
            &mut self.cache,
            &depleted_b,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.eat_count_view.mark_dirty();
        self.food_consumed_view.mark_dirty();
        self.starved_count_view.mark_dirty();
        self.depleted_count_view.mark_dirty();
        self.tick += 1;

        // CPU-side lifecycle post-processing.
        let births = self.process_births();
        let alive_now = self.count_alive_ants();
        let deaths = self.last_alive_ants.saturating_sub(alive_now) + births; // alive went down by deaths-births
        // Re-derive deaths from starved_count for accuracy:
        let starved_total: f32 = self.starved_count_view.readback(&self.gpu).iter().sum();
        let starved_int = starved_total as u32;
        let deaths_this_tick = starved_int.saturating_sub(self.deaths_so_far);
        self.deaths_so_far = starved_int;

        let total_eats = self.eat_count_view.readback(&self.gpu).iter().sum::<f32>();
        let eats_this_tick = (total_eats - self.last_total_eats).round() as u32;
        self.last_total_eats = total_eats;
        self.last_alive_ants = alive_now;

        self.last_lifecycle = TickLifecycle {
            births_this_tick: births,
            deaths_this_tick,
            eats_this_tick,
        };
        // Suppress the unused-binding warning for `deaths` (kept as
        // a sanity-check shadow).
        let _ = deaths;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(ForagingRealState::new(seed, agent_count))
}
