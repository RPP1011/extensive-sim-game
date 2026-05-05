//! Per-fixture runtime for `assets/sim/tower_defense.sim` — the FIRST
//! recognizable game-shaped sim (not a coverage probe, not a duel).
//!
//! ## Composition
//!
//! All agents share the engine's per-agent SoA. Slot layout, by
//! construction in [`TowerDefenseState::new`]:
//!
//!   - Slot 0: Base. mana=1.0, hp=1000.0, alive=1, pos=(50,0,0).
//!   - Slots 1..=10: Defenders. mana=2.0, hp=100.0, alive=1, pos
//!     evenly spread on the defender line at x≈40.
//!   - Slots 11..=110: Enemy slots. Initially mana=3.0, hp=20.0,
//!     alive=0 (waiting for a wave to fill the slot). Spawn position
//!     written when the CPU-side wave fills the slot.
//!
//! ## Per-tick chain (mirrors duel_1v1's structure)
//!
//!   1. clear ring + mask + scoring_output
//!   2. mask_verb_Shoot — PerPair, writes mask_0 (Shoot can fire iff
//!      both mana bands match + distance gates pass).
//!   3. scoring — PerAgent argmax (only one verb here — argmax over a
//!      single mask just emits the picked target). Emits one
//!      ActionSelected per gated defender.
//!   4. physics_verb_chronicle_Shoot — gates on action_id==0u, emits
//!      Damaged{source, target, amount=defender_damage}.
//!   5. physics_MarchEnemies — per_agent: every alive enemy steps
//!      toward the base; if within base_hit_radius, emits BaseHit and
//!      sets alive=0.
//!   6. physics_ApplyDamage — PerEvent: reads Damaged events, writes
//!      target HP. On HP<=0 sets alive=0 and emits Defeated.
//!   7. seed_indirect_0 — keeps the indirect-args buffer warm.
//!   8. fold_defender_damage_dealt — per-source f32 accumulator.
//!   9. fold_base_damage_taken — per-source f32 accumulator.
//!
//! ## Wave spawning (CPU-side, runtime-owned)
//!
//! Every `WAVE_INTERVAL_TICKS` ticks, [`TowerDefenseState::maybe_spawn_wave`]
//! walks the enemy slot range, picks the first N free slots
//! (alive==0), and writes hp=20.0, alive=1, pos=(spawn_x, 0, 0) for
//! each. Wave size escalates: 5, 8, 12, 16, ... until WAVE_COUNT
//! waves have been spawned.
//!
//! ## Win/loss
//!
//!   - Defenders win when all WAVE_COUNT waves are spawned AND no
//!     enemy slot has alive==1 AND base HP > 0.
//!   - Defenders lose when base HP (CPU-side counter, decremented by
//!     `base_damage_taken` view delta) drops to ≤ 0.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

// ---- Slot layout constants (wired into both DSL semantics and
//      runtime initialization). The .sim's mana role-bands depend on
//      these values; keep aligned.

/// Total agent slot capacity: 1 base + 10 defenders + 100 enemy slots.
pub const TOTAL_AGENT_COUNT: u32 = 111;

/// Slot 0: the Base.
pub const BASE_SLOT: u32 = 0;
/// Defenders occupy slots [1, 11). Ten in total.
pub const DEFENDER_SLOT_START: u32 = 1;
pub const DEFENDER_COUNT: u32 = 10;
/// Enemy slots occupy [11, 111). Hundred in total.
pub const ENEMY_SLOT_START: u32 = 11;
pub const ENEMY_SLOT_COUNT: u32 = 100;

// ---- Wave spawn schedule (matches the .sim's CPU-side responsibility) --

/// Spawn a new wave every K ticks.
pub const WAVE_INTERVAL_TICKS: u64 = 50;
/// How many waves the run spawns before declaring "all waves repelled".
pub const WAVE_COUNT: u32 = 10;
/// Wave sizes (length WAVE_COUNT). Each entry is the number of new
/// enemies the wave spawns. Escalates so the final waves stress the
/// defender line. Sum = 110 — one full pass through the enemy slot
/// range, but with kills happening in between, slots free up for
/// reuse so 110 spawns fit in 100 slots provided the early waves get
/// killed before the late waves arrive (≈1 wave/50 ticks, 5 dmg/sec
/// from ~10 defenders → ~50 dmg/sec ÷ 20 hp/enemy = 2.5 kills/sec, so
/// waves up to ~10 kills can be cleared in 4 sec).
pub const WAVE_SIZES: [u32; WAVE_COUNT as usize] = [
    5, 6, 8, 10, 12, 12, 14, 14, 14, 15,
];

// ---- Initial agent SoA values (must match the .sim's role bands). ----

const BASE_HP: f32 = 1000.0;
const BASE_MANA: f32 = 1.0;
const BASE_X: f32 = 50.0;

const DEFENDER_HP: f32 = 100.0;
const DEFENDER_MANA: f32 = 2.0;
const DEFENDER_X: f32 = 40.0;

const ENEMY_HP: f32 = 20.0;
const ENEMY_MANA: f32 = 3.0;
const ENEMY_SPAWN_X: f32 = -50.0;

/// 16-byte WGSL `vec3<f32>` interop (mirrors foraging_runtime's shape).
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

/// Per-fixture state for the tower_defense simulation.
pub struct TowerDefenseState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_pos_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_hp_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmap (only one verb: Shoot, mask_0) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    defender_damage_dealt: ViewStorage,
    defender_damage_dealt_cfg_buf: wgpu::Buffer,
    base_damage_taken: ViewStorage,
    base_damage_taken_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_shoot_cfg_buf: wgpu::Buffer,
    apply_damage_cfg_buf: wgpu::Buffer,
    march_enemies_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    // -- CPU-side game state --
    /// Base HP — decremented by the per-tick delta in
    /// base_damage_taken view. The base is "destroyed" when this drops
    /// ≤ 0 (the harness's loss condition).
    pub base_hp: f32,
    /// Last observed cumulative base_damage_taken sum across all enemy
    /// slots (the BaseHit view total). Per-tick delta against this is
    /// the new damage to apply to base_hp.
    last_base_damage_total: f32,
    /// Index of the next wave to spawn (0..WAVE_COUNT). Reaches
    /// WAVE_COUNT when all waves have been spawned.
    pub next_wave_idx: u32,
    /// Tick when the next wave is allowed to spawn. Starts at 0 (the
    /// first wave fires at tick 0).
    next_wave_tick: u64,
    /// Total enemies spawned so far (sum of wave sizes that have
    /// fired). For end-of-run reporting.
    pub total_enemies_spawned: u32,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl TowerDefenseState {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_AGENT_COUNT;
        let n = agent_count as usize;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // ---- Agent SoA initial state ----
        let mut pos_padded: Vec<Vec3Padded> = vec![Vec3Padded::default(); n];
        let mut alive_init: Vec<u32> = vec![0u32; n];
        let mut hp_init: Vec<f32> = vec![0.0_f32; n];
        let mut mana_init: Vec<f32> = vec![0.0_f32; n];

        // Base — slot 0.
        pos_padded[BASE_SLOT as usize] = Vec3::new(BASE_X, 0.0, 0.0).into();
        alive_init[BASE_SLOT as usize] = 1;
        hp_init[BASE_SLOT as usize] = BASE_HP;
        mana_init[BASE_SLOT as usize] = BASE_MANA;

        // Defenders — slots 1..=10. Spread along the y-axis so the
        // shoot verb's distance check on each defender→enemy pair
        // resolves against the defender's individual position rather
        // than the whole defender line collapsing onto one point.
        for i in 0..DEFENDER_COUNT {
            let slot = (DEFENDER_SLOT_START + i) as usize;
            // Defenders span y ∈ [-9, +9].
            let y = -9.0 + (i as f32) * 2.0;
            pos_padded[slot] = Vec3::new(DEFENDER_X, y, 0.0).into();
            alive_init[slot] = 1;
            hp_init[slot] = DEFENDER_HP;
            mana_init[slot] = DEFENDER_MANA;
        }

        // Enemy slots — slots 11..=110. Initially alive=0 (no enemy
        // present). When a wave fires, the runtime walks these slots
        // and writes the spawn state. We pre-populate mana with the
        // enemy band value so when a slot is later reused the verb's
        // mana-band gate fires immediately (writing pos+hp+alive is
        // enough at spawn time).
        for i in 0..ENEMY_SLOT_COUNT {
            let slot = (ENEMY_SLOT_START + i) as usize;
            let y = -10.0 + ((i % 21) as f32) - 10.0;
            // Park the slot off-screen at the spawn x. Doesn't matter
            // because alive=0 — the per-agent rules' alive-guard
            // skips the row.
            pos_padded[slot] = Vec3::new(ENEMY_SPAWN_X - 100.0, y, 0.0).into();
            mana_init[slot] = ENEMY_MANA;
            // alive_init[slot] stays 0; hp_init[slot] stays 0.
        }

        let agent_pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::agent_pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Mask bitmap (1 bitmap, one per verb) ----
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_0_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tower_defense_runtime::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // ---- Scoring output (4 × u32 per agent) ----
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tower_defense_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // ---- Event ring + view storage ----
        let event_ring = EventRing::new(&gpu, "tower_defense_runtime");
        let defender_damage_dealt = ViewStorage::new(
            &gpu,
            "tower_defense_runtime::defender_damage_dealt",
            agent_count,
            false,
            false,
        );
        let base_damage_taken = ViewStorage::new(
            &gpu,
            "tower_defense_runtime::base_damage_taken",
            agent_count,
            false,
            false,
        );

        // ---- Per-kernel cfg uniforms ----
        let mask_cfg_init = mask_verb_Shoot::MaskVerbShootCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let chronicle_shoot_cfg_init =
            physics_verb_chronicle_Shoot::PhysicsVerbChronicleShootCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_shoot_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::chronicle_shoot_cfg"),
            contents: bytemuck::bytes_of(&chronicle_shoot_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let apply_damage_cfg_init = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let apply_damage_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::apply_damage_cfg"),
            contents: bytemuck::bytes_of(&apply_damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let march_enemies_cfg_init = physics_MarchEnemies::PhysicsMarchEnemiesCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let march_enemies_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::march_enemies_cfg"),
            contents: bytemuck::bytes_of(&march_enemies_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tower_defense_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dd_cfg_init = fold_defender_damage_dealt::FoldDefenderDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let defender_damage_dealt_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("tower_defense_runtime::defender_damage_dealt_cfg"),
                contents: bytemuck::bytes_of(&dd_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let bd_cfg_init = fold_base_damage_taken::FoldBaseDamageTakenCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let base_damage_taken_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("tower_defense_runtime::base_damage_taken_cfg"),
                contents: bytemuck::bytes_of(&bd_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_pos_buf,
            agent_alive_buf,
            agent_hp_buf,
            agent_mana_buf,
            mask_0_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            defender_damage_dealt,
            defender_damage_dealt_cfg_buf,
            base_damage_taken,
            base_damage_taken_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_shoot_cfg_buf,
            apply_damage_cfg_buf,
            march_enemies_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            base_hp: BASE_HP,
            last_base_damage_total: 0.0,
            next_wave_idx: 0,
            next_wave_tick: 0,
            total_enemies_spawned: 0,
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }

    /// Per-source defender_damage_dealt readback.
    pub fn defender_damage_dealt(&mut self) -> &[f32] {
        self.defender_damage_dealt.readback(&self.gpu)
    }

    /// Per-source base_damage_taken readback. Sum over all enemy slots
    /// gives the cumulative damage to the base.
    pub fn base_damage_taken(&mut self) -> &[f32] {
        self.base_damage_taken.readback(&self.gpu)
    }

    /// Total damage dealt to the base across all enemy slots.
    pub fn base_damage_taken_total(&mut self) -> f32 {
        self.base_damage_taken().iter().copied().sum()
    }

    /// Per-agent alive readback (1 = alive, 0 = dead/empty slot).
    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    /// Per-agent HP readback.
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    /// Count of currently-alive enemy slots (slots 11..=110 with alive==1).
    pub fn alive_enemy_count(&self) -> u32 {
        let alive = self.read_alive();
        let lo = ENEMY_SLOT_START as usize;
        let hi = (ENEMY_SLOT_START + ENEMY_SLOT_COUNT) as usize;
        alive[lo..hi].iter().filter(|&&a| a == 1).count() as u32
    }

    /// Count of currently-alive defender slots (1..=10). Useful for
    /// tracking attrition (none today — defenders are never targeted —
    /// but the harness reports it anyway).
    pub fn alive_defender_count(&self) -> u32 {
        let alive = self.read_alive();
        let lo = DEFENDER_SLOT_START as usize;
        let hi = (DEFENDER_SLOT_START + DEFENDER_COUNT) as usize;
        alive[lo..hi].iter().filter(|&&a| a == 1).count() as u32
    }

    /// CPU-side wave spawner: every WAVE_INTERVAL_TICKS ticks, find
    /// the first N free enemy slots (alive==0) and write spawn state
    /// for them. Returns the number of enemies spawned this call (0 if
    /// no wave was due, or no waves remain).
    ///
    /// Spawn writes go through small staging buffers — for this scale
    /// (up to ~15 enemies per wave) we just push each slot's update
    /// individually. A future optimisation would batch into one
    /// COPY_DST submission.
    pub fn maybe_spawn_wave(&mut self) -> u32 {
        if self.next_wave_idx >= WAVE_COUNT {
            return 0;
        }
        if self.tick < self.next_wave_tick {
            return 0;
        }
        let wave_size = WAVE_SIZES[self.next_wave_idx as usize];

        // Walk current alive flags to find free enemy slots.
        let alive = self.read_alive();
        let mut free_slots: Vec<u32> = Vec::with_capacity(wave_size as usize);
        for s in ENEMY_SLOT_START..(ENEMY_SLOT_START + ENEMY_SLOT_COUNT) {
            if alive[s as usize] == 0 {
                free_slots.push(s);
                if free_slots.len() == wave_size as usize {
                    break;
                }
            }
        }

        // If we somehow can't fit a full wave (defender line is
        // overwhelmed), spawn whatever fits — the run continues but
        // future waves will be pinched too. Acceptable for slice A.
        let actual_spawn = free_slots.len() as u32;

        // Per-slot writes: pos = (-50, y_lane, 0), hp = ENEMY_HP,
        // alive = 1. We use queue.write_buffer for each (3 writes per
        // slot — pos, alive, hp).
        for (i, &slot) in free_slots.iter().enumerate() {
            // Spread enemies in a vertical strip at the spawn line so
            // they don't all overlap.
            let y_lane = -10.0 + ((i as f32 / actual_spawn as f32) * 20.0);
            let pos = Vec3Padded::from(Vec3::new(ENEMY_SPAWN_X, y_lane, 0.0));
            let pos_offset = (slot as u64) * std::mem::size_of::<Vec3Padded>() as u64;
            self.gpu.queue.write_buffer(
                &self.agent_pos_buf, pos_offset, bytemuck::bytes_of(&pos),
            );
            let hp_offset = (slot as u64) * 4;
            self.gpu.queue.write_buffer(
                &self.agent_hp_buf, hp_offset, bytemuck::bytes_of(&ENEMY_HP),
            );
            let alive_offset = (slot as u64) * 4;
            let alive_v: u32 = 1;
            self.gpu.queue.write_buffer(
                &self.agent_alive_buf, alive_offset, bytemuck::bytes_of(&alive_v),
            );
        }

        self.next_wave_idx += 1;
        self.next_wave_tick = self.tick + WAVE_INTERVAL_TICKS;
        self.total_enemies_spawned += actual_spawn;
        actual_spawn
    }

    /// Sync CPU-side base_hp from the GPU view delta. Reads the
    /// cumulative base_damage_taken total and subtracts its delta from
    /// `base_hp`. Call AFTER `step` each tick.
    pub fn sync_base_hp(&mut self) {
        let total = self.base_damage_taken_total();
        let delta = total - self.last_base_damage_total;
        if delta > 0.0 {
            self.base_hp -= delta;
            self.last_base_damage_total = total;
        }
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("tower_defense_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tower_defense_runtime::read_f32") },
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
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("tower_defense_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tower_defense_runtime::read_u32") },
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

impl CompiledSim for TowerDefenseState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tower_defense_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // The Damaged + BaseHit emitters fire conditionally per tick;
        // clear ring headers conservatively up to agent_count*4 slots
        // per tick (matches event_count_estimate below).
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        // Clear mask + scoring output.
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf, 0, &self.mask_0_bitmap_buf, 0, mask_bytes.max(4),
        );
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — single PerPair Shoot mask.
        let mask_cfg = mask_verb_Shoot::MaskVerbShootCfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = mask_verb_Shoot::MaskVerbShootBindings {
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        // GAP WORKAROUND (compiler kernel.rs:3243): the PerPair mask
        // body emits `let mask_0_k = cfg.agent_cap;` (was `1u`), so
        // each thread visits one (agent, candidate) pair. We must
        // dispatch `agent_cap * agent_cap` threads to cover the full
        // pair grid — the duel_1v1 runtime gets away with passing
        // `agent_cap` because agent_count=2 fits in one workgroup,
        // but tower_defense's 111 slots need 12321 threads ≈ 193
        // workgroups, not 111. Pass agent_cap² as the dispatch
        // "thread count" so the host's `(N + 63) / 64` math sizes
        // the workgroup grid correctly.
        let pair_thread_count = self.agent_count.saturating_mul(self.agent_count);
        dispatch::dispatch_mask_verb_shoot(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            pair_thread_count,
        );

        // (3) Scoring — argmax per defender. The PerPair Shoot
        // candidate space is `(self_slot, target_slot)` for every
        // pair. Each defender (alive in the defender mana band)
        // ranks every alive enemy in range and emits one
        // ActionSelected with target=picked-enemy.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Shoot chronicle — gates on action_id==0u, emits Damaged.
        let shoot_cfg = physics_verb_chronicle_Shoot::PhysicsVerbChronicleShootCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_shoot_cfg_buf, 0, bytemuck::bytes_of(&shoot_cfg),
        );
        let shoot_bindings = physics_verb_chronicle_Shoot::PhysicsVerbChronicleShootBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_shoot_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_shoot(
            &mut self.cache, &shoot_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) MarchEnemies — per_agent: alive enemies step toward base
        // OR emit BaseHit + self-expire when in melee range.
        let march_cfg = physics_MarchEnemies::PhysicsMarchEnemiesCfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.march_enemies_cfg_buf, 0, bytemuck::bytes_of(&march_cfg),
        );
        let march_bindings = physics_MarchEnemies::PhysicsMarchEnemiesBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.agent_pos_buf,
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            cfg: &self.march_enemies_cfg_buf,
        };
        dispatch::dispatch_physics_marchenemies(
            &mut self.cache, &march_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) ApplyDamage — PerEvent: reads Damaged events, writes target HP.
        let event_count_estimate = self.agent_count * 4;
        let apply_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_damage_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyDamage::PhysicsApplyDamageBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_damage_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (7) seed_indirect_0 — keep indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg),
        );
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache, &seed_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (8) fold_defender_damage_dealt — RMW per Damaged event.
        let dd_cfg = fold_defender_damage_dealt::FoldDefenderDamageDealtCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.defender_damage_dealt_cfg_buf, 0, bytemuck::bytes_of(&dd_cfg),
        );
        let dd_bindings = fold_defender_damage_dealt::FoldDefenderDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.defender_damage_dealt.primary(),
            view_storage_anchor: self.defender_damage_dealt.anchor(),
            view_storage_ids: self.defender_damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.defender_damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_defender_damage_dealt(
            &mut self.cache, &dd_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (9) fold_base_damage_taken — RMW per BaseHit event.
        let bd_cfg = fold_base_damage_taken::FoldBaseDamageTakenCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.base_damage_taken_cfg_buf, 0, bytemuck::bytes_of(&bd_cfg),
        );
        let bd_bindings = fold_base_damage_taken::FoldBaseDamageTakenBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.base_damage_taken.primary(),
            view_storage_anchor: self.base_damage_taken.anchor(),
            view_storage_ids: self.base_damage_taken.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.base_damage_taken_cfg_buf,
        };
        dispatch::dispatch_fold_base_damage_taken(
            &mut self.cache, &bd_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.defender_damage_dealt.mark_dirty();
        self.base_damage_taken.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TowerDefenseState::new(seed))
}
