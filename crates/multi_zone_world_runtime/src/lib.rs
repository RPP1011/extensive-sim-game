//! Per-fixture runtime for `assets/sim/multi_zone_world.sim` — the
//! NINETEENTH real sim. First sim with PER-AGENT LOCATION-BASED
//! CONTEXT (zone migration).
//!
//! Each agent's per-agent `level` SoA slot encodes a discrete zone
//! id (0=Forest, 1=Town, 2=Dungeon). Each verb gates on `self.level
//! == X` so the rule cascade fires per-zone. Migration happens
//! CPU-side: each tick the runtime reads back hp/mana/alive and
//! writes back into `agent_level`.
//!
//! Per-tick chain (mirrors duel_1v1's structure with three mask
//! rows / three chronicles / one fused apply):
//!
//!   1. clear_tail + clear 3 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_GatherWood — PerPair fused mask kernel,
//!      writes mask_0 (GatherWood, level==0), mask_1 (TradeWoodFor
//!      Gold, level==1+wood), mask_2 (AttackMonster, level==2+
//!      target alive+target.level==2)
//!   3. scoring — PerAgent argmax over the 3 rows; emits one
//!      ActionSelected per gated agent
//!   4. physics_verb_chronicle_GatherWood — gates on action_id==0u,
//!      emits WoodGathered{gatherer=self, amount=1.0}
//!   5. physics_verb_chronicle_TradeWoodForGold — gates on
//!      action_id==1u, emits TradeMade{trader=self, ...}
//!   6. physics_verb_chronicle_AttackMonster — gates on action_id==2u,
//!      emits Damaged{source=self, target, amount=12.0}
//!   7. physics_ApplyGather_and_ApplyTrade_and_ApplyDamage — fused
//!      PerEvent kernel that reads each event kind and writes per-
//!      target hp/mana via agents.set_hp / agents.set_mana / agents.
//!      set_alive
//!   8. seed_indirect_0 — keeps indirect-args buffer warm
//!   9. fold_wood_gathered_total — per-source f32 accumulator
//!  10. fold_gold_earned_total — per-source f32 accumulator
//!  11. fold_damage_dealt — per-source f32 accumulator
//!  12. CPU migration pass: read hp/mana/level → compute new zones →
//!      queue.write_buffer agent_level

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

pub const ADVENTURER_COUNT: u32 = 30;
pub const MONSTER_COUNT: u32 = 5;
pub const TOTAL_AGENTS: u32 = ADVENTURER_COUNT + MONSTER_COUNT;

/// Zone discriminants — must match the literals in
/// `assets/sim/multi_zone_world.sim` (`self.level == N`).
pub const ZONE_FOREST: u32 = 0;
pub const ZONE_TOWN: u32 = 1;
pub const ZONE_DUNGEON: u32 = 2;

/// Migration thresholds — kept CPU-side because the verb DSL doesn't
/// expose a per-tick `self.level := X` write surface today. (The
/// migration arithmetic itself is trivial; surfacing it inside the
/// DSL is a follow-up to Slice A — see GAP comment in lib.rs.)
const MIGRATE_FOREST_TO_TOWN_WOOD: f32 = 20.0;
const MIGRATE_TOWN_TO_DUNGEON_GOLD: f32 = 30.0;
/// Hp threshold below which a Dungeon adventurer flees to Forest.
/// Combined with the per-monster-kill mass-retreat rule, dungeon
/// stays don't last forever.
const MIGRATE_DUNGEON_TO_FOREST_HP: f32 = 10.0;

const MONSTER_INITIAL_HP: f32 = 100.0;
/// Adventurer's initial HP (== gold) at start. Also the value to
/// reset to after a successful dungeon run (so the gold loop reflects
/// "spent all gold on the fight, recovered hp").
const ADVENTURER_INITIAL_HP: f32 = 0.0;

/// Per-fixture state for the multi-zone world.
pub struct MultiZoneWorldState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// Per-agent f32 HP. For Adventurers, hp doubles as **gold**
    /// (cumulative, +5 per successful trade, set back down on dungeon
    /// migration). For Monsters, hp is real HP (decremented by
    /// AttackMonster damage; CPU respawn on death).
    agent_hp_buf: wgpu::Buffer,
    /// Per-agent u32 alive (1 = alive, 0 = dead). All agents start
    /// alive. ApplyDamage writes 0 on hp<=0; CPU respawn handler
    /// resets it.
    agent_alive_buf: wgpu::Buffer,
    /// Per-agent f32 mana. For Adventurers, mana = **wood inventory**
    /// (+1 per gather, -10 per trade). For Monsters, unused.
    agent_mana_buf: wgpu::Buffer,
    /// Per-agent u32 level. **Repurposed as ZONE id** for this sim
    /// (0=Forest, 1=Town, 2=Dungeon). Adventurers all start in Forest.
    /// Monsters always at Dungeon (never migrate).
    agent_level_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order: GatherWood=0,
    //    TradeWoodForGold=1, AttackMonster=2) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    wood_gathered_total: ViewStorage,
    wood_gathered_total_cfg_buf: wgpu::Buffer,
    gold_earned_total: ViewStorage,
    gold_earned_total_cfg_buf: wgpu::Buffer,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_gather_cfg_buf: wgpu::Buffer,
    chronicle_trade_cfg_buf: wgpu::Buffer,
    chronicle_strike_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,

    // CPU-side mirror of agent_level for migration tracking. Updated
    // each tick after readback; reflects the *new* zones written back
    // to GPU. Stored here so the harness can report per-zone counts
    // without extra readbacks.
    pub level_mirror: Vec<u32>,
    // CPU-side mirror of agent_hp/agent_mana. Maintained alongside the
    // GPU buffers because the apply-physics rules are removed (see
    // sidestep in multi_zone_world.sim) — the runtime computes hp/mana
    // updates by replaying the per-tick view deltas, then writes the
    // mirror back to the GPU agent SoA.
    hp_mirror: Vec<f32>,
    mana_mirror: Vec<f32>,
    // Last-tick view totals so we can compute per-tick deltas.
    last_wood_total: Vec<f32>,
    last_gold_total: Vec<f32>,
    // Last-tick damage_dealt totals (per agent) so we can compute
    // per-tick dungeon-attack deltas for CPU monster-damage resolution.
    last_dmg_total: Vec<f32>,

    // CPU-side counters. Bumped by the migration pass + monster
    // respawn loop.
    pub kill_count: u64,
    pub forest_to_town_migrations: u64,
    pub town_to_dungeon_migrations: u64,
    pub dungeon_to_forest_migrations: u64,
    pub monster_respawns: u64,
}

impl MultiZoneWorldState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert!(
            agent_count >= TOTAL_AGENTS,
            "multi_zone_world expects at least {TOTAL_AGENTS} agents \
             ({ADVENTURER_COUNT} adventurers + {MONSTER_COUNT} monsters); \
             got {agent_count}",
        );

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");
        let n = agent_count as usize;

        // hp init: adventurers start at gold=0; monsters start at HP=100.
        let mut hp_init = vec![ADVENTURER_INITIAL_HP; n];
        for i in (ADVENTURER_COUNT as usize)..(TOTAL_AGENTS as usize).min(n) {
            hp_init[i] = MONSTER_INITIAL_HP;
        }
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // alive init: all 1.
        let alive_init: Vec<u32> = vec![1u32; n];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // mana (wood) init: all 0.
        let mana_init: Vec<f32> = vec![0.0_f32; n];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // level (zone) init: adventurers in Forest (0); monsters in
        // Dungeon (2).
        let mut level_init = vec![ZONE_FOREST; n];
        for i in (ADVENTURER_COUNT as usize)..(TOTAL_AGENTS as usize).min(n) {
            level_init[i] = ZONE_DUNGEON;
        }
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Three mask bitmaps + a zero source.
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
        let mask_0_bitmap_buf = mk_mask("multi_zone_world_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("multi_zone_world_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("multi_zone_world_runtime::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent. Cleared per tick.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi_zone_world_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi_zone_world_runtime::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Event ring + view storage (3 views, all per-agent f32, no decay).
        let event_ring = EventRing::new(&gpu, "multi_zone_world_runtime");
        let wood_gathered_total = ViewStorage::new(
            &gpu,
            "multi_zone_world_runtime::wood_gathered_total",
            agent_count,
            false,
            false,
        );
        let gold_earned_total = ViewStorage::new(
            &gpu,
            "multi_zone_world_runtime::gold_earned_total",
            agent_count,
            false,
            false,
        );
        let damage_dealt = ViewStorage::new(
            &gpu,
            "multi_zone_world_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = fused_mask_verb_GatherWood::FusedMaskVerbGatherWoodCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_gather_cfg_init =
            physics_verb_chronicle_GatherWood::PhysicsVerbChronicleGatherWoodCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_gather_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::chronicle_gather_cfg"),
            contents: bytemuck::bytes_of(&chronicle_gather_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_trade_cfg_init =
            physics_verb_chronicle_TradeWoodForGold::PhysicsVerbChronicleTradeWoodForGoldCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_trade_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::chronicle_trade_cfg"),
            contents: bytemuck::bytes_of(&chronicle_trade_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_DungeonStrike::PhysicsVerbChronicleDungeonStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi_zone_world_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mk_view_cfg = |label: &str| -> wgpu::Buffer {
            let cfg = fold_wood_gathered_total::FoldWoodGatheredTotalCfg {
                event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let wood_gathered_total_cfg_buf = mk_view_cfg("multi_zone_world_runtime::wood_gathered_total_cfg");
        let gold_earned_total_cfg_buf = mk_view_cfg("multi_zone_world_runtime::gold_earned_total_cfg");
        let damage_dealt_cfg_buf = mk_view_cfg("multi_zone_world_runtime::damage_dealt_cfg");

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_level_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            wood_gathered_total,
            wood_gathered_total_cfg_buf,
            gold_earned_total,
            gold_earned_total_cfg_buf,
            damage_dealt,
            damage_dealt_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_gather_cfg_buf,
            chronicle_trade_cfg_buf,
            chronicle_strike_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
            level_mirror: level_init,
            hp_mirror: hp_init,
            mana_mirror: mana_init,
            last_wood_total: vec![0.0; n],
            last_gold_total: vec![0.0; n],
            last_dmg_total: vec![0.0; n],
            kill_count: 0,
            forest_to_town_migrations: 0,
            town_to_dungeon_migrations: 0,
            dungeon_to_forest_migrations: 0,
            monster_respawns: 0,
        }
    }

    pub fn wood_gathered_total(&mut self) -> &[f32] {
        self.wood_gathered_total.readback(&self.gpu)
    }
    pub fn gold_earned_total(&mut self) -> &[f32] {
        self.gold_earned_total.readback(&self.gpu)
    }
    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }
    pub fn read_mana(&self) -> Vec<f32> {
        self.read_f32(&self.agent_mana_buf, "mana")
    }
    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }
    pub fn read_level(&self) -> Vec<u32> {
        self.read_u32(&self.agent_level_buf, "level")
    }

    /// Per-agent scoring output (4 × u32 per agent: best_action,
    /// best_target, bitcast<u32>(best_utility), 0).
    pub fn read_scoring_output(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi_zone_world_runtime::scoring_output_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("multi_zone_world_runtime::read_scoring") },
        );
        encoder.copy_buffer_to_buffer(&self.scoring_output_buf, 0, &staging, 0, bytes);
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

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("multi_zone_world_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("multi_zone_world_runtime::read_f32") },
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
            label: Some(&format!("multi_zone_world_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("multi_zone_world_runtime::read_u32") },
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

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }

    /// Per-zone counts over Adventurer slots (0..30). Indexed
    /// [forest, town, dungeon].
    pub fn zone_counts(&self) -> [u32; 3] {
        let mut c = [0u32; 3];
        for slot in 0..(ADVENTURER_COUNT as usize) {
            let z = self.level_mirror.get(slot).copied().unwrap_or(0) as usize;
            if z < 3 {
                c[z] += 1;
            }
        }
        c
    }

    /// CPU-side migration pass + apply-physics replacement.
    ///
    /// The GPU side runs the verb cascade (mask → scoring →
    /// chronicle) which produces WoodGathered / TradeMade /
    /// DungeonAttacked events folded into per-source views. The
    /// apply-physics that would normally write hp/mana on the GPU
    /// is removed (see sidestep in multi_zone_world.sim) — instead
    /// the CPU replays the per-tick view deltas:
    ///
    ///   - `wood_total[i]` jump → mana_mirror[i] += delta
    ///   - `gold_total[i]` jump → mana_mirror[i] -= delta * 2,
    ///                            hp_mirror[i]   += delta
    ///                            (each gold = 5, costs 10 wood)
    ///   - dungeon_attacker contributes one swing → CPU subtracts
    ///                            attack_damage from a chosen
    ///                            monster's hp.
    ///
    /// Then the migration rules run, the mirrors get pushed back to
    /// the GPU SoA, and dead monsters get respawned.
    fn cpu_migration_step(&mut self) {
        let n = self.agent_count as usize;

        // Snapshot view totals (cumulative since tick 0).
        let wood_total = self.wood_gathered_total().to_vec();
        let gold_total = self.gold_earned_total().to_vec();
        let dmg_total = self.damage_dealt().to_vec();

        // Per-tick deltas + apply to mirrors.
        for i in 0..n.min(wood_total.len()) {
            let dw = wood_total[i] - self.last_wood_total[i];
            if dw > 0.0 {
                self.mana_mirror[i] += dw;
            }
        }
        for i in 0..n.min(gold_total.len()) {
            let dg = gold_total[i] - self.last_gold_total[i];
            if dg > 0.0 {
                // Each gold piece earned = 5 (one trade). Each trade
                // costs 10 wood. So `dg / 5 * 10 = dg * 2` wood
                // burned, `dg` gold gained.
                let trades = dg / 5.0;
                self.mana_mirror[i] -= trades * 10.0;
                self.hp_mirror[i] += dg;
                if self.mana_mirror[i] < 0.0 {
                    self.mana_mirror[i] = 0.0;
                }
            }
        }

        // Dungeon combat resolution: each adventurer-attacker delta
        // contributes to monster damage. Pick the first alive monster
        // and apply damage to its hp_mirror. Monster death triggers
        // adventurer retreat below.
        let mut monster_hp_changed = false;
        for i in 0..n.min(dmg_total.len()) {
            // Only count attackers that are adventurers (slot < 30).
            // Monsters also fire DungeonStrike (their level=2) but
            // we don't want monsters damaging themselves; skip them.
            if i >= ADVENTURER_COUNT as usize {
                continue;
            }
            let dd = dmg_total[i] - self.last_dmg_total[i];
            if dd > 0.0 {
                // Find first alive monster and deal damage to it.
                for m in (ADVENTURER_COUNT as usize)..(TOTAL_AGENTS as usize) {
                    if self.hp_mirror[m] > 0.0 {
                        self.hp_mirror[m] -= dd;
                        monster_hp_changed = true;
                        break;
                    }
                }
            }
        }

        self.last_wood_total = wood_total;
        self.last_gold_total = gold_total;
        self.last_dmg_total = dmg_total;

        let mut new_levels = self.level_mirror.clone();
        let alive_dirty = false;
        let new_alive = vec![1u32; n];
        // Read back current alive (in case GPU side did something —
        // it shouldn't here, but for safety).
        // Initialise from existing GPU state? We've never modified
        // alive on GPU (no Damaged events apply). So all alive=1
        // until CPU sets dead.

        // Adventurer migration based on mirrored hp/mana.
        for slot in 0..(ADVENTURER_COUNT as usize) {
            let cur_zone = new_levels[slot];
            let cur_hp = self.hp_mirror[slot];
            let cur_mana = self.mana_mirror[slot];
            match cur_zone {
                ZONE_FOREST => {
                    if cur_mana >= MIGRATE_FOREST_TO_TOWN_WOOD {
                        new_levels[slot] = ZONE_TOWN;
                        self.forest_to_town_migrations += 1;
                    }
                }
                ZONE_TOWN => {
                    if cur_mana < 10.0 {
                        if cur_hp >= MIGRATE_TOWN_TO_DUNGEON_GOLD {
                            new_levels[slot] = ZONE_DUNGEON;
                            self.town_to_dungeon_migrations += 1;
                        } else {
                            // Out of wood, not enough gold for the
                            // dungeon — head back to Forest to gather
                            // more.
                            new_levels[slot] = ZONE_FOREST;
                            // (intentionally not counted as a
                            // migration metric — the round-trip
                            // foresters dominate; metrics focus on
                            // forward progress.)
                        }
                    }
                }
                ZONE_DUNGEON => {
                    if cur_hp < MIGRATE_DUNGEON_TO_FOREST_HP {
                        new_levels[slot] = ZONE_FOREST;
                        self.dungeon_to_forest_migrations += 1;
                    }
                }
                _ => {}
            }
        }

        // Monster respawn: when a monster's hp_mirror dropped to <=0,
        // award the kill, retreat dungeon adventurers (with -50 gold
        // cost), and respawn the monster. ALSO mark the monster dead
        // briefly so alive=0 propagates (then back to 1 next tick
        // when respawn fires).
        let mut any_monster_died = false;
        if monster_hp_changed {
            for m in (ADVENTURER_COUNT as usize)..(TOTAL_AGENTS as usize) {
                if self.hp_mirror[m] <= 0.0 {
                    self.hp_mirror[m] = MONSTER_INITIAL_HP;
                    self.monster_respawns += 1;
                    self.kill_count += 1;
                    any_monster_died = true;
                }
            }
        }
        if any_monster_died {
            for slot in 0..(ADVENTURER_COUNT as usize) {
                if new_levels[slot] == ZONE_DUNGEON {
                    new_levels[slot] = ZONE_FOREST;
                    self.dungeon_to_forest_migrations += 1;
                    self.hp_mirror[slot] =
                        (self.hp_mirror[slot] - MIGRATE_TOWN_TO_DUNGEON_GOLD / 2.0).max(0.0);
                }
            }
        }

        // Push mirrors → GPU. Always write hp+mana back since the
        // GPU side never wrote them this tick (the apply rules are
        // removed) — but the next tick's verb predicates need to see
        // the updated mana for the trade gate.
        self.gpu.queue.write_buffer(
            &self.agent_mana_buf, 0, bytemuck::cast_slice(&self.mana_mirror),
        );
        self.gpu.queue.write_buffer(
            &self.agent_hp_buf, 0, bytemuck::cast_slice(&self.hp_mirror),
        );
        if new_levels != self.level_mirror {
            self.gpu.queue.write_buffer(
                &self.agent_level_buf, 0, bytemuck::cast_slice(&new_levels),
            );
            self.level_mirror = new_levels;
        }
        if alive_dirty {
            self.gpu.queue.write_buffer(
                &self.agent_alive_buf, 0, bytemuck::cast_slice(&new_alive),
            );
        }
    }
}

impl CompiledSim for MultiZoneWorldState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("multi_zone_world_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [&self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf, &self.mask_2_bitmap_buf] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round.
        let mask_cfg = fused_mask_verb_GatherWood::FusedMaskVerbGatherWoodCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_GatherWood::FusedMaskVerbGatherWoodBindings {
            agent_alive: &self.agent_alive_buf,
            agent_level: &self.agent_level_buf,
            agent_mana: &self.agent_mana_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_gatherwood(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (3) Scoring.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) GatherWood chronicle (action_id == 0): emits
        // WoodGathered{gatherer=actor, amount}. The wood/mana write
        // back into agent_mana happens CPU-side after step (sidestep
        // documented in multi_zone_world.sim — the schedule fuses
        // chronicles + apply otherwise, breaking visibility).
        let cg = physics_verb_chronicle_GatherWood::PhysicsVerbChronicleGatherWoodCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_gather_cfg_buf, 0, bytemuck::bytes_of(&cg));
        let cb = physics_verb_chronicle_GatherWood::PhysicsVerbChronicleGatherWoodBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_gather_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_gatherwood(
            &mut self.cache, &cb, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (5) TradeWoodForGold chronicle (action_id == 1).
        let cg = physics_verb_chronicle_TradeWoodForGold::PhysicsVerbChronicleTradeWoodForGoldCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_trade_cfg_buf, 0, bytemuck::bytes_of(&cg));
        let cb = physics_verb_chronicle_TradeWoodForGold::PhysicsVerbChronicleTradeWoodForGoldBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_trade_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_tradewoodforgold(
            &mut self.cache, &cb, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (6) DungeonStrike chronicle (action_id == 2).
        let cg = physics_verb_chronicle_DungeonStrike::PhysicsVerbChronicleDungeonStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&cg));
        let cb = physics_verb_chronicle_DungeonStrike::PhysicsVerbChronicleDungeonStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_dungeonstrike(
            &mut self.cache, &cb, &self.gpu.device, &mut encoder, self.agent_count,
        );

        let event_count_estimate = self.agent_count * 4;

        // (8) seed_indirect_0 — keep indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
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

        // (9) fold_wood_gathered_total.
        let view_cfg = fold_wood_gathered_total::FoldWoodGatheredTotalCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.wood_gathered_total_cfg_buf, 0, bytemuck::bytes_of(&view_cfg));
        let view_bindings = fold_wood_gathered_total::FoldWoodGatheredTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.wood_gathered_total.primary(),
            view_storage_anchor: self.wood_gathered_total.anchor(),
            view_storage_ids: self.wood_gathered_total.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.wood_gathered_total_cfg_buf,
        };
        dispatch::dispatch_fold_wood_gathered_total(
            &mut self.cache, &view_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (10) fold_gold_earned_total.
        let view_cfg = fold_gold_earned_total::FoldGoldEarnedTotalCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.gold_earned_total_cfg_buf, 0, bytemuck::bytes_of(&view_cfg));
        let view_bindings = fold_gold_earned_total::FoldGoldEarnedTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.gold_earned_total.primary(),
            view_storage_anchor: self.gold_earned_total.anchor(),
            view_storage_ids: self.gold_earned_total.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.gold_earned_total_cfg_buf,
        };
        dispatch::dispatch_fold_gold_earned_total(
            &mut self.cache, &view_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (11) fold_damage_dealt.
        let view_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.damage_dealt_cfg_buf, 0, bytemuck::bytes_of(&view_cfg));
        let view_bindings = fold_damage_dealt::FoldDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_dealt.primary(),
            view_storage_anchor: self.damage_dealt.anchor(),
            view_storage_ids: self.damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_damage_dealt(
            &mut self.cache, &view_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.wood_gathered_total.mark_dirty();
        self.gold_earned_total.mark_dirty();
        self.damage_dealt.mark_dirty();

        // (12) CPU-side migration pass + monster respawn.
        self.cpu_migration_step();

        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(MultiZoneWorldState::new(seed, agent_count))
}
