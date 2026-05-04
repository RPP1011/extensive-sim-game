//! Per-fixture runtime for `assets/sim/tactical_horde_500.sim` —
//! the ELEVENTH real gameplay-shaped fixture. Combines mass_battle's
//! 3-role distribution (Tank/Healer/DPS) with megaswarm_1000's
//! 1000-agent population AND megaswarm's team-mirrored verb pattern.
//!
//! Composition: 500 Red + 500 Blue (= 1000 agents). Each team:
//!   - 50 Tanks   (HP=200, level=1 Red / 4 Blue)
//!   - 50 Healers (HP=80,  level=2 Red / 5 Blue)
//!   - 400 DPS    (HP=120, level=3 Red / 6 Blue)
//!
//! Verbs: SIX, two per role × team-mirror (per the megaswarm_1000
//! pattern that sidesteps the inner-argmax-no-per-candidate-filter
//! gap). Source order is fixed:
//!
//!   verb 0 = RedStrike   (Red Tank   → Blue any role)
//!   verb 1 = BlueStrike  (Blue Tank  → Red  any role)
//!   verb 2 = RedSnipe    (Red DPS    → Blue any role)
//!   verb 3 = BlueSnipe   (Blue DPS   → Red  any role)
//!   verb 4 = RedHeal     (Red Healer → Red  any role)
//!   verb 5 = BlueHeal    (Blue Healer→ Blue any role)
//!
//! Per-tick chain (mirrors the megaswarm_1000 + mass_battle
//! shape, scaled to 6 verbs):
//!
//!   1. clear_tail + clear 6 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_RedStrike — fused PerPair kernel writes
//!      ALL 6 mask bitmaps. Dispatches `agent_cap²` threads
//!      (1M at agent_cap=1000); the kernel internally executes
//!      6 mask predicate bodies per pair = 6M predicate checks/tick.
//!   3. scoring — PerAgent argmax over 6 candidate verbs per actor;
//!      inner loop over `agent_cap` candidates per row. 6M
//!      comparisons per tick.
//!   4-9. physics_verb_chronicle_<Verb> — gates action_id==Nu.
//!  10. physics_ApplyDamage_and_ApplyHeal — fused PerEvent kernel.
//!  11. seed_indirect_0 — keeps indirect-args buffer warm.
//!  12. fold_damage_dealt + fold_healing_done — per-source f32
//!      accumulators.
//!
//! After step(), the harness MUST call sweep_dead_to_sentinel() to
//! repair the apply_damage non-atomic RMW pile-on race (proven by
//! megaswarm_1000 at the same 1000-agent scale).

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-team / per-role agent populations. 50+50+400=500 per team,
/// 1000 total. Same role split as mass_battle_100v100 scaled by 5×
/// (which kept the 1:1:8 ratio).
pub const TANKS_PER_TEAM: u32 = 50;
pub const HEALERS_PER_TEAM: u32 = 50;
pub const DPS_PER_TEAM: u32 = 400;
pub const PER_TEAM: u32 = TANKS_PER_TEAM + HEALERS_PER_TEAM + DPS_PER_TEAM;
pub const TOTAL_AGENTS: u32 = PER_TEAM * 2;

// Per-role baseline HP — same magnitudes as tactical_squad_5v5 /
// mass_battle_100v100.
pub const TANK_HP: f32 = 200.0;
pub const HEALER_HP: f32 = 80.0;
pub const DPS_HP: f32 = 120.0;

/// Encode (team, role) as the per-agent `level` slot.
///
/// team: 0 = Red, 1 = Blue. role: 0 = Tank, 1 = Healer, 2 = DPS.
/// Result: 1..=3 for Red, 4..=6 for Blue. Matches the .sim header.
pub fn level_for(team: u32, role: u32) -> u32 {
    team * 3 + role + 1
}

fn role_hp(role: u32) -> f32 {
    match role {
        0 => TANK_HP,
        1 => HEALER_HP,
        _ => DPS_HP,
    }
}

/// Per-fixture state for the tactical_horde_500.
pub struct TacticalHorde500State {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    0=RedStrike, 1=BlueStrike, 2=RedSnipe, 3=BlueSnipe,
    //    4=RedHeal,   5=BlueHeal). --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_3_bitmap_buf: wgpu::Buffer,
    mask_4_bitmap_buf: wgpu::Buffer,
    mask_5_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,
    healing_done: ViewStorage,
    healing_done_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_red_strike_cfg_buf: wgpu::Buffer,
    chronicle_blue_strike_cfg_buf: wgpu::Buffer,
    chronicle_red_snipe_cfg_buf: wgpu::Buffer,
    chronicle_blue_snipe_cfg_buf: wgpu::Buffer,
    chronicle_red_heal_cfg_buf: wgpu::Buffer,
    chronicle_blue_heal_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl TacticalHorde500State {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_AGENTS;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Layout convention:
        //   slots 0..PER_TEAM         → Red team
        //   slots PER_TEAM..2*PER_TEAM → Blue team
        // Within each team:
        //   0..TANKS_PER_TEAM           → Tank   (level=1 Red / 4 Blue)
        //   TANKS_PER_TEAM..+HEALERS    → Healer (level=2 Red / 5 Blue)
        //   +HEALERS..+DPS              → DPS    (level=3 Red / 6 Blue)
        let mut hp_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        let mut alive_init: Vec<u32> = Vec::with_capacity(agent_count as usize);
        let mut level_init: Vec<u32> = Vec::with_capacity(agent_count as usize);

        for team in 0..2u32 {
            for role in 0..3u32 {
                let n = match role {
                    0 => TANKS_PER_TEAM,
                    1 => HEALERS_PER_TEAM,
                    _ => DPS_PER_TEAM,
                };
                for _ in 0..n {
                    hp_init.push(role_hp(role));
                    alive_init.push(1);
                    level_init.push(level_for(team, role));
                }
            }
        }
        debug_assert_eq!(hp_init.len(), agent_count as usize);

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Six mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("tactical_horde_500::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("tactical_horde_500::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("tactical_horde_500::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("tactical_horde_500::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("tactical_horde_500::mask_4_bitmap");
        let mask_5_bitmap_buf = mk_mask("tactical_horde_500::mask_5_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tactical_horde_500::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let event_ring = EventRing::new(&gpu, "tactical_horde_500");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "tactical_horde_500::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "tactical_horde_500::healing_done",
            agent_count,
            false,
            false,
        );

        let mask_cfg_init = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Six chronicle cfgs (one per verb).
        let chronicle_red_strike_cfg_init =
            physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_red_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::chronicle_red_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_red_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_blue_strike_cfg_init =
            physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_blue_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::chronicle_blue_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_blue_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_red_snipe_cfg_init =
            physics_verb_chronicle_RedSnipe::PhysicsVerbChronicleRedSnipeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_red_snipe_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::chronicle_red_snipe_cfg"),
            contents: bytemuck::bytes_of(&chronicle_red_snipe_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_blue_snipe_cfg_init =
            physics_verb_chronicle_BlueSnipe::PhysicsVerbChronicleBlueSnipeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_blue_snipe_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::chronicle_blue_snipe_cfg"),
            contents: bytemuck::bytes_of(&chronicle_blue_snipe_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_red_heal_cfg_init =
            physics_verb_chronicle_RedHeal::PhysicsVerbChronicleRedHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_red_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::chronicle_red_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_red_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_blue_heal_cfg_init =
            physics_verb_chronicle_BlueHeal::PhysicsVerbChronicleBlueHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_blue_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::chronicle_blue_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_blue_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_horde_500::healing_done_cfg"),
            contents: bytemuck::bytes_of(&healing_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_level_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_3_bitmap_buf,
            mask_4_bitmap_buf,
            mask_5_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            damage_dealt,
            damage_dealt_cfg_buf,
            healing_done,
            healing_done_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_red_strike_cfg_buf,
            chronicle_blue_strike_cfg_buf,
            chronicle_red_snipe_cfg_buf,
            chronicle_blue_snipe_cfg_buf,
            chronicle_red_heal_cfg_buf,
            chronicle_blue_heal_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    pub fn healing_done(&mut self) -> &[f32] {
        self.healing_done.readback(&self.gpu)
    }

    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    pub fn read_level(&self) -> Vec<u32> {
        self.read_u32(&self.agent_level_buf, "level")
    }

    /// Scoring output buffer: per-agent (action_id, target, utility_bits, _pad)
    /// = 4 u32 per agent. Returns the raw `4 * agent_count` u32 vector.
    pub fn read_scoring_output(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tactical_horde_500::scoring_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_horde_500::read_scoring") },
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
            label: Some(&format!("tactical_horde_500::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_horde_500::read_f32") },
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
            label: Some(&format!("tactical_horde_500::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_horde_500::read_u32") },
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

    /// CPU-side workaround for the apply_damage non-atomic RMW pile-on
    /// race (proven by megaswarm_1000 at the same 1000-agent scale).
    /// The compiler-emitted apply kernel uses non-atomic RMW on
    /// agent_hp; when N>>1 strikes hit the same target in one tick,
    /// a death-branch sentinel-HP write can be clobbered by an else-
    /// branch stale-value write, leaving "dead but HP=10" agents that
    /// continue to win the next tick's argmax.
    ///
    /// Sweep restores sentinel HP=1e9 for any alive=0 agent whose HP
    /// is not already at sentinel. Single readback + small writes per
    /// tick — bandwidth-cheap at agent_count=1000.
    pub fn sweep_dead_to_sentinel(&mut self) {
        let alive = self.read_alive();
        let hp = self.read_hp();
        let mut updates: Vec<(u32, f32)> = Vec::new();
        for i in 0..self.agent_count as usize {
            if alive[i] == 0 && hp[i] < 1.0e8 {
                updates.push((i as u32, 1.0e9));
            }
        }
        if updates.is_empty() {
            return;
        }
        for (slot, sentinel) in updates {
            self.gpu.queue.write_buffer(
                &self.agent_hp_buf,
                (slot as u64) * 4,
                bytemuck::bytes_of(&sentinel),
            );
        }
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    }
}

impl CompiledSim for TacticalHorde500State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_horde_500::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Per tick: ~agent_count ActionSelected + up to ~agent_count
        // Damaged + ~agent_count Healed + small Defeated. Use
        // agent_cap*8 for headroom.
        let max_slots_per_tick = self.agent_count * 8;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [
            &self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf, &self.mask_3_bitmap_buf,
            &self.mask_4_bitmap_buf, &self.mask_5_bitmap_buf,
        ] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — fused PerPair kernel writes ALL 6 mask
        // bitmaps. Dispatches `agent_cap × agent_cap` threads (=
        // 1000×1000 = 1 000 000 at full scale); the kernel internally
        // executes 6 mask predicate bodies per pair.
        let mask_cfg = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeBindings {
            agent_alive: &self.agent_alive_buf,
            agent_level: &self.agent_level_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            mask_4_bitmap: &self.mask_4_bitmap_buf,
            mask_5_bitmap: &self.mask_5_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_redstrike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over the 6 rows. Inner loop over
        // `cfg.agent_cap` candidates per pair-field row. Emits one
        // ActionSelected per gated agent.
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
            agent_hp: &self.agent_hp_buf,
            agent_level: &self.agent_level_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            mask_4_bitmap: &self.mask_4_bitmap_buf,
            mask_5_bitmap: &self.mask_5_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4..9) Six per-verb chronicle dispatches (gates action_id).
        // Inlined per-verb (macro path interpolation rejected by
        // rustc — see commit history).
        let agent_count = self.agent_count;
        let tick_u32 = self.tick as u32;

        let cfg = physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeCfg {
            event_count: agent_count, tick: tick_u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_red_strike_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let bindings = physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_red_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_redstrike(
            &mut self.cache, &bindings, &self.gpu.device, &mut encoder, agent_count,
        );

        let cfg = physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeCfg {
            event_count: agent_count, tick: tick_u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_blue_strike_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let bindings = physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_blue_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bluestrike(
            &mut self.cache, &bindings, &self.gpu.device, &mut encoder, agent_count,
        );

        let cfg = physics_verb_chronicle_RedSnipe::PhysicsVerbChronicleRedSnipeCfg {
            event_count: agent_count, tick: tick_u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_red_snipe_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let bindings = physics_verb_chronicle_RedSnipe::PhysicsVerbChronicleRedSnipeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_red_snipe_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_redsnipe(
            &mut self.cache, &bindings, &self.gpu.device, &mut encoder, agent_count,
        );

        let cfg = physics_verb_chronicle_BlueSnipe::PhysicsVerbChronicleBlueSnipeCfg {
            event_count: agent_count, tick: tick_u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_blue_snipe_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let bindings = physics_verb_chronicle_BlueSnipe::PhysicsVerbChronicleBlueSnipeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_blue_snipe_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bluesnipe(
            &mut self.cache, &bindings, &self.gpu.device, &mut encoder, agent_count,
        );

        let cfg = physics_verb_chronicle_RedHeal::PhysicsVerbChronicleRedHealCfg {
            event_count: agent_count, tick: tick_u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_red_heal_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let bindings = physics_verb_chronicle_RedHeal::PhysicsVerbChronicleRedHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_red_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_redheal(
            &mut self.cache, &bindings, &self.gpu.device, &mut encoder, agent_count,
        );

        let cfg = physics_verb_chronicle_BlueHeal::PhysicsVerbChronicleBlueHealCfg {
            event_count: agent_count, tick: tick_u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.chronicle_blue_heal_cfg_buf, 0, bytemuck::bytes_of(&cfg));
        let bindings = physics_verb_chronicle_BlueHeal::PhysicsVerbChronicleBlueHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_blue_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_blueheal(
            &mut self.cache, &bindings, &self.gpu.device, &mut encoder, agent_count,
        );

        // (10) Apply damage + heal (fused PerEvent).
        let event_count_estimate = self.agent_count * 8;
        let apply_cfg = physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage_and_applyheal(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (11) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
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

        // (12) fold_damage_dealt — RMW per Damaged event.
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_dealt_cfg_buf, 0, bytemuck::bytes_of(&damage_cfg),
        );
        let damage_bindings = fold_damage_dealt::FoldDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_dealt.primary(),
            view_storage_anchor: self.damage_dealt.anchor(),
            view_storage_ids: self.damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_damage_dealt(
            &mut self.cache, &damage_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (13) fold_healing_done — RMW per Healed event.
        let healing_cfg = fold_healing_done::FoldHealingDoneCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.healing_done_cfg_buf, 0, bytemuck::bytes_of(&healing_cfg),
        );
        let healing_bindings = fold_healing_done::FoldHealingDoneBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.healing_done.primary(),
            view_storage_anchor: self.healing_done.anchor(),
            view_storage_ids: self.healing_done.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.healing_done_cfg_buf,
        };
        dispatch::dispatch_fold_healing_done(
            &mut self.cache, &healing_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.healing_done.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TacticalHorde500State::new(seed))
}
