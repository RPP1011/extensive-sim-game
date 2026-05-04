//! Per-fixture runtime for `assets/sim/mass_battle_100v100.sim` —
//! the SIXTH real gameplay-shaped fixture and the first SCALE-UP for
//! pair-field scoring (200 agents, agent_cap × agent_cap mask grid).
//!
//! Composition: 10 Tanks + 10 Healers + 80 DPS per team × 2 teams =
//! 200 agents. Role + team encoded into the per-agent `level` u32
//! (1=Red Tank, 2=Red Healer, 3=Red DPS, 4=Blue Tank, 5=Blue Healer,
//! 6=Blue DPS).
//!
//! Per-tick chain mirrors the duel_1v1 cascade with one extra
//! verb (Snipe):
//!
//!   1. clear_tail + clear 3 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_Strike — PerPair, dispatches `agent_cap²`
//!      threads (40 000 at agent_cap=200), writes mask_0 (Strike,
//!      Tank-vs-enemy), mask_1 (Snipe, DPS-vs-enemy), mask_2 (Heal,
//!      Healer-vs-ally)
//!   3. scoring — PerAgent argmax over 3 candidate verbs per actor;
//!      inner loop over `agent_cap` candidates per pair-field row.
//!      Emits one ActionSelected{actor, action_id, target} per
//!      gated agent.
//!   4. physics_verb_chronicle_Strike — gates action_id==0u, emits
//!      Damaged{source, target, amount=18.0}
//!   5. physics_verb_chronicle_Snipe — gates action_id==1u, emits
//!      Damaged{source, target, amount=14.0}
//!   6. physics_verb_chronicle_Heal — gates action_id==2u, emits
//!      Healed{source, target, amount=22.0}
//!   7. physics_ApplyDamage_and_ApplyHeal — fused PerEvent kernel.
//!      Reads Damaged/Healed events, writes per-target HP via
//!      agents.set_hp; on HP<=0 also flips alive=0 + emits Defeated.
//!   8. seed_indirect_0 — keeps indirect-args buffer warm
//!   9. fold_damage_dealt — per-source f32 accumulator
//!  10. fold_healing_done — per-source f32 accumulator
//!
//! The compiler change shipped with this fixture (mask body alias
//! `mask_<ID>_k = cfg.agent_cap`) means the runtime MUST dispatch
//! `agent_cap × agent_cap` threads for the mask kernel; otherwise
//! the per-pair grid only covers a slice of the (actor, candidate)
//! space and verbs that gate on `target.*` will silently see a
//! zero-bit mask. See `step()` below.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-team / per-role agent populations.
pub const TANKS_PER_TEAM: u32 = 10;
pub const HEALERS_PER_TEAM: u32 = 10;
pub const DPS_PER_TEAM: u32 = 80;
pub const PER_TEAM: u32 = TANKS_PER_TEAM + HEALERS_PER_TEAM + DPS_PER_TEAM;
pub const TOTAL_AGENTS: u32 = PER_TEAM * 2;

// Per-role baseline HP (matches the "stats" section of the task
// brief — Tank 200, Healer 80, DPS 120). Initial HP at spawn equals
// MaxHp; the chronicle never increases HP past the SoA buffer
// initialiser (no clamp lowering today), so the runtime
// over-provisions MaxHp via initial_hp == max_hp and post-clamps on
// readback when the harness wants a "live" hp display.
pub const TANK_HP: f32 = 200.0;
pub const HEALER_HP: f32 = 80.0;
pub const DPS_HP: f32 = 120.0;

/// Encode (team, role) as the per-agent `level` slot.
///
/// team: 0 = Red, 1 = Blue. role: 0 = Tank, 1 = Healer, 2 = DPS.
fn level_for(team: u32, role: u32) -> u32 {
    // 1..=3 for Red, 4..=6 for Blue. Matches the encoding documented
    // in `assets/sim/mass_battle_100v100.sim`.
    team * 3 + role + 1
}

fn role_hp(role: u32) -> f32 {
    match role {
        0 => TANK_HP,
        1 => HEALER_HP,
        _ => DPS_HP,
    }
}

/// Per-fixture state for the mass-battle.
pub struct MassBattle100v100State {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    Strike=0, Snipe=1, Heal=2) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
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
    chronicle_strike_cfg_buf: wgpu::Buffer,
    chronicle_snipe_cfg_buf: wgpu::Buffer,
    chronicle_heal_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl MassBattle100v100State {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_AGENTS;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Build per-agent SoA inits. Layout convention:
        //   slots 0..PER_TEAM         → Red team
        //   slots PER_TEAM..2*PER_TEAM → Blue team
        // Within each team, the role layout is
        //   0..TANKS_PER_TEAM           → Tank
        //   TANKS_PER_TEAM..+HEALERS    → Healer
        //   +HEALERS..+DPS              → DPS
        let mut hp_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        let mut alive_init: Vec<u32> = Vec::with_capacity(agent_count as usize);
        let mut level_init: Vec<u32> = Vec::with_capacity(agent_count as usize);

        for team in 0..2u32 {
            // Place teams 50 units apart on +X / -X. Within a team
            // arrange roles in shallow rings so the spatial layout
            // is interpretable in the trace (front-line tanks,
            // back-line healers, mid-line DPS). The exact positions
            // don't drive any predicate today (perception_radius is
            // fixed at 999.0 in the sim) but they make the metric
            // stream debuggable.
            let team_x = if team == 0 { -50.0 } else { 50.0 };
            for role in 0..3u32 {
                let n = match role {
                    0 => TANKS_PER_TEAM,
                    1 => HEALERS_PER_TEAM,
                    _ => DPS_PER_TEAM,
                };
                let role_y = match role {
                    0 => 0.0_f32, // Tanks at front
                    1 => 8.0,     // Healers behind
                    _ => 4.0,     // DPS mid
                };
                let _ = (team_x, role_y);
                for _ in 0..n {
                    hp_init.push(role_hp(role));
                    alive_init.push(1);
                    level_init.push(level_for(team, role));
                }
            }
        }
        debug_assert_eq!(hp_init.len(), agent_count as usize);

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // Three mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("mass_battle_100v100::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("mass_battle_100v100::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("mass_battle_100v100::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mass_battle_100v100::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let event_ring = EventRing::new(&gpu, "mass_battle_100v100");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "mass_battle_100v100::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "mass_battle_100v100::healing_done",
            agent_count,
            false,
            false,
        );

        let mask_cfg_init = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_snipe_cfg_init =
            physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_snipe_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_snipe_cfg"),
            contents: bytemuck::bytes_of(&chronicle_snipe_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_heal_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::healing_done_cfg"),
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
            chronicle_strike_cfg_buf,
            chronicle_snipe_cfg_buf,
            chronicle_heal_cfg_buf,
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

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("mass_battle_100v100::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("mass_battle_100v100::read_f32") },
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
            label: Some(&format!("mass_battle_100v100::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("mass_battle_100v100::read_u32") },
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
}

impl CompiledSim for MassBattle100v100State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("mass_battle_100v100::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Each producer can emit at most one event per actor per tick;
        // 4 producers (ActionSelected, Damaged from Strike, Damaged
        // from Snipe, Healed from Heal) × agent_cap upper bound
        // covers the worst case. Use agent_cap*8 for headroom (covers
        // Defeated emissions too).
        let max_slots_per_tick = self.agent_count * 8;
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

        // (2) Mask round — fused PerPair kernel writes all 3 mask
        // bitmaps. Dispatches `agent_cap × agent_cap` threads (=
        // 200×200 = 40 000 at full scale), one per (actor, candidate)
        // pair. The compiler change to `mask_<ID>_k = cfg.agent_cap`
        // shipped with this fixture means every cand in 0..agent_cap
        // is checked per actor.
        let mask_cfg = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_Strike::FusedMaskVerbStrikeBindings {
            agent_alive: &self.agent_alive_buf,
            agent_level: &self.agent_level_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        // Dispatch agent_cap × agent_cap threads. The `agent_cap`
        // parameter to `dispatch_*` is interpreted as the total
        // thread count to round up against the workgroup_x.
        dispatch::dispatch_fused_mask_verb_strike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over the 3 rows. Inner loop over
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

        // (4) Strike chronicle — gates action_id==0u, emits Damaged
        // (Tank attacks).
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&strike_cfg),
        );
        let strike_bindings = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_strike(
            &mut self.cache, &strike_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) Snipe chronicle — gates action_id==1u (DPS attacks).
        let snipe_cfg = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_snipe_cfg_buf, 0, bytemuck::bytes_of(&snipe_cfg),
        );
        let snipe_bindings = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_snipe_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_snipe(
            &mut self.cache, &snipe_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) Heal chronicle — gates action_id==2u (Healer heals).
        let heal_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_heal_cfg_buf, 0, bytemuck::bytes_of(&heal_cfg),
        );
        let heal_bindings = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_heal(
            &mut self.cache, &heal_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) Apply damage + heal (fused PerEvent).
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

        // (8) seed_indirect_0 — keeps indirect-args buffer warm.
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

        // (9) fold_damage_dealt — RMW per Damaged event.
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

        // (10) fold_healing_done — RMW per Healed event.
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
    // agent_count is fixed by the per-team / per-role layout.
    Box::new(MassBattle100v100State::new(seed))
}
