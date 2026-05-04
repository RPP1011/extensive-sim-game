//! Per-fixture runtime for `assets/sim/megaswarm_1000.sim` —
//! the TENTH real gameplay-shaped fixture and the second SCALE-UP
//! for pair-field scoring (1000 agents → 1M mask cells per tick).
//!
//! Composition: 500 Red (level=1) + 500 Blue (level=2) on a single
//! Agent SoA. TWO team-mirrored verbs (RedStrike + BlueStrike) so
//! the per-actor argmax can score-out same-team targets via the
//! `if (target.level == <enemy>) ... else { -1e9 }` pattern. The
//! single-verb shape from the original task brief degenerates at
//! 1000 agents because the inner argmax has no per-candidate
//! predicate filter and the score expr can't see `self.*` — both
//! teams end up targeting the same lowest-HP slot.
//!
//! Per-tick chain (manual sequencing, mirrors mass_battle_100v100
//! shape — the compiler-emitted SCHEDULE table is a planner hint,
//! not a hard constraint):
//!
//!   1. clear_tail + clear both mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_RedStrike — PerPair, dispatches
//!      `agent_cap²` threads (1 000 000 at agent_cap=1000), writes
//!      both mask_0 (RedStrike) + mask_1 (BlueStrike).
//!   3. scoring — PerAgent argmax over 2 candidate verbs per actor;
//!      inner loop over `agent_cap` candidates per pair-field row.
//!      Emits one ActionSelected{actor, action_id, target} per
//!      gated agent.
//!   4. physics_verb_chronicle_RedStrike — gates action_id==0u,
//!      emits Damaged (Red attacks Blue).
//!   5. physics_verb_chronicle_BlueStrike — gates action_id==1u,
//!      emits Damaged (Blue attacks Red).
//!   6. physics_ApplyDamage — PerEvent kernel. Reads Damaged events,
//!      writes per-target HP; on HP<=0 also flips alive=0 + emits
//!      Defeated.
//!   7. seed_indirect_0 — keeps indirect-args buffer warm
//!   8. fold_damage_dealt — per-source f32 accumulator

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-team agent populations. 500 + 500 = 1000.
pub const PER_TEAM: u32 = 500;
pub const TOTAL_AGENTS: u32 = PER_TEAM * 2;

/// All combatants spawn with HP=100. Strike does 10 dmg, so each
/// kill takes 10 strikes worth of pile-on (the deterministic
/// argmax piles every same-team striker on the same lowest-HP
/// enemy).
pub const COMBATANT_HP: f32 = 100.0;

/// Encode team as the per-agent `level` slot.
/// Red = 1u, Blue = 2u. Matches the encoding documented in the .sim.
fn level_for_team(team: u32) -> u32 {
    team + 1
}

/// Per-fixture state for the megaswarm.
pub struct Megaswarm1000State {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_level_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    RedStrike=0, BlueStrike=1) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_red_strike_cfg_buf: wgpu::Buffer,
    chronicle_blue_strike_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl Megaswarm1000State {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_AGENTS;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Per-agent SoA inits. Layout:
        //   slots 0..PER_TEAM         → Red team   (level=1)
        //   slots PER_TEAM..2*PER_TEAM → Blue team (level=2)
        let mut hp_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        let mut alive_init: Vec<u32> = Vec::with_capacity(agent_count as usize);
        let mut level_init: Vec<u32> = Vec::with_capacity(agent_count as usize);

        for team in 0..2u32 {
            for _ in 0..PER_TEAM {
                hp_init.push(COMBATANT_HP);
                alive_init.push(1);
                level_init.push(level_for_team(team));
            }
        }
        debug_assert_eq!(hp_init.len(), agent_count as usize);

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Two mask bitmaps (RedStrike + BlueStrike). Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("megaswarm_1000::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("megaswarm_1000::mask_1_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("megaswarm_1000::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let event_ring = EventRing::new(&gpu, "megaswarm_1000");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "megaswarm_1000::damage_dealt",
            agent_count,
            false,
            false,
        );

        let mask_cfg_init = fused_mask_verb_RedStrike::FusedMaskVerbRedStrikeCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_red_strike_cfg_init =
            physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_red_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::chronicle_red_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_red_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_blue_strike_cfg_init =
            physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_blue_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::chronicle_blue_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_blue_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megaswarm_1000::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_level_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            damage_dealt,
            damage_dealt_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_red_strike_cfg_buf,
            chronicle_blue_strike_cfg_buf,
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
            label: Some("megaswarm_1000::scoring_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_1000::read_scoring") },
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
            label: Some(&format!("megaswarm_1000::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_1000::read_f32") },
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
            label: Some(&format!("megaswarm_1000::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_1000::read_u32") },
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

    /// CPU-side sweep that resets HP=sentinel for any agent whose
    /// alive=0 but whose HP is NOT already at sentinel. Required to
    /// sidestep a deeper-layer gap: the compiler-emitted apply
    /// kernel uses non-atomic read-modify-write on agent_hp, so when
    /// many concurrent threads target the same agent the death
    /// branch's sentinel-HP write can be overwritten by an else-
    /// branch's stale-value write. The result is "dead but HP=10"
    /// agents that continue to win the next tick's argmax, freezing
    /// the simulation. See header for full discussion.
    ///
    /// Runs each tick after step(). A single readback + small write
    /// per tick — bandwidth-cheap at agent_count=1000.
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
        // Coalesce contiguous slot ranges into one write per range.
        // For sparse death sets (~few dozen agents) this is ~1
        // queue write per dead agent; well within budget.
        for (slot, sentinel) in updates {
            self.gpu.queue.write_buffer(
                &self.agent_hp_buf,
                (slot as u64) * 4,
                bytemuck::bytes_of(&sentinel),
            );
        }
        // Force the writes to flush before the next step's scoring
        // kernel reads agent_hp.
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    }
}

impl CompiledSim for Megaswarm1000State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("megaswarm_1000::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Per tick: ~agent_count ActionSelected + ~agent_count Damaged
        // + up to ~agent_count Defeated. Use agent_cap*8 for headroom.
        let max_slots_per_tick = self.agent_count * 8;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [&self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — fused PerPair kernel writes BOTH mask
        // bitmaps (RedStrike → mask_0, BlueStrike → mask_1).
        // Dispatches `agent_cap × agent_cap` threads
        // (= 1000×1000 = 1 000 000 at full scale).
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
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_redstrike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over the single row. Inner loop over
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
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) RedStrike chronicle — gates action_id==0u, emits Damaged
        // (Red attacks Blue).
        let red_strike_cfg = physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_red_strike_cfg_buf, 0, bytemuck::bytes_of(&red_strike_cfg),
        );
        let red_strike_bindings = physics_verb_chronicle_RedStrike::PhysicsVerbChronicleRedStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_red_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_redstrike(
            &mut self.cache, &red_strike_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4b) BlueStrike chronicle — gates action_id==1u, emits Damaged
        // (Blue attacks Red).
        let blue_strike_cfg = physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_blue_strike_cfg_buf, 0, bytemuck::bytes_of(&blue_strike_cfg),
        );
        let blue_strike_bindings = physics_verb_chronicle_BlueStrike::PhysicsVerbChronicleBlueStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_blue_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bluestrike(
            &mut self.cache, &blue_strike_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) Apply damage (PerEvent).
        let event_count_estimate = self.agent_count * 8;
        let apply_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyDamage::PhysicsApplyDamageBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (6) seed_indirect_0 — keeps indirect-args buffer warm.
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

        // (7) fold_damage_dealt — RMW per Damaged event.
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

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    // agent_count is fixed by the per-team layout (500+500=1000).
    Box::new(Megaswarm1000State::new(seed))
}
