//! Per-fixture runtime for `assets/sim/tactical_squad_5v5.sim` —
//! THE FIFTH real gameplay-shaped fixture (after duel_1v1, duel_25v25,
//! foraging_real, predator_prey_real). 5 vs 5 with role distribution.
//!
//! ## Per-tick pipeline
//!
//!   1. clear_tail + clear 3 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_Strike — PerPair, writes mask_0 (Strike,
//!      Tank-only, enemy-team), mask_1 (Snipe, DPS-only, enemy-team),
//!      mask_2 (Heal, Healer-only, ally-team). Dispatched with
//!      `agent_count * agent_count` threads so every (actor, candidate)
//!      pair is visited (the verb-injected mask_k literal now reads
//!      `cfg.agent_cap` per the inline-fix in
//!      `dsl_compiler::cg::emit::kernel::mask_predicate_per_pair_body`).
//!   3. scoring — PerAgent argmax across the 3 rows; the per-pair
//!      candidate inner loop runs over all `agent_cap` candidates and
//!      picks the lowest-HP enemy (Snipe) or lowest-HP ally (Heal).
//!      Strike has constant score 1.0 — falls back to "any enemy".
//!      Emits one ActionSelected{actor, action_id, target} per gated
//!      agent.
//!   4. physics_verb_chronicle_Strike — gates on action_id==0u, emits
//!      Damaged{source, target, amount=tank_damage=10.0}.
//!   5. physics_verb_chronicle_Snipe — gates on action_id==1u, emits
//!      Damaged{source, target, amount=dps_damage=22.0}.
//!   6. physics_verb_chronicle_Heal — gates on action_id==2u, emits
//!      Healed{source, target, amount=heal_amount=18.0}.
//!   7. physics_ApplyDamage_and_ApplyHeal — fused PerEvent kernel that
//!      reads Damaged/Healed events and writes per-target HP via
//!      `agents.set_hp`. On HP<=0 sets alive=0 and emits Defeated.
//!   8. seed_indirect_0 — keeps indirect-args buffer warm.
//!   9. fold_damage_dealt — per-source f32 accumulator.
//!  10. fold_healing_done — per-source f32 accumulator.
//!
//! ## Agent layout
//!
//!   - 10 slots: Red[0..5], Blue[5..10]
//!   - Per team: 1 Tank, 1 Healer, 3 DPS
//!   - `creature_type[i]`: 0=Tank, 1=Healer, 2=Dps (per the
//!     `entity Tank/Healer/Dps : Agent` declaration order in the .sim)
//!   - `level[i]`: 0=Red, 1=Blue (team membership discriminant)
//!   - `hp[i]`: 200 for Tank, 80 for Healer, 120 for DPS

use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

// Role discriminants — match `entity Tank/Healer/Dps : Agent`
// declaration order in `assets/sim/tactical_squad_5v5.sim`.
pub const ROLE_TANK: u32 = 0;
pub const ROLE_HEALER: u32 = 1;
pub const ROLE_DPS: u32 = 2;

// Team discriminants — packed into the `level` u32 SoA slot.
pub const TEAM_RED: u32 = 0;
pub const TEAM_BLUE: u32 = 1;

// Per-role HP totals.
pub const HP_TANK: f32 = 200.0;
pub const HP_HEALER: f32 = 80.0;
pub const HP_DPS: f32 = 120.0;

/// Per-fixture state for the 5v5 tactical squad simulation.
pub struct TacticalSquad5v5State {
    gpu: GpuContext,

    // -- Agent SoA --
    /// Per-agent f32 HP. Initialised per role: Tank=200, Healer=80,
    /// DPS=120. ApplyDamage subtracts amount per Damaged; ApplyHeal
    /// adds per Healed.
    agent_hp_buf: wgpu::Buffer,
    /// Per-agent u32 alive (1 = alive, 0 = dead). All start at 1.
    /// ApplyDamage flips to 0 when HP <= 0 and emits Defeated.
    agent_alive_buf: wgpu::Buffer,
    /// Per-agent f32 mana. Not consumed today (every verb's `when`
    /// clause skips the mana check); initialised to 100.0 for parity
    /// with duel_1v1's SoA shape.
    #[allow(dead_code)]
    agent_mana_buf: wgpu::Buffer,
    /// Per-agent u32 `creature_type` discriminant (Tank=0, Healer=1,
    /// Dps=2). Mask predicates gate on `self.creature_type == <Role>`.
    agent_creature_type_buf: wgpu::Buffer,
    /// Per-agent u32 `level` SoA — repurposed as TEAM membership
    /// (Red=0, Blue=1). Mask predicates compare
    /// `target.level != self.level` (enemies) or
    /// `target.level == self.level` (allies).
    agent_level_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order: Strike=0,
    //    Snipe=1, Heal=2) --
    mask_0_bitmap_buf: wgpu::Buffer, // Strike
    mask_1_bitmap_buf: wgpu::Buffer, // Snipe
    mask_2_bitmap_buf: wgpu::Buffer, // Heal
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
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

impl TacticalSquad5v5State {
    /// Build a 5 vs 5 simulation. The agent layout is fixed:
    ///
    /// | slot | team | role   | hp  |
    /// |------|------|--------|-----|
    /// |   0  | Red  | Tank   | 200 |
    /// |   1  | Red  | Healer |  80 |
    /// |   2  | Red  | DPS    | 120 |
    /// |   3  | Red  | DPS    | 120 |
    /// |   4  | Red  | DPS    | 120 |
    /// |   5  | Blue | Tank   | 200 |
    /// |   6  | Blue | Healer |  80 |
    /// |   7  | Blue | DPS    | 120 |
    /// |   8  | Blue | DPS    | 120 |
    /// |   9  | Blue | DPS    | 120 |
    ///
    /// `agent_count` is enforced to 10 (the constant for the fixed
    /// composition); future scaling would parameterise the per-team
    /// counts.
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert_eq!(
            agent_count, 10,
            "tactical_squad_5v5 requires agent_count=10 (5 per team × 2 teams)"
        );
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Per-slot init — 5 Red (0..5), 5 Blue (5..10).
        let role_for = |slot: u32| -> u32 {
            // First slot per team is Tank, second is Healer, rest are DPS.
            let local = slot % 5;
            match local {
                0 => ROLE_TANK,
                1 => ROLE_HEALER,
                _ => ROLE_DPS,
            }
        };
        let team_for = |slot: u32| -> u32 {
            if slot < 5 { TEAM_RED } else { TEAM_BLUE }
        };
        let hp_for = |slot: u32| -> f32 {
            match role_for(slot) {
                ROLE_TANK => HP_TANK,
                ROLE_HEALER => HP_HEALER,
                _ => HP_DPS,
            }
        };

        let hp_init: Vec<f32> = (0..agent_count).map(hp_for).collect();
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let mana_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let creature_type_init: Vec<u32> = (0..agent_count).map(role_for).collect();
        let agent_creature_type_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::agent_creature_type"),
            contents: bytemuck::cast_slice(&creature_type_init),
            // COPY_SRC so `snapshot()` can stage the per-slot role
            // discriminant back to host for glyph encoding.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let level_init: Vec<u32> = (0..agent_count).map(team_for).collect();
        let agent_level_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::agent_level"),
            contents: bytemuck::cast_slice(&level_init),
            // COPY_SRC so `snapshot()` can stage the per-slot team
            // discriminant back to host for glyph encoding.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
        let mask_0_bitmap_buf = mk_mask("tactical_squad_5v5_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("tactical_squad_5v5_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("tactical_squad_5v5_runtime::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tactical_squad_5v5_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "tactical_squad_5v5_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "tactical_squad_5v5_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "tactical_squad_5v5_runtime::healing_done",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_snipe_cfg_init =
            physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_snipe_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_snipe_cfg"),
            contents: bytemuck::bytes_of(&chronicle_snipe_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_heal_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::healing_done_cfg"),
            contents: bytemuck::bytes_of(&healing_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_creature_type_buf,
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

    /// Per-source damage_dealt readback (one f32 per agent slot).
    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    /// Per-source healing_done readback (one f32 per agent slot).
    pub fn healing_done(&mut self) -> &[f32] {
        self.healing_done.readback(&self.gpu)
    }

    /// Per-agent HP readback.
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    /// Per-agent alive readback (1 = alive, 0 = dead).
    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    /// Per-agent scoring output (4 × u32 per agent: best_action,
    /// best_target, bitcast<u32>(best_utility), 0). Used by the harness
    /// to assert per-tick targeting decisions (e.g. DPS picked
    /// lowest-HP enemy; Healer picked lowest-HP ally).
    pub fn read_scoring_output(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tactical_squad_5v5_runtime::scoring_output_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_squad_5v5_runtime::read_scoring") },
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
            label: Some(&format!("tactical_squad_5v5_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_squad_5v5_runtime::read_f32") },
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
            label: Some(&format!("tactical_squad_5v5_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_squad_5v5_runtime::read_u32") },
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

impl CompiledSim for TacticalSquad5v5State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tactical_squad_5v5_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Bound at agent_count*4 — same shape as duel_1v1's
        // event_count_estimate, scaled to the higher event volume
        // (5 verbs × 10 actors / tick at peak). Without a header
        // clear, stale Damaged/Healed slots from prior ticks would
        // be re-folded by the per-tick view aggregators.
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

        // (2) Mask round — one fused PerPair kernel writes all 3 mask
        // bitmaps. The verb-injected mask_k literal now reads
        // `cfg.agent_cap` (per the inline-fix in
        // `dsl_compiler::cg::emit::kernel::mask_predicate_per_pair_body`),
        // so we MUST dispatch `agent_cap * agent_cap` threads to cover
        // every (actor, candidate) pair. Without that, only
        // `pair < agent_cap` would execute, collapsing each actor to
        // candidate=0 and breaking the team-membership predicate
        // (`target.level != self.level`) for every actor whose
        // candidate=0 fails the predicate (e.g. team-Red actors
        // scanning team-Red agent-0 → predicate false → entire
        // actor's mask bit never sets).
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
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        // PAIR DISPATCH: agent_count * agent_count = 100 threads (10 × 10).
        // The dispatch helper takes a "logical work count" and
        // computes `(count + 63) / 64` workgroups; passing
        // `agent_count * agent_count` gets us 100 threads, which round
        // up to 2 workgroups (128 threads — 28 are no-op `return`
        // under the bound check `mask_X_agent >= cfg.agent_cap`).
        dispatch::dispatch_fused_mask_verb_strike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over the 3 rows. Emits one
        // ActionSelected per gated agent. Snipe + Heal rows iterate
        // every candidate and pick lowest HP target; Strike picks
        // any-enemy with a constant score (any candidate that flipped
        // the mask bit wins by tie-break on first hit).
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
            // The score expression `if target.alive { ... } else { ... }`
            // adds `agent_alive` to the scoring kernel's binding set —
            // without it the if-branch's `target.alive` predicate has
            // nothing to read.
            agent_alive: &self.agent_alive_buf,
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

        // (4) Strike chronicle — gates on action_id==0, emits Damaged.
        let event_count_estimate = self.agent_count * 4;
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, _pad0: 0,
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
            event_count_estimate,
        );

        // (5) Snipe chronicle — gates on action_id==1, emits Damaged.
        let snipe_cfg = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, _pad0: 0,
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
            event_count_estimate,
        );

        // (6) Heal chronicle — gates on action_id==2, emits Healed.
        let heal_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, _pad0: 0,
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
            event_count_estimate,
        );

        // (7) ApplyDamage_and_ApplyHeal — fused PerEvent kernel.
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

    /// Snapshot per-agent state for the universal `viz_app` renderer.
    ///
    /// Like `duel_abilities_runtime`, this fixture has no movement
    /// physics — the .sim declares `pos: vec3` on the role entities but
    /// no kernel writes a position buffer. We therefore SYNTHESIZE a
    /// stationary 2-D layout from the slot index: Red on the left
    /// (x = -3..1), Blue on the right (x = 3..7), with role-driven y
    /// stacking (Tank at y=+1, Healer at y=0, DPS rows at y=-1, -2, -3).
    /// This keeps both teams legible in the ASCII grid even though the
    /// actual sim is purely event-driven HP/heal arithmetic.
    ///
    /// `creature_types` encoding (12 entries, indexed by
    /// `team * 6 + role * 2 + dead_bit`):
    ///
    /// |  i | team | role   | state |
    /// |----|------|--------|-------|
    /// |  0 | Red  | Tank   | alive |
    /// |  1 | Red  | Tank   | dead  |
    /// |  2 | Red  | Healer | alive |
    /// |  3 | Red  | Healer | dead  |
    /// |  4 | Red  | DPS    | alive |
    /// |  5 | Red  | DPS    | dead  |
    /// |  6 | Blue | Tank   | alive |
    /// |  7 | Blue | Tank   | dead  |
    /// |  8 | Blue | Healer | alive |
    /// |  9 | Blue | Healer | dead  |
    /// | 10 | Blue | DPS    | alive |
    /// | 11 | Blue | DPS    | dead  |
    ///
    /// Both team AND role come from REAL SoA reads
    /// (`agent_creature_type_buf`, `agent_level_buf`) rather than
    /// index-derived heuristics — slots could in principle be reordered
    /// without breaking the encoding, although the constructor today
    /// hard-codes the index→(team, role) layout. The `alive` field is
    /// read from `agent_alive_buf`; `agent_count` stays constant
    /// (no spawn/despawn) so dead slots remain in the snapshot at
    /// their original positions, just rendered with the tombstone glyph.
    ///
    /// Initial-state safe: GPU buffers are populated by
    /// `create_buffer_init` at construction, so calling `snapshot()`
    /// before any `step()` returns 10 alive slots with deterministic
    /// team+role discriminants.
    fn snapshot(&mut self) -> AgentSnapshot {
        let n = self.agent_count as usize;
        // Synthetic 2-D layout — see method doc for the per-slot map.
        // Red [0..5] occupies x = -3..1 (1-unit spacing); Blue [5..10]
        // occupies x = 3..7. Within each team, slot 0 = Tank (y=+1),
        // slot 1 = Healer (y=0), slots 2..4 = DPS (y = -1, -2, -3).
        let positions: Vec<Vec3> = (0..n)
            .map(|i| {
                let team = if i < 5 { 0u32 } else { 1u32 };
                let local = (i % 5) as i32;
                // Red on the left, Blue on the right; ~6 units apart.
                let x = if team == 0 { -3.0 } else { 3.0 };
                // Stack roles vertically: Tank=+1, Healer=0, DPS rows below.
                let y = match local {
                    0 => 1.0,                 // Tank
                    1 => 0.0,                 // Healer
                    k => -((k - 1) as f32),   // DPS rows: -1, -2, -3
                };
                Vec3::new(x, y, 0.0)
            })
            .collect();

        let role: Vec<u32> = self.read_u32(&self.agent_creature_type_buf, "creature_type_snap");
        let team: Vec<u32> = self.read_u32(&self.agent_level_buf, "level_snap");
        let alive_raw: Vec<u32> = self.read_alive();
        let hp: Vec<f32> = self.read_hp();
        // Defence-in-depth: drop slots whose HP fell to 0 even if the
        // alive bit hasn't been written yet by ApplyDamage (mirrors the
        // duel_abilities approach).
        let alive: Vec<u32> = alive_raw
            .iter()
            .zip(hp.iter())
            .map(|(&a, &h)| if a != 0 && h > 0.0 { 1 } else { 0 })
            .collect();

        // 12-entry encoding (2 teams × 3 roles × 2 alive states).
        // Clamp role to 0..3 just in case an out-of-range disc ever
        // sneaks in — keeps the index inside the glyph table.
        let creature_types: Vec<u32> = (0..n)
            .map(|i| {
                let t = (team[i] & 1) as u32; // 0 = Red, 1 = Blue
                let r = role[i].min(2) as u32; // 0=Tank, 1=Healer, 2=DPS
                let dead_bit = if alive[i] == 0 { 1u32 } else { 0u32 };
                t * 6 + r * 2 + dead_bit
            })
            .collect();

        AgentSnapshot { positions, creature_types, alive }
    }

    /// 12 glyphs matching the `snapshot.creature_types` encoding (2
    /// teams × 3 roles × 2 alive states). Layout per the table in
    /// `snapshot()`'s doc-comment.
    ///
    /// - Red team: bright red (196) for alive, fades to grey × on death.
    /// - Blue team: bright cyan (51) for alive, same grey × on death.
    /// - Per-role glyph letter: T = Tank, H = Healer, D = DPS.
    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![
            // Red team (0..6)
            VizGlyph::new('T', 196),        // 0: Red Tank alive
            VizGlyph::new('\u{00D7}', 240), // 1: Red Tank dead (grey ×)
            VizGlyph::new('H', 196),        // 2: Red Healer alive
            VizGlyph::new('\u{00D7}', 240), // 3: Red Healer dead
            VizGlyph::new('D', 196),        // 4: Red DPS alive
            VizGlyph::new('\u{00D7}', 240), // 5: Red DPS dead
            // Blue team (6..12)
            VizGlyph::new('T', 51),         // 6: Blue Tank alive (bright cyan)
            VizGlyph::new('\u{00D7}', 240), // 7: Blue Tank dead
            VizGlyph::new('H', 51),         // 8: Blue Healer alive
            VizGlyph::new('\u{00D7}', 240), // 9: Blue Healer dead
            VizGlyph::new('D', 51),         // 10: Blue DPS alive
            VizGlyph::new('\u{00D7}', 240), // 11: Blue DPS dead
        ]
    }

    /// Default viewport tight around the synthetic stationary layout
    /// from `snapshot()` — Red at x=-3, Blue at x=+3, role rows at
    /// y ∈ [-3, +1]. ±5 keeps every slot on screen with breathing room.
    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        Some((Vec3::new(-5.0, -5.0, 0.0), Vec3::new(5.0, 5.0, 0.0)))
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TacticalSquad5v5State::new(seed, agent_count))
}

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// Snapshot before any tick must report initial state: 10 slots
    /// (5 Red, 5 Blue), every slot alive, and `creature_types` reflecting
    /// the deterministic per-slot (team, role) layout from `new()`.
    /// Guards the construction-only readback path so `viz_app` can
    /// render frame 0 with content instead of a blank grid.
    #[test]
    fn snapshot_after_construction_returns_initial_state() {
        let mut state = TacticalSquad5v5State::new(0xCAFE_F00D, 10);
        let snap = state.snapshot();

        assert_eq!(snap.positions.len(), 10, "positions length");
        assert_eq!(snap.creature_types.len(), 10, "creature_types length");
        assert_eq!(snap.alive.len(), 10, "alive length");

        // No combat at tick 0 — every slot must be alive.
        let alive_total: u32 = snap.alive.iter().sum();
        assert_eq!(
            alive_total, 10,
            "every slot must be alive at construction; got alive={:?}",
            snap.alive,
        );

        // Per-slot encoding must match the constructor's hard-coded
        // index → (team, role) layout (Red[0..5] = Tank/Healer/Dps×3,
        // Blue[5..10] = Tank/Healer/Dps×3). With dead_bit=0 everywhere,
        // expected creature_types = team*6 + role*2.
        let expected: Vec<u32> = (0..10u32)
            .map(|i| {
                let team = if i < 5 { 0 } else { 1 };
                let role = match i % 5 {
                    0 => 0, // Tank
                    1 => 1, // Healer
                    _ => 2, // DPS
                };
                team * 6 + role * 2
            })
            .collect();
        assert_eq!(
            snap.creature_types, expected,
            "creature_types must reflect index→(team,role) layout from new()",
        );

        // Glyph table must be addressable for every encoded value the
        // snapshot can produce (12 entries, max index = 11).
        let glyphs = state.glyph_table();
        assert_eq!(glyphs.len(), 12, "glyph_table must have 12 entries");
        for (i, &ct) in snap.creature_types.iter().enumerate() {
            assert!(
                (ct as usize) < glyphs.len(),
                "slot {i}: creature_type {ct} out of glyph_table range",
            );
        }
    }

    /// After ticking the simulation forward, either at least one HP
    /// readback must have moved off its starting value (Strike/Snipe
    /// landing or Heal applying) or the alive count must have dropped
    /// (a kill happened). Proves the snapshot reflects live GPU state
    /// rather than cached construction-time values.
    #[test]
    fn snapshot_after_tick_reflects_state_change() {
        let mut state = TacticalSquad5v5State::new(0xCAFE_F00D, 10);
        let initial_hp = state.read_hp();
        let initial_alive_total: u32 = state.snapshot().alive.iter().sum();

        for _ in 0..100 {
            state.step();
        }

        let snap = state.snapshot();
        assert_eq!(snap.positions.len(), 10);
        assert_eq!(snap.alive.len(), 10);

        let hp_now = state.read_hp();
        let any_hp_moved = initial_hp.iter().zip(hp_now.iter()).any(|(a, b)| {
            (a - b).abs() > 0.01
        });
        let alive_total_now: u32 = snap.alive.iter().sum();
        let alive_changed = alive_total_now != initial_alive_total;

        assert!(
            any_hp_moved || alive_changed,
            "after 100 ticks, expected HP movement or kill; saw HP unchanged \
             (initial={:?}, now={:?}) and alive_total stable ({})",
            initial_hp, hp_now, alive_total_now,
        );
    }
}
