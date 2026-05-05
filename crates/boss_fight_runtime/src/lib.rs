//! Per-fixture runtime for `assets/sim/boss_fight.sim` — first
//! ASYMMETRIC RPG combat fixture. 1 Boss + 5 Heroes in a shared Agent
//! SoA, distinguished via the `creature_type` discriminant.
//!
//! Mirrors `duel_1v1_runtime` shape with these deltas:
//!   - AGENT_COUNT = 6 (slot 0 = Boss, slots 1..=5 = Heroes)
//!   - Per-slot HP init: Boss=5000, Heroes=200
//!   - Adds `agent_creature_type` SoA buffer (u32 per slot;
//!     init: [0, 1, 1, 1, 1, 1] — Boss=0, Hero=1 in entity decl order)
//!   - 4 verbs (BossStrike, BossSelfHeal, HeroAttack, HeroHeal)
//!     producing 4 mask bitmaps + 4 chronicle physics rules
//!
//! The init pattern relies on the entity-declaration order in
//! `boss_fight.sim`: `entity Boss : Agent { ... }` first, then
//! `entity Hero : Agent { ... }`. The DSL compiler assigns
//! `EntityRef.0` in declaration order — see expr.rs:644-664 — and
//! the runtime mirrors those discriminants when seeding the
//! `agent_creature_type` SoA.

use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the boss fight.
pub struct BossFightState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    /// Creature-type discriminant per slot. Boss=0 (slot 0),
    /// Hero=1 (slots 1..=5). Read by mask predicates AND scoring
    /// rows that test `target.creature_type == ...`.
    agent_creature_type_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    BossStrike=0, BossSelfHeal=1, HeroAttack=2, HeroHeal=3) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_3_bitmap_buf: wgpu::Buffer,
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
    chronicle_boss_strike_cfg_buf: wgpu::Buffer,
    chronicle_boss_heal_cfg_buf: wgpu::Buffer,
    chronicle_hero_attack_cfg_buf: wgpu::Buffer,
    chronicle_hero_heal_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl BossFightState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert!(
            agent_count >= 6,
            "boss_fight requires at least 6 agents (1 boss + 5 heroes); got {agent_count}",
        );
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Per-slot HP init: slot 0 = boss (5000), slots 1..=5 = heroes (200).
        let mut hp_init: Vec<f32> = vec![200.0_f32; agent_count as usize];
        hp_init[0] = 5000.0;
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // creature_type: slot 0 = Boss (discriminant 0), 1..=5 = Hero
        // (discriminant 1). Slots beyond 5 (if agent_count > 6) keep
        // discriminant 1 — they'd be additional heroes structurally
        // but the harness only initialises HP/alive for the canonical
        // 6 slots.
        let mut creature_type_init: Vec<u32> = vec![1u32; agent_count as usize];
        creature_type_init[0] = 0; // Boss
        let agent_creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("boss_fight_runtime::agent_creature_type"),
                contents: bytemuck::cast_slice(&creature_type_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Four mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("boss_fight_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("boss_fight_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("boss_fight_runtime::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("boss_fight_runtime::mask_3_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boss_fight_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "boss_fight_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "boss_fight_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "boss_fight_runtime::healing_done",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = fused_mask_verb_BossStrike::FusedMaskVerbBossStrikeCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::mask_cfg"),
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
            label: Some("boss_fight_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_boss_strike_cfg_init =
            physics_verb_chronicle_BossStrike::PhysicsVerbChronicleBossStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_boss_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::chronicle_boss_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_boss_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_boss_heal_cfg_init =
            physics_verb_chronicle_BossSelfHeal::PhysicsVerbChronicleBossSelfHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_boss_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::chronicle_boss_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_boss_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_hero_attack_cfg_init =
            physics_verb_chronicle_HeroAttack::PhysicsVerbChronicleHeroAttackCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_hero_attack_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::chronicle_hero_attack_cfg"),
            contents: bytemuck::bytes_of(&chronicle_hero_attack_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_hero_heal_cfg_init =
            physics_verb_chronicle_HeroHeal::PhysicsVerbChronicleHeroHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_hero_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::chronicle_hero_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_hero_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::healing_done_cfg"),
            contents: bytemuck::bytes_of(&healing_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_creature_type_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_3_bitmap_buf,
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
            chronicle_boss_strike_cfg_buf,
            chronicle_boss_heal_cfg_buf,
            chronicle_hero_attack_cfg_buf,
            chronicle_hero_heal_cfg_buf,
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

    pub fn read_creature_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_creature_type_buf, "creature_type")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("boss_fight_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("boss_fight_runtime::read_f32") },
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
            label: Some(&format!("boss_fight_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("boss_fight_runtime::read_u32") },
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

impl CompiledSim for BossFightState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("boss_fight_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // boss_fight emits up to 1 ActionSelected + 1 verb-emit (Damaged
        // or Healed) per agent per tick, plus chronicle Defeated emits.
        // 4*agent_count is the same conservative upper bound duel uses.
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [
            &self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf, &self.mask_3_bitmap_buf,
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

        // (2) Mask round — fused PerPair kernel writes all 4 mask
        // bitmaps. Predicates are pure self-checks (no `target.*`
        // reads in `when`) so the mask_k=1u hardcoding doesn't matter
        // — every actor's predicate evaluates the same regardless of
        // candidate slot.
        let mask_cfg = fused_mask_verb_BossStrike::FusedMaskVerbBossStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_BossStrike::FusedMaskVerbBossStrikeBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_bossstrike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (3) Scoring — argmax over 4 rows. Each row's score expression
        // iterates per-pair candidates (because the verb's score reads
        // `target.alive` and `target.creature_type`); the row whose
        // `if (target ok) { 1.0 } else { -1000.0 }` returns 1.0 with
        // the lowest-slot candidate wins argmax for that row.
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
            agent_alive: &self.agent_alive_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) BossStrike chronicle — gates on action_id==0, emits Damaged.
        let bs_cfg = physics_verb_chronicle_BossStrike::PhysicsVerbChronicleBossStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_boss_strike_cfg_buf, 0, bytemuck::bytes_of(&bs_cfg),
        );
        let bs_bindings = physics_verb_chronicle_BossStrike::PhysicsVerbChronicleBossStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_boss_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bossstrike(
            &mut self.cache, &bs_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) BossSelfHeal chronicle — gates on action_id==1, emits Healed.
        let bh_cfg = physics_verb_chronicle_BossSelfHeal::PhysicsVerbChronicleBossSelfHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_boss_heal_cfg_buf, 0, bytemuck::bytes_of(&bh_cfg),
        );
        let bh_bindings = physics_verb_chronicle_BossSelfHeal::PhysicsVerbChronicleBossSelfHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_boss_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bossselfheal(
            &mut self.cache, &bh_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) HeroAttack chronicle — gates on action_id==2, emits Damaged.
        let ha_cfg = physics_verb_chronicle_HeroAttack::PhysicsVerbChronicleHeroAttackCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_hero_attack_cfg_buf, 0, bytemuck::bytes_of(&ha_cfg),
        );
        let ha_bindings = physics_verb_chronicle_HeroAttack::PhysicsVerbChronicleHeroAttackBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_hero_attack_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_heroattack(
            &mut self.cache, &ha_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) HeroHeal chronicle — gates on action_id==3, emits Healed.
        let hh_cfg = physics_verb_chronicle_HeroHeal::PhysicsVerbChronicleHeroHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_hero_heal_cfg_buf, 0, bytemuck::bytes_of(&hh_cfg),
        );
        let hh_bindings = physics_verb_chronicle_HeroHeal::PhysicsVerbChronicleHeroHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_hero_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_heroheal(
            &mut self.cache, &hh_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (8) ApplyDamage_and_ApplyHeal — fused PerEvent kernel.
        let event_count_estimate = self.agent_count * 4;
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

        // (9) seed_indirect_0.
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

        // (10) fold_damage_dealt.
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

        // (11) fold_healing_done.
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
    /// `boss_fight.sim` declares `pos: vec3` on both Boss and Hero
    /// entities, but the runtime never allocates an `agent_pos_buf`
    /// (combat is purely event-driven HP arithmetic — no kernel reads
    /// or writes positions). Following `mass_battle_100v100_runtime`'s
    /// approach for sims without a real position buffer, we SYNTHESISE
    /// a deterministic 2-D layout: the boss anchors the centre and
    /// heroes ring around it. This gives the renderer an asymmetric
    /// arena that visually reflects the 1-vs-N topology.
    ///
    /// Layout:
    /// - Slot 0 (Boss):   origin (0, 0, 0)
    /// - Slots 1..N (Heroes): evenly-spaced ring at radius 3.0,
    ///   angle = 2π·(slot-1) / (agent_count-1)
    ///
    /// `creature_types` encoding (4 entries, indexed by
    /// `boss_bit << 1 | dead_bit`):
    ///
    /// |  i | role   | state |
    /// |----|--------|-------|
    /// |  0 | minion | alive |
    /// |  1 | minion | dead  |
    /// |  2 | boss   | alive |
    /// |  3 | boss   | dead  |
    ///
    /// The `boss_bit` is derived from a REAL SoA read of
    /// `agent_creature_type_buf`: the .sim's entity-declaration order
    /// pins Boss=0 / Hero=1, so `boss_bit = (creature_type == 0)`.
    /// The `alive` field is read from `agent_alive_buf`; `agent_count`
    /// stays constant (no spawn/despawn) so dead slots remain in the
    /// snapshot at their original ring positions, just rendered with
    /// the tombstone glyph. HP defence-in-depth zeros the alive bit if
    /// hp <= 0 even when the alive buffer hasn't been flipped yet
    /// (mirrors duel_25v25_runtime + tactical_squad_5v5_runtime).
    ///
    /// Initial-state safe: GPU buffers are populated by
    /// `create_buffer_init` at construction, so calling `snapshot()`
    /// before any `step()` returns `agent_count` alive slots with the
    /// boss at index 0 and the rest as minions.
    fn snapshot(&mut self) -> AgentSnapshot {
        let n = self.agent_count as usize;

        // Synthetic 2-D layout: boss at origin, heroes ringed around.
        let positions: Vec<Vec3> = (0..n)
            .map(|i| {
                if i == 0 {
                    // Boss anchors the centre.
                    Vec3::new(0.0, 0.0, 0.0)
                } else {
                    // Hero ring at radius 3.0 in the (x,y) plane. The
                    // renderer projects on x/y; z stays 0.
                    let ring_n = (n - 1).max(1) as f32;
                    let theta =
                        std::f32::consts::TAU * ((i - 1) as f32) / ring_n;
                    Vec3::new(3.0 * theta.cos(), 3.0 * theta.sin(), 0.0)
                }
            })
            .collect();

        let ctype: Vec<u32> = self.read_creature_type();
        let alive_raw: Vec<u32> = self.read_alive();
        let hp: Vec<f32> = self.read_hp();
        // Defence-in-depth: treat hp<=0 as dead even if the alive bit
        // hasn't been written yet by ApplyDamage.
        let alive: Vec<u32> = alive_raw
            .iter()
            .zip(hp.iter())
            .map(|(&a, &h)| if a != 0 && h > 0.0 { 1 } else { 0 })
            .collect();

        // Encode: Boss (creature_type==0) → boss_bit=1; Hero
        // (creature_type==1) → boss_bit=0. Encoded value is
        // `boss_bit << 1 | dead_bit`.
        let creature_types: Vec<u32> = (0..n)
            .map(|i| {
                let boss_bit = if ctype[i] == 0 { 1u32 } else { 0u32 };
                let dead_bit = if alive[i] == 0 { 1u32 } else { 0u32 };
                (boss_bit << 1) | dead_bit
            })
            .collect();

        AgentSnapshot { positions, creature_types, alive }
    }

    /// 4 glyphs matching the `snapshot.creature_types` encoding:
    ///
    /// - `m` in bright cyan (51) for alive minion (Hero)
    /// - tombstone × in grey (240) for dead minion
    /// - `B` in bright red (196) for alive Boss
    /// - tombstone × in grey (240) for dead Boss
    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![
            VizGlyph::new('m', 51),         // 0: minion alive (cyan)
            VizGlyph::new('\u{00D7}', 240), // 1: minion dead (grey ×)
            VizGlyph::new('B', 196),        // 2: boss alive (red)
            VizGlyph::new('\u{00D7}', 240), // 3: boss dead (grey ×)
        ]
    }

    /// Default viewport tight around the synthetic layout from
    /// `snapshot()`: boss at origin, heroes ringed at radius 3.0 in
    /// the (x,y) plane. ±4 keeps the full ring on screen with breathing
    /// room. Positions are stationary today (no kernel writes a `pos`
    /// buffer), so this framing stays valid for the full encounter.
    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        Some((Vec3::new(-4.0, -4.0, 0.0), Vec3::new(4.0, 4.0, 0.0)))
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(BossFightState::new(seed, agent_count))
}

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// Snapshot before any tick must report the asymmetric initial
    /// state: exactly 1 Boss (slot 0) and N-1 minions, every slot
    /// alive. Guards the construction-only readback path so `viz_app`
    /// can render frame 0 with content instead of a blank arena.
    #[test]
    fn snapshot_after_construction_returns_initial_state() {
        const N: u32 = 6;
        let mut state = BossFightState::new(0xCAFE_F00D, N);
        let snap = state.snapshot();

        assert_eq!(snap.positions.len(), N as usize, "positions length");
        assert_eq!(
            snap.creature_types.len(),
            N as usize,
            "creature_types length",
        );
        assert_eq!(snap.alive.len(), N as usize, "alive length");

        // Construction state — every slot must be alive.
        let alive_total: u32 = snap.alive.iter().sum();
        assert_eq!(
            alive_total, N,
            "every slot must be alive at construction; got {}",
            alive_total,
        );

        // Asymmetric topology check: exactly 1 boss + N-1 minions.
        // Encoded value 2 = boss alive, 0 = minion alive.
        let boss_count = snap
            .creature_types
            .iter()
            .filter(|&&ct| ct == 2)
            .count();
        let minion_count = snap
            .creature_types
            .iter()
            .filter(|&&ct| ct == 0)
            .count();
        assert_eq!(boss_count, 1, "must have exactly 1 boss; got {boss_count}");
        assert_eq!(
            minion_count,
            (N - 1) as usize,
            "must have exactly N-1 minions; got {minion_count}",
        );
        // Slot 0 must specifically be the boss (creature_type=Boss=0
        // discriminant pinned by .sim entity-declaration order).
        assert_eq!(
            snap.creature_types[0], 2,
            "slot 0 must be the boss (encoded 2)",
        );

        // Glyph table must be addressable for every encoded value the
        // snapshot can produce (4 entries, max index = 3).
        let glyphs = state.glyph_table();
        assert_eq!(glyphs.len(), 4, "glyph_table must have 4 entries");
        for (i, &ct) in snap.creature_types.iter().enumerate() {
            assert!(
                (ct as usize) < glyphs.len(),
                "slot {i}: creature_type {ct} out of glyph_table range",
            );
        }

        // Synthetic positions: boss at origin, heroes on ring radius 3.
        let (vmin, vmax) = state.default_viewport().expect("viewport");
        assert_eq!(snap.positions[0], Vec3::new(0.0, 0.0, 0.0));
        for (i, p) in snap.positions.iter().enumerate().skip(1) {
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!(
                (r - 3.0).abs() < 0.001,
                "slot {i} hero must lie on radius-3.0 ring; got {p:?} (r={r})",
            );
            assert!(
                p.x >= vmin.x - 0.001
                    && p.x <= vmax.x + 0.001
                    && p.y >= vmin.y - 0.001
                    && p.y <= vmax.y + 0.001,
                "slot {i} pos {p:?} outside default viewport [{vmin:?}, {vmax:?}]",
            );
        }
    }

    /// After ticking the simulation forward, either at least one HP
    /// readback must have moved off its starting value (a Damaged or
    /// Healed event landing) or the alive count must have dropped (a
    /// Defeated event firing). Proves the snapshot reflects live GPU
    /// state rather than cached construction-time values.
    #[test]
    fn snapshot_after_tick_reflects_state_change() {
        const N: u32 = 6;
        let mut state = BossFightState::new(0xCAFE_F00D, N);
        let initial_hp = state.read_hp();
        let initial_alive_total: u32 = state.snapshot().alive.iter().sum();

        // 30 ticks: HeroAttack fires every 3 ticks (so ~10 attacks per
        // hero against the boss), well above the noise floor.
        for _ in 0..30 {
            state.step();
        }

        let snap = state.snapshot();
        assert_eq!(snap.positions.len(), N as usize);
        assert_eq!(snap.alive.len(), N as usize);

        let hp_now = state.read_hp();
        let any_hp_moved = initial_hp
            .iter()
            .zip(hp_now.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        let alive_total_now: u32 = snap.alive.iter().sum();
        let alive_changed = alive_total_now != initial_alive_total;

        assert!(
            any_hp_moved || alive_changed,
            "after 30 ticks, expected HP movement or kill; saw HP unchanged \
             and alive_total stable ({})",
            alive_total_now,
        );
    }
}
