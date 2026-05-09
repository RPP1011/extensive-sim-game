//! Per-fixture runtime for `assets/sim/spy_network.sim` — multi-faction
//! social-cascade fixture (Disguise + Slander + Strike).
//!
//! ## Per-tick chain
//!
//! Mirrors `duel_abilities_runtime`'s shape at N=10 with three verbs.
//! Post-rewrite (2026-05-07) every verb body dispatches via
//! `apply_ability N by self target X` instead of inlining `emit
//! ChronicleEvent { ... }`, so the per-tick chain runs the full
//! ability-registry-driven dispatcher → consumer pipeline:
//!
//!   1. Per-tick clears (event_tail, ring headers, mask bitmaps,
//!      scoring_output)
//!   2. fused_mask_verb_Disguise — PerPair, writes mask_0 (Disguise),
//!      mask_1 (Slander), mask_2 (Strike).
//!   3. scoring — PerAgent argmax over 3 rows; emits ActionSelected.
//!   4. physics_verb_chronicle_Disguise — gates action_id==0u, walks
//!      Disguise's program from the ability registry, writes
//!      EffectDisguiseApplied chronicle records (kind=67).
//!   5. physics_verb_chronicle_Slander — gates action_id==1u, walks
//!      Slander's program, writes EffectPlantBeliefApplied (kind=63).
//!   6. physics_verb_chronicle_Strike — gates action_id==2u, walks
//!      Strike's program, writes EffectDamageApplied (kind=26).
//!   7. physics_ApplyDamageFromChronicle_and_ApplySlanderFromChronicle
//!      — consumes EffectDamageApplied + EffectPlantBeliefApplied;
//!      re-emits Damaged for kind=26 and bumps target's mana for
//!      kind=63.
//!   8. physics_ApplyDamage_and_ApplyDisguiseFromChronicle — consumes
//!      Damaged (drains hp + emits Defeated on hp<=0) and
//!      EffectDisguiseApplied (writes per-agent disguise SoA).
//!   9. seed_indirect_0
//!  10. fold_damage_dealt
//!
//! ## Faction layout (10 agents)
//!
//! Slot 0..2 → spy      (creature_type=1)
//! Slot 3..5 → noble    (creature_type=2)
//! Slot 6..9 → commoner (creature_type=3)
//!
//! ## Behavioral pin
//!
//! `viz_tests::noble_dies_after_slander_cascade` — at least one noble
//! flips alive=0 within 200 ticks. Slander piles on every 10 ticks
//! (each ordered noble pair); Strike fires after 2 slanders against
//! the same target accumulate ≥100.0 mana (= suspicion).
//!
//! ## `mana` re-purpose
//!
//! The per-agent f32 `mana` SoA column is re-purposed as the slander
//! counter. Init = 0; ApplySlanderFromChronicle adds 50 per cast;
//! Strike's `when` clause gates on `target.mana >= 100`. No verb in
//! this fixture uses mana for its original combat-resource role.
//!
//! ## AbilityRegistry
//!
//! The runtime builds + uploads the spy_network ability registry once
//! at construction. The chronicle kernels read `effect_kinds` /
//! `effect_payload_a` / `effect_payload_b` etc. to walk each program's
//! effects when their gated action fires. Registry contents are pinned
//! by `binding_check::assert_ability_registry_matches_sim_constants`
//! — drift surfaces as a panic before any GPU work runs.

use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

mod binding_check;

pub use binding_check::{
    assert_ability_registry_matches_sim_constants, DISGUISE_EXPECTED_ABILITY_ID,
    SLANDER_EXPECTED_ABILITY_ID, STRIKE_EXPECTED_ABILITY_ID,
};

/// Faction discriminants — must match `assets/sim/spy_network.sim`'s
/// `config.combat.type_*` fields. The runtime seeds
/// `agent_creature_type[slot]` with these values; the .sim mask kernel
/// gates on `self.creature_type == config.combat.type_*` to drive the
/// faction-specific verb wiring.
pub const CREATURE_TYPE_SPY: u32 = 1;
pub const CREATURE_TYPE_NOBLE: u32 = 2;
pub const CREATURE_TYPE_COMMONER: u32 = 3;

/// Per-fixture state for the spy_network sim.
pub struct SpyNetworkState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    /// Re-purposed as the per-agent slander counter. Init=0; bumped by
    /// ApplySlanderFromChronicle; read by Strike's gate
    /// (`target.mana >= 100.0`) and Slander's per-candidate score.
    agent_mana_buf: wgpu::Buffer,
    /// Faction discriminant (1=spy, 2=noble, 3=commoner). Bound by the
    /// fused mask kernel and scoring; the verb gates compare against
    /// the config.combat.type_* literals.
    agent_creature_type_buf: wgpu::Buffer,
    /// Per-agent disguise SoA. Init=0; ApplyDisguiseFromChronicle
    /// writes both columns from chronicle records produced by Disguise's
    /// dispatcher arm. Subsequent observers (or runtime readback)
    /// consult these columns to substitute fake_type for true
    /// creature_type while `disguise_expires_at_tick > world.tick`.
    agent_disguise_expires_at_tick_buf: wgpu::Buffer,
    agent_disguise_fake_type_buf: wgpu::Buffer,

    // -- Stat columns the chronicle dispatchers + scoring kernel both
    // bind. Init zeros; spy_network's verbs don't scale on any of
    // these (no `+ N% stat` modifiers in the .ability files), but the
    // BGL contract demands the bindings exist.
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask bitmaps (Disguise=0, Slander=1, Strike=2) --
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
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_disguise_cfg_buf: wgpu::Buffer,
    chronicle_slander_cfg_buf: wgpu::Buffer,
    chronicle_strike_cfg_buf: wgpu::Buffer,
    apply_dmgchron_slander_cfg_buf: wgpu::Buffer,
    apply_dmg_disguise_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Packed AbilityRegistry uploaded to the GPU. The chronicle
    /// dispatcher kernels (Disguise / Slander / Strike) bind
    /// `effect_kinds` / `effect_payload_a` / `effect_payload_b` etc. to
    /// walk each program's effects. Built once at construction from the
    /// `.ability` corpus via `binding_check::build_spy_network_registry`,
    /// then uploaded to GPU storage buffers.
    registry_gpu: PackedAbilityRegistryGpu,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl SpyNetworkState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert_eq!(
            agent_count, 10,
            "spy_network expects exactly 10 agents (3 spy + 3 noble + 4 commoner); got {agent_count}",
        );

        // Wave 1 acceptance binding-check — runs once before GPU init.
        // Asserts every .ability under assets/ability_test/spy_network/
        // lowers to the slot the binding-check consts pin.
        binding_check::assert_ability_registry_matches_sim_constants();

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Build the AbilityRegistry from the .ability corpus and upload
        // it to GPU storage buffers. The chronicle dispatcher kernels
        // bind the effect_kinds + payload columns to walk each program
        // when its gated action_id fires.
        let built_registry = binding_check::build_spy_network_registry();
        let packed = PackedAbilityRegistry::pack(&built_registry.registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed,
            &gpu,
            "spy_network_runtime",
        );

        // Agent SoA — HP=50.0 (lower than 1v1's 100 so 2 strikes finish a
        // noble), alive=1, mana=0 (= zero suspicion at start). Faction
        // discriminants split by slot index.
        let n = agent_count as usize;
        let hp_init = vec![50.0_f32; n];
        let alive_init = vec![1u32; n];
        let mana_init = vec![0.0_f32; n];
        let mut creature_init = vec![0u32; n];
        // 3 spies (slots 0..2), 3 nobles (slots 3..5), 4 commoners (6..9).
        for (slot, ct) in creature_init.iter_mut().enumerate() {
            *ct = match slot {
                0..=2 => CREATURE_TYPE_SPY,
                3..=5 => CREATURE_TYPE_NOBLE,
                _ => CREATURE_TYPE_COMMONER,
            };
        }

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spy_network_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spy_network_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spy_network_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_creature_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::agent_creature_type"),
                contents: bytemuck::cast_slice(&creature_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Disguise SoA — init=0 for both columns. ApplyDisguiseFromChronicle
        // writes `expires_at_tick = world.tick + duration`, `fake_type`
        // from the chronicle payload's low byte.
        let zero_u32 = vec![0u32; n];
        let agent_disguise_expires_at_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::agent_disguise_expires_at_tick"),
                contents: bytemuck::cast_slice(&zero_u32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let agent_disguise_fake_type_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::agent_disguise_fake_type"),
                contents: bytemuck::cast_slice(&zero_u32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Stat columns the chronicle dispatchers + scoring kernel both
        // bind. spy_network's verbs don't scale on any of these (no
        // `+ N% stat` in the .ability files); init zero. The mock_max_hp
        // matters slightly for downstream eval predicates that may peek
        // at it — set to the same 50.0 the actual hp seeds, just so
        // any %max_hp-style read is sensible if a future verb adds one.
        let max_hp_init = vec![50.0_f32; n];
        let agent_max_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spy_network_runtime::agent_max_hp"),
            contents: bytemuck::cast_slice(&max_hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let zeros_f32 = vec![0.0_f32; n];
        let mk_stat = |label: &'static str| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf =
            mk_stat("spy_network_runtime::agent_attack_damage");
        let agent_armor_buf = mk_stat("spy_network_runtime::agent_armor");
        let agent_magic_resist_buf = mk_stat("spy_network_runtime::agent_magic_resist");
        let agent_move_speed_buf = mk_stat("spy_network_runtime::agent_move_speed");

        // Three mask bitmaps — one per verb, cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("spy_network_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("spy_network_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("spy_network_runtime::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 × u32 per agent, cleared per tick.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spy_network_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "spy_network_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "spy_network_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = fused_mask_verb_Disguise::FusedMaskVerbDisguiseCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spy_network_runtime::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scoring_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let chronicle_disguise_cfg_init =
            physics_verb_chronicle_Disguise::PhysicsVerbChronicleDisguiseCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let chronicle_disguise_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::chronicle_disguise_cfg"),
                contents: bytemuck::bytes_of(&chronicle_disguise_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let chronicle_slander_cfg_init =
            physics_verb_chronicle_Slander::PhysicsVerbChronicleSlanderCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let chronicle_slander_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::chronicle_slander_cfg"),
                contents: bytemuck::bytes_of(&chronicle_slander_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let chronicle_strike_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::chronicle_strike_cfg"),
                contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let apply_dmgchron_slander_cfg_init =
            physics_ApplyDamageFromChronicle_and_ApplySlanderFromChronicle::PhysicsApplyDamageFromChronicleAndApplySlanderFromChronicleCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let apply_dmgchron_slander_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::apply_dmgchron_slander_cfg"),
                contents: bytemuck::bytes_of(&apply_dmgchron_slander_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let apply_dmg_disguise_cfg_init =
            physics_ApplyDamage_and_ApplyDisguiseFromChronicle::PhysicsApplyDamageAndApplyDisguiseFromChronicleCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                agent_cap: 0,
            };
        let apply_dmg_disguise_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::apply_dmg_disguise_cfg"),
                contents: bytemuck::bytes_of(&apply_dmg_disguise_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spy_network_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let damage_dealt_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spy_network_runtime::damage_dealt_cfg"),
                contents: bytemuck::bytes_of(&damage_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_creature_type_buf,
            agent_disguise_expires_at_tick_buf,
            agent_disguise_fake_type_buf,
            agent_max_hp_buf,
            agent_attack_damage_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
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
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_disguise_cfg_buf,
            chronicle_slander_cfg_buf,
            chronicle_strike_cfg_buf,
            apply_dmgchron_slander_cfg_buf,
            apply_dmg_disguise_cfg_buf,
            seed_cfg_buf,
            registry_gpu,
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

    pub fn read_mana(&self) -> Vec<f32> {
        self.read_f32(&self.agent_mana_buf, "mana")
    }

    pub fn read_creature_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_creature_type_buf, "creature_type")
    }

    pub fn read_disguise_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_disguise_expires_at_tick_buf, "disguise_expires_at_tick")
    }

    pub fn read_disguise_fake_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_disguise_fake_type_buf, "disguise_fake_type")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("spy_network_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("spy_network_runtime::read_f32"),
            },
        );
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
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("spy_network_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("spy_network_runtime::read_u32"),
            },
        );
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

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl CompiledSim for SpyNetworkState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("spy_network_runtime::step"),
            },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, max_slots_per_tick);
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [
            &self.mask_0_bitmap_buf,
            &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf,
        ] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf,
                0,
                buf,
                0,
                mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf,
            0,
            &self.scoring_output_buf,
            0,
            scoring_output_bytes.max(16),
        );

        // (2) Mask round — fused PerPair kernel writes all 3 mask bitmaps.
        let mask_cfg = fused_mask_verb_Disguise::FusedMaskVerbDisguiseCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_Disguise::FusedMaskVerbDisguiseBindings {
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_disguise(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over 3 rows. Binds the full agent SoA +
        // ability_registry when_pred_* columns for the predicate-aware
        // gating path (the chronicle dispatcher and scoring share these
        // bindings — see `cg::lower::driver::wire_scoring_predicate_reads`).
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana: &self.agent_mana_buf,
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) Disguise chronicle — gates action_id==0u; walks Disguise's
        // program from the ability registry; emits EffectDisguiseApplied
        // (kind=67).
        let disguise_cfg =
            physics_verb_chronicle_Disguise::PhysicsVerbChronicleDisguiseCfg {
                event_count: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                agent_cap: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_disguise_cfg_buf,
            0,
            bytemuck::bytes_of(&disguise_cfg),
        );
        let disguise_bindings =
            physics_verb_chronicle_Disguise::PhysicsVerbChronicleDisguiseBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_hp: &self.agent_hp_buf,
                agent_max_hp: &self.agent_max_hp_buf,
                agent_move_speed: &self.agent_move_speed_buf,
                agent_armor: &self.agent_armor_buf,
                agent_magic_resist: &self.agent_magic_resist_buf,
                agent_attack_damage: &self.agent_attack_damage_buf,
                agent_mana: &self.agent_mana_buf,
                ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
                ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
                ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
                ability_registry_chances: &self.registry_gpu.chances,
                ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
                ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
                ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
                ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
                ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
                ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
                ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
                ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
                ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
                cfg: &self.chronicle_disguise_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_disguise(
            &mut self.cache,
            &disguise_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5) Slander chronicle — gates action_id==1u; walks Slander's
        // program; emits EffectPlantBeliefApplied (kind=63).
        let slander_cfg = physics_verb_chronicle_Slander::PhysicsVerbChronicleSlanderCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_slander_cfg_buf,
            0,
            bytemuck::bytes_of(&slander_cfg),
        );
        let slander_bindings = physics_verb_chronicle_Slander::PhysicsVerbChronicleSlanderBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana: &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_chances: &self.registry_gpu.chances,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            cfg: &self.chronicle_slander_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_slander(
            &mut self.cache,
            &slander_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6) Strike chronicle — gates action_id==2u; walks Strike's
        // program; emits EffectDamageApplied (kind=26).
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf,
            0,
            bytemuck::bytes_of(&strike_cfg),
        );
        let strike_bindings = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana: &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_chances: &self.registry_gpu.chances,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_strike(
            &mut self.cache,
            &strike_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (7) ApplyDamageFromChronicle + ApplySlanderFromChronicle
        // (fused PerEvent kernel). Drains EffectDamageApplied (kind=26)
        // → emits Damaged events; drains EffectPlantBeliefApplied
        // (kind=63) → bumps target's mana SoA.
        let event_count_estimate = self.agent_count * 4;
        let dmgchron_slander_cfg = physics_ApplyDamageFromChronicle_and_ApplySlanderFromChronicle::PhysicsApplyDamageFromChronicleAndApplySlanderFromChronicleCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_dmgchron_slander_cfg_buf,
            0,
            bytemuck::bytes_of(&dmgchron_slander_cfg),
        );
        let dmgchron_slander_bindings = physics_ApplyDamageFromChronicle_and_ApplySlanderFromChronicle::PhysicsApplyDamageFromChronicleAndApplySlanderFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_mana: &self.agent_mana_buf,
            cfg: &self.apply_dmgchron_slander_cfg_buf,
        };
        dispatch::dispatch_physics_applydamagefromchronicle_and_applyslanderfromchronicle(
            &mut self.cache,
            &dmgchron_slander_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (8) ApplyDamage + ApplyDisguiseFromChronicle (fused PerEvent
        // kernel). Drains Damaged (kind=1) → updates hp + emits
        // Defeated; drains EffectDisguiseApplied (kind=67) → unpacks
        // (duration<<8 | fake_type) and writes per-agent disguise SoA.
        let dmg_disguise_cfg = physics_ApplyDamage_and_ApplyDisguiseFromChronicle::PhysicsApplyDamageAndApplyDisguiseFromChronicleCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_dmg_disguise_cfg_buf,
            0,
            bytemuck::bytes_of(&dmg_disguise_cfg),
        );
        let dmg_disguise_bindings = physics_ApplyDamage_and_ApplyDisguiseFromChronicle::PhysicsApplyDamageAndApplyDisguiseFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_disguise_expires_at_tick: &self.agent_disguise_expires_at_tick_buf,
            agent_disguise_fake_type: &self.agent_disguise_fake_type_buf,
            cfg: &self.apply_dmg_disguise_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage_and_applydisguisefromchronicle(
            &mut self.cache,
            &dmg_disguise_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (9) seed_indirect_0 — keeps args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf,
            0,
            bytemuck::bytes_of(&seed_cfg),
        );
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (10) fold_damage_dealt — RMW per Damaged event.
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_dealt_cfg_buf,
            0,
            bytemuck::bytes_of(&damage_cfg),
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
            &mut self.cache,
            &damage_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &[]
    }

    fn snapshot(&mut self) -> AgentSnapshot {
        let hp = self.read_hp();
        let alive_raw = self.read_alive();
        let creature_types_raw = self.read_creature_type();
        let alive: Vec<u32> = alive_raw
            .iter()
            .zip(hp.iter())
            .map(|(&a, &h)| if a != 0 && h > 0.0 { 1 } else { 0 })
            .collect();
        let positions: Vec<Vec3> = (0..self.agent_count as usize)
            .map(|i| Vec3::new(i as f32 * 5.0, 0.0, 0.0))
            .collect();
        AgentSnapshot {
            positions,
            creature_types: creature_types_raw,
            alive,
        }
    }

    fn glyph_table(&self) -> Vec<VizGlyph> {
        // 1=spy, 2=noble, 3=commoner — index by creature_type.
        // Index 0 is unused (creature_type starts at 1).
        vec![
            VizGlyph::new('?', 240),  // 0 — unused
            VizGlyph::new('S', 39),   // 1 — spy
            VizGlyph::new('N', 196),  // 2 — noble
            VizGlyph::new('c', 27),   // 3 — commoner
        ]
    }

    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        Some((
            Vec3::new(-1.5, -1.5, 0.0),
            Vec3::new(((self.agent_count as f32) * 5.0) + 1.5, 1.5, 0.0),
        ))
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(SpyNetworkState::new(seed, agent_count))
}

// ---------------------------------------------------------------------------
// Behavioral pin tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// Full cascade: noble dies after enough Slanders pile suspicion onto
    /// them (mana counter crosses threshold) and Strike fires from peer
    /// nobles.
    ///
    /// Failure mode: if `disguise/slander/strike` argmax row gating ever
    /// regresses (e.g. mask kernel incorrectly clears mask_2 for nobles),
    /// the cascade stalls — no nobles strike, every slot stays alive at
    /// tick 200, and the assertion fires.
    #[test]
    fn noble_dies_after_slander_cascade() {
        let mut state = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SpyNetworkState::new(0, 10)
        })) {
            Ok(s) => s,
            Err(_) => {
                eprintln!(
                    "[noble_dies_after_slander_cascade] skipping: GPU init failed (no wgpu adapter on host?)"
                );
                return;
            }
        };

        // Step until the first noble dies, capping at 200 ticks. Lets the
        // diagnostic print show the tick the cascade tipped at.
        let mut first_noble_death_tick: Option<u64> = None;
        for _ in 0..200 {
            state.step();
            if first_noble_death_tick.is_none() {
                let alive = state.read_alive();
                let cts = state.read_creature_type();
                if (0..10).any(|s| cts[s] == CREATURE_TYPE_NOBLE && alive[s] == 0) {
                    first_noble_death_tick = Some(state.tick());
                }
            }
        }
        eprintln!(
            "[spy_network] first noble death @ tick={:?}",
            first_noble_death_tick,
        );

        let creature_types = state.read_creature_type();
        let alive = state.read_alive();
        let hp = state.read_hp();
        let mana = state.read_mana();

        eprintln!("[spy_network] post-cascade state @ tick={}", state.tick());
        for slot in 0..10 {
            eprintln!(
                "  slot {} ct={} alive={} hp={:.1} mana={:.1}",
                slot, creature_types[slot], alive[slot], hp[slot], mana[slot],
            );
        }

        // Pin 1: at least one noble (ct=2) is dead.
        let dead_nobles: Vec<usize> = (0..10)
            .filter(|&slot| creature_types[slot] == CREATURE_TYPE_NOBLE && alive[slot] == 0)
            .collect();
        assert!(
            !dead_nobles.is_empty(),
            "expected at least one noble (creature_type=2) to flip alive=0 \
             after slander cascade in 200 ticks; got slots={:?} alive={:?} \
             mana={:?}",
            dead_nobles,
            alive,
            mana,
        );

        // Pin 2: no spy or commoner died (the cascade is faction-scoped).
        for slot in 0..10 {
            if creature_types[slot] != CREATURE_TYPE_NOBLE {
                assert_eq!(
                    alive[slot], 1,
                    "non-noble slot {} (creature_type={}) should not have \
                     died — strikes only target nobles",
                    slot, creature_types[slot],
                );
            }
        }

        // Pin 3: at least one spy fired Disguise (its disguise SoA entry
        // has `expires_at_tick > 0`). Exercises the EffectDisguiseApplied
        // chronicle + ApplyDisguiseFromChronicle consumer end-to-end.
        let disguise_expires = state.read_disguise_expires_at_tick();
        let disguise_fake_type = state.read_disguise_fake_type();
        let disguised_spies: Vec<usize> = (0..10)
            .filter(|&slot| {
                creature_types[slot] == CREATURE_TYPE_SPY && disguise_expires[slot] > 0
            })
            .collect();
        assert!(
            !disguised_spies.is_empty(),
            "expected at least one spy (creature_type=1) to have \
             disguise_expires_at_tick > 0 after the cascade; got \
             expires={:?} fake_type={:?}",
            disguise_expires,
            disguise_fake_type,
        );
    }
}
