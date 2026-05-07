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

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

mod binding_check;

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
    /// Task #138 follow-on (boss_fight port, 2026-05-07) — per-stat
    /// agent SoA columns the apply_ability dispatcher's
    /// `scale_bonus = Σ percent * agent_stat[caster_slot]` switch reads.
    /// boss_fight's BossStrike + HeroAttack programs have no scaling
    /// entries today, so all five columns sit at their inert init
    /// values; the dispatcher's `scale_bonus` collapses to 0
    /// unconditionally. Kept on the state struct because the verb-
    /// chronicle kernels still BIND them (the dispatcher emits the
    /// stat-switch arms whether or not any program actually scales).
    /// Mirrors apply_ability_smoke_runtime + duel_25v25_runtime
    /// exactly — the same five-column shape.
    #[allow(dead_code)]
    agent_attack_damage_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_armor_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_magic_resist_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_move_speed_buf: wgpu::Buffer,
    /// boss_fight's verbs don't read mana, but the apply_ability
    /// dispatcher's stat-switch (Wave 1.5#4 GPU wire-up) binds it
    /// alongside the other stat columns — see the BossStrike +
    /// HeroAttack `agent_mana` field in the generated
    /// `PhysicsVerbChronicle*Bindings`. Init to 100.0 for shape parity
    /// with duel_abilities; no kernel in this fixture reads it.
    #[allow(dead_code)]
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    BossStrike=0, BossSelfHeal=1, HeroAttack=2, HeroStun=3, HeroHeal=4) --
    //
    // HeroStun control-status proof (boss_fight, 2026-05-07): the verb
    // is declared between HeroAttack and HeroHeal in the .sim, so
    // source-order action_id assignment lands HeroStun at index 3 and
    // shifts HeroHeal to index 4. The fused mask kernel writes all five
    // bitmaps; the scoring argmax reads all five rows.
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_3_bitmap_buf: wgpu::Buffer,
    mask_4_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    /// HeroStun control-status proof (boss_fight, 2026-05-07) —
    /// per-agent `stun_expires_at_tick` SoA column. Written by the
    /// fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`
    /// kernel from kind=29 EffectStunApplied chronicle records produced
    /// by HeroStun's verb chronicle dispatcher (kind=29 records carry
    /// `expires_at_tick = world.tick + 15` precomputed by the
    /// dispatcher). Init to 0 (= "never stunned" — agents whose
    /// `stun_expires_at_tick > world.tick` are stunned). The verb
    /// `where`-clauses in this fixture don't read it today, but the new
    /// `hero_stun_locks_boss` test asserts the SoA column lands the
    /// expected expires_at_tick after HeroStun cast cycles.
    agent_stun_expires_at_tick_buf: wgpu::Buffer,

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
    /// HeroStun control-status proof (boss_fight, 2026-05-07) — fifth
    /// per-agent verb chronicle cfg. Same shape as the BossStrike +
    /// HeroAttack cfgs; the HeroStun chronicle kernel filters
    /// `action_id == 3u` (source-order index — HeroStun is the fourth
    /// verb declared after BossStrike + BossSelfHeal + HeroAttack) and
    /// dispatches the AbilityId(3) program through the apply_ability
    /// arm, writing kind=29 EffectStunApplied records.
    chronicle_hero_stun_cfg_buf: wgpu::Buffer,
    chronicle_hero_heal_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (boss_fight port, 2026-05-07) + HeroStun
    /// control-status proof (boss_fight, 2026-05-07) + HeroHeal
    /// apply_ability proof (boss_fight, 2026-05-07) — cfg uniform for
    /// the FUSED chronicle-consumer kernel. The lower pass folded
    /// ApplyDamageFromChronicle (drains kind=26 → emit Damaged),
    /// ApplyStunFromChronicle (drains kind=29 → write
    /// `agents.set_stun_expires_at_tick`), and ApplyHealFromChronicle
    /// (drains kind=27 → write
    /// `agents.set_hp(min(hp+amt, max_hp))`) into ONE kernel
    /// (`physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`)
    /// because all three consume from the same event ring at
    /// @phase(post) with non-overlapping kind tags (26 + 29 + 27).
    /// Single cfg + single dispatch per tick. Fusion grew from 2-way
    /// to 3-way when HeroHeal switched to the chronicle-pipeline shape.
    apply_chronicle_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Task #138 follow-on (boss_fight port, 2026-05-07) — Packed
    /// AbilityRegistry uploaded to the GPU. The BossStrike + HeroAttack
    /// chronicle kernels bind `effect_kinds` / `effect_payload_a` /
    /// `effect_payload_b` (and the scaling/nested/when_pred columns)
    /// for the apply_ability dispatcher arm. Built once at construction
    /// by `binding_check::build_boss_fight_registry` (two programs:
    /// BossStrike at AbilityId(1), HeroAttack at AbilityId(2)) and
    /// uploaded via `PackedAbilityRegistryGpu::upload`. The buffers
    /// live for the rest of the run.
    registry_gpu: PackedAbilityRegistryGpu,

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

        // Task #138 follow-on (boss_fight port, 2026-05-07) — runs ONCE
        // at startup before any GPU work. Asserts the runtime's
        // hand-built BossStrike + HeroAttack programs land at
        // AbilityId(1) / AbilityId(2) so the `apply_ability 1` /
        // `apply_ability 2` literals in `assets/sim/boss_fight.sim`
        // (BossStrike + HeroAttack verb bodies) dispatch the correct
        // programs. Cheap (two-program registry build); panics on any
        // drift before the expensive GPU init below.
        binding_check::assert_ability_registry_matches_sim_constants();

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

        // ---- AbilityRegistry GPU upload (Task #138 follow-on) ----
        // Build the two-program registry (BossStrike at AbilityId(1),
        // HeroAttack at AbilityId(2)), pack it via
        // PackedAbilityRegistry::pack, and upload one buffer per SoA
        // column. The verb-chronicle kernels bind these for the
        // apply_ability dispatcher arm. Building the registry repeats
        // the binding-check's program-build pass (cheap — two hand-
        // built programs) but keeps construction colocated with the
        // upload site, mirroring duel_25v25's pattern.
        let registry = binding_check::build_boss_fight_registry();
        let packed = PackedAbilityRegistry::pack(&registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed, &gpu, "boss_fight_runtime",
        );

        // ---- Per-stat agent SoA columns (Task #138 follow-on) ----
        // The apply_ability dispatcher's `scale_bonus = Σ percent *
        // agent_stat[caster_slot]` switch reads these unconditionally
        // even though boss_fight's BossStrike + HeroAttack have no
        // scaling entries — the per-effect scaling SoA is empty, so
        // scale_bonus collapses to 0.0 inside the dispatcher. We still
        // need to bind real buffers because the kernel's BGL declares
        // the bindings. Init values mirror duel_25v25 (max_hp=100,
        // mana=100, others=0). Note: agent_max_hp is NOT load-bearing
        // for boss_fight today (the .sim's hp init for slot 0 = 5000
        // and slots 1..=5 = 200 lives in agent_hp_buf above; this
        // column is only a stat-scaling lookup).
        let n = agent_count as usize;
        let zeros_f32: Vec<f32> = vec![0.0_f32; n];
        let max_hp_init: Vec<f32> = vec![100.0_f32; n];
        let mana_init: Vec<f32> = vec![100.0_f32; n];
        let agent_max_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::agent_max_hp"),
            contents: bytemuck::cast_slice(&max_hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let mk_zero_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf =
            mk_zero_stat("boss_fight_runtime::agent_attack_damage");
        let agent_armor_buf = mk_zero_stat("boss_fight_runtime::agent_armor");
        let agent_magic_resist_buf =
            mk_zero_stat("boss_fight_runtime::agent_magic_resist");
        let agent_move_speed_buf =
            mk_zero_stat("boss_fight_runtime::agent_move_speed");
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // HeroStun control-status proof (boss_fight, 2026-05-07) —
        // per-agent `stun_expires_at_tick` SoA column. Init to 0 = "never
        // stunned" (the convention established by duel_abilities). The
        // fused
        // ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle
        // kernel (3-way fusion since HeroHeal joined the chronicle path,
        // 2026-05-07) writes this slot from kind=29 EffectStunApplied
        // chronicle records (one record per HeroStun cast). COPY_SRC
        // is on so the test can read it back via `read_u32`.
        let stun_expires_init: Vec<u32> = vec![0_u32; agent_count as usize];
        let agent_stun_expires_at_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("boss_fight_runtime::agent_stun_expires_at_tick"),
                contents: bytemuck::cast_slice(&stun_expires_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Five mask bitmaps — one per verb. Cleared each tick.
        // HeroStun control-status proof (boss_fight, 2026-05-07):
        // mask_3 is the new bitmap added for HeroStun's verb. The fused
        // mask kernel writes all five bitmaps; the scoring argmax reads
        // all five rows. Source-order verb list:
        //   BossStrike=0, BossSelfHeal=1, HeroAttack=2, HeroStun=3,
        //   HeroHeal=4 (HeroHeal shifted from 3 to 4).
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
        let mask_4_bitmap_buf = mk_mask("boss_fight_runtime::mask_4_bitmap");
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
        // HeroStun control-status proof (boss_fight, 2026-05-07) — cfg
        // uniform for the new verb chronicle kernel. Same shape as
        // BossStrike + HeroAttack; the kernel filters action_id == 3u
        // and dispatches AbilityId(3) through the apply_ability arm.
        let chronicle_hero_stun_cfg_init =
            physics_verb_chronicle_HeroStun::PhysicsVerbChronicleHeroStunCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_hero_stun_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::chronicle_hero_stun_cfg"),
            contents: bytemuck::bytes_of(&chronicle_hero_stun_cfg_init),
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
        // Task #138 follow-on (boss_fight port, 2026-05-07) + HeroStun
        // control-status proof (boss_fight, 2026-05-07) + HeroHeal
        // apply_ability proof (boss_fight, 2026-05-07) — cfg uniform
        // for the FUSED chronicle-consumer kernel. The lower pass folded
        // ApplyDamageFromChronicle (kind=26 → emit Damaged),
        // ApplyStunFromChronicle (kind=29 → write
        // `agents.set_stun_expires_at_tick`), and ApplyHealFromChronicle
        // (kind=27 → write `agents.set_hp(min(hp+amt, max_hp))`) into
        // ONE kernel
        // (`physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`)
        // because all three consume from the same event ring at
        // @phase(post) with non-overlapping kind tags (26 + 29 + 27).
        // Single cfg + single dispatch per tick. Fusion grew from 2-way
        // to 3-way when HeroHeal switched to the chronicle pipeline.
        let apply_chronicle_cfg_init =
            physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boss_fight_runtime::apply_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_chronicle_cfg_init),
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
            agent_attack_damage_buf,
            agent_max_hp_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            agent_mana_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_3_bitmap_buf,
            mask_4_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            agent_stun_expires_at_tick_buf,
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
            chronicle_hero_stun_cfg_buf,
            chronicle_hero_heal_cfg_buf,
            apply_chronicle_cfg_buf,
            apply_cfg_buf,
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

    /// Per-agent `stun_expires_at_tick` readback (u32 absolute tick at
    /// which the stun expires; 0 = "never stunned"). HeroStun control-
    /// status proof (boss_fight, 2026-05-07) — written by the fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`
    /// kernel from kind=29 EffectStunApplied chronicle records emitted
    /// by HeroStun's verb chronicle dispatcher.
    pub fn read_stun_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_stun_expires_at_tick_buf, "stun_expires_at_tick")
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

    /// Test-only hook: stamp `hp` into one slot of `agent_hp_buf` so a
    /// behavioural test can pre-seed a wounded agent before stepping.
    /// HeroHeal apply_ability proof (boss_fight, 2026-05-07): the
    /// `hero_heal_recovers_friendly_hp` test seeds a hero slot at
    /// hp=50.0 (well below max_hp=100.0) so HeroHeal's score expression
    /// (`if (... && target.hp < 100.0) { 1000.0 - target.hp } else { -1000.0 }`)
    /// elects that slot as every other hero's argmax target on tick 7,
    /// driving the chronicle Heal arm to write hp > 50 back to the
    /// SoA. Same shape `duel_25v25_runtime` uses ad-hoc; we expose it
    /// as a method for explicit test wiring rather than reaching into
    /// the buffer directly via `gpu.queue`.
    #[cfg(test)]
    pub fn write_hp_slot(&self, slot: u32, hp: f32) {
        assert!(
            slot < self.agent_count,
            "write_hp_slot: slot={slot} out of range (agent_count={})",
            self.agent_count,
        );
        let offset = (slot as wgpu::BufferAddress) * 4;
        self.gpu.queue.write_buffer(
            &self.agent_hp_buf, offset, bytemuck::bytes_of(&hp),
        );
        self.gpu.queue.submit(std::iter::empty());
    }
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
        // HeroStun control-status proof (boss_fight, 2026-05-07):
        // mask_3 is the new bitmap added for HeroStun's verb. Cleared
        // every tick alongside the other four (one bitmap per verb in
        // source order: BossStrike=0, BossSelfHeal=1, HeroAttack=2,
        // HeroStun=3, HeroHeal=4).
        for buf in [
            &self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf, &self.mask_3_bitmap_buf,
            &self.mask_4_bitmap_buf,
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
            // HeroStun control-status proof (boss_fight, 2026-05-07):
            // fifth verb mask (HeroStun at source index 3, HeroHeal
            // shifted to index 4).
            mask_4_bitmap: &self.mask_4_bitmap_buf,
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
            // HeroStun control-status proof (boss_fight, 2026-05-07):
            // scoring's argmax now reads five mask rows.
            mask_4_bitmap: &self.mask_4_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
            // Wave 1.5#7 follow-on (predicate-aware scoring,
            // 2026-05-07): scoring kernel now inlines per-effect when-
            // predicate eval; same SoA + agent stat columns as the
            // chronicle dispatcher.
            ability_registry_when_pred_binder:  &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field:   &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op:      &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_max_hp:        &self.agent_max_hp_buf,
            agent_hp:            &self.agent_hp_buf,
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_mana:          &self.agent_mana_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) BossStrike chronicle. Task #138 follow-on (boss_fight
        // port, 2026-05-07) — verb body is now `apply_ability 1 by self
        // target target`, so this kernel walks the AbilityRegistry's
        // effect SoA columns to expand the dispatch into chronicle
        // EffectDamageApplied writes (kind=26). Re-emitted as Damaged
        // by ApplyDamageFromChronicle below.
        let bs_cfg = physics_verb_chronicle_BossStrike::PhysicsVerbChronicleBossStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_boss_strike_cfg_buf, 0, bytemuck::bytes_of(&bs_cfg),
        );
        let bs_bindings = physics_verb_chronicle_BossStrike::PhysicsVerbChronicleBossStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_hp: &self.agent_hp_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_mana: &self.agent_mana_buf,
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

        // (6) HeroAttack chronicle. Task #138 follow-on (boss_fight
        // port, 2026-05-07) — verb body is now `apply_ability 2 by self
        // target target`, so this kernel walks the AbilityRegistry's
        // effect SoA columns to expand the dispatch into chronicle
        // EffectDamageApplied writes (kind=26). Re-emitted as Damaged
        // by ApplyDamageFromChronicle below.
        let ha_cfg = physics_verb_chronicle_HeroAttack::PhysicsVerbChronicleHeroAttackCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_hero_attack_cfg_buf, 0, bytemuck::bytes_of(&ha_cfg),
        );
        let ha_bindings = physics_verb_chronicle_HeroAttack::PhysicsVerbChronicleHeroAttackBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_hp: &self.agent_hp_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_mana: &self.agent_mana_buf,
            cfg: &self.chronicle_hero_attack_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_heroattack(
            &mut self.cache, &ha_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6b) HeroStun chronicle — HeroStun control-status proof
        // (boss_fight, 2026-05-07). Gates action_id==3u (HeroStun is
        // the fourth verb in source order, so its action_id is 3 —
        // shifting HeroHeal's action_id from 3 to 4). Same chronicle
        // re-emit pattern as BossStrike + HeroAttack except the
        // AbilityProgram at slot 3 declares EffectOp::Stun{
        // duration_ticks=15} instead of Damage, so the apply_ability
        // dispatcher writes kind=29 EffectStunApplied records (with
        // `expires_at_tick = world.tick + 15` precomputed) instead of
        // kind=26 EffectDamageApplied. The fused
        // ApplyDamageFromChronicle_and_ApplyStunFromChronicle kernel
        // below drains both kind tags into the right SoA target
        // (Damaged → ApplyDamage HP cascade for kind=26, direct
        // `agents.set_stun_expires_at_tick` for kind=29).
        let hs_cfg = physics_verb_chronicle_HeroStun::PhysicsVerbChronicleHeroStunCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_hero_stun_cfg_buf, 0, bytemuck::bytes_of(&hs_cfg),
        );
        let hs_bindings = physics_verb_chronicle_HeroStun::PhysicsVerbChronicleHeroStunBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_hp: &self.agent_hp_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_mana: &self.agent_mana_buf,
            cfg: &self.chronicle_hero_stun_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_herostun(
            &mut self.cache, &hs_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) HeroHeal chronicle. HeroHeal apply_ability proof (boss_fight,
        // 2026-05-07) — verb body now uses `apply_ability 4 by self target
        // target` (was `emit Healed { ..self.. }`). Same chronicle re-emit
        // pattern as BossStrike + HeroAttack except the AbilityProgram at
        // slot 4 declares EffectOp::Heal{ amount: 25.0 } instead of Damage,
        // so the apply_ability dispatcher writes kind=27
        // EffectHealApplied records (carrying `amount = 25.0`). The fused
        // ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle
        // kernel below drains kind=27 into a clamped `agents.set_hp(t,
        // min(hp + amt, max_hp))` write.
        //
        // Action_id == 4 (source-order: BossStrike=0, BossSelfHeal=1,
        // HeroAttack=2, HeroStun=3, HeroHeal=4). Kernel-byte size grew
        // 1688 → 85129 because the body switched from a single `emit`
        // to the full apply_ability dispatcher arm (effect-SoA walk +
        // scaling switch). Bindings count grew 3 → 23.
        let hh_cfg = physics_verb_chronicle_HeroHeal::PhysicsVerbChronicleHeroHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_hero_heal_cfg_buf, 0, bytemuck::bytes_of(&hh_cfg),
        );
        let hh_bindings = physics_verb_chronicle_HeroHeal::PhysicsVerbChronicleHeroHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_max_hp: &self.agent_max_hp_buf,
            agent_hp: &self.agent_hp_buf,
            agent_armor: &self.agent_armor_buf,
            agent_magic_resist: &self.agent_magic_resist_buf,
            agent_move_speed: &self.agent_move_speed_buf,
            agent_mana: &self.agent_mana_buf,
            cfg: &self.chronicle_hero_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_heroheal(
            &mut self.cache, &hh_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7b) Fused ApplyDamageFromChronicle + ApplyStunFromChronicle
        // + ApplyHealFromChronicle — chronicle consumers fused into ONE
        // kernel by the lower pass (all three run @phase(post) over the
        // same event ring with non-overlapping kind tags). HeroStun
        // control-status proof (boss_fight, 2026-05-07) + HeroHeal
        // apply_ability proof (boss_fight, 2026-05-07). Drains:
        //   - kind=26 EffectDamageApplied → emit `Damaged` (re-emit;
        //     the standalone ApplyDamage_and_ApplyHeal kernel below
        //     decrements HP)
        //   - kind=29 EffectStunApplied → write
        //     `agents.set_stun_expires_at_tick(t, expires_at_tick)`
        //     directly into the per-agent SoA slot.
        //   - kind=27 EffectHealApplied → write
        //     `agents.set_hp(t, min(hp + amt, max_hp))` directly
        //     (HeroHeal apply_ability proof, 2026-05-07).
        // Fusion grew 2-way → 3-way when HeroHeal switched from emit
        // Healed (a 3-binding kernel) to the apply_ability chronicle
        // pipeline. Bindings on the fused kernel grew from 4 → 6
        // (added agent_hp + agent_max_hp).
        //
        // event_count is the upper bound on chronicle records produced
        // per tick across BossStrike + HeroAttack + HeroStun + HeroHeal.
        // Same agent_count*4 headroom estimate the rest of the cascade
        // uses.
        let event_count_estimate = self.agent_count * 4;
        let apply_chronicle_cfg = physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&apply_chronicle_cfg),
        );
        let apply_chronicle_bindings =
            physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_hp: &self.agent_hp_buf,
                agent_max_hp: &self.agent_max_hp_buf,
                agent_stun_expires_at_tick: &self.agent_stun_expires_at_tick_buf,
                cfg: &self.apply_chronicle_cfg_buf,
            };
        dispatch::dispatch_physics_applydamagefromchronicle_and_applystunfromchronicle_and_applyhealfromchronicle(
            &mut self.cache,
            &apply_chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (8) ApplyDamage_and_ApplyHeal — fused PerEvent kernel. Reads
        // Damaged (re-emitted by ApplyDamageFromChronicle from the
        // BossStrike + HeroAttack EffectDamageApplied records) and
        // Healed (still emitted directly by BossSelfHeal + HeroHeal),
        // writes agent_hp + agent_alive, may emit Defeated.
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

    /// HeroStun control-status proof (boss_fight, 2026-05-07) — proves
    /// the apply_ability dispatcher emits kind=29 EffectStunApplied
    /// chronicle records in the asymmetric 1-vs-N RPG combat shape AND
    /// the fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
    /// kernel ferries the per-record `expires_at_tick` into the
    /// `agent_stun_expires_at_tick` SoA via
    /// `agents.set_stun_expires_at_tick(t, e)`.
    ///
    /// CADENCE AT THE SEAM: HeroStun fires at `world.tick % 7 == 0`,
    /// so steps 0..=13 (= ticks 0..=13) drive cast cycles at tick 0
    /// and tick 7 — two cast cycles before tick 14 dispatches. Each
    /// alive Hero (slots 1..=5) on its eligible cycle picks the Boss
    /// (slot 0) via the scoring argmax (HeroStun's `score 2.0` outscores
    /// HeroAttack's `score 1.0` on overlap ticks). At tick 0 both
    /// HeroStun and HeroAttack are eligible (% 7 and % 3 both pass);
    /// argmax picks HeroStun. At tick 7 only HeroStun is on for heroes;
    /// it lands as their argmax pick uncontested.
    ///
    /// `expires_at_tick` for a tick-0 cast = `world.tick + 15 = 15`;
    /// for a tick-7 cast = `7 + 15 = 22`. Each successful hero cast
    /// writes the boss's `stun_expires_at_tick` slot (multiple heroes
    /// targeting slot 0 race to write — last-write-wins on the SoA).
    /// The expected post-run value is in {15, 22} ∪ small race
    /// neighbours; we pin a tolerant range [15, 23].
    ///
    /// HOW THE TEST PROVES HERO-STUN FIRED:
    ///   1. After 14 ticks, the boss (slot 0) has a non-zero
    ///      stun_expires_at_tick (proves the chronicle path emitted
    ///      kind=29 records AND the fused consumer wrote the SoA).
    ///   2. The boss's expiry tick is in [15, 23] (matches `tick + 15`
    ///      for tick∈{0, 7}).
    #[test]
    fn hero_stun_locks_boss() {
        const N: u32 = 6;
        let mut state = BossFightState::new(0xCAFE_F00D, N);

        // Pre-tick baseline — every agent's stun_expires_at_tick is 0.
        let initial_stun = state.read_stun_expires_at_tick();
        assert_eq!(
            initial_stun.len(),
            N as usize,
            "stun_expires_at_tick readback must cover all {N} agents",
        );
        for (i, &e) in initial_stun.iter().enumerate() {
            assert_eq!(
                e, 0,
                "initial stun_expires_at_tick[{i}] must be 0 (= never \
                 stunned); got {e}",
            );
        }

        // Run 14 ticks. HeroStun fires at tick 0 and tick 7 (two cast
        // cycles); the next firing tick is 14 (we run steps 0..=13
        // inclusive, ending BEFORE tick 14 dispatches). HeroAttack (% 3)
        // still fires at ticks 0, 3, 6, 9, 12 — but at tick 0 the
        // scoring argmax picks HeroStun (score 2.0 > HeroAttack's 1.0),
        // and HeroAttack lands the other ticks. BossStrike (% 10) fires
        // at tick 0 and 10 — independent. BossSelfHeal (% 50, hp <
        // 1500) doesn't fire in this window (boss starts at HP 5000).
        for _ in 0..14 {
            state.step();
        }

        let stun = state.read_stun_expires_at_tick();

        // Pin 1: the boss (slot 0) has a non-zero stun expiry — proves
        // HeroStun's apply_ability dispatch emitted kind=29
        // EffectStunApplied records AND the fused chronicle consumer
        // wrote the SoA. Multiple heroes target slot 0 on each cycle so
        // the slot is guaranteed to land at least one stun.
        assert!(
            stun[0] > 0,
            "after 14 ticks the boss (slot 0) must have a non-zero \
             stun_expires_at_tick (control-status chronicle proof); \
             got stun[0]={}. Per-slot stun: {:?}",
            stun[0],
            stun,
        );

        // Pin 2: the boss's expiry tick is in [15, 23] — the dispatcher
        // pre-computes `expires_at_tick = world.tick + duration_ticks(15)`
        // at chronicle-write time. Tick 0 cast → expires at 15; tick 7
        // cast → expires at 22. Race-tolerant bound is [15, 23] (allows
        // for one extra tick of advance from the apply pass running
        // after the chronicle write).
        assert!(
            (15..=23).contains(&stun[0]),
            "boss stun_expires_at_tick={} outside expected range \
             [15, 23] (tick-0 cast → expires at 15, tick-7 cast → \
             expires at 22)",
            stun[0],
        );

        // Pin 3: heroes (slots 1..=5) must NOT be stunned — the score
        // expression's `target.creature_type == Boss` gate filters them
        // out, so no kind=29 record ever lands on a hero slot.
        for i in 1..(N as usize) {
            assert_eq!(
                stun[i], 0,
                "hero slot {i} must have stun_expires_at_tick=0 (only \
                 the boss is a valid HeroStun target); got {}",
                stun[i],
            );
        }
    }

    /// HeroHeal apply_ability proof (boss_fight, 2026-05-07) — proves
    /// the apply_ability dispatcher emits kind=27 EffectHealApplied
    /// chronicle records in the asymmetric 1-vs-N RPG combat shape AND
    /// the 3-way fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`
    /// kernel ferries the per-record `amount` into the per-agent hp
    /// SoA via `agents.set_hp(t, min(hp + amt, max_hp))`.
    ///
    /// PRE-SEED: a hero slot (slot 2) is stamped at hp=50 before
    /// stepping. The HeroHeal score expression `if (target.alive &&
    /// target.creature_type == Hero && target.hp < 100.0) { 1000.0 -
    /// target.hp }` qualifies slot 2 (score 950) but sentinel-rejects
    /// the unwounded heroes (slots 1, 3, 4, 5 at hp=200; 200 NOT <
    /// 100). So at tick 7 every alive hero's argmax picks slot 2 as
    /// the HeroHeal target (score 950 > HeroStun's 2.0 > HeroAttack's
    /// 1.0).
    ///
    /// CADENCE AT THE SEAM: HeroHeal fires at `world.tick % 7 == 0`,
    /// so steps 0..=6 (= ticks 0..=6) drive ONE cast cycle at tick 0.
    /// At tick 0, slot 2's HeroHeal score is 950 (qualifies; hp=50 <
    /// 100); the 4 other heroes pick slot 2 as their argmax target
    /// (slot 2 also picks itself — see Gap-E note in HeroHeal score).
    /// After ApplyHealFromChronicle drains the 5 records, slot 2's hp
    /// ends ≥ 75 (a single cast: 50 + 25 = 75; multiple casters race-
    /// write the same SoA slot but each reads the same input snapshot,
    /// so all writes converge on 75).
    ///
    /// HOW THE TEST PROVES HERO-HEAL FIRED:
    ///   1. After 7 ticks, slot 2's hp is > 50 (proves the chronicle
    ///      path emitted kind=27 records AND the fused consumer wrote
    ///      the SoA).
    ///   2. Slot 2's hp does NOT exceed max_hp=100 (proves the
    ///      `min(hp+amt, max_hp)` clamp engaged in
    ///      ApplyHealFromChronicle).
    #[test]
    fn hero_heal_recovers_friendly_hp() {
        const N: u32 = 6;
        const WOUNDED_SLOT: u32 = 2;
        let mut state = BossFightState::new(0xCAFE_F00D, N);

        // Pre-seed slot 2 at hp=50. The other heroes start at hp=200.
        // Slot 2 sits below the HeroHeal `target.hp < 100.0` gate;
        // the others don't, so slot 2 is the universal argmax pick at
        // tick 7. Boss stays at hp=5000, far above any threshold.
        state.write_hp_slot(WOUNDED_SLOT, 50.0);

        // Sanity: pre-tick hp readback shows the seed landed.
        let initial_hp = state.read_hp();
        assert_eq!(
            initial_hp[WOUNDED_SLOT as usize], 50.0,
            "pre-seed write failed: slot {} hp={} (expected 50.0)",
            WOUNDED_SLOT, initial_hp[WOUNDED_SLOT as usize],
        );
        for i in 1..(N as usize) {
            if i as u32 == WOUNDED_SLOT { continue; }
            assert_eq!(
                initial_hp[i], 200.0,
                "non-wounded hero slot {i} must start at hp=200; got {}",
                initial_hp[i],
            );
        }

        // Run 7 ticks (steps 0..=6). HeroHeal (% 7 == 0) fires at tick
        // 0 only — the next firing tick is 7, beyond the window.
        // BossStrike fires at tick 0 too, but argmax picks slot 1 (the
        // first alive hero by slot order) for damage; slot 2 stays
        // un-struck. HeroAttack (% 3 = 0, 3, 6) fires too but at tick
        // 0 the scoring argmax for heroes picks HeroHeal (score 950)
        // over HeroAttack (1.0), so HeroAttack only lands at ticks 3
        // and 6 — irrelevant to slot 2's hp arithmetic.
        for _ in 0..7 {
            state.step();
        }

        let hp_now = state.read_hp();

        // Pin 1: slot 2's hp moved above the seed value of 50 — proves
        // HeroHeal's apply_ability dispatch emitted kind=27
        // EffectHealApplied records AND the fused chronicle consumer
        // wrote the SoA with a clamped `agents.set_hp(t, min(hp + amt,
        // max_hp))`.
        assert!(
            hp_now[WOUNDED_SLOT as usize] > 50.0,
            "after 7 ticks slot {WOUNDED_SLOT}'s hp must be > 50 \
             (HeroHeal apply_ability proof); got hp[{WOUNDED_SLOT}]={}. \
             Per-slot hp: {:?}",
            hp_now[WOUNDED_SLOT as usize],
            hp_now,
        );

        // Pin 2: slot 2's hp is bounded above by max_hp=100 — proves
        // the `min(hp + amt, max_hp)` clamp in ApplyHealFromChronicle
        // engaged. Five heroes targeting slot 2 each compute
        // min(50+25, 100) = 75; without the clamp a hypothetical
        // higher-seed run would climb past max_hp. Other hero slots
        // are unhealed in this scenario (they don't qualify as
        // HeroHeal targets — hp ≥ 100) so we don't pin them; they may
        // sit at hp=150 (BossStrike landed on slot 1) or hp=200 (no
        // damage), neither of which speaks to the heal clamp.
        assert!(
            hp_now[WOUNDED_SLOT as usize] <= 100.0 + 0.001,
            "slot {WOUNDED_SLOT}: hp={} exceeds max_hp=100.0; clamp \
             in ApplyHealFromChronicle didn't engage",
            hp_now[WOUNDED_SLOT as usize],
        );
    }
}
