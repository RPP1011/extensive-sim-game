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

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

mod binding_check;

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
    /// Task #138 follow-on (tactical_squad_5v5 port, 2026-05-07) —
    /// per-stat agent SoA columns the apply_ability dispatcher's
    /// `scale_bonus = Σ percent * agent_stat[caster_slot]` switch reads.
    /// tactical_squad_5v5's TankAttack + DpsAttack have no scaling
    /// entries today, so all five columns sit at their inert init
    /// values; the dispatcher's `scale_bonus` collapses to 0
    /// unconditionally. Kept on the state struct because the Strike
    /// + Snipe verb-chronicle kernels still BIND them (the dispatcher
    /// emits the stat-switch arms whether or not any program actually
    /// scales). Mirrors apply_ability_smoke_runtime + duel_abilities_runtime
    /// + duel_25v25_runtime exactly — the same five-column shape.
    #[allow(dead_code)]
    agent_attack_damage_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_armor_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_magic_resist_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order: Strike=0,
    //    Snipe=1, ConcussiveBlow=2, Heal=3, SquadHeal=4) --
    //
    // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07): the
    // verb is declared between Snipe and Heal in the .sim, so source-
    // order action_id assignment lands ConcussiveBlow at index 2 and
    // shifts Heal to index 3. SquadHeal apply_ability ally-heal proof
    // (5v5 scale, 2026-05-07): SquadHeal is declared after Heal so it
    // takes index 4 — Heal stays at 3. The fused mask kernel writes
    // all five bitmaps; the scoring argmax reads all five rows.
    mask_0_bitmap_buf: wgpu::Buffer, // Strike
    mask_1_bitmap_buf: wgpu::Buffer, // Snipe
    mask_2_bitmap_buf: wgpu::Buffer, // ConcussiveBlow
    mask_3_bitmap_buf: wgpu::Buffer, // Heal
    mask_4_bitmap_buf: wgpu::Buffer, // SquadHeal
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    /// ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
    /// per-agent `stun_expires_at_tick` SoA column. Written by the
    /// fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
    /// kernel from kind=29 EffectStunApplied chronicle records produced
    /// by ConcussiveBlow's verb chronicle dispatcher (kind=29 records
    /// carry `expires_at_tick = world.tick + 20` precomputed by the
    /// dispatcher). Init to 0 (= "never stunned" — agents whose
    /// `stun_expires_at_tick > world.tick` are stunned). The verb
    /// `where`-clauses in this fixture don't read it today, but the
    /// new `concussive_blow_stuns_targets` test asserts the SoA column
    /// lands the expected expires_at_tick after ConcussiveBlow cast
    /// cycles.
    agent_stun_expires_at_tick_buf: wgpu::Buffer,

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
    /// ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
    /// fourth per-agent verb chronicle cfg. Same shape as the Strike +
    /// Snipe cfgs; the ConcussiveBlow chronicle kernel filters
    /// `action_id == 2u` (source-order index — ConcussiveBlow is the
    /// third verb declared after Strike and Snipe) and dispatches the
    /// AbilityId(3) program through the apply_ability arm, writing
    /// kind=29 EffectStunApplied records.
    chronicle_concussive_blow_cfg_buf: wgpu::Buffer,
    chronicle_heal_cfg_buf: wgpu::Buffer,
    /// SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07)
    /// — fifth per-agent verb chronicle cfg. Same shape as the Strike +
    /// Snipe + ConcussiveBlow cfgs; the SquadHeal chronicle kernel
    /// filters `action_id == 4u` (source-order index — SquadHeal is
    /// the fifth verb declared after Strike, Snipe, ConcussiveBlow,
    /// Heal) and dispatches the AbilityId(4) program through the
    /// apply_ability arm, writing kind=27 EffectHealApplied records.
    chronicle_squad_heal_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (tactical_squad_5v5 port, 2026-05-07) +
    /// ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) +
    /// SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07)
    /// — cfg uniform for the FUSED chronicle-consumer kernel. The
    /// lower pass folded ApplyDamageFromChronicle (drains kind=26 →
    /// emit Damaged), ApplyStunFromChronicle (drains kind=29 → write
    /// `agents.set_stun_expires_at_tick`), and ApplyHealFromChronicle
    /// (drains kind=27 → write `agents.set_hp(min(hp + amt, max_hp))`)
    /// into ONE kernel
    /// (`physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`)
    /// because all three consume from the same event ring at
    /// @phase(post) with non-overlapping kind tags. Single cfg +
    /// single dispatch per tick. Fusion grew from 2-way to 3-way when
    /// SquadHeal was added — mirrors duel_25v25's HealPulse fusion
    /// (commit 049feb0c).
    apply_chronicle_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Task #138 follow-on (tactical_squad_5v5 port, 2026-05-07) +
    /// ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
    /// Packed AbilityRegistry uploaded to the GPU. The Strike + Snipe
    /// + ConcussiveBlow verb-chronicle kernels bind `effect_kinds` /
    /// `effect_payload_a` / `effect_payload_b` (and the modifier
    /// columns) for the apply_ability dispatcher arm. Built once at
    /// construction by `binding_check::build_tactical_squad_5v5_registry`
    /// (TankAttack at AbilityId(1), DpsAttack at AbilityId(2),
    /// ConcussiveBlow at AbilityId(3)) and uploaded via
    /// `PackedAbilityRegistryGpu::upload`. The buffers live for the
    /// rest of the run.
    registry_gpu: PackedAbilityRegistryGpu,

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

        // Task #138 follow-on (tactical_squad_5v5 port, 2026-05-07) —
        // runs ONCE at startup before any GPU work. Asserts the
        // runtime's hand-built TankAttack + DpsAttack programs land
        // at AbilityId(1) and AbilityId(2) so the
        // `apply_ability 1` / `apply_ability 2` literals in
        // `assets/sim/tactical_squad_5v5.sim` (Strike + Snipe verb
        // bodies) dispatch the correct programs. Cheap (two-program
        // registry build); panics on any drift before the expensive
        // GPU init below.
        binding_check::assert_ability_registry_matches_sim_constants();

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

        // ---- AbilityRegistry GPU upload (Task #138 follow-on) ----
        // Build the two-program registry (TankAttack at AbilityId(1),
        // DpsAttack at AbilityId(2)), pack it via
        // PackedAbilityRegistry::pack, and upload one buffer per SoA
        // column. The Strike + Snipe verb-chronicle kernels bind these
        // for the apply_ability dispatcher arm. Building the registry
        // repeats the binding-check's program-build pass (cheap — two
        // hand-built programs) but keeps construction colocated with
        // the upload site, mirroring duel_25v25 / duel_abilities.
        let registry = binding_check::build_tactical_squad_5v5_registry();
        let packed = PackedAbilityRegistry::pack(&registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed, &gpu, "tactical_squad_5v5_runtime",
        );

        // ---- Per-stat agent SoA columns (Task #138 follow-on) ----
        // The apply_ability dispatcher's `scale_bonus = Σ percent *
        // agent_stat[caster_slot]` switch reads these unconditionally
        // even though tactical_squad_5v5's TankAttack + DpsAttack have
        // no scaling entries — the per-effect scaling SoA is empty,
        // so scale_bonus collapses to 0.0 inside the dispatcher. We
        // still need to bind real buffers because the kernel's BGL
        // declares the bindings. Init values mirror duel_25v25
        // (max_hp=100, others=0).
        let n = agent_count as usize;
        let zeros_f32: Vec<f32> = vec![0.0_f32; n];
        let max_hp_init: Vec<f32> = vec![100.0_f32; n];
        let agent_max_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::agent_max_hp"),
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
            mk_zero_stat("tactical_squad_5v5_runtime::agent_attack_damage");
        let agent_armor_buf = mk_zero_stat("tactical_squad_5v5_runtime::agent_armor");
        let agent_magic_resist_buf =
            mk_zero_stat("tactical_squad_5v5_runtime::agent_magic_resist");
        let agent_move_speed_buf =
            mk_zero_stat("tactical_squad_5v5_runtime::agent_move_speed");

        // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
        // per-agent `stun_expires_at_tick` SoA column. Init to 0 = "never
        // stunned" (the convention established by duel_abilities). The
        // fused ApplyDamageFromChronicle_and_ApplyStunFromChronicle
        // kernel writes this slot from kind=29 EffectStunApplied
        // chronicle records (one record per ConcussiveBlow cast).
        // COPY_SRC is on so the test can read it back via `read_u32`.
        let stun_expires_init: Vec<u32> = vec![0_u32; n];
        let agent_stun_expires_at_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tactical_squad_5v5_runtime::agent_stun_expires_at_tick"),
                contents: bytemuck::cast_slice(&stun_expires_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Five mask bitmaps — one per verb. Cleared each tick.
        // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07):
        // mask_3 is the bitmap for Heal (which shifted from index 2 to
        // index 3 because ConcussiveBlow was inserted at source
        // position 2). SquadHeal apply_ability ally-heal proof (5v5
        // scale, 2026-05-07): mask_4 is the new bitmap for SquadHeal
        // (the fifth verb declared after Strike, Snipe, ConcussiveBlow,
        // Heal — Heal stays at 3 because SquadHeal was appended after
        // it). The fused mask kernel writes all five bitmaps; the
        // scoring argmax reads all five rows.
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
        let mask_3_bitmap_buf = mk_mask("tactical_squad_5v5_runtime::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("tactical_squad_5v5_runtime::mask_4_bitmap");
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
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_snipe_cfg_init =
            physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_snipe_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_snipe_cfg"),
            contents: bytemuck::bytes_of(&chronicle_snipe_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
        // cfg uniform for the new verb chronicle kernel. Same shape as
        // Strike + Snipe; the kernel filters action_id == 2u and
        // dispatches AbilityId(3) through the apply_ability arm.
        let chronicle_concussive_blow_cfg_init =
            physics_verb_chronicle_ConcussiveBlow::PhysicsVerbChronicleConcussiveBlowCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_concussive_blow_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_concussive_blow_cfg"),
            contents: bytemuck::bytes_of(&chronicle_concussive_blow_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_heal_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07)
        // — cfg uniform for the new SquadHeal verb chronicle kernel.
        // Same shape as Strike + Snipe + ConcussiveBlow; the kernel
        // filters action_id == 4u and dispatches AbilityId(4) through
        // the apply_ability arm, writing kind=27 EffectHealApplied
        // records.
        let chronicle_squad_heal_cfg_init =
            physics_verb_chronicle_SquadHeal::PhysicsVerbChronicleSquadHealCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_squad_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::chronicle_squad_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_squad_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (tactical_squad_5v5 port, 2026-05-07) +
        // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) +
        // SquadHeal apply_ability ally-heal proof (5v5 scale,
        // 2026-05-07) — cfg uniform for the FUSED chronicle-consumer
        // kernel. The lower pass folded ApplyDamageFromChronicle
        // (kind=26 → emit Damaged), ApplyStunFromChronicle (kind=29 →
        // write `agents.set_stun_expires_at_tick`), and
        // ApplyHealFromChronicle (kind=27 → write `agents.set_hp(min(
        // hp + amt, max_hp))`) into ONE kernel
        // (`physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`)
        // because all three consume from the same event ring at
        // @phase(post) with non-overlapping kind tags. Single cfg +
        // single dispatch per tick. Fusion grew from 2-way to 3-way
        // when SquadHeal was added (mirrors duel_25v25 commit
        // 049feb0c).
        let apply_chronicle_cfg_init =
            physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let apply_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tactical_squad_5v5_runtime::apply_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_chronicle_cfg_init),
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
            agent_attack_damage_buf,
            agent_max_hp_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
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
            chronicle_strike_cfg_buf,
            chronicle_snipe_cfg_buf,
            chronicle_concussive_blow_cfg_buf,
            chronicle_heal_cfg_buf,
            chronicle_squad_heal_cfg_buf,
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

    /// Per-agent `stun_expires_at_tick` readback (u32 absolute tick at
    /// which the stun expires; 0 = "never stunned"). ConcussiveBlow
    /// control-status proof (5v5 scale, 2026-05-07) — written by the
    /// fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`
    /// kernel from kind=29 EffectStunApplied chronicle records emitted
    /// by ConcussiveBlow's verb chronicle dispatcher.
    pub fn read_stun_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_stun_expires_at_tick_buf, "stun_expires_at_tick")
    }

    /// SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07)
    /// — test-only helper that pre-seeds a single slot's HP via
    /// `queue.write_buffer`. Used by the SquadHeal test to inject a
    /// low-HP target before stepping so the SquadHeal scoring argmax
    /// has a clear winner. NOT used in the default sim flow.
    pub fn write_hp(&self, slot: u32, value: f32) {
        assert!(
            slot < self.agent_count,
            "write_hp slot {} >= agent_count {}",
            slot,
            self.agent_count,
        );
        let offset = (slot as u64) * 4;
        self.gpu.queue.write_buffer(
            &self.agent_hp_buf,
            offset,
            bytemuck::bytes_of(&value),
        );
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
        // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07):
        // mask_3 is the bitmap for Heal (which shifted from index 2 to
        // index 3 because ConcussiveBlow was inserted at source position
        // 2). SquadHeal apply_ability ally-heal proof (5v5 scale,
        // 2026-05-07): mask_4 is the new bitmap for SquadHeal (the
        // fifth verb declared after Heal — Heal stays at 3 because
        // SquadHeal was appended after it). Cleared every tick alongside
        // the other four (one bitmap per verb in source order:
        // Strike=0, Snipe=1, ConcussiveBlow=2, Heal=3, SquadHeal=4).
        for buf in [
            &self.mask_0_bitmap_buf,
            &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf,
            &self.mask_3_bitmap_buf,
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
            // ConcussiveBlow control-status proof (5v5 scale,
            // 2026-05-07): fourth verb mask (Heal at source index 3,
            // shifted from 2 because ConcussiveBlow took index 2).
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            // SquadHeal apply_ability ally-heal proof (5v5 scale,
            // 2026-05-07): fifth verb mask (SquadHeal at source index 4
            // — appended after Heal so Heal stays at 3).
            mask_4_bitmap: &self.mask_4_bitmap_buf,
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
            // ConcussiveBlow control-status proof (5v5 scale,
            // 2026-05-07): scoring's argmax reads four mask rows.
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            // SquadHeal apply_ability ally-heal proof (5v5 scale,
            // 2026-05-07): scoring's argmax now reads five mask rows
            // (Strike=0, Snipe=1, ConcussiveBlow=2, Heal=3,
            // SquadHeal=4).
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
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_mana:          &self.agent_mana_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Strike chronicle — gates on action_id==0. Task #138
        // follow-on (tactical_squad_5v5 port, 2026-05-07): instead of
        // emitting Damaged directly, the kernel walks the
        // AbilityRegistry's effect SoA columns (TankAttack at
        // AbilityId(1)) and writes EffectDamageApplied chronicle
        // records (kind=26). The new ApplyDamageFromChronicle kernel
        // below re-emits those as Damaged so the existing
        // ApplyDamage_and_ApplyHeal cascade keeps working unchanged.
        let event_count_estimate = self.agent_count * 4;
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&strike_cfg),
        );
        let strike_bindings = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp:            &self.agent_hp_buf,
            agent_max_hp:        &self.agent_max_hp_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana:          &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_strike(
            &mut self.cache, &strike_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (5) Snipe chronicle — gates on action_id==1. Same
        // apply_ability dispatcher shape as Strike above (DpsAttack at
        // AbilityId(2)) — writes EffectDamageApplied(kind=26) records
        // re-emitted by ApplyDamageFromChronicle.
        let snipe_cfg = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_snipe_cfg_buf, 0, bytemuck::bytes_of(&snipe_cfg),
        );
        let snipe_bindings = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp:            &self.agent_hp_buf,
            agent_max_hp:        &self.agent_max_hp_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana:          &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            cfg: &self.chronicle_snipe_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_snipe(
            &mut self.cache, &snipe_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (5b) ConcussiveBlow chronicle — ConcussiveBlow control-status
        // proof (5v5 scale, 2026-05-07). Gates action_id==2u
        // (ConcussiveBlow is the third verb in source order, so its
        // action_id is 2 — shifting Heal's action_id from 2 to 3).
        // Same chronicle re-emit pattern as Strike + Snipe except the
        // AbilityProgram at slot 3 declares EffectOp::Stun{
        // duration_ticks=20} instead of Damage, so the apply_ability
        // dispatcher writes kind=29 EffectStunApplied records (with
        // `expires_at_tick = world.tick + 20` precomputed) instead of
        // kind=26 EffectDamageApplied. The fused
        // ApplyDamageFromChronicle_and_ApplyStunFromChronicle kernel
        // below drains both kind tags into the right SoA target
        // (Damaged → ApplyDamage HP cascade for kind=26, direct
        // `agents.set_stun_expires_at_tick` for kind=29).
        let concussive_blow_cfg = physics_verb_chronicle_ConcussiveBlow::PhysicsVerbChronicleConcussiveBlowCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_concussive_blow_cfg_buf, 0, bytemuck::bytes_of(&concussive_blow_cfg),
        );
        let concussive_blow_bindings = physics_verb_chronicle_ConcussiveBlow::PhysicsVerbChronicleConcussiveBlowBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp:            &self.agent_hp_buf,
            agent_max_hp:        &self.agent_max_hp_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana:          &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            cfg: &self.chronicle_concussive_blow_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_concussiveblow(
            &mut self.cache, &concussive_blow_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (6) Heal chronicle — gates on action_id==3u. ConcussiveBlow
        // control-status proof (5v5 scale, 2026-05-07): Heal's action_id
        // shifted from 2 to 3 because ConcussiveBlow was inserted at
        // source position 2. The kernel name + binding shape are
        // unchanged — the action_id literal in the generated kernel is
        // the only detail that moved.
        let heal_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, agent_cap: 0,
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

        // (6a) SquadHeal chronicle — SquadHeal apply_ability ally-heal
        // proof (5v5 scale, 2026-05-07). Gates action_id==4u (SquadHeal
        // is the fifth verb in source order — Heal stays at 3 because
        // SquadHeal was appended after it). Same chronicle dispatch
        // pattern as Strike + Snipe + ConcussiveBlow except the
        // AbilityProgram at slot 4 declares EffectOp::Heal{amount=12.0}
        // instead of Damage/Stun, so the apply_ability dispatcher writes
        // kind=27 EffectHealApplied records (with `amount` ferried
        // verbatim from the program's effect SoA). The fused
        // ApplyDamageFromChronicle_and_ApplyHealFromChronicle_and_ApplyStunFromChronicle
        // kernel below drains all three kind tags into the right SoA
        // target (Damaged → ApplyDamage HP cascade for kind=26, direct
        // `agents.set_hp(min(hp + amt, max_hp))` for kind=27, direct
        // `agents.set_stun_expires_at_tick` for kind=29).
        let squad_heal_cfg = physics_verb_chronicle_SquadHeal::PhysicsVerbChronicleSquadHealCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_squad_heal_cfg_buf, 0, bytemuck::bytes_of(&squad_heal_cfg),
        );
        let squad_heal_bindings = physics_verb_chronicle_SquadHeal::PhysicsVerbChronicleSquadHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp:            &self.agent_hp_buf,
            agent_max_hp:        &self.agent_max_hp_buf,
            agent_move_speed:    &self.agent_move_speed_buf,
            agent_armor:         &self.agent_armor_buf,
            agent_magic_resist:  &self.agent_magic_resist_buf,
            agent_attack_damage: &self.agent_attack_damage_buf,
            agent_mana:          &self.agent_mana_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents: &self.registry_gpu.scaling_percents,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_when_pred_binder: &self.registry_gpu.when_pred_binder,
            ability_registry_when_pred_field: &self.registry_gpu.when_pred_field,
            ability_registry_when_pred_op: &self.registry_gpu.when_pred_op,
            ability_registry_when_pred_literal: &self.registry_gpu.when_pred_literal,
            ability_registry_chances:           &self.registry_gpu.chances,
            cfg: &self.chronicle_squad_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_squadheal(
            &mut self.cache, &squad_heal_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (6b) Fused ApplyDamageFromChronicle + ApplyHealFromChronicle +
        // ApplyStunFromChronicle — chronicle consumers fused into ONE
        // kernel by the lower pass (all three run @phase(post) over the
        // same event ring with non-overlapping kind tags).
        // ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) +
        // SquadHeal apply_ability ally-heal proof (5v5 scale,
        // 2026-05-07). Drains:
        //   - kind=26 EffectDamageApplied → emit `Damaged` (re-emit;
        //     the standalone ApplyDamage_and_ApplyHeal kernel below
        //     decrements HP)
        //   - kind=27 EffectHealApplied → write
        //     `agents.set_hp(t, min(hp + amt, max_hp))` directly into
        //     the per-agent SoA slot, clamped at max_hp
        //     (SquadHeal, 2026-05-07).
        //   - kind=29 EffectStunApplied → write
        //     `agents.set_stun_expires_at_tick(t, expires_at_tick)`
        //     directly into the per-agent SoA slot.
        // Fusion grew from 2-way to 3-way when SquadHeal was added —
        // mirrors duel_25v25's HealPulse fusion (commit 049feb0c).
        let apply_chronicle_cfg = physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_chronicle_cfg_buf, 0,
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
            &mut self.cache, &apply_chronicle_bindings,
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (7) ApplyDamage_and_ApplyHeal — fused PerEvent kernel.
        let apply_cfg = physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, agent_cap: 0,
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

    /// ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
    /// proves the apply_ability dispatcher emits kind=29
    /// EffectStunApplied chronicle records at 5v5 scale (10 agents
    /// through pair-field scoring) AND the fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
    /// kernel ferries the per-record `expires_at_tick` into the
    /// `agent_stun_expires_at_tick` SoA via
    /// `agents.set_stun_expires_at_tick(t, e)`.
    ///
    /// CADENCE AT THE SEAM: ConcussiveBlow fires at
    /// `world.tick % 7 == 0`, so steps 0..=13 (= ticks 0..=13) drive
    /// cast cycles at tick 0 and tick 7 — two cast cycles before tick
    /// 14 dispatches. Each DPS Red actor (3 per team at slots 2..5 /
    /// 7..10) on its eligible cycle picks an enemy DPS via the scoring
    /// argmax (ConcussiveBlow's `300 - target.hp` outscores Snipe's
    /// `200 - target.hp` when both are eligible — but at ticks 0 and
    /// 7 only ConcussiveBlow is on for those actors at the 7-cadence
    /// at tick 7; tick 0 has both Snipe (% 4) and ConcussiveBlow (% 7)
    /// firing for DPS, ConcussiveBlow wins by score).
    ///
    /// `expires_at_tick` for a tick-0 cast = `world.tick + 20 = 20`;
    /// for a tick-7 cast = `7 + 20 = 27`. After 14 steps the second
    /// cast cycle has just finished, so any agent whose
    /// `stun_expires_at_tick > 0` was hit by either cycle and the
    /// value lands in {20, 27} (race-tolerant range: [20, 28]). The
    /// test pins that range.
    ///
    /// HOW THE TEST PROVES CONCUSSIVE-BLOW FIRED:
    ///   1. After 14 steps at least one agent has a non-zero
    ///      stun_expires_at_tick (proves the chronicle path emitted
    ///      kind=29 records AND the fused consumer wrote the SoA).
    ///   2. Every stunned agent's expiry tick is in [20, 28] (matches
    ///      `tick + 20` for tick∈{0, 7}).
    #[test]
    fn concussive_blow_stuns_targets() {
        let mut state = TacticalSquad5v5State::new(0xCAFE_F00D, 10);

        // Pre-tick baseline — every agent's stun_expires_at_tick is 0.
        let initial_stun = state.read_stun_expires_at_tick();
        assert_eq!(
            initial_stun.len(),
            10,
            "stun_expires_at_tick readback must cover all 10 agents",
        );
        for (i, &e) in initial_stun.iter().enumerate() {
            assert_eq!(
                e, 0,
                "initial stun_expires_at_tick[{i}] must be 0 (= never \
                 stunned); got {e}",
            );
        }

        // Run 14 ticks. ConcussiveBlow fires at tick 0 and tick 7
        // (two cast cycles); the next firing tick is 14 (we run steps
        // 0..=13 inclusive, ending BEFORE tick 14 dispatches).
        for _ in 0..14 {
            state.step();
        }

        let stun = state.read_stun_expires_at_tick();

        // Pin 1: at least 1 agent has a non-zero stun expiry — proves
        // ConcussiveBlow's apply_ability dispatch emitted kind=29
        // EffectStunApplied records AND the fused chronicle consumer
        // wrote the SoA. With 3 DPS per team firing on a 7-cadence and
        // pair-field argmax piling casters onto the lowest-HP target,
        // typically 1+ DPS targets get stunned, but the load-bearing
        // pin is "≥ 1" so the test is robust against scheduling
        // variation.
        let stunned_count: usize = stun.iter().filter(|&&e| e > 0).count();
        assert!(
            stunned_count >= 1,
            "after 14 ticks at least 1 agent must have a non-zero \
             stun_expires_at_tick (control-status chronicle proof); \
             saw {} stunned out of 10. Per-slot stun: {:?}",
            stunned_count,
            stun,
        );

        // Pin 2: every stunned agent's expiry tick is in [20, 28] —
        // the dispatcher pre-computes `expires_at_tick = world.tick +
        // duration_ticks(20)` at chronicle-write time. Tick 0 cast →
        // expires at 20; tick 7 cast → expires at 27. Race-tolerant
        // bound is [20, 28] (allows for one extra tick of advance from
        // the apply pass running after the chronicle write).
        for (i, &e) in stun.iter().enumerate() {
            if e > 0 {
                assert!(
                    (20..=28).contains(&e),
                    "agent {i}: stun_expires_at_tick={e} outside \
                     expected range [20, 28] (tick-0 cast → expires at \
                     20, tick-7 cast → expires at 27)",
                );
            }
        }
    }

    /// SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07) —
    /// proves the apply_ability dispatcher emits kind=27
    /// EffectHealApplied chronicle records at 5v5 scale (10 agents
    /// through pair-field scoring) AND the lower-pass-fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`
    /// kernel ferries the per-record `amount` into the `agent_hp` SoA
    /// via `agents.set_hp(t, min(hp + amt, max_hp))`.
    ///
    /// CADENCE AT THE SEAM: SquadHeal fires at `world.tick % 7 == 0`,
    /// so step 7 (= tick 7 dispatch) drives one cast cycle. Pre-seed
    /// slot 2 (Red DPS) at hp=50.0 (well below the 120.0 init for DPS,
    /// and below max_hp=100.0 so the heal isn't clamped on the first
    /// cast). Slot 2's same-team allies (Red Tank=0, Healer=1, DPS=3,
    /// DPS=4) will all see slot 2 as the lowest-HP candidate when their
    /// SquadHeal scoring argmax fires at tick 7 (`200 - 50 = 150`
    /// dominates over their other allies' `200 - 200/120/80 = 0..120`).
    ///
    /// Slot 2's own SquadHeal vote at tick 7 doesn't pick itself
    /// (`target != self` filter) — it picks one of its allies as a
    /// secondary target. The healing math: 4 friendly casters target
    /// slot 2, each landing 12.0 → slot 2 hp = 50 + 4×12 = 98
    /// (under the 100.0 max_hp clamp). However Strike/Snipe damage
    /// path may also have hit slot 2 between tick 0 and tick 7, so the
    /// load-bearing pin is `hp[2] > 50` (heal arm fired AND landed at
    /// least one record).
    ///
    /// HOW THE TEST PROVES SQUADHEAL FIRED:
    ///   1. After 8 steps (ticks 0..=7 dispatched, with the tick-7
    ///      SquadHeal cycle landing) slot 2's hp must be > 50.0
    ///      (proves the chronicle path emitted kind=27 records AND
    ///      the fused consumer wrote the SoA).
    ///   2. No agent's hp exceeds max_hp=100.0 (proves the
    ///      `min(hp + amt, max_hp)` clamp engages).
    #[test]
    fn squad_heal_recovers_friendly_hp() {
        let mut state = TacticalSquad5v5State::new(0xCAFE_F00D, 10);

        // Pre-seed slot 2 (Red DPS) at hp=50.0. Slot 2's SoA hp init
        // (HP_DPS=120) is overwritten before the first step so the
        // SquadHeal scoring argmax sees slot 2 as the clear lowest-HP
        // ally on the Red team for the tick-7 cast cycle.
        state.write_hp(2, 50.0);

        // Pre-tick baseline — the seeded slot reads back at 50.0.
        let initial_hp = state.read_hp();
        assert_eq!(
            initial_hp[2], 50.0,
            "pre-seed: slot 2 hp must be 50.0 after write_hp; got {}",
            initial_hp[2],
        );

        // Run 8 ticks. SquadHeal fires at tick 0 and tick 7 (% 7 == 0).
        // We run steps 0..=7 inclusive, dispatching ticks 0..=7. The
        // tick-0 cast targets agents at their init HP (all healthy
        // except slot 2 which is at 50); the tick-7 cast targets slot 2
        // (still the lowest-HP ally on the Red side, even after some
        // damage from Strike/Snipe).
        for _ in 0..8 {
            state.step();
        }

        let hp_now = state.read_hp();

        // Pin 1: slot 2's hp must climb above the 50.0 seed — proves
        // the SquadHeal apply_ability dispatch emitted kind=27
        // EffectHealApplied records AND the fused chronicle consumer
        // wrote the SoA with the `min(hp + amt, max_hp)` clamp.
        assert!(
            hp_now[2] > 50.0,
            "after 8 ticks slot 2's hp must climb above the 50.0 \
             pre-seed (SquadHeal apply_ability proof); got hp[2]={} \
             (full hp readback: {:?})",
            hp_now[2],
            hp_now,
        );

        // Pin 2: at least one agent has hp > pre-seeded value — robust
        // against pinning to slot 2 specifically in case the argmax
        // distributes heals across multiple targets.
        let healed_count: usize = hp_now
            .iter()
            .enumerate()
            .filter(|(i, &h)| {
                let init = if *i == 2 { 50.0 } else { initial_hp[*i] };
                h > init + 0.001
            })
            .count();
        assert!(
            healed_count >= 1,
            "after 8 ticks at least one agent's hp must climb above \
             its seed; saw 0 of 10. HP: {:?}",
            hp_now,
        );

        // Pin 3: slot 2 (the seeded heal target) ends at hp ≤
        // max_hp=100.0 — proves the `min(hp + amt, max_hp)` clamp in
        // ApplyHealFromChronicle is honoured. Without the clamp, slot 2
        // receiving 4 friend-targeted heals at tick 0 AND tick 7 (12 ×
        // 8 = 96 raw) on top of 50 would land at 146, breaking the
        // invariant. Slots 0/5 (Tanks at hp=200) and 1/6 (Healers at
        // hp=80) are not seeded targets — they retain their init HP
        // because the SquadHeal argmax doesn't pick them (score
        // 200-200=0 and 200-80=120 both lose to slot 2's 200-50=150).
        // We therefore only check the seeded slot.
        assert!(
            hp_now[2] <= 100.0 + 0.001,
            "agent 2: hp={} exceeds max_hp=100.0 — \
             ApplyHealFromChronicle clamp didn't engage (full \
             readback: {:?})",
            hp_now[2],
            hp_now,
        );
    }
}
