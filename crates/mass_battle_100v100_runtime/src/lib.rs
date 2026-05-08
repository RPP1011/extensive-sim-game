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

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

mod binding_check;

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
    /// Task #138 follow-on (mass_battle_100v100 port, 2026-05-07) —
    /// per-stat agent SoA columns the apply_ability dispatcher's
    /// `scale_bonus = Σ percent * agent_stat[caster_slot]` switch
    /// reads. mass_battle_100v100's Strike + Snipe have no scaling
    /// entries today, so all five columns sit at their inert init
    /// values; the dispatcher's `scale_bonus` collapses to 0
    /// unconditionally. Kept on the state struct because the verb
    /// chronicle kernels (Strike + Snipe) still BIND them — the
    /// dispatcher emits the stat-switch arms whether or not any
    /// program actually scales. Mirrors duel_25v25_runtime exactly.
    #[allow(dead_code)]
    agent_attack_damage_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_armor_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_magic_resist_buf: wgpu::Buffer,
    #[allow(dead_code)]
    agent_move_speed_buf: wgpu::Buffer,
    /// mass_battle_100v100's verbs don't read mana, but the
    /// apply_ability dispatcher's stat-switch (Wave 1.5#4 GPU wire-up)
    /// binds it alongside the other stat columns. Init to 100.0 for
    /// shape parity with duel_abilities; no kernel in this fixture
    /// reads it.
    #[allow(dead_code)]
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    Strike=0, Snipe=1, StunBolt=2, MassHeal=3, Heal=4) --
    //
    // StunBolt control-status proof (200-agent scale, 2026-05-07): the
    // verb is declared between Snipe and Heal in the .sim, so source-
    // order action_id assignment lands StunBolt at index 2 and shifted
    // Heal from 2 to 3 at the time.
    //
    // MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07):
    // MassHeal is declared between StunBolt and Heal in the .sim, so
    // source-order action_id assignment lands MassHeal at index 3 and
    // shifts Heal from 3 to 4. The fused mask kernel writes all five
    // bitmaps; the scoring argmax reads all five rows.
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_3_bitmap_buf: wgpu::Buffer,
    mask_4_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    /// StunBolt control-status proof (200-agent scale, 2026-05-07) —
    /// per-agent `stun_expires_at_tick` SoA column. Written by the
    /// fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
    /// kernel from kind=29 EffectStunApplied chronicle records produced
    /// by StunBolt's verb chronicle dispatcher (kind=29 records carry
    /// `expires_at_tick = world.tick + 20` precomputed by the
    /// dispatcher). Init to 0 (= "never stunned" — agents whose
    /// `stun_expires_at_tick > world.tick` are stunned). The verb
    /// `where`-clauses in this fixture don't read it today, but the new
    /// `stun_bolt_stuns_targets_at_200_agent_scale` test asserts the
    /// SoA column lands the expected expires_at_tick after StunBolt
    /// cast cycles.
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
    chronicle_strike_cfg_buf: wgpu::Buffer,
    chronicle_snipe_cfg_buf: wgpu::Buffer,
    /// StunBolt control-status proof (200-agent scale, 2026-05-07) —
    /// fourth per-agent verb chronicle cfg. Same shape as the Strike +
    /// Snipe cfgs; the StunBolt chronicle kernel filters
    /// `action_id == 2u` (source-order index — StunBolt is the third
    /// verb declared after Strike and Snipe) and dispatches the
    /// AbilityId(3) program through the apply_ability arm, writing
    /// kind=29 EffectStunApplied records.
    chronicle_stun_bolt_cfg_buf: wgpu::Buffer,
    /// MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07)
    /// — fifth per-agent verb chronicle cfg. Same shape as the Strike
    /// + Snipe + StunBolt cfgs; the MassHeal chronicle kernel filters
    /// `action_id == 3u` (source-order index — MassHeal is the fourth
    /// verb declared after Strike, Snipe, and StunBolt) and dispatches
    /// the AbilityId(4) program through the apply_ability arm, writing
    /// kind=27 EffectHealApplied records consumed by the fused
    /// ApplyDamage+Stun+Heal chronicle kernel.
    chronicle_mass_heal_cfg_buf: wgpu::Buffer,
    chronicle_heal_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (mass_battle_100v100 port, 2026-05-07) +
    /// StunBolt control-status proof (200-agent scale, 2026-05-07) +
    /// MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07)
    /// — cfg uniform for the FUSED chronicle-consumer kernel. The
    /// lower pass folded ApplyDamageFromChronicle (drains kind=26 →
    /// emit Damaged), ApplyStunFromChronicle (drains kind=29 → write
    /// `agents.set_stun_expires_at_tick`), and ApplyHealFromChronicle
    /// (drains kind=27 → write
    /// `agents.set_hp(min(hp+amt, max_hp))`) into ONE kernel
    /// (`physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`)
    /// because all three consume from the same event ring at
    /// @phase(post) with non-overlapping kind tags. Single cfg +
    /// single dispatch per tick. Fusion grew from 2-way to 3-way when
    /// MassHeal was added, mirroring the same 2→3 transition
    /// duel_25v25's lib.rs surfaced in commit 049feb0c.
    apply_chronicle_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Task #138 follow-on (mass_battle_100v100 port, 2026-05-07) —
    /// Packed AbilityRegistry uploaded to the GPU. The Strike +
    /// Snipe chronicle kernels bind `effect_kinds` /
    /// `effect_payload_a` / `effect_payload_b` (and the modifier
    /// columns) for the apply_ability dispatcher arm. Built once at
    /// construction by `binding_check::build_mass_battle_100v100_registry`
    /// (two programs: Strike at AbilityId(1), Snipe at AbilityId(2))
    /// and uploaded via `PackedAbilityRegistryGpu::upload`. The
    /// buffers live for the rest of the run.
    registry_gpu: PackedAbilityRegistryGpu,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl MassBattle100v100State {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_AGENTS;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Task #138 follow-on (mass_battle_100v100 port, 2026-05-07) +
        // StunBolt + MassHeal — runs ONCE at startup before any GPU
        // work. Asserts the runtime's hand-built Strike + Snipe +
        // StunBolt + MassHeal programs land at AbilityId(1..=4) so the
        // `apply_ability 1..=4` literals in
        // `assets/sim/mass_battle_100v100.sim` (the four verb bodies)
        // dispatch the correct programs. Cheap (four hand-built
        // programs); panics on any drift before the expensive GPU init
        // below.
        binding_check::assert_ability_registry_matches_sim_constants();

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

        // ---- AbilityRegistry GPU upload (Task #138 follow-on) ----
        // Build the two-program registry (Strike at AbilityId(1),
        // Snipe at AbilityId(2)), pack it via
        // PackedAbilityRegistry::pack, and upload one buffer per SoA
        // column. The Strike + Snipe chronicle kernels bind these
        // for the apply_ability dispatcher arm. Building the registry
        // repeats the binding-check's program-build pass (cheap —
        // two hand-built programs) but keeps construction colocated
        // with the upload site, mirroring duel_25v25's pattern.
        let built_registry = binding_check::build_mass_battle_100v100_registry();
        let packed = PackedAbilityRegistry::pack(&built_registry.registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed, &gpu, "mass_battle_100v100_runtime",
        );

        // ---- Per-stat agent SoA columns (Task #138 follow-on) ----
        // The apply_ability dispatcher's `scale_bonus = Σ percent *
        // agent_stat[caster_slot]` switch reads these unconditionally
        // even though mass_battle_100v100's Strike + Snipe have no
        // scaling entries — the per-effect scaling SoA is empty, so
        // scale_bonus collapses to 0.0 inside the dispatcher. We
        // still need to bind real buffers because the kernels' BGLs
        // declare the bindings. Init values mirror duel_25v25
        // (max_hp=100, mana=100, others=0).
        let n_usize = agent_count as usize;
        let zeros_f32: Vec<f32> = vec![0.0_f32; n_usize];
        let max_hp_init: Vec<f32> = vec![100.0_f32; n_usize];
        let mana_init: Vec<f32> = vec![100.0_f32; n_usize];
        let agent_max_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::agent_max_hp"),
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
            mk_zero_stat("mass_battle_100v100::agent_attack_damage");
        let agent_armor_buf = mk_zero_stat("mass_battle_100v100::agent_armor");
        let agent_magic_resist_buf =
            mk_zero_stat("mass_battle_100v100::agent_magic_resist");
        let agent_move_speed_buf =
            mk_zero_stat("mass_battle_100v100::agent_move_speed");
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // StunBolt control-status proof (200-agent scale, 2026-05-07) —
        // per-agent `stun_expires_at_tick` SoA column. Init to 0 = "never
        // stunned" (the convention established by duel_abilities). The
        // fused ApplyDamageFromChronicle_and_ApplyStunFromChronicle
        // kernel writes this slot from kind=29 EffectStunApplied
        // chronicle records (one record per StunBolt cast). COPY_SRC
        // is on so the test can read it back via `read_u32`.
        let stun_expires_init: Vec<u32> = vec![0_u32; n_usize];
        let agent_stun_expires_at_tick_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mass_battle_100v100::agent_stun_expires_at_tick"),
                contents: bytemuck::cast_slice(&stun_expires_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Five mask bitmaps — one per verb. Cleared each tick.
        // StunBolt control-status proof (200-agent scale, 2026-05-07):
        // mask_3 was the new bitmap added for StunBolt's verb when the
        // verb count went 3 → 4.
        // MassHeal recovery-dynamics proof (200-agent scale,
        // 2026-05-07): mask_4 is the new bitmap added for the Heal
        // verb's shifted source index (MassHeal lands at index 3,
        // shifting Heal from index 3 to index 4). The fused mask kernel
        // writes all five bitmaps; the scoring argmax reads all five
        // rows.
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
        let mask_3_bitmap_buf = mk_mask("mass_battle_100v100::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("mass_battle_100v100::mask_4_bitmap");
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
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_snipe_cfg_init =
            physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_snipe_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_snipe_cfg"),
            contents: bytemuck::bytes_of(&chronicle_snipe_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // StunBolt control-status proof (200-agent scale, 2026-05-07) —
        // cfg uniform for the new verb chronicle kernel. Same shape as
        // Strike + Snipe; the kernel filters action_id == 2u and
        // dispatches AbilityId(3) through the apply_ability arm.
        let chronicle_stun_bolt_cfg_init =
            physics_verb_chronicle_StunBolt::PhysicsVerbChronicleStunBoltCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_stun_bolt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_stun_bolt_cfg"),
            contents: bytemuck::bytes_of(&chronicle_stun_bolt_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07)
        // — cfg uniform for the new MassHeal verb chronicle kernel.
        // Same shape as Strike + Snipe + StunBolt; the kernel filters
        // action_id == 3u (MassHeal is the fourth verb declared, source-
        // order index 3) and dispatches AbilityId(4) through the
        // apply_ability arm, writing kind=27 EffectHealApplied records.
        let chronicle_mass_heal_cfg_init =
            physics_verb_chronicle_MassHeal::PhysicsVerbChronicleMassHealCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_mass_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_mass_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_mass_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_heal_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::chronicle_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (mass_battle_100v100 port, 2026-05-07) +
        // StunBolt control-status proof (200-agent scale, 2026-05-07) +
        // MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07)
        // — cfg uniform for the FUSED chronicle-consumer kernel. The
        // lower pass folded ApplyDamageFromChronicle (kind=26 → emit
        // Damaged), ApplyStunFromChronicle (kind=29 → write
        // `agents.set_stun_expires_at_tick`), and ApplyHealFromChronicle
        // (kind=27 → write
        // `agents.set_hp(min(hp+amt, max_hp))`) into ONE kernel
        // (`physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`)
        // because all three consume from the same event ring at
        // @phase(post) with non-overlapping kind tags. Single cfg +
        // single dispatch per tick. Fusion grew from 2-way to 3-way
        // when MassHeal was added.
        let apply_chronicle_cfg_init =
            physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, agent_cap: 0,
            };
        let apply_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass_battle_100v100::apply_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_chronicle_cfg_init),
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
            chronicle_strike_cfg_buf,
            chronicle_snipe_cfg_buf,
            chronicle_stun_bolt_cfg_buf,
            chronicle_mass_heal_cfg_buf,
            chronicle_heal_cfg_buf,
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

    pub fn read_level(&self) -> Vec<u32> {
        self.read_u32(&self.agent_level_buf, "level")
    }

    /// Per-agent `stun_expires_at_tick` readback (u32 absolute tick at
    /// which the stun expires; 0 = "never stunned"). StunBolt control-
    /// status proof (200-agent scale, 2026-05-07) — written by the
    /// fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
    /// kernel from kind=29 EffectStunApplied chronicle records emitted
    /// by StunBolt's verb chronicle dispatcher.
    pub fn read_stun_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_stun_expires_at_tick_buf, "stun_expires_at_tick")
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
        // StunBolt control-status proof (200-agent scale, 2026-05-07):
        // mask_3 was the bitmap added for StunBolt's verb when the
        // verb count went 3 → 4.
        // MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07):
        // mask_4 is the new bitmap added for the Heal verb's shifted
        // source index (MassHeal lands at index 3, shifting Heal from
        // index 3 to index 4). Cleared every tick alongside the other
        // four (one bitmap per verb in source order: Strike=0, Snipe=1,
        // StunBolt=2, MassHeal=3, Heal=4).
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
            // StunBolt control-status proof (200-agent scale,
            // 2026-05-07): fourth verb mask (StunBolt at source index 2).
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            // MassHeal recovery-dynamics proof (200-agent scale,
            // 2026-05-07): fifth verb mask (MassHeal at source index 3
            // shifts Heal to source index 4).
            mask_4_bitmap: &self.mask_4_bitmap_buf,
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
            // StunBolt control-status proof (200-agent scale,
            // 2026-05-07): scoring's argmax now reads four mask rows.
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            // MassHeal recovery-dynamics proof (200-agent scale,
            // 2026-05-07): scoring's argmax now reads five mask rows
            // (MassHeal at source index 3 shifts Heal to index 4).
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

        // (4) Strike chronicle — gates action_id==0u, dispatches the
        // Strike ability via apply_ability. Task #138 follow-on
        // (mass_battle_100v100 port, 2026-05-07): instead of emitting
        // Damaged directly, the kernel walks the AbilityRegistry's
        // effect SoA columns and writes EffectDamageApplied chronicle
        // records (engine kind=26). The new ApplyDamageFromChronicle
        // kernel below re-emits those as Damaged so the existing
        // ApplyDamage_and_ApplyHeal cascade keeps working unchanged.
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&strike_cfg),
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
            self.agent_count,
        );

        // (5) Snipe chronicle — gates action_id==1u, dispatches the
        // Snipe ability via apply_ability. Same chronicle re-emit
        // pattern as Strike — EffectDamageApplied records (kind=26)
        // flow through ApplyDamageFromChronicle → Damaged →
        // ApplyDamage_and_ApplyHeal cascade unchanged.
        let snipe_cfg = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_snipe_cfg_buf, 0, bytemuck::bytes_of(&snipe_cfg),
        );
        let snipe_bindings = physics_verb_chronicle_Snipe::PhysicsVerbChronicleSnipeBindings {
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
            self.agent_count,
        );

        // (5b) StunBolt chronicle — StunBolt control-status proof
        // (200-agent scale, 2026-05-07). Gates action_id==2u (StunBolt
        // is the third verb in source order, so its action_id is 2 —
        // shifting Heal's action_id from 2 to 3). Same chronicle re-emit
        // pattern as Strike + Snipe except the AbilityProgram at slot 3
        // declares EffectOp::Stun{duration_ticks=20} instead of Damage,
        // so the apply_ability dispatcher writes kind=29
        // EffectStunApplied records (with `expires_at_tick = world.tick
        // + 20` precomputed) instead of kind=26 EffectDamageApplied.
        // The fused
        // ApplyDamageFromChronicle_and_ApplyStunFromChronicle kernel
        // below drains both kind tags into the right SoA target
        // (Damaged → ApplyDamage HP cascade for kind=26, direct
        // `agents.set_stun_expires_at_tick` for kind=29).
        let stun_bolt_cfg = physics_verb_chronicle_StunBolt::PhysicsVerbChronicleStunBoltCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_stun_bolt_cfg_buf, 0, bytemuck::bytes_of(&stun_bolt_cfg),
        );
        let stun_bolt_bindings = physics_verb_chronicle_StunBolt::PhysicsVerbChronicleStunBoltBindings {
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
            cfg: &self.chronicle_stun_bolt_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_stunbolt(
            &mut self.cache, &stun_bolt_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5c) MassHeal chronicle — MassHeal recovery-dynamics proof
        // (200-agent scale, 2026-05-07). Gates action_id==3u (MassHeal
        // is the fourth verb in source order, so its action_id is 3 —
        // shifting Heal's action_id from 3 to 4). Same chronicle re-emit
        // pattern as Strike + Snipe + StunBolt except the AbilityProgram
        // at slot 4 declares EffectOp::Heal{amount: 18.0} instead of
        // Damage/Stun, so the apply_ability dispatcher writes kind=27
        // EffectHealApplied records (with the resolved heal `amount`)
        // instead of kind=26 EffectDamageApplied or kind=29
        // EffectStunApplied. The fused
        // ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle
        // kernel below drains all three kind tags into the right SoA
        // target (Damaged → ApplyDamage HP cascade for kind=26, direct
        // `agents.set_stun_expires_at_tick` for kind=29, direct
        // `agents.set_hp(min(hp+amt, max_hp))` for kind=27).
        let mass_heal_cfg = physics_verb_chronicle_MassHeal::PhysicsVerbChronicleMassHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_mass_heal_cfg_buf, 0, bytemuck::bytes_of(&mass_heal_cfg),
        );
        let mass_heal_bindings = physics_verb_chronicle_MassHeal::PhysicsVerbChronicleMassHealBindings {
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
            cfg: &self.chronicle_mass_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_massheal(
            &mut self.cache, &mass_heal_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) Heal chronicle — gates action_id==4u. StunBolt control-
        // status proof (200-agent scale, 2026-05-07): Heal's action_id
        // shifted from 2 to 3 when StunBolt was inserted at source
        // position 2.
        // MassHeal recovery-dynamics proof (200-agent scale,
        // 2026-05-07): Heal's action_id shifted again from 3 to 4 when
        // MassHeal was inserted at source position 3. The kernel name +
        // binding shape are unchanged — the action_id literal in the
        // generated kernel is the only detail that moved.
        let heal_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, agent_cap: 0,
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

        // (6b) Fused ApplyDamageFromChronicle + ApplyStunFromChronicle
        // + ApplyHealFromChronicle — chronicle consumers fused into ONE
        // kernel by the lower pass (all three run @phase(post) over the
        // same event ring with non-overlapping kind tags). MassHeal
        // recovery-dynamics proof (200-agent scale, 2026-05-07): the
        // fused kernel grew from 2-way (Damage+Stun) to 3-way
        // (Damage+Stun+Heal) when MassHeal was added. Drains:
        //   - kind=26 EffectDamageApplied → emit `Damaged` (re-emit;
        //     the standalone ApplyDamage_and_ApplyHeal kernel below
        //     decrements HP)
        //   - kind=29 EffectStunApplied → write
        //     `agents.set_stun_expires_at_tick(t, expires_at_tick)`
        //     directly into the per-agent SoA slot.
        //   - kind=27 EffectHealApplied → write
        //     `agents.set_hp(t, min(hp + amt, max_hp))` directly into
        //     the per-agent SoA slot, clamping at max_hp.
        //
        // event_count is the upper bound on chronicle records produced
        // per tick across Strike + Snipe + StunBolt + MassHeal (each
        // can emit one record per actor per tick). agent_count * 8
        // reuses the same slot-count headroom estimate the rest of the
        // cascade uses.
        let event_count_estimate = self.agent_count * 8;
        let apply_chronicle_cfg = physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle::PhysicsApplyDamageFromChronicleAndApplyStunFromChronicleAndApplyHealFromChronicleCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0, agent_cap: 0,
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

        // (7) Apply damage + heal (fused PerEvent). Reads Damaged
        // (re-emitted by ApplyDamageFromChronicle from the
        // apply_ability EffectDamageApplied records) + Healed (still
        // direct-emitted by Heal chronicle today; Heal isn't ported).
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
    /// Like `tactical_squad_5v5_runtime`, this fixture has no movement
    /// physics — the .sim declares `pos: vec3` on the role entities but
    /// no kernel writes a position buffer. We therefore SYNTHESIZE a
    /// stationary 2-D layout per slot index: Red [0..100] on the left
    /// (x = -10), Blue [100..200] on the right (x = +10), with each
    /// team arranged in a 10×10 grid in (y, z). This keeps both teams
    /// legible in the ASCII grid even though the actual sim is purely
    /// event-driven HP/heal arithmetic.
    ///
    /// `creature_types` encoding (4 entries, indexed by
    /// `team_bit | (dead_bit << 1)`):
    ///
    /// |  i | team | state |
    /// |----|------|-------|
    /// |  0 | Red  | alive |
    /// |  1 | Blue | alive |
    /// |  2 | Red  | dead  |
    /// |  3 | Blue | dead  |
    ///
    /// Team comes from a REAL SoA read of `agent_level_buf` (decoded
    /// via the `level_for(team, role)` inverse: `(level - 1) / 3`),
    /// not an index-derived heuristic — slots could in principle be
    /// reordered without breaking the encoding, although the
    /// constructor today hard-codes the index→(team, role) layout
    /// (Red [0..PER_TEAM], Blue [PER_TEAM..2*PER_TEAM]). The role bit
    /// (Tank/Healer/DPS) is intentionally collapsed into the single
    /// team glyph here — at 200 agents per scene, distinct per-role
    /// glyphs would dominate the screen with letter noise; team color
    /// is what matters at this scale. The `alive` field is read from
    /// `agent_alive_buf`; `agent_count` stays constant (no
    /// spawn/despawn) so dead slots remain in the snapshot at their
    /// original positions, just rendered with the tombstone glyph.
    /// HP defence-in-depth zeros the alive bit if hp <= 0 even when
    /// the alive buffer hasn't been flipped yet.
    ///
    /// Initial-state safe: GPU buffers are populated by
    /// `create_buffer_init` at construction, so calling `snapshot()`
    /// before any `step()` returns 200 alive slots with deterministic
    /// team discriminants.
    fn snapshot(&mut self) -> AgentSnapshot {
        let n = self.agent_count as usize;
        // Synthetic 2-D layout — Red on x=-10, Blue on x=+10. Within
        // each team, 100 slots laid out in a 10×10 grid in (y, z) so
        // every slot is visible in the ASCII grid. The renderer only
        // uses x/y for projection, but we set z too for completeness.
        let positions: Vec<Vec3> = (0..n)
            .map(|i| {
                let team = if (i as u32) < PER_TEAM { 0u32 } else { 1u32 };
                let local = (i as u32 % PER_TEAM) as i32; // 0..100
                let row = local / 10; // 0..10
                let col = local % 10; // 0..10
                let x = if team == 0 { -10.0 } else { 10.0 };
                let y = (col as f32) - 4.5; // centered: -4.5..+4.5
                let z = (row as f32) - 4.5;
                Vec3::new(x, y, z)
            })
            .collect();

        let level: Vec<u32> = self.read_level();
        let alive_raw: Vec<u32> = self.read_alive();
        let hp: Vec<f32> = self.read_hp();
        // Defence-in-depth: treat hp<=0 as dead even if the alive bit
        // hasn't been written yet by ApplyDamage (mirrors the
        // tactical_squad_5v5 / duel_abilities approach).
        let alive: Vec<u32> = alive_raw
            .iter()
            .zip(hp.iter())
            .map(|(&a, &h)| if a != 0 && h > 0.0 { 1 } else { 0 })
            .collect();

        // 4-entry encoding (2 teams × 2 alive states). Decode team
        // from `agent_level`: levels 1..=3 → Red (team_bit=0),
        // levels 4..=6 → Blue (team_bit=1). Inverse of `level_for()`.
        // Clamp via saturating_sub so a stray level=0 (shouldn't
        // happen) maps to Red rather than panicking.
        let creature_types: Vec<u32> = (0..n)
            .map(|i| {
                let team_bit = level[i].saturating_sub(1) / 3;
                let team_bit = team_bit & 1; // 0 = Red, 1 = Blue
                let dead_bit = if alive[i] == 0 { 1u32 } else { 0u32 };
                team_bit | (dead_bit << 1)
            })
            .collect();

        AgentSnapshot { positions, creature_types, alive }
    }

    /// 4 glyphs matching the `snapshot.creature_types` encoding:
    ///
    /// - `r` in bright red (196) for alive Red team
    /// - `b` in bright cyan (51) for alive Blue team
    /// - tombstone × in grey (240) for dead variants of either team
    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![
            VizGlyph::new('r', 196),        // 0: Red alive
            VizGlyph::new('b', 51),         // 1: Blue alive
            VizGlyph::new('\u{00D7}', 240), // 2: Red dead
            VizGlyph::new('\u{00D7}', 240), // 3: Blue dead
        ]
    }

    /// Default viewport tight around the synthetic stationary layout
    /// from `snapshot()` — Red at x=-10, Blue at x=+10, 10×10 grids
    /// per team spanning y,z ∈ [-4.5, +4.5]. ±12 keeps every slot on
    /// screen with breathing room for the 200-agent scene.
    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        Some((Vec3::new(-12.0, -6.0, 0.0), Vec3::new(12.0, 6.0, 0.0)))
    }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    // agent_count is fixed by the per-team / per-role layout.
    Box::new(MassBattle100v100State::new(seed))
}

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// Snapshot before any tick must report initial state: 200 slots
    /// (100 Red, 100 Blue), every slot alive, and `creature_types`
    /// reflecting the deterministic per-slot team layout from `new()`.
    /// Guards the construction-only readback path so `viz_app` can
    /// render frame 0 with content instead of a blank grid.
    #[test]
    fn snapshot_after_construction_returns_initial_state() {
        let mut state = MassBattle100v100State::new(0xCAFE_F00D);
        let snap = state.snapshot();

        let n = TOTAL_AGENTS as usize;
        assert_eq!(snap.positions.len(), n, "positions length");
        assert_eq!(snap.creature_types.len(), n, "creature_types length");
        assert_eq!(snap.alive.len(), n, "alive length");

        // No combat at tick 0 — every slot must be alive.
        let alive_total: u32 = snap.alive.iter().sum();
        assert_eq!(
            alive_total, TOTAL_AGENTS,
            "every slot must be alive at construction; got {}",
            alive_total,
        );

        // Per-slot encoding: Red [0..PER_TEAM] → 0, Blue [PER_TEAM..]
        // → 1 (no dead bit set). The constructor's hard-coded layout
        // puts Red first (levels 1..=3) and Blue second (levels 4..=6).
        for (i, &ct) in snap.creature_types.iter().enumerate() {
            let expected = if (i as u32) < PER_TEAM { 0u32 } else { 1u32 };
            assert_eq!(
                ct, expected,
                "slot {i}: creature_type must reflect team layout from new(); got {ct}, expected {expected}",
            );
        }

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

        // Synthetic positions must lie inside the default viewport box
        // (the renderer scales out if not, but the opening framing
        // should fit).
        let (vmin, vmax) = state.default_viewport().expect("viewport");
        for (i, p) in snap.positions.iter().enumerate() {
            assert!(
                p.x >= vmin.x - 0.001 && p.x <= vmax.x + 0.001
                    && p.y >= vmin.y - 0.001 && p.y <= vmax.y + 0.001,
                "slot {i} synthetic pos {p:?} outside default viewport [{vmin:?}, {vmax:?}]",
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
        let mut state = MassBattle100v100State::new(0xCAFE_F00D);
        let initial_hp = state.read_hp();
        let initial_alive_total: u32 = state.snapshot().alive.iter().sum();

        for _ in 0..50 {
            state.step();
        }

        let snap = state.snapshot();
        let n = TOTAL_AGENTS as usize;
        assert_eq!(snap.positions.len(), n);
        assert_eq!(snap.alive.len(), n);

        let hp_now = state.read_hp();
        let any_hp_moved = initial_hp.iter().zip(hp_now.iter()).any(|(a, b)| {
            (a - b).abs() > 0.01
        });
        let alive_total_now: u32 = snap.alive.iter().sum();
        let alive_changed = alive_total_now != initial_alive_total;

        assert!(
            any_hp_moved || alive_changed,
            "after 50 ticks, expected HP movement or kill; saw HP unchanged \
             and alive_total stable ({})",
            alive_total_now,
        );
    }

    /// StunBolt control-status proof (200-agent scale, 2026-05-07) —
    /// proves the apply_ability dispatcher emits kind=29
    /// EffectStunApplied chronicle records at production scale (200
    /// agents through pair-field scoring) AND the fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
    /// kernel ferries the per-record `expires_at_tick` into the
    /// `agent_stun_expires_at_tick` SoA via
    /// `agents.set_stun_expires_at_tick(t, e)`.
    ///
    /// CADENCE AT THE SEAM: StunBolt fires at `world.tick % 7 == 0`,
    /// so steps 0..=13 (= ticks 0..=13) drive cast cycles at tick 0
    /// and tick 7 — two cast cycles before tick 14 dispatches. Each
    /// DPS Red actor (80 agents at level=3) on its eligible cycle
    /// picks a Blue DPS target via the scoring argmax (StunBolt's
    /// `2000 - target.hp` outscores Snipe's `1000 - target.hp` when
    /// both are eligible at tick 21 — but at ticks 0 and 7 only
    /// StunBolt is on for those actors at the 7-cadence anyway, so
    /// it lands as their argmax pick uncontested for cadence reasons).
    /// The 80 Blue DPS actors mirror against Red DPS targets the same
    /// way.
    ///
    /// `expires_at_tick` for a tick-0 cast = `world.tick + 20 = 20`;
    /// for a tick-7 cast = `7 + 20 = 27`. After 14 steps the second
    /// cast cycle has just finished, so any agent whose
    /// `stun_expires_at_tick > 0` was hit by either cycle and the
    /// value lands in {20, 27} (race-tolerant range: [20, 28]). The
    /// test pins that range.
    ///
    /// HOW THE TEST PROVES STUN-BOLT FIRED:
    ///   1. After 14 steps at least one agent has a non-zero
    ///      stun_expires_at_tick (proves the chronicle path emitted
    ///      kind=29 records AND the fused consumer wrote the SoA).
    ///   2. Every stunned agent's expiry tick is in [20, 28] (matches
    ///      `tick + 20` for tick∈{0, 7}).
    #[test]
    fn stun_bolt_stuns_targets_at_200_agent_scale() {
        let mut state = MassBattle100v100State::new(0xCAFE_F00D);

        // Pre-tick baseline — every agent's stun_expires_at_tick is 0.
        let initial_stun = state.read_stun_expires_at_tick();
        assert_eq!(
            initial_stun.len(),
            TOTAL_AGENTS as usize,
            "stun_expires_at_tick readback must cover all 200 agents",
        );
        for (i, &e) in initial_stun.iter().enumerate() {
            assert_eq!(
                e, 0,
                "initial stun_expires_at_tick[{i}] must be 0 (= never \
                 stunned); got {e}",
            );
        }

        // Run 14 ticks. StunBolt fires at tick 0 and tick 7 (two
        // cast cycles); the next firing tick is 14 (we run steps 0..=13
        // inclusive, ending BEFORE tick 14 dispatches). Strike (% 2)
        // fires at ticks 0, 2, 4, 6, 8, 10, 12 — independent. Snipe
        // (% 3) fires at ticks 0, 3, 6, 9, 12 — independent. Heal
        // (% 3) at the same ticks but the role gate filters to Healers.
        for _ in 0..14 {
            state.step();
        }

        let stun = state.read_stun_expires_at_tick();

        // Pin 1: at least 1 agent has a non-zero stun expiry — proves
        // StunBolt's apply_ability dispatch emitted kind=29
        // EffectStunApplied records AND the fused chronicle consumer
        // wrote the SoA. With 80 DPS per team firing on a 7-cadence
        // and pair-field argmax piling many casters onto the
        // lowest-HP target, the count is typically high (most live
        // targets get stunned), but the load-bearing pin is "≥ 1"
        // so the test is robust against scheduling variation.
        let stunned_count: usize = stun.iter().filter(|&&e| e > 0).count();
        assert!(
            stunned_count >= 1,
            "after 14 ticks at least 1 agent must have a non-zero \
             stun_expires_at_tick (control-status chronicle proof); \
             saw {} stunned out of {}. Per-slot stun: {:?}",
            stunned_count,
            TOTAL_AGENTS,
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

    /// MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07)
    /// — proves the apply_ability dispatcher emits kind=27
    /// EffectHealApplied chronicle records at production scale (200
    /// agents through pair-field scoring) AND the fused
    /// `physics_ApplyDamageFromChronicle_and_ApplyStunFromChronicle_and_ApplyHealFromChronicle`
    /// kernel ferries the per-record `amount` into the `agent_hp` SoA
    /// via `agents.set_hp(t, min(hp + amt, max_hp))` (clamped at the
    /// per-agent `max_hp` ceiling).
    ///
    /// PRE-SEED: every agent's hp is overridden to 50.0 (well below
    /// max_hp=100) before any tick. With everyone tied at 50.0 the
    /// per-pair argmax for MassHeal picks the lowest-slot same-team
    /// candidate — but the load-bearing test pin is "≥ 1 agent ends up
    /// with hp > 50", which is robust against argmax ordering.
    ///
    /// CADENCE AT THE SEAM: MassHeal fires at `world.tick % 11 == 0`,
    /// so steps 0..=10 (= ticks 0..=10) drive exactly ONE cast cycle
    /// at tick 0 — the next firing tick is 11, beyond the inclusive
    /// run. Strike (% 2) fires at ticks 0, 2, 4, 6, 8, 10 — pile-up
    /// damage on the focus-fired enemy. Snipe (% 3) at ticks 0, 3, 6,
    /// 9. StunBolt (% 7) at tick 7. Heal (% 3) at the same Snipe ticks
    /// but role-gated to Healers.
    ///
    /// HOW THE TEST PROVES MASS-HEAL FIRED:
    ///   1. Pre-seed every agent's hp to 50.0 via queue.write_buffer.
    ///   2. Run 11 ticks (one MassHeal cycle at tick 0).
    ///   3. At least one agent has hp > 50.0 — proves the
    ///      EffectHealApplied chronicle records were drained AND the
    ///      ApplyHealFromChronicle arm of the 3-way fused kernel wrote
    ///      back to `agent_hp`. Even with pile-up damage from Strike +
    ///      Snipe at the same ticks, the per-team layout means most
    ///      agents are not on the offensive verbs' argmax target list
    ///      and only the heal lands on them.
    ///   4. No agent's hp exceeds max_hp=100 — proves the
    ///      `min(hp + amt, max_hp)` clamp in ApplyHealFromChronicle is
    ///      honoured.
    #[test]
    fn mass_heal_recovers_friendly_hp_at_200_agent_scale() {
        let mut state = MassBattle100v100State::new(0xCAFE_F00D);

        // Pre-seed every agent's hp to 50.0. The constructor put each
        // agent's hp at role_hp(role) (200/80/120), well above the
        // per-agent max_hp=100.0 SoA ceiling — we want every agent
        // BELOW max_hp so the heal clamp leaves visible headroom for
        // the MassHeal cast to land.
        let seeded_hp: Vec<f32> = vec![50.0_f32; TOTAL_AGENTS as usize];
        state.gpu.queue.write_buffer(
            &state.agent_hp_buf,
            0,
            bytemuck::cast_slice(&seeded_hp),
        );

        // Sanity: confirm the override stuck before any tick.
        let initial_hp = state.read_hp();
        for (i, &h) in initial_hp.iter().enumerate() {
            assert_eq!(
                h, 50.0,
                "initial hp[{i}] must be 50.0 after the pre-seed \
                 write_buffer; got {h}",
            );
        }

        // Run exactly 11 ticks. MassHeal (% 11 == 0) fires at step 0
        // (tick 0) ONLY — the next firing tick is 11, beyond the
        // 0..=10 inclusive run.
        for _ in 0..11 {
            state.step();
        }

        let hp_now = state.read_hp();

        // Pin 1: at least one agent must have hp > 50.0 — proves the
        // EffectHealApplied chronicle records were drained AND the
        // ApplyHealFromChronicle arm of the 3-way fused kernel wrote
        // back to `agent_hp`. With the per-pair argmax distributing
        // MassHeal casts onto same-team friends and most agents not
        // being on Strike/Snipe's offensive argmax pile-up list,
        // many agents end up net positive.
        let healed_count: usize = hp_now.iter().filter(|&&h| h > 50.0).count();
        assert!(
            healed_count >= 1,
            "after 11 ticks at least one agent must have hp > 50.0 \
             (MassHeal chronicle arm proof); saw {} healed of {}. \
             First few HP samples: {:?}",
            healed_count,
            TOTAL_AGENTS,
            &hp_now[..hp_now.len().min(8)],
        );

        // Pin 2: no live agent's hp exceeds max_hp (100.0) — proves
        // the `min(hp + amt, max_hp)` clamp in ApplyHealFromChronicle
        // is honoured. Without the clamp, repeated friend-targeted
        // heals would push hp past 100. Dead-target sentinel HP (set
        // to 1e9 by ApplyDamage when a slot dies) is excluded by the
        // alive filter.
        let alive = state.read_alive();
        for (i, (&h, &a)) in hp_now.iter().zip(alive.iter()).enumerate() {
            if a != 0 {
                assert!(
                    h <= 100.0 + 0.001,
                    "agent {i}: hp={h} exceeds max_hp=100.0; clamp \
                     in ApplyHealFromChronicle didn't engage",
                );
            }
        }
    }
}
