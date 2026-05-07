//! Per-fixture runtime for `assets/sim/duel_abilities.sim` — Wave 1
//! acceptance fixture for the .ability DSL pipeline.
//!
//! ## Why this fixture exists
//!
//! Wave 1.0 + 1.6 + 1.7 + 1.9 landed:
//!   - `dsl_ast::parse_ability_file` (parser)
//!   - `dsl_compiler::ability_lower` (AST → AbilityProgram lowering)
//!   - `dsl_compiler::ability_registry::build_registry` (cross-file
//!      AbilityRegistry assembly + cast resolution)
//!   - `engine::ability::PackedAbilityRegistry::pack` (SoA repacking
//!      for GPU consumption)
//!
//! What WASN'T landed: a kernel-emit path that lets a compiled .sim
//! kernel actually consume the `PackedAbilityRegistry` storage buffer
//! and dispatch ability values dynamically. There is no engine-wide
//! cast cascade today. (Per `crates/engine/src/ability/mod.rs:11-15`,
//! the docs claim cast dispatch is "compiler-emitted from
//! `assets/sim/physics.sim`" — that file does not exist.) Hooking
//! kernels up to the registry is Wave 2+ work.
//!
//! So Wave 1's *real* acceptance is a **binding test**: prove the
//! `.ability` files flow through parser → lowering → registry → packed
//! buffer with values that MATCH the corresponding hand-authored
//! `.sim` verb constants. The runtime then runs the duel using the
//! hand-mirrored .sim constants. The binding assertion at
//! `assert_ability_registry_matches_sim_constants` is the proof that
//! the lowering pipeline is correct end-to-end.
//!
//! ## Tick chain (mirror of duel_1v1)
//!
//! Two `Combatant : Agent` entities (Hero A vs Hero B) with five
//! abilities (Strike, ShieldUp, Mend, Bleed, Reap). Per-tick:
//!
//!   1. clear_tail + clear 5 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_Strike — PerPair, writes mask_0 (Strike,
//!      cooldown=10), mask_1 (ShieldUp, cooldown=40 + self.hp<90),
//!      mask_2 (Mend, cooldown=30 + self HP < 50), mask_3 (Bleed,
//!      cooldown=50 + self.hp > 50), mask_4 (Reap, cooldown=20 +
//!      target.hp < 20). Kernel name stays `fused_mask_verb_Strike` —
//!      the compiler names fused mask kernels after the first verb in
//!      source order, not all verbs.
//!   3. scoring — PerAgent argmax over the 5 competing rows
//!   4. physics_verb_chronicle_Strike   — gates action_id==0u, emits Damaged
//!   5. physics_verb_chronicle_ShieldUp — gates action_id==1u, emits Shielded
//!   6. physics_verb_chronicle_Mend     — gates action_id==2u, emits Healed
//!   7. physics_verb_chronicle_Bleed    — gates action_id==3u, emits
//!      Damaged{source=self,target=self,amount=5}. Reuses the existing
//!      ApplyDamage chronicle (no new physics block); shield_hp
//!      absorbs first, then bleed-through hits hp.
//!   8. physics_verb_chronicle_Reap     — gates action_id==4u, emits
//!      Defeated{combatant=target}. Wave 2 piece N Execute E2E demo;
//!      conditional-emit gated by the verb's `target.hp < threshold`
//!      `when` clause. Drained by the new ApplyDefeat physics block,
//!      which the compiler fuses into the existing PerEvent group →
//!      kernel renamed `physics_ApplyDamage_and_ApplyHeal_and_ApplyShield_and_ApplyDefeat`.
//!   9. physics_ApplyDamage_and_ApplyHeal_and_ApplyShield_and_ApplyDefeat —
//!      fused PerEvent kernel that reads Damaged/Healed/Shielded/Defeated
//!      events. ApplyDamage's hp<=0 branch still emits Defeated INLINE
//!      and calls `set_alive(t, false)`; ApplyDefeat handles
//!      Reap-emitted Defeated events the same way. Both paths
//!      idempotently set alive=false — no write conflict.
//!  10. seed_indirect_0
//!  11. fold_damage_dealt
//!  12. fold_healing_done
//!
//! ## Shield modelling note
//!
//! There is no `shield_hp` SoA field or `set_shield` setter in the
//! engine today (`crates/dsl_compiler/src/cg/lower/physics.rs`
//! `agents_setter_field` recognises hp/alive/mana/hunger only).
//! Adding one is engine work that belongs in a later wave. The
//! ShieldUp chronicle in `assets/sim/duel_abilities.sim` therefore
//! emits a distinct `Shielded` event but applies it as +HP via the
//! existing `set_hp` setter — semantically a heal, but with its own
//! event kind so the chronicle topology mirrors what a real shield
//! handler will look like once `shield_hp` lands. The binding-check
//! still asserts the .ability lowered to `EffectOp::Shield(50.0)`;
//! only the .sim's runtime *behaviour* is shield-as-buffer-hp.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

mod binding_check;

/// Per-fixture state for the duel.
pub struct DuelAbilitiesState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    /// Mana stays at 100.0 — no verb in this fixture gates on or
    /// reads mana, so the generated mask/scoring kernels do NOT bind
    /// it. The buffer is kept on the state struct for parity with
    /// duel_1v1's interface (and so a future fixture extending mana
    /// gates wires through cleanly), hence `#[allow(dead_code)]`.
    #[allow(dead_code)]
    agent_mana_buf: wgpu::Buffer,
    /// Per-agent shield HP — written by ApplyShield (Shielded event
    /// handler), read back via the agent_shield_hp() getter for
    /// observability. Starts at 0.0 for every slot.
    agent_shield_hp_buf: wgpu::Buffer,
    /// Per-agent lifesteal fraction (q8 fixed-point, 128 == 0.5x).
    /// Written by ApplyLifestealActivation (SetLifesteal event handler);
    /// read by ApplyDamage to decide whether the source heals back a
    /// fraction of the damage they dealt. Starts at 0 (no lifesteal).
    /// Wave 2 piece N LifeSteal E2E demo.
    agent_lifesteal_frac_q8_buf: wgpu::Buffer,
    /// Per-agent lifesteal expiry tick stamp. ApplyDamage gates on
    /// `expires_at > world.tick`, so a 0 expiry never grants lifesteal.
    /// Written by ApplyLifestealActivation alongside frac_q8.
    agent_lifesteal_expires_at_tick_buf: wgpu::Buffer,
    /// Per-agent damage_taken multiplier (q8 fixed-point, 256 == 1.0×).
    /// Written by ApplyDamageModActivation (SetDamageMod event handler);
    /// read by ApplyDamage to scale incoming bleed-through damage by
    /// `mult_q8 / 256`. Initialised to 256 (1.0×) so the buff branch
    /// is a no-op while inactive — the per-tick `expires_at > world.tick`
    /// gate keeps an EXPIRED window from applying. Wave 2 piece N
    /// DamageModify E2E demo.
    agent_damage_taken_mult_q8_buf: wgpu::Buffer,
    /// Per-agent damage_taken_mult expiry tick stamp. ApplyDamage gates
    /// on `expires_at > world.tick`, so a 0 expiry never scales damage
    /// even though the default mult_q8 is 256 (1.0×). Written by
    /// ApplyDamageModActivation alongside mult_q8.
    agent_damage_taken_mult_expires_at_tick_buf: wgpu::Buffer,
    /// Per-agent stun expiry tick. Wave 2 piece N — first cast-gating
    /// status SoA in this fixture. Written by ApplyStun (Stunned event
    /// handler emitted by Daze); read by EVERY offensive verb's mask
    /// kernel via `agents.stun_expires_at_tick(self) <= world.tick`.
    /// Initialised to 0 (= "never stunned"), so the gate is a no-op
    /// until Daze fires. Buffer is bound by both the fused mask kernel
    /// (read-side) and the fused PerEvent kernel (write-side via the
    /// ApplyStun arm).
    agent_stun_expires_at_tick_buf: wgpu::Buffer,
    /// Wave 1.5#4 GPU wire-up: per-stat columns the dispatcher reads at
    /// `caster_slot` for the `scale_bonus = Σ percent * caster_stat`
    /// computation. Bleed's `+5% max_hp` lands here — `agent_max_hp[i]`
    /// is seeded to 100.0 so a +5% scaling produces +5.0 on the
    /// chronicle write (= 5 + 5 = 10 base+scaled). The other four
    /// (attack_damage / armor / magic_resist / move_speed) are
    /// 0-initialized — duel_abilities verbs scale only on max_hp today.
    agent_attack_damage_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order: Strike=0,
    //    ShieldUp=1, Mend=2, Bleed=3, Reap=4, Vampirize=5, Fortify=6,
    //    Daze=7) --
    mask_0_bitmap_buf: wgpu::Buffer, // Strike
    mask_1_bitmap_buf: wgpu::Buffer, // ShieldUp
    mask_2_bitmap_buf: wgpu::Buffer, // Mend
    mask_3_bitmap_buf: wgpu::Buffer, // Bleed (Wave 2 SelfDamage demo)
    mask_4_bitmap_buf: wgpu::Buffer, // Reap  (Wave 2 Execute demo)
    mask_5_bitmap_buf: wgpu::Buffer, // Vampirize (Wave 2 LifeSteal demo)
    mask_6_bitmap_buf: wgpu::Buffer, // Fortify   (Wave 2 DamageModify demo)
    mask_7_bitmap_buf: wgpu::Buffer, // Daze      (Wave 2 Stun E2E demo + cast-gate)
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
    /// Cfg uniform for the FUSED kernel that drains
    /// Healed/Shielded/Defeated/SetLifesteal/SetDamageMod events AND
    /// emits Damaged from the Strike chronicle. Adding
    /// ApplyDamageModActivation grew the fusion to:
    /// `physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_
    /// ApplyLifestealActivation_and_ApplyDamageModActivation_and_
    /// verb_chronicle_Strike`. Field name `chronicle_strike_cfg_buf`
    /// retained for continuity — Strike's chronicle still needs an
    /// event_count uniform and this is the kernel that runs it.
    chronicle_strike_cfg_buf: wgpu::Buffer,
    chronicle_shieldup_cfg_buf: wgpu::Buffer,
    chronicle_mend_cfg_buf: wgpu::Buffer,
    chronicle_bleed_cfg_buf: wgpu::Buffer,
    chronicle_reap_cfg_buf: wgpu::Buffer,
    chronicle_vampirize_cfg_buf: wgpu::Buffer,
    chronicle_fortify_cfg_buf: wgpu::Buffer,
    /// Cfg uniform for the Daze chronicle — Wave 2 piece N Stun E2E
    /// demo. Standalone PerAgent kernel: emits Stunned events drained
    /// by the ApplyStun arm of the fused PerEvent kernel below.
    chronicle_daze_cfg_buf: wgpu::Buffer,
    /// Cfg uniform for the standalone `physics_ApplyDamage` kernel —
    /// split out of the previous PerEvent fusion because ApplyDamage
    /// now emits Healed events for source-side lifesteal and that
    /// production conflicts with the consumers in the same fusion
    /// group. With the DamageModify demo it now also reads the target's
    /// `damage_taken_mult_q8` + `damage_taken_mult_expires_at_tick`
    /// SoA fields (added to its bind group).
    apply_damage_cfg_buf: wgpu::Buffer,
    /// Task #138 — Cfg uniform for the new
    /// `physics_ApplyDamageFromChronicle` kernel that translates
    /// `EffectDamageApplied` records (kind=26, written by the
    /// apply_ability dispatcher in the fused kernel) back into
    /// `Damaged` events (kind=1) so the existing ApplyDamage cascade
    /// keeps working unchanged.
    apply_damage_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on — Cfg uniform for the new
    /// `physics_ApplyShieldFromChronicle` kernel that translates
    /// `EffectShieldApplied` records (kind=28, written by the
    /// apply_ability dispatcher in the standalone ShieldUp chronicle
    /// kernel) back into `Shielded` events so the existing ApplyShield
    /// cascade keeps working unchanged.
    apply_shield_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on — Cfg uniform for the new
    /// `physics_ApplyHealFromChronicle` kernel that translates
    /// `EffectHealApplied` records (kind=27, written by the
    /// apply_ability dispatcher in the standalone Mend chronicle
    /// kernel) back into `Healed` events so the existing ApplyHeal
    /// cascade keeps working unchanged.
    apply_heal_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (Daze) — Cfg uniform for the new
    /// `physics_ApplyStunFromChronicle` kernel that translates
    /// `EffectStunApplied` records (kind=29, written by the
    /// apply_ability dispatcher in the standalone Daze chronicle
    /// kernel) back into `Stunned` events so the existing ApplyStun
    /// cascade keeps working unchanged. Same shape as
    /// ApplyDamageFromChronicle / ApplyShieldFromChronicle: PerEvent
    /// + emit-only kernel, no AgentField writes (so no P6 trip).
    apply_stun_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (Bleed, 2026-05-06) — Cfg uniform for the
    /// new `physics_ApplyDamageFromSelfDamageChronicle` kernel that
    /// translates `EffectSelfDamageApplied` records (kind=39, written
    /// by the apply_ability dispatcher in the standalone Bleed
    /// chronicle kernel) back into `Damaged` events so the existing
    /// ApplyDamage cascade (shield_hp absorption, lifesteal, damage-
    /// modify) keeps working unchanged. Same shape as
    /// ApplyDamageFromChronicle: PerEvent + emit-only kernel, no
    /// AgentField writes (so no P6 trip).
    apply_damage_from_self_damage_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (Vampirize, mirror of Bleed at `486eb08f`)
    /// — Cfg uniform for the new `physics_ApplyLifestealFromChronicle`
    /// kernel that translates `EffectLifeStealApplied` records
    /// (kind=40, written by the apply_ability dispatcher in the
    /// standalone Vampirize chronicle kernel) back into `SetLifesteal`
    /// events so the existing ApplyLifestealActivation cascade (which
    /// writes the per-agent lifesteal SoA fields) keeps working
    /// unchanged. Same shape as ApplyDamageFromSelfDamageChronicle:
    /// PerEvent + emit-only kernel, no AgentField writes (so no P6
    /// trip).
    apply_lifesteal_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (Fortify, mirror of Vampirize at `60115f64`)
    /// — Cfg uniform for the new `physics_ApplyDamageModFromChronicle`
    /// kernel that translates `EffectDamageModifyApplied` records
    /// (kind=41, written by the apply_ability dispatcher in the
    /// standalone Fortify chronicle kernel) back into `SetDamageMod`
    /// events so the existing ApplyDamageModActivation cascade (which
    /// writes the per-agent damage_taken_mult SoA fields) keeps working
    /// unchanged. Same shape as ApplyLifestealFromChronicle: PerEvent
    /// + emit-only kernel, no AgentField writes (so no P6 trip).
    apply_damagemod_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 follow-on (Reap, mirror of Fortify at `001ae9a6`) —
    /// Cfg uniform for the new `physics_ApplyExecuteFromChronicle`
    /// kernel that translates `EffectExecuteApplied` records (kind=42,
    /// written by the apply_ability dispatcher in the standalone Reap
    /// chronicle kernel) back into `Defeated` events so the existing
    /// ApplyDefeat cascade (per-agent `set_alive`) keeps working
    /// unchanged. Same shape as ApplyDamageFromChronicle: PerEvent +
    /// emit-only kernel, no AgentField writes (so no P6 trip). Closes
    /// the slice across all 8 duel_abilities verbs.
    apply_execute_from_chronicle_cfg_buf: wgpu::Buffer,
    /// Task #138 — Packed AbilityRegistry uploaded to the GPU. The
    /// fused kernel binds `effect_kinds` / `effect_payload_a` /
    /// `effect_payload_b` for the apply_ability dispatcher arm
    /// (verb_chronicle_Strike). Built once at construction from the
    /// `.ability` corpus via `binding_check::build_duel_abilities_registry`,
    /// then uploaded to GPU storage buffers.
    registry_gpu: PackedAbilityRegistryGpu,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl DuelAbilitiesState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        // === ACCEPTANCE BINDING CHECK ===
        // Runs ONCE at startup before any GPU work. Re-parses the
        // source-of-truth `.ability` files and asserts every program
        // lowers to constants that match this fixture's hand-mirrored
        // .sim verb constants. If any assertion fails, the panic
        // points at the .sim/.ability divergence.
        //
        // Task #138 — also asserts Strike's AbilityId matches the
        // literal `apply_ability 1` in duel_abilities.sim so any drift
        // in the registry build order surfaces immediately.
        binding_check::assert_ability_registry_matches_sim_constants();

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Task #138 — build the AbilityRegistry from the .ability
        // corpus and upload it to GPU storage buffers. The fused kernel
        // binds the effect_kinds + payload columns for the
        // apply_ability dispatcher arm (verb_chronicle_Strike). Built
        // once here; the GPU buffers live for the rest of the run.
        // Constructing the registry repeats the binding-check's parse
        // pass (cheap — 8 small text files) but keeps the registry
        // build colocated with its consumer.
        let built_registry = binding_check::build_duel_abilities_registry();
        let packed = PackedAbilityRegistry::pack(&built_registry.registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed, &gpu, "duel_abilities_runtime",
        );

        // Agent SoA — HP=100.0, alive=1, mana=100.0 for every slot.
        let hp_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let mana_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let shield_hp_init: Vec<f32> = vec![0.0_f32; agent_count as usize];
        let agent_shield_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_shield_hp"),
            contents: bytemuck::cast_slice(&shield_hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // Lifesteal SoA — Wave 2 piece N. The compiler types
        // `lifesteal_frac_q8` as `i16` but the WGSL emit reads it as
        // `array<i32>` (see `cg/emit/kernel.rs`'s `AgentFieldTy::I16
        // => "array<i32>"` arm), so the GPU buffer is one i32 (4
        // bytes) per agent. Init to 0 (no lifesteal); ApplyDamage's
        // `src_frac > 0` gate keeps the heal branch dormant until
        // Vampirize fires.
        let lifesteal_frac_init: Vec<i32> = vec![0_i32; agent_count as usize];
        let agent_lifesteal_frac_q8_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_lifesteal_frac_q8"),
            contents: bytemuck::cast_slice(&lifesteal_frac_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let lifesteal_expires_init: Vec<u32> = vec![0_u32; agent_count as usize];
        let agent_lifesteal_expires_at_tick_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_lifesteal_expires_at_tick"),
            contents: bytemuck::cast_slice(&lifesteal_expires_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // Damage-taken-mult SoA — Wave 2 piece N DamageModify demo.
        // Same i16-as-array<i32> emit path as lifesteal_frac_q8 (see
        // `cg/emit/kernel.rs`'s `AgentFieldTy::I16 => "array<i32>"`),
        // so the GPU buffer is one i32 per agent. Init to 256 (=1.0×
        // in q8) so the `bleed * mult_q8 / 256.0` arithmetic is the
        // identity when the buff is inactive — the per-tick
        // `expires_at > world.tick` gate in ApplyDamage is the actual
        // activation switch (an EXPIRED window falls back to the raw
        // bleed via the if-expr's else branch).
        let damage_taken_mult_init: Vec<i32> = vec![256_i32; agent_count as usize];
        let agent_damage_taken_mult_q8_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_damage_taken_mult_q8"),
            contents: bytemuck::cast_slice(&damage_taken_mult_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let damage_taken_mult_expires_init: Vec<u32> = vec![0_u32; agent_count as usize];
        let agent_damage_taken_mult_expires_at_tick_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_damage_taken_mult_expires_at_tick"),
            contents: bytemuck::cast_slice(&damage_taken_mult_expires_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // Stun-expiry SoA — Wave 2 piece N Stun E2E demo + first
        // cast-gate. u32 per agent, init 0 (= "never stunned"). Read by
        // every offensive verb's mask kernel via
        // `agents.stun_expires_at_tick(self) <= world.tick`; written by
        // the ApplyStun arm of the fused PerEvent kernel.
        let stun_expires_init: Vec<u32> = vec![0_u32; agent_count as usize];
        let agent_stun_expires_at_tick_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_stun_expires_at_tick"),
            contents: bytemuck::cast_slice(&stun_expires_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Wave 1.5#4 GPU scaling: per-stat columns the dispatcher reads
        // at `caster_slot` for `scale_bonus`. Bleed declares
        // `+5% max_hp`, so `agent_max_hp` is seeded to 100.0 (matching
        // the prior hand-mirrored constant in `bleed_amount = 10.0 = 5
        // + 5%·100`). The other four (attack_damage / armor /
        // magic_resist / move_speed) stay zero — duel_abilities verbs
        // scale only on max_hp today.
        let max_hp_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_max_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::agent_max_hp"),
            contents: bytemuck::cast_slice(&max_hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let zeros_f32: Vec<f32> = vec![0.0_f32; agent_count as usize];
        let mk_stat = |label: &str| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("duel_abilities_runtime::agent_attack_damage");
        let agent_armor_buf         = mk_stat("duel_abilities_runtime::agent_armor");
        let agent_magic_resist_buf  = mk_stat("duel_abilities_runtime::agent_magic_resist");
        let agent_move_speed_buf    = mk_stat("duel_abilities_runtime::agent_move_speed");

        // Eight mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("duel_abilities_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("duel_abilities_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("duel_abilities_runtime::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("duel_abilities_runtime::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("duel_abilities_runtime::mask_4_bitmap");
        let mask_5_bitmap_buf = mk_mask("duel_abilities_runtime::mask_5_bitmap");
        let mask_6_bitmap_buf = mk_mask("duel_abilities_runtime::mask_6_bitmap");
        let mask_7_bitmap_buf = mk_mask("duel_abilities_runtime::mask_7_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_abilities_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "duel_abilities_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "duel_abilities_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "duel_abilities_runtime::healing_done",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::mask_cfg"),
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
            label: Some("duel_abilities_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Wave 2 piece N Stun E2E + cast-gate demo grew the fused-PerEvent
        // kernel ONCE MORE: ApplyStun joined the existing fusion group
        // (ApplyHeal/ApplyShield/ApplyDefeat/ApplyLifestealActivation/
        // ApplyDamageModActivation/verb_chronicle_Strike). The cfg type
        // now lives at
        // `physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_
        // ApplyLifestealActivation_and_ApplyDamageModActivation_and_
        // ApplyStun_and_verb_chronicle_Strike`. Field name
        // `chronicle_strike_cfg_buf` retained for continuity — Strike's
        // chronicle still needs an event_count uniform and this is
        // the kernel that runs it.
        let chronicle_strike_cfg_init =
            physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_ApplyLifestealActivation_and_ApplyDamageModActivation_and_ApplyStun_and_verb_chronicle_Strike::PhysicsApplyHealAndApplyShieldAndApplyDefeatAndApplyLifestealActivationAndApplyDamageModActivationAndApplyStunAndVerbChronicleStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_shieldup_cfg_init =
            physics_verb_chronicle_ShieldUp::PhysicsVerbChronicleShieldUpCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_shieldup_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_shieldup_cfg"),
            contents: bytemuck::bytes_of(&chronicle_shieldup_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_mend_cfg_init =
            physics_verb_chronicle_Mend::PhysicsVerbChronicleMendCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_mend_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_mend_cfg"),
            contents: bytemuck::bytes_of(&chronicle_mend_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_bleed_cfg_init =
            physics_verb_chronicle_Bleed::PhysicsVerbChronicleBleedCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_bleed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_bleed_cfg"),
            contents: bytemuck::bytes_of(&chronicle_bleed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_reap_cfg_init =
            physics_verb_chronicle_Reap::PhysicsVerbChronicleReapCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_reap_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_reap_cfg"),
            contents: bytemuck::bytes_of(&chronicle_reap_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Vampirize chronicle — Wave 2 piece N LifeSteal demo. The
        // compiler emitted this as a standalone kernel since
        // SetLifesteal events are produced HERE and consumed by
        // ApplyLifestealActivation downstream — same shape as
        // Strike/ShieldUp/Mend/Bleed/Reap chronicles but writes a
        // different event variant.
        let chronicle_vampirize_cfg_init =
            physics_verb_chronicle_Vampirize::PhysicsVerbChronicleVampirizeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_vampirize_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_vampirize_cfg"),
            contents: bytemuck::bytes_of(&chronicle_vampirize_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Fortify chronicle — Wave 2 piece N DamageModify demo. Same
        // standalone-kernel shape as Vampirize: produces SetDamageMod
        // events consumed by ApplyDamageModActivation (which itself is
        // fused into the big PerEvent group with ApplyHeal/ApplyShield/
        // ApplyDefeat/ApplyLifestealActivation/verb_chronicle_Strike).
        let chronicle_fortify_cfg_init =
            physics_verb_chronicle_Fortify::PhysicsVerbChronicleFortifyCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_fortify_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_fortify_cfg"),
            contents: bytemuck::bytes_of(&chronicle_fortify_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Daze chronicle — Wave 2 piece N Stun E2E demo. Same standalone
        // PerAgent shape as the other verb chronicles: produces Stunned
        // events drained by the ApplyStun arm of the fused PerEvent
        // kernel.
        let chronicle_daze_cfg_init =
            physics_verb_chronicle_Daze::PhysicsVerbChronicleDazeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_daze_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::chronicle_daze_cfg"),
            contents: bytemuck::bytes_of(&chronicle_daze_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Wave 2 piece N: `ApplyDamage` is now a STANDALONE kernel
        // (not the fused PerEvent group it used to be) because the
        // block now emits Healed events for source-side lifesteal
        // restoration. The compiler split it out and re-fused the rest
        // of the consumers (ApplyHeal/ApplyShield/ApplyDefeat/
        // ApplyLifestealActivation) WITH the Strike chronicle producer
        // into a single kernel. So the runtime now has TWO cfg
        // uniforms where it used to have one fused `apply_cfg`.
        let apply_damage_cfg_init =
            physics_ApplyDamage::PhysicsApplyDamageCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_damage_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_damage_cfg"),
            contents: bytemuck::bytes_of(&apply_damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 — Cfg uniform for the new ApplyDamageFromChronicle
        // kernel that translates EffectDamageApplied (kind=26) records
        // emitted by the apply_ability dispatcher into Damaged (kind=1)
        // events the existing ApplyDamage cascade consumes.
        let apply_damage_from_chronicle_cfg_init =
            physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_damage_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_damage_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_damage_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on — Cfg uniform for ApplyShieldFromChronicle
        // (kind=28 → Shielded) and ApplyHealFromChronicle (kind=27 →
        // Healed). Same shape as ApplyDamageFromChronicle: PerEvent +
        // emit-only kernels with no AgentField writes (so they don't
        // trip P6).
        let apply_shield_from_chronicle_cfg_init =
            physics_ApplyShieldFromChronicle::PhysicsApplyShieldFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_shield_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_shield_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_shield_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_heal_from_chronicle_cfg_init =
            physics_ApplyHealFromChronicle::PhysicsApplyHealFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_heal_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_heal_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_heal_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (Daze) — Cfg uniform for the new
        // ApplyStunFromChronicle kernel that translates EffectStunApplied
        // (kind=29) records emitted by the apply_ability dispatcher into
        // Stunned events the existing ApplyStun cascade consumes.
        let apply_stun_from_chronicle_cfg_init =
            physics_ApplyStunFromChronicle::PhysicsApplyStunFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_stun_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_stun_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_stun_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (Bleed, 2026-05-06) — Cfg uniform for the
        // new ApplyDamageFromSelfDamageChronicle kernel that translates
        // EffectSelfDamageApplied (kind=39) records emitted by the
        // apply_ability dispatcher into Damaged events the existing
        // ApplyDamage cascade consumes (with shield_hp absorption etc).
        let apply_damage_from_self_damage_chronicle_cfg_init =
            physics_ApplyDamageFromSelfDamageChronicle::PhysicsApplyDamageFromSelfDamageChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_damage_from_self_damage_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_damage_from_self_damage_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_damage_from_self_damage_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (Vampirize, mirror of Bleed at `486eb08f`)
        // — Cfg uniform for the new ApplyLifestealFromChronicle kernel
        // that translates EffectLifeStealApplied (kind=40) records
        // emitted by the apply_ability dispatcher into SetLifesteal
        // events the existing ApplyLifestealActivation cascade
        // consumes (writing per-agent lifesteal SoA fields).
        let apply_lifesteal_from_chronicle_cfg_init =
            physics_ApplyLifestealFromChronicle::PhysicsApplyLifestealFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_lifesteal_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_lifesteal_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_lifesteal_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (Fortify, mirror of Vampirize at `60115f64`)
        // — Cfg uniform for the new ApplyDamageModFromChronicle kernel
        // that translates EffectDamageModifyApplied (kind=41) records
        // emitted by the apply_ability dispatcher into SetDamageMod
        // events the existing ApplyDamageModActivation cascade
        // consumes (writing per-agent damage_taken_mult SoA fields).
        let apply_damagemod_from_chronicle_cfg_init =
            physics_ApplyDamageModFromChronicle::PhysicsApplyDamageModFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_damagemod_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_damagemod_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_damagemod_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Task #138 follow-on (Reap, mirror of Fortify at `001ae9a6`) —
        // Cfg uniform for the new ApplyExecuteFromChronicle kernel that
        // translates EffectExecuteApplied (kind=42) records emitted by
        // the apply_ability dispatcher into Defeated events the existing
        // ApplyDefeat cascade consumes (per-agent set_alive). Closes the
        // slice across all 8 duel_abilities verbs.
        let apply_execute_from_chronicle_cfg_init =
            physics_ApplyExecuteFromChronicle::PhysicsApplyExecuteFromChronicleCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_execute_from_chronicle_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_execute_from_chronicle_cfg"),
            contents: bytemuck::bytes_of(&apply_execute_from_chronicle_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::healing_done_cfg"),
            contents: bytemuck::bytes_of(&healing_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_shield_hp_buf,
            agent_lifesteal_frac_q8_buf,
            agent_lifesteal_expires_at_tick_buf,
            agent_damage_taken_mult_q8_buf,
            agent_damage_taken_mult_expires_at_tick_buf,
            agent_stun_expires_at_tick_buf,
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
            mask_5_bitmap_buf,
            mask_6_bitmap_buf,
            mask_7_bitmap_buf,
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
            chronicle_shieldup_cfg_buf,
            chronicle_mend_cfg_buf,
            chronicle_bleed_cfg_buf,
            chronicle_reap_cfg_buf,
            chronicle_vampirize_cfg_buf,
            chronicle_fortify_cfg_buf,
            chronicle_daze_cfg_buf,
            apply_damage_cfg_buf,
            apply_damage_from_chronicle_cfg_buf,
            apply_shield_from_chronicle_cfg_buf,
            apply_heal_from_chronicle_cfg_buf,
            apply_stun_from_chronicle_cfg_buf,
            apply_damage_from_self_damage_chronicle_cfg_buf,
            apply_lifesteal_from_chronicle_cfg_buf,
            apply_damagemod_from_chronicle_cfg_buf,
            apply_execute_from_chronicle_cfg_buf,
            registry_gpu,
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
    /// Per-agent shield HP, in agent-slot order. Mirrors `read_hp`'s
    /// staging-buffer + map-await pattern. Documented at the field site
    /// as the "agent_shield_hp() getter for observability"; defined here
    /// so `snapshot()` (and downstream eyeballing) can surface buff
    /// state without poking at the buffer directly.
    pub fn read_shield_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_shield_hp_buf, "shield_hp")
    }
    /// Per-agent lifesteal fraction (q8: 128 == 0.5x). Reads the GPU
    /// buffer via the shared u32 staging path — the field's storage is
    /// `array<i32>` per the compiler's WGSL emit (see `cg/emit/kernel.rs`),
    /// so `read_u32` returns the raw bit-pattern that the test
    /// reinterprets to `i32`.
    pub fn read_lifesteal_frac_q8(&self) -> Vec<i32> {
        self.read_u32(&self.agent_lifesteal_frac_q8_buf, "lifesteal_frac_q8")
            .into_iter()
            .map(|u| u as i32)
            .collect()
    }
    /// Per-agent lifesteal window expiry tick (in world ticks).
    pub fn read_lifesteal_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_lifesteal_expires_at_tick_buf, "lifesteal_expires_at_tick")
    }
    /// Per-agent damage_taken multiplier (q8: 256 == 1.0×, 128 == 0.5×).
    /// Reads via the shared u32 staging path — the field's storage is
    /// `array<i32>` per the compiler's WGSL emit (`AgentFieldTy::I16
    /// => "array<i32>"` in `cg/emit/kernel.rs`), so `read_u32` returns
    /// the raw bit-pattern that the test reinterprets to `i32`.
    pub fn read_damage_taken_mult_q8(&self) -> Vec<i32> {
        self.read_u32(&self.agent_damage_taken_mult_q8_buf, "damage_taken_mult_q8")
            .into_iter()
            .map(|u| u as i32)
            .collect()
    }
    /// Per-agent damage_taken_mult window expiry tick (in world ticks).
    pub fn read_damage_taken_mult_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_damage_taken_mult_expires_at_tick_buf, "damage_taken_mult_expires_at_tick")
    }
    /// Per-agent stun-window expiry tick (in world ticks). 0 means
    /// "never stunned"; the cast-gate `agents.stun_expires_at_tick(self)
    /// <= world.tick` evaluates to TRUE for that default, so an init=0
    /// agent can act normally.
    pub fn read_stun_expires_at_tick(&self) -> Vec<u32> {
        self.read_u32(&self.agent_stun_expires_at_tick_buf, "stun_expires_at_tick")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("duel_abilities_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_abilities_runtime::read_f32") },
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
            label: Some(&format!("duel_abilities_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_abilities_runtime::read_u32") },
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

    /// Test-only HP override. Writes the supplied values directly to the
    /// `agent_hp` SoA so a test can preconfigure a state where Reap's
    /// `target.hp < threshold` gate is satisfied at the next tick%20==0
    /// boundary. The natural duel never produces a target.hp ∈ (0, 20)
    /// at a tick%20==0 boundary — Strike's 30-damage step skips the
    /// (0, 10] window — so we engineer the state to surface the
    /// Defeated event from Reap rather than Strike's inline emit.
    ///
    /// Length must equal `agent_count`. Panics on mismatch.
    #[doc(hidden)]
    pub fn override_hp_for_test(&self, hp: &[f32]) {
        assert_eq!(
            hp.len(),
            self.agent_count as usize,
            "override_hp_for_test: length must match agent_count",
        );
        self.gpu.queue.write_buffer(
            &self.agent_hp_buf,
            0,
            bytemuck::cast_slice(hp),
        );
        // No submit needed — the queue serialises writes ahead of the
        // next encoder.submit on `step()`.
    }

    /// Test-only stun-expiry override. Writes the supplied per-agent
    /// `expires_at_tick` values directly to the
    /// `agent_stun_expires_at_tick` SoA so a test can preconfigure a
    /// stun window without having to engineer a Daze cast onto the
    /// target. Length must equal `agent_count`. Panics on mismatch.
    ///
    /// Used by `stunned_agent_skips_strike` to verify that ANY
    /// offensive verb's `when` clause skips when
    /// `agents.stun_expires_at_tick(self) > world.tick`.
    #[doc(hidden)]
    pub fn override_stun_for_test(&self, expires_at: &[u32]) {
        assert_eq!(
            expires_at.len(),
            self.agent_count as usize,
            "override_stun_for_test: length must match agent_count",
        );
        self.gpu.queue.write_buffer(
            &self.agent_stun_expires_at_tick_buf,
            0,
            bytemuck::cast_slice(expires_at),
        );
    }
}

impl CompiledSim for DuelAbilitiesState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_abilities_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // 8 verbs in source order; +1 for ApplyDamage's source-side
        // Healed emit (lifesteal); +1 for SetLifesteal each Vampirize
        // cast emits; +1 for SetDamageMod each Fortify cast emits;
        // +1 for Stunned each Daze cast emits. Task #138 grew the
        // worst-case by two more records per agent: Strike's
        // apply_ability now emits one EffectDamageApplied (kind=26)
        // per cast, and ApplyDamageFromChronicle re-emits each into
        // a Damaged event. The Task #138 follow-on extended the swap
        // to ShieldUp (kind=28 EffectShieldApplied + Shielded re-emit)
        // and Mend (kind=27 EffectHealApplied + Healed re-emit) — but
        // the verbs are mutually exclusive per tick (scoring picks
        // one), so the worst case is +2 records per agent (whichever
        // verb fires + its re-emit). Task #138 follow-on (Daze) adds
        // a fourth swap (kind=29 EffectStunApplied + Stunned re-emit)
        // — same +2 per-agent overhead when the chance gate fires,
        // and same mutual-exclusivity argument with the other verbs.
        // Task #138 follow-on (Bleed, 2026-05-06) extended the swap to
        // Bleed (kind=39 EffectSelfDamageApplied + Damaged re-emit) —
        // again +2 per-agent overhead, again mutually exclusive with
        // the other verbs.
        // Task #138 follow-on (Vampirize, mirror of Bleed) extended the
        // swap to Vampirize (kind=40 EffectLifeStealApplied +
        // SetLifesteal re-emit) — again +2 per-agent overhead, again
        // mutually exclusive with the other verbs.
        // Task #138 follow-on (Fortify, mirror of Vampirize) extended the
        // swap to Fortify (kind=41 EffectDamageModifyApplied +
        // SetDamageMod re-emit) — again +2 per-agent overhead, again
        // mutually exclusive with the other verbs.
        // Task #138 follow-on (Reap, mirror of Fortify) extended the
        // swap to Reap (kind=42 EffectExecuteApplied + Defeated
        // re-emit) — again +2 per-agent overhead, again mutually
        // exclusive with the other verbs (Reap is the verb that fires
        // when target.hp < threshold, so it competes with Strike at the
        // 20-tick boundary). Closes the slice across all 8
        // duel_abilities verbs.
        // Bump the upper bound to 30 slots per agent to keep
        // clear_ring_headers from leaving stale slots between ticks.
        // (28 + 2 = 30; the per-tick sum stays in the same
        // verb-mutually-exclusive band but the bump gives every
        // chronicle re-emit its own headroom.)
        let max_slots_per_tick = self.agent_count * 30;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [
            &self.mask_0_bitmap_buf,
            &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf,
            &self.mask_3_bitmap_buf,
            &self.mask_4_bitmap_buf,
            &self.mask_5_bitmap_buf,
            &self.mask_6_bitmap_buf,
            &self.mask_7_bitmap_buf,
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

        // (2) Mask round.
        let mask_cfg = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        // Mask kernel: no verb gates on mana, so the generated bindings
        // omit agent_mana (compiler emits only what's read). Mana SoA
        // stays in this fixture for parity with duel_1v1's interface
        // but isn't bound to any kernel.
        //
        // Wave 2 piece N — `agent_stun_expires_at_tick` is now bound
        // because EVERY verb's `when` clause reads it for the cast-gate
        // `agents.stun_expires_at_tick(self) <= world.tick`. This is
        // the FIRST status-SoA field bound to the mask kernel — the
        // previous fixture had only purely-functional reads (hp, alive).
        let mask_bindings = fused_mask_verb_Strike::FusedMaskVerbStrikeBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_stun_expires_at_tick: &self.agent_stun_expires_at_tick_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            mask_4_bitmap: &self.mask_4_bitmap_buf,
            mask_5_bitmap: &self.mask_5_bitmap_buf,
            mask_6_bitmap: &self.mask_6_bitmap_buf,
            mask_7_bitmap: &self.mask_7_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_strike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count * self.agent_count,
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
        // Scoring kernel binds agent_hp because Strike's score
        // formula `(200.0 - target.hp)` is a pair-field read — the
        // scoring kernel iterates per_pair_candidate and looks up
        // agent_hp[candidate]. Without that pair iteration the
        // best_target slot stays at the 0xFFFFFFFF sentinel and the
        // chronicle's emitted Damaged event addresses an OOB slot.
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            mask_4_bitmap: &self.mask_4_bitmap_buf,
            mask_5_bitmap: &self.mask_5_bitmap_buf,
            mask_6_bitmap: &self.mask_6_bitmap_buf,
            mask_7_bitmap: &self.mask_7_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Strike chronicle is now FUSED into the
        // `physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_
        // ApplyLifestealActivation_and_verb_chronicle_Strike` kernel
        // dispatched at step (8b) below. The compiler split out
        // ApplyDamage (which now emits Healed events for source-side
        // lifesteal) and re-fused Strike with the remaining consumers
        // since Strike is now the only producer the others can ride.

        // (5) ShieldUp chronicle. Task #138 follow-on — verb body is
        // now `apply_ability 2 by self target self`, so this kernel
        // walks the AbilityRegistry's effect SoA columns to expand the
        // dispatch into chronicle EffectShieldApplied writes (kind=28).
        // Re-emitted as Shielded by ApplyShieldFromChronicle below.
        let shieldup_cfg = physics_verb_chronicle_ShieldUp::PhysicsVerbChronicleShieldUpCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_shieldup_cfg_buf, 0, bytemuck::bytes_of(&shieldup_cfg),
        );
        let shieldup_bindings = physics_verb_chronicle_ShieldUp::PhysicsVerbChronicleShieldUpBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_shieldup_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_shieldup(
            &mut self.cache, &shieldup_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) Mend chronicle. Task #138 follow-on — verb body is now
        // `apply_ability 3 by self target self`, so this kernel walks
        // the AbilityRegistry's effect SoA columns to expand the
        // dispatch into chronicle EffectHealApplied writes (kind=27).
        // Re-emitted as Healed by ApplyHealFromChronicle below.
        let mend_cfg = physics_verb_chronicle_Mend::PhysicsVerbChronicleMendCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_mend_cfg_buf, 0, bytemuck::bytes_of(&mend_cfg),
        );
        let mend_bindings = physics_verb_chronicle_Mend::PhysicsVerbChronicleMendBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_mend_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_mend(
            &mut self.cache, &mend_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) Bleed chronicle — Wave 2 SelfDamage demo. Task #138
        // follow-on (Bleed, 2026-05-06) — verb body is now
        // `apply_ability 4 by self target self`, so this kernel walks
        // the AbilityRegistry's effect SoA columns to expand the
        // dispatch into chronicle EffectSelfDamageApplied writes
        // (kind=39). Re-emitted as Damaged by
        // ApplyDamageFromSelfDamageChronicle below; the existing
        // ApplyDamage cascade drains shield first then hp, so the
        // caster's hp drops by min(5, max(0, 5 - shield)) per cast.
        let bleed_cfg = physics_verb_chronicle_Bleed::PhysicsVerbChronicleBleedCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_bleed_cfg_buf, 0, bytemuck::bytes_of(&bleed_cfg),
        );
        let bleed_bindings = physics_verb_chronicle_Bleed::PhysicsVerbChronicleBleedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_bleed_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bleed(
            &mut self.cache, &bleed_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7b) Reap chronicle — Wave 2 Execute demo. Task #138 follow-on
        // (Reap, mirror of Fortify at `001ae9a6`) — verb body is now
        // `apply_ability 5 by self target target`, so this kernel walks
        // the AbilityRegistry's effect SoA columns to expand the
        // dispatch into chronicle EffectExecuteApplied writes (kind=42).
        // Re-emitted as Defeated by ApplyExecuteFromChronicle below;
        // the existing ApplyDefeat cascade drains Defeated into per-agent
        // `set_alive(t, false)`. Closes the slice across all 8
        // duel_abilities verbs.
        let reap_cfg = physics_verb_chronicle_Reap::PhysicsVerbChronicleReapCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_reap_cfg_buf, 0, bytemuck::bytes_of(&reap_cfg),
        );
        let reap_bindings = physics_verb_chronicle_Reap::PhysicsVerbChronicleReapBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_reap_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_reap(
            &mut self.cache, &reap_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7c) Vampirize chronicle — Wave 2 LifeSteal demo. Task #138
        // follow-on (Vampirize, mirror of Bleed at `486eb08f`) — verb
        // body is now `apply_ability 6 by self target self`, so this
        // kernel walks the AbilityRegistry's effect SoA columns to
        // expand the dispatch into chronicle EffectLifeStealApplied
        // writes (kind=40). Re-emitted as SetLifesteal by
        // ApplyLifestealFromChronicle below; the existing
        // ApplyLifestealActivation cascade drains SetLifesteal into
        // the per-agent lifesteal_frac_q8 + lifesteal_expires_at_tick
        // SoA slots so ApplyDamage's source lookup can heal them on
        // subsequent hits.
        let vampirize_cfg = physics_verb_chronicle_Vampirize::PhysicsVerbChronicleVampirizeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_vampirize_cfg_buf, 0, bytemuck::bytes_of(&vampirize_cfg),
        );
        let vampirize_bindings = physics_verb_chronicle_Vampirize::PhysicsVerbChronicleVampirizeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_vampirize_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_vampirize(
            &mut self.cache, &vampirize_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7d) Fortify chronicle — Wave 2 DamageModify demo. Task #138
        // follow-on (Fortify, mirror of Vampirize at `60115f64`) — verb
        // body is now `apply_ability 7 by self target self`, so this
        // kernel walks the AbilityRegistry's effect SoA columns to
        // expand the dispatch into chronicle EffectDamageModifyApplied
        // writes (kind=41). Re-emitted as SetDamageMod by
        // ApplyDamageModFromChronicle below; the existing
        // ApplyDamageModActivation cascade drains SetDamageMod into
        // the per-agent damage_taken_mult_q8 +
        // damage_taken_mult_expires_at_tick SoA slots so ApplyDamage's
        // target lookup scales bleed by mult/256 on subsequent hits.
        let fortify_cfg = physics_verb_chronicle_Fortify::PhysicsVerbChronicleFortifyCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_fortify_cfg_buf, 0, bytemuck::bytes_of(&fortify_cfg),
        );
        let fortify_bindings = physics_verb_chronicle_Fortify::PhysicsVerbChronicleFortifyBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_fortify_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_fortify(
            &mut self.cache, &fortify_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7e) Daze chronicle — Wave 2 piece N Stun E2E demo + first
        // verb-status cast-gate. Task #138 follow-on (Daze) — verb
        // body is now `apply_ability 8 by self target target`, so this
        // kernel walks the AbilityRegistry's effect SoA columns to
        // expand the dispatch into chronicle EffectStunApplied writes
        // (kind=29). Re-emitted as Stunned by ApplyStunFromChronicle
        // below; ApplyStun (fused kernel) drains those Stunned events
        // and writes the target's stun_expires_at_tick SoA slot so
        // EVERY offensive verb's mask kernel reads
        // `stun_expires_at_tick > world.tick` and skips casting for
        // the duration of the window.
        let daze_cfg = physics_verb_chronicle_Daze::PhysicsVerbChronicleDazeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_daze_cfg_buf, 0, bytemuck::bytes_of(&daze_cfg),
        );
        let daze_bindings = physics_verb_chronicle_Daze::PhysicsVerbChronicleDazeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_daze_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_daze(
            &mut self.cache, &daze_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7f) Task #138 follow-on — ApplyShieldFromChronicle.
        // The ShieldUp chronicle (step 5) just wrote EffectShieldApplied
        // records (kind=28) into the event ring via the apply_ability
        // dispatcher arm. This kernel filters those records and re-emits
        // them as `Shielded` (kind=2.x source-declared) events so the
        // ApplyShield arm of the fused kernel below can drain them with
        // the existing per-agent set_shield_hp accumulation intact.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 28. Setting event_count to the
        // generous estimate is safe because cleared slots read kind=0
        // (skipped) and non-EffectShieldApplied records also skipped.
        // Reuses the per-tick max_slots_per_tick (= agent_count * 20)
        // computed at the top of step() for clear_ring_headers.
        let apply_shield_from_chronicle_cfg = physics_ApplyShieldFromChronicle::PhysicsApplyShieldFromChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_shield_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_shield_from_chronicle_cfg),
        );
        let apply_shield_from_chronicle_bindings = physics_ApplyShieldFromChronicle::PhysicsApplyShieldFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_shield_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applyshieldfromchronicle(
            &mut self.cache, &apply_shield_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (7g) Task #138 follow-on — ApplyHealFromChronicle.
        // The Mend chronicle (step 6) just wrote EffectHealApplied
        // records (kind=27) into the event ring via the apply_ability
        // dispatcher arm. This kernel filters those records and
        // re-emits them as `Healed` events so the ApplyHeal arm of the
        // fused kernel below can drain them with per-agent set_hp
        // intact.
        let apply_heal_from_chronicle_cfg = physics_ApplyHealFromChronicle::PhysicsApplyHealFromChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_heal_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_heal_from_chronicle_cfg),
        );
        let apply_heal_from_chronicle_bindings = physics_ApplyHealFromChronicle::PhysicsApplyHealFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_heal_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applyhealfromchronicle(
            &mut self.cache, &apply_heal_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (7h) Task #138 follow-on (Daze) — ApplyStunFromChronicle.
        // The Daze chronicle (step 7e) just wrote EffectStunApplied
        // records (kind=29) into the event ring via the apply_ability
        // dispatcher arm — when the verb's `chance 50%` `when` gate
        // fires (the dispatcher itself does NOT consult program.chances
        // today, so the verb gate is the only chance gate). This kernel
        // filters those records and re-emits them as `Stunned` events
        // so the ApplyStun arm of the fused kernel below can drain
        // them with per-agent set_stun_expires_at_tick intact.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 29. The chronicle's third
        // payload word is `expires_at_tick` (= dispatcher's `tick +
        // duration_ticks`), which the existing `Stunned` event already
        // speaks — re-emit ferries verbatim, no conversion required.
        let apply_stun_from_chronicle_cfg = physics_ApplyStunFromChronicle::PhysicsApplyStunFromChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_stun_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_stun_from_chronicle_cfg),
        );
        let apply_stun_from_chronicle_bindings = physics_ApplyStunFromChronicle::PhysicsApplyStunFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_stun_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applystunfromchronicle(
            &mut self.cache, &apply_stun_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (7i) Task #138 follow-on (Bleed, 2026-05-06) —
        // ApplyDamageFromSelfDamageChronicle. The Bleed chronicle
        // (step 7) just wrote EffectSelfDamageApplied records (kind=39)
        // into the event ring via the apply_ability dispatcher arm.
        // This kernel filters those records and re-emits them as
        // `Damaged` events so the existing ApplyDamage standalone
        // kernel below can drain them with shield/lifesteal/damage-
        // modify processing intact.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 39. Same pattern as
        // ApplyDamageFromChronicle (kind=26), just with the new
        // EffectSelfDamageApplied discriminant. The dispatcher writes
        // caster_slot into BOTH actor (slot 2) and target (slot 3) for
        // the SelfDamage arm, so the re-emit carries source==target
        // and ApplyDamage routes the bleed-through hp loss to the
        // caster.
        let apply_damage_from_self_damage_chronicle_cfg = physics_ApplyDamageFromSelfDamageChronicle::PhysicsApplyDamageFromSelfDamageChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_damage_from_self_damage_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_damage_from_self_damage_chronicle_cfg),
        );
        let apply_damage_from_self_damage_chronicle_bindings = physics_ApplyDamageFromSelfDamageChronicle::PhysicsApplyDamageFromSelfDamageChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_damage_from_self_damage_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applydamagefromselfdamagechronicle(
            &mut self.cache, &apply_damage_from_self_damage_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (7j) Task #138 follow-on (Vampirize, mirror of Bleed at
        // `486eb08f`) — ApplyLifestealFromChronicle. The Vampirize
        // chronicle (step 7c) just wrote EffectLifeStealApplied records
        // (kind=40) into the event ring via the apply_ability dispatcher
        // arm. This kernel filters those records and re-emits them as
        // `SetLifesteal` events so the fused ApplyLifestealActivation
        // kernel below (step 8a) can drain them and write the per-agent
        // lifesteal_frac_q8 + lifesteal_expires_at_tick SoA fields.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 40. Same pattern as
        // ApplyDamageFromSelfDamageChronicle (kind=39), just with the
        // EffectLifeStealApplied discriminant + the 4-payload-word
        // shape (actor + target + expires_at_tick + fraction_q8). The
        // dispatcher writes caster_slot into BOTH actor (slot 2) and
        // target (slot 3) for the LifeSteal arm (self-cast), so the
        // re-emit's `caster: c` reads the caster id directly.
        let apply_lifesteal_from_chronicle_cfg = physics_ApplyLifestealFromChronicle::PhysicsApplyLifestealFromChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_lifesteal_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_lifesteal_from_chronicle_cfg),
        );
        let apply_lifesteal_from_chronicle_bindings = physics_ApplyLifestealFromChronicle::PhysicsApplyLifestealFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_lifesteal_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applylifestealfromchronicle(
            &mut self.cache, &apply_lifesteal_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (7k) Task #138 follow-on (Fortify, mirror of Vampirize at
        // `60115f64`) — ApplyDamageModFromChronicle. The Fortify
        // chronicle (step 7d) just wrote EffectDamageModifyApplied
        // records (kind=41) into the event ring via the apply_ability
        // dispatcher arm. This kernel filters those records and
        // re-emits them as `SetDamageMod` events so the fused
        // ApplyDamageModActivation kernel below (step 8a) can drain
        // them and write the per-agent damage_taken_mult_q8 +
        // damage_taken_mult_expires_at_tick SoA fields.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 41. Same pattern as
        // ApplyLifestealFromChronicle (kind=40), just with the
        // EffectDamageModifyApplied discriminant + the 4-payload-word
        // shape (actor + target + expires_at_tick + multiplier_q8). The
        // dispatcher writes caster_slot into BOTH actor (slot 2) and
        // target (slot 3) for the DamageModify arm (self-cast), so the
        // re-emit's `actor: c` reads the caster id directly.
        let apply_damagemod_from_chronicle_cfg = physics_ApplyDamageModFromChronicle::PhysicsApplyDamageModFromChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_damagemod_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_damagemod_from_chronicle_cfg),
        );
        let apply_damagemod_from_chronicle_bindings = physics_ApplyDamageModFromChronicle::PhysicsApplyDamageModFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_damagemod_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applydamagemodfromchronicle(
            &mut self.cache, &apply_damagemod_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (7l) Task #138 follow-on (Reap, mirror of Fortify at
        // `001ae9a6`) — ApplyExecuteFromChronicle. The Reap chronicle
        // (step 7b) just wrote EffectExecuteApplied records (kind=42)
        // into the event ring via the apply_ability dispatcher arm.
        // This kernel filters those records and re-emits them as
        // `Defeated` events so the fused ApplyDefeat kernel below
        // (step 8a) can drain them and write per-agent
        // `set_alive(t, false)`. Closes the slice across all 8
        // duel_abilities verbs.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 42. Same pattern as
        // ApplyDamageFromChronicle (kind=26) — 3-payload-word shape
        // (actor + target + hp_threshold), with the re-emit pulling
        // `target: t` directly into Defeated's `combatant` field. The
        // hp_threshold payload is decorative because:
        //   1. apply_program doesn't evaluate `when_per_effect[i]`
        //      today, so the dispatcher writes the record
        //      unconditionally once the verb mask passes.
        //   2. Reap's verb gate already enforces
        //      `target.hp < config.combat.reap_threshold` upstream.
        // Together those mean the re-emit can ferry the record into
        // Defeated without re-checking hp.
        let apply_execute_from_chronicle_cfg = physics_ApplyExecuteFromChronicle::PhysicsApplyExecuteFromChronicleCfg {
            event_count: max_slots_per_tick, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_execute_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_execute_from_chronicle_cfg),
        );
        let apply_execute_from_chronicle_bindings = physics_ApplyExecuteFromChronicle::PhysicsApplyExecuteFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_execute_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applyexecutefromchronicle(
            &mut self.cache, &apply_execute_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, max_slots_per_tick,
        );

        // (8a) Fused ApplyHeal + ApplyShield + ApplyDefeat +
        // ApplyLifestealActivation + ApplyDamageModActivation +
        // ApplyStun + verb_chronicle_Strike. The compiler re-fused these
        // because Strike's chronicle is the lone producer feeding the
        // Healed/Shielded/Defeated/SetLifesteal/SetDamageMod/Stunned
        // consumers. **Runs BEFORE ApplyDamage** so Strike's emitted
        // Damaged events are visible to ApplyDamage at step (8b);
        // ShieldUp/Mend/Bleed/Reap/Vampirize/Fortify/Daze chronicles
        // already emitted earlier (steps 5-7e), and their consumer arms
        // (ApplyHeal/ApplyShield/ApplyDefeat/ApplyLifestealActivation/
        // ApplyDamageModActivation/ApplyStun) drain those here.
        //
        // The bind-group binds the stun SoA write-side (ApplyStun
        // writes hot_stun_expires_at_tick), alongside the lifesteal +
        // damage_taken_mult SoA write-side (their own ApplyXActivation
        // arms). The compiler-emitted SCHEDULE places ApplyDamage first;
        // we transpose because the Strike→ApplyDamage chain MUST happen
        // within a single tick.
        //
        // Wave 2 piece N — fusion topology shift: ApplyStun joined the
        // existing fusion group, growing the kernel name by one
        // `_and_ApplyStun_` segment and adding agent_stun_expires_at_tick
        // to the bind group. No new pass introduced; same single
        // dispatch per tick.
        let event_count_estimate = self.agent_count * 30;
        let apply_heal_cfg = physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_ApplyLifestealActivation_and_ApplyDamageModActivation_and_ApplyStun_and_verb_chronicle_Strike::PhysicsApplyHealAndApplyShieldAndApplyDefeatAndApplyLifestealActivationAndApplyDamageModActivationAndApplyStunAndVerbChronicleStrikeCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&apply_heal_cfg),
        );
        // Task #138 — the fused kernel now binds the
        // PackedAbilityRegistry's effect SoA columns because the
        // apply_ability dispatcher arm (verb_chronicle_Strike) walks
        // the registry to expand `apply_ability 1 by self target target`
        // into chronicle EffectDamageApplied writes.
        let apply_heal_bindings = physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_ApplyLifestealActivation_and_ApplyDamageModActivation_and_ApplyStun_and_verb_chronicle_Strike::PhysicsApplyHealAndApplyShieldAndApplyDefeatAndApplyLifestealActivationAndApplyDamageModActivationAndApplyStunAndVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_shield_hp: &self.agent_shield_hp_buf,
            agent_stun_expires_at_tick: &self.agent_stun_expires_at_tick_buf,
            agent_lifesteal_frac_q8: &self.agent_lifesteal_frac_q8_buf,
            agent_lifesteal_expires_at_tick: &self.agent_lifesteal_expires_at_tick_buf,
            agent_damage_taken_mult_q8: &self.agent_damage_taken_mult_q8_buf,
            agent_damage_taken_mult_expires_at_tick: &self.agent_damage_taken_mult_expires_at_tick_buf,
            ability_registry_effect_kinds: &self.registry_gpu.effect_kinds,
            ability_registry_effect_payload_a: &self.registry_gpu.effect_payload_a,
            ability_registry_effect_payload_b: &self.registry_gpu.effect_payload_b,
            ability_registry_nested_effect_kinds: &self.registry_gpu.nested_effect_kinds,
            ability_registry_nested_effect_payload_a: &self.registry_gpu.nested_effect_payload_a,
            ability_registry_nested_effect_payload_b: &self.registry_gpu.nested_effect_payload_b,
            ability_registry_scaling_stat_refs: &self.registry_gpu.scaling_stat_refs,
            ability_registry_scaling_percents:  &self.registry_gpu.scaling_percents,
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
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_applyheal_and_applyshield_and_applydefeat_and_applylifestealactivation_and_applydamagemodactivation_and_applystun_and_verb_chronicle_strike(
            &mut self.cache, &apply_heal_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (8a.5) Task #138 — ApplyDamageFromChronicle.
        // The fused kernel above just wrote EffectDamageApplied records
        // (kind=26) into the event ring via the apply_ability dispatcher
        // arm (one record per cast). This kernel filters those records
        // and re-emits them as `Damaged` (kind=1) events so the existing
        // ApplyDamage standalone kernel below can drain them with
        // shield/lifesteal/damage-modify processing intact.
        //
        // PerEvent shape: scans every event_ring slot up to event_count
        // and skips slots whose kind != 26. Setting event_count to the
        // generous estimate is safe because cleared slots read kind=0
        // (skipped) and non-EffectDamageApplied records also skipped.
        let apply_damage_from_chronicle_cfg = physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_damage_from_chronicle_cfg_buf, 0,
            bytemuck::bytes_of(&apply_damage_from_chronicle_cfg),
        );
        let apply_damage_from_chronicle_bindings = physics_ApplyDamageFromChronicle::PhysicsApplyDamageFromChronicleBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.apply_damage_from_chronicle_cfg_buf,
        };
        dispatch::dispatch_physics_applydamagefromchronicle(
            &mut self.cache, &apply_damage_from_chronicle_bindings,
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (8b) ApplyDamage — STANDALONE PerEvent kernel. Wave 2 piece N
        // refactor: ApplyDamage now emits Healed events for source-side
        // lifesteal restoration, so it became a producer and the
        // compiler split it out of the previous fusion group. Runs
        // AFTER step (8a) so Strike's Damaged emits (and Bleed's
        // self-Damaged emits from step 7) are visible. The lifesteal
        // SoA fields are read-only here (written upstream by the
        // ApplyLifestealActivation arm of the (8a) kernel).
        //
        // CAVEAT: ApplyDamage's source-side Healed emit lands in the
        // ring AFTER the (8a) ApplyHeal arm has already drained it for
        // this tick — so source-side healing from lifesteal materialises
        // ONE TICK LATER (next tick's (8a) drains the Healed events
        // emitted here). This is acceptable for the demo because
        // Vampirize sets a 50-tick window and individual hits land on
        // 10-tick intervals (Strike cooldown), so the heal still
        // arrives well within the lifesteal window.
        let apply_damage_cfg = physics_ApplyDamage::PhysicsApplyDamageCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_damage_cfg_buf, 0, bytemuck::bytes_of(&apply_damage_cfg),
        );
        let apply_damage_bindings = physics_ApplyDamage::PhysicsApplyDamageBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_shield_hp: &self.agent_shield_hp_buf,
            agent_lifesteal_frac_q8: &self.agent_lifesteal_frac_q8_buf,
            agent_lifesteal_expires_at_tick: &self.agent_lifesteal_expires_at_tick_buf,
            agent_damage_taken_mult_q8: &self.agent_damage_taken_mult_q8_buf,
            agent_damage_taken_mult_expires_at_tick: &self.agent_damage_taken_mult_expires_at_tick_buf,
            cfg: &self.apply_damage_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage(
            &mut self.cache, &apply_damage_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (8) seed_indirect_0.
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

        // (9) fold_damage_dealt.
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

        // (10) fold_healing_done.
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
    /// The duel doesn't move agents — combat is purely event-driven HP
    /// edits — so positions are a deterministic 1-D fixed grid laid out
    /// along +X (`agent_id * 5.0`). The renderer's grid + glyph-table
    /// pipeline does the rest.
    ///
    /// Per-agent fields populated:
    /// - `positions`: stationary grid (Hero A at origin, Hero B 5 units east).
    /// - `creature_types`: HP-banded discriminant — hero index encoded
    ///   in the low bit (0=A, 1=B), HP bucket (full=0, hurt=1, low=1)
    ///   in the next two bits, so the glyph table can colour-shift as
    ///   the duel progresses without inventing new glyphs. Dead slots
    ///   land in the `2|hero_id` "tombstone" rows.
    /// - `alive`: read directly from `agent_alive_buf` AND gated by
    ///   HP > 0 (defence-in-depth — the chronicle->ApplyDamage kernel
    ///   sets alive=0 on HP<=0 but a partial step or a future bug
    ///   shouldn't render a corpse on the field).
    ///
    /// Initial-state safe: the GPU buffers are populated by
    /// `create_buffer_init` at construction, so calling `snapshot()`
    /// before any `step()` returns hp=100, alive=1 for every slot.
    fn snapshot(&mut self) -> AgentSnapshot {
        let hp = self.read_hp();
        let alive_raw = self.read_alive();
        // Defence-in-depth: drop slots whose HP fell to 0 even if the
        // alive bit hasn't been written yet by ApplyDamage.
        let alive: Vec<u32> = alive_raw
            .iter()
            .zip(hp.iter())
            .map(|(&a, &h)| if a != 0 && h > 0.0 { 1 } else { 0 })
            .collect();
        let positions: Vec<Vec3> = (0..self.agent_count as usize)
            .map(|i| Vec3::new(i as f32 * 5.0, 0.0, 0.0))
            .collect();
        // creature_type encoding: 4 entries per hero index in the glyph
        // table — full HP, hurt (<75%), low (<33%), dead (×). Hero
        // index in low bit (0=A,1=B); HP bucket in upper bits → table
        // index = bucket * 2 + hero_id.
        let creature_types: Vec<u32> = (0..self.agent_count as usize)
            .map(|i| {
                let hero_id = (i & 1) as u32;
                let bucket = if alive[i] == 0 {
                    3
                } else if hp[i] < 33.0 {
                    2
                } else if hp[i] < 75.0 {
                    1
                } else {
                    0
                };
                bucket * 2 + hero_id
            })
            .collect();
        AgentSnapshot {
            positions,
            creature_types,
            alive,
        }
    }

    /// 4 HP-banded glyphs × 2 hero ids = 8 entries.
    /// Layout: `[full_A, full_B, hurt_A, hurt_B, low_A, low_B, dead_A, dead_B]`.
    /// Colours: A in cyan tones, B in red tones — both desaturate as HP
    /// drops, then go grey on death.
    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![
            VizGlyph::new('A', 51),  // full A: bright cyan
            VizGlyph::new('B', 196), // full B: bright red
            VizGlyph::new('a', 39),  // hurt A: dim cyan
            VizGlyph::new('b', 160), // hurt B: dim red
            VizGlyph::new('a', 27),  // low A: deep blue
            VizGlyph::new('b', 88),  // low B: deep red
            VizGlyph::new('\u{00D7}', 240), // dead A: grey ×
            VizGlyph::new('\u{00D7}', 240), // dead B: grey ×
        ]
    }

    /// Tight viewport around the two stationary heroes. Hero A sits at
    /// x=0, Hero B at x=5 (see `snapshot`). 8-unit window keeps both on
    /// screen with breathing room.
    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        Some((Vec3::new(-1.5, -1.5, 0.0), Vec3::new(6.5, 1.5, 0.0)))
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(DuelAbilitiesState::new(seed, agent_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tick-200 acceptance smoke test. Constructs the fixture (which
    /// runs the binding check at startup), ticks 200 times, and asserts
    /// at least one Combatant has hp <= 0 (combat resolved).
    ///
    /// Skipped when no GPU adapter is available — the binding check
    /// still ran during construction and would have panicked otherwise.
    #[test]
    fn tick_200_resolves_combat() {
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 2);
        for _ in 0..200 {
            state.step();
        }
        let alive = state.read_alive();
        let hp = state.read_hp();
        assert!(
            alive[0] == 0 || alive[1] == 0 || hp[0] <= 0.0 || hp[1] <= 0.0,
            "expected at least one Combatant defeated by tick 200, \
             got alive=[{}, {}], hp=[{:.2}, {:.2}]",
            alive[0], alive[1], hp[0], hp[1],
        );
    }

    /// Standalone binding-check unit test — runs the assertion without
    /// needing the GPU adapter. If the .ability files drift from the
    /// .sim hand-mirrored constants this test fails first.
    #[test]
    fn binding_check_passes() {
        binding_check::assert_ability_registry_matches_sim_constants();
    }

    /// #138 Step 2 pin — Wave 1 ability programs as construction-helper
    /// references for the dispatcher path.
    ///
    /// Loads each ability's `AbilityProgram` from the actual
    /// `AbilityRegistry` (built from the .ability source files via
    /// `binding_check::build_duel_abilities_registry`) so the test
    /// catches any drift from the .ability source automatically — no
    /// hand-mirrored constants.
    ///
    /// All three pin the apply_program output the dispatcher path
    /// will need to reproduce when #138 swaps the verb body's
    /// `emit X` with `apply_ability <id>`:
    ///   - Strike   → ApplyEvent::Damage  (amount = strike.damage)
    ///   - ShieldUp → ApplyEvent::Shield  (amount = shieldup.shield)
    ///   - Mend     → ApplyEvent::Heal    (amount = mend.heal)
    #[test]
    fn strike_apply_program_produces_expected_damage_event() {
        use engine::ability::apply::{apply_program, ApplyEvent};
        use engine::ability::program::CasterStats;
        use engine::ids::AgentId;

        let built = binding_check::build_duel_abilities_registry();
        let strike_id = *built.names.get("Strike").expect("Strike registered");
        let strike = built.registry.get(strike_id).expect("Strike resolves");

        let caster = AgentId::new(1).expect("AgentId::new");
        let target = AgentId::new(2).expect("AgentId::new");
        let events = apply_program(
            strike, caster, target, 0, 0xCAFE_F00D, &CasterStats::default(),
            &CasterStats::default(),
        );

        assert_eq!(events.len(), 1, "Strike has one Damage effect");
        match events[0] {
            ApplyEvent::Damage { source, target: t, amount } => {
                assert_eq!(source, caster);
                assert_eq!(t, target);
                // Mirrors the hand-mirrored .sim verb constant; the
                // binding check enforces .ability → .sim agreement.
                assert_eq!(amount, 30.0);
            }
            other => panic!("expected ApplyEvent::Damage, got {:?}", other),
        }
    }

    #[test]
    fn shieldup_apply_program_produces_expected_shield_event() {
        use engine::ability::apply::{apply_program, ApplyEvent};
        use engine::ability::program::CasterStats;
        use engine::ids::AgentId;

        let built = binding_check::build_duel_abilities_registry();
        let shieldup_id = *built.names.get("ShieldUp").expect("ShieldUp registered");
        let shieldup = built.registry.get(shieldup_id).expect("ShieldUp resolves");

        // Self-cast: caster == target.
        let caster = AgentId::new(1).expect("AgentId::new");
        let events = apply_program(
            shieldup, caster, caster, 0, 0xCAFE_F00D, &CasterStats::default(),
            &CasterStats::default(),
        );

        assert_eq!(events.len(), 1, "ShieldUp has one Shield effect");
        match events[0] {
            ApplyEvent::Shield { source, target, amount } => {
                assert_eq!(source, caster);
                assert_eq!(target, caster, "self-cast: target == source");
                assert_eq!(amount, 50.0);
            }
            other => panic!("expected ApplyEvent::Shield, got {:?}", other),
        }
    }

    #[test]
    fn mend_apply_program_produces_expected_heal_event() {
        use engine::ability::apply::{apply_program, ApplyEvent};
        use engine::ability::program::CasterStats;
        use engine::ids::AgentId;

        let built = binding_check::build_duel_abilities_registry();
        let mend_id = *built.names.get("Mend").expect("Mend registered");
        let mend = built.registry.get(mend_id).expect("Mend resolves");

        let caster = AgentId::new(1).expect("AgentId::new");
        let events = apply_program(
            mend, caster, caster, 0, 0xCAFE_F00D, &CasterStats::default(),
            &CasterStats::default(),
        );

        assert_eq!(events.len(), 1, "Mend has one Heal effect");
        match events[0] {
            ApplyEvent::Heal { source, target, amount } => {
                assert_eq!(source, caster);
                assert_eq!(target, caster, "self-cast: target == source");
                assert_eq!(amount, 25.0);
            }
            other => panic!("expected ApplyEvent::Heal, got {:?}", other),
        }
    }

    /// Wave 2 sibling pin — Fortify's apply_program. Same registry-loaded
    /// shape as Strike/ShieldUp/Mend; exercises the DamageModify EffectOp
    /// (target + duration_ticks + multiplier_q8 payload triple).
    ///
    /// Fortify.ability: `damage_modify 0.5 5s stacking refresh` → 0.5
    /// multiplier × 256 = 128 in q8 fixed-point; 5s × 10 ticks/s = 50
    /// duration_ticks. The stacking modifier is captured on the program
    /// but is enforced at the cascade-handler level, not by apply_program.
    #[test]
    fn fortify_apply_program_produces_expected_damage_modify_event() {
        use engine::ability::apply::{apply_program, ApplyEvent};
        use engine::ability::program::CasterStats;
        use engine::ids::AgentId;

        let built = binding_check::build_duel_abilities_registry();
        let fortify_id = *built.names.get("Fortify").expect("Fortify registered");
        let fortify = built.registry.get(fortify_id).expect("Fortify resolves");

        let caster = AgentId::new(1).expect("AgentId::new");
        let events = apply_program(
            fortify, caster, caster, 0, 0xCAFE_F00D, &CasterStats::default(),
            &CasterStats::default(),
        );

        assert_eq!(events.len(), 1, "Fortify has one DamageModify effect");
        match events[0] {
            ApplyEvent::DamageModify { target, duration_ticks, multiplier_q8 } => {
                assert_eq!(target, caster, "self-cast");
                assert_eq!(duration_ticks, 50, "5s × 10 ticks/s");
                assert_eq!(multiplier_q8, 128, "0.5 × 256 = 128 in q8");
            }
            other => panic!("expected ApplyEvent::DamageModify, got {:?}", other),
        }
    }

    /// Wave 2 sibling pin — Vampirize's apply_program. Exercises the
    /// LifeSteal EffectOp (target + duration_ticks + fraction_q8 triple).
    ///
    /// Vampirize.ability: `lifesteal 0.5 5s` → 0.5 fraction × 256 = 128
    /// in q8; 5s × 10 ticks/s = 50 duration_ticks. The cascade handler
    /// (ApplyLifestealActivation) folds the resulting LifeSteal event
    /// into per-agent lifesteal SoA fields.
    #[test]
    fn vampirize_apply_program_produces_expected_lifesteal_event() {
        use engine::ability::apply::{apply_program, ApplyEvent};
        use engine::ability::program::CasterStats;
        use engine::ids::AgentId;

        let built = binding_check::build_duel_abilities_registry();
        let vampirize_id = *built.names.get("Vampirize").expect("Vampirize registered");
        let vampirize = built.registry.get(vampirize_id).expect("Vampirize resolves");

        let caster = AgentId::new(1).expect("AgentId::new");
        let events = apply_program(
            vampirize, caster, caster, 0, 0xCAFE_F00D, &CasterStats::default(),
            &CasterStats::default(),
        );

        assert_eq!(events.len(), 1, "Vampirize has one LifeSteal effect");
        match events[0] {
            ApplyEvent::LifeSteal { target, duration_ticks, fraction_q8 } => {
                assert_eq!(target, caster, "self-cast");
                assert_eq!(duration_ticks, 50, "5s × 10 ticks/s");
                assert_eq!(fraction_q8, 128, "0.5 × 256 = 128 in q8");
            }
            other => panic!("expected ApplyEvent::LifeSteal, got {:?}", other),
        }
    }

    /// Wave 2 sibling pin — Daze's apply_program. The `chance 50%`
    /// modifier is the additional complexity vs Strike/ShieldUp/Mend:
    /// apply_program rolls `(per_agent_u32(world_seed, caster, tick,
    /// purpose) & 0xFFFF) < q16` per effect slot, so a single
    /// (caster, target, tick) call may emit ZERO or ONE Stun event
    /// depending on the seed.
    ///
    /// This pin sweeps (caster, tick) combinations and asserts:
    ///   1. program.chances[0] is populated with q16=0x8000 (50%).
    ///   2. At least one combo produces an empty event vec
    ///      (chance gate suppressed).
    ///   3. At least one combo produces an ApplyEvent::Stun
    ///      (chance gate fired).
    ///   4. Determinism: same (caster, target, tick, world_seed) →
    ///      same output (P5 keyed-PCG contract).
    ///
    /// Daze.ability: `stun 1s chance 50%` → Stun { duration_ticks: 10 }
    /// at chance q16=0x8000.
    #[test]
    fn daze_apply_program_honors_chance_gate_deterministically() {
        use engine::ability::apply::{apply_program, ApplyEvent};
        use engine::ability::program::CasterStats;
        use engine::ids::AgentId;

        let built = binding_check::build_duel_abilities_registry();
        let daze_id = *built.names.get("Daze").expect("Daze registered");
        let daze = built.registry.get(daze_id).expect("Daze resolves");

        // Pin (1): the chance modifier landed in program.chances.
        assert_eq!(
            daze.chances.get(0),
            Some(&Some(0x8000)),
            "Daze.ability `stun 1s chance 50%` must populate chances[0]=Some(0x8000) \
             (50% as q16); got {:?}",
            daze.chances,
        );

        let target = AgentId::new(99).expect("AgentId::new");
        let world_seed = 0xCAFE_F00D;

        let mut any_fire = false;
        let mut any_skip = false;
        for caster_seed in [1u32, 2, 3, 5, 7, 11, 13, 17] {
            for tick in [10u32, 50, 100, 200] {
                let caster = AgentId::new(caster_seed).expect("AgentId::new");

                // Pin (4): determinism — same input twice → same output.
                let run1 = apply_program(
                    daze, caster, target, tick as u64, world_seed,
                    &CasterStats::default(),
            &CasterStats::default(),
                );
                let run2 = apply_program(
                    daze, caster, target, tick as u64, world_seed,
                    &CasterStats::default(),
            &CasterStats::default(),
                );
                assert_eq!(
                    run1.len(), run2.len(),
                    "P5 violation: same input → different output (caster={caster_seed} tick={tick})"
                );

                match run1.as_slice() {
                    [] => any_skip = true,
                    [ApplyEvent::Stun { target: t, duration_ticks }] => {
                        assert_eq!(*t, target, "Stun targets correct agent");
                        assert_eq!(*duration_ticks, 10, "1s × 10 ticks/s");
                        any_fire = true;
                    }
                    other => panic!(
                        "expected [] or [Stun{{...}}] for Daze; got {:?} \
                         (caster={caster_seed} tick={tick})",
                        other,
                    ),
                }
            }
        }

        // Pins (2) + (3): both halves of the chance fork were exercised.
        assert!(any_fire, "no chance-fire across 32 sweep combos at 50% gate");
        assert!(any_skip, "no chance-skip across 32 sweep combos at 50% gate");
    }

    /// Snapshot before any tick must report initial state: both heroes
    /// alive at full HP (100.0), shields zero, and the renderer-visible
    /// fields (positions/creature_types/alive) populated for every
    /// slot. Guards the construction-only readback path so `viz_app`
    /// can render frame 0 instead of an empty grid.
    #[test]
    fn snapshot_after_construction_returns_initial_state() {
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 2);
        let snap = state.snapshot();
        assert_eq!(snap.positions.len(), 2, "two-agent snapshot");
        assert_eq!(snap.creature_types.len(), 2);
        assert_eq!(snap.alive.len(), 2);
        // Both alive, full-HP bucket → table entries 0 (hero_id=0) and
        // 1 (hero_id=1).
        assert_eq!(snap.alive, vec![1u32, 1u32]);
        assert_eq!(snap.creature_types, vec![0u32, 1u32]);
        // Stationary grid: A at origin, B 5 units east.
        assert_eq!(snap.positions[0], Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(snap.positions[1], Vec3::new(5.0, 0.0, 0.0));
        // HP/shield readback paths separately exposed for the harness.
        let hp = state.read_hp();
        let shield = state.read_shield_hp();
        assert_eq!(hp, vec![100.0_f32, 100.0_f32]);
        assert_eq!(shield, vec![0.0_f32, 0.0_f32]);
    }

    /// After ticking the duel forward, at least one hero's HP must
    /// have moved off 100.0 (Strike landing, Mend healing back, or
    /// ShieldUp adding buffer). Proves snapshot reflects live GPU
    /// state rather than cached construction-time values.
    #[test]
    fn snapshot_after_tick_reflects_state_change() {
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 2);
        for _ in 0..50 {
            state.step();
        }
        let snap = state.snapshot();
        let hp = state.read_hp();
        assert_eq!(snap.positions.len(), 2);
        assert!(
            (hp[0] - 100.0).abs() > 0.01 || (hp[1] - 100.0).abs() > 0.01,
            "expected HP movement after 50 ticks, got hp=[{:.2}, {:.2}]",
            hp[0], hp[1],
        );
    }

    /// Wave 2 SelfDamage E2E demo. Constructs a 1-agent fixture so
    /// Strike (which requires `target != self`) cannot fire — leaving
    /// Bleed as the only verb whose gates pass at tick 0 (hp=100.0,
    /// so Mend's `hp < 50` fails and ShieldUp's `hp < 90` fails;
    /// Bleed's `hp > 50` is the only self-target gate that holds).
    ///
    /// Trace expectations:
    ///   - tick 0: Bleed mask wins argmax (sole eligible verb), the
    ///     Bleed chronicle emits Damaged{source=0,target=0,amount=5},
    ///     ApplyDamage drains shield (0) then hp (-5). Result: hp=95.
    ///   - ticks 1..49: no gates fire; hp stays 95.
    ///   - tick 50: Bleed fires again; hp=90.
    ///
    /// After 51 step() calls (tick advancing from 0 → 50 inclusive),
    /// hp must have dropped by AT LEAST 5 (the bleed amount). Concrete
    /// expected value: 90.0 (two Bleed cycles, no shield absorption).
    /// We assert the floor (`<= 95.0`) rather than exact equality so
    /// the test stays robust to per-cycle ordering tweaks while still
    /// proving the SelfDamage flow .ability → AbilityRegistry →
    /// chronicle Damaged → ApplyDamage chain is wired end-to-end.
    #[test]
    fn bleed_drains_caster_hp_when_selected() {
        // 1-agent fixture: Strike's `target != self` gate always
        // fails, so Bleed is the only verb that can win argmax at
        // tick 0 (Mend/ShieldUp's hp-low gates also fail at hp=100).
        //
        // Wave 1.5#4 GPU scaling (2026-05-07): the dispatcher now
        // reads `scalings_per_effect` SoA + per-stat agent SoA at
        // `caster_slot` to compute `scale_bonus = Σ percent * stat`.
        // Bleed declares `self_damage 5 + 5% max_hp`; with
        // `agent_max_hp[*] = 100.0`, the dispatcher writes
        // `5 + 0.05 * 100 = 10.0` into the chronicle (= the prior
        // hand-mirrored `bleed_amount = 10.0` constant). Bleed fires
        // at tick 0 and tick 50 → hp = 100 - 10 - 10 = 80.0.
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 1);
        for _ in 0..51 {
            state.step();
        }
        let hp = state.read_hp();
        let shield = state.read_shield_hp();
        let alive = state.read_alive();
        assert_eq!(hp.len(), 1);
        assert!(
            (hp[0] - 80.0).abs() < 1e-3,
            "expected hp = 80.0 (Bleed scaled amount 10.0 fires twice at \
             ticks 0 and 50 — registry-driven scaling dispatch reads \
             5 + 5%·MaxHp = 10.0 from agent_max_hp[*]=100.0), got \
             hp={:.4}, shield={:.4}, alive={}",
            hp[0], shield[0], alive[0],
        );
        // Sanity: the agent must still be alive; Bleed is supposed
        // to be a self-cost, not a suicide.
        assert_eq!(
            alive[0], 1,
            "Bleed must not kill the caster — alive=0 after 51 ticks \
             means Bleed fired far too many times or shield_hp is \
             negative; got hp={:.2}, shield={:.2}",
            hp[0], shield[0],
        );
    }

    /// Wave 2 Execute E2E demo. Reap fires a Defeated event when its
    /// `target.hp < threshold` gate is satisfied at a tick%20==0
    /// boundary, drained by the new ApplyDefeat physics block (fused
    /// into the existing PerEvent kernel as
    /// `physics_ApplyDamage_and_ApplyHeal_and_ApplyShield_and_ApplyDefeat`).
    ///
    /// Why we engineer the HP rather than play it out: Strike's 30-
    /// damage step skips the (0, 10] HP window, so the natural duel
    /// never lands target.hp ∈ (0, 20) at a tick%20==0 boundary. The
    /// trace from `0xCAFE_F00D` shows agent A dies at tick 30 (Mend
    /// +25 then Strike −30 → hp=5 with alive=0 from Strike's inline
    /// Defeated emit). To exercise Reap specifically we override HP to
    /// 15.0 for both agents BEFORE tick 0 — Reap's gate then fires
    /// for both at tick 0:
    ///
    ///   * tick%20==0 → cooldown gate satisfied
    ///   * target.hp=15 < reap_threshold=20 → finisher gate satisfied
    ///   * target.alive && target!=self → both true in 2-agent fixture
    ///   * Reap score 500 dominates Strike (200-15=185), Mend (300),
    ///     ShieldUp (250); Bleed not eligible (hp=15 ≯ 50)
    ///
    /// **Reap-killed signal:** the Defeated event from Reap sets
    /// alive=false WITHOUT touching HP. So if Reap killed the agent,
    /// HP at death is the unmodified 15.0. If Strike had been the
    /// killer instead, HP would be at most 15-30 = -15. We therefore
    /// assert `alive==0 && hp > 0.0`, which can ONLY be produced by
    /// the Reap → Defeated → ApplyDefeat path.
    ///
    /// Cooldown=20 means Reap fires at tick 0 (the very first step);
    /// 5 steps gives plenty of margin even if argmax ordering changes.
    /// Wave 2 piece N — LifeSteal E2E demo. Verifies that the
    /// `lifesteal 0.5 5s` effect on `Vampirize.ability` makes it all
    /// the way through:
    ///
    ///   1. parser → ability_lower → AbilityProgram { effects:
    ///      [EffectOp::LifeSteal { duration_ticks: 50, fraction_q8:
    ///      128 }] } (asserted by the binding-check at construction)
    ///   2. mirrored .sim verb gate (`world.tick % cooldown_vampirize
    ///      == 0` AND `self.hp < hp_vampire_floor`)
    ///   3. Vampirize chronicle emits SetLifesteal{caster, frac_q8=128,
    ///      expires_at=tick+50}
    ///   4. ApplyLifestealActivation chronicle drains SetLifesteal into
    ///      the per-agent lifesteal SoA fields (agent_lifesteal_frac_q8
    ///      + agent_lifesteal_expires_at_tick)
    ///
    /// **Test shape:** 1-agent fixture so Strike (`target != self`)
    /// cannot fire. Agent overridden to hp=25 so the Vampirize gate
    /// (`hp < hp_vampire_floor=30`) passes. After one tick:
    ///   * lifesteal_frac_q8[0] == 128 (0.5x in q8)
    ///   * lifesteal_expires_at_tick[0] == 50 (tick 0 + duration_ticks
    ///     50)
    ///
    /// **Why the SoA-set test instead of an end-to-end heal observation:**
    /// observing the source-side heal on agent 0 requires a Damaged
    /// event with `source=agent 0` to fire WITHIN the lifesteal window
    /// (ticks 1..49). In the 1-agent fixture only Bleed can produce a
    /// self-Damaged event, but Bleed's `hp > hp_bleed_floor=50` gate is
    /// incompatible with the Vampirize `hp < 30` gate. In the 2-agent
    /// fixture the only way to land Strike from agent 0 inside the
    /// window is at tick 10, but agent 1's reciprocal Strike (score
    /// `200 - 25 = 175`) lands the same tick and kills agent 0 (hp
    /// 25 → -5) before the source-side Healed event from ApplyDamage
    /// drains in the next tick's (8a) ApplyHeal arm. The SoA-set check
    /// here proves the Vampirize → SetLifesteal → ApplyLifestealActivation
    /// → SoA chain is wired end-to-end; the source-side Healed emit on
    /// ApplyDamage is exercised by inspection (its branch executes
    /// every Damaged event but only emits when src_frac > 0 AND
    /// expires > world.tick AND bleed > 0.0, all read from the
    /// per-agent lifesteal SoA written by this test's code path).
    #[test]
    fn vampirize_heals_caster_when_dealing_damage() {
        // 1-agent fixture: Strike's `target != self` gate always fails,
        // so Vampirize is the only verb that can win argmax at tick 0
        // with hp=25 (Mend score 300 < Vampirize 350).
        // Task #138 follow-on (Vampirize, mirror of Bleed at
        // `486eb08f`) — Vampirize now flows .ability →
        // AbilityRegistry → apply_ability dispatcher →
        // EffectLifeStealApplied (kind=40) →
        // ApplyLifestealFromChronicle re-emit → SetLifesteal →
        // ApplyLifestealActivation → SoA write. The chronicle's
        // expires_at_tick is computed by the dispatcher as
        // `tick + duration_ticks` (= 0 + 50 = 50 at tick 0); the
        // re-emit ferries it verbatim into SetLifesteal's `expires_at`
        // field (no `world.tick + d` recomputation, which would
        // compound the offset).
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 1);
        // Engineer the state: agent 0 at hp=25, well under
        // hp_vampire_floor=30.
        state.override_hp_for_test(&[25.0]);
        // Lifesteal SoA must start zeroed (no lifesteal).
        let pre = state.read_lifesteal_frac_q8();
        assert_eq!(
            pre, vec![0_i32],
            "lifesteal_frac_q8 must initialise to zero — saw {:?}",
            pre,
        );
        // Tick 0 satisfies tick%80==0; Vampirize fires for agent 0.
        // The Vampirize chronicle emits EffectLifeStealApplied at step
        // (7c); ApplyLifestealFromChronicle re-emits SetLifesteal at
        // step (7j); ApplyLifestealActivation drains it inside the
        // (8a) fused kernel.
        state.step();
        let frac = state.read_lifesteal_frac_q8();
        let expires = state.read_lifesteal_expires_at_tick();
        assert_eq!(
            frac, vec![128_i32],
            "lifesteal_frac_q8 must be 128 (0.5 in q8) after Vampirize \
             fires — saw {:?}, expires={:?}",
            frac, expires,
        );
        assert_eq!(
            expires, vec![50_u32],
            "lifesteal_expires_at_tick must be tick(0) + \
             vampirize_dur(50) = 50 — saw {:?}",
            expires,
        );
        // Sanity: the agent must still be alive (Vampirize doesn't
        // damage; the SoA write goes through cleanly).
        let alive = state.read_alive();
        let hp = state.read_hp();
        assert_eq!(
            alive, vec![1_u32],
            "Vampirize must not kill the caster — saw alive={:?}, hp={:?}",
            alive, hp,
        );
        assert_eq!(
            hp, vec![25.0_f32],
            "Vampirize is a state-set verb; caster's hp must stay at \
             the post-override value (25.0) — saw {:?}",
            hp,
        );
    }

    /// Wave 2 piece N — DamageModify E2E demo. Verifies that the
    /// `damage_modify 0.5 5s` effect on `Fortify.ability` makes it all
    /// the way through:
    ///
    ///   1. parser → ability_lower → AbilityProgram { effects:
    ///      [EffectOp::DamageModify { duration_ticks: 50,
    ///      multiplier_q8: 128 }] } (asserted by the binding-check at
    ///      construction)
    ///   2. mirrored .sim verb gate (`world.tick % cooldown_fortify
    ///      == 0` AND `self.hp < hp_fortify_floor`)
    ///   3. Fortify chronicle emits SetDamageMod{target_agent, mult_q8=128,
    ///      expires_at=tick+50}
    ///   4. ApplyDamageModActivation chronicle (fused with the rest of
    ///      the PerEvent group) drains SetDamageMod into the per-agent
    ///      damage_taken_mult SoA fields
    ///   5. ApplyDamage reads target's damage_taken_mult on each
    ///      Damaged event and scales bleed by `mult_q8/256`
    ///
    /// **Test shape:** 2-agent fixture so Strike can fire from agent 1
    /// onto agent 0 within the SAME tick that Fortify activates.
    /// Agent 0 overridden to hp=65 so the Fortify gate
    /// (`hp < hp_fortify_floor=70`) passes at tick 0; agent 1 stays at
    /// hp=100 so its Fortify gate fails and it instead lands a Strike
    /// (cooldown gate tick%10==0 also satisfied at tick 0).
    ///
    /// At tick 0 the chain is:
    ///   * Fortify chronicle emits SetDamageMod{target=0, mult_q8=128,
    ///     expires_at=50}
    ///   * Fused kernel drains SetDamageMod (writes mult_q8[0]=128,
    ///     expires[0]=50) AND emits Strike's Damaged{src=1,target=0,
    ///     amount=30} via the verb_chronicle_Strike arm
    ///   * ApplyDamage standalone kernel runs AFTER the fused kernel,
    ///     reads mult_q8[0]=128 and expires[0]=50 > tick=0 → scales
    ///     bleed: bleed_raw=30, scaled=30*128/256=15, hp[0]=65-15=50
    ///
    /// **Without Fortify** the same Strike would drop hp 65 → 35
    /// (default mult_q8=256 stays at 1.0×, but expires=0 so the
    /// if-expr's else branch picks `bleed_raw` directly). **WITH
    /// Fortify** the assertion is hp[0] ≈ 50, ±a small tolerance for
    /// the q8 fixed-point rounding.
    ///
    /// Agent 1's Fortify gate fails (hp=100 ≥ 70), so it scores Strike
    /// at `200 - target.hp = 200 - 65 = 135` — wins over Bleed (75)
    /// for agent 1's argmax. Mend (300) and Vampirize (350) fail their
    /// hp-low gates at hp=100. So agent 1 strikes and the scenario
    /// surfaces both branches in the same tick.
    #[test]
    fn fortify_halves_incoming_damage() {
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 2);
        // Agent 0 at hp=65 (Fortify gate hp<70 passes; Vampirize gate
        // hp<30 fails). Agent 1 at hp=100 (Fortify gate fails so it
        // strikes instead).
        state.override_hp_for_test(&[65.0, 100.0]);
        // SoA must start at default mult_q8=256 (1.0×) AND expires=0.
        let pre_mult = state.read_damage_taken_mult_q8();
        let pre_expires = state.read_damage_taken_mult_expires_at_tick();
        assert_eq!(
            pre_mult, vec![256_i32, 256_i32],
            "damage_taken_mult_q8 must initialise to 256 (1.0×) — saw {:?}",
            pre_mult,
        );
        assert_eq!(
            pre_expires, vec![0_u32, 0_u32],
            "damage_taken_mult_expires_at_tick must initialise to 0 — \
             saw {:?}",
            pre_expires,
        );
        // Tick 0: Fortify fires for agent 0; Strike fires from agent 1
        // onto agent 0 in the same tick. The fused kernel drains
        // SetDamageMod BEFORE ApplyDamage runs (ApplyDamage is a
        // SEPARATE compute pass that follows the fused-kernel pass —
        // the cross-pass barrier guarantees the SoA write is visible).
        state.step();
        // SoA write side: agent 0's mult_q8 = 128 (0.5× in q8), expires
        // at tick 0 + fortify_dur(50) = 50. Agent 1 untouched.
        let post_mult = state.read_damage_taken_mult_q8();
        let post_expires = state.read_damage_taken_mult_expires_at_tick();
        assert_eq!(
            post_mult, vec![128_i32, 256_i32],
            "agent 0's damage_taken_mult_q8 must be 128 (0.5×) after \
             Fortify fires; agent 1 stays at default 256 (1.0×) — \
             saw {:?}, expires={:?}",
            post_mult, post_expires,
        );
        assert_eq!(
            post_expires, vec![50_u32, 0_u32],
            "agent 0's damage_taken_mult_expires must be tick(0) + \
             fortify_dur(50) = 50; agent 1 stays at 0 — saw {:?}",
            post_expires,
        );
        // Damage-scaling side: agent 0 absorbed Strike (30) scaled by
        // 0.5× → lost 15 hp → 65 - 15 = 50.0. Without Fortify the same
        // Strike would have dropped hp 65 → 35. We assert hp[0] is
        // close to 50, not 35.
        let hp = state.read_hp();
        let alive = state.read_alive();
        let shield = state.read_shield_hp();
        assert_eq!(alive, vec![1_u32, 1_u32],
            "both agents must still be alive after one tick — \
             saw alive={:?}, hp={:?}, shield={:?}",
            alive, hp, shield);
        // q8 rounding: 30.0 * 128 / 256.0 = 15.0 exactly (no rounding
        // since 30 * 128 = 3840 and 3840 / 256 = 15). Tolerance of
        // 0.001 covers any f32 IEEE-754 quirk.
        assert!(
            (hp[0] - 50.0).abs() < 0.001,
            "agent 0's hp must be ~50.0 (Fortify halved Strike's 30 \
             damage to 15) — saw hp={:?}, mult_q8={:?}, expires={:?}, \
             shield={:?}. WITHOUT Fortify hp[0] would have been 35.0.",
            hp, post_mult, post_expires, shield,
        );
    }

    #[test]
    fn reap_kills_enemy_when_below_threshold() {
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 2);
        // Engineer the state: both agents at HP=15, well under
        // reap_threshold=20.
        state.override_hp_for_test(&[15.0, 15.0]);
        // Tick 0 satisfies tick%20==0; Reap fires for both agents.
        // 5 ticks is overkill but cheap and robust.
        for _ in 0..5 {
            state.step();
        }
        let hp = state.read_hp();
        let alive = state.read_alive();
        // At least one agent must have died (alive=0). The Reap signal
        // is hp>0 at death — Strike would have driven hp to ≤-15.
        let reap_killed_a = alive[0] == 0 && hp[0] > 0.0;
        let reap_killed_b = alive[1] == 0 && hp[1] > 0.0;
        assert!(
            reap_killed_a || reap_killed_b,
            "expected Reap to kill at least one agent (alive=0 && hp>0) — \
             Strike-kill leaves hp<=0 from the inline ApplyDamage path; \
             got alive=[{}, {}], hp=[{:.2}, {:.2}]",
            alive[0], alive[1], hp[0], hp[1],
        );
    }

    /// Wave 2 piece N — Stun E2E demo + FIRST verb-status cast-gate.
    /// This test is the *acceptance* of the cast-gate: a stunned agent
    /// must NOT cast any offensive verb during the stun window. We
    /// verify that by:
    ///
    ///   1. Constructing a 2-agent fixture (both alive, hp=100).
    ///   2. Overriding agent 0's `hot_stun_expires_at_tick` to 50
    ///      BEFORE any `step()` runs. The mask kernel reads
    ///      `agents.stun_expires_at_tick(self) <= world.tick` and
    ///      gates EVERY offensive verb (Strike/ShieldUp/Mend/Bleed/
    ///      Reap/Vampirize/Fortify/Daze) on `expires <= tick`. With
    ///      expires=50 the gate FAILS for ticks 0..49.
    ///   3. Ticking 50 times. Agent 0's mask kernel sees
    ///      stun_expires=50 > tick∈[0,49] → no verb is selected;
    ///      Strike never emits a Damaged event with target=agent 1.
    ///   4. Asserting agent 1's hp is *exactly* 100.0 after 50 ticks.
    ///      Any drift means the cast-gate didn't suppress agent 0's
    ///      Strike — i.e. the new `agents.stun_expires_at_tick(self)`
    ///      read in the verb's `when` clause failed to lower or wasn't
    ///      bound to the mask kernel.
    ///
    /// Agent 1 is NOT stunned and would normally Strike agent 0 every
    /// 10 ticks, dropping agent 0's hp by 30 per cycle. We don't
    /// assert on agent 0's hp because that path is unrelated to the
    /// cast-gate (and would noise up the test). The single load-bearing
    /// signal is `agent 1's hp == 100.0`: only agent 0's Strike could
    /// have damaged agent 1, and that Strike is the thing the gate
    /// suppresses.
    ///
    /// Why expires=50 (not 49 or 51): the gate is `<=`, so
    /// `expires <= tick` is TRUE when tick==50. Setting expires=50
    /// means ticks 0..49 fail the gate (expires=50 > tick∈[0,49]) and
    /// tick 50 onward passes. We tick 50 times (ticks 0..49 are
    /// processed; the post-tick counter advances to 50 but no further
    /// step occurs), so the entire window is gate-failed. Strike
    /// would land at tick 0 if not gated (cooldown 10, tick%10==0).
    #[test]
    fn stunned_agent_skips_strike() {
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 2);
        // Agent 0 stunned until tick 50; agent 1 never stunned.
        // Agents start at hp=100, alive=1 by construction.
        state.override_stun_for_test(&[50_u32, 0_u32]);
        // Sanity: confirm the override landed on the right slot.
        let pre_stun = state.read_stun_expires_at_tick();
        assert_eq!(
            pre_stun, vec![50_u32, 0_u32],
            "stun override didn't land — expected [50, 0], saw {:?}. \
             override_stun_for_test must write the agent_stun_expires_at_tick \
             buffer; if this fails the test below means nothing.",
            pre_stun,
        );
        // Tick 50 times — agent 0's gate fails for tick∈[0,49] because
        // stun_expires=50 > tick. Agent 1's stun is 0, so its mask
        // kernel passes its own gate (0 <= tick) — but since it would
        // only Strike agent 0, agent 1's HP is unaffected by its own
        // actions. The load-bearing assertion is on agent 1's HP: any
        // Strike from agent 0 onto agent 1 would drop agent 1's HP by
        // 30 per cooldown cycle (5 strikes possible in the 50-tick
        // window).
        for _ in 0..50 {
            state.step();
        }
        let hp = state.read_hp();
        let alive = state.read_alive();
        let stun = state.read_stun_expires_at_tick();
        // Agent 1's hp must be exactly 100.0 — the cast-gate prevented
        // every one of agent 0's would-be Strikes.
        assert_eq!(
            hp[1], 100.0,
            "stun cast-gate FAILED — agent 1's hp dropped from 100 to {} \
             over 50 ticks while agent 0 was stunned. Expected exactly \
             100.0 (no Strike from agent 0 lands during the stun window). \
             Saw hp={:?}, alive={:?}, stun_expires={:?}.",
            hp[1], hp, alive, stun,
        );
        // Defence-in-depth: agent 1 must still be alive (5 strikes at
        // 30 dmg each = 150 dmg, would have killed it from hp=100).
        assert_eq!(
            alive[1], 1,
            "stun cast-gate FAILED — agent 1 died during the stun window. \
             Expected alive=1 (no damage taken). Saw hp={:?}, alive={:?}.",
            hp, alive,
        );
        // Sanity: the stun field wasn't accidentally overwritten by an
        // ApplyStun emit (no Daze fires in this fixture — the .ability
        // exists but the verb gate `target.hp > 80` only fires at the
        // 40-tick boundary, where target.hp is still 100; what matters
        // is that agent 0's preconfigured stun is still 50, not that
        // agent 1's stun stays 0).
        assert_eq!(
            stun[0], 50_u32,
            "agent 0's stun_expires_at_tick must remain at the preconfigured \
             50 throughout the test — saw {:?}. If this changed, ApplyStun \
             ran (a Daze landed) and the test isn't actually exercising the \
             preconfigured-stun path.",
            stun,
        );
    }
}
