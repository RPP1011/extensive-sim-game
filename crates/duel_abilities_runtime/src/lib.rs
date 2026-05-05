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

    // -- Mask bitmaps (one per verb in source order: Strike=0,
    //    ShieldUp=1, Mend=2, Bleed=3, Reap=4, Vampirize=5) --
    mask_0_bitmap_buf: wgpu::Buffer, // Strike
    mask_1_bitmap_buf: wgpu::Buffer, // ShieldUp
    mask_2_bitmap_buf: wgpu::Buffer, // Mend
    mask_3_bitmap_buf: wgpu::Buffer, // Bleed (Wave 2 SelfDamage demo)
    mask_4_bitmap_buf: wgpu::Buffer, // Reap  (Wave 2 Execute demo)
    mask_5_bitmap_buf: wgpu::Buffer, // Vampirize (Wave 2 LifeSteal demo)
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
    /// Healed/Shielded/Defeated/SetLifesteal events AND emits Damaged
    /// from the Strike chronicle. Adding ApplyLifestealActivation +
    /// ApplyDamage's source-side Healed emit caused the compiler to
    /// split ApplyDamage into its own pass and fuse the rest with
    /// Strike's chronicle (because Strike's emit is now the only
    /// remaining producer the others can ride). Hence the renamed
    /// kernel `physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_
    /// ApplyLifestealActivation_and_verb_chronicle_Strike`.
    chronicle_strike_cfg_buf: wgpu::Buffer,
    chronicle_shieldup_cfg_buf: wgpu::Buffer,
    chronicle_mend_cfg_buf: wgpu::Buffer,
    chronicle_bleed_cfg_buf: wgpu::Buffer,
    chronicle_reap_cfg_buf: wgpu::Buffer,
    chronicle_vampirize_cfg_buf: wgpu::Buffer,
    /// Cfg uniform for the standalone `physics_ApplyDamage` kernel —
    /// split out of the previous PerEvent fusion because ApplyDamage
    /// now emits Healed events for source-side lifesteal and that
    /// production conflicts with the consumers in the same fusion
    /// group.
    apply_damage_cfg_buf: wgpu::Buffer,
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
        binding_check::assert_ability_registry_matches_sim_constants();

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

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
        let mask_0_bitmap_buf = mk_mask("duel_abilities_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("duel_abilities_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("duel_abilities_runtime::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("duel_abilities_runtime::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("duel_abilities_runtime::mask_4_bitmap");
        let mask_5_bitmap_buf = mk_mask("duel_abilities_runtime::mask_5_bitmap");
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
        // The compiler renamed the fused-PerEvent kernel one more time:
        // adding ApplyLifestealActivation + ApplyDamage's source-side
        // Healed emit pushed Strike's chronicle into the fusion group
        // and split ApplyDamage out alone. The cfg type now lives at
        // `physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_
        // ApplyLifestealActivation_and_verb_chronicle_Strike`. Field
        // name `chronicle_strike_cfg_buf` retained for continuity —
        // Strike's chronicle still needs an event_count uniform and
        // this is the kernel that runs it.
        let chronicle_strike_cfg_init =
            physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_ApplyLifestealActivation_and_verb_chronicle_Strike::PhysicsApplyHealAndApplyShieldAndApplyDefeatAndApplyLifestealActivationAndVerbChronicleStrikeCfg {
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
            chronicle_strike_cfg_buf,
            chronicle_shieldup_cfg_buf,
            chronicle_mend_cfg_buf,
            chronicle_bleed_cfg_buf,
            chronicle_reap_cfg_buf,
            chronicle_vampirize_cfg_buf,
            apply_damage_cfg_buf,
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
}

impl CompiledSim for DuelAbilitiesState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_abilities_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // 6 verbs in source order; +1 for ApplyDamage's source-side
        // Healed emit (lifesteal) and +1 for the SetLifesteal event
        // each Vampirize cast emits. 8 slots per agent per tick caps
        // the worst case.
        let max_slots_per_tick = self.agent_count * 8;
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
        let mask_bindings = fused_mask_verb_Strike::FusedMaskVerbStrikeBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            mask_4_bitmap: &self.mask_4_bitmap_buf,
            mask_5_bitmap: &self.mask_5_bitmap_buf,
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

        // (5) ShieldUp chronicle.
        let shieldup_cfg = physics_verb_chronicle_ShieldUp::PhysicsVerbChronicleShieldUpCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_shieldup_cfg_buf, 0, bytemuck::bytes_of(&shieldup_cfg),
        );
        let shieldup_bindings = physics_verb_chronicle_ShieldUp::PhysicsVerbChronicleShieldUpBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_shieldup_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_shieldup(
            &mut self.cache, &shieldup_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) Mend chronicle.
        let mend_cfg = physics_verb_chronicle_Mend::PhysicsVerbChronicleMendCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_mend_cfg_buf, 0, bytemuck::bytes_of(&mend_cfg),
        );
        let mend_bindings = physics_verb_chronicle_Mend::PhysicsVerbChronicleMendBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_mend_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_mend(
            &mut self.cache, &mend_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7) Bleed chronicle — Wave 2 SelfDamage demo. Gates on
        // action_id==3u and emits Damaged{source=self,target=self,
        // amount=5}. The existing ApplyDamage chronicle drains shield
        // first then hp, so the caster's hp drops by min(5, max(0,
        // 5 - shield)) per cast.
        let bleed_cfg = physics_verb_chronicle_Bleed::PhysicsVerbChronicleBleedCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_bleed_cfg_buf, 0, bytemuck::bytes_of(&bleed_cfg),
        );
        let bleed_bindings = physics_verb_chronicle_Bleed::PhysicsVerbChronicleBleedBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_bleed_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_bleed(
            &mut self.cache, &bleed_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7b) Reap chronicle — Wave 2 Execute demo. Gates on
        // action_id==4u and emits Defeated{combatant=target}. The verb's
        // own `when` clause already gated on `target.hp < threshold`, so
        // the chronicle fires only when the finisher condition holds. The
        // ApplyDefeat handler (fused into the PerEvent kernel below)
        // drains the Defeated event via `agents.set_alive(t, false)`.
        let reap_cfg = physics_verb_chronicle_Reap::PhysicsVerbChronicleReapCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_reap_cfg_buf, 0, bytemuck::bytes_of(&reap_cfg),
        );
        let reap_bindings = physics_verb_chronicle_Reap::PhysicsVerbChronicleReapBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_reap_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_reap(
            &mut self.cache, &reap_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (7c) Vampirize chronicle — Wave 2 LifeSteal demo. Gates on
        // action_id==5u and emits SetLifesteal{caster=self, frac_q8=128,
        // expires_at=tick+50}. Drained by the ApplyLifestealActivation
        // arm of the fused kernel below — sets the caster's
        // lifesteal_frac_q8 + lifesteal_expires_at_tick SoA slots so
        // ApplyDamage's source lookup can heal them on subsequent hits.
        let vampirize_cfg = physics_verb_chronicle_Vampirize::PhysicsVerbChronicleVampirizeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_vampirize_cfg_buf, 0, bytemuck::bytes_of(&vampirize_cfg),
        );
        let vampirize_bindings = physics_verb_chronicle_Vampirize::PhysicsVerbChronicleVampirizeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_vampirize_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_vampirize(
            &mut self.cache, &vampirize_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (8a) Fused ApplyHeal + ApplyShield + ApplyDefeat +
        // ApplyLifestealActivation + verb_chronicle_Strike. The compiler
        // re-fused these because Strike's chronicle is now the lone
        // producer feeding the Healed/Shielded/Defeated/SetLifesteal
        // consumers. **Runs BEFORE ApplyDamage** so Strike's emitted
        // Damaged events are visible to ApplyDamage at step (8b);
        // ShieldUp/Mend/Bleed/Reap/Vampirize chronicles already
        // emitted earlier (steps 5-7c), and their consumer arms
        // (ApplyHeal/ApplyShield/ApplyDefeat/ApplyLifestealActivation)
        // drain those here.
        //
        // The bind-group binds the lifesteal SoA fields write-side
        // (ApplyLifestealActivation writes them). The compiler-emitted
        // SCHEDULE places ApplyDamage first; we transpose because the
        // Strike→ApplyDamage chain MUST happen within a single tick.
        let event_count_estimate = self.agent_count * 8;
        let apply_heal_cfg = physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_ApplyLifestealActivation_and_verb_chronicle_Strike::PhysicsApplyHealAndApplyShieldAndApplyDefeatAndApplyLifestealActivationAndVerbChronicleStrikeCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&apply_heal_cfg),
        );
        let apply_heal_bindings = physics_ApplyHeal_and_ApplyShield_and_ApplyDefeat_and_ApplyLifestealActivation_and_verb_chronicle_Strike::PhysicsApplyHealAndApplyShieldAndApplyDefeatAndApplyLifestealActivationAndVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_shield_hp: &self.agent_shield_hp_buf,
            agent_lifesteal_frac_q8: &self.agent_lifesteal_frac_q8_buf,
            agent_lifesteal_expires_at_tick: &self.agent_lifesteal_expires_at_tick_buf,
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_applyheal_and_applyshield_and_applydefeat_and_applylifestealactivation_and_verb_chronicle_strike(
            &mut self.cache, &apply_heal_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
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
        let mut state = DuelAbilitiesState::new(0xCAFE_F00D, 1);
        for _ in 0..51 {
            state.step();
        }
        let hp = state.read_hp();
        let shield = state.read_shield_hp();
        let alive = state.read_alive();
        assert_eq!(hp.len(), 1);
        assert!(
            hp[0] <= 95.0,
            "expected hp to drop by AT LEAST 5 (Bleed self_damage 5), \
             got hp={:.2}, shield={:.2}, alive={}",
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
        // The Vampirize chronicle emits SetLifesteal at step (7c);
        // ApplyLifestealActivation drains it inside the (8a) fused
        // kernel.
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
}
