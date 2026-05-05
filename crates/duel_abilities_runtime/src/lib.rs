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
//! Two `Combatant : Agent` entities (Hero A vs Hero B) with three
//! abilities (Strike, ShieldUp, Mend). Per-tick:
//!
//!   1. clear_tail + clear 3 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_Strike — PerPair, writes mask_0 (Strike,
//!      cooldown=10), mask_1 (ShieldUp, cooldown=40 + self.hp<90),
//!      mask_2 (Mend, cooldown=30 + self HP < 50)
//!   3. scoring — PerAgent argmax over the 3 competing rows
//!   4. physics_verb_chronicle_Strike   — gates action_id==0u, emits Damaged
//!   5. physics_verb_chronicle_ShieldUp — gates action_id==1u, emits Shielded
//!   6. physics_verb_chronicle_Mend     — gates action_id==2u, emits Healed
//!   7. physics_ApplyDamage_and_ApplyHeal_and_ApplyShield — fused PerEvent
//!      kernel that reads Damaged/Healed/Shielded events and writes
//!      per-target HP via `agents.set_hp`. On HP<=0 also sets alive=0
//!      and emits Defeated.
//!   8. seed_indirect_0
//!   9. fold_damage_dealt
//!  10. fold_healing_done
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

use engine::sim_trait::CompiledSim;
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

    // -- Mask bitmaps (one per verb in source order: Strike=0,
    //    ShieldUp=1, Mend=2) --
    mask_0_bitmap_buf: wgpu::Buffer, // Strike
    mask_1_bitmap_buf: wgpu::Buffer, // ShieldUp
    mask_2_bitmap_buf: wgpu::Buffer, // Mend
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
    chronicle_shieldup_cfg_buf: wgpu::Buffer,
    chronicle_mend_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
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
        let mask_0_bitmap_buf = mk_mask("duel_abilities_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("duel_abilities_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("duel_abilities_runtime::mask_2_bitmap");
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
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
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
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal_and_ApplyShield::PhysicsApplyDamageAndApplyHealAndApplyShieldCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_abilities_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
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
            chronicle_shieldup_cfg_buf,
            chronicle_mend_cfg_buf,
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
}

impl CompiledSim for DuelAbilitiesState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_abilities_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
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
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Strike chronicle.
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

        // (7) ApplyDamage + ApplyHeal + ApplyShield — fused PerEvent kernel.
        let event_count_estimate = self.agent_count * 4;
        let apply_cfg = physics_ApplyDamage_and_ApplyHeal_and_ApplyShield::PhysicsApplyDamageAndApplyHealAndApplyShieldCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyDamage_and_ApplyHeal_and_ApplyShield::PhysicsApplyDamageAndApplyHealAndApplyShieldBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage_and_applyheal_and_applyshield(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
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
}
