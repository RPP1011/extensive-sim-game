//! Per-fixture runtime for `assets/sim/duel_1v1.sim` — the FIRST
//! real gameplay-shaped fixture (not a coverage probe).
//!
//! Two `Combatant : Agent` entities (Hero A vs Hero B) with three
//! abilities (Strike, Spell, Heal). Per-tick chain:
//!
//!   1. clear_tail + clear 3 mask bitmaps + zero scoring_output
//!   2. fused_mask_verb_Strike — PerPair, writes mask_0 (Strike,
//!      cooldown=2), mask_1 (Spell, cooldown=5 + mana check),
//!      mask_2 (Heal, cooldown=4 + self HP < threshold)
//!   3. scoring — PerAgent argmax over the 3 competing rows;
//!      emits one ActionSelected{actor, action_id, target} per
//!      gated agent
//!   4. physics_verb_chronicle_Strike — gates on action_id==0u,
//!      emits Damaged{source, target, amount=12.0}
//!   5. physics_verb_chronicle_Spell — gates on action_id==1u,
//!      emits Damaged{source, target, amount=25.0}
//!   6. physics_verb_chronicle_Heal — gates on action_id==2u,
//!      emits Healed{source, target=self, amount=18.0}
//!   7. physics_ApplyDamage_and_ApplyHeal — fused PerEvent kernel
//!      that reads Damaged/Healed events and writes per-target HP
//!      via the new `agents.set_hp` setter (registered + non-self
//!      target lifted in this commit). On HP<=0 also sets alive=0
//!      and emits Defeated.
//!   8. seed_indirect_0 — keeps indirect-args buffer warm
//!   9. fold_damage_dealt — per-source f32 accumulator
//!  10. fold_healing_done — per-source f32 accumulator
//!
//! ## Observable
//!
//! After up to 200 ticks (or when a Defeated event fires, whichever
//! comes first):
//!   - hp[A], hp[B] — per-tick traces show HP draining from 100
//!   - alive[A], alive[B] — one drops to 0 when a hero is defeated
//!   - damage_dealt[A] + damage_dealt[B] > 0 (combat fired)
//!   - the survivor (either A or B) declares the winner
//!
//! ## Discovery doc
//!
//! `docs/superpowers/notes/2026-05-04-duel_1v1.md` — gap punch list:
//!   - Closed: agents.set_{hp,alive,mana} + non-self target setter
//!   - Closed: PerPair mask kernel preamble bind `tick`
//!   - Closed: WGSL bool LHS rewrite (no `(x != 0u) = ...`)
//!   - Deferred: P6 violation false-positive for @phase(post)
//!   - Deferred: cycle in read/write graph (Damaged → ApplyDamage
//!     writes hp → Spell scoring reads hp)
//!   - Deferred: Heal verb's `score (... - self.hp)` lowers
//!     `self.hp` to `agent_hp[target_expr_<N>]` (binder shadowing
//!     in pair-field scoring)

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the duel.
pub struct Duel1v1State {
    gpu: GpuContext,

    // -- Agent SoA --
    /// Per-agent f32 HP. Initialised to 100.0 for both Combatants;
    /// ApplyDamage subtracts amount per Damaged event, ApplyHeal
    /// adds per Healed.
    agent_hp_buf: wgpu::Buffer,
    /// Per-agent u32 alive (1 = alive, 0 = dead). Initialised to 1
    /// for both. ApplyDamage writes 0 when HP <= 0.
    agent_alive_buf: wgpu::Buffer,
    /// Per-agent f32 mana. Initialised to 100.0 for both. Spell
    /// `when` clause checks `self.mana >= config.combat.spell_cost`
    /// (= 30.0). NOT decremented by the cast today (the verb body
    /// only emits Damaged; mana cost is read-only gate). A future
    /// extension would add `agents.set_mana(self, mana - cost)` to
    /// the Spell chronicle.
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order: Strike=0,
    //    Spell=1, Heal=2) --
    mask_0_bitmap_buf: wgpu::Buffer, // Strike
    mask_1_bitmap_buf: wgpu::Buffer, // Spell
    mask_2_bitmap_buf: wgpu::Buffer, // Heal
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent: [best_action,
    //    best_target, bitcast<u32>(best_utility), 0]) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-source damage_dealt accumulator (f32 per agent slot).
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,
    /// Per-source healing_done accumulator (f32 per agent slot).
    healing_done: ViewStorage,
    healing_done_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_strike_cfg_buf: wgpu::Buffer,
    chronicle_spell_cfg_buf: wgpu::Buffer,
    chronicle_heal_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl Duel1v1State {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — HP=100.0, alive=1, mana=100.0 for every slot.
        let hp_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let mana_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::agent_mana"),
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
        let mask_0_bitmap_buf = mk_mask("duel_1v1_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("duel_1v1_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("duel_1v1_runtime::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent. Cleared each tick to
        // avoid stale entries from the previous tick.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_1v1_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "duel_1v1_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "duel_1v1_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "duel_1v1_runtime::healing_done",
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
            label: Some("duel_1v1_runtime::mask_cfg"),
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
            label: Some("duel_1v1_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_strike_cfg_init =
            physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_spell_cfg_init =
            physics_verb_chronicle_Spell::PhysicsVerbChronicleSpellCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_spell_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::chronicle_spell_cfg"),
            contents: bytemuck::bytes_of(&chronicle_spell_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_heal_cfg_init =
            physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::chronicle_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("duel_1v1_runtime::healing_done_cfg"),
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
            chronicle_spell_cfg_buf,
            chronicle_heal_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-source damage_dealt readback (one f32 per Combatant).
    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    /// Per-source healing_done readback (one f32 per Combatant).
    pub fn healing_done(&mut self) -> &[f32] {
        self.healing_done.readback(&self.gpu)
    }

    /// Per-agent HP readback (allocates a staging buffer + maps).
    /// Used by the duel_1v1_app harness for per-tick traces.
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    /// Per-agent alive readback (1 = alive, 0 = dead).
    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    /// Per-agent scoring output (4 × u32 per agent: best_action,
    /// best_target, bitcast<u32>(best_utility), 0). Useful for
    /// diagnosing which row won argmax per agent per tick.
    pub fn read_scoring_output(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("duel_1v1_runtime::scoring_output_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_1v1_runtime::read_scoring") },
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
            label: Some(&format!("duel_1v1_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_1v1_runtime::read_f32") },
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
            label: Some(&format!("duel_1v1_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_1v1_runtime::read_u32") },
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

impl CompiledSim for Duel1v1State {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("duel_1v1_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // The duel is a stochastic / conditional producer (Strike +
        // Spell + Heal fire on cooldown gates and HP threshold), so
        // per-tick emit count is variable. Without a header clear,
        // stale `kind=1u` (Damaged) slots from previous ticks would
        // be re-folded by fold_damage_dealt and re-applied by the
        // ApplyDamage chronicle, double-counting damage. Bound the
        // clear at agent_count*4 (matches event_count_estimate
        // below — the upper bound on producer slots per tick).
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

        // (2) Mask round — one fused PerPair kernel writes all 3
        // mask bitmaps. With agent_count=2 the PerPair grid is 2*2=4
        // (agent, candidate) pairs but the mask emitter today
        // hardcodes mask_k=1u (TODO task-5.7), so only cand=0 is
        // checked per actor. For the duel this is fine since both
        // combatants are slot 0 or slot 1 and `target != self`
        // routes correctly.
        let mask_cfg = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_Strike::FusedMaskVerbStrikeBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        // The PerPair mask body now derives `mask_0_k = cfg.agent_cap`
        // (compiler kernel.rs change) so each thread visits one
        // (agent, candidate) pair. Dispatch agent_cap² threads to
        // cover the full pair grid. With agent_count=2 that's 4
        // threads — well under one workgroup.
        let pair_thread_count = self.agent_count.saturating_mul(self.agent_count);
        dispatch::dispatch_fused_mask_verb_strike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            pair_thread_count,
        );

        // (3) Scoring — argmax over the 3 rows. Emits one
        // ActionSelected per gated agent. Spell row reads
        // agent_hp[per_pair_candidate] (target.hp) for finisher
        // urgency.
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
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Strike chronicle — gates on action_id==0, emits Damaged.
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

        // (5) Spell chronicle — gates on action_id==1, emits Damaged.
        let spell_cfg = physics_verb_chronicle_Spell::PhysicsVerbChronicleSpellCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_spell_cfg_buf, 0, bytemuck::bytes_of(&spell_cfg),
        );
        let spell_bindings = physics_verb_chronicle_Spell::PhysicsVerbChronicleSpellBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_spell_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_spell(
            &mut self.cache, &spell_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (6) Heal chronicle — gates on action_id==2, emits Healed.
        let heal_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
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

        // (7) ApplyDamage_and_ApplyHeal — fused PerEvent kernel.
        // Reads Damaged (kind=1u) + Healed (kind=2u) events and
        // writes per-target HP via the new agents.set_hp setter.
        // event_count sized for ActionSelected + Damaged/Healed
        // emissions per tick; over-provision is safe (the kernel's
        // per-handler tag check ignores foreign kinds).
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

        // (9) fold_damage_dealt — RMW per Damaged event (kind=1u).
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

        // (10) fold_healing_done — RMW per Healed event (kind=2u).
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
    Box::new(Duel1v1State::new(seed, agent_count))
}
