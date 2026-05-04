//! Per-fixture runtime for `assets/sim/scripted_battle.sim` — the
//! SECOND real gameplay-shaped fixture (after duel_1v1) and the FIRST
//! sim with a one-way scripted narrative arc.
//!
//! Three phases gated on `world.tick`:
//!   - Phase 1 (ticks 0-199):   Peaceful   — ForagePeaceful (heals)
//!   - Phase 2 (ticks 200-499): Assault    — VillagerStrike + EnemyStrike
//!   - Phase 3 (ticks 500-799): Aftermath  — ForageAftermath (heals)
//!
//! ## Per-tick chain (mirrors duel_1v1)
//!
//!   1. Per-tick clears (event-ring tail, mask bitmaps, scoring out)
//!   2. Fused mask kernel — one PerPair pass writes the four per-verb
//!      mask bitmaps. Phase gates `world.tick >= L && world.tick < R`
//!      land here as predicates on the per-(actor, candidate) thread.
//!   3. Scoring kernel — argmax over the 4 rows; emits
//!      ActionSelected{actor, action_id, target} per gated agent.
//!   4. Per-verb chronicle kernels — one per action_id (Forage*,
//!      VillagerStrike, EnemyStrike) emit Healed / Damaged events.
//!   5. ApplyDamage_and_ApplyHeal — fused PerEvent kernel applies
//!      HP deltas; `if (hp <= 0) set_alive(t, false)` flips the
//!      alive bitmap on death.
//!   6. seed_indirect_0 + fold_damage_dealt + fold_healing_done
//!
//! ## Scripted-narrative interventions (CPU-side, host-driven)
//!
//! Two `set_alive` mass-toggles are applied straight to the
//! `agent_alive_buf` between ticks:
//!   - At tick 200: every Enemy slot flipped alive=0 → 1. Wakes all
//!     enemies simultaneously (assault begins).
//!   - At tick 500: every Enemy slot flipped alive=1 → 0. Quiets the
//!     assault (aftermath begins). Surviving villagers stop emitting
//!     Damaged because the `target.alive` predicate fails for every
//!     non-self candidate.
//!
//! The host-side toggles are deterministic (stamped at fixed ticks)
//! so the run is bit-stable across replays. P5 (keyed PCG) is not
//! exercised — no randomness — so this is also P5-clean.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the scripted_battle.
#[allow(dead_code)]
pub struct ScriptedBattleState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,
    /// Host-staged buffer of `1u` per agent slot — used to flip
    /// every Enemy slot's alive bit to 1 at the Phase 1→2 boundary.
    enemy_alive_one_buf: wgpu::Buffer,
    /// Host-staged buffer of `0u` per agent slot — used to flip
    /// every Enemy slot's alive bit back to 0 at the Phase 2→3 boundary.
    enemy_alive_zero_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order:
    //    ForagePeaceful=0, VillagerStrike=1, EnemyStrike=2,
    //    ForageAftermath=3) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_3_bitmap_buf: wgpu::Buffer,
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
    chronicle_forage_peaceful_cfg_buf: wgpu::Buffer,
    chronicle_villager_strike_cfg_buf: wgpu::Buffer,
    chronicle_enemy_strike_cfg_buf: wgpu::Buffer,
    chronicle_forage_aftermath_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    villager_count: u32,
    enemy_count: u32,
    seed: u64,
}

/// Phase boundaries (must match the `when` clauses in
/// `assets/sim/scripted_battle.sim`).
pub const PHASE1_END: u64 = 200;
pub const PHASE2_END: u64 = 500;
pub const PHASE3_END: u64 = 800;

impl ScriptedBattleState {
    /// Build a scripted_battle with the given villager + enemy counts.
    /// Slots [0, villager_count) are villagers (alive=1, hp=100).
    /// Slots [villager_count, villager_count + enemy_count) are enemy
    /// reserve slots (alive=0 initially; flipped alive=1 by the host
    /// at tick 200, alive=0 at tick 500).
    pub fn new(seed: u64, villager_count: u32, enemy_count: u32) -> Self {
        let agent_count = villager_count + enemy_count;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // -- Agent SoA --
        // HP starts at 80.0 for villagers (so the Phase 1 ForagePeaceful
        // gate `self.hp < 100.0` actually fires; otherwise full-health
        // villagers would never emit Healed events in P1) and 100.0 for
        // enemy reserves (so they enter P2 at full strength). alive=1
        // for villagers, alive=0 for enemy reserves. mana=100.0 for
        // everyone (unused as gate today but allocated so the SoA
        // layout matches duel_1v1's binding shape).
        let mut hp_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        for v in 0..(villager_count as usize) {
            hp_init[v] = 80.0;
        }
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let mut alive_init: Vec<u32> = vec![0u32; agent_count as usize];
        for v in 0..(villager_count as usize) {
            alive_init[v] = 1;
        }
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let mana_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Host-side staging buffers: full agent_count × u32 of `1u`
        // and `0u` respectively. The Phase 1→2 transition copies the
        // `1u` slice into the enemy slot range; the Phase 2→3
        // transition copies the `0u` slice into the same range. The
        // copy size is exactly enemy_count*4, the destination offset
        // is villager_count*4.
        let ones_init: Vec<u32> = vec![1u32; agent_count as usize];
        let enemy_alive_one_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::enemy_alive_one"),
            contents: bytemuck::cast_slice(&ones_init),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let zeros_init: Vec<u32> = vec![0u32; agent_count as usize];
        let enemy_alive_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::enemy_alive_zero"),
            contents: bytemuck::cast_slice(&zeros_init),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // -- Mask bitmaps (4 verbs) --
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
        let mask_0_bitmap_buf = mk_mask("scripted_battle_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("scripted_battle_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("scripted_battle_runtime::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("scripted_battle_runtime::mask_3_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // -- Scoring output --
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scripted_battle_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // -- Event ring + view storage --
        let event_ring = EventRing::new(&gpu, "scripted_battle_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu,
            "scripted_battle_runtime::damage_dealt",
            agent_count,
            false,
            false,
        );
        let healing_done = ViewStorage::new(
            &gpu,
            "scripted_battle_runtime::healing_done",
            agent_count,
            false,
            false,
        );

        // -- Per-kernel cfg uniforms --
        let mask_cfg_init = fused_mask_verb_ForagePeaceful::FusedMaskVerbForagePeacefulCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::mask_cfg"),
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
            label: Some("scripted_battle_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_forage_peaceful_cfg_init =
            physics_verb_chronicle_ForagePeaceful::PhysicsVerbChronicleForagePeacefulCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_forage_peaceful_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::chronicle_forage_peaceful_cfg"),
            contents: bytemuck::bytes_of(&chronicle_forage_peaceful_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_villager_strike_cfg_init =
            physics_verb_chronicle_VillagerStrike::PhysicsVerbChronicleVillagerStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_villager_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::chronicle_villager_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_villager_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_enemy_strike_cfg_init =
            physics_verb_chronicle_EnemyStrike::PhysicsVerbChronicleEnemyStrikeCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_enemy_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::chronicle_enemy_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_enemy_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_forage_aftermath_cfg_init =
            physics_verb_chronicle_ForageAftermath::PhysicsVerbChronicleForageAftermathCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let chronicle_forage_aftermath_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::chronicle_forage_aftermath_cfg"),
            contents: bytemuck::bytes_of(&chronicle_forage_aftermath_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init =
            physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scripted_battle_runtime::healing_done_cfg"),
            contents: bytemuck::bytes_of(&healing_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            enemy_alive_one_buf,
            enemy_alive_zero_buf,
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
            chronicle_forage_peaceful_cfg_buf,
            chronicle_villager_strike_cfg_buf,
            chronicle_enemy_strike_cfg_buf,
            chronicle_forage_aftermath_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            villager_count,
            enemy_count,
            seed,
        }
    }

    /// Per-source damage_dealt readback (one f32 per slot).
    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }

    /// Per-source healing_done readback (one f32 per slot).
    pub fn healing_done(&mut self) -> &[f32] {
        self.healing_done.readback(&self.gpu)
    }

    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    pub fn villager_count(&self) -> u32 { self.villager_count }
    pub fn enemy_count(&self) -> u32 { self.enemy_count }

    /// Current narrative phase based on `self.tick`.
    pub fn phase(&self) -> u8 {
        match self.tick {
            t if t < PHASE1_END => 1,
            t if t < PHASE2_END => 2,
            _ => 3,
        }
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("scripted_battle_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("scripted_battle_runtime::read_f32") },
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
            label: Some(&format!("scripted_battle_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("scripted_battle_runtime::read_u32") },
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

impl CompiledSim for ScriptedBattleState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("scripted_battle_runtime::step") },
        );

        // -- Scripted-narrative interventions (host-driven). --
        // The transitions fire on the FIRST tick of the new phase
        // (i.e. when `self.tick` already equals the boundary value
        // BEFORE we run the boundary tick — so `step()` for the tick
        // numbered PHASE1_END applies the wake just before the masks
        // for that tick are computed). This keeps the assault visible
        // in tick 200 itself.
        if self.tick == PHASE1_END && self.enemy_count > 0 {
            // Phase 1 → 2: wake every enemy slot. Copy
            // enemy_count * 4 bytes from the `1u` staging buffer
            // straight into the alive bitmap at offset
            // villager_count * 4.
            let dst_off = (self.villager_count as u64) * 4;
            let bytes = (self.enemy_count as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &self.enemy_alive_one_buf, 0,
                &self.agent_alive_buf, dst_off, bytes,
            );
        } else if self.tick == PHASE2_END && self.enemy_count > 0 {
            // Phase 2 → 3: clear every enemy slot (mass despawn).
            let dst_off = (self.villager_count as u64) * 4;
            let bytes = (self.enemy_count as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &self.enemy_alive_zero_buf, 0,
                &self.agent_alive_buf, dst_off, bytes,
            );
        }

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Generous upper bound: every alive agent could emit one
        // ActionSelected + one Damaged/Healed per tick (= agent_cap*2).
        // ApplyDamage may also emit a Defeated event per kill so
        // round up to *4 (matches duel_1v1's bound).
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [&self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf,
                    &self.mask_2_bitmap_buf, &self.mask_3_bitmap_buf] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — fused PerPair kernel.
        let mask_cfg = fused_mask_verb_ForagePeaceful::FusedMaskVerbForagePeacefulCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = fused_mask_verb_ForagePeaceful::FusedMaskVerbForagePeacefulBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_foragepeaceful(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over 4 rows.
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
            mask_3_bitmap: &self.mask_3_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Per-verb chronicle kernels — one per action_id.
        let fp_cfg = physics_verb_chronicle_ForagePeaceful::PhysicsVerbChronicleForagePeacefulCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_forage_peaceful_cfg_buf, 0, bytemuck::bytes_of(&fp_cfg),
        );
        let fp_bindings = physics_verb_chronicle_ForagePeaceful::PhysicsVerbChronicleForagePeacefulBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_forage_peaceful_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_foragepeaceful(
            &mut self.cache, &fp_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        let vs_cfg = physics_verb_chronicle_VillagerStrike::PhysicsVerbChronicleVillagerStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_villager_strike_cfg_buf, 0, bytemuck::bytes_of(&vs_cfg),
        );
        let vs_bindings = physics_verb_chronicle_VillagerStrike::PhysicsVerbChronicleVillagerStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_villager_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_villagerstrike(
            &mut self.cache, &vs_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        let es_cfg = physics_verb_chronicle_EnemyStrike::PhysicsVerbChronicleEnemyStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_enemy_strike_cfg_buf, 0, bytemuck::bytes_of(&es_cfg),
        );
        let es_bindings = physics_verb_chronicle_EnemyStrike::PhysicsVerbChronicleEnemyStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_enemy_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_enemystrike(
            &mut self.cache, &es_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        let fa_cfg = physics_verb_chronicle_ForageAftermath::PhysicsVerbChronicleForageAftermathCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_forage_aftermath_cfg_buf, 0, bytemuck::bytes_of(&fa_cfg),
        );
        let fa_bindings = physics_verb_chronicle_ForageAftermath::PhysicsVerbChronicleForageAftermathBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_forage_aftermath_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_forageaftermath(
            &mut self.cache, &fa_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) ApplyDamage_and_ApplyHeal.
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

        // (6) seed_indirect_0 + folds.
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
    // Default split for the generic harness: 40% villagers, 60% enemy
    // reserves. The dedicated `scripted_battle_app` calls
    // `ScriptedBattleState::new` directly with explicit counts.
    let villagers = (agent_count * 4) / 10;
    let enemies = agent_count - villagers;
    Box::new(ScriptedBattleState::new(seed, villagers, enemies))
}
