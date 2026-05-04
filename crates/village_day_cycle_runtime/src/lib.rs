//! Per-fixture runtime for `assets/sim/village_day_cycle.sim` —
//! the NINTH real gameplay-shaped fixture and FIRST composite
//! gameplay sim. 30 Villagers each cycle through a 4-phase day:
//!
//!   - Morning   (tick % 100 ∈ [0,30)):  Work — harvest food (mana += 1)
//!   - Midday    (tick % 100 ∈ [30,50)): Trade — small snack (mana -1, hp +3)
//!   - Evening   (tick % 100 ∈ [50,80)): Eat — full meal (mana -2, hp +15)
//!   - Night     (tick % 100 ∈ [80,100)): Rest — energy decay paused
//!
//! Energy (`hp`) drains every tick during day phases (DrainEnergy
//! verb fires on `(tick % 100) < 80 && tick % 2 == 0`); when hp ≤ 0
//! the villager starves (set_alive(false)).
//!
//! ## Per-tick chain (Conservative schedule = one kernel per op)
//!
//!   1. clear_tail + clear 5 mask bitmaps + zero scoring_output
//!   2. mask_verb_WorkHarvest..mask_verb_DrainEnergy — 5 PerPair
//!      kernels, one per verb. Each writes its own mask_<N>_bitmap.
//!   3. scoring — PerAgent argmax over the 5 phase-mutually-exclusive
//!      rows. WorkHarvest, TradeFood, EatFood, Rest are gated on
//!      disjoint tick%100 windows; DrainEnergy fires through any
//!      day-phase tick. With phase windows non-overlapping for the
//!      4 action verbs, at most 2 rows can fire per agent per tick
//!      (DrainEnergy + one of {Work, Trade, Eat, Rest}); argmax
//!      picks DrainEnergy (score 50 < 100) only when no action verb
//!      passes its cooldown gate.
//!   4. physics_verb_chronicle_{WorkHarvest, TradeFood, EatFood,
//!      Rest, DrainEnergy} — 5 PerEvent kernels, each gates on
//!      action_id == N and emits the matching event.
//!   5. physics_ApplyWork — PerEvent, reads WorkDone, writes
//!      mana = mana + 1.
//!   6. physics_ApplyTrade — PerEvent, reads TradeDone, writes
//!      mana = mana - 1, hp = hp + 3.
//!   7. physics_ApplyEat — PerEvent, reads AteFood, writes
//!      mana = mana - 2, hp = hp + 15.
//!   8. physics_ApplyEnergyDecay — PerEvent, reads EnergyDrained,
//!      writes hp = hp - 0.4. If new_hp <= 0 also writes
//!      alive = false.
//!   9. seed_indirect_0 — keeps indirect-args buffer warm.
//!  10. fold_total_work, fold_total_trades, fold_total_eats,
//!      fold_total_rests — 4 per-villager f32 accumulators
//!      (CAS+add path, P11-trivial).
//!
//! Conservative schedule is required because the Apply* handlers
//! READ the chronicle's emitted events. Default fusion would put
//! the apply ops in the same kernel pass as the chronicle, and
//! under per-event-idx single-pass semantics the apply runs BEFORE
//! the emit lands in the ring — same root cause as quest_arc_real
//! and duel_1v1.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the village day cycle.
pub struct VillageDayCycleState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// Per-agent u32 alive (1 = alive, 0 = dead). Initialised to 1.
    /// ApplyEnergyDecay writes 0 when hp <= 0 (starvation).
    agent_alive_buf: wgpu::Buffer,
    /// Per-agent f32 hp = energy. Initialised to 100.0. Drains via
    /// EnergyDrained events (-0.4 each); restored by Trade (+3) and
    /// Eat (+15).
    agent_hp_buf: wgpu::Buffer,
    /// Per-agent f32 mana = stored food. Initialised to 0.0.
    /// Incremented by WorkDone (+1), decremented by TradeDone (-1)
    /// and AteFood (-2).
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order) --
    mask_0_bitmap_buf: wgpu::Buffer, // WorkHarvest
    mask_1_bitmap_buf: wgpu::Buffer, // TradeFood
    mask_2_bitmap_buf: wgpu::Buffer, // EatFood
    mask_3_bitmap_buf: wgpu::Buffer, // Rest
    mask_4_bitmap_buf: wgpu::Buffer, // DrainEnergy
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    total_work: ViewStorage,
    total_work_cfg_buf: wgpu::Buffer,
    total_trades: ViewStorage,
    total_trades_cfg_buf: wgpu::Buffer,
    total_eats: ViewStorage,
    total_eats_cfg_buf: wgpu::Buffer,
    total_rests: ViewStorage,
    total_rests_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms (one per kernel) --
    mask_work_cfg_buf: wgpu::Buffer,
    mask_trade_cfg_buf: wgpu::Buffer,
    mask_eat_cfg_buf: wgpu::Buffer,
    mask_rest_cfg_buf: wgpu::Buffer,
    mask_drain_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_work_cfg_buf: wgpu::Buffer,
    chronicle_trade_cfg_buf: wgpu::Buffer,
    chronicle_eat_cfg_buf: wgpu::Buffer,
    chronicle_rest_cfg_buf: wgpu::Buffer,
    chronicle_drain_cfg_buf: wgpu::Buffer,
    apply_work_cfg_buf: wgpu::Buffer,
    apply_trade_cfg_buf: wgpu::Buffer,
    apply_eat_cfg_buf: wgpu::Buffer,
    apply_decay_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl VillageDayCycleState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — alive=1, hp=100 (full energy), mana=0 (no food yet).
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let hp_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let mana_init: Vec<f32> = vec![0.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Five mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("village_day_cycle_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("village_day_cycle_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("village_day_cycle_runtime::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("village_day_cycle_runtime::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("village_day_cycle_runtime::mask_4_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_day_cycle_runtime::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("village_day_cycle_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_day_cycle_runtime::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "village_day_cycle_runtime");
        let total_work = ViewStorage::new(
            &gpu,
            "village_day_cycle_runtime::total_work",
            agent_count,
            false,
            false,
        );
        let total_trades = ViewStorage::new(
            &gpu,
            "village_day_cycle_runtime::total_trades",
            agent_count,
            false,
            false,
        );
        let total_eats = ViewStorage::new(
            &gpu,
            "village_day_cycle_runtime::total_eats",
            agent_count,
            false,
            false,
        );
        let total_rests = ViewStorage::new(
            &gpu,
            "village_day_cycle_runtime::total_rests",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mk_mask_cfg = |label: &str| -> wgpu::Buffer {
            let init = mask_verb_WorkHarvest::MaskVerbWorkHarvestCfg {
                agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mask_work_cfg_buf = mk_mask_cfg("village_day_cycle_runtime::mask_work_cfg");
        let mask_trade_cfg_buf = mk_mask_cfg("village_day_cycle_runtime::mask_trade_cfg");
        let mask_eat_cfg_buf = mk_mask_cfg("village_day_cycle_runtime::mask_eat_cfg");
        let mask_rest_cfg_buf = mk_mask_cfg("village_day_cycle_runtime::mask_rest_cfg");
        let mask_drain_cfg_buf = mk_mask_cfg("village_day_cycle_runtime::mask_drain_cfg");

        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mk_per_event_cfg = |label: &str| -> wgpu::Buffer {
            let init = physics_verb_chronicle_WorkHarvest::PhysicsVerbChronicleWorkHarvestCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let chronicle_work_cfg_buf =
            mk_per_event_cfg("village_day_cycle_runtime::chronicle_work_cfg");
        let chronicle_trade_cfg_buf =
            mk_per_event_cfg("village_day_cycle_runtime::chronicle_trade_cfg");
        let chronicle_eat_cfg_buf =
            mk_per_event_cfg("village_day_cycle_runtime::chronicle_eat_cfg");
        let chronicle_rest_cfg_buf =
            mk_per_event_cfg("village_day_cycle_runtime::chronicle_rest_cfg");
        let chronicle_drain_cfg_buf =
            mk_per_event_cfg("village_day_cycle_runtime::chronicle_drain_cfg");
        let apply_work_cfg_buf = mk_per_event_cfg("village_day_cycle_runtime::apply_work_cfg");
        let apply_trade_cfg_buf = mk_per_event_cfg("village_day_cycle_runtime::apply_trade_cfg");
        let apply_eat_cfg_buf = mk_per_event_cfg("village_day_cycle_runtime::apply_eat_cfg");
        let apply_decay_cfg_buf =
            mk_per_event_cfg("village_day_cycle_runtime::apply_decay_cfg");

        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let work_cfg_init = fold_total_work::FoldTotalWorkCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let total_work_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::total_work_cfg"),
            contents: bytemuck::bytes_of(&work_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let trade_cfg_init = fold_total_trades::FoldTotalTradesCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let total_trades_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::total_trades_cfg"),
            contents: bytemuck::bytes_of(&trade_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let eat_cfg_init = fold_total_eats::FoldTotalEatsCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let total_eats_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::total_eats_cfg"),
            contents: bytemuck::bytes_of(&eat_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let rest_cfg_init = fold_total_rests::FoldTotalRestsCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let total_rests_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_day_cycle_runtime::total_rests_cfg"),
            contents: bytemuck::bytes_of(&rest_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_alive_buf,
            agent_hp_buf,
            agent_mana_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_3_bitmap_buf,
            mask_4_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            total_work,
            total_work_cfg_buf,
            total_trades,
            total_trades_cfg_buf,
            total_eats,
            total_eats_cfg_buf,
            total_rests,
            total_rests_cfg_buf,
            mask_work_cfg_buf,
            mask_trade_cfg_buf,
            mask_eat_cfg_buf,
            mask_rest_cfg_buf,
            mask_drain_cfg_buf,
            scoring_cfg_buf,
            chronicle_work_cfg_buf,
            chronicle_trade_cfg_buf,
            chronicle_eat_cfg_buf,
            chronicle_rest_cfg_buf,
            chronicle_drain_cfg_buf,
            apply_work_cfg_buf,
            apply_trade_cfg_buf,
            apply_eat_cfg_buf,
            apply_decay_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn total_work(&mut self) -> &[f32] {
        self.total_work.readback(&self.gpu)
    }
    pub fn total_trades(&mut self) -> &[f32] {
        self.total_trades.readback(&self.gpu)
    }
    pub fn total_eats(&mut self) -> &[f32] {
        self.total_eats.readback(&self.gpu)
    }
    pub fn total_rests(&mut self) -> &[f32] {
        self.total_rests.readback(&self.gpu)
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }
    pub fn read_mana(&self) -> Vec<f32> {
        self.read_f32(&self.agent_mana_buf, "mana")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("village_day_cycle_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("village_day_cycle_runtime::read_f32"),
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
            label: Some(&format!("village_day_cycle_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("village_day_cycle_runtime::read_u32"),
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

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
}

impl CompiledSim for VillageDayCycleState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("village_day_cycle_runtime::step"),
            },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 4;
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

        let tick = self.tick as u32;
        let event_count_estimate = self.agent_count * 4;

        // (2) Mask round — 5 PerPair kernels, one per verb. Each
        // writes its own mask_<N>_bitmap. mask_k=1u (TODO task-5.7
        // in the compiler), so the per-pair grid degenerates to one
        // thread per agent. Each row gates on its own phase window
        // (`tick % 100`) plus a per-row cooldown (`tick % {3,4,5,10,2}`).

        let mask_cfg_work = mask_verb_WorkHarvest::MaskVerbWorkHarvestCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_work_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_work),
        );
        dispatch::dispatch_mask_verb_workharvest(
            &mut self.cache,
            &mask_verb_WorkHarvest::MaskVerbWorkHarvestBindings {
                agent_alive: &self.agent_alive_buf,
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                cfg: &self.mask_work_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_trade = mask_verb_TradeFood::MaskVerbTradeFoodCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_trade_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_trade),
        );
        dispatch::dispatch_mask_verb_tradefood(
            &mut self.cache,
            &mask_verb_TradeFood::MaskVerbTradeFoodBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                cfg: &self.mask_trade_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_eat = mask_verb_EatFood::MaskVerbEatFoodCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_eat_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_eat),
        );
        dispatch::dispatch_mask_verb_eatfood(
            &mut self.cache,
            &mask_verb_EatFood::MaskVerbEatFoodBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_2_bitmap: &self.mask_2_bitmap_buf,
                cfg: &self.mask_eat_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_rest = mask_verb_Rest::MaskVerbRestCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_rest_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_rest),
        );
        dispatch::dispatch_mask_verb_rest(
            &mut self.cache,
            &mask_verb_Rest::MaskVerbRestBindings {
                agent_alive: &self.agent_alive_buf,
                mask_3_bitmap: &self.mask_3_bitmap_buf,
                cfg: &self.mask_rest_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_drain = mask_verb_DrainEnergy::MaskVerbDrainEnergyCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_drain_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_drain),
        );
        dispatch::dispatch_mask_verb_drainenergy(
            &mut self.cache,
            &mask_verb_DrainEnergy::MaskVerbDrainEnergyBindings {
                agent_alive: &self.agent_alive_buf,
                mask_4_bitmap: &self.mask_4_bitmap_buf,
                cfg: &self.mask_drain_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (3) Scoring — argmax over the 5 rows. Phase windows for
        // Work/Trade/Eat/Rest are mutually exclusive, so at most one
        // of them passes per tick; DrainEnergy's score (50) is below
        // the action verbs' score (100), so it only wins when no
        // action verb passed its cooldown gate this tick.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring::ScoringBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                mask_2_bitmap: &self.mask_2_bitmap_buf,
                mask_3_bitmap: &self.mask_3_bitmap_buf,
                mask_4_bitmap: &self.mask_4_bitmap_buf,
                scoring_output: &self.scoring_output_buf,
                cfg: &self.scoring_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (4) Five chronicle kernels — each gates on action_id == N
        // and emits its respective event.
        let chr_work_cfg = physics_verb_chronicle_WorkHarvest::PhysicsVerbChronicleWorkHarvestCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_work_cfg_buf, 0, bytemuck::bytes_of(&chr_work_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_workharvest(
            &mut self.cache,
            &physics_verb_chronicle_WorkHarvest::PhysicsVerbChronicleWorkHarvestBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_work_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chr_trade_cfg = physics_verb_chronicle_TradeFood::PhysicsVerbChronicleTradeFoodCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_trade_cfg_buf, 0, bytemuck::bytes_of(&chr_trade_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_tradefood(
            &mut self.cache,
            &physics_verb_chronicle_TradeFood::PhysicsVerbChronicleTradeFoodBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_trade_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chr_eat_cfg = physics_verb_chronicle_EatFood::PhysicsVerbChronicleEatFoodCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_eat_cfg_buf, 0, bytemuck::bytes_of(&chr_eat_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_eatfood(
            &mut self.cache,
            &physics_verb_chronicle_EatFood::PhysicsVerbChronicleEatFoodBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_eat_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chr_rest_cfg = physics_verb_chronicle_Rest::PhysicsVerbChronicleRestCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_rest_cfg_buf, 0, bytemuck::bytes_of(&chr_rest_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_rest(
            &mut self.cache,
            &physics_verb_chronicle_Rest::PhysicsVerbChronicleRestBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_rest_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chr_drain_cfg = physics_verb_chronicle_DrainEnergy::PhysicsVerbChronicleDrainEnergyCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_drain_cfg_buf, 0, bytemuck::bytes_of(&chr_drain_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_drainenergy(
            &mut self.cache,
            &physics_verb_chronicle_DrainEnergy::PhysicsVerbChronicleDrainEnergyBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_drain_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (5) ApplyWork — reads WorkDone, writes mana += 1.
        let aw_cfg = physics_ApplyWork::PhysicsApplyWorkCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_work_cfg_buf, 0, bytemuck::bytes_of(&aw_cfg),
        );
        dispatch::dispatch_physics_applywork(
            &mut self.cache,
            &physics_ApplyWork::PhysicsApplyWorkBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_mana: &self.agent_mana_buf,
                cfg: &self.apply_work_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (6) ApplyTrade — reads TradeDone, writes mana -= 1, hp += 3.
        let at_cfg = physics_ApplyTrade::PhysicsApplyTradeCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_trade_cfg_buf, 0, bytemuck::bytes_of(&at_cfg),
        );
        dispatch::dispatch_physics_applytrade(
            &mut self.cache,
            &physics_ApplyTrade::PhysicsApplyTradeBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_hp: &self.agent_hp_buf,
                agent_mana: &self.agent_mana_buf,
                cfg: &self.apply_trade_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (7) ApplyEat — reads AteFood, writes mana -= 2, hp += 15.
        let ae_cfg = physics_ApplyEat::PhysicsApplyEatCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_eat_cfg_buf, 0, bytemuck::bytes_of(&ae_cfg),
        );
        dispatch::dispatch_physics_applyeat(
            &mut self.cache,
            &physics_ApplyEat::PhysicsApplyEatBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_hp: &self.agent_hp_buf,
                agent_mana: &self.agent_mana_buf,
                cfg: &self.apply_eat_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (8) ApplyEnergyDecay — reads EnergyDrained, writes hp -= 0.4.
        // If new_hp <= 0 also writes alive = false.
        let ad_cfg = physics_ApplyEnergyDecay::PhysicsApplyEnergyDecayCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_decay_cfg_buf, 0, bytemuck::bytes_of(&ad_cfg),
        );
        dispatch::dispatch_physics_applyenergydecay(
            &mut self.cache,
            &physics_ApplyEnergyDecay::PhysicsApplyEnergyDecayBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_hp: &self.agent_hp_buf,
                agent_alive: &self.agent_alive_buf,
                cfg: &self.apply_decay_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (9) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg),
        );
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_indirect_0::SeedIndirect0Bindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                indirect_args_0: self.event_ring.indirect_args_0(),
                cfg: &self.seed_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (10) Folds — per-event RMW per villager slot.
        let work_cfg = fold_total_work::FoldTotalWorkCfg {
            event_count: event_count_estimate, tick, second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.total_work_cfg_buf, 0, bytemuck::bytes_of(&work_cfg),
        );
        dispatch::dispatch_fold_total_work(
            &mut self.cache,
            &fold_total_work::FoldTotalWorkBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.total_work.primary(),
                view_storage_anchor: self.total_work.anchor(),
                view_storage_ids: self.total_work.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.total_work_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let trade_cfg = fold_total_trades::FoldTotalTradesCfg {
            event_count: event_count_estimate, tick, second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.total_trades_cfg_buf, 0, bytemuck::bytes_of(&trade_cfg),
        );
        dispatch::dispatch_fold_total_trades(
            &mut self.cache,
            &fold_total_trades::FoldTotalTradesBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.total_trades.primary(),
                view_storage_anchor: self.total_trades.anchor(),
                view_storage_ids: self.total_trades.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.total_trades_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let eat_cfg = fold_total_eats::FoldTotalEatsCfg {
            event_count: event_count_estimate, tick, second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.total_eats_cfg_buf, 0, bytemuck::bytes_of(&eat_cfg),
        );
        dispatch::dispatch_fold_total_eats(
            &mut self.cache,
            &fold_total_eats::FoldTotalEatsBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.total_eats.primary(),
                view_storage_anchor: self.total_eats.anchor(),
                view_storage_ids: self.total_eats.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.total_eats_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let rest_cfg = fold_total_rests::FoldTotalRestsCfg {
            event_count: event_count_estimate, tick, second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.total_rests_cfg_buf, 0, bytemuck::bytes_of(&rest_cfg),
        );
        dispatch::dispatch_fold_total_rests(
            &mut self.cache,
            &fold_total_rests::FoldTotalRestsBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.total_rests.primary(),
                view_storage_anchor: self.total_rests.anchor(),
                view_storage_ids: self.total_rests.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.total_rests_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.total_work.mark_dirty();
        self.total_trades.mark_dirty();
        self.total_eats.mark_dirty();
        self.total_rests.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(VillageDayCycleState::new(seed, agent_count))
}
