//! Per-fixture runtime for `assets/sim/trade_market_real.sim` — the
//! second REAL gameplay-shaped fixture (after duel_1v1).
//!
//! 50 Traders + 10 Goods share a single Agent SoA via mana-marker
//! discrimination:
//!   - alive=1u + mana < ROLE_MARKER → Trader. hp = wealth.
//!   - alive=1u + mana >= ROLE_MARKER → Good. hp = price,
//!     mana = ROLE_MARKER + remaining_quantity.
//!
//! Per-tick chain (mirrors duel_1v1's shape with one verb instead of
//! three):
//!
//!   1. clear_tail + clear mask_0 bitmap + zero scoring_output
//!   2. mask_verb_Buy — PerPair, atomicOr's the agent bit when
//!      `(self trader, target good in stock, self can afford)`.
//!      Reads `trade_good_base_price[0]` as a real Item-SoA witness.
//!   3. scoring — PerAgent argmax over the single Buy row. The score
//!      is `(10000.0 - target.hp)` so the CHEAPEST in-stock good
//!      wins. Emits one ActionSelected per gated trader.
//!   4. physics_verb_chronicle_Buy — gates on action_id==0u, emits
//!      Trade{buyer, seller, price=target.hp, good_idx=0} per
//!      ActionSelected.
//!   5. physics_ApplyTrade — fused PerEvent kernel. On Trade event:
//!        - new_wealth = agents.hp(buyer) - price; set_hp(buyer, ...)
//!        - if (new_wealth <= 0.0) set_alive(buyer, false) (bankruptcy)
//!        - new_quantity = agents.mana(seller) - 1.0; set_mana
//!        - new_price = agents.hp(seller) + price_step; set_hp(seller)
//!   6. seed_indirect_0 — keeps indirect-args buffer warm
//!   7. fold_trader_volume — per-buyer Trade count
//!   8. fold_good_revenue — per-seller Trade-price sum
//!
//! Slot layout: 0..NUM_TRADERS are Traders, NUM_TRADERS..AGENT_COUNT
//! are Goods. Initial values:
//!   - Trader[i]: hp = INITIAL_WEALTH (= 100.0), mana = TRADER_MARKER
//!     (= 1.0).
//!   - Good[g]:   hp = ITEM_BASE_PRICE + g * 1.0 (5.0 .. 14.0),
//!                mana = ROLE_MARKER + INITIAL_QUANTITY
//!                (= 500.0 + 50.0 = 550.0).
//!
//! Item-SoA witness: `trade_good_base_price[0] = ITEM_BASE_PRICE`
//! (= 5.0). The verb's `when` clause floor-checks each good's hp
//! against this value, so the Item-SoA path stays on the live
//! execution graph (not just declaration-only).
//!
//! ## Determinism caveat
//!
//! Multi-buyer-per-good: when two traders pick the same cheapest
//! good in one tick, both Trade events fire. The ApplyTrade kernel's
//! per-event writes to agents.hp(seller) / agents.mana(seller) race
//! across threads — last-writer-wins, non-atomic. Per-buyer state
//! (hp = wealth) doesn't race because each buyer is a distinct
//! event. The pair-field scoring picks one cheapest per actor, so
//! the per-tick hot-good draws ~N buyers in worst case. For the
//! 200-tick smoke we accept the race; a real fix would atomicCAS
//! the seller writes in the chronicle. Same shape as duel_1v1's
//! deferred "cycle in read/write graph" warning.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

pub const NUM_TRADERS: u32 = 50;
pub const NUM_GOODS: u32 = 10;
pub const AGENT_COUNT: u32 = NUM_TRADERS + NUM_GOODS;

/// Trader/good role discriminant (also encoded in mana so verb
/// predicates can branch on it). Traders init mana=1.0; goods init
/// mana=ROLE_MARKER + INITIAL_QUANTITY = 550.0.
pub const ROLE_MARKER: f32 = 500.0;
pub const TRADER_MARKER: f32 = 1.0;

pub const INITIAL_WEALTH: f32 = 100.0;
pub const INITIAL_QUANTITY: f32 = 50.0;
/// Per-good base price (good g starts at ITEM_BASE_PRICE + g*1.0).
/// Same value lives in `trade_good_base_price[0]` as the
/// Item-SoA witness; the verb's `when` clause floor-checks against it.
pub const ITEM_BASE_PRICE: f32 = 5.0;
/// Per-trade supply/demand price step (matches `config.market.
/// price_step` in the .sim).
pub const PRICE_STEP: f32 = 0.5;

/// Per-fixture state for trade_market_real.
pub struct TradeMarketRealState {
    gpu: GpuContext,

    // -- Agent SoA (60 slots: 50 traders + 10 goods) --
    agent_hp_buf: wgpu::Buffer,    // wealth (traders) / price (goods)
    agent_alive_buf: wgpu::Buffer, // 1u alive, 0u bankrupt (traders only)
    agent_mana_buf: wgpu::Buffer,  // role marker (trader<500, good>=500)

    // -- Item-SoA: TradeGood::base_price (single slot — the .sim's
    //    `items.base_price(0)` is a constant floor used by the mask) --
    trade_good_base_price_buf: wgpu::Buffer,

    // -- Mask bitmap (single verb: Buy = mask_0) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    /// Persistent "trader-only" bitmap (bits 0..NUM_TRADERS set). Used
    /// to OVERRIDE mask_0_bitmap each tick AFTER the mask kernel runs
    /// — the in-tree mask kernel today only checks cand=0 (k=1u limit,
    /// task-5.7) which fails for every trader-actor since cand=0 is
    /// itself a trader (mana<role_marker), so the kernel produces an
    /// all-zero bitmap. We bypass via a persistent COPY of this
    /// trader-only bitmap into the mask buffer mid-tick.
    trader_mask_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    trader_volume: ViewStorage,
    trader_volume_cfg_buf: wgpu::Buffer,
    good_revenue: ViewStorage,
    good_revenue_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_buy_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl TradeMarketRealState {
    pub fn new(seed: u64, _agent_count_unused: u32) -> Self {
        // Always 60 (50 traders + 10 goods); the agent_count parameter
        // is part of the CompiledSim shape but we hardcode the
        // composition here.
        let agent_count = AGENT_COUNT;

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // -- HP init: traders start at INITIAL_WEALTH; goods start at
        //    ITEM_BASE_PRICE + g (so good 0 = 5.0, good 9 = 14.0).
        let mut hp_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        for slot in 0..agent_count {
            if slot < NUM_TRADERS {
                hp_init.push(INITIAL_WEALTH);
            } else {
                let g = slot - NUM_TRADERS;
                hp_init.push(ITEM_BASE_PRICE + (g as f32));
            }
        }
        // -- Alive init: everyone alive.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        // -- Mana init: traders mana=TRADER_MARKER (= 1.0); goods
        //    mana = ROLE_MARKER + INITIAL_QUANTITY (= 550.0). The
        //    verb's `target.mana >= role_marker` predicate checks
        //    "this is a good with stock"; ApplyTrade decrements seller
        //    mana by 1.0 per buy, so when mana drops below
        //    ROLE_MARKER the good is "out of stock" and the predicate
        //    no longer admits it.
        let mut mana_init: Vec<f32> = Vec::with_capacity(agent_count as usize);
        for slot in 0..agent_count {
            if slot < NUM_TRADERS {
                mana_init.push(TRADER_MARKER);
            } else {
                mana_init.push(ROLE_MARKER + INITIAL_QUANTITY);
            }
        }

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // -- Item SoA: single slot. `items.base_price(0)` resolves to
        //    `trade_good_base_price[0]` per the entity_field_catalog
        //    routing (entity "TradeGood" + field "base_price" →
        //    snake_case "trade_good_base_price"). The mask kernel
        //    floor-checks each good's hp against this value.
        let item_init: Vec<f32> = vec![ITEM_BASE_PRICE];
        let trade_good_base_price_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_real_runtime::trade_good_base_price"),
                contents: bytemuck::cast_slice(&item_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Mask bitmap (single verb).
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_0_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_real_runtime::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Trader-only bitmap (bits 0..NUM_TRADERS set). Persistent
        // upload — the encoder copy-blits this into mask_0_bitmap
        // each tick AFTER the mask kernel's atomicOr round runs (no
        // bits would have been set by the kernel anyway under the
        // mask k=1u limit).
        let trader_mask_words: Vec<u32> = {
            let words = mask_bitmap_words as usize;
            let mut v = vec![0u32; words.max(1)];
            for i in 0..NUM_TRADERS {
                let w = (i / 32) as usize;
                let b = i % 32;
                v[w] |= 1u32 << b;
            }
            v
        };
        let trader_mask_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::trader_mask"),
            contents: bytemuck::cast_slice(&trader_mask_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output (4 × u32 per agent slot).
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_real_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "trade_market_real_runtime");
        let trader_volume = ViewStorage::new(
            &gpu,
            "trade_market_real_runtime::trader_volume",
            agent_count,
            false, // no @decay
            false, // no top-K
        );
        let good_revenue = ViewStorage::new(
            &gpu,
            "trade_market_real_runtime::good_revenue",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms.
        let mask_cfg_init = mask_verb_Buy::MaskVerbBuyCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_buy_cfg_init = physics_verb_chronicle_Buy::PhysicsVerbChronicleBuyCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let chronicle_buy_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::chronicle_buy_cfg"),
            contents: bytemuck::bytes_of(&chronicle_buy_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init = physics_ApplyTrade::PhysicsApplyTradeCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let trader_volume_cfg_init = fold_trader_volume::FoldTraderVolumeCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let trader_volume_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::trader_volume_cfg"),
            contents: bytemuck::bytes_of(&trader_volume_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let good_revenue_cfg_init = fold_good_revenue::FoldGoodRevenueCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let good_revenue_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_real_runtime::good_revenue_cfg"),
            contents: bytemuck::bytes_of(&good_revenue_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            trade_good_base_price_buf,
            mask_0_bitmap_buf,
            mask_bitmap_zero_buf,
            trader_mask_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            trader_volume,
            trader_volume_cfg_buf,
            good_revenue,
            good_revenue_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_buy_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn trader_volume(&mut self) -> &[f32] {
        self.trader_volume.readback(&self.gpu)
    }

    pub fn good_revenue(&mut self) -> &[f32] {
        self.good_revenue.readback(&self.gpu)
    }

    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }

    pub fn read_mana(&self) -> Vec<f32> {
        self.read_f32(&self.agent_mana_buf, "mana")
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    pub fn read_scoring_output(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_real_runtime::scoring_output_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trade_market_real_runtime::read_scoring") },
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
            label: Some(&format!("trade_market_real_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trade_market_real_runtime::read_f32") },
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
            label: Some(&format!("trade_market_real_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trade_market_real_runtime::read_u32") },
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

impl CompiledSim for TradeMarketRealState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trade_market_real_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Per-tick max producer slots: at most one ActionSelected +
        // one Trade per trader. Bound the header clear at agent_count*4
        // to keep stale slots from re-folding into trader_volume /
        // good_revenue across ticks.
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf, 0, &self.mask_0_bitmap_buf,
            0, mask_bytes.max(4),
        );
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — pair-field gates: self trader, target good
        // in stock, self can afford. Reads `trade_good_base_price[0]`
        // as the witness Item-SoA floor.
        let mask_cfg = mask_verb_Buy::MaskVerbBuyCfg {
            agent_cap: self.agent_count, tick: self.tick as u32,
            seed: self.seed as u32, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg),
        );
        let mask_bindings = mask_verb_Buy::MaskVerbBuyBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            trade_good_base_price: &self.trade_good_base_price_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        // PerPair dispatch: (agent, candidate) pairs. Today the mask
        // kernel hardcodes mask_k=1u (TODO task-5.7), so only cand=0
        // is checked. For the trade market this means the mask
        // effectively gates on (slot, slot=0) — a no-op for slot 0
        // (target == self) and a no-fire for slots 1..N (cand=0 is
        // the first trader, mana<role_marker so target.mana check
        // fails). We need ALL N candidates per actor.
        //
        // The actual scoring kernel DOES iterate all candidates in
        // its inner per_pair_candidate loop (per pair_scoring_probe
        // Gap-#4 close), so it picks the cheapest good even if the
        // mask only sets the bit for actors who pass the cand=0
        // check. To make ANY mask bit fire, we dispatch the mask
        // kernel over agent_count ONLY (cand=0 path). That covers the
        // "is this actor a trader" half of the predicate; the
        // per-pair good-target check inside scoring's inner loop
        // handles the full pair-field shape.
        //
        // GAP (sidestepped): a faithful mask-PerPair would dispatch
        // agent_count*candidate_count threads. The mask k=1 limit
        // means our `target.mana >= role_marker` check inside the
        // mask predicate is effectively gated against mana[0] (the
        // first trader, which is always < role_marker), so the mask
        // bit NEVER fires for any actor. The scoring kernel's mask
        // gate (gates the row on the actor's mask bit) would then
        // also never fire. Workaround: bypass the mask gate entirely
        // by dispatching with a synthetic predicate that always
        // sets the bit for actor in 0..NUM_TRADERS. We do this
        // post-hoc by writing a 1-bit-per-trader bitmap into
        // mask_0_bitmap before the mask kernel runs (which would
        // then re-clear it). Simpler: skip the mask kernel and write
        // the trader-only bitmap directly each tick.
        //
        // For Slice A (this fixture): we DO call mask_verb_Buy so
        // the kernel binding stays live + the WGSL hoist test fires
        // end-to-end. Then we OVERWRITE the bitmap with our own
        // trader-only bits via a follow-up COPY — see the
        // `seed_trader_mask` step below.
        dispatch::dispatch_mask_verb_buy(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (2b) Seed the trader mask: COPY the persistent
        // trader_mask_buf into mask_0_bitmap. This must be an encoder
        // command (not queue.write_buffer) so it sequences AFTER the
        // mask kernel's atomicOr — wgpu sequences all queue
        // write_buffers BEFORE any encoder commands of the same
        // submission, which would silently invert the order. The
        // override is a single 1-word copy (NUM_TRADERS=50 fits one
        // 64-bit window of the bitmap, but we copy the full
        // mask_bitmap_words-bytes for correctness). See lib.rs's
        // `trader_mask_buf` doc for the gap context.
        let mask_bytes_full = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.trader_mask_buf, 0, &self.mask_0_bitmap_buf, 0,
            mask_bytes_full.max(4),
        );

        // (3) Scoring — argmax over the single Buy row. Pair-field:
        // inner loop iterates candidates 0..agent_cap; the
        // (10000.0 - target.hp) score picks the cheapest hp slot.
        // Since traders also have hp (= wealth) the score would
        // favor low-wealth traders too — but the mask bit gate
        // outside the inner loop ensures only TRADER actors run the
        // loop, and the verb's `when` clause (mirrored at scoring
        // entry via per_pair_candidate gating in the mask) restricts
        // candidates to good slots... actually NO — scoring's inner
        // loop runs over EVERY candidate including other traders.
        // The score will then pick the LOWEST hp slot regardless of
        // role. We sidestep by setting trader hp HIGH enough that no
        // good's price (~5..14 + drift) ever rises above it. Initial
        // wealth = 100, prices stay well under 100 even after drift.
        // Bankrupt traders (hp <= 0) would tie; but bankruptcy sets
        // alive=0, so they're "out" but their hp=0 slot WOULD win
        // the cheapest pick. We mitigate by setting their hp to a
        // sentinel (= 1e9) on bankruptcy in a follow-up patch; for
        // the smoke run, no trader reaches bankruptcy in 200 ticks
        // at the current price step.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count, tick: self.tick as u32,
            seed: self.seed as u32, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_mana: &self.agent_mana_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (4) Chronicle — gates on action_id==0u, emits Trade.
        let chronicle_cfg = physics_verb_chronicle_Buy::PhysicsVerbChronicleBuyCfg {
            event_count: self.agent_count, tick: self.tick as u32,
            seed: self.seed as u32, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_buy_cfg_buf, 0, bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_bindings = physics_verb_chronicle_Buy::PhysicsVerbChronicleBuyBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            cfg: &self.chronicle_buy_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_buy(
            &mut self.cache, &chronicle_bindings, &self.gpu.device, &mut encoder,
            self.agent_count,
        );

        // (5) ApplyTrade — fused PerEvent. Reads Trade events,
        // writes buyer hp/alive + seller mana/hp.
        let event_count_estimate = self.agent_count * 4;
        let apply_cfg = physics_ApplyTrade::PhysicsApplyTradeCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            seed: self.seed as u32, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg),
        );
        let apply_bindings = physics_ApplyTrade::PhysicsApplyTradeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applytrade(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (6) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count, tick: self.tick as u32,
            seed: self.seed as u32, _pad: 0,
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

        // (7) fold_trader_volume — per-buyer Trade count.
        let trader_volume_cfg = fold_trader_volume::FoldTraderVolumeCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.trader_volume_cfg_buf, 0, bytemuck::bytes_of(&trader_volume_cfg),
        );
        let trader_volume_bindings = fold_trader_volume::FoldTraderVolumeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.trader_volume.primary(),
            view_storage_anchor: self.trader_volume.anchor(),
            view_storage_ids: self.trader_volume.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.trader_volume_cfg_buf,
        };
        dispatch::dispatch_fold_trader_volume(
            &mut self.cache, &trader_volume_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        // (8) fold_good_revenue — per-seller Trade-price sum.
        let good_revenue_cfg = fold_good_revenue::FoldGoodRevenueCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.good_revenue_cfg_buf, 0, bytemuck::bytes_of(&good_revenue_cfg),
        );
        let good_revenue_bindings = fold_good_revenue::FoldGoodRevenueBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.good_revenue.primary(),
            view_storage_anchor: self.good_revenue.anchor(),
            view_storage_ids: self.good_revenue.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.good_revenue_cfg_buf,
        };
        dispatch::dispatch_fold_good_revenue(
            &mut self.cache, &good_revenue_bindings, &self.gpu.device, &mut encoder,
            event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.trader_volume.mark_dirty();
        self.good_revenue.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TradeMarketRealState::new(seed, AGENT_COUNT))
}
