//! Per-fixture runtime for `assets/sim/trade_market_probe.sim` —
//! v2 of the multi-resource market discovery probe. Closes gap #4
//! (verb cascade integration) and gap #5 (spatial body-form INSIDE
//! a multi-emit physics rule) from the v1 discovery doc.
//!
//! ## Per-tick chain
//!
//! Three chained subsystems share a single event ring:
//!
//! 1. **Spatial-grid build** (5 phases — count, scan_local, scan_carry,
//!    scan_add, scatter). Required input for the body-form spatial
//!    walk in `physics_WanderAndTrade`.
//!
//! 2. **Verb cascade** (mask -> scoring -> chronicle). The verb
//!    `ExecuteTrade` is the SOLE producer of `TradeExecuted` events
//!    (option (a) from the gap doc — verb-only emit, no two-producer
//!    fold ambiguity):
//!     - `mask_verb_ExecuteTrade` — sets bitmap bit when `self.alive`.
//!     - `scoring` — argmax picks `ExecuteTrade` (action_id=0) per
//!       agent, atomically appends one `ActionSelected{actor=agent,
//!       action_id=0, target=NO_TARGET}` event (kind tag = 5u).
//!     - `physics_verb_chronicle_ExecuteTrade` — reads each
//!       ActionSelected slot, gates on `action_id == 0u` (Trade),
//!       emits one `TradeExecuted{buyer:self, seller:self, amount:1.0,
//!       price:1.0, resource:0u}` (kind tag = 1u). Placeholder buyer
//!       == seller == self (the verb body has no spatial access).
//!
//! 3. **Multi-emit physics + spatial walk** (`physics_WanderAndTrade`).
//!    Per alive agent: integrate position, then walk the 27-cell
//!    spatial neighbourhood (slice-2b body-form, commit `134c5df8`)
//!    and per candidate emit BOTH a PriceObserved (kind tag = 2u, 4
//!    fields) AND a PriceGossip (kind tag = 3u, 5 fields). Two emits
//!    inside one for-body — the gap-#5 combination.
//!
//! ## Per-tick event count
//!
//! Per alive agent the events are:
//!   - 1 ActionSelected (scoring)
//!   - 1 TradeExecuted (chronicle, gated on ActionSelected.action_id)
//!   - K_per_agent PriceObserved (one per spatial-neighbour candidate)
//!   - K_per_agent PriceGossip   (one per spatial-neighbour candidate)
//!
//! K_per_agent = number of candidate slots in the agent's 27-cell
//! neighbourhood. With auto-spread + 32 agents the density is sparse
//! (~1/cell where occupied), so K_per_agent is typically a small
//! constant per agent. Total per-tick ≈ N * (2 + 2 * K). The fold
//! event_count uses a generous upper bound (`N * MAX_PER_CELL * 27 *
//! 2 + N * 2`) capped at the ring's 65536-slot capacity.
//!
//! ## Three-view shape (unchanged storage layout, new dynamics)
//!
//!   - `price_belief` — pair_map u32 view, atomicOr-folded. NOW
//!     populated from BOTH `PriceObserved` (kind 2) and `PriceGossip`
//!     (kind 3) by per-handler tag filter inside fold_price_belief.
//!     Off-diagonal cells are NO LONGER zero — every spatial-neighbour
//!     pair flips its bit to 1u.
//!   - `trader_volume` — f32 with @decay, fed by TradeExecuted
//!     (kind 1) from the verb chronicle. Per-slot dynamics
//!     unchanged from v1 (steady ≈ 20.0 with decay=0.95).
//!   - `hub_volume` — f32 no-decay, fed by TradeExecuted (kind 1)
//!     from the verb chronicle. Per-slot grows by 1.0 per tick =
//!     100.0 after 100 ticks.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Padded {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z, _pad: 0.0 }
    }
}

impl From<Vec3Padded> for Vec3 {
    fn from(p: Vec3Padded) -> Self {
        Vec3::new(p.x, p.y, p.z)
    }
}

/// Per-fixture state for the v2 trade_market probe.
pub struct TradeMarketState {
    gpu: GpuContext,

    // -- Agent SoA --
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-agent alive flag. All slots start alive so every tick fires
    /// the verb cascade + the multi-emit physics.
    alive_buf: wgpu::Buffer,
    physics_cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    // -- Shared event ring (TradeExecuted + PriceObserved +
    //    PriceGossip + ActionSelected) --
    event_ring: EventRing,

    // -- Verb cascade state --
    /// Mask bitmap (one bit per agent in u32 words). Cleared per tick.
    mask_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,
    /// Scoring output (4 × u32 per agent). Mostly inert here — the
    /// scoring kernel populates it but the chronicle reads from the
    /// event ring (ActionSelected) instead.
    scoring_output_buf: wgpu::Buffer,
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_cfg_buf: wgpu::Buffer,

    // -- Spatial grid (slice 2b body-form spatial query) --
    spatial_grid_cells: wgpu::Buffer,
    spatial_grid_offsets: wgpu::Buffer,
    spatial_grid_starts: wgpu::Buffer,
    spatial_chunk_sums: wgpu::Buffer,
    spatial_offsets_zero: wgpu::Buffer,

    // -- price_belief (pair_map u32, atomicOr) --
    price_belief_primary: wgpu::Buffer,
    price_belief_staging: wgpu::Buffer,
    price_belief_cache: Vec<u32>,
    price_belief_dirty: bool,
    price_belief_cfg_buf: wgpu::Buffer,

    // -- trader_volume (f32 with @decay) --
    trader_volume: ViewStorage,
    trader_volume_cfg_buf: wgpu::Buffer,
    trader_volume_decay_cfg_buf: wgpu::Buffer,

    // -- hub_volume (f32 no decay) --
    hub_volume: ViewStorage,
    hub_volume_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,
    pos_cache: Vec<Vec3>,
    dirty: bool,
    tick: u64,
    agent_count: u32,
    seed: u64,
}

fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl TradeMarketState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let spread = (agent_count as f32).cbrt().max(1.0);

        // Agent init — keyed PCG per slot.
        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id =
                AgentId::new(slot + 1).expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"trade_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"trade_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"trade_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"trade_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"trade_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"trade_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }
        let alive_init: Vec<u32> = vec![1u32; n];

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trade_market_runtime::alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let physics_cfg = physics_WanderAndTrade::PhysicsWanderAndTradeCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Event ring — the chronicle WGSL gates on the event-kind tag
        // (`if (atomicLoad(&event_ring[event_idx * 10u + 0u]) == 5u)`,
        // commit b90a43a4 — the per-handler tag filter inside
        // PerEventEmit kernels). Empty slots have tag 0u so they
        // skip the chronicle's body, eliminating the need for
        // the older sentinel-pre-stamp pattern.
        let event_ring = EventRing::new(&gpu, "trade_market_runtime");

        // ---- Verb cascade buffers ----
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::mask_0_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let scoring_output_bytes = (agent_count as u64) * 4 * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask_cfg_init = mask_verb_ExecuteTrade::MaskVerbExecuteTradeCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::mask_cfg"),
                contents: bytemuck::bytes_of(&mask_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let chronicle_cfg_init =
            physics_verb_chronicle_ExecuteTrade::PhysicsVerbChronicleExecuteTradeCfg {
                event_count: 0,
                tick: 0,
                seed: 0,
                _pad0: 0,
            };
        let chronicle_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::chronicle_cfg"),
                contents: bytemuck::bytes_of(&chronicle_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // ---- Spatial grid buffers (mirrors particle_collision_runtime)
        use dsl_compiler::cg::emit::spatial as sp;
        let agent_cap_bytes = (agent_count as u64) * 4;
        let offsets_size = sp::offsets_bytes();
        let starts_size = ((sp::num_cells() as u64) + 1) * 4;
        let spatial_grid_cells = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::spatial_grid_cells"),
            size: agent_cap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::spatial_grid_offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_grid_starts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::spatial_grid_starts"),
            size: starts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = dsl_compiler::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
        let num_chunks = sp::num_cells().div_ceil(chunk_size);
        let chunk_sums_size = (num_chunks as u64) * 4;
        let spatial_chunk_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::spatial_chunk_sums"),
            size: chunk_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let zeros: Vec<u8> = vec![0u8; offsets_size as usize];
        let spatial_offsets_zero =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::spatial_offsets_zero"),
                contents: &zeros,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // ---- price_belief: pair_map u32 storage allocated locally ----
        let belief_slot_count = (agent_count as u64) * (agent_count as u64);
        let belief_bytes = (belief_slot_count * 4).max(16);
        let price_belief_primary = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::price_belief_primary"),
            size: belief_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let price_belief_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trade_market_runtime::price_belief_staging"),
            size: belief_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pb_cfg_init = fold_price_belief::FoldPriceBeliefCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: agent_count,
            _pad: 0,
        };
        let price_belief_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::price_belief_cfg"),
                contents: bytemuck::bytes_of(&pb_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // ---- trader_volume: ViewStorage (f32 with @decay) ----
        let trader_volume = ViewStorage::new(
            &gpu,
            "trade_market_runtime::trader_volume",
            agent_count,
            true,
            false,
        );
        let tv_cfg_init = fold_trader_volume::FoldTraderVolumeCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let trader_volume_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::trader_volume_cfg"),
                contents: bytemuck::bytes_of(&tv_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let tv_decay_init = decay_trader_volume::DecayTraderVolumeCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let trader_volume_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::trader_volume_decay_cfg"),
                contents: bytemuck::bytes_of(&tv_decay_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // ---- hub_volume: ViewStorage (f32, no decay) ----
        let hub_volume = ViewStorage::new(
            &gpu,
            "trade_market_runtime::hub_volume",
            agent_count,
            false,
            false,
        );
        let hv_cfg_init = fold_hub_volume::FoldHubVolumeCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let hub_volume_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("trade_market_runtime::hub_volume_cfg"),
                contents: bytemuck::bytes_of(&hv_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            alive_buf,
            physics_cfg_buf,
            pos_staging,
            event_ring,
            mask_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_cfg_buf,
            spatial_grid_cells,
            spatial_grid_offsets,
            spatial_grid_starts,
            spatial_chunk_sums,
            spatial_offsets_zero,
            price_belief_primary,
            price_belief_staging,
            price_belief_cache: vec![0u32; belief_slot_count as usize],
            price_belief_dirty: false,
            price_belief_cfg_buf,
            trader_volume,
            trader_volume_cfg_buf,
            trader_volume_decay_cfg_buf,
            hub_volume,
            hub_volume_cfg_buf,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    /// Per-Trader trade volume accumulator (verb chronicle TradeExecuted
    /// with placeholder buyer=self). With @decay(0.95) and trade_amount
    /// = 1.0, per-slot steady ≈ 20.0 (same as v1).
    pub fn trader_volumes(&mut self) -> &[f32] {
        self.trader_volume.readback(&self.gpu)
    }

    /// Per-Trader hub volume accumulator (verb chronicle TradeExecuted
    /// with placeholder seller=self). After T ticks per slot = T *
    /// trade_amount = 100.0 (same as v1).
    pub fn hub_volumes(&mut self) -> &[f32] {
        self.hub_volume.readback(&self.gpu)
    }

    /// Per-(observer, hub) belief bitset, flattened row-major.
    /// Length = `agent_count × agent_count`. v2 dynamic: every
    /// spatial-neighbour pair flips its bit; off-diagonal cells are
    /// no longer all-zero.
    pub fn price_belief(&mut self) -> &[u32] {
        if self.price_belief_dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("trade_market_runtime::price_belief::readback"),
                },
            );
            let bytes = (self.price_belief_cache.len() as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &self.price_belief_primary,
                0,
                &self.price_belief_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.price_belief_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.price_belief_cache.copy_from_slice(raw);
            drop(mapped);
            self.price_belief_staging.unmap();
            self.price_belief_dirty = false;
        }
        &self.price_belief_cache
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("trade_market_runtime::positions::copy"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.pos_buf,
                0,
                &self.pos_staging,
                0,
                (self.agent_count as u64) * std::mem::size_of::<Vec3Padded>() as u64,
            );
            self.gpu.queue.submit(Some(encoder.finish()));

            let slice = self.pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");

            let bytes = slice.get_mapped_range();
            let padded: &[Vec3Padded] = bytemuck::cast_slice(&bytes);
            for (cache, p) in self.pos_cache.iter_mut().zip(padded.iter()) {
                *cache = (*p).into();
            }
            drop(bytes);
            self.pos_staging.unmap();
            self.dirty = false;
        }
        &self.pos_cache
    }
}

impl CompiledSim for TradeMarketState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trade_market_runtime::step"),
            });

        // (0) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_bitmap_buf,
            0,
            mask_bytes.max(4),
        );
        let offsets_size = dsl_compiler::cg::emit::spatial::offsets_bytes();
        encoder.copy_buffer_to_buffer(
            &self.spatial_offsets_zero,
            0,
            &self.spatial_grid_offsets,
            0,
            offsets_size,
        );

        // Common cfg upload — the spatial-build kernels and the
        // physics/chronicle kernels share `physics_cfg_buf` for
        // `agent_cap` + `tick`.
        let physics_cfg = physics_WanderAndTrade::PhysicsWanderAndTradeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&physics_cfg),
        );

        // (1) Spatial-hash counting sort (5 phases). Same shape as
        // particle_collision_runtime — sort agents into 27-cell grid
        // for the body-form spatial walk in WanderAndTrade.
        let count_b = spatial_build_hash_count::SpatialBuildHashCountBindings {
            agent_pos: &self.pos_buf,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_count(
            &mut self.cache,
            &count_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_local_b =
            spatial_build_hash_scan_local::SpatialBuildHashScanLocalBindings {
                spatial_grid_offsets: &self.spatial_grid_offsets,
                spatial_grid_starts: &self.spatial_grid_starts,
                spatial_chunk_sums: &self.spatial_chunk_sums,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_spatial_build_hash_scan_local(
            &mut self.cache,
            &scan_local_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_carry_b =
            spatial_build_hash_scan_carry::SpatialBuildHashScanCarryBindings {
                spatial_chunk_sums: &self.spatial_chunk_sums,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_spatial_build_hash_scan_carry(
            &mut self.cache,
            &scan_carry_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scan_add_b = spatial_build_hash_scan_add::SpatialBuildHashScanAddBindings {
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            spatial_chunk_sums: &self.spatial_chunk_sums,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scan_add(
            &mut self.cache,
            &scan_add_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );
        let scatter_b = spatial_build_hash_scatter::SpatialBuildHashScatterBindings {
            agent_pos: &self.pos_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_spatial_build_hash_scatter(
            &mut self.cache,
            &scatter_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) Verb cascade — Round 1: mask -> scoring.
        let mask_cfg = mask_verb_ExecuteTrade::MaskVerbExecuteTradeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_b = mask_verb_ExecuteTrade::MaskVerbExecuteTradeBindings {
            agent_alive: &self.alive_buf,
            mask_0_bitmap: &self.mask_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_mask_verb_executetrade(
            &mut self.cache,
            &mask_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_b = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            mask_0_bitmap: &self.mask_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) Verb cascade — Round 2: chronicle. Reads ActionSelected
        // events, gates on action_id == 0u, emits TradeExecuted.
        // Sized to `agent_count` since scoring emits exactly one
        // ActionSelected per alive slot (and the ring is sentinel-
        // pre-stamped so workgroup-rounding threads early-exit).
        let chronicle_cfg =
            physics_verb_chronicle_ExecuteTrade::PhysicsVerbChronicleExecuteTradeCfg {
                event_count: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                _pad0: 0,
            };
        self.gpu.queue.write_buffer(
            &self.chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_b =
            physics_verb_chronicle_ExecuteTrade::PhysicsVerbChronicleExecuteTradeBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_cfg_buf,
            };
        dispatch::dispatch_physics_verb_chronicle_executetrade(
            &mut self.cache,
            &chronicle_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) WanderAndTrade — multi-emit physics + spatial body-form
        // walk. Per alive agent: integrate position, then walk the
        // 27-cell neighbourhood; per candidate emit one PriceObserved
        // (kind=2u) AND one PriceGossip (kind=3u).
        let physics_b = physics_WanderAndTrade::PhysicsWanderAndTradeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_alive: &self.alive_buf,
            agent_vel: &self.vel_buf,
            spatial_grid_cells: &self.spatial_grid_cells,
            spatial_grid_offsets: &self.spatial_grid_offsets,
            spatial_grid_starts: &self.spatial_grid_starts,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_physics_wanderandtrade(
            &mut self.cache,
            &physics_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_b = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6) Folds — sized generously to cover all event kinds:
        //   - N ActionSelected (kind 5u, scoring)
        //   - N TradeExecuted  (kind 1u, chronicle)
        //   - up to N * MAX_PER_CELL * 27 PriceObserved (kind 2u,
        //     physics body-form spatial walk per candidate)
        //   - up to N * MAX_PER_CELL * 27 PriceGossip (kind 3u, same)
        //
        // Cap at the ring's 65536 slots (DEFAULT_EVENT_RING_CAP_SLOTS);
        // the kernel `if (event_idx >= cfg.event_count) return;`
        // gate skips empty slots, and the per-handler tag filter
        // partitions events into the correct fold body.
        use dsl_compiler::cg::emit::spatial as sp;
        let upper_per_kind =
            self.agent_count.saturating_mul(sp::MAX_PER_CELL).saturating_mul(27);
        let event_count = std::cmp::min(
            self.agent_count
                .saturating_mul(2) // ActionSelected + TradeExecuted
                .saturating_add(upper_per_kind.saturating_mul(2)), // 2 emits per pair
            65536,
        );

        // (6a) fold_price_belief — atomicOr u32 fold filtered on tags
        // 2u (PriceObserved) and 3u (PriceGossip). second_key_pop =
        // agent_count so the `[obs * N + hub]` index lands at the
        // pair_map flat offset.
        let pb_cfg = fold_price_belief::FoldPriceBeliefCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.price_belief_cfg_buf,
            0,
            bytemuck::bytes_of(&pb_cfg),
        );
        let pb_b = fold_price_belief::FoldPriceBeliefBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.price_belief_primary,
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.price_belief_cfg_buf,
        };
        dispatch::dispatch_fold_price_belief(
            &mut self.cache,
            &pb_b,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (6b) decay_trader_volume — anchor multiply pre-pass.
        let tv_decay_cfg = decay_trader_volume::DecayTraderVolumeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.trader_volume_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&tv_decay_cfg),
        );
        let tv_decay_b = decay_trader_volume::DecayTraderVolumeBindings {
            view_storage_primary: self.trader_volume.primary(),
            cfg: &self.trader_volume_decay_cfg_buf,
        };
        dispatch::dispatch_decay_trader_volume(
            &mut self.cache,
            &tv_decay_b,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6c) fold_trader_volume — RMW per TradeExecuted (kind=1u).
        let tv_cfg = fold_trader_volume::FoldTraderVolumeCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.trader_volume_cfg_buf,
            0,
            bytemuck::bytes_of(&tv_cfg),
        );
        let tv_b = fold_trader_volume::FoldTraderVolumeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.trader_volume.primary(),
            view_storage_anchor: self.trader_volume.anchor(),
            view_storage_ids: self.trader_volume.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.trader_volume_cfg_buf,
        };
        dispatch::dispatch_fold_trader_volume(
            &mut self.cache,
            &tv_b,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (6d) fold_hub_volume — RMW per TradeExecuted (kind=1u).
        let hv_cfg = fold_hub_volume::FoldHubVolumeCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.hub_volume_cfg_buf,
            0,
            bytemuck::bytes_of(&hv_cfg),
        );
        let hv_b = fold_hub_volume::FoldHubVolumeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.hub_volume.primary(),
            view_storage_anchor: self.hub_volume.anchor(),
            view_storage_ids: self.hub_volume.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.hub_volume_cfg_buf,
        };
        dispatch::dispatch_fold_hub_volume(
            &mut self.cache,
            &hv_b,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.price_belief_dirty = true;
        self.trader_volume.mark_dirty();
        self.hub_volume.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        self.read_positions()
    }
}

/// Build a boxed `CompiledSim`.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TradeMarketState::new(seed, agent_count))
}
