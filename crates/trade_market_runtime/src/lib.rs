//! Per-fixture runtime for `assets/sim/trade_market_probe.sim` —
//! multi-resource market discovery probe combining ALL the recently-
//! landed compiler surfaces:
//!
//!   - 4 distinct Item entity declarations (Wood/Iron/Grain/Cloth) +
//!     1 Group entity (Guild) — declaration-only today; the active
//!     rule body doesn't read the per-Item base_price fields, so the
//!     compiler doesn't emit external bindings for them. The catalog
//!     populates correctly (verified by lower diagnostics + the
//!     compile-gate test); a follow-up probe with `items.base_price(
//!     <id>)` reads in the rule body would surface 4 distinct
//!     `wood_base_price` / `iron_base_price` / `grain_base_price` /
//!     `cloth_base_price` external bindings via
//!     `cg/emit/kernel.rs::item_field_external_name` (commit
//!     524ae43f).
//!
//!   - **Multi-event-kind producer**: physics_WanderAndTrade emits
//!     ONE TradeExecuted (kind=1u, 7 ring slots used: kind+tick+
//!     buyer+seller+resource+amount+price) AND ONE PriceObserved
//!     (kind=2u, 6 ring slots used: kind+tick+observer+hub+resource+
//!     price_q8) per alive agent per tick. Both events share the
//!     same event ring (no per-kind ring partitioning yet); the
//!     consumer-side per-handler tag filter (commit cb24fd69)
//!     guards each fold body on the kind tag at offset 0:
//!       - `fold_trader_volume`  — `if (... + 0u] == 1u)` (tag=1)
//!       - `fold_hub_volume`     — `if (... + 0u] == 1u)` (tag=1)
//!       - `fold_price_belief`   — `if (... + 0u] == 2u)` (tag=2)
//!     Per-tick event count = 2 × agent_count (each agent emits 2).
//!
//!   - **Three views with mixed storage**:
//!       - `price_belief` — `view ... -> u32`, `pair_map` storage.
//!         Atomically OR'd via `atomicOr` in the compiler-emitted
//!         fold (post-`51b5853b`, the same shape as tom_probe).
//!         Allocated locally as an `agent_cap × agent_cap × u32`
//!         buffer (NOT through `engine::gpu::ViewStorage`, whose
//!         host-side cache is `Vec<f32>`).
//!       - `trader_volume` — `view ... -> f32` with `@decay(0.95)`.
//!         The ViewStorage helper handles the f32 case + the
//!         per-decay anchor binding fallback.
//!       - `hub_volume` — `view ... -> f32`, no decay. Same f32
//!         path, has_anchor=false.
//!
//! ## Expected (FULL FIRE) observable
//!
//! After T=100 ticks at agent_count = 32:
//!   - `trader_volume[i]` per slot ≈ `trade_amount / (1 - 0.95)` =
//!     `1.0 / 0.05` = 20.0 (geometric series steady state; with
//!     `0.95^100 ≈ 5.9e-3`, the per-slot value is within 0.6% of
//!     the analytical limit).
//!   - `hub_volume[i]` per slot = `T * trade_amount` = 100.0 (no
//!     decay, monotonic accumulator).
//!   - `price_belief[i*N + i]` == 1u (diagonal — observation_bit
//!     OR'd into the (self, self) cell every tick; idempotent on
//!     the second-and-subsequent ticks).
//!   - `price_belief[i*N + j]` == 0u for every i != j (off-diagonal
//!     stays at 0u because no event with observer != hub is ever
//!     emitted under the placeholder routing).

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// 16-byte WGSL `vec3<f32>` interop. Same shape as the sibling
/// runtimes use; duplicated here so each fixture-runtime crate stays
/// self-contained.
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

/// Per-fixture state for the trade_market probe.
pub struct TradeMarketState {
    gpu: GpuContext,

    // -- Agent SoA (read by physics_WanderAndTrade) --
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-agent alive flag; the `where (self.alive)` guard reads
    /// this. All slots start alive so every tick fires both emits.
    alive_buf: wgpu::Buffer,
    physics_cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    // -- Shared event ring (TradeExecuted + PriceObserved) --
    event_ring: EventRing,

    // -- price_belief storage (pair_map u32, allocated locally) --
    /// `agent_cap × agent_cap × u32` buffer. Indexed
    /// `[observer * agent_cap + subject]` per the compiler-emitted
    /// fold body (`local_0 * cfg.second_key_pop + local_1`).
    price_belief_primary: wgpu::Buffer,
    price_belief_staging: wgpu::Buffer,
    price_belief_cache: Vec<u32>,
    price_belief_dirty: bool,
    price_belief_cfg_buf: wgpu::Buffer,

    // -- trader_volume storage (f32, @decay) --
    trader_volume: ViewStorage,
    trader_volume_cfg_buf: wgpu::Buffer,
    trader_volume_decay_cfg_buf: wgpu::Buffer,

    // -- hub_volume storage (f32, no decay) --
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
            _pad: [0; 2],
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

        let event_ring = EventRing::new(&gpu, "trade_market_runtime");

        // ---- price_belief: pair_map u32 storage allocated locally ----
        // Tom_probe shape: NxN u32 buffer + staging. atomicOr fold
        // body indexes `[observer * agent_cap + subject]`. We expose
        // STORAGE | COPY_SRC | COPY_DST so the readback round-trip
        // through the staging buffer works.
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
            true,  // has_anchor (carries @decay)
            false, // no top-K storage hint
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
            false, // no @decay
            false, // no top-K storage hint
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

    /// Per-Trader trade volume accumulator. With @decay(0.95) and
    /// trade_amount=1.0, per-slot steady state ≈ 20.0.
    pub fn trader_volumes(&mut self) -> &[f32] {
        self.trader_volume.readback(&self.gpu)
    }

    /// Per-Trader hub volume accumulator (no decay). Per-slot grows
    /// by trade_amount=1.0 every tick under placeholder routing
    /// (seller=self) — after T ticks per slot = T.
    pub fn hub_volumes(&mut self) -> &[f32] {
        self.hub_volume.readback(&self.gpu)
    }

    /// Per-(observer, hub) belief bitset, flattened row-major.
    /// Length = `agent_count × agent_count`. Diagonal slots
    /// `[i * N + i] == observation_bit (1u)`; off-diagonal stay
    /// at 0u under the placeholder routing.
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

        // (1) Per-tick clear of event_tail.
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) WanderAndTrade physics — 2 emits per alive agent per
        // tick (TradeExecuted kind=1u + PriceObserved kind=2u).
        let physics_cfg = physics_WanderAndTrade::PhysicsWanderAndTradeCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&physics_cfg),
        );
        let physics_bindings =
            physics_WanderAndTrade::PhysicsWanderAndTradeBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_pos: &self.pos_buf,
                agent_alive: &self.alive_buf,
                agent_vel: &self.vel_buf,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_physics_wanderandtrade(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // Per-tick event count = 2 × agent_count (each alive agent
        // emits exactly one TradeExecuted + one PriceObserved). The
        // per-handler tag filter inside each fold partitions the
        // ring by event kind, so the dispatch sizing is the EXACT
        // upper bound (no per-kind subdivision needed at the
        // dispatch level).
        let event_count = self.agent_count * 2;

        // (4) fold_price_belief — atomicOr u32 fold filtered on tag
        // 2u (PriceObserved). Indexes pair_map storage as
        // `[observer * agent_cap + subject]`; second_key_pop must
        // equal agent_count for the diagonal to land at i*N+i.
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
        let pb_bindings = fold_price_belief::FoldPriceBeliefBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.price_belief_primary,
            // No @decay, no top-K → anchor / ids fall back to primary
            // per the kernel's `unwrap_or(primary)` shape.
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.price_belief_cfg_buf,
        };
        dispatch::dispatch_fold_price_belief(
            &mut self.cache,
            &pb_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (5a) decay_trader_volume — anchor multiply BEFORE the fold.
        // `view_storage_primary[slot] *= 0.95`.
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
        let tv_decay_bindings = decay_trader_volume::DecayTraderVolumeBindings {
            view_storage_primary: self.trader_volume.primary(),
            cfg: &self.trader_volume_decay_cfg_buf,
        };
        dispatch::dispatch_decay_trader_volume(
            &mut self.cache,
            &tv_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5b) fold_trader_volume — RMW
        // `view_storage_primary[event.buyer] += event.amount` per
        // TradeExecuted (filtered on tag 1u).
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
        let tv_bindings = fold_trader_volume::FoldTraderVolumeBindings {
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
            &tv_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (6) fold_hub_volume — RMW
        // `view_storage_primary[event.seller] += event.amount` per
        // TradeExecuted (filtered on tag 1u). No decay → no anchor
        // multiply pre-pass.
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
        let hv_bindings = fold_hub_volume::FoldHubVolumeBindings {
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
            &hv_bindings,
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

/// Build a boxed `CompiledSim` so the `sim_app` runner can switch
/// between fixture runtimes via a one-line constructor swap.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TradeMarketState::new(seed, agent_count))
}
