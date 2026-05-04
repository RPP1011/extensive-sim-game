//! Per-fixture runtime for `assets/sim/auction_market.sim` — first
//! fixture to exercise the `auctions.*` namespace lowering. Each
//! Trader emits a single Bid event per tick AND calls
//! `auctions.place_bid(self, self, config.market.bid_amount)` (B1
//! stub: `return true`). The Allocated event is declared but no
//! rule fires it today (no `auctions.allocate(...)` body wired).
//!
//! Two views land here, both Agent-keyed:
//! - `bid_activity(trader: Agent) -> f32` — `@decay(rate=0.90)`
//!   anchored. Per-event fold matches `trader: who` — the binder
//!   means the slot indexed is the Bid's `trader` field, which is
//!   `self` for every emit. Per-slot steady state: 1/(1-0.90)=10.0.
//! - `good_bid_total(good: Agent) -> f32` — no decay,
//!   `@materialized(on_event = [Bid])`. Per-event fold accumulates
//!   `amount` keyed on the Bid's `good` field (also `self` per
//!   emit). After T=100 ticks each per-Trader slot accumulates
//!   `T * bid_amount = 100 * 10.0 = 1000.0`.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// 16-byte WGSL `vec3<f32>` interop. Same shape as the sibling
/// runtimes use; duplicated here to keep each fixture-runtime crate
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

/// Per-fixture state for the auction_market simulation.
pub struct AuctionState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    /// Per-Trader alive flag. The `where (self.alive)` guard on
    /// WanderAndBid reads this slot; traders with `alive == 0`
    /// neither integrate nor emit a Bid. Left at 1u everywhere
    /// here for the uniform-population observable.
    alive_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    /// Single shared event-ring (Bid is the only emitted event;
    /// Allocated is declared but unfired today).
    event_ring: EventRing,

    /// `bid_activity(trader: Agent) -> f32` storage.
    /// `@decay(rate=0.90)` → has_anchor=true. No top-K hint → ids
    /// stays absent.
    bid_activity: ViewStorage,
    bid_activity_cfg_buf: wgpu::Buffer,
    bid_activity_decay_cfg_buf: wgpu::Buffer,

    /// `good_bid_total(good: Agent) -> f32` storage. No `@decay`
    /// → has_anchor=false. The fold accumulates `amount` per Bid
    /// event keyed on `good` (= `self` per emit), so each
    /// per-Trader slot grows by `bid_amount` per tick.
    good_bid_total: ViewStorage,
    good_bid_total_cfg_buf: wgpu::Buffer,

    /// `faction_pressure(faction: Group) -> f32` storage. SLICE 2
    /// PROBE — first Group-keyed view in the auction context.
    /// `@decay(rate=0.95)` → has_anchor=true. The Bid event has no
    /// `faction` field, so the compiler-emitted fold defaults the
    /// key resolution to slot 2 (`trader`), making this view
    /// dispatch identically to an Agent-keyed view today: each
    /// per-Trader slot accumulates `bid_amount` per tick under
    /// decay=0.95, so per-slot steady ≈ 10.0 / (1 - 0.95) = 200.0.
    /// Once Group-population-aware sizing + a real `faction` field
    /// land, the storage shrinks to `faction_count` slots and the
    /// fold accumulates per-Faction. Wiring it through here proves
    /// the compile + dispatch path is healthy end-to-end in the
    /// auction context.
    faction_pressure: ViewStorage,
    faction_pressure_cfg_buf: wgpu::Buffer,
    faction_pressure_decay_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,
    pos_cache: Vec<Vec3>,
    dirty: bool,
    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Map a u32 hash output into a uniform `[-1, 1)` float — same
/// pattern the sibling runtimes use for keyed-PCG-derived initial
/// state.
fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl AuctionState {
    /// Construct an N-Trader simulation with deterministic initial
    /// positions + velocities derived from `seed` via the engine's
    /// keyed PCG (P5: `per_agent_u32(seed, agent_id, tick=0,
    /// purpose)`). All slots start alive (alive=1u everywhere).
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let spread = (agent_count as f32).cbrt().max(1.0);

        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id =
                AgentId::new(slot + 1).expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"auction_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"auction_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"auction_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"auction_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"auction_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"auction_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }
        let alive_init: Vec<u32> = vec![1u32; n];

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auction_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auction_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auction_runtime::alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cfg = physics_WanderAndBid::PhysicsWanderAndBidCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auction_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auction_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Shared event-ring + view storages ----
        let event_ring = EventRing::new(&gpu, "auction_runtime");

        // bid_activity — Agent-keyed, @decay(rate=0.90). One slot
        // per Trader.
        let bid_activity = ViewStorage::new(
            &gpu,
            "auction_runtime::bid_activity",
            agent_count,
            true,  // has_anchor (carries @decay)
            false, // no top-K storage hint
        );
        let ba_cfg = fold_bid_activity::FoldBidActivityCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let bid_activity_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("auction_runtime::bid_activity_cfg"),
                contents: bytemuck::bytes_of(&ba_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let ba_decay_cfg = decay_bid_activity::DecayBidActivityCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let bid_activity_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("auction_runtime::bid_activity_decay_cfg"),
                contents: bytemuck::bytes_of(&ba_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // good_bid_total — Agent-keyed, NO decay. One slot per
        // Trader. The fold matches `good: g` (binder) and
        // accumulates the Bid's `amount` field.
        let good_bid_total = ViewStorage::new(
            &gpu,
            "auction_runtime::good_bid_total",
            agent_count,
            false, // no @decay → has_anchor=false
            false, // no top-K storage hint
        );
        let gbt_cfg = fold_good_bid_total::FoldGoodBidTotalCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let good_bid_total_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("auction_runtime::good_bid_total_cfg"),
                contents: bytemuck::bytes_of(&gbt_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // SLICE 2 — faction_pressure (Group-keyed) view storage.
        // Sized at `agent_count` slots because the emit currently
        // resolves the key as the Bid's `trader` field (no
        // `faction` field on the event today, no
        // Group-population-aware sizing).
        let faction_pressure = ViewStorage::new(
            &gpu,
            "auction_runtime::faction_pressure",
            agent_count,
            true,  // has_anchor (carries @decay rate=0.95)
            false, // no top-K storage hint
        );
        let fp_cfg = fold_faction_pressure::FoldFactionPressureCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let faction_pressure_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("auction_runtime::faction_pressure_cfg"),
                contents: bytemuck::bytes_of(&fp_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let fp_decay_cfg = decay_faction_pressure::DecayFactionPressureCfg {
            agent_cap: agent_count,
            tick: 0,
            slot_count: agent_count,
            _pad: 0,
        };
        let faction_pressure_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("auction_runtime::faction_pressure_decay_cfg"),
                contents: bytemuck::bytes_of(&fp_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            alive_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            bid_activity,
            bid_activity_cfg_buf,
            bid_activity_decay_cfg_buf,
            good_bid_total,
            good_bid_total_cfg_buf,
            faction_pressure,
            faction_pressure_cfg_buf,
            faction_pressure_decay_cfg_buf,
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

    /// Per-Trader bid_activity accumulator. Each tick one Bid lands
    /// on each slot; @decay multiplies the running total by 0.90
    /// BEFORE the fold lands. Steady state per slot ≈
    /// `1 / (1 - 0.90)` = 10.0.
    pub fn bid_activity(&mut self) -> &[f32] {
        self.bid_activity.readback(&self.gpu)
    }

    /// Per-Trader good_bid_total accumulator. No decay; per Bid
    /// emit the slot indexed by `good` accumulates `amount`. With
    /// `good: self`, each Trader's slot grows by `bid_amount` per
    /// tick. After T ticks: `T * bid_amount`.
    pub fn good_bid_totals(&mut self) -> &[f32] {
        self.good_bid_total.readback(&self.gpu)
    }

    /// SLICE 2 — faction_pressure (Group-keyed) view readback.
    /// Today the Bid event has no `faction` field so the
    /// compiler-emitted fold defaults the key to slot 2 (= the
    /// `trader` field), making this view dispatch identically to
    /// an Agent-keyed view: each per-Trader slot accumulates
    /// `bid_amount` per tick under decay 0.95 → steady ≈ 200.0.
    pub fn faction_pressures(&mut self) -> &[f32] {
        self.faction_pressure.readback(&self.gpu)
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("auction_runtime::positions::copy"),
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

impl CompiledSim for AuctionState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("auction_runtime::step"),
            });

        // Per-tick clear of the shared event_tail. The single Bid
        // emitter atomicAdds against this counter; the count gets
        // read back to size the fold dispatch.
        self.event_ring.clear_tail_in(&mut encoder);

        // (1) WanderAndBid — emits 1 Bid event per alive Trader per
        // tick AND calls auctions.place_bid(self, self, amount)
        // (B1 stub `return true`).
        let bindings = physics_WanderAndBid::PhysicsWanderAndBidBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_alive: &self.alive_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_wanderandbid(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (2) seed_indirect_0 — keeps the indirect-args buffer warm
        // for the eventual `dispatch_workgroups_indirect` wire-up.
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // Per-tick event count = agent_count (every alive Trader
        // emits exactly one Bid per tick; all Traders stay alive).
        let event_count = self.agent_count;

        // (3a) decay_bid_activity — anchor multiply BEFORE the
        // fold. `view_storage_primary[slot] *= 0.90`.
        let ba_decay_cfg = decay_bid_activity::DecayBidActivityCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.bid_activity_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&ba_decay_cfg),
        );
        let ba_decay_bindings =
            decay_bid_activity::DecayBidActivityBindings {
                view_storage_primary: self.bid_activity.primary(),
                cfg: &self.bid_activity_decay_cfg_buf,
            };
        dispatch::dispatch_decay_bid_activity(
            &mut self.cache,
            &ba_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3b) fold_bid_activity — RMW
        // `view_storage_primary[event.trader] += 1.0` per Bid event.
        let ba_cfg = fold_bid_activity::FoldBidActivityCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.bid_activity_cfg_buf,
            0,
            bytemuck::bytes_of(&ba_cfg),
        );
        let ba_bindings = fold_bid_activity::FoldBidActivityBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.bid_activity.primary(),
            view_storage_anchor: self.bid_activity.anchor(),
            view_storage_ids: self.bid_activity.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.bid_activity_cfg_buf,
        };
        dispatch::dispatch_fold_bid_activity(
            &mut self.cache,
            &ba_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (4) fold_good_bid_total — no decay, just RMW
        // `view_storage_primary[event.good] += event.amount`.
        let gbt_cfg = fold_good_bid_total::FoldGoodBidTotalCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.good_bid_total_cfg_buf,
            0,
            bytemuck::bytes_of(&gbt_cfg),
        );
        let gbt_bindings = fold_good_bid_total::FoldGoodBidTotalBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.good_bid_total.primary(),
            view_storage_anchor: self.good_bid_total.anchor(),
            view_storage_ids: self.good_bid_total.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.good_bid_total_cfg_buf,
        };
        dispatch::dispatch_fold_good_bid_total(
            &mut self.cache,
            &gbt_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        // (5a) decay_faction_pressure — anchor multiply BEFORE the
        // (Group-keyed) fold. Today this multiplies the `agent_count`
        // slots that the fold actually writes (the `faction` key
        // resolves to the Bid's `trader` field — see SLICE 2 notes).
        let fp_decay_cfg = decay_faction_pressure::DecayFactionPressureCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            slot_count: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.faction_pressure_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&fp_decay_cfg),
        );
        let fp_decay_bindings = decay_faction_pressure::DecayFactionPressureBindings {
            view_storage_primary: self.faction_pressure.primary(),
            cfg: &self.faction_pressure_decay_cfg_buf,
        };
        dispatch::dispatch_decay_faction_pressure(
            &mut self.cache,
            &fp_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5b) fold_faction_pressure — RMW
        // `view_storage_primary[event.trader] += event.amount` per
        // Bid event today (the `faction` key resolution defaults to
        // `trader`).
        let fp_cfg = fold_faction_pressure::FoldFactionPressureCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.faction_pressure_cfg_buf,
            0,
            bytemuck::bytes_of(&fp_cfg),
        );
        let fp_bindings = fold_faction_pressure::FoldFactionPressureBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.faction_pressure.primary(),
            view_storage_anchor: self.faction_pressure.anchor(),
            view_storage_ids: self.faction_pressure.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.faction_pressure_cfg_buf,
        };
        dispatch::dispatch_fold_faction_pressure(
            &mut self.cache,
            &fp_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.bid_activity.mark_dirty();
        self.good_bid_total.mark_dirty();
        self.faction_pressure.mark_dirty();
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
    Box::new(AuctionState::new(seed, agent_count))
}
