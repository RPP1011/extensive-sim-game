//! Per-fixture runtime for `assets/sim/bartering.sim` — entity-root
//! coverage probe for the Item + Group entity-root surfaces (the
//! pre-existing fixtures all declare only `entity ... : Agent`).
//!
//! The `bartering.sim` fixture declares one entity of each root
//! kind plus a `Trade` event whose payload carries an `ItemId`. The
//! IdleDrift physics rule emits one Trade event per alive agent per
//! tick, routing the receiver via the `engaged_with` SoA slot. The
//! `trade_count` view folds the receiver field into a per-slot
//! counter. After T ticks with N alive agents that's N*T total
//! Trades; with the deterministic "every slot points at slot 0"
//! engaged_with topology used here, slot 0 receives all of them.
//!
//! GAP carry-through: the runtime allocates SoA storage only for
//! Agent entities; the `Coin : Item` and `Caravan : Group`
//! declarations in the .sim file have no corresponding buffers
//! here because the compiler doesn't emit any (see the GAP note in
//! `bartering.sim` for the full description). The runtime mirrors
//! the swarm_storm shape because that's the smallest existing
//! shape that has both an event-ring producer and a fold consumer
//! — the bartering observable falls out of those primitives plus
//! the engaged_with target buffer that target_chaser introduced.

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

/// Coin entity SoA: a single Item slot whose `weight: f32` field the
/// IdleDrift physics rule multiplies the per-tick velocity drift by.
/// Sized to one record so `items.weight(0)` (the only access pattern
/// in the .sim file) lands on a populated slot. The chosen value
/// (`COIN_WEIGHT_VALUE`) is the multiplier the bartering_app
/// observable assertion keys on.
pub const COIN_WEIGHT_VALUE: f32 = 2.0;

/// Caravan entity SoA: a single Group slot whose `size: f32` field
/// the IdleDrift physics rule also multiplies the per-tick velocity
/// drift by. Sized to one record so `groups.size(0)` lands on a
/// populated slot. The chosen value (`CARAVAN_SIZE_VALUE`) combines
/// with `COIN_WEIGHT_VALUE` in the bartering_app drift assertion
/// (expected drift ≈ baseline × COIN_WEIGHT_VALUE × CARAVAN_SIZE_VALUE).
pub const CARAVAN_SIZE_VALUE: f32 = 1.5;

pub struct BarteringState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    alive_buf: wgpu::Buffer,
    /// Per-agent target (receiver of trades). `array<u32>` of
    /// length `agent_cap`. Initialized so every slot points at
    /// slot 0 — the resulting fold per-slot signature is "slot 0
    /// receives N*T trades, every other slot receives 0", which
    /// makes the observable trivially predictable.
    engaged_with_buf: wgpu::Buffer,
    /// Per-Item SoA — `coin_weight: array<f32>`. The .sim file's
    /// IdleDrift rule reads `items.weight(0)`, so this buffer carries
    /// one record (`COIN_WEIGHT_VALUE`). The compiler-generated
    /// `PhysicsIdleDriftBindings` carries a `coin_weight` slot whose
    /// name comes from the program's `entity_field_catalog`
    /// (entity_name "Coin" + field_name "weight" → snake_case
    /// `coin_weight`).
    coin_weight_buf: wgpu::Buffer,
    /// Per-Group SoA — `caravan_size: array<f32>`. Symmetric mirror
    /// of `coin_weight_buf` for the Group entity-root surface. The
    /// .sim file's IdleDrift rule reads `groups.size(0)`, so this
    /// buffer carries one record (`CARAVAN_SIZE_VALUE`). The
    /// compiler-generated `PhysicsIdleDriftBindings` carries a
    /// `caravan_size` slot whose name comes from the program's
    /// `entity_field_catalog` (entity_name "Caravan" + field_name
    /// "size" → snake_case `caravan_size`).
    caravan_size_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    event_ring: EventRing,

    /// Per-receiver Trade counter. No @decay → has_anchor=false.
    trade_count: ViewStorage,
    trade_count_cfg_buf: wgpu::Buffer,

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

impl BarteringState {
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
                normalise(per_agent_u32(seed, agent_id, 0, b"bart_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"bart_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"bart_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"bart_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"bart_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"bart_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }
        let alive_init: Vec<u32> = vec![1u32; n];
        // Hub-and-spokes: every slot routes its Trade to slot 0.
        // The fold's per-slot signature is "slot 0 = N*T, all
        // others = 0", which makes the observable trivially
        // predictable for the bartering_app assertion.
        let engaged_with_init: Vec<u32> = vec![0u32; n];

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bartering_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bartering_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bartering_runtime::alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let engaged_with_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bartering_runtime::engaged_with"),
                contents: bytemuck::cast_slice(&engaged_with_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Per-Item SoA — the WGSL `coin_weight: array<f32>` binding
        // produced by lowering the `entity Coin : Item { weight: f32 }`
        // declaration + the `items.weight(0)` access in the IdleDrift
        // physics rule. Sized to a single Item slot.
        let coin_weight_init: Vec<f32> = vec![COIN_WEIGHT_VALUE];
        let coin_weight_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("bartering_runtime::coin_weight"),
                contents: bytemuck::cast_slice(&coin_weight_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        );
        // Per-Group SoA — the WGSL `caravan_size: array<f32>` binding
        // produced by lowering the `entity Caravan : Group { size: f32 }`
        // declaration + the `groups.size(0)` access in the IdleDrift
        // physics rule. Sized to a single Group slot. This buffer is
        // the runtime-side proof that the Group READ path is wired
        // symmetrically to the Item READ path (compiler symmetric
        // wiring claim from commit 524ae43f).
        let caravan_size_init: Vec<f32> = vec![CARAVAN_SIZE_VALUE];
        let caravan_size_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("bartering_runtime::caravan_size"),
                contents: bytemuck::cast_slice(&caravan_size_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        );
        let cfg = physics_IdleDrift::PhysicsIdleDriftCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0, _pad: 0,
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bartering_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bartering_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let event_ring = EventRing::new(&gpu, "bartering_runtime");
        let trade_count = ViewStorage::new(
            &gpu,
            "bartering_runtime::trade_count",
            agent_count,
            false,
            false,
        );
        let tc_cfg = fold_trade_count::FoldTradeCountCfg {
            event_count: 0,
            tick: 0,
            second_key_pop: 1,
            _pad: 0,
        };
        let trade_count_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("bartering_runtime::trade_count_cfg"),
                contents: bytemuck::bytes_of(&tc_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            alive_buf,
            engaged_with_buf,
            coin_weight_buf,
            caravan_size_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            trade_count,
            trade_count_cfg_buf,
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

    pub fn trade_counts(&mut self) -> &[f32] {
        self.trade_count.readback(&self.gpu)
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("bartering_runtime::positions::copy"),
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

impl CompiledSim for BarteringState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bartering_runtime::step"),
            });

        self.event_ring.clear_tail_in(&mut encoder);

        // (1) IdleDrift — emits 1 Trade event per alive agent.
        let bindings = physics_IdleDrift::PhysicsIdleDriftBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_alive: &self.alive_buf,
            agent_engaged_with: &self.engaged_with_buf,
            agent_vel: &self.vel_buf,
            coin_weight: &self.coin_weight_buf,
            caravan_size: &self.caravan_size_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_idledrift(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

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

        // 1 Trade event per alive agent per tick.
        let event_count = self.agent_count;
        let tc_cfg = fold_trade_count::FoldTradeCountCfg {
            event_count,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.trade_count_cfg_buf,
            0,
            bytemuck::bytes_of(&tc_cfg),
        );

        // (2) fold_trade_count — RMW primary by 1.0 per Trade.
        // Dispatch sized by event count (same caller-side sizing
        // workaround the swarm_storm runtime uses; see ses_app's
        // note for context).
        let tc_bindings = fold_trade_count::FoldTradeCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.trade_count.primary(),
            view_storage_anchor: self.trade_count.anchor(),
            view_storage_ids: self.trade_count.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.trade_count_cfg_buf,
        };
        dispatch::dispatch_fold_trade_count(
            &mut self.cache,
            &tc_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.trade_count.mark_dirty();
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

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(BarteringState::new(seed, agent_count))
}
