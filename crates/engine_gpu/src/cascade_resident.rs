//! GPU-resident cascade driver. Records `MAX_CASCADE_ITERATIONS`
//! physics iterations into a single command encoder with no Rust-side
//! convergence check and no per-iter readback — convergence is encoded
//! as indirect dispatch args written by each iteration's physics kernel
//! (0 workgroups = no-op for downstream iters once events stop
//! propagating).
//!
//! The sync cascade in [`crate::cascade`] is untouched — that path
//! remains authoritative for callers that need determinism or per-tick
//! CPU observability. This module is scaffolding for Phase D's
//! `step_batch(n)` which encodes one submit + one poll across N ticks.
//!
//! ### Scope of Task C1
//!
//! This module encodes the full per-iteration physics dispatch sequence
//! plus the two resident spatial queries + the indirect-args seed
//! kernel. It does *not* yet run the fold kernels that project events
//! into view storage — `cascade::fold_iteration_events` is sync-only
//! and re-entering it here would reintroduce per-iter submit+poll.
//! Task D4 is scheduled to add a resident fold kernel; until then
//! callers that need view storage updated must invoke the sync fold
//! path separately.

use bytemuck::{Pod, Zeroable};

use engine::state::SimState;

use crate::cascade::{CascadeCtx, MAX_CASCADE_ITERATIONS};
use crate::event_ring::{GpuEventRing, DEFAULT_CAPACITY, PAYLOAD_WORDS};
use crate::gpu_util::indirect::IndirectArgsBuffer;
use crate::physics::{
    GpuAgentSlot, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PHYSICS_WORKGROUP_SIZE,
    MAX_ABILITIES, MAX_EFFECTS,
};
use crate::spatial_gpu::{GpuQueryResult, SpatialOutputs};

/// Capacity of the ping-pong physics event rings owned by
/// [`CascadeResidentCtx`]. Matches [`crate::cascade::APPLY_EVENT_RING_CAPACITY`]
/// so the rings have headroom equivalent to the apply-path ring the
/// cascade consumes on iter 0.
pub const PHYSICS_EVENT_RING_CAPACITY: u32 = crate::cascade::APPLY_EVENT_RING_CAPACITY;

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

/// Errors surfaced by the resident cascade driver. Wraps sub-component
/// error types so the caller gets a single `Result` type.
#[derive(Debug)]
pub enum CascadeResidentError {
    Physics(crate::physics::PhysicsError),
    Spatial(crate::spatial_gpu::SpatialError),
    /// Any other kernel init / dispatch failure originating in this
    /// module's own helpers (e.g. seed kernel compile).
    Kernel(String),
}

impl std::fmt::Display for CascadeResidentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CascadeResidentError::Physics(e) => write!(f, "cascade_resident physics: {e}"),
            CascadeResidentError::Spatial(e) => write!(f, "cascade_resident spatial: {e}"),
            CascadeResidentError::Kernel(s) => write!(f, "cascade_resident kernel: {s}"),
        }
    }
}

impl std::error::Error for CascadeResidentError {}

impl From<crate::physics::PhysicsError> for CascadeResidentError {
    fn from(e: crate::physics::PhysicsError) -> Self {
        CascadeResidentError::Physics(e)
    }
}

impl From<crate::spatial_gpu::SpatialError> for CascadeResidentError {
    fn from(e: crate::spatial_gpu::SpatialError) -> Self {
        CascadeResidentError::Spatial(e)
    }
}

// ---------------------------------------------------------------------------
// Resident spatial output buffer trios
// ---------------------------------------------------------------------------

/// Two caller-owned spatial output trios kept alive across ticks. One
/// trio per radius: `kin` (12 m — feeds `nearby_kin`) and `engagement`
/// (2 m — feeds `nearest_hostile`). Reallocated iff `agent_cap` grows.
struct ResidentSpatialBuffers {
    agent_cap: u32,
    // kin-radius trio
    kin_within: wgpu::Buffer,
    kin_kin: wgpu::Buffer,
    kin_nearest: wgpu::Buffer,
    // engagement-radius trio
    eng_within: wgpu::Buffer,
    eng_kin: wgpu::Buffer,
    eng_nearest: wgpu::Buffer,
}

impl ResidentSpatialBuffers {
    fn new(device: &wgpu::Device, agent_cap: u32) -> Self {
        let qr_bytes = (agent_cap as u64) * (std::mem::size_of::<GpuQueryResult>() as u64);
        let nearest_bytes = (agent_cap as u64) * 4;
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mk = |size: u64, label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        Self {
            agent_cap,
            kin_within: mk(qr_bytes, "cascade_resident::spatial::kin_within"),
            kin_kin: mk(qr_bytes, "cascade_resident::spatial::kin_kin"),
            kin_nearest: mk(nearest_bytes, "cascade_resident::spatial::kin_nearest"),
            eng_within: mk(qr_bytes, "cascade_resident::spatial::eng_within"),
            eng_kin: mk(qr_bytes, "cascade_resident::spatial::eng_kin"),
            eng_nearest: mk(nearest_bytes, "cascade_resident::spatial::eng_nearest"),
        }
    }

    fn kin_outputs(&self) -> SpatialOutputs<'_> {
        SpatialOutputs {
            within: &self.kin_within,
            kin: &self.kin_kin,
            nearest: &self.kin_nearest,
        }
    }

    fn engagement_outputs(&self) -> SpatialOutputs<'_> {
        SpatialOutputs {
            within: &self.eng_within,
            kin: &self.eng_kin,
            nearest: &self.eng_nearest,
        }
    }
}

// ---------------------------------------------------------------------------
// Resident ability buffers (caller-owned for the physics resident path)
// ---------------------------------------------------------------------------

/// Caller-owned ability-registry buffers the physics resident path
/// reads from. Sized for `MAX_ABILITIES` × `MAX_EFFECTS` at construction
/// — fixed by the kernel's WGSL bounds so the buffers never need to
/// grow.
///
/// Registry contents don't change on most ticks (only on unit
/// composition / ability lineage changes). `upload` hashes the packed
/// registry and skips the 4× queue.write_buffer call if unchanged —
/// saves ~64 KB of host-to-GPU traffic + 4 encoder-internal
/// serialisation points per tick on the hot path.
struct ResidentAbilityBuffers {
    known: wgpu::Buffer,
    cooldown: wgpu::Buffer,
    effects_count: wgpu::Buffer,
    effects: wgpu::Buffer,
    /// Hash of the last uploaded `PackedAbilityRegistry`. `None` =
    /// never uploaded. Computed over the concatenation of the four
    /// slices that get written; collision-resistant enough for a
    /// same-tick dirty check (FxHash over ~64 KB is microseconds).
    last_hash: std::cell::Cell<Option<u64>>,
}

impl ResidentAbilityBuffers {
    fn new(device: &wgpu::Device) -> Self {
        use crate::physics::GpuEffectOp;
        let u32_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (MAX_ABILITIES * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let effects = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade_resident::abilities::effects"),
            size: (MAX_ABILITIES * MAX_EFFECTS) as u64
                * std::mem::size_of::<GpuEffectOp>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            known: u32_buf("cascade_resident::abilities::known"),
            cooldown: u32_buf("cascade_resident::abilities::cooldown"),
            effects_count: u32_buf("cascade_resident::abilities::effects_count"),
            effects,
            last_hash: std::cell::Cell::new(None),
        }
    }

    fn upload(&self, queue: &wgpu::Queue, abilities: &PackedAbilityRegistry) {
        // Content-addressed dirty check: skip the 4× write_buffer if
        // the registry bytes haven't changed since last upload. The
        // hash is over all four slice contents — any single-byte
        // change anywhere triggers a full re-upload. `std::hash`'s
        // DefaultHasher is SipHash-1-3 which is fine for a
        // non-adversarial equality check (the alternative — a per-tick
        // memcmp against the old bytes — would need keeping a shadow
        // copy, which defeats the point).
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytemuck::cast_slice::<_, u8>(&abilities.known).hash(&mut hasher);
        bytemuck::cast_slice::<_, u8>(&abilities.cooldown).hash(&mut hasher);
        bytemuck::cast_slice::<_, u8>(&abilities.effects_count).hash(&mut hasher);
        bytemuck::cast_slice::<_, u8>(&abilities.effects).hash(&mut hasher);
        let h = hasher.finish();
        if self.last_hash.get() == Some(h) {
            return;
        }
        queue.write_buffer(&self.known, 0, bytemuck::cast_slice(&abilities.known));
        queue.write_buffer(&self.cooldown, 0, bytemuck::cast_slice(&abilities.cooldown));
        queue.write_buffer(
            &self.effects_count,
            0,
            bytemuck::cast_slice(&abilities.effects_count),
        );
        queue.write_buffer(&self.effects, 0, bytemuck::cast_slice(&abilities.effects));
        self.last_hash.set(Some(h));
    }
}

// ---------------------------------------------------------------------------
// Seed-indirect kernel
// ---------------------------------------------------------------------------

/// Uniform for the seed kernel. `cap_wg` clamps the seeded workgroup
/// count at `ceil(agent_cap / PHYSICS_WORKGROUP_SIZE)` so a huge
/// initial-event batch can't overflow the kernel's one-thread-per-agent
/// bound.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct SeedIndirectCfg {
    cap_wg: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// WGSL source for the seed kernel. Writes slot 0 of the indirect args
/// buffer + slot 0 of the num_events buffer from the apply-path ring
/// tail atomic. One workgroup, one thread.
const SEED_INDIRECT_WGSL: &str = r#"
struct SeedCfg {
    cap_wg: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Must match `engine_gpu::sim_cfg::SimCfg` byte-for-byte. The
// `sim_cfg_layout` regression test fences the Rust-side offsets;
// any drift surfaces as subtle corruption of world-scalar fields.
struct SimCfg {
    tick:                          atomic<u32>,
    world_seed_lo:                 u32,
    world_seed_hi:                 u32,
    _pad0:                         u32,
    engagement_range:              f32,
    attack_damage:                 f32,
    attack_range:                  f32,
    move_speed:                    f32,
    move_speed_mult:               f32,
    kin_radius:                    f32,
    cascade_max_iterations:        u32,
    rules_registry_generation:     u32,
    abilities_registry_generation: u32,
    _reserved0:                    u32,
    _reserved1:                    u32,
    _reserved2:                    u32,
};

@group(0) @binding(0) var<storage, read>       apply_tail: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> indirect_args: array<u32>;
@group(0) @binding(2) var<storage, read_write> num_events: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: SeedCfg;
@group(0) @binding(4) var<storage, read_write> sim_cfg: SimCfg;

const WG: u32 = 64u;

@compute @workgroup_size(1)
fn seed() {
    let n = atomicLoad(&apply_tail[0]);
    num_events[0] = n;
    let req = (n + WG - 1u) / WG;
    var wg = req;
    if (wg > cfg.cap_wg) { wg = cfg.cap_wg; }
    indirect_args[0] = wg;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
    // Advance GPU-side tick once per cascade dispatch. Single thread,
    // single atomic — no race. CPU still increments `state.tick` until
    // Task 2.10 removes that; during the overlap both paths advance in
    // lockstep (no observer reads `sim_cfg.tick` yet — Task 2.11 wires
    // it for snapshot readback).
    atomicAdd(&sim_cfg.tick, 1u);
}
"#;

/// Small compute kernel that writes slot 0 of both the cascade
/// indirect-args buffer and the per-iter num_events buffer from the
/// apply-path event ring's atomic tail. Runs as a single 1-workgroup,
/// 1-thread dispatch at the top of every tick's resident cascade.
struct SeedIndirectKernel {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cfg_buf: wgpu::Buffer,
    /// `cap_wg` the cfg buffer was last uploaded with; used to skip
    /// redundant uploads on stable agent_cap.
    last_cap_wg: u32,
    /// Cached BG keyed by the 3 caller-supplied buffer identities.
    /// All are stable across a batch, so the cache hits 100% after
    /// tick 1.
    cached_bg: Option<(SeedBgKey, wgpu::BindGroup)>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct SeedBgKey {
    apply_tail: wgpu::Buffer,
    indirect_args: wgpu::Buffer,
    num_events: wgpu::Buffer,
    sim_cfg: wgpu::Buffer,
}

impl SeedIndirectKernel {
    fn new(device: &wgpu::Device) -> Result<Self, CascadeResidentError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cascade_resident::seed::shader"),
            source: wgpu::ShaderSource::Wgsl(SEED_INDIRECT_WGSL.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(CascadeResidentError::Kernel(format!(
                "seed shader compile: {err}"
            )));
        }

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cascade_resident::seed::bgl"),
            // apply_tail: read-only storage (but contains atomic<u32>; read-only is fine for atomicLoad)
            // binding 4: sim_cfg (rw storage) — kernel atomically increments `tick` at end-of-dispatch.
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                uniform(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<crate::sim_cfg::SimCfg>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cascade_resident::seed::pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cascade_resident::seed::pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("seed"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade_resident::seed::cfg"),
            size: std::mem::size_of::<SeedIndirectCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            pipeline,
            bgl,
            cfg_buf,
            last_cap_wg: u32::MAX,
            cached_bg: None,
        })
    }

    /// Encode a 1-thread dispatch that reads `apply_tail[0]` and writes
    /// slot 0 of `indirect_args` + `num_events`. Uploads the `cap_wg`
    /// uniform iff it changed since the last call (typical case:
    /// agent_cap is stable across ticks).
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        apply_tail_buf: &wgpu::Buffer,
        indirect_args: &IndirectArgsBuffer,
        num_events_buf: &wgpu::Buffer,
        sim_cfg_buf: &wgpu::Buffer,
        agent_cap: u32,
    ) {
        let cap_wg = agent_cap.div_ceil(PHYSICS_WORKGROUP_SIZE).max(1);
        if cap_wg != self.last_cap_wg {
            let cfg = SeedIndirectCfg {
                cap_wg,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(&self.cfg_buf, 0, bytemuck::bytes_of(&cfg));
            self.last_cap_wg = cap_wg;
        }
        let key = SeedBgKey {
            apply_tail: apply_tail_buf.clone(),
            indirect_args: indirect_args.buffer().clone(),
            num_events: num_events_buf.clone(),
            sim_cfg: sim_cfg_buf.clone(),
        };
        let need_rebuild = match &self.cached_bg {
            Some((k, _)) => *k != key,
            None => true,
        };
        if need_rebuild {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cascade_resident::seed::bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: apply_tail_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: indirect_args.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: num_events_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: sim_cfg_buf.as_entire_binding(),
                    },
                ],
            });
            self.cached_bg = Some((key, bg));
        }
        let bg = &self.cached_bg.as_ref().expect("cached_bg populated").1;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cascade_resident::seed::cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// Append-events kernel (C1 fix)
// ---------------------------------------------------------------------------
//
// `step_batch` clears `apply_event_ring.tail` at the top of every tick so
// the per-iter cascade seed kernel sees a fresh count. That clear means
// only the *last* tick's apply+movement events are observable at end-of-
// batch — all prior ticks' events are lost to the snapshot consumer.
//
// This kernel accumulates apply+movement events into a dedicated
// `batch_events_ring` across every tick in a batch. Per-tick flow:
//
//   1. apply_actions + movement run, appending to `apply_event_ring`.
//   2. This append kernel dispatches: each thread i < apply_tail copies
//      record i from apply_event_ring into batch_events_ring at slot
//      atomic-reserved on batch_tail.
//   3. Seed kernel reads apply_tail for the cascade.
//   4. apply_event_ring.tail is cleared (at the top of the *next* tick).
//
// The batch ring is reset once per `step_batch` call so
// snapshot()-observed events are scoped to that batch's execution.
//
// Dispatch size is computed on CPU from a conservative upper bound
// (apply + movement together can emit at most `agent_cap * 4`
// events — 2 events per agent per kernel, very loose). Threads past
// `apply_tail` early-exit after reading the atomic once, so the
// over-dispatch cost is a few hundred no-op threads per tick.

/// WGSL source for the append kernel. Each thread handles one record.
/// `apply_tail[0]` bounds the work; `batch_tail[0]` is atomicAdd'd once
/// per copied record to reserve a destination slot.
///
/// Record layout: 10 u32s (kind + tick + 8 payload words). See
/// [`crate::event_ring::RECORD_BYTES`].
const APPEND_EVENTS_WGSL: &str = r#"
const RECORD_U32S: u32 = 10u;

@group(0) @binding(0) var<storage, read>       apply_tail: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read>       apply_records: array<u32>;
@group(0) @binding(2) var<storage, read_write> batch_tail: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> batch_records: array<u32>;
@group(0) @binding(4) var<uniform>             cfg: AppendCfg;

struct AppendCfg {
    batch_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@compute @workgroup_size(64)
fn append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = atomicLoad(&apply_tail[0]);
    let i = gid.x;
    if (i >= n) { return; }
    let dst = atomicAdd(&batch_tail[0], 1u);
    if (dst >= cfg.batch_cap) { return; }
    let src_off = i * RECORD_U32S;
    let dst_off = dst * RECORD_U32S;
    for (var w: u32 = 0u; w < RECORD_U32S; w = w + 1u) {
        batch_records[dst_off + w] = apply_records[src_off + w];
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct AppendCfg {
    batch_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Small compute kernel that copies `apply_event_ring.records[0..tail]`
/// into `batch_events_ring.records[batch_tail..]`, advancing the batch
/// ring's atomic tail by the apply-tail count. Used by `step_batch`
/// after apply+movement and before the cascade seed, so the batch ring
/// accumulates ALL apply+movement events across every tick in the
/// batch (rather than just the last tick's, which is all that
/// `apply_event_ring` still holds by end-of-batch because its tail is
/// cleared at the top of each tick for cascade-seed correctness).
struct AppendEventsKernel {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cfg_buf: wgpu::Buffer,
    /// `batch_cap` the cfg buffer was last uploaded with.
    last_cap: u32,
    /// Cached BG keyed by the five caller-supplied buffer identities.
    /// apply_ring and batch_ring are stable across a batch, so the
    /// cache hits 100% after tick 1.
    cached_bg: Option<(AppendBgKey, wgpu::BindGroup)>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct AppendBgKey {
    apply_tail: wgpu::Buffer,
    apply_records: wgpu::Buffer,
    batch_tail: wgpu::Buffer,
    batch_records: wgpu::Buffer,
}

impl AppendEventsKernel {
    fn new(device: &wgpu::Device) -> Result<Self, CascadeResidentError> {
        // Sanity: WGSL assumes 10 u32s per record.
        debug_assert_eq!(2 + PAYLOAD_WORDS, 10);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cascade_resident::append::shader"),
            source: wgpu::ShaderSource::Wgsl(APPEND_EVENTS_WGSL.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(CascadeResidentError::Kernel(format!(
                "append shader compile: {err}"
            )));
        }

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cascade_resident::append::bgl"),
            entries: &[
                storage(0, true),  // apply_tail (atomic — read via atomicLoad)
                storage(1, true),  // apply_records
                storage(2, false), // batch_tail (atomicAdd)
                storage(3, false), // batch_records
                uniform(4),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cascade_resident::append::pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cascade_resident::append::pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("append"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade_resident::append::cfg"),
            size: std::mem::size_of::<AppendCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            pipeline,
            bgl,
            cfg_buf,
            last_cap: u32::MAX,
            cached_bg: None,
        })
    }

    /// Encode the dispatch. `agent_cap` is used to compute a
    /// conservative workgroup count — apply + movement together emit at
    /// most ~2 events per agent per kernel (very loose upper bound: 4).
    /// Threads past the observed `apply_tail` early-exit after a single
    /// atomic load.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        apply_ring: &GpuEventRing,
        batch_ring: &GpuEventRing,
        agent_cap: u32,
    ) {
        let batch_cap = batch_ring.capacity();
        if batch_cap != self.last_cap {
            let cfg = AppendCfg {
                batch_cap,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(&self.cfg_buf, 0, bytemuck::bytes_of(&cfg));
            self.last_cap = batch_cap;
        }
        let key = AppendBgKey {
            apply_tail: apply_ring.tail_buffer().clone(),
            apply_records: apply_ring.records_buffer().clone(),
            batch_tail: batch_ring.tail_buffer().clone(),
            batch_records: batch_ring.records_buffer().clone(),
        };
        let need_rebuild = match &self.cached_bg {
            Some((k, _)) => *k != key,
            None => true,
        };
        if need_rebuild {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cascade_resident::append::bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: apply_ring.tail_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: apply_ring.records_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: batch_ring.tail_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: batch_ring.records_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.cfg_buf.as_entire_binding(),
                    },
                ],
            });
            self.cached_bg = Some((key, bg));
        }
        let bg = &self.cached_bg.as_ref().expect("cached_bg populated").1;
        // Conservative upper bound: 4 events per agent across apply +
        // movement (real max ~2). Threads past apply_tail early-exit.
        let max_events = agent_cap.saturating_mul(4).max(64);
        let workgroups = max_events.div_ceil(PHYSICS_WORKGROUP_SIZE).max(1);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cascade_resident::append::cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// Driver context
// ---------------------------------------------------------------------------

/// Per-backend state the resident cascade driver owns across ticks.
/// Holds the caller-owned buffer trios the resident kernels consume so
/// we don't pay a fresh allocation per tick. Construct once on the
/// first `run_cascade_resident` call; buffers grow lazily with
/// `agent_cap`.
///
/// Phase D moves this onto `GpuBackend` alongside `cascade_ctx`. For
/// C1 it's constructed inline by [`run_cascade_resident`] via
/// [`Self::ensure`] on a caller-held `Option<CascadeResidentCtx>` — the
/// caller owns the lifetime so the Phase D wiring is a rename away.
pub struct CascadeResidentCtx {
    spatial_bufs: Option<ResidentSpatialBuffers>,
    ability_bufs: ResidentAbilityBuffers,
    seed_kernel: SeedIndirectKernel,
    /// Append kernel that copies apply_event_ring contents into the
    /// batch-scoped `batch_events_ring` on each tick. See
    /// [`AppendEventsKernel`] for rationale.
    append_kernel: AppendEventsKernel,
    /// Ping-pong pair of physics event rings. Iter `i` (starting at
    /// 0) reads from `apply_event_ring` (for i=0) or the ring at index
    /// `(i - 1) & 1` (for i>0), and writes into the ring at index
    /// `i & 1`. Both rings live here so the driver can bind whichever
    /// pair the current iteration needs without the caller caring.
    physics_ring_a: GpuEventRing,
    physics_ring_b: GpuEventRing,
    /// Per-iter event count buffer. Slot `i` holds the number of events
    /// the iter-`i` dispatch must process (slot 0 is written by the
    /// seed kernel; slots 1..=MAX are written by physics in its
    /// epilogue). Sized for `MAX_CASCADE_ITERATIONS + 1` u32s so the
    /// final iteration's epilogue has a valid write target.
    num_events_buf: wgpu::Buffer,
    /// Chronicle ring — caller-owned in the resident path. Shared
    /// across all iterations (append-only).
    pub(crate) chronicle_ring: crate::event_ring::GpuChronicleRing,
    /// Batch-scoped accumulator for apply + movement events. The
    /// physics `apply_event_ring` is cleared at the top of every tick
    /// in `step_batch` (the cascade seed kernel reads its tail and we
    /// can't re-seed without zeroing), which means only the *last*
    /// tick's events remain observable at end-of-batch. The append
    /// kernel copies each tick's apply+movement events into this ring
    /// before the cascade seed runs, so `snapshot()` can observe the
    /// full N-tick batch. Reset at the start of each `step_batch` call.
    pub(crate) batch_events_ring: GpuEventRing,
    /// Persistent staging buffer for num_events readback, sized for
    /// `MAX_CASCADE_ITERATIONS + 1` u32 slots. Used by
    /// [`Self::encode_num_events_readback`] + [`Self::read_num_events_blocking`]
    /// to observe cascade convergence at end-of-batch without paying a
    /// per-tick fence. See [`Self::batch_observed_max_iters`] and
    /// `lib.rs::step_batch`'s iter-cap heuristic for consumers.
    num_events_staging: wgpu::Buffer,
    /// Cascade convergence observed on the final tick of the previous
    /// `step_batch` submit. Used to cap the number of physics
    /// iterations recorded per tick on subsequent batches — workloads
    /// that converge at iter 1-2 save 5-6 per-iter encodes each tick
    /// (each costs ~50-200 µs of CPU compute-pass + bind-group cost,
    /// even when the indirect dispatch args are `(x=0, y=1, z=1)` no-ops).
    ///
    /// Initialised to `MAX_CASCADE_ITERATIONS` (the conservative
    /// bootstrap cap) on construction. Updated after each batch submit
    /// by `GpuBackend::step_batch` via
    /// [`Self::read_num_events_blocking`]. Next batch reads this
    /// field + a `+2` margin and passes the clamped value as
    /// `run_cascade_resident_with_iter_cap`'s `max_iters` param.
    pub(crate) batch_observed_max_iters: u32,
    /// `agent_cap` the resident physics ability buffers + agent-cap
    /// dependent buffers were last sized for.
    last_agent_cap: u32,
}

impl CascadeResidentCtx {
    /// Build a fresh driver context on the supplied device. Allocates
    /// the ability-registry buffers (fixed size — `MAX_ABILITIES * MAX_EFFECTS`),
    /// the two physics event rings at `PHYSICS_EVENT_RING_CAPACITY`,
    /// the per-iter `num_events` buffer sized for `MAX_CASCADE_ITERATIONS + 1`
    /// u32s, the seed kernel, and a fresh chronicle ring at
    /// `DEFAULT_CHRONICLE_CAPACITY`. Spatial output buffers are lazy —
    /// allocated on first `run_cascade_resident` once `state.agent_cap`
    /// is known.
    pub fn new(device: &wgpu::Device) -> Result<Self, CascadeResidentError> {
        let ability_bufs = ResidentAbilityBuffers::new(device);
        let seed_kernel = SeedIndirectKernel::new(device)?;
        let append_kernel = AppendEventsKernel::new(device)?;
        let physics_ring_a = GpuEventRing::new(device, PHYSICS_EVENT_RING_CAPACITY);
        let physics_ring_b = GpuEventRing::new(device, PHYSICS_EVENT_RING_CAPACITY);
        // `MAX_CASCADE_ITERATIONS + 1` u32 slots: iter `i` reads
        // num_events[i] and physics writes num_events[i+1] in its
        // epilogue (write_slot is `i+1`). Max i is `MAX_CASCADE_ITERATIONS - 1`,
        // so max write_slot is `MAX_CASCADE_ITERATIONS`.
        let num_events_slots = (MAX_CASCADE_ITERATIONS + 1) as u64;
        let num_events_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade_resident::num_events"),
            size: num_events_slots * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Persistent staging buffer for per-batch convergence readback
        // (Stage B.1). Sized for `MAX_CASCADE_ITERATIONS + 1` u32 slots
        // (36 B). MAP_READ | COPY_DST usage so we can copy num_events
        // into it at end-of-batch and map_async after submit+poll.
        let num_events_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade_resident::num_events_staging"),
            size: num_events_slots * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let chronicle_ring = crate::event_ring::GpuChronicleRing::new(
            device,
            crate::event_ring::DEFAULT_CHRONICLE_CAPACITY,
        );
        // Batch-scoped event accumulator sized at the main-ring default
        // capacity. At 655_360 records × 40 B that's ~25 MiB — matches
        // the `apply_event_ring` envelope the batch path already budgets
        // and gives ample room for ~N ticks × ~10× overflow.
        let batch_events_ring = GpuEventRing::new(device, DEFAULT_CAPACITY);
        Ok(Self {
            spatial_bufs: None,
            ability_bufs,
            seed_kernel,
            append_kernel,
            physics_ring_a,
            physics_ring_b,
            num_events_buf,
            chronicle_ring,
            batch_events_ring,
            num_events_staging,
            // Conservative bootstrap: first batch must record the full
            // `MAX_CASCADE_ITERATIONS` dispatches. Subsequent batches
            // narrow this based on observed convergence.
            batch_observed_max_iters: MAX_CASCADE_ITERATIONS,
            last_agent_cap: 0,
        })
    }

    /// Encode the append dispatch that copies `apply_event_ring`'s
    /// contents into the batch-scoped accumulator. Used by
    /// `step_batch` after apply+movement and before the cascade seed.
    pub(crate) fn encode_append_apply_events(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        apply_ring: &GpuEventRing,
        agent_cap: u32,
    ) {
        self.append_kernel.record(
            device,
            queue,
            encoder,
            apply_ring,
            &self.batch_events_ring,
            agent_cap,
        );
    }

    /// Reset the batch-scoped accumulator's tail. Called at the top of
    /// each `step_batch` so each batch begins with a fresh events view.
    /// Emits a GPU `clear_buffer` op (ordered inside the command buffer,
    /// not a queue write).
    pub(crate) fn reset_batch_events_ring(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.batch_events_ring.tail_buffer(), 0, None);
    }

    /// Reset the chronicle ring's tail. Called at the top of each
    /// `step_batch` so chronicle emissions are scoped to the current
    /// batch — otherwise the append-only ring would accumulate across
    /// every batch call in a long-running session and eventually
    /// overflow its `DEFAULT_CHRONICLE_CAPACITY`.
    pub(crate) fn reset_chronicle_ring(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.chronicle_ring.tail_buffer(), 0, None);
    }

    /// Read-only handle to the batch-scoped events accumulator.
    /// `GpuBackend::snapshot()` reads this ring instead of
    /// `apply_event_ring` so the snapshot covers every tick in the
    /// batch (the apply ring holds only the last tick's events because
    /// its tail is cleared per-tick for cascade-seed correctness).
    pub(crate) fn batch_events_ring(&self) -> &GpuEventRing {
        &self.batch_events_ring
    }

    /// Stage B.1 — encode a `copy_buffer_to_buffer` from `num_events_buf`
    /// into the persistent `num_events_staging` buffer. Called by
    /// `step_batch` after the final tick's cascade is encoded, before
    /// `queue.submit`. Does NOT poll, does NOT map — caller owns the
    /// fence via the step_batch submit+poll that already runs at
    /// end-of-batch.
    ///
    /// The staging buffer holds the final tick's per-iter event counts:
    /// slot `i` = count of events the iter-`i` physics dispatch read.
    /// `slot[k] == 0` implies iter `k-1` converged (emitted no events).
    /// [`Self::read_num_events_blocking`] reads the mapped range once
    /// the submit's poll has returned.
    pub(crate) fn encode_num_events_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let byte_len = (MAX_CASCADE_ITERATIONS as u64 + 1) * 4;
        encoder.copy_buffer_to_buffer(
            &self.num_events_buf,
            0,
            &self.num_events_staging,
            0,
            byte_len,
        );
    }

    /// Stage B.1 — map + read + unmap the `num_events_staging` buffer,
    /// returning the `MAX_CASCADE_ITERATIONS + 1` per-iter event counts
    /// from the LAST tick of the most recent submit.
    ///
    /// **Precondition:** the caller has already run `queue.submit` +
    /// `device.poll(Wait)` AFTER [`Self::encode_num_events_readback`]
    /// was invoked on the submitted encoder. Calling before the poll
    /// returns a `map_async` error; calling without a prior encode
    /// returns stale data (or zeros if this is the first invocation).
    ///
    /// Uses `map_async` + a blocking `device.poll(Wait)` to flush the
    /// map operation itself. The `Wait` here is cheap (no compute
    /// work in flight once the batch submit has poll-completed), so
    /// this does NOT re-fence the batch path's non-fence property.
    pub(crate) fn read_num_events_blocking(
        &self,
        device: &wgpu::Device,
    ) -> Result<Vec<u32>, String> {
        let slice = self.num_events_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // The flush poll is required for map_async to complete on
        // wgpu backends that don't auto-drive the mapping queue.
        // This does NOT block on compute work — the batch submit's
        // poll already returned before we called this.
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv()
            .map_err(|e| format!("num_events_staging channel closed: {e}"))?
            .map_err(|e| format!("num_events_staging map_async: {e:?}"))?;
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.num_events_staging.unmap();
        Ok(out)
    }

    /// Stage B.1 — from a per-iter num_events readback, compute the
    /// highest iter index that actually processed events (i.e. read a
    /// non-zero `num_events[i]` and thus was not a GPU-side no-op).
    /// Returns `0` if all iters were no-ops (nothing to cascade).
    ///
    /// `num_events[0]` is the seed-kernel write (always the
    /// apply_event_ring tail). `num_events[i]` for `i >= 1` is written
    /// by iter-`(i-1)`'s physics epilogue and read by iter-`i`'s
    /// dispatch. So if `num_events[i] > 0`, iter `i` ran real work.
    ///
    /// The caller adds a `+2` margin (for run-to-run variance) and
    /// clamps to `MAX_CASCADE_ITERATIONS` before using as
    /// `run_cascade_resident_with_iter_cap`'s `max_iters`.
    pub(crate) fn observed_last_active_iter(counts: &[u32]) -> u32 {
        let mut last_active: u32 = 0;
        // counts[i] drives iter i; i in 0..MAX. counts[MAX] is the
        // final iter's epilogue-write (an exhaustion counter).
        let max_probe = counts.len().min(MAX_CASCADE_ITERATIONS as usize) as u32;
        for i in 0..max_probe {
            if counts[i as usize] > 0 {
                last_active = i + 1;
            }
        }
        last_active
    }

    /// Read-only handle to the dedicated chronicle ring.
    /// `GpuBackend::snapshot()` reads this ring's tail and records
    /// buffer to populate `GpuSnapshot::chronicle_since_last`.
    pub(crate) fn chronicle_ring(&self) -> &crate::event_ring::GpuChronicleRing {
        &self.chronicle_ring
    }

    /// Test-only accessor for `num_events_buf` so parent-crate test
    /// harnesses can read back the per-iter event counts the cascade
    /// writes. Not part of a stable public API.
    #[doc(hidden)]
    pub(crate) fn num_events_buf_for_test(&self) -> &wgpu::Buffer {
        &self.num_events_buf
    }

    fn ensure_spatial(&mut self, device: &wgpu::Device, agent_cap: u32) {
        let need = !matches!(
            &self.spatial_bufs,
            Some(b) if b.agent_cap >= agent_cap
        );
        if need {
            self.spatial_bufs = Some(ResidentSpatialBuffers::new(device, agent_cap));
        }
    }

    /// For iter `i`, returns `(events_in_records, events_out_ring)` —
    /// the input records buffer the physics kernel reads from and the
    /// output ring it appends to. `apply_event_ring_records` is the
    /// buffer handed in by the caller that holds the initial events
    /// (consumed on iter 0 only).
    fn iter_rings<'a>(
        &'a self,
        iter: u32,
        apply_event_ring_records: &'a wgpu::Buffer,
    ) -> (&'a wgpu::Buffer, &'a GpuEventRing) {
        let out = if iter & 1 == 0 {
            &self.physics_ring_a
        } else {
            &self.physics_ring_b
        };
        let in_buf = if iter == 0 {
            apply_event_ring_records
        } else if (iter - 1) & 1 == 0 {
            self.physics_ring_a.records_buffer()
        } else {
            self.physics_ring_b.records_buffer()
        };
        (in_buf, out)
    }
}

// ---------------------------------------------------------------------------
// Public driver entry point
// ---------------------------------------------------------------------------

/// One-tick resident cascade. Encodes into `encoder`:
///
///   1. Two spatial queries (kin radius + engagement range) into the
///      driver's caller-owned output buffer trios.
///   2. Seed of `indirect_args[0]` + `num_events[0]` from
///      `apply_event_ring`'s atomic tail.
///   3. `MAX_CASCADE_ITERATIONS` indirect physics dispatches, each
///      reading `indirect_args[iter]` and writing
///      `indirect_args[iter+1]` via the physics kernel's epilogue. On
///      convergence (an iteration that emits zero events) every
///      subsequent iteration indirect-dispatches with
///      `(x=0, y=1, z=1)`, so the trailing kernels run but do no work.
///
/// Preconditions:
///   * The caller has already run mask → scoring → apply+movement
///     earlier in the same `encoder` (or earlier in the submit batch),
///     so `apply_event_ring` contains this tick's initial events.
///   * `agents_buf` is sized to `state.agent_cap() *
///     size_of::<GpuAgentSlot>()` bytes with usage `STORAGE`.
///   * `indirect_args` has at least `MAX_CASCADE_ITERATIONS + 1` slots.
///   * `ctx.abilities` is populated with the current tick's registry.
///
/// Does NOT submit, does NOT poll, does NOT drain events. The caller
/// owns those steps — the whole point of this driver is to keep the
/// cascade resident in a single command buffer.
///
/// TODO(D4): view-storage folding is deferred until a resident fold
/// kernel lands. Callers that need view storage updated by the
/// cascade's emissions must invoke
/// [`crate::cascade::fold_iteration_events`] on a separately-drained
/// event list — that path submits + polls internally which is at odds
/// with the resident driver's one-submit-per-tick goal.
#[allow(clippy::too_many_arguments)]
pub fn run_cascade_resident(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    state: &SimState,
    cascade_ctx: &mut CascadeCtx,
    resident_ctx: &mut CascadeResidentCtx,
    agents_buf: &wgpu::Buffer,
    apply_event_ring: &GpuEventRing,
    indirect_args: &IndirectArgsBuffer,
    sim_cfg_buf: &wgpu::Buffer,
    gold_buf: &wgpu::Buffer,
    standing_records_buf: &wgpu::Buffer,
    standing_counts_buf: &wgpu::Buffer,
    memory_records_buf: &wgpu::Buffer,
    memory_cursors_buf: &wgpu::Buffer,
    alive_bitmap_buf: &wgpu::Buffer,
) -> Result<(), CascadeResidentError> {
    run_cascade_resident_with_iter_cap(
        device,
        queue,
        encoder,
        state,
        cascade_ctx,
        resident_ctx,
        agents_buf,
        apply_event_ring,
        indirect_args,
        sim_cfg_buf,
        gold_buf,
        standing_records_buf,
        standing_counts_buf,
        memory_records_buf,
        memory_cursors_buf,
        alive_bitmap_buf,
        MAX_CASCADE_ITERATIONS,
        None,
    )
}

/// Variant of [`run_cascade_resident`] that caps the number of physics
/// dispatches recorded at `max_iters`. When `max_iters <
/// MAX_CASCADE_ITERATIONS`, subsequent iterations are skipped entirely
/// — neither encoded nor dispatched — saving the per-iter CPU encode
/// cost for workloads where convergence is known (from a prior tick's
/// observed iteration count). If a given tick happens to need more
/// iterations than the cap, remaining propagation is truncated;
/// callers SHOULD only set a low cap when the application tolerates
/// this (e.g. perf-critical paths with monitored convergence means).
#[allow(clippy::too_many_arguments)]
pub fn run_cascade_resident_with_iter_cap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    state: &SimState,
    cascade_ctx: &mut CascadeCtx,
    resident_ctx: &mut CascadeResidentCtx,
    agents_buf: &wgpu::Buffer,
    apply_event_ring: &GpuEventRing,
    indirect_args: &IndirectArgsBuffer,
    sim_cfg_buf: &wgpu::Buffer,
    gold_buf: &wgpu::Buffer,
    standing_records_buf: &wgpu::Buffer,
    standing_counts_buf: &wgpu::Buffer,
    memory_records_buf: &wgpu::Buffer,
    memory_cursors_buf: &wgpu::Buffer,
    alive_bitmap_buf: &wgpu::Buffer,
    max_iters: u32,
    // Perf Stage A.1 — optional GPU timestamp profiler. `None` means
    // no timestamps are emitted (original behaviour). When `Some`, the
    // driver emits one `write_timestamp` at seed-begin, one at
    // seed-end, and one at the begin+end of each cascade iteration so
    // the test harness can surface per-phase GPU µs.
    profiler: Option<&mut crate::gpu_profiling::GpuProfiler>,
) -> Result<(), CascadeResidentError> {
    let max_iters = max_iters.clamp(1, MAX_CASCADE_ITERATIONS);
    let agent_cap = state.agent_cap();

    // Perf Stage A.1 — reborrow the profiler into an Option<&mut> once,
    // then use `as_deref_mut` via a shadowed binding. The cascade
    // driver emits marks at: spatial+abilities begin, seed_kernel,
    // per-iter begins, cascade_end. Each mark is a no-op when the
    // profiler is disabled.
    let mut prof = profiler;
    if let Some(p) = prof.as_deref_mut() {
        p.mark(encoder, "spatial+abilities");
    }

    // ---- 1. Spatial queries ---------------------------------------------
    // Two resident dispatches into distinct caller-owned output trios
    // so the engagement query doesn't clobber the kin query. Each call
    // allocates a per-call `qcfg` uniform internally so the two radii
    // don't race on a shared uniform.
    resident_ctx.ensure_spatial(device, agent_cap);
    resident_ctx.last_agent_cap = agent_cap;
    let spatial_bufs = resident_ctx
        .spatial_bufs
        .as_ref()
        .expect("spatial_bufs ensured");

    // Both radii are now designer-tunable via `state.config.combat.*`
    // (`kin_radius` was promoted from a hardcoded const on 2026-04-22;
    // `engagement_range` has always been config-driven). SimCfg mirrors
    // both fields, but the spatial query kernel takes the radius as a
    // per-call parameter on its own uniform, so we read from Config
    // directly here (batch-constant, no GPU readback needed).
    let kin_radius = state.config.combat.kin_radius;
    let engagement_range = state.config.combat.engagement_range;

    // Split the spatial pipeline: the CPU SoA pack + clear/count/scan/
    // scatter/sort passes are radius-independent and run exactly once
    // per tick. Only the query kernel (which reads qcfg) runs per
    // radius, against its own caller-owned output trio.
    cascade_ctx.spatial.rebuild_resident(device, queue, encoder, state)?;
    cascade_ctx.spatial.query_resident(
        device,
        encoder,
        agent_cap,
        kin_radius,
        spatial_bufs.kin_outputs(),
    );
    cascade_ctx.spatial.query_resident(
        device,
        encoder,
        agent_cap,
        engagement_range,
        spatial_bufs.engagement_outputs(),
    );

    // ---- 2. Upload ability registry to resident buffers -----------------
    // Must happen before the first physics dispatch binds the buffers.
    resident_ctx.ability_bufs.upload(queue, &cascade_ctx.abilities);

    // ---- 3. Reset physics event rings + seed indirect slot 0 ------------
    // Both physics rings start each tick at tail=0 so the ping-pong
    // binds a known-clean output buffer. The chronicle ring is *not*
    // reset — it's append-only across the whole run (apps drain it
    // separately).
    //
    // We use `encoder.clear_buffer` rather than `queue.write_buffer`
    // because the clears must be ordered *inside* the command buffer:
    // later iterations re-use the same ring as their output and
    // re-clearing its tail between iterations needs to land *between*
    // adjacent compute passes, not at submit time (queue writes all
    // execute before any command-buffer command, collapsing multiple
    // writes to the same offset). `clear_buffer` emits a transfer
    // operation at the current encoder position, so the ordering
    // matches the iteration schedule.
    encoder.clear_buffer(resident_ctx.physics_ring_a.tail_buffer(), 0, None);
    encoder.clear_buffer(resident_ctx.physics_ring_b.tail_buffer(), 0, None);
    // Zero the entire num_events buffer so slots 1..=MAX_ITER start at
    // 0. The seed kernel overwrites slot 0 on dispatch. Ordered inside
    // the encoder for the same reason as the ring-tail clears above —
    // the seed dispatch must see the zeroed slots, and later
    // iterations' epilogues overwrite their write-slot anyway.
    encoder.clear_buffer(&resident_ctx.num_events_buf, 0, None);

    // Perf Stage A.1 — `seed_kernel` mark lands after spatial rebuild
    // + query + ability-registry upload so their GPU time is credited
    // to `spatial+abilities` rather than leaking into `seed_kernel`.
    if let Some(p) = prof.as_deref_mut() {
        // Between append_events and seed_kernel (spatial_done boundary).
        p.write_between_pass_timestamp(
            encoder,
            crate::gpu_profiling::BETWEEN_PASS_LABELS_PRE_CASCADE[6],
        );
        p.mark(encoder, "seed_kernel");
    }

    // Seed dispatch: reads apply_event_ring.tail, writes
    // indirect_args[0] + num_events[0].
    resident_ctx.seed_kernel.record(
        device,
        queue,
        encoder,
        apply_event_ring.tail_buffer(),
        indirect_args,
        &resident_ctx.num_events_buf,
        sim_cfg_buf,
        agent_cap,
    );

    // Between seed_kernel and cascade iter 0.
    if let Some(p) = prof.as_deref_mut() {
        p.write_between_pass_timestamp(
            encoder,
            crate::gpu_profiling::BETWEEN_PASS_LABELS_PRE_CASCADE[7],
        );
    }

    // ---- 4. Encode N physics iterations ---------------------------------
    // Task 2.8 — world-scalars (`tick`, `combat_engagement_range`,
    // `cascade_max_iterations`) migrated to the shared SimCfg storage
    // buffer. `PhysicsCfg` now carries only kernel-local fields; physics
    // reads `state.tick` from `sim_cfg.tick` (atomically incremented
    // by the seed-indirect kernel, not taken from this `state`).
    let cfg_template = PhysicsCfg {
        // `num_events` in the `PhysicsCfg` uniform is *unused* by the
        // resident entry point — it reads its count from
        // `num_events_buf[read_slot]` instead. We keep a 0 default so
        // any stale read surfaces obviously as a no-op.
        num_events: 0,
        agent_cap,
        max_abilities: MAX_ABILITIES as u32,
        max_effects: MAX_EFFECTS as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };

    for iter in 0..max_iters {
        // Pick this iter's input records buffer + output ring. Iter 0
        // reads from `apply_event_ring`; iter 1+ alternates between
        // the two resident physics rings.
        let (events_in_buf, events_out_ring) =
            resident_ctx.iter_rings(iter, apply_event_ring.records_buffer());

        // Perf Stage A.1 — mark begin of this cascade iter.
        if let Some(p) = prof.as_deref_mut() {
            let label = crate::gpu_profiling::CASCADE_ITER_BEGIN_LABELS
                .get(iter as usize)
                .copied()
                .unwrap_or("cascade iter N");
            p.mark(encoder, label);
        }

        cascade_ctx.physics.run_batch_resident(
            device,
            queue,
            encoder,
            agents_buf,
            &resident_ctx.ability_bufs.known,
            &resident_ctx.ability_bufs.cooldown,
            &resident_ctx.ability_bufs.effects_count,
            &resident_ctx.ability_bufs.effects,
            // Spatial kin buffer: the spatial kernel's `kin` output
            // and the physics `KinList` have the same byte layout
            // (u32 count at offset 0, ids[32] at offset 16 — see
            // `GpuKinList` and `GpuQueryResult`), so we bind the
            // kin-radius trio's `kin` buffer directly.
            &spatial_bufs.kin_kin,
            &spatial_bufs.eng_nearest,
            events_in_buf,
            events_out_ring,
            &resident_ctx.chronicle_ring,
            indirect_args,
            &resident_ctx.num_events_buf,
            sim_cfg_buf,
            gold_buf,
            standing_records_buf,
            standing_counts_buf,
            memory_records_buf,
            memory_cursors_buf,
            alive_bitmap_buf,
            iter,       // read_slot
            iter + 1,   // write_slot
            cfg_template,
        )?;

        // For iter 1+ the *next* iter's output ring was used two
        // iterations ago (ping-pong partner); its tail still holds the
        // count from that earlier dispatch. Zero it here via
        // `encoder.clear_buffer` so the next iter's atomicAdds start
        // at 0. `clear_buffer` is ordered inside the command buffer,
        // so this clear lands AFTER the current iter's dispatch reads
        // that ring's records buffer as events_in (iter 1 reads
        // ring_a; iter 2 reads ring_b; etc.) but BEFORE the next
        // iter's dispatch writes to its tail.
        if iter + 2 < max_iters {
            let next_next_out = if (iter + 2) & 1 == 0 {
                &resident_ctx.physics_ring_a
            } else {
                &resident_ctx.physics_ring_b
            };
            encoder.clear_buffer(next_next_out.tail_buffer(), 0, None);
        }

        // Per-dispatch attribution: emit a between-pass mark right after
        // this iter's physics dispatch (and the ping-pong ring clear).
        // Paired with the NEXT iter's begin mark, the delta isolates the
        // inter-iter barrier + pipeline-swap cost from the iter's own
        // compute time.
        if let Some(p) = prof.as_deref_mut() {
            let label = crate::gpu_profiling::BETWEEN_CASCADE_LABELS
                .get(iter as usize)
                .copied()
                .unwrap_or("gap:cascade_iter_N_done");
            p.write_between_pass_timestamp(encoder, label);
        }
    }

    // Perf Stage A.1 — close the last cascade iter's timing interval.
    // The profiler reports `(label_i, delta(ts_{i+1} - ts_i))`, so the
    // final iter needs a trailing stamp to produce its µs entry.
    if let Some(p) = prof.as_deref_mut() {
        p.mark(encoder, "cascade_end");
    }

    // Silence the "imports unused" lint if MAX_ABILITIES/MAX_EFFECTS
    // or GpuAgentSlot/GpuKinList are only referenced through doc
    // paths (not the case today — keeping a tiny anchor).
    let _ = std::mem::size_of::<GpuAgentSlot>();
    let _ = std::mem::size_of::<GpuKinList>();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Stage B.1 — pure-decode tests for `observed_last_active_iter`.
    // Guards the convergence-detection logic so a one-off GPU-readback
    // regression can't silently mis-classify a converged tick.

    #[test]
    fn observed_last_active_iter_all_zeros_reports_zero() {
        let counts = vec![0u32; (MAX_CASCADE_ITERATIONS + 1) as usize];
        assert_eq!(CascadeResidentCtx::observed_last_active_iter(&counts), 0);
    }

    #[test]
    fn observed_last_active_iter_only_slot_0_reports_one() {
        let mut counts = vec![0u32; (MAX_CASCADE_ITERATIONS + 1) as usize];
        counts[0] = 42;
        assert_eq!(CascadeResidentCtx::observed_last_active_iter(&counts), 1);
    }

    #[test]
    fn observed_last_active_iter_contiguous_range_reports_last() {
        // counts[0..=2] nonzero, counts[3..] zero: iter 2 was the last
        // one that read a nonzero event count, so last_active is 3.
        let mut counts = vec![0u32; (MAX_CASCADE_ITERATIONS + 1) as usize];
        counts[0] = 100;
        counts[1] = 20;
        counts[2] = 3;
        assert_eq!(CascadeResidentCtx::observed_last_active_iter(&counts), 3);
    }

    #[test]
    fn observed_last_active_iter_sparse_tail_reports_deepest_active() {
        // A gap in the middle should not hide a deeper active iter.
        let mut counts = vec![0u32; (MAX_CASCADE_ITERATIONS + 1) as usize];
        counts[0] = 5;
        counts[1] = 0; // physically can't happen — seed always positive
        counts[3] = 7; // but be robust to degenerate readbacks
        assert_eq!(CascadeResidentCtx::observed_last_active_iter(&counts), 4);
    }

    #[test]
    fn observed_last_active_iter_ignores_trailing_epilogue_write() {
        // counts[MAX] is the exhaustion epilogue write — the decoder
        // should not treat it as an "active iter" (it's one PAST the
        // last iter that actually ran).
        let mut counts = vec![0u32; (MAX_CASCADE_ITERATIONS + 1) as usize];
        counts[MAX_CASCADE_ITERATIONS as usize] = 999;
        assert_eq!(CascadeResidentCtx::observed_last_active_iter(&counts), 0);
    }

    #[test]
    fn observed_last_active_iter_full_cascade_reports_max() {
        let counts = vec![1u32; (MAX_CASCADE_ITERATIONS + 1) as usize];
        assert_eq!(
            CascadeResidentCtx::observed_last_active_iter(&counts),
            MAX_CASCADE_ITERATIONS
        );
    }
}
