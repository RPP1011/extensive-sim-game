//! Task 199 — GPU movement kernel.
//!
//! WGSL port of the MoveToward / Flee position updates from
//! `engine::step::apply_actions`. One thread per agent slot reads the
//! scoring output, computes the destination, writes `agents[slot].pos_*`,
//! and emits `AgentMoved` / `AgentFled` into the GPU event ring.
//!
//! # Scope (landed)
//!
//! * **MoveToward** — `new_pos = self_pos + normalize(target_pos -
//!   self_pos) * move_speed`. Reads target slot from
//!   `ScoreOutput.chosen_target`. No engagement-slow or effect-slow
//!   multiplier yet — follow-up.
//! * **Flee (pure-away)** — `new_pos = self_pos + normalize(self_pos -
//!   threat_pos) * move_speed`. Threat slot is `ScoreOutput.chosen_target`
//!   on the scorer side (Flee is self-only; CPU backend resolves threat
//!   via `nearest_hostile` so the GPU port needs the threat slot plumbed
//!   in — currently falls back to `NO_TARGET` meaning no flee).
//!
//! # Scope (deferred)
//!
//! * **Kin-flee-bias** — `spatial::flee_direction_with_kin_bias`
//!   blends a kin-centroid pull into the flee vector for species with
//!   `Capabilities.herds_when_fleeing` (Deer). Not ported — the
//!   fixtures the perf bench exercises are wolves + humans, which
//!   don't herd. The module has a stub that reads from a
//!   `kin_lists` binding but the core dispatch path only uses
//!   pure-away. Wire the kin-bias blend once the rebuilt
//!   precompute feeds into movement.
//! * **Effect slow multiplier** — q8 fixed-point slow factor; handled
//!   by physics kernel's `slow_factor_q8` field on `GpuAgentSlot` but
//!   not consumed here yet.
//! * **Engagement slow + OpportunityAttackTriggered** — the CPU path
//!   slows non-toward-engager movement and emits OpportunityAttack on
//!   both MoveToward and Flee when engaged. Skipped here.
//!
//! # Bindings
//!
//!   * `@group(0) @binding(0)` — `agents: array<AgentSlot>` (read_write).
//!     *Different group index* from scoring's `agent_data` binding so
//!     scoring can stay `read_only` on the same underlying buffer
//!     (task 198 forbids touching scoring's bindings).
//!   * `@group(0) @binding(1)` — `scoring: array<ScoreOutput>` (read)
//!   * `@group(0) @binding(2)` — `cfg: MovementCfg` (uniform)
//!   * `@group(0) @binding(3)` — `event_ring: array<EventRecord>` (read_write)
//!   * `@group(0) @binding(4)` — `event_ring_tail: atomic<u32>` (read_write)
//!
//! # Determinism
//!
//! Each thread only writes its own slot's pos; no cross-slot data
//! races. Event emit uses `atomicAdd(&tail, 1)` for claim ordering,
//! then the host drain sorts by `(tick, kind, payload[0])` to
//! reconstruct the CPU-backend push order.

#![cfg(feature = "gpu")]

use std::fmt;

use bytemuck::{Pod, Zeroable};
use engine::ids::AgentId;
use engine::state::SimState;

use crate::event_ring::{wgsl_prefix, GpuEventRing, EVENT_RING_WGSL};
use crate::physics::GpuAgentSlot;
use crate::scoring::ScoreOutput;

pub const WORKGROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// GPU-POD wire types
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct MovementCfg {
    pub agent_cap: u32,
    pub tick: u32,
    pub move_speed_mps: f32,
    pub max_move_radius: f32,
    pub kin_flee_bias: f32,
    pub kin_flee_radius: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<MovementCfg>() == 32);

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum MovementError {
    ShaderCompile(String),
    Dispatch(String),
}

impl fmt::Display for MovementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MovementError::ShaderCompile(s) => write!(f, "movement shader compile: {s}"),
            MovementError::Dispatch(s) => write!(f, "movement dispatch: {s}"),
        }
    }
}

impl std::error::Error for MovementError {}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

pub struct MovementKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pool: Option<BufferPool>,
    /// Cached resident-path bind group keyed by the caller-supplied
    /// `agents_buf`, `scoring_buf`, and `event_ring` buffer identities.
    /// Stable across a batch — pool rebuild invalidates the cache.
    cached_resident_bg: Option<(ResidentBgKey, wgpu::BindGroup)>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct ResidentBgKey {
    agent_cap: u32,
    agents: wgpu::Buffer,
    scoring: wgpu::Buffer,
    event_ring_records: wgpu::Buffer,
    event_ring_tail: wgpu::Buffer,
}

struct BufferPool {
    agent_cap: u32,
    agents_buf: wgpu::Buffer,
    agents_readback: wgpu::Buffer,
    scoring_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
}

impl MovementKernel {
    pub fn new(device: &wgpu::Device, event_ring_capacity: u32) -> Result<Self, MovementError> {
        let wgsl = build_shader(event_ring_capacity);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::movement::wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(MovementError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{wgsl}"
            )));
        }

        let storage_rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl_entries = [
            storage_rw(0), // agents
            storage_ro(1), // scoring
            uniform(2),    // cfg
            storage_rw(3), // event_ring records
            storage_rw(4), // event_ring tail
        ];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::movement::bgl"),
            entries: &bgl_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::movement::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::movement::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_movement"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            pool: None,
            cached_resident_bg: None,
        })
    }

    fn ensure_pool(&mut self, device: &wgpu::Device, agent_cap: u32) {
        if let Some(p) = &self.pool {
            if p.agent_cap == agent_cap {
                return;
            }
        }
        let agents_bytes = (agent_cap as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64);
        let scoring_bytes = (agent_cap as u64) * (std::mem::size_of::<ScoreOutput>() as u64);

        let agents_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::movement::agents"),
            size: agents_bytes.max(1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let agents_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::movement::agents_rb"),
            size: agents_bytes.max(1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scoring_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::movement::scoring"),
            size: scoring_bytes.max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::movement::cfg"),
            size: std::mem::size_of::<MovementCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.pool = Some(BufferPool {
            agent_cap,
            agents_buf,
            agents_readback,
            scoring_buf,
            cfg_buf,
        });
        // Pool rebuilt — cached BG references the old pool.cfg_buf.
        self.cached_resident_bg = None;
    }

    /// Run the movement kernel. Returns the mutated agent slots. The
    /// event ring is shared across apply_actions / movement / physics;
    /// the caller drains at a coordinated boundary.
    pub fn run_and_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        agent_slots_in: &[GpuAgentSlot],
        scoring: &[ScoreOutput],
        cfg: MovementCfg,
        event_ring: &GpuEventRing,
    ) -> Result<Vec<GpuAgentSlot>, MovementError> {
        let agent_cap = cfg.agent_cap;
        if (agent_slots_in.len() as u32) < agent_cap {
            return Err(MovementError::Dispatch(format!(
                "agent_slots_in len {} < agent_cap {}",
                agent_slots_in.len(),
                agent_cap
            )));
        }
        if (scoring.len() as u32) < agent_cap {
            return Err(MovementError::Dispatch(format!(
                "scoring len {} < agent_cap {}",
                scoring.len(),
                agent_cap
            )));
        }
        self.ensure_pool(device, agent_cap);
        let pool = self.pool.as_ref().expect("pool ensured");

        queue.write_buffer(
            &pool.agents_buf,
            0,
            bytemuck::cast_slice(&agent_slots_in[..agent_cap as usize]),
        );
        queue.write_buffer(
            &pool.scoring_buf,
            0,
            bytemuck::cast_slice(&scoring[..agent_cap as usize]),
        );
        queue.write_buffer(&pool.cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::movement::bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.agents_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.scoring_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.cfg_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: event_ring.records_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: event_ring.tail_buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::movement::enc"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::movement::cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        let agents_bytes = (agent_cap as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64);
        encoder.copy_buffer_to_buffer(
            &pool.agents_buf,
            0,
            &pool.agents_readback,
            0,
            agents_bytes,
        );
        queue.submit(Some(encoder.finish()));

        let slice = pool.agents_readback.slice(..agents_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let map_result = rx
            .recv()
            .map_err(|e| MovementError::Dispatch(format!("channel closed: {e}")))?;
        map_result.map_err(|e| MovementError::Dispatch(format!("map_async: {e:?}")))?;
        let data = slice.get_mapped_range();
        let agent_slots_out: Vec<GpuAgentSlot> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        pool.agents_readback.unmap();

        Ok(agent_slots_out)
    }

    /// Ensure the internal buffer pool can hold `agent_cap` slots.
    /// Resident entry point (B6) uses this as the owner-agnostic knob
    /// since agents / scoring / event_ring are all caller-supplied —
    /// only the cfg uniform is pool-owned.
    fn ensure_pool_cap(&mut self, device: &wgpu::Device, agent_cap: u32) {
        self.ensure_pool(device, agent_cap);
    }

    // Phase B6 (resident path): the resident path's `run_resident`
    // accepts a caller-supplied `agents_buf` (packed `GpuAgentSlot`
    // AoS, shared with the physics + apply_actions kernels) bound as
    // read_write — position updates land directly in the caller's
    // buffer. `scoring_buf` is read-only, `event_ring` is append-only
    // (records + tail are both read_write since the kernel emits
    // AgentMoved / AgentFled).
    //
    // Like apply_actions (B5), movement has no internal SoA to upload
    // — the caller owns `agents_buf` and the scorer's `scoring_buf`.
    // The only pool-owned input is the cfg uniform, which
    // [`Self::upload_soa_from_state`] writes from `SimState`. The
    // helper's name is kept (`_soa_`) for API symmetry with B3/B4
    // even though nothing SoA-shaped is uploaded here.

    /// Upload the cfg uniform from `SimState` into the pool. Must be
    /// called once per tick before [`Self::run_resident`], so the
    /// dispatch sees current `agent_cap`, `tick`, and movement speed.
    ///
    /// Separated from `run_resident` so the resident path's signature
    /// does not require a `&SimState` at dispatch time — matches the
    /// B3/B4/B5 pattern where SimState ingress is staged ahead of the
    /// cascade dispatch.
    ///
    /// Unlike the mask / scoring kernels' `upload_soa_from_state`,
    /// this helper uploads ONLY the cfg uniform — the agent and
    /// scoring buffers flow in as caller-supplied parameters to
    /// `run_resident` (since movement reuses the physics
    /// `GpuAgentSlot` layout directly, there's no per-kernel SoA to
    /// pack here).
    pub fn upload_soa_from_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &SimState,
    ) {
        let agent_cap = state.agent_cap();
        self.ensure_pool_cap(device, agent_cap);
        let pool = self.pool.as_ref().expect("pool ensured");
        let cfg = cfg_from_state(state);
        queue.write_buffer(&pool.cfg_buf, 0, bytemuck::bytes_of(&cfg));
    }

    /// Resident-path sibling to [`Self::run_and_readback`].
    ///
    /// Records the movement dispatch into `encoder`, binding
    /// caller-supplied `agents_buf` (read_write, packed
    /// `GpuAgentSlot`), `scoring_buf` (read-only, packed
    /// `ScoreOutput`), and the shared `event_ring`. Does NOT submit,
    /// does NOT read back.
    ///
    /// Agent position writes and AgentMoved / AgentFled emissions all
    /// land in the caller's buffers — no internal pool-owned mutation.
    ///
    /// ### Preconditions
    ///
    /// * [`Self::upload_soa_from_state`] must have been called on
    ///   this tick with a `state` whose `agent_cap()` equals the
    ///   `agent_cap` argument passed here. That helper uploads the
    ///   cfg uniform into the pool.
    /// * `agents_buf` must be at least `agent_cap * size_of::<GpuAgentSlot>()`
    ///   bytes and usable as `STORAGE` (read_write).
    /// * `scoring_buf` must be at least `agent_cap * size_of::<ScoreOutput>()`
    ///   bytes and usable as `STORAGE` (read-only).
    pub fn run_resident(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        scoring_buf: &wgpu::Buffer,
        event_ring: &GpuEventRing,
        agent_cap: u32,
    ) -> Result<(), MovementError> {
        self.ensure_pool_cap(device, agent_cap);

        let key = ResidentBgKey {
            agent_cap,
            agents: agents_buf.clone(),
            scoring: scoring_buf.clone(),
            event_ring_records: event_ring.records_buffer().clone(),
            event_ring_tail: event_ring.tail_buffer().clone(),
        };
        let need_rebuild = match &self.cached_resident_bg {
            Some((k, _)) => *k != key,
            None => true,
        };
        if need_rebuild {
            let pool = self.pool.as_ref().expect("pool ensured");
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::movement::bg_resident"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: agents_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: scoring_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: event_ring.records_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: event_ring.tail_buffer().as_entire_binding(),
                    },
                ],
            });
            self.cached_resident_bg = Some((key, bg));
        }
        let bind_group = &self
            .cached_resident_bg
            .as_ref()
            .expect("cached_resident_bg populated above")
            .1;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::movement::cpass_resident"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            let groups = agent_cap.div_ceil(WORKGROUP_SIZE).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        let _ = queue; // kept for signature parity with B3/B4; cfg writes go through upload_soa_from_state.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------

fn build_shader(event_ring_capacity: u32) -> String {
    let mut out = String::new();
    out.push_str(&wgsl_prefix(event_ring_capacity));

    out.push_str(
        r#"
struct AgentSlot {
    hp: f32,
    max_hp: f32,
    shield_hp: f32,
    attack_damage: f32,
    alive: u32,
    creature_type: u32,
    engaged_with: u32,
    stun_expires_at: u32,
    slow_expires_at: u32,
    slow_factor_q8: u32,
    cooldown_next_ready: u32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    _pad0: u32,
    _pad1: u32,
};

struct ScoreOutput {
    chosen_action: u32,
    chosen_target: u32,
    best_score_bits: u32,
    debug: u32,
};

struct MovementCfg {
    agent_cap: u32,
    tick: u32,
    move_speed_mps: f32,
    max_move_radius: f32,
    kin_flee_bias: f32,
    kin_flee_radius: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> agents: array<AgentSlot>;
@group(0) @binding(1) var<storage, read> scoring: array<ScoreOutput>;
@group(0) @binding(2) var<uniform> cfg: MovementCfg;
@group(0) @binding(3) var<storage, read_write> event_ring: array<EventRecord>;
@group(0) @binding(4) var<storage, read_write> event_ring_tail: atomic<u32>;
"#,
    );

    out.push_str(EVENT_RING_WGSL);
    out.push('\n');

    out.push_str(&format!(
        r#"
const ACTION_MOVE_TOWARD: u32 = 1u;
const ACTION_FLEE: u32 = 2u;
const NO_TARGET: u32 = 0xFFFFFFFFu;

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {{
    let len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 <= 0.0) {{ return vec3<f32>(0.0, 0.0, 0.0); }}
    let inv = 1.0 / sqrt(len2);
    return v * inv;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn cs_movement(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let slot = gid.x;
    if (slot >= cfg.agent_cap) {{ return; }}
    if (agents[slot].alive == 0u) {{ return; }}

    let self_id = slot + 1u;
    let so = scoring[slot];
    let action = so.chosen_action;
    let t_slot_raw = so.chosen_target;
    let tick = cfg.tick;

    let sx = agents[slot].pos_x;
    let sy = agents[slot].pos_y;
    let sz = agents[slot].pos_z;
    let self_pos = vec3<f32>(sx, sy, sz);

    if (action == ACTION_MOVE_TOWARD) {{
        if (t_slot_raw == NO_TARGET) {{ return; }}
        if (t_slot_raw >= cfg.agent_cap) {{ return; }}
        if (agents[t_slot_raw].alive == 0u) {{ return; }}
        let tx = agents[t_slot_raw].pos_x;
        let ty = agents[t_slot_raw].pos_y;
        let tz = agents[t_slot_raw].pos_z;
        let target_pos = vec3<f32>(tx, ty, tz);
        let delta = target_pos - self_pos;
        if (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z <= 0.0) {{ return; }}
        let dir = normalize_or_zero(delta);
        // Speed = move_speed_mps. Engagement-slow / effect-slow are
        // deferred (see module header); the perf path treats movement
        // as the unslowed baseline.
        let speed = cfg.move_speed_mps;
        let new_pos = self_pos + dir * speed;
        agents[slot].pos_x = new_pos.x;
        agents[slot].pos_y = new_pos.y;
        agents[slot].pos_z = new_pos.z;
        let _mv_idx = gpu_emit_agent_moved(self_id, tick,
                                           self_pos.x, self_pos.y, self_pos.z,
                                           new_pos.x, new_pos.y, new_pos.z);
        return;
    }}

    if (action == ACTION_FLEE) {{
        if (t_slot_raw == NO_TARGET) {{ return; }}
        if (t_slot_raw >= cfg.agent_cap) {{ return; }}
        // Note: unlike MoveToward, `chosen_target` on a Flee row is
        // the *threat* slot, which the scorer resolves via the Flee
        // mask's candidate list. Dead threats mean no flee — matches
        // the CPU guard.
        if (agents[t_slot_raw].alive == 0u) {{ return; }}
        let tx = agents[t_slot_raw].pos_x;
        let ty = agents[t_slot_raw].pos_y;
        let tz = agents[t_slot_raw].pos_z;
        let threat_pos = vec3<f32>(tx, ty, tz);
        // Pure-away flee. Kin-flee-bias blend is deferred (see module
        // header) — requires kin_lists precompute + species capability
        // lookup.
        let away = normalize_or_zero(self_pos - threat_pos);
        if (away.x * away.x + away.y * away.y + away.z * away.z <= 0.0) {{ return; }}
        let speed = cfg.move_speed_mps;
        let new_pos = self_pos + away * speed;
        agents[slot].pos_x = new_pos.x;
        agents[slot].pos_y = new_pos.y;
        agents[slot].pos_z = new_pos.z;
        let _fl_idx = gpu_emit_agent_fled(self_id, tick,
                                          self_pos.x, self_pos.y, self_pos.z,
                                          new_pos.x, new_pos.y, new_pos.z);
        return;
    }}
}}
"#
    ));

    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn cfg_from_state(state: &SimState) -> MovementCfg {
    MovementCfg {
        agent_cap: state.agent_cap(),
        tick: state.tick,
        move_speed_mps: state.config.movement.move_speed_mps,
        max_move_radius: state.config.movement.max_move_radius,
        kin_flee_bias: state.config.combat.kin_flee_bias,
        kin_flee_radius: state.config.combat.kin_flee_radius,
        _pad0: 0,
        _pad1: 0,
    }
}

/// Commit the movement kernel's pos updates onto `SimState`. Only
/// writes `pos` — alive / hp / shield / etc. are owned by other
/// kernels (apply_actions / physics).
pub fn unpack_movement_slots(state: &mut SimState, slots: &[GpuAgentSlot]) {
    use glam::Vec3;
    for (slot_idx, s) in slots.iter().enumerate() {
        let id = match AgentId::new(slot_idx as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        if !state.agent_alive(id) || s.alive == 0 {
            continue;
        }
        state.set_agent_pos(id, Vec3::new(s.pos_x, s.pos_y, s.pos_z));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<MovementCfg>(), 32);
    }

    #[test]
    fn shader_parses_through_naga() {
        let wgsl = build_shader(1024);
        if let Err(e) = naga::front::wgsl::parse_str(&wgsl) {
            panic!(
                "movement shader failed naga parse:\n{e}\n--- WGSL source ---\n{wgsl}"
            );
        }
    }
}
