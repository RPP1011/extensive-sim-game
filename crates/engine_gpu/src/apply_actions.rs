//! Task 199 — GPU `apply_actions` kernel.
//!
//! WGSL port of the hot subset of `engine::step::apply_actions`. One
//! thread per agent slot reads the companion scoring kernel's
//! `ScoreOutput`, dispatches on `chosen_action`, mutates the agent SoA
//! (hp, shield, hunger, thirst, rest_timer), and emits replayable
//! events (AgentAttacked, AgentAte, AgentDrank, AgentRested, AgentDied)
//! directly into the GPU event ring.
//!
//! # What this kernel covers
//!
//! * **Attack** — range-check (self.pos vs target.pos <= attack_range),
//!   apply `damage` to target's hp (post-shield), emit `AgentAttacked`,
//!   emit `AgentDied` + set target alive=0 if new_hp <= 0.
//! * **Eat / Drink / Rest** — delta-based need restore with `min(1.0)`
//!   clamp, emit `AgentAte` / `AgentDrank` / `AgentRested`.
//! * **Hold / MoveToward / Flee** — no events (movement kernel emits
//!   AgentMoved / AgentFled), no state mutation.
//! * **Cast** — skipped (matches CPU physics kernel's cast handling;
//!   the current sim never produces Cast from the scorer anyway).
//!
//! # What this kernel DOESN'T cover (intentional gap vs CPU)
//!
//! The CPU `apply_actions` does a lot more work that doesn't show up
//! hot on the profile and would balloon the WGSL:
//!
//! * **Opportunity attacks + engagement slow** on MoveToward — the
//!   engagement-aware speed scaling + `OpportunityAttackTriggered`
//!   emit is still in the CPU path. Moving this to WGSL requires
//!   `agent_engaged_with` as mutable SoA on GPU (which physics already
//!   has) plus a hostile-pos lookup on the opponent.
//! * **Effect slow multiplier** on movement — the q8 fixed-point slow
//!   read is available to the physics kernel but would need to feed
//!   apply_actions too.
//! * **Announce + overhear + channel-gated communication** — these
//!   fire for the `Announce` macro action, which the scorer never
//!   selects in the hot combat fixtures. Ignored.
//! * **Cast** — see above.
//!
//! This is the same deliberate subsetting task 197 did for the
//! "scoring → actions → CPU apply_actions" bridge: cover the hot path
//! that the N=1000 fixture actually exercises; defer the tail to a
//! follow-up if profile reveals them as bottlenecks.
//!
//! # Bindings
//!
//!   * `@group(0) @binding(0)` — `agents: array<ActionApplyAgent>` (read_write)
//!   * `@group(0) @binding(1)` — `scoring: array<ScoreOutput>` (read)
//!   * `@group(0) @binding(2)` — `cfg: ApplyActionsCfg` (uniform)
//!   * `@group(0) @binding(3)` — `event_ring: array<EventRecord>` (read_write)
//!   * `@group(0) @binding(4)` — `event_ring_tail: atomic<u32>` (read_write)
//!   * `@group(0) @binding(5)` — `sim_cfg: SimCfg` (storage, read-only) — Task 2.6
//!
//! 6 bindings, well under the 16-per-group cap. Event ring borrows
//! `GpuEventRing`'s buffers; the physics + apply_actions kernels share
//! the same event ring (different dispatches, same underlying storage).
//!
//! Task 2.6 of the GPU sim-state refactor migrated the `tick` +
//! `attack_range_default` reads out of the per-kernel cfg uniform and
//! onto the shared `SimCfg` storage buffer bound at `@binding(5)`. The
//! remaining per-kernel cfg fields are all kernel-local
//! (`agent_cap` for dispatch bounds; unused restore stubs kept for
//! future Eat/Drink/Rest porting).

#![cfg(feature = "gpu")]

use std::fmt;

use bytemuck::{Pod, Zeroable};
use engine::ids::AgentId;
use engine::state::SimState;

use crate::event_ring::{wgsl_prefix, GpuEventRing, EVENT_RING_WGSL};
use crate::physics::GpuAgentSlot;
use crate::scoring::ScoreOutput;

/// Workgroup size for the apply_actions kernel. 64 threads × ceil(N/64)
/// groups — one thread per agent slot. Matches the mask/scoring
/// kernels for consistency.
pub const WORKGROUP_SIZE: u32 = 64;

/// Binding number of the shared `SimCfg` storage buffer (Task 2.6).
/// Sits immediately past the event-ring tail binding (4) so it's the
/// last entry in the BGL; the shared WGSL emitter writes
/// `@binding(SIM_CFG_BINDING)` into the shader source.
const SIM_CFG_BINDING: u32 = 5;

// ---------------------------------------------------------------------------
// GPU-POD wire types
// ---------------------------------------------------------------------------

/// Per-slot agent record the apply_actions kernel reads and writes.
/// Strict subset of `GpuAgentSlot` (from physics.rs) so callers can
/// share buffers — we upload a `Vec<GpuAgentSlot>` verbatim and the
/// kernel only touches the fields documented here.
///
/// Kept as a WGSL-struct-equivalent for documentation; the Rust side
/// uses `GpuAgentSlot` directly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ActionApplyAgent {
    pub hp: f32,
    pub max_hp: f32,
    pub shield_hp: f32,
    pub attack_damage: f32,
    pub alive: u32,
    pub creature_type: u32,
    pub attack_range: f32,
    pub hunger: f32,
    pub thirst: f32,
    pub fatigue: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<ActionApplyAgent>() == 64);

/// Uniform config carried per-dispatch. 32 bytes (WGSL uniform rule
/// minimum alignment).
///
/// Task 2.6 of the GPU sim-state refactor migrated `tick` +
/// `attack_range_default` out of this struct; they now live in the
/// shared `SimCfg` storage buffer (`sim_cfg.tick` /
/// `sim_cfg.attack_range`). The remaining fields are either
/// subsystem-local (dispatch bound `agent_cap`) or unused stubs
/// preserved for a future Eat/Drink/Rest port (see the `Needs` comment
/// in `cs_apply_actions`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct ApplyActionsCfg {
    pub agent_cap: u32,
    pub attack_damage_default: f32,
    pub eat_restore: f32,
    pub drink_restore: f32,
    pub rest_restore: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<ApplyActionsCfg>() == 32);

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ApplyActionsError {
    ShaderCompile(String),
    Dispatch(String),
}

impl fmt::Display for ApplyActionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyActionsError::ShaderCompile(s) => write!(f, "apply_actions shader compile: {s}"),
            ApplyActionsError::Dispatch(s) => write!(f, "apply_actions dispatch: {s}"),
        }
    }
}

impl std::error::Error for ApplyActionsError {}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

pub struct ApplyActionsKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pool: Option<BufferPool>,
    /// Cached resident-path bind group keyed by the caller-supplied
    /// `agents_buf`, `scoring_buf`, and `event_ring` buffer identities.
    /// All of these are stable across a batch (resident buffers don't
    /// swap), so the cache hits 100% after tick 1. Invalidated when
    /// the pool is rebuilt (agent_cap grow) since the bound
    /// `pool.cfg_buf` would become stale.
    cached_resident_bg: Option<(ResidentBgKey, wgpu::BindGroup)>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct ResidentBgKey {
    agent_cap: u32,
    agents: wgpu::Buffer,
    scoring: wgpu::Buffer,
    event_ring_records: wgpu::Buffer,
    event_ring_tail: wgpu::Buffer,
    /// Task 2.6: caller-supplied shared `SimCfg` buffer. Stable across
    /// a batch (backend holds one resident buffer), so adding it here
    /// still amortises to a single BG build per batch.
    sim_cfg: wgpu::Buffer,
}

struct BufferPool {
    agent_cap: u32,
    agents_buf: wgpu::Buffer,
    agents_readback: wgpu::Buffer,
    scoring_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    /// Pool-owned `SimCfg` snapshot buffer — used only by the sync
    /// path (`run_and_readback`) so it stays self-contained. The
    /// resident path (`run_resident`) binds the caller-supplied
    /// `sim_cfg_buf` at `SIM_CFG_BINDING` instead.
    sync_sim_cfg_buf: wgpu::Buffer,
}

impl ApplyActionsKernel {
    pub fn new(device: &wgpu::Device, event_ring_capacity: u32) -> Result<Self, ApplyActionsError> {
        let wgsl = build_shader(event_ring_capacity);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::apply_actions::wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(ApplyActionsError::ShaderCompile(format!(
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
        let sim_cfg_storage_ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(
                    std::mem::size_of::<crate::sim_cfg::SimCfg>() as u64,
                ),
            },
            count: None,
        };
        let bgl_entries = [
            storage_rw(0), // agents
            storage_ro(1), // scoring
            uniform(2),    // cfg
            storage_rw(3), // event_ring records
            storage_rw(4), // event_ring tail
            sim_cfg_storage_ro(SIM_CFG_BINDING), // Task 2.6: shared SimCfg
        ];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::apply_actions::bgl"),
            entries: &bgl_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::apply_actions::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::apply_actions::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_apply_actions"),
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
            label: Some("engine_gpu::apply_actions::agents"),
            size: agents_bytes.max(1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let agents_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::apply_actions::agents_rb"),
            size: agents_bytes.max(1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scoring_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::apply_actions::scoring"),
            size: scoring_bytes.max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::apply_actions::cfg"),
            size: std::mem::size_of::<ApplyActionsCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Task 2.6: sync-path fallback `SimCfg` buffer. `run_and_readback`
        // uploads a fresh `SimCfg::from_state(state)` here every tick
        // so the sync path stays self-contained without requiring a
        // caller-supplied resident `sim_cfg_buf`. The resident path
        // ignores this and binds the caller's buffer instead.
        let sync_sim_cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::apply_actions::sync_sim_cfg"),
            size: std::mem::size_of::<crate::sim_cfg::SimCfg>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.pool = Some(BufferPool {
            agent_cap,
            agents_buf,
            agents_readback,
            scoring_buf,
            cfg_buf,
            sync_sim_cfg_buf,
        });
        // Pool rebuild — cached BG references the old pool.cfg_buf
        // (and implicitly the old agent_cap). Drop it.
        self.cached_resident_bg = None;
    }

    /// Run the apply_actions kernel against `agent_slots_in`, the
    /// scoring outputs, and the caller-owned event ring. Returns the
    /// mutated agent slots. The event ring is drained by the caller
    /// (the ring is shared with the physics kernel so the caller
    /// orchestrates the drain timing).
    ///
    /// `sim_cfg` carries the world-scalar sim state (tick + attack
    /// range) that this kernel reads via the shared `SimCfg` binding.
    /// The sync path uploads this into the pool-owned fallback buffer
    /// each tick; the resident path (which binds a caller-supplied
    /// `sim_cfg_buf` instead) does not touch it. Migrated from
    /// `ApplyActionsCfg` in Task 2.6 of the GPU sim-state refactor.
    pub fn run_and_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        agent_slots_in: &[GpuAgentSlot],
        scoring: &[ScoreOutput],
        cfg: ApplyActionsCfg,
        sim_cfg: &crate::sim_cfg::SimCfg,
        event_ring: &GpuEventRing,
    ) -> Result<Vec<GpuAgentSlot>, ApplyActionsError> {
        let agent_cap = cfg.agent_cap;
        if (agent_slots_in.len() as u32) < agent_cap {
            return Err(ApplyActionsError::Dispatch(format!(
                "agent_slots_in len {} < agent_cap {}",
                agent_slots_in.len(),
                agent_cap
            )));
        }
        if (scoring.len() as u32) < agent_cap {
            return Err(ApplyActionsError::Dispatch(format!(
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
        // Task 2.6: upload the SimCfg snapshot into the pool-owned
        // fallback buffer so the sync bind group's `sim_cfg` binding
        // sees current `tick` + `attack_range`.
        crate::sim_cfg::upload_sim_cfg(queue, &pool.sync_sim_cfg_buf, sim_cfg);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::apply_actions::bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pool.agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool.scoring_buf.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: SIM_CFG_BINDING,
                    resource: pool.sync_sim_cfg_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::apply_actions::enc"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::apply_actions::cpass"),
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
            .map_err(|e| ApplyActionsError::Dispatch(format!("channel closed: {e}")))?;
        map_result.map_err(|e| ApplyActionsError::Dispatch(format!("map_async: {e:?}")))?;
        let data = slice.get_mapped_range();
        let agent_slots_out: Vec<GpuAgentSlot> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        pool.agents_readback.unmap();

        Ok(agent_slots_out)
    }

    /// Ensure the internal buffer pool can hold `agent_cap` slots.
    /// Resident entry point (B5) uses this as the owner-agnostic knob
    /// since agents / scoring / event_ring are all caller-supplied —
    /// only the cfg uniform is pool-owned.
    fn ensure_pool_cap(&mut self, device: &wgpu::Device, agent_cap: u32) {
        self.ensure_pool(device, agent_cap);
    }

    // Phase B5 (resident path): the resident path's `run_resident`
    // accepts a caller-supplied `agents_buf` (packed `GpuAgentSlot`
    // AoS, shared with the physics kernel) bound as read_write — HP
    // deltas and alive=0 writes land directly in the caller's buffer.
    // `scoring_buf` is read-only, `event_ring` is append-only (records
    // + tail are both read_write since the kernel emits
    // AgentAttacked / AgentDied).
    //
    // Unlike B3 (mask) + B4 (scoring), apply_actions has no internal
    // SoA to upload — the caller owns `agents_buf` and the scorer's
    // `scoring_buf`. The only pool-owned input is the cfg uniform,
    // which [`Self::upload_soa_from_state`] writes from `SimState`.
    // The helper's name is kept (`_soa_`) for API symmetry with B3/B4
    // even though nothing SoA-shaped is uploaded here.

    /// Upload the cfg uniform from `SimState` into the pool. Must be
    /// called once per tick before [`Self::run_resident`], so the
    /// dispatch sees current `agent_cap`, `tick`, and restore deltas.
    ///
    /// Separated from `run_resident` so the resident path's signature
    /// does not require a `&SimState` at dispatch time — matches the
    /// B3/B4 pattern where SimState ingress is staged ahead of the
    /// cascade dispatch.
    ///
    /// Unlike the mask / scoring kernels' `upload_soa_from_state`,
    /// this helper uploads ONLY the cfg uniform — the agent and
    /// scoring buffers flow in as caller-supplied parameters to
    /// `run_resident` (since apply_actions reuses the physics
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
    /// Records the apply_actions dispatch into `encoder`, binding
    /// caller-supplied `agents_buf` (read_write, packed
    /// `GpuAgentSlot`), `scoring_buf` (read-only, packed
    /// `ScoreOutput`), `sim_cfg_buf` (read-only shared world-scalars,
    /// Task 2.6), and the shared `event_ring`. Does NOT submit, does
    /// NOT read back.
    ///
    /// Agent HP deltas, alive=0 writes, and event emissions all land
    /// in the caller's buffers — no internal pool-owned mutation.
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
    /// * `sim_cfg_buf` must be at least `size_of::<SimCfg>()` bytes and
    ///   usable as `STORAGE` (read-only). The backend populates it
    ///   once per batch via `SimCfg::from_state(state)` + atomic tick
    ///   increments on the seed-indirect kernel.
    pub fn run_resident(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        scoring_buf: &wgpu::Buffer,
        sim_cfg_buf: &wgpu::Buffer,
        event_ring: &GpuEventRing,
        agent_cap: u32,
    ) -> Result<(), ApplyActionsError> {
        self.ensure_pool_cap(device, agent_cap);

        let key = ResidentBgKey {
            agent_cap,
            agents: agents_buf.clone(),
            scoring: scoring_buf.clone(),
            event_ring_records: event_ring.records_buffer().clone(),
            event_ring_tail: event_ring.tail_buffer().clone(),
            sim_cfg: sim_cfg_buf.clone(),
        };
        let need_rebuild = match &self.cached_resident_bg {
            Some((k, _)) => *k != key,
            None => true,
        };
        if need_rebuild {
            let pool = self.pool.as_ref().expect("pool ensured");
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::apply_actions::bg_resident"),
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
                    wgpu::BindGroupEntry {
                        binding: SIM_CFG_BINDING,
                        resource: sim_cfg_buf.as_entire_binding(),
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
                label: Some("engine_gpu::apply_actions::cpass_resident"),
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
// Shader construction
// ---------------------------------------------------------------------------

fn build_shader(event_ring_capacity: u32) -> String {
    let mut out = String::new();
    // EVENT_RING_CAP / EVENT_RING_PAYLOAD_WORDS consts.
    out.push_str(&wgsl_prefix(event_ring_capacity));

    // Structs. AgentSlot here mirrors `physics::GpuAgentSlot` so the
    // backend can share the same agent buffer with the physics kernel.
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

struct ApplyActionsCfg {
    agent_cap: u32,
    attack_damage_default: f32,
    eat_restore: f32,
    drink_restore: f32,
    rest_restore: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> agents: array<AgentSlot>;
@group(0) @binding(1) var<storage, read> scoring: array<ScoreOutput>;
@group(0) @binding(2) var<uniform> cfg: ApplyActionsCfg;
@group(0) @binding(3) var<storage, read_write> event_ring: array<EventRecord>;
@group(0) @binding(4) var<storage, read_write> event_ring_tail: atomic<u32>;
"#,
    );

    // SimCfg binding (Task 2.6) — shared world-scalars, read-only
    // storage. This kernel reads `sim_cfg.tick` (event emission
    // timestamps) and `sim_cfg.attack_range` (range-check guard on the
    // Attack head). The struct declaration + binding come from the
    // dsl_compiler shared helper so the WGSL stays in lockstep with
    // `SimCfg`'s Rust layout.
    dsl_compiler::emit_sim_cfg::emit_sim_cfg_struct_wgsl(&mut out, SIM_CFG_BINDING);

    // Pull in the event_ring WGSL — defines `EventRecord` and the
    // `gpu_emit_event_*` helpers. Note: we use only a tiny subset
    // (attacked, died, ate, drank, rested).
    //
    // EVENT_RING_WGSL declares its own `event_ring` + `event_ring_tail`
    // bindings in a doc comment — the actual bindings are the ones we
    // declared above. The helper fns reference them as globals which
    // WGSL resolves through module scope.
    out.push_str(EVENT_RING_WGSL);
    out.push('\n');

    // Action discriminants — match MicroKind ordinals from
    // `engine::mask::MicroKind`.
    out.push_str(
        r#"
const ACTION_HOLD: u32 = 0u;
const ACTION_MOVE_TOWARD: u32 = 1u;
const ACTION_FLEE: u32 = 2u;
const ACTION_ATTACK: u32 = 3u;
const ACTION_CAST: u32 = 4u;
const ACTION_EAT: u32 = 7u;
const ACTION_DRINK: u32 = 8u;
const ACTION_REST: u32 = 9u;

const NO_TARGET: u32 = 0xFFFFFFFFu;

fn slot_of(id: u32) -> u32 {
    if (id == 0u) { return 0xFFFFFFFFu; }
    let s = id - 1u;
    if (s >= cfg.agent_cap) { return 0xFFFFFFFFu; }
    return s;
}

fn restore_need(current: f32, desired_delta: f32) -> vec2<f32> {
    // vec2<f32>(new_value, applied_delta).
    let new_val = min(current + desired_delta, 1.0);
    let applied = new_val - current;
    return vec2<f32>(new_val, applied);
}
"#,
    );

    out.push_str(&format!(
        r#"
@compute @workgroup_size({WORKGROUP_SIZE})
fn cs_apply_actions(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let slot = gid.x;
    if (slot >= cfg.agent_cap) {{ return; }}
    // Dead slots do nothing — matches the CPU `for id in state.agents_alive()`.
    if (agents[slot].alive == 0u) {{ return; }}

    let self_id = slot + 1u;
    let so = scoring[slot];
    let action = so.chosen_action;
    let tgt_slot_raw = so.chosen_target;  // already 0-based slot
    let tick = sim_cfg.tick;

    // --- Attack ---
    if (action == ACTION_ATTACK) {{
        if (tgt_slot_raw == NO_TARGET) {{ return; }}
        let t_slot = tgt_slot_raw;
        if (t_slot >= cfg.agent_cap) {{ return; }}
        if (agents[t_slot].alive == 0u) {{ return; }}
        let tgt_id = t_slot + 1u;

        // Range check: self.pos to target.pos.
        let sx = agents[slot].pos_x;
        let sy = agents[slot].pos_y;
        let sz = agents[slot].pos_z;
        let tx = agents[t_slot].pos_x;
        let ty = agents[t_slot].pos_y;
        let tz = agents[t_slot].pos_z;
        let dx = sx - tx;
        let dy = sy - ty;
        let dz = sz - tz;
        let dist = sqrt(dx * dx + dy * dy + dz * dz);
        // Use the world-scalar attack_range from SimCfg — per-agent
        // attack_range is not in `GpuAgentSlot` yet. The mask kernel
        // already radius-filters with per-agent range on the SoA
        // scoring buffer, so this kernel can trust the mask's gate;
        // the SimCfg value is a defence-in-depth guard. (Migrated from
        // the per-kernel `cfg.attack_range_default` uniform in Task 2.6
        // of the GPU sim-state refactor.)
        let range = sim_cfg.attack_range;
        if (dist > range) {{ return; }}

        let dmg = agents[slot].attack_damage;
        let old_hp = agents[t_slot].hp;
        let new_hp = max(old_hp - dmg, 0.0);
        agents[t_slot].hp = new_hp;
        // AgentAttacked: payload = [actor_raw_id, target_raw_id, damage_bits].
        let _atk_idx = gpu_emit_agent_attacked(self_id, tgt_id, dmg, tick);
        if (new_hp <= 0.0) {{
            agents[t_slot].alive = 0u;
            let _die_idx = gpu_emit_agent_died(tgt_id, tick);
        }}
        return;
    }}

    // --- Needs (Eat / Drink / Rest) ---
    //
    // The CPU path reads `state.agent_hunger/thirst/rest_timer` and
    // clamps via `restore_need`. Our agent SoA here doesn't carry
    // hunger/thirst/fatigue (those live on cold state in
    // `SimState.hot_*`). Skip emission until a future revision adds
    // them — the scorer rarely picks these heads in the combat
    // fixtures the perf bench exercises.
    //
    // Preserving the stub here so the kernel shape is correct once the
    // fields land.

    // --- Hold / Move / Flee ---
    // No state mutation, no events. Movement kernel handles pos updates.
}}
"#
    ));

    out
}

// ---------------------------------------------------------------------------
// Helpers exposed to the backend
// ---------------------------------------------------------------------------

/// Build an `ApplyActionsCfg` from a SimState. Default-attack-damage
/// comes off the config block; the kernel falls back to it when a
/// per-agent stat isn't plumbed. `tick` + `attack_range` moved to
/// `SimCfg` in Task 2.6 and are NOT set here — callers populate the
/// shared buffer via `SimCfg::from_state(state)`.
pub fn cfg_from_state(state: &SimState) -> ApplyActionsCfg {
    ApplyActionsCfg {
        agent_cap: state.agent_cap(),
        attack_damage_default: state.config.combat.attack_damage,
        eat_restore: state.config.needs.eat_restore,
        drink_restore: state.config.needs.drink_restore,
        rest_restore: state.config.needs.rest_restore,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

/// Unpack an `agent_slots_out` back onto `SimState`. Mirrors
/// `physics::unpack_agent_slots` but only touches the fields
/// `apply_actions` mutates (hp, alive). Used when the backend opts
/// into the GPU apply path.
pub fn unpack_apply_slots(state: &mut SimState, slots: &[GpuAgentSlot]) {
    for (slot_idx, s) in slots.iter().enumerate() {
        let id = match AgentId::new(slot_idx as u32 + 1) {
            Some(id) => id,
            None => continue,
        };
        let currently_alive = state.agent_alive(id);
        if !currently_alive && s.alive == 0 {
            continue;
        }
        state.set_agent_hp(id, s.hp);
        if currently_alive && s.alive == 0 {
            state.kill_agent(id);
        }
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
        assert_eq!(std::mem::size_of::<ApplyActionsCfg>(), 32);
    }

    #[test]
    fn action_apply_agent_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<ActionApplyAgent>(), 64);
    }

    #[test]
    fn shader_parses_through_naga() {
        let wgsl = build_shader(1024);
        if let Err(e) = naga::front::wgsl::parse_str(&wgsl) {
            panic!(
                "apply_actions shader failed naga parse:\n{e}\n--- WGSL source ---\n{wgsl}"
            );
        }
    }
}
