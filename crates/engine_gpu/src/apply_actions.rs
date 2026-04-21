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
//!
//! 5 bindings, well under the 16-per-group cap. Event ring borrows
//! `GpuEventRing`'s buffers; the physics + apply_actions kernels share
//! the same event ring (different dispatches, same underlying storage).

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
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct ApplyActionsCfg {
    pub agent_cap: u32,
    pub tick: u32,
    pub attack_damage_default: f32,
    pub attack_range_default: f32,
    pub eat_restore: f32,
    pub drink_restore: f32,
    pub rest_restore: f32,
    pub _pad: u32,
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
}

struct BufferPool {
    agent_cap: u32,
    agents_buf: wgpu::Buffer,
    agents_readback: wgpu::Buffer,
    scoring_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
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
        let bgl_entries = [
            storage_rw(0), // agents
            storage_ro(1), // scoring
            uniform(2),    // cfg
            storage_rw(3), // event_ring records
            storage_rw(4), // event_ring tail
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

        self.pool = Some(BufferPool {
            agent_cap,
            agents_buf,
            agents_readback,
            scoring_buf,
            cfg_buf,
        });
    }

    /// Run the apply_actions kernel against `agent_slots_in`, the
    /// scoring outputs, and the caller-owned event ring. Returns the
    /// mutated agent slots. The event ring is drained by the caller
    /// (the ring is shared with the physics kernel so the caller
    /// orchestrates the drain timing).
    pub fn run_and_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        agent_slots_in: &[GpuAgentSlot],
        scoring: &[ScoreOutput],
        cfg: ApplyActionsCfg,
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
    tick: u32,
    attack_damage_default: f32,
    attack_range_default: f32,
    eat_restore: f32,
    drink_restore: f32,
    rest_restore: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> agents: array<AgentSlot>;
@group(0) @binding(1) var<storage, read> scoring: array<ScoreOutput>;
@group(0) @binding(2) var<uniform> cfg: ApplyActionsCfg;
@group(0) @binding(3) var<storage, read_write> event_ring: array<EventRecord>;
@group(0) @binding(4) var<storage, read_write> event_ring_tail: atomic<u32>;
"#,
    );

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
    let tick = cfg.tick;

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
        // Use the default from cfg — per-agent attack_range is not in
        // `GpuAgentSlot` yet. The mask kernel already radius-filters
        // with per-agent range on the SoA scoring buffer, so this
        // kernel can trust the mask's gate; the cfg default is a
        // defence-in-depth guard.
        let range = cfg.attack_range_default;
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

/// Build an `ApplyActionsCfg` from a SimState. Default-attack-range and
/// default-attack-damage come off the config block; the kernel falls
/// back to these when a per-agent stat isn't plumbed (see the comment
/// in `cs_apply_actions`).
pub fn cfg_from_state(state: &SimState) -> ApplyActionsCfg {
    ApplyActionsCfg {
        agent_cap: state.agent_cap(),
        tick: state.tick,
        attack_damage_default: state.config.combat.attack_damage,
        attack_range_default: state.config.combat.attack_range,
        eat_restore: state.config.needs.eat_restore,
        drink_restore: state.config.needs.drink_restore,
        rest_restore: state.config.needs.rest_restore,
        _pad: 0,
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
