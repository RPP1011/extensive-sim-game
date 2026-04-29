//! Pod helpers for the GPU backend's sync surface.
//!
//! Real Pod types and pack/unpack utilities that survived the T16
//! hand-written-kernel deletion (commit `4474566c`). Lives here, not in a
//! kernel module, so callers that just need a `GpuAgentSlot` (the
//! snapshot decoder, the resident-init pack uploader) don't pull in any
//! deleted-kernel state.
//!
//! # What's here
//! * [`GpuAgentSlot`] — `#[repr(C)]` POD shape the resident agent buffer
//!   stores one-per-slot. Field order is the WGSL `AgentSlot` layout.
//! * [`pack_agent_slots`] / [`unpack_agent_slots`] — bidirectional
//!   conversion between `SimState`'s SoA fields and the packed slot
//!   array.
//! * [`alive_bitmap_words`] / [`alive_bitmap_bytes`] /
//!   [`create_alive_bitmap_buffer`] — sizing + allocation helpers for
//!   the per-tick alive-bitmap. The pack kernel itself lives in
//!   `engine_gpu_rules::alive_pack` post-T16; this module only exposes
//!   the buffer-allocation surface the resident init path needs.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};
use engine::ids::AgentId;
use engine::state::SimState;

/// Packed per-slot agent state the GPU agent buffer stores. One struct
/// per agent slot; dead slots have `alive = 0`.
///
/// Field order pins the WGSL `AgentSlot` layout in
/// `engine_gpu_rules`'s emitted shaders — any reorder/insert forces a
/// lockstep update there.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuAgentSlot {
    pub hp: f32,
    pub max_hp: f32,
    pub shield_hp: f32,
    pub attack_damage: f32,
    pub alive: u32,
    pub creature_type: u32,
    /// 1-based raw `AgentId` of partner, or `0xFFFF_FFFF` for "none".
    pub engaged_with: u32,
    pub stun_expires_at: u32,
    pub slow_expires_at: u32,
    /// `i16` promoted to `u32` via sign-preserving `as u16 as u32`.
    pub slow_factor_q8: u32,
    pub cooldown_next_ready: u32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl GpuAgentSlot {
    pub const ENGAGED_NONE: u32 = 0xFFFF_FFFFu32;
}

/// Pack a `SimState`'s agent SoA into the resident buffer's wire layout.
///
/// Returns one `GpuAgentSlot` per `state.agent_cap()` slot in slot-id
/// order; dead slots emit a zeroed slot with `alive = 0` and
/// `engaged_with = ENGAGED_NONE` so the kernel's "no partner" branch
/// fires correctly.
pub fn pack_agent_slots(state: &SimState) -> Vec<GpuAgentSlot> {
    let cap = state.agent_cap() as usize;
    let mut out = Vec::with_capacity(cap);
    for slot in 0..cap {
        let id = match AgentId::new(slot as u32 + 1) {
            Some(id) => id,
            None => {
                out.push(GpuAgentSlot::zeroed());
                continue;
            }
        };
        let alive = state.agent_alive(id);
        if !alive {
            let mut z = GpuAgentSlot::zeroed();
            z.alive = 0;
            z.engaged_with = GpuAgentSlot::ENGAGED_NONE;
            out.push(z);
            continue;
        }
        let pos = state.agent_pos(id).unwrap_or(glam::Vec3::ZERO);
        let engaged = state
            .agent_engaged_with(id)
            .map(|p| p.raw())
            .unwrap_or(GpuAgentSlot::ENGAGED_NONE);
        let slow_factor = state.agent_slow_factor_q8(id).unwrap_or(0);
        // `slow_factor_q8` is an `i16`; encode as `u16` bits then widen
        // so the round-trip preserves negative values.
        let slow_factor_u32 = (slow_factor as u16) as u32;
        out.push(GpuAgentSlot {
            hp: state.agent_hp(id).unwrap_or(0.0),
            max_hp: state.agent_max_hp(id).unwrap_or(0.0),
            shield_hp: state.agent_shield_hp(id).unwrap_or(0.0),
            attack_damage: state.agent_attack_damage(id).unwrap_or(0.0),
            alive: 1,
            creature_type: state
                .agent_creature_type(id)
                .map(|c| c as u32)
                .unwrap_or(u32::MAX),
            engaged_with: engaged,
            stun_expires_at: state.agent_stun_expires_at(id).unwrap_or(0),
            slow_expires_at: state.agent_slow_expires_at(id).unwrap_or(0),
            slow_factor_q8: slow_factor_u32,
            cooldown_next_ready: state.agent_cooldown_next_ready(id).unwrap_or(0),
            pos_x: pos.x,
            pos_y: pos.y,
            pos_z: pos.z,
            _pad0: 0,
            _pad1: 0,
        });
    }
    out
}

/// Apply a readback of `Vec<GpuAgentSlot>` to `SimState` — used by the
/// snapshot path to materialise the GPU's mutated state back onto the
/// authoritative `SimState`.
///
/// Field writes always fire (even for newly-killed slots) so the CPU
/// `hot_*` arrays match the GPU's post-kill snapshot. The alive-flag
/// transition is deferred to a `kill_agent` call AFTER the field
/// writes so the kill-teardown (pool drop, spatial remove) runs on
/// the correct field state.
pub fn unpack_agent_slots(state: &mut SimState, slots: &[GpuAgentSlot]) {
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
        state.set_agent_shield_hp(id, s.shield_hp);
        state.set_agent_stun_expires_at(id, s.stun_expires_at);
        state.set_agent_slow_expires_at(id, s.slow_expires_at);
        let factor_i16 = (s.slow_factor_q8 & 0xFFFF) as u16 as i16;
        state.set_agent_slow_factor_q8(id, factor_i16);
        state.set_agent_cooldown_next_ready(id, s.cooldown_next_ready);
        let engaged = if s.engaged_with == GpuAgentSlot::ENGAGED_NONE {
            None
        } else {
            AgentId::new(s.engaged_with)
        };
        state.set_agent_engaged_with(id, engaged);
        if currently_alive && s.alive == 0 {
            state.kill_agent(id);
        }
    }
}

/// Number of `u32` words the alive bitmap needs for `agent_cap` slots:
/// one bit per slot, 32 slots per word.
#[inline]
pub fn alive_bitmap_words(agent_cap: u32) -> u32 {
    (agent_cap + 31) / 32
}

/// Byte size of the bitmap storage buffer for a given `agent_cap`.
/// Clamped to at least 4 B so wgpu never sees a zero-sized buffer
/// (`agent_cap == 0` is rare but legal in tests).
#[inline]
pub fn alive_bitmap_bytes(agent_cap: u32) -> u64 {
    (alive_bitmap_words(agent_cap) as u64 * 4).max(4)
}

/// Create an empty bitmap storage buffer sized for `agent_cap` slots.
/// Zero-initialised by wgpu — safe as a pre-pack default since the
/// emitted `AlivePackKernel` overwrites every word at the top of every
/// tick.
pub fn create_alive_bitmap_buffer(device: &wgpu::Device, agent_cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::alive_bitmap"),
        size: alive_bitmap_bytes(agent_cap),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Read the GPU agents buffer back into a `Vec<GpuAgentSlot>`.
/// Used by `step_batch` (Phase 0 of physics-wgsl-runtime) to bring
/// GPU-mutated agent state back to the CPU mirror after a batch.
///
/// Caller must have already issued `device.poll(Wait)` after the
/// batch's `queue.submit()`; this function is sync-only.
#[cfg(feature = "gpu")]
pub fn readback_agents_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    agents_buf: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<GpuAgentSlot> {
    let bytes = (agent_cap as usize) * std::mem::size_of::<GpuAgentSlot>();
    let raw = crate::gpu_util::readback::readback_typed::<u8>(device, queue, agents_buf, bytes)
        .expect("readback_agents_buf: GPU readback failed");
    bytemuck::cast_slice::<u8, GpuAgentSlot>(&raw).to_vec()
}
