//! WGSL emitter for `ApplyActionsKernel` — applies the ACTION_ATTACK
//! head selected by scoring this tick: range-checks the chosen target,
//! deals damage, mutates HP / alive state, and emits `AgentAttacked` /
//! `AgentDied` events into the cascade ring.
//!
//! Recovered from the pre-T16 hand-written apply_actions WGSL
//! (`crates/engine_gpu/src/apply_actions.rs:660` at commit `4474566c~1`)
//! and adapted to the post-T16 raw-u32 binding layout. The kernel
//! `include_str!`s `emit_runtime_prelude_wgsl` for event-emission
//! helpers.
//!
//! ## Action coverage
//!
//! Only `ACTION_ATTACK` causes state mutation here. Hold / Move / Flee
//! either do nothing or are handled by the movement kernel. Eat / Drink
//! / Rest are stubbed pending hunger/thirst/fatigue fields landing on
//! `GpuAgentSlot` (matches the pre-T16 stub).
//!
//! ## Field offsets
//!
//! `agents` is `GpuAgentSlot` SoA (16 u32 stride):
//! - hp: 0
//! - attack_damage: 3
//! - alive: 4
//! - pos_x: 11, pos_y: 12, pos_z: 13
//!
//! `scoring` (named `scoring_out` in the new BGL) is `ScoreOutput` SoA
//! (4 u32 stride):
//! - chosen_action: 0
//! - chosen_target: 1
//!
//! `sim_cfg` is `SimCfg` SoA (in u32 indices):
//! - tick: 0
//! - attack_range: 6

use crate::emit_runtime_prelude_wgsl::{
    emit_runtime_prelude_consts, emit_runtime_prelude_wgsl,
};

/// Default event ring capacity. Mirrors
/// `engine_gpu::event_ring::DEFAULT_CAPACITY`.
const DEFAULT_EVENT_RING_CAPACITY: u32 = 65536;

/// Emit the body of `engine_gpu_rules/src/apply_actions.wgsl`.
pub fn emit_apply_actions_wgsl() -> String {
    let mut out = String::new();
    out.push_str(&emit_runtime_prelude_consts(DEFAULT_EVENT_RING_CAPACITY));
    out.push_str(
        "\n\
@group(0) @binding(0) var<storage, read_write> agents:             array<u32>;\n\
@group(0) @binding(1) var<storage, read>       scoring_out:        array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> event_ring_records: array<u32>;\n\
@group(0) @binding(3) var<storage, read_write> event_ring_tail:    atomic<u32>;\n\
@group(0) @binding(4) var<storage, read>       sim_cfg:            array<u32>;\n\
struct ApplyActionsCfg { agent_cap: u32, tick: u32, event_ring_capacity: u32, _pad: u32 };\n\
@group(0) @binding(5) var<uniform>             cfg:                ApplyActionsCfg;\n\
\n\
// GpuAgentSlot stride (16 u32 = 64 bytes per agent).\n\
const SLOT_STRIDE_U32: u32 = 16u;\n\
const SLOT_OFFSET_HP: u32 = 0u;\n\
const SLOT_OFFSET_ATTACK_DAMAGE: u32 = 3u;\n\
const SLOT_OFFSET_ALIVE: u32 = 4u;\n\
const SLOT_OFFSET_POS_X: u32 = 11u;\n\
const SLOT_OFFSET_POS_Y: u32 = 12u;\n\
const SLOT_OFFSET_POS_Z: u32 = 13u;\n\
\n\
// ScoreOutput stride (4 u32 = 16 bytes per agent).\n\
const SCORING_STRIDE_U32: u32 = 4u;\n\
const SCORING_OFFSET_ACTION: u32 = 0u;\n\
const SCORING_OFFSET_TARGET: u32 = 1u;\n\
\n\
// SimCfg field offsets (in u32 indices).\n\
const SIM_CFG_OFFSET_TICK: u32 = 0u;\n\
const SIM_CFG_OFFSET_ATTACK_RANGE: u32 = 6u;\n\
\n\
// Action ids (mirror MicroKind ordinals).\n\
const ACTION_HOLD: u32 = 0u;\n\
const ACTION_MOVE_TOWARD: u32 = 1u;\n\
const ACTION_FLEE: u32 = 2u;\n\
const ACTION_ATTACK: u32 = 3u;\n\
const ACTION_CAST: u32 = 4u;\n\
const NO_TARGET: u32 = 0xFFFFFFFFu;\n\
\n\
fn agent_alive(slot: u32) -> u32 {\n\
    return agents[slot * SLOT_STRIDE_U32 + SLOT_OFFSET_ALIVE];\n\
}\n\
\n\
fn agent_hp(slot: u32) -> f32 {\n\
    return bitcast<f32>(agents[slot * SLOT_STRIDE_U32 + SLOT_OFFSET_HP]);\n\
}\n\
\n\
fn set_agent_hp(slot: u32, hp: f32) {\n\
    agents[slot * SLOT_STRIDE_U32 + SLOT_OFFSET_HP] = bitcast<u32>(hp);\n\
}\n\
\n\
fn set_agent_dead(slot: u32) {\n\
    agents[slot * SLOT_STRIDE_U32 + SLOT_OFFSET_ALIVE] = 0u;\n\
}\n\
\n\
fn agent_attack_damage(slot: u32) -> f32 {\n\
    return bitcast<f32>(agents[slot * SLOT_STRIDE_U32 + SLOT_OFFSET_ATTACK_DAMAGE]);\n\
}\n\
\n\
fn agent_pos(slot: u32) -> vec3<f32> {\n\
    let base = slot * SLOT_STRIDE_U32;\n\
    return vec3<f32>(\n\
        bitcast<f32>(agents[base + SLOT_OFFSET_POS_X]),\n\
        bitcast<f32>(agents[base + SLOT_OFFSET_POS_Y]),\n\
        bitcast<f32>(agents[base + SLOT_OFFSET_POS_Z]),\n\
    );\n\
}\n\
\n\
fn scoring_action(slot: u32) -> u32 {\n\
    return scoring_out[slot * SCORING_STRIDE_U32 + SCORING_OFFSET_ACTION];\n\
}\n\
\n\
fn scoring_target(slot: u32) -> u32 {\n\
    return scoring_out[slot * SCORING_STRIDE_U32 + SCORING_OFFSET_TARGET];\n\
}\n\
\n\
fn sim_cfg_tick() -> u32 {\n\
    return sim_cfg[SIM_CFG_OFFSET_TICK];\n\
}\n\
\n\
fn sim_cfg_attack_range() -> f32 {\n\
    return bitcast<f32>(sim_cfg[SIM_CFG_OFFSET_ATTACK_RANGE]);\n\
}\n\
\n",
    );
    out.push_str(&emit_runtime_prelude_wgsl());
    out.push_str(
        "\n\
@compute @workgroup_size(64)\n\
fn cs_apply_actions(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let slot = gid.x;\n\
    if (slot >= cfg.agent_cap) { return; }\n\
    if (agent_alive(slot) == 0u) { return; }\n\
\n\
    let self_id = slot + 1u;\n\
    let action = scoring_action(slot);\n\
    let tgt_slot_raw = scoring_target(slot);\n\
    let tick = sim_cfg_tick();\n\
\n\
    // ATTACK — only action that mutates state in this kernel.\n\
    if (action == ACTION_ATTACK) {\n\
        if (tgt_slot_raw == NO_TARGET) { return; }\n\
        let t_slot = tgt_slot_raw;\n\
        if (t_slot >= cfg.agent_cap) { return; }\n\
        if (agent_alive(t_slot) == 0u) { return; }\n\
        let tgt_id = t_slot + 1u;\n\
\n\
        // Range check (defence in depth — mask kernel already gates).\n\
        let self_pos = agent_pos(slot);\n\
        let target_pos = agent_pos(t_slot);\n\
        let delta = self_pos - target_pos;\n\
        let dist = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);\n\
        let range = sim_cfg_attack_range();\n\
        if (dist > range) { return; }\n\
\n\
        let dmg = agent_attack_damage(slot);\n\
        let old_hp = agent_hp(t_slot);\n\
        let new_hp = max(old_hp - dmg, 0.0);\n\
        set_agent_hp(t_slot, new_hp);\n\
        let _atk_idx = gpu_emit_agent_attacked(self_id, tgt_id, dmg, tick);\n\
        if (new_hp <= 0.0) {\n\
            set_agent_dead(t_slot);\n\
            let _die_idx = gpu_emit_agent_died(tgt_id, tick);\n\
        }\n\
        return;\n\
    }\n\
\n\
    // HOLD / MOVE / FLEE / CAST / Eat / Drink / Rest:\n\
    // - HOLD/MOVE/FLEE: no state mutation; movement kernel owns pos.\n\
    // - CAST: handled by physics fixed-point via cascade events.\n\
    // - Eat/Drink/Rest: pending hunger/thirst/fatigue fields on\n\
    //   GpuAgentSlot. Skip emission for now (pre-T16 also stubbed).\n\
}\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_actions_naga_parses() {
        let src = emit_apply_actions_wgsl();
        naga::front::wgsl::parse_str(&src)
            .map_err(|e| {
                format!("--- WGSL ---\n{src}\n--- naga error ---\n{}", e.emit_to_string(&src))
            })
            .expect("emit_apply_actions_wgsl should parse cleanly");
    }

    #[test]
    fn apply_actions_uses_prelude_helpers() {
        let src = emit_apply_actions_wgsl();
        assert!(src.contains("gpu_emit_agent_attacked"));
        assert!(src.contains("gpu_emit_agent_died"));
    }
}
