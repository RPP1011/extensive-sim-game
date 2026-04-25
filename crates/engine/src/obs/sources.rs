//! Built-in `FeatureSource` impls. Tasks 6-8 of Plan 3 fill these in.

use super::packer::FeatureSource;
use crate::ids::AgentId;
use crate::state::{MovementMode, SimState};

/// Vitals feature window: `[hp_frac, hunger, thirst, rest_timer]` (dim 4).
pub struct VitalsSource;

impl FeatureSource for VitalsSource {
    fn dim(&self) -> usize {
        4
    }
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]) {
        let hp = state.agent_hp(agent).unwrap_or(0.0);
        let max_hp = state.agent_max_hp(agent).unwrap_or(1.0).max(1e-6);
        out[0] = hp / max_hp;
        out[1] = state.agent_hunger(agent).unwrap_or(0.0);
        out[2] = state.agent_thirst(agent).unwrap_or(0.0);
        out[3] = state.agent_rest_timer(agent).unwrap_or(0.0);
    }
}

/// Position + movement-mode one-hot:
/// `[pos.x, pos.y, pos.z, is_walk, is_fly, is_swim, is_climb]` (dim 7).
///
/// `MovementMode::Fall` is not represented in the one-hot (it's a
/// transient transition state); fallers zero out the 4 mode slots.
pub struct PositionSource;

impl FeatureSource for PositionSource {
    fn dim(&self) -> usize {
        7
    }
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]) {
        let pos = state.agent_pos(agent).unwrap_or(glam::Vec3::ZERO);
        out[0] = pos.x;
        out[1] = pos.y;
        out[2] = pos.z;
        let mode = state.agent_movement_mode(agent).unwrap_or(MovementMode::Walk);
        out[3] = f32::from(u8::from(mode == MovementMode::Walk));
        out[4] = f32::from(u8::from(mode == MovementMode::Fly));
        out[5] = f32::from(u8::from(mode == MovementMode::Swim));
        out[6] = f32::from(u8::from(mode == MovementMode::Climb));
    }
}

/// Top-K nearest other agents, each contributing
/// `[rel_x, rel_y, rel_z, dist, hp_frac, present_flag]` (dim `6 * K`).
///
/// Fewer than K other alive agents: remaining slots are zero-filled;
/// `present_flag` is 0 for those slots.
///
/// Per-tick allocation: a temporary `Vec<(f32, AgentId)>` is built per call.
/// A `SimScratch`-backed zero-alloc variant is a later plan's concern.
pub struct NeighborSource<const K: usize>;

impl<const K: usize> FeatureSource for NeighborSource<K> {
    fn dim(&self) -> usize {
        K * 6
    }
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]) {
        out.fill(0.0);
        let Some(sp) = state.agent_pos(agent) else {
            return;
        };
        let mut neighbors: Vec<(f32, AgentId)> = state
            .agents_alive()
            .filter(|id| *id != agent)
            .filter_map(|id| state.agent_pos(id).map(|op| ((op - sp).length(), id)))
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(K);
        for (i, (dist, other)) in neighbors.iter().enumerate() {
            let Some(op) = state.agent_pos(*other) else {
                continue;
            };
            let rel = op - sp;
            let hp = state.agent_hp(*other).unwrap_or(0.0);
            let max_hp = state.agent_max_hp(*other).unwrap_or(1.0).max(1e-6);
            let base = i * 6;
            out[base] = rel.x;
            out[base + 1] = rel.y;
            out[base + 2] = rel.z;
            out[base + 3] = *dist;
            out[base + 4] = hp / max_hp;
            out[base + 5] = 1.0;
        }
    }
}
