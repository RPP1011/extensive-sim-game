//! Observation extraction for BuildingEnv.
//!
//! Extracts per-cell grid channels + scalar context from VoxelWorld into Vec<f32>.

use crate::world_sim::state::{EntityKind, WorldState};
use crate::world_sim::voxel::world_to_voxel;
use super::env_config::GRID_SIZE;

/// Number of per-cell channels in the grid observation.
pub const GRID_CHANNELS: usize = 8;

/// Number of scalar context features appended after the grid.
pub const SCALAR_CONTEXT_DIM: usize = 12;

/// Total observation dimension.
pub const OBS_DIM: usize = GRID_SIZE * GRID_SIZE * GRID_CHANNELS + SCALAR_CONTEXT_DIM;

/// Extract observation from world state as a flat Vec<f32>.
///
/// Grid channels (per cell, 128x128 centered on first settlement):
///   0: surface height (normalized, 0-1)
///   1: building presence (0 or 1)
///   2: material solidity (0 = air, 1 = solid at surface)
///   3: zone type (normalized, 0-1)
///   4: building type (normalized index / 25)
///   5: structural integrity (0-1)
///   6: friendly NPC density (0-1, clamped count / 5)
///   7: enemy density (0-1, clamped count / 5)
///
/// Scalar context:
///   0: settlement level (0-1)
///   1-4: resource stockpiles (0-1)
///   5: population / housing capacity (0-1)
///   6: tick / tick_budget (0-1)
///   7: seasonal phase (0-1)
///   8: challenge severity (0-1)
///   9-10: challenge direction x, y (-1 to 1)
///   11: alive monster count (0-1, / 50)
pub fn extract_observation(
    state: &WorldState,
    tick_budget: u64,
    challenge_severity: f32,
    challenge_direction: Option<(f32, f32)>,
) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_DIM];

    // Settlement center in voxel coords
    let settlement_pos = state.settlements.first().map(|s| s.pos).unwrap_or((0.0, 0.0));
    let (center_vx, center_vy, _) = world_to_voxel(settlement_pos.0, settlement_pos.1, 0.0);
    let half = (GRID_SIZE / 2) as i32;
    let origin_vx = center_vx - half;
    let origin_vy = center_vy - half;

    // Grid channels from VoxelWorld
    for r in 0..GRID_SIZE {
        for c in 0..GRID_SIZE {
            let vx = origin_vx + c as i32;
            let vy = origin_vy + r as i32;
            let base = (r * GRID_SIZE + c) * GRID_CHANNELS;

            // Channel 0: surface height (normalize to 0-1 assuming max ~64)
            let surface_z = state.voxel_world.surface_height(vx, vy);
            obs[base] = (surface_z as f32 / 64.0).min(1.0);

            // Get the surface voxel
            let surface_voxel = state.voxel_world.get_voxel(vx, vy, surface_z.saturating_sub(1).max(0));

            // Channel 1: building presence
            obs[base + 1] = if surface_voxel.building_id.is_some() { 1.0 } else { 0.0 };

            // Channel 2: material solidity
            obs[base + 2] = if surface_voxel.material.is_solid() { 1.0 } else { 0.0 };

            // Channel 3: zone type (normalized)
            obs[base + 3] = surface_voxel.zone as u8 as f32 / 7.0;

            // Channel 4: building type (from entity lookup)
            if let Some(bid) = surface_voxel.building_id {
                if let Some(entity) = state.entities.iter().find(|e| e.id == bid) {
                    if let Some(bd) = &entity.building {
                        obs[base + 4] = bd.building_type as u8 as f32 / 25.0;
                    }
                }
            }

            // Channel 5: structural integrity
            obs[base + 5] = surface_voxel.integrity;
        }
    }

    // Channels 6-7: NPC/monster density
    for entity in &state.entities {
        if !entity.alive { continue; }
        let (evx, evy, _) = world_to_voxel(entity.pos.0, entity.pos.1, 0.0);
        let gc = (evx - origin_vx) as usize;
        let gr = (evy - origin_vy) as usize;
        if gc >= GRID_SIZE || gr >= GRID_SIZE { continue; }
        let base = (gr * GRID_SIZE + gc) * GRID_CHANNELS;
        match entity.kind {
            EntityKind::Npc => obs[base + 6] = (obs[base + 6] + 0.2).min(1.0),
            EntityKind::Monster => obs[base + 7] = (obs[base + 7] + 0.2).min(1.0),
            _ => {}
        }
    }

    // Scalar context
    let sc = GRID_SIZE * GRID_SIZE * GRID_CHANNELS;

    obs[sc] = state.settlements.first()
        .map(|s| (s.infrastructure_level / 5.0).min(1.0))
        .unwrap_or(0.0);

    if let Some(s) = state.settlements.first() {
        for i in 0..4.min(s.stockpile.len()) {
            obs[sc + 1 + i] = (s.stockpile[i] / 500.0).min(1.0);
        }
    }

    let alive_npcs = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Npc && e.alive).count() as f32;
    let housing_cap: f32 = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .filter_map(|e| e.building.as_ref())
        .map(|b| b.residential_capacity as f32).sum();
    obs[sc + 5] = if housing_cap > 0.0 { (alive_npcs / housing_cap).min(1.0) } else { 0.0 };

    obs[sc + 6] = if tick_budget > 0 { state.tick as f32 / tick_budget as f32 } else { 0.0 };
    obs[sc + 7] = (state.tick % 4800) as f32 / 4800.0;
    obs[sc + 8] = challenge_severity;

    let (dx, dy) = challenge_direction.unwrap_or((0.0, 0.0));
    obs[sc + 9] = dx;
    obs[sc + 10] = dy;

    let alive_monsters = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Monster && e.alive).count() as f32;
    obs[sc + 11] = (alive_monsters / 50.0).min(1.0);

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_has_correct_length() {
        let state = WorldState::new(42);
        let obs = extract_observation(&state, 5000, 0.5, Some((0.0, -1.0)));
        assert_eq!(obs.len(), OBS_DIM);
    }

    #[test]
    fn obs_values_in_range() {
        let state = WorldState::new(42);
        let obs = extract_observation(&state, 5000, 0.5, Some((0.0, -1.0)));
        for (i, &v) in obs.iter().enumerate() {
            assert!(v >= -1.0 && v <= 1.0, "obs[{}] = {} out of range", i, v);
        }
    }
}
