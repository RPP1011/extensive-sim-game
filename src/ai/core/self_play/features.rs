//! Feature extraction: encode_unit, extract_features.

use super::{
    UNIT_FEATURES, MAX_ABILITIES, ABILITY_FEATURES_LEGACY, NUM_ENEMIES, NUM_ALLIES,
    GLOBAL_FEATURES, TERRAIN_RAYS, TERRAIN_RAY_MAX, FEATURE_DIM,
};
use crate::ai::core::{distance, is_alive, SimState, SimVec2};
use crate::ai::pathing::raycast_distances;

/// Encode a single unit's raw state into a fixed-size slice.
fn encode_unit(u: &crate::ai::core::UnitState, ref_pos: SimVec2) -> [f32; UNIT_FEATURES] {
    let mut f = [0.0f32; UNIT_FEATURES];
    f[0] = u.hp as f32;
    f[1] = u.max_hp as f32;
    f[2] = u.hp as f32 / u.max_hp.max(1) as f32;
    f[3] = u.position.x - ref_pos.x;  // relative position
    f[4] = u.position.y - ref_pos.y;
    f[5] = distance(u.position, ref_pos);
    f[6] = u.move_speed_per_sec;
    f[7] = u.attack_damage as f32;
    f[8] = u.attack_range;
    f[9] = u.attack_cooldown_ms as f32;
    f[10] = u.cooldown_remaining_ms as f32;
    f[11] = u.ability_damage as f32;
    f[12] = u.ability_range;
    f[13] = u.ability_cooldown_remaining_ms as f32;
    f[14] = u.heal_amount as f32;
    f[15] = u.heal_range;
    f[16] = u.heal_cooldown_remaining_ms as f32;
    f[17] = u.control_range;
    f[18] = u.control_duration_ms as f32;
    f[19] = u.control_cooldown_remaining_ms as f32;
    f[20] = u.control_remaining_ms as f32;
    f[21] = u.shield_hp as f32;
    f[22] = u.resource as f32;
    f[23] = u.max_resource as f32;
    f[24] = u.armor;
    f[25] = u.magic_resist;
    f[26] = u.total_healing_done as f32;
    f[27] = u.total_damage_done as f32;
    f[28] = if u.casting.is_some() { 1.0 } else { 0.0 };
    f[29] = u.status_effects.len() as f32;
    f
}

/// Encode an ability slot (legacy — 4 features, no semantic info).
fn encode_ability_legacy(slot: &crate::ai::effects::AbilitySlot, unit: &crate::ai::core::UnitState) -> [f32; ABILITY_FEATURES_LEGACY] {
    let ready = slot.cooldown_remaining_ms == 0
        && (slot.def.resource_cost <= 0 || unit.resource >= slot.def.resource_cost);
    [
        if ready { 1.0 } else { 0.0 },
        slot.def.range,
        slot.def.cooldown_ms as f32,
        slot.cooldown_remaining_ms as f32,
    ]
}

/// Extract the full raw feature vector for a unit.
pub fn extract_features(state: &SimState, unit_id: u32) -> [f32; FEATURE_DIM] {
    let mut f = [0.0f32; FEATURE_DIM];
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return f,
    };
    let ref_pos = unit.position;

    // Self
    let self_enc = encode_unit(unit, ref_pos);
    f[..UNIT_FEATURES].copy_from_slice(&self_enc);
    let mut offset = UNIT_FEATURES;

    // Sort enemies by distance
    let mut enemies: Vec<&crate::ai::core::UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    enemies.sort_by(|a, b| {
        distance(ref_pos, a.position)
            .partial_cmp(&distance(ref_pos, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in 0..NUM_ENEMIES {
        if let Some(e) = enemies.get(i) {
            let enc = encode_unit(e, ref_pos);
            f[offset..offset + UNIT_FEATURES].copy_from_slice(&enc);
        }
        offset += UNIT_FEATURES;
    }

    // Sort allies by distance (exclude self)
    let mut allies: Vec<&crate::ai::core::UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit_id)
        .collect();
    allies.sort_by(|a, b| {
        distance(ref_pos, a.position)
            .partial_cmp(&distance(ref_pos, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in 0..NUM_ALLIES {
        if let Some(a) = allies.get(i) {
            let enc = encode_unit(a, ref_pos);
            f[offset..offset + UNIT_FEATURES].copy_from_slice(&enc);
        }
        offset += UNIT_FEATURES;
    }

    // Ability slots (legacy)
    for i in 0..MAX_ABILITIES {
        if let Some(slot) = unit.abilities.get(i) {
            let enc = encode_ability_legacy(slot, unit);
            f[offset..offset + ABILITY_FEATURES_LEGACY].copy_from_slice(&enc);
        }
        offset += ABILITY_FEATURES_LEGACY;
    }

    // Global
    f[offset] = state.tick as f32;
    f[offset + 1] = enemies.len() as f32;
    f[offset + 2] = allies.len() as f32 + 1.0; // +1 for self
    f[offset + 3] = (allies.len() as f32 + 1.0) - enemies.len() as f32;
    f[offset + 4] = state.zones.len() as f32;
    offset += GLOBAL_FEATURES;

    // Terrain raycasts (64 directions from self)
    if let Some(ref nav) = state.grid_nav {
        let rays = raycast_distances(nav, ref_pos, TERRAIN_RAYS, TERRAIN_RAY_MAX);
        for (i, &d) in rays.iter().enumerate() {
            f[offset + i] = d;
        }
    } else {
        // No terrain: all rays at max distance
        for i in 0..TERRAIN_RAYS {
            f[offset + i] = TERRAIN_RAY_MAX;
        }
    }

    f
}
