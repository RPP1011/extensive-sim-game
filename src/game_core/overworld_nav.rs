use bevy::prelude::*;

use super::overworld_types::*;
use super::types::*;

pub fn overworld_cycle_selection(overworld: &mut OverworldMap, forward: bool) -> bool {
    let current = overworld
        .current_region
        .min(overworld.regions.len().saturating_sub(1));
    let Some(region) = overworld.regions.get(current) else {
        return false;
    };
    if region.neighbors.is_empty() {
        return false;
    }

    if !region.neighbors.contains(&overworld.selected_region) {
        overworld.selected_region = region.neighbors[0];
        return true;
    }

    let pos = region
        .neighbors
        .iter()
        .position(|id| *id == overworld.selected_region)
        .unwrap_or(0);
    let next = if forward {
        (pos + 1) % region.neighbors.len()
    } else if pos == 0 {
        region.neighbors.len() - 1
    } else {
        pos - 1
    };
    overworld.selected_region = region.neighbors[next];
    true
}

/// Returns `Some(slot_index)` if travel succeeds and the destination region has a mission slot.
pub fn try_overworld_travel(
    overworld: &mut OverworldMap,
    attention: &mut AttentionState,
) -> Option<usize> {
    if overworld.travel_cooldown_turns > 0 || attention.global_energy < overworld.travel_cost {
        return None;
    }
    let current = overworld
        .current_region
        .min(overworld.regions.len().saturating_sub(1));
    let target = overworld
        .selected_region
        .min(overworld.regions.len().saturating_sub(1));
    if current == target {
        return None;
    }
    let Some(region) = overworld.regions.get(current) else {
        return None;
    };
    if !region.neighbors.contains(&target) {
        return None;
    }

    overworld.current_region = target;
    attention.global_energy = (attention.global_energy - overworld.travel_cost).max(0.0);
    overworld.travel_cooldown_turns = overworld.travel_cooldown_max;
    let neighbors = overworld.regions[target].neighbors.clone();
    if let Some(region) = overworld.regions.get_mut(target) {
        region.intel_level = (region.intel_level + 24.0).clamp(0.0, 100.0);
    }
    for neighbor in neighbors {
        if let Some(region) = overworld.regions.get_mut(neighbor) {
            region.intel_level = (region.intel_level + 10.0).clamp(0.0, 100.0);
        }
    }
    overworld.regions[target].mission_slot
}

pub fn overworld_hub_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut commands: Commands,
    mut overworld: ResMut<OverworldMap>,
    board: Res<MissionBoard>,
    mut attention: ResMut<AttentionState>,
    active_query: Query<Entity, With<ActiveMission>>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };
    if keyboard.just_pressed(KeyCode::KeyJ) {
        let _ = overworld_cycle_selection(&mut overworld, false);
    }
    if keyboard.just_pressed(KeyCode::KeyL) {
        let _ = overworld_cycle_selection(&mut overworld, true);
    }
    if keyboard.just_pressed(KeyCode::KeyT) {
        if let Some(slot) = try_overworld_travel(&mut overworld, &mut attention) {
            if let Some(&new_entity) = board
                .entities
                .get(slot.min(board.entities.len().saturating_sub(1)))
            {
                for old in active_query.iter() {
                    commands.entity(old).remove::<ActiveMission>();
                }
                commands.entity(new_entity).insert(ActiveMission);
            }
        }
    }
}
