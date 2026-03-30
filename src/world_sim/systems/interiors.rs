//! Building interiors — NPCs enter buildings and occupy specific rooms.
//!
//! When an NPC arrives at a building (work, home, inn), they "enter" it:
//! - `inside_building_id` is set to the building entity ID
//! - `current_room` is set to the appropriate room index
//! - The building's room occupant_id is set to the NPC
//! - NPC's local_pos is set to the room's offset (for visualization)
//!
//! When the NPC leaves (goal changes, walking away), they "exit":
//! - Fields cleared, room vacated
//!
//! Room assignment based on action:
//! - Working → Workshop or building-specific (Farm→Entrance, Forge→Workshop)
//! - Eating → Kitchen or Counter
//! - Resting → Bedroom
//! - Socializing → Hearth
//! - Trading → Counter
//!
//! Cadence: every 5 ticks (post-apply).

use crate::world_sim::state::*;

const INTERIOR_INTERVAL: u64 = 5;
const ENTER_DIST_SQ: f32 = 9.0; // 3 units — close enough to enter

/// Manage NPC building entry/exit. Called post-apply from runtime.rs.
pub fn advance_interiors(state: &mut WorldState) {
    if state.tick % INTERIOR_INTERVAL != 0 { return; }

    // Phase 1: Clear stale occupancies (NPC died, moved away, etc.).
    // Collect (building_entity_idx, room_idx) to clear.
    let mut clears: Vec<(usize, u8)> = Vec::new();
    for (bi, entity) in state.entities.iter().enumerate() {
        if entity.kind != EntityKind::Building { continue; }
        let bld = match &entity.building { Some(b) => b, None => continue };
        for (ri, room) in bld.rooms.iter().enumerate() {
            if let Some(occ_id) = room.occupant_id {
                // Check if occupant is still alive and nearby.
                let still_valid = state.entities.iter().any(|e| {
                    e.id == occ_id && e.alive
                        && e.npc.as_ref().map(|n| n.inside_building_id == Some(entity.id)).unwrap_or(false)
                });
                if !still_valid {
                    clears.push((bi, ri as u8));
                }
            }
        }
    }
    for (bi, ri) in clears {
        if let Some(bld) = state.entities[bi].building.as_mut() {
            if (ri as usize) < bld.rooms.len() {
                bld.rooms[ri as usize].occupant_id = None;
            }
        }
    }

    // Phase 2: NPCs that are near their target building enter it.
    let entity_count = state.entities.len();
    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Already inside a building?
        if npc.inside_building_id.is_some() {
            // Check if we should exit (goal changed, walking away).
            let should_exit = match &npc.action {
                NpcAction::Walking { .. } => true,
                NpcAction::Fleeing => true,
                NpcAction::Hauling { .. } => true,
                NpcAction::Idle => {
                    // Exit if no goal targeting this building.
                    npc.goal_stack.current()
                        .and_then(|g| g.target_entity)
                        .map(|te| te != npc.inside_building_id.unwrap_or(u32::MAX))
                        .unwrap_or(true)
                }
                _ => false,
            };
            if should_exit {
                // Exit building.
                let building_id = npc.inside_building_id.unwrap();
                let room_idx = npc.current_room;
                let npc = state.entities[i].npc.as_mut().unwrap();
                npc.inside_building_id = None;
                npc.current_room = None;
                // Clear room occupancy.
                if let Some(ri) = room_idx {
                    if let Some(bld_entity) = state.entity_mut(building_id) {
                        if let Some(bld) = &mut bld_entity.building {
                            if (ri as usize) < bld.rooms.len() {
                                bld.rooms[ri as usize].occupant_id = None;
                            }
                        }
                    }
                }
            }
            continue;
        }

        // Not inside — check if we should enter.
        // Need a target building from current goal or work/home assignment.
        let target_building_id = npc.goal_stack.current()
            .and_then(|g| g.target_entity)
            .or(npc.work_building_id)
            .or(npc.home_building_id);

        let target_building_id = match target_building_id {
            Some(id) => id,
            None => continue,
        };

        // Check distance to target building.
        let bld_pos = state.entities.iter()
            .find(|e| e.id == target_building_id)
            .map(|e| e.pos);
        let bld_pos = match bld_pos { Some(p) => p, None => continue };

        let entity_pos = entity.pos;
        let dx = bld_pos.0 - entity_pos.0;
        let dy = bld_pos.1 - entity_pos.1;
        if dx * dx + dy * dy > ENTER_DIST_SQ { continue; }

        // Close enough — enter building. Pick appropriate room.
        let desired_room = match &npc.action {
            NpcAction::Working { .. } => RoomKind::Workshop,
            NpcAction::Eating { .. } => RoomKind::Kitchen,
            NpcAction::Resting { .. } => RoomKind::Bedroom,
            NpcAction::Socializing { .. } => RoomKind::Hearth,
            NpcAction::Trading { .. } => RoomKind::Counter,
            _ => RoomKind::Entrance,
        };

        // Find available room of desired kind (or fallback to entrance).
        let room_idx = state.entities.iter()
            .find(|e| e.id == target_building_id)
            .and_then(|e| e.building.as_ref())
            .and_then(|bld| {
                // Preferred room.
                let preferred = bld.rooms.iter().enumerate()
                    .find(|(_, r)| r.kind == desired_room && r.occupant_id.is_none())
                    .map(|(idx, _)| idx as u8);
                // Fallback: any empty room.
                preferred.or_else(|| {
                    bld.rooms.iter().enumerate()
                        .find(|(_, r)| r.occupant_id.is_none())
                        .map(|(idx, _)| idx as u8)
                })
            });

        let room_idx = match room_idx { Some(ri) => ri, None => continue }; // building full

        // Get room offset for local_pos.
        let room_offset = state.entities.iter()
            .find(|e| e.id == target_building_id)
            .and_then(|e| e.building.as_ref())
            .and_then(|bld| bld.rooms.get(room_idx as usize))
            .map(|r| r.offset)
            .unwrap_or((0.0, 0.0));

        // Enter: set NPC fields.
        let npc = state.entities[i].npc.as_mut().unwrap();
        npc.inside_building_id = Some(target_building_id);
        npc.current_room = Some(room_idx);
        // Set local_pos relative to building.
        state.entities[i].local_pos = Some((bld_pos.0 + room_offset.0, bld_pos.1 + room_offset.1));

        // Set room occupant.
        let entity_id = state.entities[i].id;
        if let Some(bld_entity) = state.entity_mut(target_building_id) {
            if let Some(bld) = &mut bld_entity.building {
                if (room_idx as usize) < bld.rooms.len() {
                    bld.rooms[room_idx as usize].occupant_id = Some(entity_id);
                }
            }
        }
    }
}
