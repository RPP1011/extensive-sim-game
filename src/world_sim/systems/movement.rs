//! Unified movement system — the ONLY system that moves entities.
//!
//! All other systems set entity.move_target to request movement.
//! This system reads move_target, moves the entity toward it at
//! entity.move_speed * entity.move_speed_mult, and clears it on arrival.

use crate::world_sim::state::*;

const ARRIVAL_DIST: f32 = 1.5;

pub fn advance_movement(state: &mut WorldState) {
    let dt = crate::world_sim::DT_SEC;

    // Snapshot tile data for movement cost queries (avoid borrow conflicts).
    let tiles = &state.tiles;

    for entity in &mut state.entities {
        if !entity.alive { continue; }
        let target = match entity.move_target {
            Some(t) => t,
            None => continue,
        };

        // CC blocks movement — stun/root/freeze prevent moving
        let cc_blocked = entity.status_effects.iter().any(|s| matches!(s.kind,
            StatusEffectKind::Stun | StatusEffectKind::Root
        ));
        if cc_blocked { continue; }

        let dx = target.0 - entity.pos.0;
        let dy = target.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < ARRIVAL_DIST {
            entity.move_target = None;
            continue;
        }

        // Tile-based movement cost at current position.
        let tile_pos = TilePos::from_world(entity.pos.0, entity.pos.1);
        let tile_cost = tiles.get(&tile_pos)
            .map(|t| {
                let cost = t.tile_type.movement_cost();
                // Fences block monsters but not NPCs.
                if t.tile_type.blocks_monsters_only() && entity.kind == EntityKind::Monster {
                    return f32::MAX;
                }
                cost
            })
            .unwrap_or(1.0); // no tile = open ground, normal speed

        // Impassable tile: stop movement.
        if tile_cost >= f32::MAX {
            entity.move_target = None;
            continue;
        }

        // Also check the tile we're moving INTO for blocking.
        let step_x = entity.pos.0 + dx / dist * 1.0;
        let step_y = entity.pos.1 + dy / dist * 1.0;
        let next_tile_pos = TilePos::from_world(step_x, step_y);
        let next_cost = tiles.get(&next_tile_pos)
            .map(|t| {
                if t.tile_type.is_solid() { return f32::MAX; }
                if t.tile_type.blocks_monsters_only() && entity.kind == EntityKind::Monster {
                    return f32::MAX;
                }
                t.tile_type.movement_cost()
            })
            .unwrap_or(1.0);

        if next_cost >= f32::MAX {
            entity.move_target = None;
            continue;
        }

        // Apply movement with tile cost modifier (lower cost = faster).
        let effective_cost = (tile_cost + next_cost) * 0.5; // average current + next
        let speed = entity.move_speed * entity.move_speed_mult * dt / effective_cost;
        entity.pos.0 += dx / dist * speed;
        entity.pos.1 += dy / dist * speed;
    }
}
