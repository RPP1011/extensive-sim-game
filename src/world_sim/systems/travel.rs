#![allow(unused)]
//! Party travel — every tick.
//!
//! Moves NPC entities toward their travel destinations via Move deltas.
//! Arrival detection: when distance < threshold, the entity is close enough
//! that the apply phase will snap it.
//!
//! Original: `crates/headless_campaign/src/systems/travel.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, Entity, EntityKind, WorldState};

/// Distance threshold below which an entity is considered "arrived".
const ARRIVAL_THRESHOLD: f32 = 0.5;

/// Speed multiplier for overworld travel (tiles per tick).
const OVERWORLD_SPEED_SCALE: f32 = 0.1;

pub fn compute_travel(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }

        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Determine travel destination from economic intent.
        let destination = match &npc.economic_intent {
            EconomicIntent::Travel { destination } => Some(*destination),
            EconomicIntent::Trade {
                destination_settlement_id,
            } => state.settlement(*destination_settlement_id).map(|s| s.pos),
            // Not actively traveling.
            _ => continue,
        };

        let dest = match destination {
            Some(d) => d,
            None => continue,
        };

        let dx = dest.0 - entity.pos.0;
        let dy = dest.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < ARRIVAL_THRESHOLD {
            // Already at destination — no movement delta needed.
            // The runtime / apply phase handles intent clearing on arrival.
            continue;
        }

        // Normalize direction and scale by entity speed.
        let speed = entity.move_speed * OVERWORLD_SPEED_SCALE;
        let move_dist = speed.min(dist); // Don't overshoot.
        let nx = dx / dist;
        let ny = dy / dist;

        out.push(WorldDelta::Move {
            entity_id: entity.id,
            force: (nx * move_dist, ny * move_dist),
        });

        // Consume food while traveling (commodity 0 = food).
        let food_drain: f32 = 0.005;
        if npc.carried_goods[0] > 0.0 {
            out.push(WorldDelta::ConsumeCommodity {
                location_id: entity.id, // consumed from personal inventory
                commodity: 0,
                amount: food_drain.min(npc.carried_goods[0]),
            });
        }
    }
}
