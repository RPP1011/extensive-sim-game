#![allow(unused)]
//! Wound persistence — heals injured NPCs over time, every 3 ticks.
//!
//! Battle survivors sustain persistent damage. Idle NPCs heal faster than
//! traveling or fighting ones. Maps to Heal deltas for gradual recovery.
//!
//! Original: `crates/headless_campaign/src/systems/wound_persistence.rs`
//!
//! NEEDS STATE: `wounds: Vec<PersistentWound>` on Entity/NpcData
//! NEEDS STATE: `activity_status` on NpcData (idle/traveling/fighting)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, EntityKind, WorldState};

/// Heal interval in ticks.
const WOUND_HEAL_INTERVAL: u64 = 3;

/// Per-tick heal amount for idle NPCs.
const HEAL_RATE_IDLE: f32 = 1.0;

/// Per-tick heal amount for traveling NPCs.
const HEAL_RATE_TRAVELING: f32 = 0.5;

/// Per-tick heal amount for fighting NPCs (no healing).
const HEAL_RATE_FIGHTING: f32 = 0.0;

pub fn compute_wound_persistence(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % WOUND_HEAL_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }
            // Only heal entities that are below max HP
            if entity.hp >= entity.max_hp {
                continue;
            }

            let npc = match &entity.npc {
                Some(n) => n,
                None => continue,
            };

            // Determine heal rate based on activity
            let heal_rate = match &npc.economic_intent {
                EconomicIntent::Idle => HEAL_RATE_IDLE,
                EconomicIntent::Travel { .. } | EconomicIntent::Trade { .. } => HEAL_RATE_TRAVELING,
                // Produce/Buy/Sell are "at settlement" activities — heal at idle rate
                EconomicIntent::Produce | EconomicIntent::Buy { .. } | EconomicIntent::Sell { .. } => {
                    HEAL_RATE_IDLE
                }
            };

            if heal_rate <= 0.0 {
                continue;
            }

            // Don't overheal
            let missing = entity.max_hp - entity.hp;
            let amount = heal_rate.min(missing);
            if amount > 0.0 {
                out.push(WorldDelta::Heal {
                    target_id: entity.id,
                    amount,
                    source_id: entity.id, // self-healing
                });
            }
        }
    }
}
