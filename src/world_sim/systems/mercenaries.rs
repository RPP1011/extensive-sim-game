#![allow(unused)]
//! Mercenary contract system — fires every 7 ticks.
//!
//! Hired mercenaries drain gold each cycle and may desert or betray if
//! unpaid. In the delta architecture, mercenary costs map to TransferGold
//! (payment to mercenary), and betrayal maps to Damage.
//!
//! Original: `crates/headless_campaign/src/systems/mercenaries.rs`
//!
//! NEEDS STATE: `hired_mercenaries: Vec<MercenaryCompany>` on WorldState
//! NEEDS STATE: `available_mercenaries: Vec<MercenaryCompany>` on WorldState
//! NEEDS DELTA: HireMercenary, DismissMercenary, MercenaryBetray

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// Mercenary tick interval.
const MERCENARY_INTERVAL: u64 = 7;

/// Gold cost per mercenary entity per tick cycle (maintenance).
const MERCENARY_MAINTENANCE_COST: f32 = 5.0;

/// Damage from mercenary betrayal.
const BETRAYAL_DAMAGE: f32 = 20.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_mercenaries(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MERCENARY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without mercenary tracking, we approximate:
    // Settlements with low treasury that have monsters nearby may attract
    // mercenary interest. This results in treasury drain (mercenary fees).

    for settlement in &state.settlements {
        // Only settlements under threat need mercenaries
        let under_threat = state.regions.iter().any(|r| r.threat_level > 50.0);
        if !under_threat {
            continue;
        }

        // If settlement can afford it, pay mercenary maintenance
        if settlement.treasury > MERCENARY_MAINTENANCE_COST * 3.0 && settlement.treasury > -100.0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -MERCENARY_MAINTENANCE_COST,
            });
        } else {
            // Can't pay — risk of betrayal: damage NPCs
            let roll = tick_hash(state.tick, settlement.id as u64 ^ 0xBE4C);
            if roll < 0.05 {
                if let Some(grid_id) = settlement.grid_id {
                    if let Some(grid) = state.grid(grid_id) {
                        for &entity_id in &grid.entity_ids {
                            if let Some(entity) = state.entity(entity_id) {
                                if entity.alive && entity.kind == EntityKind::Npc {
                                    out.push(WorldDelta::Damage {
                                        target_id: entity_id,
                                        amount: BETRAYAL_DAMAGE,
                                        source_id: 0,
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
