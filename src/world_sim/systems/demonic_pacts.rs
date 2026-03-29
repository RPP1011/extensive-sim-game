#![allow(unused)]
//! Demonic pacts — fires every 7 ticks.
//!
//! NPCs with demonic pacts accrue debt. At thresholds, escalating
//! consequences: minor (morale damage to allies via DoT), moderate
//! (gold drain), severe (possession — damage allies). Maps to Damage,
//! TransferGold, and ApplyStatus deltas.
//!
//! Original: `crates/headless_campaign/src/systems/demonic_pacts.rs`
//!
//! NEEDS STATE: `demonic_pacts: Vec<DemonicPact>` on WorldState
//! NEEDS DELTA: ModifyDebt, SpawnNemesis

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// Pact system tick interval.
const PACT_INTERVAL: u64 = 7;

/// Damage from minor pact corruption.
const MINOR_CORRUPTION_DAMAGE: f32 = 1.0;

/// Gold drained at moderate corruption.
const MODERATE_GOLD_DRAIN: f32 = 10.0;

/// Damage from severe possession event.
const POSSESSION_DAMAGE: f32 = 15.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_demonic_pacts(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PACT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without demonic_pacts state, we use a proxy: NPCs with very low HP
    // and existing debuffs are "pact-holders" who suffer escalating effects.
    // This is a placeholder until full pact tracking is added.

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Proxy: entities with multiple debuffs are "corrupted"
        let debuff_count = entity
            .status_effects
            .iter()
            .filter(|s| matches!(s.kind, StatusEffectKind::Debuff { .. }))
            .count();

        if debuff_count == 0 {
            continue;
        }

        // Minor corruption: periodic damage
        if debuff_count >= 1 {
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: MINOR_CORRUPTION_DAMAGE,
                source_id: 0,
            });
        }

        // Moderate: gold drain
        if debuff_count >= 2 && npc.gold > MODERATE_GOLD_DRAIN {
            if let Some(home) = npc.home_settlement_id {
                out.push(WorldDelta::TransferGold {
                    from_id: entity.id,
                    to_id: home,
                    amount: MODERATE_GOLD_DRAIN,
                });
            }
        }

        // Severe: possession — damage a nearby ally
        if debuff_count >= 3 {
            let roll = tick_hash(state.tick, entity.id as u64 ^ 0xDE30);
            if roll < 0.15 {
                // Find a nearby friendly NPC to damage
                if let Some(grid_id) = entity.grid_id {
                    if let Some(grid) = state.grid(grid_id) {
                        for &other_id in &grid.entity_ids {
                            if other_id == entity.id {
                                continue;
                            }
                            if let Some(other) = state.entity(other_id) {
                                if other.alive && other.kind == EntityKind::Npc {
                                    out.push(WorldDelta::Damage {
                                        target_id: other_id,
                                        amount: POSSESSION_DAMAGE,
                                        source_id: entity.id,
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
