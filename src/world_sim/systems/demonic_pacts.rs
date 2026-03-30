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

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

/// Pact system tick interval.
const PACT_INTERVAL: u64 = 7;

/// Damage from minor pact corruption.
const MINOR_CORRUPTION_DAMAGE: f32 = 1.0;

/// Gold drained at moderate corruption.
const MODERATE_GOLD_DRAIN: f32 = 10.0;

/// Damage from severe possession event.
const POSSESSION_DAMAGE: f32 = 15.0;


pub fn compute_demonic_pacts(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PACT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_demonic_pacts_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_demonic_pacts_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % PACT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without demonic_pacts state, we use a proxy: NPCs with very low HP
    // and existing debuffs are "pact-holders" who suffer escalating effects.
    // This is a placeholder until full pact tracking is added.

    for entity in entities {
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
                    from_entity: entity.id,
                    to_entity: home,
                    amount: MODERATE_GOLD_DRAIN,
                });
            }
        }

        // Severe: possession — damage a nearby ally in the same settlement
        if debuff_count >= 3 {
            let roll = entity_hash_f32(entity.id, state.tick, 0xDE30);
            if roll < 0.15 {
                for other in entities {
                    if other.id == entity.id {
                        continue;
                    }
                    if other.alive && other.kind == EntityKind::Npc {
                        out.push(WorldDelta::Damage {
                            target_id: other.id,
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
