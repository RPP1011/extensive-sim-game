//! Counter-espionage system — every 7 ticks.
//!
//! Settlements detect hostile NPCs in their vicinity. NPCs with stealth
//! behavior tags are harder to detect. Detection chance is based on
//! settlement population vs spy stealth. Detected spies receive Damage
//! or Die deltas.
//!
//! This complements the espionage system: espionage handles spy intel
//! gathering, while counter-espionage handles settlement-initiated
//! detection sweeps.
//!
//! Ported from `crates/headless_campaign/src/systems/counter_espionage.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, DiplomaticStance, EntityField, EntityKind, FactionField,
    WorldEvent, WorldState, tags,
};
use crate::world_sim::state::pair_hash_f32;

/// Cadence: runs every 7 ticks.
const COUNTER_ESPIONAGE_INTERVAL: u64 = 7;

/// Range (squared) within which a settlement can detect hostiles.
const DETECTION_RANGE_SQ: f32 = 625.0; // 25 units

/// Base detection chance per hostile NPC near a settlement.
const BASE_DETECTION_CHANCE: f32 = 0.08;

/// Stealth value that fully suppresses counter-espionage detection.
const STEALTH_SUPPRESS_AT: f32 = 25.0;

/// Population factor: sqrt(pop) * this scales detection up.
const POP_DETECTION_SCALE: f32 = 0.04;

/// Military outpost bonus to detection chance.
const OUTPOST_BONUS: f32 = 0.10;

/// Damage dealt to a detected hostile NPC.
const DETECTED_SPY_DAMAGE: f32 = 60.0;

/// Lethal threshold: if detection roll is under this fraction of the
/// detection chance, the spy is killed outright.
const LETHAL_FRACTION: f32 = 0.3;

/// Morale boost for settlement-aligned NPCs when a spy is caught.
const MORALE_BOOST_ON_CATCH: f32 = 3.0;


pub fn compute_counter_espionage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % COUNTER_ESPIONAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.settlements.is_empty() || state.factions.is_empty() {
        return;
    }

    // Build a lookup: faction_id -> is_hostile.
    let hostile_faction_ids: Vec<u32> = state
        .factions
        .iter()
        .filter(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            )
        })
        .map(|f| f.id)
        .collect();

    if hostile_faction_ids.is_empty() {
        return;
    }

    // For each settlement, scan nearby entities for hostile faction NPCs.
    for settlement in &state.settlements {
        // Compute settlement detection strength.
        let pop_bonus = (settlement.population as f32).sqrt() * POP_DETECTION_SCALE;
        let is_military = matches!(
            settlement.specialty,
            crate::world_sim::state::SettlementSpecialty::MilitaryOutpost
        );
        let outpost_bonus = if is_military { OUTPOST_BONUS } else { 0.0 };
        let detection_strength = BASE_DETECTION_CHANCE + pop_bonus + outpost_bonus;

        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }
            let npc = match &entity.npc {
                Some(n) => n,
                None => continue,
            };

            // Must belong to a hostile faction.
            let spy_faction_id = match npc.faction_id {
                Some(fid) if hostile_faction_ids.contains(&fid) => fid,
                _ => continue,
            };

            // Must not be at their own faction's settlement.
            if settlement.faction_id == Some(spy_faction_id) {
                continue;
            }

            // Range check.
            let dx = entity.pos.0 - settlement.pos.0;
            let dy = entity.pos.1 - settlement.pos.1;
            if dx * dx + dy * dy > DETECTION_RANGE_SQ {
                continue;
            }

            // Stealth reduces detection chance.
            let stealth_val = npc.behavior_value(tags::STEALTH);
            let stealth_factor = (1.0 - stealth_val / STEALTH_SUPPRESS_AT).clamp(0.05, 1.0);

            let final_detection = detection_strength * stealth_factor;
            let roll = pair_hash_f32(settlement.id, entity.id, state.tick, 0);

            if roll >= final_detection {
                continue; // Not detected this tick.
            }

            // Detected! Determine severity.
            let is_lethal = roll < final_detection * LETHAL_FRACTION;

            if is_lethal {
                // Kill the spy outright.
                out.push(WorldDelta::Die {
                    entity_id: entity.id,
                });

                out.push(WorldDelta::RecordEvent {
                    event: WorldEvent::EntityDied {
                        entity_id: entity.id,
                        cause: format!(
                            "Executed as spy by {} counter-intelligence",
                            settlement.name
                        ),
                    },
                });
            } else {
                // Wound the spy.
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: DETECTED_SPY_DAMAGE,
                    source_id: settlement.id,
                });

                out.push(WorldDelta::RecordEvent {
                    event: WorldEvent::Generic {
                        category: ChronicleCategory::Diplomacy,
                        text: format!(
                            "Hostile agent (entity {}) detected and wounded near {}",
                            entity.id, settlement.name
                        ),
                    },
                });
            }

            // Penalize the spy's faction diplomatically.
            out.push(WorldDelta::UpdateFaction {
                faction_id: spy_faction_id,
                field: FactionField::RelationshipToGuild,
                value: -10.0,
            });

            // Boost morale of friendly NPCs at this settlement.
            for ally in &state.entities {
                if !ally.alive || ally.kind != EntityKind::Npc || ally.id == entity.id {
                    continue;
                }
                let ally_npc = match &ally.npc {
                    Some(n) => n,
                    None => continue,
                };
                if ally_npc.home_settlement_id == Some(settlement.id) {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: ally.id,
                        field: EntityField::Morale,
                        value: MORALE_BOOST_ON_CATCH,
                    });
                }
            }
        }
    }
}
