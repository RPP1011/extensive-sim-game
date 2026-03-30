//! Espionage system — every 7 ticks.
//!
//! Hostile factions send spies (NPCs from hostile factions positioned near
//! enemy settlements). Spies gather intel (reducing target faction military
//! effectiveness) and risk detection. Detected spies take Damage.
//!
//! Uses existing WorldState types: entities with npc.faction_id identify spies
//! vs locals. UpdateFaction for intel effects. Damage/Die for caught spies.
//!
//! Ported from `crates/headless_campaign/src/systems/espionage.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ActionTags, DiplomaticStance, EntityKind, FactionField, WorldEvent, WorldState, tags,
};
use crate::world_sim::state::pair_hash_f32;

/// Cadence: runs every 7 ticks.
const ESPIONAGE_INTERVAL: u64 = 7;

/// Distance squared within which an NPC is "near" a settlement (spy range).
const SPY_RANGE_SQ: f32 = 400.0; // 20 units

/// Base military strength reduction per spy per tick.
const BASE_INTEL_DRAIN: f32 = 2.0;

/// Extra drain when the spy has stealth behavior tags.
const STEALTH_BONUS_DRAIN: f32 = 1.5;

/// Cover threshold: if detection roll exceeds this, the spy is caught.
const BASE_DETECTION_CHANCE: f32 = 0.12;

/// Stealth tag value that fully suppresses detection (linear scale).
const STEALTH_SUPPRESS_AT: f32 = 20.0;

/// Damage dealt to a caught spy.
const CAUGHT_SPY_DAMAGE: f32 = 80.0;

/// Faction relation penalty when a spy is caught.
const CAUGHT_RELATION_PENALTY: f32 = 15.0;


pub fn compute_espionage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ESPIONAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Build set of hostile faction IDs for quick lookup.
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

    // For each alive NPC entity belonging to a hostile faction, check if they
    // are near a settlement owned by a different (non-hostile-to-guild) faction.
    // If so, they are acting as a spy.
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        let spy_faction_id = match npc.faction_id {
            Some(fid) if hostile_faction_ids.contains(&fid) => fid,
            _ => continue,
        };

        // Check proximity to settlements NOT owned by the spy's faction.
        for settlement in &state.settlements {
            // Skip settlements owned by the spy's own faction.
            if settlement.faction_id == Some(spy_faction_id) {
                continue;
            }

            let dx = entity.pos.0 - settlement.pos.0;
            let dy = entity.pos.1 - settlement.pos.1;
            if dx * dx + dy * dy > SPY_RANGE_SQ {
                continue;
            }

            // This NPC is a spy near a target settlement. Compute effects.

            // --- Intel gathering: reduce target settlement's faction military strength ---
            let stealth_val = npc.behavior_value(tags::STEALTH);
            let deception_val = npc.behavior_value(tags::DECEPTION);
            let spy_skill = 1.0 + (stealth_val + deception_val) * 0.05;
            let drain = BASE_INTEL_DRAIN * spy_skill + if stealth_val > 5.0 { STEALTH_BONUS_DRAIN } else { 0.0 };

            // Drain military strength of the settlement's owning faction.
            if let Some(target_faction_id) = settlement.faction_id {
                out.push(WorldDelta::UpdateFaction {
                    faction_id: target_faction_id,
                    field: FactionField::MilitaryStrength,
                    value: -drain,
                });
            }

            // Accumulate espionage behavior tags on the spy.
            let mut action_tags = ActionTags::empty();
            action_tags.add(tags::STEALTH, 2.0);
            action_tags.add(tags::DECEPTION, 1.0);
            out.push(WorldDelta::AddBehaviorTags {
                entity_id: entity.id,
                tags: action_tags.tags,
                count: action_tags.count,
            });

            // --- Detection check ---
            let roll = pair_hash_f32(entity.id, settlement.id, state.tick, 0);
            // Stealth reduces detection: at STEALTH_SUPPRESS_AT value, detection is 0.
            let stealth_factor = (1.0 - stealth_val / STEALTH_SUPPRESS_AT).clamp(0.0, 1.0);
            // Higher population settlements are better at detecting spies.
            let pop_factor = 1.0 + (settlement.population as f32).sqrt() * 0.02;
            let detection_threshold = BASE_DETECTION_CHANCE * stealth_factor * pop_factor;

            if roll < detection_threshold {
                // Spy caught!
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: CAUGHT_SPY_DAMAGE,
                    source_id: settlement.id,
                });

                // Penalize the spy's faction relationship with guild.
                out.push(WorldDelta::UpdateFaction {
                    faction_id: spy_faction_id,
                    field: FactionField::RelationshipToGuild,
                    value: -CAUGHT_RELATION_PENALTY,
                });

                // Record the event.
                out.push(WorldDelta::RecordEvent {
                    event: WorldEvent::Generic {
                        category: crate::world_sim::state::ChronicleCategory::Diplomacy,
                        text: format!(
                            "Spy (entity {}) from faction {} caught near settlement {}",
                            entity.id, spy_faction_id, settlement.name
                        ),
                    },
                });

                // Only process the first settlement hit per spy per tick.
                break;
            }
        }
    }
}
