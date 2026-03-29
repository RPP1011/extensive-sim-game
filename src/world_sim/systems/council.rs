#![allow(unused)]
//! Guild council voting system — every 7 ticks.
//!
//! NPCs at settlements vote on faction policy. The system counts NPCs per
//! faction at each settlement; the majority-faction's diplomatic stance
//! influences the faction's relationship with the guild.
//!
//! Simple implementation: no stored CouncilVote state required. Each interval,
//! for each settlement we tally faction representation among living NPCs and
//! emit UpdateFaction deltas based on the outcome.
//!
//! Uses only existing WorldState types and WorldDelta variants.
//!
//! Ported from `crates/headless_campaign/src/systems/council.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, ChronicleEntry, EntityKind, FactionField, WorldEvent, WorldState,
};

/// Cadence: every 7 ticks.
const COUNCIL_INTERVAL: u64 = 7;

/// Minimum number of NPCs at a settlement to hold a council vote.
const MIN_COUNCIL_SIZE: usize = 3;

/// Relationship bonus when a faction has strong NPC representation.
const MAJORITY_RELATIONSHIP_BONUS: f32 = 2.0;

/// Relationship penalty when a hostile faction dominates a settlement.
const HOSTILE_MAJORITY_PENALTY: f32 = -1.5;

/// Morale boost for NPCs on the winning side of a council vote.
const WINNER_MORALE_BOOST: f32 = 3.0;

/// Morale penalty for NPCs on the losing side.
const LOSER_MORALE_PENALTY: f32 = -2.0;

pub fn compute_council(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % COUNCIL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Skip if no factions or settlements.
    if state.factions.is_empty() || state.settlements.is_empty() {
        return;
    }

    // --- For each settlement, count NPC faction representation ---
    for settlement in &state.settlements {
        let sid = settlement.id;

        // Collect living NPCs at this settlement.
        let npcs_at_settlement: Vec<_> = state
            .entities
            .iter()
            .filter(|e| {
                e.kind == EntityKind::Npc
                    && e.alive
                    && e.npc
                        .as_ref()
                        .map(|n| n.home_settlement_id == Some(sid))
                        .unwrap_or(false)
            })
            .collect();

        if npcs_at_settlement.len() < MIN_COUNCIL_SIZE {
            continue;
        }

        // Tally faction membership among NPCs.
        let mut faction_counts: Vec<(u32, usize)> = Vec::new();
        let mut unaffiliated = 0usize;

        for npc_entity in &npcs_at_settlement {
            if let Some(ref npc) = npc_entity.npc {
                if let Some(fid) = npc.faction_id {
                    if let Some(entry) = faction_counts.iter_mut().find(|(id, _)| *id == fid) {
                        entry.1 += 1;
                    } else {
                        faction_counts.push((fid, 1));
                    }
                } else {
                    unaffiliated += 1;
                }
            }
        }

        // No factions represented — skip.
        if faction_counts.is_empty() {
            continue;
        }

        // Sort by count descending, ties broken by faction_id for determinism.
        faction_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let (majority_faction_id, majority_count) = faction_counts[0];
        let total_count = npcs_at_settlement.len();
        let majority_share = majority_count as f32 / total_count as f32;

        // Only apply effects if a clear majority (>40%) exists.
        if majority_share < 0.40 {
            continue;
        }

        // Look up the faction to determine stance effects.
        let majority_faction = match state.factions.iter().find(|f| f.id == majority_faction_id) {
            Some(f) => f,
            None => continue,
        };

        // --- Emit deltas based on majority faction's stance ---
        match majority_faction.diplomatic_stance {
            crate::world_sim::state::DiplomaticStance::Friendly
            | crate::world_sim::state::DiplomaticStance::Coalition => {
                // Friendly majority: boost relationship with guild.
                out.push(WorldDelta::UpdateFaction {
                    faction_id: majority_faction_id,
                    field: FactionField::RelationshipToGuild,
                    value: MAJORITY_RELATIONSHIP_BONUS * majority_share,
                });
            }
            crate::world_sim::state::DiplomaticStance::Hostile
            | crate::world_sim::state::DiplomaticStance::AtWar => {
                // Hostile majority: erode relationship with guild.
                out.push(WorldDelta::UpdateFaction {
                    faction_id: majority_faction_id,
                    field: FactionField::RelationshipToGuild,
                    value: HOSTILE_MAJORITY_PENALTY * majority_share,
                });
            }
            crate::world_sim::state::DiplomaticStance::Neutral => {
                // Neutral majority: slight positive drift.
                out.push(WorldDelta::UpdateFaction {
                    faction_id: majority_faction_id,
                    field: FactionField::RelationshipToGuild,
                    value: 0.5 * majority_share,
                });
            }
        }

        // --- Morale effects on NPCs ---
        // Winning-side NPCs get a morale boost; minority NPCs get a penalty.
        for npc_entity in &npcs_at_settlement {
            if let Some(ref npc) = npc_entity.npc {
                if npc.faction_id == Some(majority_faction_id) {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: npc_entity.id,
                        field: crate::world_sim::state::EntityField::Morale,
                        value: WINNER_MORALE_BOOST,
                    });
                } else if npc.faction_id.is_some() {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: npc_entity.id,
                        field: crate::world_sim::state::EntityField::Morale,
                        value: LOSER_MORALE_PENALTY,
                    });
                }
            }
        }

        // --- Chronicle entry for significant votes ---
        if majority_share > 0.60 {
            out.push(WorldDelta::RecordEvent {
                event: WorldEvent::Generic {
                    category: ChronicleCategory::Diplomacy,
                    text: format!(
                        "Council at settlement {} dominated by {} ({:.0}% representation)",
                        sid,
                        majority_faction.name,
                        majority_share * 100.0,
                    ),
                },
            });
        }
    }
}
