//! Diplomacy system — every 100 ticks.
//!
//! Manages faction diplomacy using existing WorldState fields:
//! - **Trade agreements**: Factions with positive relations generate gold for
//!   settlements they share borders with (both own settlements in same region).
//! - **Relation changes**: Factions sharing regions with high threat see improved
//!   relations (common enemy effect).
//! - **Alliance coordination**: Allied factions (relationship_to_guild > 50)
//!   share threat reduction across their settlements.
//!
//! All output is via WorldDelta — state is read-only.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, ChronicleEntry, DiplomaticStance, FactionField, SettlementField,
    WorldState,
};
use crate::world_sim::state::pair_hash_f32;

/// Cadence for diplomacy processing (every 100 ticks).
const DIPLOMACY_INTERVAL: u64 = 100;

/// Base gold per settlement pair from a trade-friendly relationship.
const TRADE_GOLD_PER_PAIR: f32 = 0.5;

/// Relation boost per shared-threat region per cycle.
const SHARED_THREAT_RELATION_BOOST: f32 = 1.0;

/// Threat level threshold to count a region as "threatened".
const THREAT_THRESHOLD: f32 = 0.3;

/// Relationship threshold for alliance coordination.
const ALLIANCE_THRESHOLD: f32 = 50.0;

/// Threat reduction applied to allied faction settlements per cycle.
const ALLIANCE_THREAT_REDUCTION: f32 = 0.02;

/// Relationship threshold for trade income generation.
const TRADE_RELATION_THRESHOLD: f32 = 20.0;


pub fn compute_diplomacy(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DIPLOMACY_INTERVAL != 0 || state.tick == 0 {
        return;
    }
    if state.factions.len() < 2 {
        return;
    }

    compute_trade_income(state, out);
    compute_shared_threat_relations(state, out);
    compute_alliance_coordination(state, out);
}

/// Trade agreements: faction pairs with positive relations generate gold income
/// for settlements where both factions have a presence in the same region.
///
/// For each pair of factions with relation > TRADE_RELATION_THRESHOLD, find
/// settlements owned by each faction that share a region. Each such settlement
/// pair generates a small treasury boost for both settlements.
fn compute_trade_income(state: &WorldState, out: &mut Vec<WorldDelta>) {
    let factions = &state.factions;

    for (i, fa) in factions.iter().enumerate() {
        for fb in factions.iter().skip(i + 1) {
            // Skip factions at war.
            if fa.at_war_with.contains(&fb.id) || fb.at_war_with.contains(&fa.id) {
                continue;
            }

            // Determine relationship between the pair.
            // We use relationship_to_guild when one faction is effectively the
            // player guild proxy (faction 0). For faction-to-faction, we infer
            // from diplomatic stance.
            let relation = estimate_bilateral_relation(fa.id, fb.id, state);
            if relation <= TRADE_RELATION_THRESHOLD {
                continue;
            }

            // Find settlements owned by each faction that share a region.
            let fa_settlements: Vec<u32> = state
                .settlements
                .iter()
                .filter(|s| s.faction_id == Some(fa.id))
                .map(|s| s.id)
                .collect();
            let fb_settlements: Vec<u32> = state
                .settlements
                .iter()
                .filter(|s| s.faction_id == Some(fb.id))
                .map(|s| s.id)
                .collect();

            if fa_settlements.is_empty() || fb_settlements.is_empty() {
                continue;
            }

            // Find shared regions (both factions own at least one settlement with
            // a grid in the same region, or we approximate by region ownership).
            let fa_regions: Vec<u32> = state
                .regions
                .iter()
                .filter(|r| r.faction_id == Some(fa.id))
                .map(|r| r.id)
                .collect();
            let fb_regions: Vec<u32> = state
                .regions
                .iter()
                .filter(|r| r.faction_id == Some(fb.id))
                .map(|r| r.id)
                .collect();

            // Check for border adjacency: do they have neighboring regions?
            // Since we don't have explicit adjacency, count settlement co-presence
            // as a proxy — if both have settlements at all, they trade.
            let has_border = !fa_regions.is_empty() && !fb_regions.is_empty();
            if !has_border && fa_settlements.is_empty() {
                continue;
            }

            // Scale gold by relation strength (0 at threshold, full at 100).
            let relation_factor =
                (relation - TRADE_RELATION_THRESHOLD) / (100.0 - TRADE_RELATION_THRESHOLD);
            let gold = TRADE_GOLD_PER_PAIR * relation_factor;

            // Each faction's settlements get a treasury boost.
            for &sid in &fa_settlements {
                out.push(WorldDelta::UpdateTreasury {
                    settlement_id: sid,
                    delta: gold,
                });
            }
            for &sid in &fb_settlements {
                out.push(WorldDelta::UpdateTreasury {
                    settlement_id: sid,
                    delta: gold,
                });
            }

            // Record notable trade events at higher relation thresholds.
            if relation > 70.0 && pair_hash_f32(fa.id, fb.id, state.tick, 10 as u64) < 0.1 {
                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: ChronicleCategory::Diplomacy,
                        text: format!(
                            "A prosperous trade agreement between {} and {} enriches border settlements.",
                            fa.name, fb.name
                        ),
                        entity_ids: vec![fa.id, fb.id],
                    },
                });
            }
        }
    }
}

/// Shared-threat relation improvement: factions controlling regions with high
/// threat levels see their relations improve (enemy of my enemy effect).
///
/// For each region with threat > THREAT_THRESHOLD, find which factions own
/// neighboring regions or settlements in the same area. Those factions get a
/// small relation boost toward the guild.
fn compute_shared_threat_relations(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Build a map: faction_id -> list of region IDs they control.
    let mut faction_regions: Vec<(u32, Vec<u32>)> = Vec::new();
    for faction in &state.factions {
        let regions: Vec<u32> = state
            .regions
            .iter()
            .filter(|r| r.faction_id == Some(faction.id))
            .map(|r| r.id)
            .collect();
        if !regions.is_empty() {
            faction_regions.push((faction.id, regions));
        }
    }

    // For each high-threat region, find all factions with presence there
    // or in adjacent regions (using region ID proximity as adjacency heuristic).
    for region in &state.regions {
        if region.threat_level < THREAT_THRESHOLD {
            continue;
        }

        // Find factions that own this region or neighboring regions (id +/- 1).
        let mut affected_factions: Vec<u32> = Vec::new();
        for (fid, regions) in &faction_regions {
            let near_threat = regions.iter().any(|&rid| {
                rid == region.id
                    || rid == region.id.wrapping_add(1)
                    || rid == region.id.wrapping_sub(1)
            });
            if near_threat {
                affected_factions.push(*fid);
            }
        }

        if affected_factions.len() < 2 {
            continue;
        }

        // All affected factions get a relation boost toward the guild,
        // scaled by threat severity.
        let boost = SHARED_THREAT_RELATION_BOOST * region.threat_level;
        for &fid in &affected_factions {
            // Clamp check: only boost if not already at max.
            let faction = match state.factions.iter().find(|f| f.id == fid) {
                Some(f) => f,
                None => continue,
            };
            if faction.relationship_to_guild >= 100.0 {
                continue;
            }
            // Skip factions at war with the guild (stance == AtWar).
            if faction.diplomatic_stance == DiplomaticStance::AtWar {
                continue;
            }

            out.push(WorldDelta::UpdateFaction {
                faction_id: fid,
                field: FactionField::RelationshipToGuild,
                value: boost,
            });
        }

        // Record chronicle for major threat-driven diplomacy.
        if affected_factions.len() >= 2
            && region.threat_level > 0.6
            && pair_hash_f32(region.id, 0, state.tick, 20 as u64) < 0.15
        {
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Diplomacy,
                    text: format!(
                        "Rising threats in {} drive neighboring factions toward cooperation.",
                        region.name
                    ),
                    entity_ids: affected_factions.clone(),
                },
            });
        }
    }
}

/// Alliance coordination: factions allied with the guild (relationship > 50)
/// share threat reduction. Allied faction settlements get reduced threat,
/// simulating coordinated patrols and intelligence sharing.
fn compute_alliance_coordination(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Find allied factions (relationship_to_guild > ALLIANCE_THRESHOLD,
    // not at war, stance is Friendly or Coalition).
    let allied_factions: Vec<u32> = state
        .factions
        .iter()
        .filter(|f| {
            f.relationship_to_guild > ALLIANCE_THRESHOLD
                && !matches!(
                    f.diplomatic_stance,
                    DiplomaticStance::AtWar | DiplomaticStance::Hostile
                )
        })
        .map(|f| f.id)
        .collect();

    if allied_factions.is_empty() {
        return;
    }

    // Reduce threat at settlements owned by allied factions.
    for settlement in &state.settlements {
        let faction_id = match settlement.faction_id {
            Some(fid) => fid,
            None => continue,
        };
        if !allied_factions.contains(&faction_id) {
            continue;
        }
        // Only reduce if threat is positive.
        if settlement.threat_level <= 0.0 {
            continue;
        }

        // Scale reduction by how strong the alliance is.
        let faction = match state.factions.iter().find(|f| f.id == faction_id) {
            Some(f) => f,
            None => continue,
        };
        let alliance_strength =
            (faction.relationship_to_guild - ALLIANCE_THRESHOLD) / (100.0 - ALLIANCE_THRESHOLD);
        let reduction = ALLIANCE_THREAT_REDUCTION * alliance_strength;

        // Don't reduce below zero.
        let clamped = reduction.min(settlement.threat_level);
        if clamped > 0.0 {
            out.push(WorldDelta::UpdateSettlementField {
                settlement_id: settlement.id,
                field: SettlementField::ThreatLevel,
                value: -clamped,
            });
        }
    }

    // Record alliance coordination events occasionally.
    if allied_factions.len() >= 2
        && pair_hash_f32(allied_factions[0], allied_factions[1], state.tick, 30 as u64) < 0.08
    {
        // Look up names for the chronicle.
        let names: Vec<&str> = allied_factions
            .iter()
            .take(3)
            .filter_map(|&fid| state.factions.iter().find(|f| f.id == fid).map(|f| f.name.as_str()))
            .collect();
        if names.len() >= 2 {
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Diplomacy,
                    text: format!(
                        "Allied factions {} coordinate patrols, reducing regional threats.",
                        names.join(" and ")
                    ),
                    entity_ids: allied_factions.clone(),
                },
            });
        }
    }
}

/// Estimate bilateral relation between two factions.
///
/// Since WorldState only stores `relationship_to_guild` per faction, we use:
/// - If either faction ID is 0 (guild proxy), use the other's `relationship_to_guild`.
/// - Otherwise, infer from diplomatic stances:
///   Friendly/Coalition = 60, Neutral = 0, Hostile = -40, AtWar = -80.
///   Average the two stances and add a small offset based on whether they share enemies.
fn estimate_bilateral_relation(fa_id: u32, fb_id: u32, state: &WorldState) -> f32 {
    // Guild is typically faction 0, but check both directions.
    if fa_id == 0 {
        return state
            .factions
            .iter()
            .find(|f| f.id == fb_id)
            .map(|f| f.relationship_to_guild)
            .unwrap_or(0.0);
    }
    if fb_id == 0 {
        return state
            .factions
            .iter()
            .find(|f| f.id == fa_id)
            .map(|f| f.relationship_to_guild)
            .unwrap_or(0.0);
    }

    let fa = state.factions.iter().find(|f| f.id == fa_id);
    let fb = state.factions.iter().find(|f| f.id == fb_id);
    let (fa, fb) = match (fa, fb) {
        (Some(a), Some(b)) => (a, b),
        _ => return 0.0,
    };

    // Direct war check.
    if fa.at_war_with.contains(&fb_id) || fb.at_war_with.contains(&fa_id) {
        return -80.0;
    }

    let stance_score = |s: DiplomaticStance| -> f32 {
        match s {
            DiplomaticStance::Coalition => 70.0,
            DiplomaticStance::Friendly => 50.0,
            DiplomaticStance::Neutral => 0.0,
            DiplomaticStance::Hostile => -40.0,
            DiplomaticStance::AtWar => -80.0,
        }
    };

    let base = (stance_score(fa.diplomatic_stance) + stance_score(fb.diplomatic_stance)) / 2.0;

    // Bonus for shared enemies.
    let shared_enemies = fa
        .at_war_with
        .iter()
        .filter(|e| fb.at_war_with.contains(e))
        .count();
    let enemy_bonus = (shared_enemies as f32) * 10.0;

    (base + enemy_bonus).clamp(-100.0, 100.0)
}
