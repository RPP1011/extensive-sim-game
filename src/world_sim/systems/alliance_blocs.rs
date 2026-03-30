//! Alliance bloc system — every 17 ticks.
//!
//! Factions with mutual friendly stance form blocs and share resources.
//! Bloc membership is derived each tick from diplomatic_stance (no stored
//! AllianceBlocState required). Friendly factions pool treasury via
//! TransferGold and gain cohesion bonuses through UpdateFaction deltas.
//!
//! Ported from `crates/headless_campaign/src/systems/alliance_blocs.rs`,
//! rewritten to use existing WorldState types and WorldDelta variants only.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, DiplomaticStance, FactionField, WorldEvent, WorldState,
};

/// Cadence: every 17 ticks.
const BLOC_INTERVAL: u64 = 17;

/// Resource-sharing fraction: each friendly faction transfers this fraction
/// of its treasury surplus (above 50) to the poorest member each interval.
const RESOURCE_SHARE_FRACTION: f32 = 0.05;

/// Relationship boost between friendly factions each interval.
const COHESION_RELATIONSHIP_BOOST: f32 = 1.0;

/// Military strength boost from having an ally (per ally).
const ALLIANCE_STRENGTH_BONUS: f32 = 0.5;


pub fn compute_alliance_blocs(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BLOC_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Identify alliance blocs ---
    // A bloc is a set of factions that are ALL mutually Friendly.
    // We derive blocs from current state each tick (no stored AllianceBlocState).
    let friendly_factions: Vec<&crate::world_sim::state::FactionState> = state
        .factions
        .iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::Friendly)
        .collect();

    if friendly_factions.len() < 2 {
        return;
    }

    // Build pairs of mutually-friendly factions (both Friendly stance, both
    // have positive relationship_to_guild as proxy for mutual goodwill).
    let mut bloc_pairs: Vec<(u32, u32)> = Vec::new();
    for (i, fa) in friendly_factions.iter().enumerate() {
        for fb in friendly_factions.iter().skip(i + 1) {
            // Both friendly to the guild and to each other (positive relations).
            if fa.relationship_to_guild > 20.0 && fb.relationship_to_guild > 20.0 {
                bloc_pairs.push((fa.id, fb.id));
            }
        }
    }

    if bloc_pairs.is_empty() {
        return;
    }

    // --- Phase 2: Resource sharing ---
    // Among friendly-bloc factions, the wealthiest transfers a fraction of
    // surplus treasury to the poorest.
    let richest = friendly_factions
        .iter()
        .max_by(|a, b| a.treasury.partial_cmp(&b.treasury).unwrap_or(std::cmp::Ordering::Equal));
    let poorest = friendly_factions
        .iter()
        .min_by(|a, b| a.treasury.partial_cmp(&b.treasury).unwrap_or(std::cmp::Ordering::Equal));

    if let (Some(rich), Some(poor)) = (richest, poorest) {
        if rich.id != poor.id {
            let surplus = (rich.treasury - 50.0).max(0.0);
            let transfer = surplus * RESOURCE_SHARE_FRACTION;
            if transfer > 0.1 {
                out.push(WorldDelta::TransferGold {
                    from_entity: rich.id,
                    to_entity: poor.id,
                    amount: transfer,
                });
            }
        }
    }

    // --- Phase 3: Cohesion bonuses ---
    // Each pair of friendly factions gains a small relationship and military boost.
    for &(fa_id, fb_id) in &bloc_pairs {
        // Boost relationship to guild for both members (cohesion reward).
        out.push(WorldDelta::UpdateFaction {
            faction_id: fa_id,
            field: FactionField::RelationshipToGuild,
            value: COHESION_RELATIONSHIP_BOOST,
        });
        out.push(WorldDelta::UpdateFaction {
            faction_id: fb_id,
            field: FactionField::RelationshipToGuild,
            value: COHESION_RELATIONSHIP_BOOST,
        });

        // Small military strength bonus from alliance synergy.
        out.push(WorldDelta::UpdateFaction {
            faction_id: fa_id,
            field: FactionField::MilitaryStrength,
            value: ALLIANCE_STRENGTH_BONUS,
        });
        out.push(WorldDelta::UpdateFaction {
            faction_id: fb_id,
            field: FactionField::MilitaryStrength,
            value: ALLIANCE_STRENGTH_BONUS,
        });
    }

    // --- Phase 4: Cohesion decay ---
    // Factions that are Friendly but have declining relationship drift apart.
    // If a friendly faction's relationship_to_guild dips below 10, push them
    // toward Neutral (emitted as a chronicle note — stance changes are handled
    // by the diplomacy/faction_ai systems).
    for f in &friendly_factions {
        if f.relationship_to_guild < 10.0 {
            out.push(WorldDelta::RecordEvent {
                event: WorldEvent::Generic {
                    category: ChronicleCategory::Diplomacy,
                    text: format!(
                        "Alliance cohesion weakening: {} shows signs of discontent",
                        f.name
                    ),
                },
            });
            // Nudge relationship further down to accelerate natural stance drift.
            out.push(WorldDelta::UpdateFaction {
                faction_id: f.id,
                field: FactionField::RelationshipToGuild,
                value: -2.0,
            });
        }
    }

    // --- Phase 5: Coordinated defense ---
    // If any friendly-bloc faction is at war, allies contribute military aid.
    for f in &friendly_factions {
        if !f.at_war_with.is_empty() {
            // All other friendly factions contribute a small strength boost.
            for ally in &friendly_factions {
                if ally.id != f.id && ally.military_strength > 30.0 {
                    let aid = ally.military_strength * 0.01;
                    out.push(WorldDelta::UpdateFaction {
                        faction_id: f.id,
                        field: FactionField::MilitaryStrength,
                        value: aid,
                    });
                    out.push(WorldDelta::UpdateFaction {
                        faction_id: ally.id,
                        field: FactionField::MilitaryStrength,
                        value: -aid * 0.5, // Aid costs the helper half as much.
                    });
                }
            }
        }
    }
}
