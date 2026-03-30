#![allow(unused)]
//! Vassalage system — every 17 ticks.
//!
//! Weak factions (military_strength < 20) vassalize to strong neighbors
//! (military_strength > 60). Vassalage is derived from the relationship
//! between faction strengths rather than stored VassalRelation state.
//!
//! Tribute flows via TransferGold. Rebellion occurs when the vassal
//! faction's relationship_to_guild diverges heavily from the lord's,
//! signaling resentment buildup.
//!
//! Uses only existing WorldState types and WorldDelta variants:
//! - UpdateFaction for military_strength, relationship_to_guild, treasury
//! - TransferGold for tribute payments
//! - RecordEvent / RecordChronicle for narrative
//!
//! Ported from `crates/headless_campaign/src/systems/vassalage.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, ChronicleEntry, DiplomaticStance, FactionField, WorldEvent, WorldState,
};
use crate::world_sim::state::pair_hash_f32;

/// Cadence: every 17 ticks.
const VASSALAGE_INTERVAL: u64 = 17;

/// Military strength below which a faction is considered weak.
const WEAK_THRESHOLD: f32 = 20.0;

/// Military strength above which a faction can be a lord.
const STRONG_THRESHOLD: f32 = 60.0;

/// Chance per interval for a weak faction to become a vassal.
const VASSALIZATION_CHANCE: f32 = 0.30;

/// Tribute rate: fraction of the weak faction's treasury transferred per interval.
const TRIBUTE_RATE: f32 = 0.10;

/// Military protection: lord contributes this fraction of their strength as a boost.
const PROTECTION_RATE: f32 = 0.02;

/// Relationship threshold below which resentment triggers rebellion.
const REBELLION_RELATIONSHIP_THRESHOLD: f32 = -30.0;

/// Chance per interval for rebellion when resentment is high.
const REBELLION_CHANCE: f32 = 0.15;

/// Relationship penalty when a faction rebels.
const REBELLION_RELATIONSHIP_PENALTY: f32 = -20.0;


pub fn compute_vassalage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % VASSALAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Identify vassalage relationships ---
    // Weak factions (military < WEAK_THRESHOLD) that are NOT at war vassalize
    // to the strongest non-hostile faction.
    compute_auto_vassalage(state, out);

    // --- Phase 2: Tribute and protection ---
    // Every weak faction adjacent to a strong faction pays tribute and receives
    // military protection.
    compute_tribute(state, out);

    // --- Phase 3: Rebellion ---
    // Weak factions with very negative relationship to guild may rebel against
    // their de-facto lord.
    compute_rebellion(state, out);
}

// ---------------------------------------------------------------------------
// Auto-vassalage: weak factions submit to strong ones
// ---------------------------------------------------------------------------

fn compute_auto_vassalage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        // Only weak, non-warring factions can become vassals.
        if faction.military_strength >= WEAK_THRESHOLD {
            continue;
        }
        if !faction.at_war_with.is_empty() {
            continue;
        }
        // Already hostile or at war — too proud to vassalize.
        if faction.diplomatic_stance == DiplomaticStance::Hostile
            || faction.diplomatic_stance == DiplomaticStance::AtWar
        {
            continue;
        }

        // Find the strongest eligible lord.
        let best_lord = state
            .factions
            .iter()
            .filter(|f| {
                f.id != faction.id
                    && f.military_strength > STRONG_THRESHOLD
                    && f.diplomatic_stance != DiplomaticStance::AtWar
            })
            .max_by(|a, b| {
                a.military_strength
                    .partial_cmp(&b.military_strength)
                    .unwrap()
            });

        let lord = match best_lord {
            Some(l) => l,
            None => continue,
        };

        let roll = pair_hash_f32(faction.id, lord.id, state.tick, 0 as u64);
        if roll >= VASSALIZATION_CHANCE {
            continue;
        }

        // Vassalization: weak faction's relationship drifts toward the lord's.
        // This represents political alignment with the stronger power.
        let rel_drift = (lord.relationship_to_guild - faction.relationship_to_guild) * 0.3;
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::RelationshipToGuild,
            value: rel_drift,
        });

        // Small military strength boost from the lord's protection umbrella.
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::MilitaryStrength,
            value: lord.military_strength * PROTECTION_RATE,
        });

        out.push(WorldDelta::RecordEvent {
            event: WorldEvent::Generic {
                category: ChronicleCategory::Diplomacy,
                text: format!(
                    "{} becomes a vassal of {} (military disparity)",
                    faction.name, lord.name
                ),
            },
        });
    }
}

// ---------------------------------------------------------------------------
// Tribute: weak factions pay strong neighbors
// ---------------------------------------------------------------------------

fn compute_tribute(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        if faction.military_strength >= WEAK_THRESHOLD {
            continue;
        }
        if faction.treasury <= 0.0 {
            continue;
        }

        // Find the strongest neighbor (de-facto lord).
        let lord = state
            .factions
            .iter()
            .filter(|f| {
                f.id != faction.id
                    && f.military_strength > STRONG_THRESHOLD
                    && !f.at_war_with.contains(&faction.id)
            })
            .max_by(|a, b| {
                a.military_strength
                    .partial_cmp(&b.military_strength)
                    .unwrap()
            });

        let lord = match lord {
            Some(l) => l,
            None => continue,
        };

        let tribute = faction.treasury * TRIBUTE_RATE;
        if tribute < 0.1 {
            continue;
        }

        // Transfer gold as tribute.
        out.push(WorldDelta::TransferGold {
            from_id: faction.id,
            to_id: lord.id,
            amount: tribute,
        });

        // Lord provides military protection in return.
        let protection = lord.military_strength * PROTECTION_RATE;
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::MilitaryStrength,
            value: protection,
        });
    }
}

// ---------------------------------------------------------------------------
// Rebellion: resentful vassals break free
// ---------------------------------------------------------------------------

fn compute_rebellion(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        // Only weak factions with deep resentment can rebel.
        if faction.military_strength >= WEAK_THRESHOLD {
            continue;
        }
        if faction.relationship_to_guild > REBELLION_RELATIONSHIP_THRESHOLD {
            continue;
        }

        // There must be a lord to rebel against.
        let lord = state
            .factions
            .iter()
            .filter(|f| {
                f.id != faction.id
                    && f.military_strength > STRONG_THRESHOLD
            })
            .max_by(|a, b| {
                a.military_strength
                    .partial_cmp(&b.military_strength)
                    .unwrap()
            });

        let lord = match lord {
            Some(l) => l,
            None => continue,
        };

        let roll = pair_hash_f32(faction.id, lord.id, state.tick, 100 as u64);
        if roll >= REBELLION_CHANCE {
            continue;
        }

        // Rebellion! Vassal breaks free with a military surge.
        let surge = 5.0 + faction.military_strength * 0.5;
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::MilitaryStrength,
            value: surge,
        });

        // Lord loses a small amount of strength from the uprising.
        out.push(WorldDelta::UpdateFaction {
            faction_id: lord.id,
            field: FactionField::MilitaryStrength,
            value: -surge * 0.3,
        });

        // Relationship craters between rebel and guild.
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::RelationshipToGuild,
            value: REBELLION_RELATIONSHIP_PENALTY,
        });

        // Lord's relationship with guild improves (sympathy).
        out.push(WorldDelta::UpdateFaction {
            faction_id: lord.id,
            field: FactionField::RelationshipToGuild,
            value: 5.0,
        });

        // Coup risk rises for the rebel faction.
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::CoupRisk,
            value: 0.15,
        });

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Crisis,
                text: format!(
                    "{} rebels against {}! The vassal rises with newfound military strength.",
                    faction.name, lord.name
                ),
                entity_ids: vec![],
            },
        });

        out.push(WorldDelta::RecordEvent {
            event: WorldEvent::Generic {
                category: ChronicleCategory::Diplomacy,
                text: format!("{} rebels against vassal lord {}", faction.name, lord.name),
            },
        });
    }
}
