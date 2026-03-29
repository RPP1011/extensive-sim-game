//! Faction vassalage system — every 500 ticks.
//!
//! Weak factions can become vassals of stronger ones. The guild can demand
//! vassalage from weak factions, accept offers, or liberate vassals.
//!
//! Vassals pay tribute to their lord and receive military protection.
//! Autonomy drifts over time: high autonomy leads to rebellion,
//! low autonomy leads to absorption.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Run the vassalage system every 500 ticks.
pub fn tick_vassalage(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 17 != 0 || state.tick == 0 {
        return;
    }

    let n_factions = state.factions.len();
    if n_factions < 2 {
        return;
    }

    // --- Phase 1: Auto-vassalage for weak NPC factions ---
    auto_vassalage(state, events);

    // --- Phase 2: Tribute collection ---
    collect_tribute(state, events);

    // --- Phase 3: Autonomy drift and rebellion ---
    autonomy_tick(state, events);

    // --- Phase 4: Absorption of very low-autonomy vassals ---
    absorb_vassals(state, events);

    // --- Phase 5: Update system trackers ---
    update_trackers(state);
}

/// Weak factions (strength < 20) with a strong neighbor (strength > 60)
/// become vassals with 30% chance.
fn auto_vassalage(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n = state.factions.len();
    let guild_id = state.diplomacy.guild_faction_id;

    // Collect faction strengths first to avoid borrow issues.
    let strengths: Vec<(usize, f32)> = state
        .factions
        .iter()
        .map(|f| (f.id, f.military_strength))
        .collect();

    // Find factions that are already vassals (either side).
    let already_vassal: Vec<usize> = state
        .vassal_relations
        .iter()
        .map(|v| v.vassal_id)
        .collect();

    for i in 0..n {
        let fid = state.factions[i].id;

        // Skip guild faction, skip already-vassals.
        if fid == guild_id || already_vassal.contains(&fid) {
            continue;
        }

        // Skip factions that are already lords.
        if state.vassal_relations.iter().any(|v| v.lord_id == fid) {
            continue;
        }

        let strength = strengths[i].1;
        if strength >= 20.0 {
            continue;
        }

        // Find strongest neighbor faction.
        let mut best_lord: Option<(usize, f32)> = None;
        for &(other_id, other_str) in &strengths {
            if other_id == fid || other_id == guild_id {
                continue;
            }
            if other_str <= 60.0 {
                continue;
            }
            // Skip if already a vassal themselves.
            if already_vassal.contains(&other_id) {
                continue;
            }
            if best_lord.map_or(true, |(_, s)| other_str > s) {
                best_lord = Some((other_id, other_str));
            }
        }

        if let Some((lord_id, _)) = best_lord {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.30 {
                let tribute_rate = 0.10 + lcg_f32(&mut state.rng) * 0.20; // 10-30%
                state.vassal_relations.push(VassalRelation {
                    vassal_id: fid,
                    lord_id,
                    tribute_rate,
                    autonomy: 50.0,
                    started_tick: state.tick,
                });

                let vassal_name = state
                    .factions
                    .iter()
                    .find(|f| f.id == fid)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", fid));
                let lord_name = state
                    .factions
                    .iter()
                    .find(|f| f.id == lord_id)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", lord_id));

                events.push(WorldEvent::VassalageEstablished {
                    vassal_id: fid,
                    lord_id,
                    vassal_name,
                    lord_name,
                });
            }
        }
    }
}

/// Vassals pay tribute (portion of income) to their lord.
/// Lord provides military protection (small strength boost).
fn collect_tribute(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let guild_id = state.diplomacy.guild_faction_id;

    for rel_idx in 0..state.vassal_relations.len() {
        let vassal_id = state.vassal_relations[rel_idx].vassal_id;
        let lord_id = state.vassal_relations[rel_idx].lord_id;
        let tribute_rate = state.vassal_relations[rel_idx].tribute_rate;

        // Base income proportional to territory.
        let vassal_territory = state
            .factions
            .iter()
            .find(|f| f.id == vassal_id)
            .map(|f| f.territory_size)
            .unwrap_or(0);
        let base_income = vassal_territory as f32 * 5.0;
        let tribute = base_income * tribute_rate;

        if lord_id == guild_id {
            // Lord is the guild — guild receives gold.
            state.guild.gold += tribute;
            events.push(WorldEvent::TributePaid {
                vassal_id,
                lord_id,
                amount: tribute,
            });
        } else if vassal_id == guild_id {
            // Guild is the vassal — guild pays gold.
            state.guild.gold = (state.guild.gold - tribute).max(0.0);
            events.push(WorldEvent::TributePaid {
                vassal_id,
                lord_id,
                amount: tribute,
            });
        } else {
            // NPC-to-NPC tribute: lord gains strength, vassal loses a bit.
            if let Some(lord) = state.factions.iter_mut().find(|f| f.id == lord_id) {
                lord.military_strength += tribute * 0.1;
            }
            events.push(WorldEvent::TributePaid {
                vassal_id,
                lord_id,
                amount: tribute,
            });
        }

        // Lord provides military protection: small strength boost to vassal.
        let lord_strength = state
            .factions
            .iter()
            .find(|f| f.id == lord_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);
        if let Some(vassal) = state.factions.iter_mut().find(|f| f.id == vassal_id) {
            vassal.military_strength += lord_strength * 0.02;
        }
    }
}

/// Autonomy drifts based on lord's treatment.
/// - Lord at war with vassal's enemies: autonomy decreases (protective).
/// - Lord very strong relative to vassal: autonomy decreases.
/// - Vassal recovering strength: autonomy increases.
/// High autonomy (>80) may trigger rebellion.
fn autonomy_tick(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut rebellions = Vec::new();

    for rel in &mut state.vassal_relations {
        let lord_strength = state
            .factions
            .iter()
            .find(|f| f.id == rel.lord_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);
        let vassal_strength = state
            .factions
            .iter()
            .find(|f| f.id == rel.vassal_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);

        // Strength ratio drives autonomy drift.
        let ratio = if lord_strength > 0.0 {
            vassal_strength / lord_strength
        } else {
            1.0
        };

        // If vassal is getting stronger relative to lord, autonomy rises.
        if ratio > 0.5 {
            rel.autonomy += 3.0;
        } else if ratio < 0.2 {
            rel.autonomy -= 2.0;
        } else {
            rel.autonomy += 1.0;
        }

        // High tribute rate increases autonomy (resentment).
        if rel.tribute_rate > 0.25 {
            rel.autonomy += 2.0;
        }

        rel.autonomy = rel.autonomy.clamp(0.0, 100.0);
    }

    // Check for rebellions (autonomy > 80, chance-based).
    for rel in &state.vassal_relations {
        if rel.autonomy > 80.0 {
            let mut rng = state.rng;
            let roll = lcg_f32(&mut rng);
            state.rng = rng;

            // 20% chance of rebellion per tick when autonomy > 80.
            if roll < 0.20 {
                rebellions.push((rel.vassal_id, rel.lord_id));
            }
        }
    }

    for (vassal_id, lord_id) in rebellions {
        // Remove the vassal relation.
        state
            .vassal_relations
            .retain(|v| !(v.vassal_id == vassal_id && v.lord_id == lord_id));

        // Vassal goes hostile toward lord.
        if let Some(vassal) = state.factions.iter_mut().find(|f| f.id == vassal_id) {
            if lord_id == state.diplomacy.guild_faction_id {
                vassal.relationship_to_guild -= 20.0;
            }
        }

        // Update diplomacy matrix.
        let n = state.diplomacy.relations.len();
        if vassal_id < n && lord_id < n {
            state.diplomacy.relations[vassal_id][lord_id] =
                (state.diplomacy.relations[vassal_id][lord_id] - 30).max(-100);
            state.diplomacy.relations[lord_id][vassal_id] =
                (state.diplomacy.relations[lord_id][vassal_id] - 20).max(-100);
        }

        let vassal_name = state
            .factions
            .iter()
            .find(|f| f.id == vassal_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", vassal_id));
        let lord_name = state
            .factions
            .iter()
            .find(|f| f.id == lord_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", lord_id));

        events.push(WorldEvent::VassalRebellion {
            vassal_id,
            lord_id,
            vassal_name,
            lord_name,
        });
    }
}

/// Vassals with autonomy < 20 are gradually absorbed by their lord.
fn absorb_vassals(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut absorbed = Vec::new();

    for rel in &state.vassal_relations {
        if rel.autonomy < 20.0 {
            let mut rng = state.rng;
            let roll = lcg_f32(&mut rng);
            state.rng = rng;

            // 15% chance of absorption when autonomy < 20.
            if roll < 0.15 {
                absorbed.push((rel.vassal_id, rel.lord_id));
            }
        }
    }

    for (vassal_id, lord_id) in absorbed {
        // Transfer territory from vassal to lord.
        let vassal_territory = state
            .factions
            .iter()
            .find(|f| f.id == vassal_id)
            .map(|f| f.territory_size)
            .unwrap_or(0);
        let vassal_strength = state
            .factions
            .iter()
            .find(|f| f.id == vassal_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);

        if let Some(lord) = state.factions.iter_mut().find(|f| f.id == lord_id) {
            lord.territory_size += vassal_territory;
            lord.military_strength += vassal_strength * 0.5;
        }

        // Remove vassal relation.
        state
            .vassal_relations
            .retain(|v| !(v.vassal_id == vassal_id && v.lord_id == lord_id));

        // Mark the vassal faction as absorbed (zero out strength/territory).
        if let Some(vassal) = state.factions.iter_mut().find(|f| f.id == vassal_id) {
            vassal.military_strength = 0.0;
            vassal.territory_size = 0;
        }

        let vassal_name = state
            .factions
            .iter()
            .find(|f| f.id == vassal_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", vassal_id));
        let lord_name = state
            .factions
            .iter()
            .find(|f| f.id == lord_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", lord_id));

        events.push(WorldEvent::CampaignMilestone {
            description: format!(
                "{} has been fully absorbed by {}.",
                vassal_name, lord_name
            ),
        });
    }
}

/// Update system trackers for vassalage.
fn update_trackers(state: &mut CampaignState) {
    let guild_id = state.diplomacy.guild_faction_id;

    state.system_trackers.vassal_count = state.vassal_relations.len() as u32;
    state.system_trackers.guild_vassal_count = state
        .vassal_relations
        .iter()
        .filter(|v| v.lord_id == guild_id)
        .count() as u32;
    state.system_trackers.guild_is_vassal = state
        .vassal_relations
        .iter()
        .any(|v| v.vassal_id == guild_id);

    let guild_vassal_autonomy: Vec<f32> = state
        .vassal_relations
        .iter()
        .filter(|v| v.lord_id == guild_id)
        .map(|v| v.autonomy)
        .collect();
    state.system_trackers.mean_vassal_autonomy = if guild_vassal_autonomy.is_empty() {
        0.0
    } else {
        guild_vassal_autonomy.iter().sum::<f32>() / guild_vassal_autonomy.len() as f32
    };
}
