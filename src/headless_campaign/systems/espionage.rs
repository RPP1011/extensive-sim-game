//! Espionage system — every 200 ticks.
//!
//! Manages spies planted by the guild in factions. Spies gather intel,
//! risk discovery, and can perform sabotage actions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 200 ticks.
const ESPIONAGE_INTERVAL: u64 = 200;

/// Base intel gathered per espionage tick.
const BASE_INTEL_PER_TICK: f32 = 5.0;

/// Base cover degradation per espionage tick.
const BASE_COVER_DECAY: f32 = 2.0;

/// Extra cover decay when the target faction is hostile.
const HOSTILE_COVER_DECAY: f32 = 5.0;

/// Cover threshold below which discovery checks occur.
const DISCOVERY_THRESHOLD: f32 = 20.0;

/// Chance (0–1) of being caught per tick when cover < threshold.
const DISCOVERY_CHANCE: f32 = 0.1;

/// Faction relation penalty when a spy is caught.
const CAUGHT_RELATION_PENALTY: f32 = 20.0;

pub fn tick_espionage(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % ESPIONAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.spies.is_empty() {
        return;
    }

    // Collect spy updates to avoid borrow conflicts.
    // For each spy: (spy_index, caught, intel_delta)
    let mut updates: Vec<(usize, bool, f32)> = Vec::new();

    for (idx, spy) in state.spies.iter().enumerate() {
        // Look up adventurer level and archetype for modifiers
        let (adv_level, is_rogue) = state
            .adventurers
            .iter()
            .find(|a| a.id == spy.adventurer_id)
            .map(|a| (a.level, a.archetype.to_lowercase().contains("rogue")))
            .unwrap_or((1, false));

        // Intel gathering: base + level bonus + rogue bonus
        let level_modifier = 1.0 + (adv_level as f32 - 1.0) * 0.1; // +10% per level above 1
        let rogue_modifier = if is_rogue { 1.5 } else { 1.0 };
        let intel_delta = BASE_INTEL_PER_TICK * level_modifier * rogue_modifier;

        // Cover degradation
        let faction_hostile = state
            .factions
            .iter()
            .find(|f| f.id == spy.target_faction_id)
            .map(|f| {
                matches!(
                    f.diplomatic_stance,
                    DiplomaticStance::Hostile | DiplomaticStance::AtWar
                )
            })
            .unwrap_or(false);
        let cover_decay = if faction_hostile {
            HOSTILE_COVER_DECAY
        } else {
            BASE_COVER_DECAY
        };

        let new_cover = (spy.cover - cover_decay).max(0.0);

        // Discovery check
        let caught = if new_cover < DISCOVERY_THRESHOLD {
            let roll = lcg_f32(&mut state.rng);
            roll < DISCOVERY_CHANCE
        } else {
            false
        };

        updates.push((idx, caught, intel_delta));
    }

    // Apply updates (process in reverse to safely remove caught spies)
    let mut caught_spies: Vec<(u32, u32, usize)> = Vec::new(); // (spy_id, adventurer_id, faction_id)
    let mut intel_events: Vec<(u32, usize, f32)> = Vec::new(); // (spy_id, faction_id, total_intel)

    for &(idx, caught, intel_delta) in &updates {
        let spy = &mut state.spies[idx];

        if caught {
            caught_spies.push((spy.id, spy.adventurer_id, spy.target_faction_id));
        } else {
            // Apply cover decay
            let faction_hostile = state
                .factions
                .iter()
                .find(|f| f.id == spy.target_faction_id)
                .map(|f| {
                    matches!(
                        f.diplomatic_stance,
                        DiplomaticStance::Hostile | DiplomaticStance::AtWar
                    )
                })
                .unwrap_or(false);
            let cover_decay = if faction_hostile {
                HOSTILE_COVER_DECAY
            } else {
                BASE_COVER_DECAY
            };
            spy.cover = (spy.cover - cover_decay).max(0.0);

            // Gather intel
            spy.intel_gathered += intel_delta;

            intel_events.push((spy.id, spy.target_faction_id, spy.intel_gathered));
        }
    }

    // Process caught spies
    for (spy_id, adventurer_id, faction_id) in &caught_spies {
        // Remove the spy
        state.spies.retain(|s| s.id != *spy_id);

        // Mark adventurer as dead (imprisoned / removed)
        if let Some(adv) = state
            .adventurers
            .iter_mut()
            .find(|a| a.id == *adventurer_id)
        {
            adv.status = AdventurerStatus::Dead;
        }

        // Damage faction relations
        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == *faction_id) {
            faction.relationship_to_guild =
                (faction.relationship_to_guild - CAUGHT_RELATION_PENALTY).max(-100.0);
        }

        events.push(WorldEvent::SpyCaught {
            spy_id: *spy_id,
            adventurer_id: *adventurer_id,
            faction_id: *faction_id,
        });
    }

    // Emit intel gathered events (only for spies that weren't caught)
    for (spy_id, faction_id, total_intel) in intel_events {
        events.push(WorldEvent::IntelGathered {
            spy_id,
            faction_id,
            total_intel,
        });
    }
}
