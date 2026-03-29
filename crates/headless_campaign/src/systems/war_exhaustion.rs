//! War exhaustion system — every 100 ticks (~10s).
//!
//! Prolonged wars drain morale, gold, and public support. Factions accumulate
//! exhaustion from time at war, casualties, and gold spent. At threshold levels
//! (25/50/75/90) escalating penalties apply: combat debuffs, morale collapse,
//! forced peace-seeking, and eventual faction collapse with auto-ceasefire.
//!
//! Guild war exhaustion tracks the guild's own involvement in wars and penalizes
//! adventurer morale and recruitment difficulty.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Cadence: every 100 ticks.
const WAR_EXHAUSTION_INTERVAL: u64 = 3;

/// Base exhaustion gain per tick while at war.
const BASE_EXHAUSTION_PER_TICK: f32 = 0.5;
/// Extra exhaustion per casualty suffered.
const EXHAUSTION_PER_CASUALTY: f32 = 2.0;
/// Extra exhaustion per gold spent on war.
const EXHAUSTION_PER_GOLD: f32 = 0.1;
/// Exhaustion decay per tick while at peace.
const PEACE_DECAY_PER_TICK: f32 = 1.0;

/// Morale penalty applied to adventurers per 25 exhaustion (guild wars only).
const GUILD_MORALE_PENALTY_PER_25: f32 = 0.5;

pub fn tick_war_exhaustion(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % WAR_EXHAUSTION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let guild_faction_id = state.diplomacy.guild_faction_id;
    let n_factions = state.factions.len();

    // --- Process each faction ---
    for fi in 0..n_factions {
        let at_war = state.factions[fi].diplomatic_stance == DiplomaticStance::AtWar
            || !state.factions[fi].at_war_with.is_empty();

        // Find or create exhaustion entry for this faction
        let entry_idx = state.war_exhaustion.iter().position(|e| e.faction_id == fi);

        if at_war {
            let entry_idx = if let Some(idx) = entry_idx {
                idx
            } else {
                // New war — create entry
                state.war_exhaustion.push(WarExhaustion {
                    faction_id: fi,
                    exhaustion_level: 0.0,
                    war_start_tick: state.tick,
                    casualties: 0,
                    gold_spent: 0.0,
                });
                state.war_exhaustion.len() - 1
            };

            let prev_level = state.war_exhaustion[entry_idx].exhaustion_level;

            // Accumulate exhaustion
            let casualties = state.war_exhaustion[entry_idx].casualties;
            let gold_spent = state.war_exhaustion[entry_idx].gold_spent;

            // Per-tick base + contribution from cumulative casualties and gold
            // We use the delta since last check: casualties and gold_spent are
            // cumulative, so we compute the marginal contribution.
            let exhaustion_gain = BASE_EXHAUSTION_PER_TICK
                + casualties as f32 * EXHAUSTION_PER_CASUALTY * 0.01  // Diminishing per-tick contribution
                + gold_spent * EXHAUSTION_PER_GOLD * 0.001;            // Diminishing per-tick contribution

            let entry = &mut state.war_exhaustion[entry_idx];
            entry.exhaustion_level = (entry.exhaustion_level + exhaustion_gain).min(100.0);
            let new_level = entry.exhaustion_level;

            // Check threshold crossings and apply effects
            check_thresholds(state, fi, prev_level, new_level, guild_faction_id, events);
        } else if let Some(idx) = entry_idx {
            // At peace — decay exhaustion
            let entry = &mut state.war_exhaustion[idx];
            entry.exhaustion_level = (entry.exhaustion_level - PEACE_DECAY_PER_TICK).max(0.0);

            // Remove entry if fully recovered
            if entry.exhaustion_level <= 0.0 {
                state.war_exhaustion.remove(idx);
            }
        }
    }

    // --- Guild war exhaustion ---
    // The guild itself may be at war (factions with AtWar stance toward guild).
    let guild_at_war = state
        .factions
        .iter()
        .any(|f| f.diplomatic_stance == DiplomaticStance::AtWar);

    let guild_entry_idx = state
        .war_exhaustion
        .iter()
        .position(|e| e.faction_id == guild_faction_id);

    if guild_at_war {
        let guild_idx = if let Some(idx) = guild_entry_idx {
            idx
        } else {
            state.war_exhaustion.push(WarExhaustion {
                faction_id: guild_faction_id,
                exhaustion_level: 0.0,
                war_start_tick: state.tick,
                casualties: 0,
                gold_spent: 0.0,
            });
            state.war_exhaustion.len() - 1
        };

        let prev_level = state.war_exhaustion[guild_idx].exhaustion_level;
        let casualties = state.war_exhaustion[guild_idx].casualties;
        let gold_spent = state.war_exhaustion[guild_idx].gold_spent;

        let exhaustion_gain = BASE_EXHAUSTION_PER_TICK
            + casualties as f32 * EXHAUSTION_PER_CASUALTY * 0.01
            + gold_spent * EXHAUSTION_PER_GOLD * 0.001;

        state.war_exhaustion[guild_idx].exhaustion_level =
            (state.war_exhaustion[guild_idx].exhaustion_level + exhaustion_gain).min(100.0);
        let new_level = state.war_exhaustion[guild_idx].exhaustion_level;

        // Apply adventurer morale penalty based on guild exhaustion
        let morale_penalty = (new_level / 25.0).floor() * GUILD_MORALE_PENALTY_PER_25;
        if morale_penalty > 0.0 {
            for adv in &mut state.adventurers {
                // Only affect guild adventurers (no faction_id)
                if adv.faction_id.is_none() {
                    adv.morale = (adv.morale - morale_penalty).max(0.0);
                }
            }
        }

        // Emit milestone events for the guild faction too
        check_guild_thresholds(prev_level, new_level, events);
    } else if let Some(idx) = guild_entry_idx {
        state.war_exhaustion[idx].exhaustion_level =
            (state.war_exhaustion[idx].exhaustion_level - PEACE_DECAY_PER_TICK).max(0.0);
        if state.war_exhaustion[idx].exhaustion_level <= 0.0 {
            state.war_exhaustion.remove(idx);
        }
    }
}

/// Record a casualty for a faction's war exhaustion tracker.
/// Called externally when a battle produces casualties.
pub fn record_casualty(state: &mut CampaignState, faction_id: usize, count: u32) {
    if let Some(entry) = state
        .war_exhaustion
        .iter_mut()
        .find(|e| e.faction_id == faction_id)
    {
        entry.casualties += count;
    }
}

/// Record gold spent on war for a faction's exhaustion tracker.
pub fn record_gold_spent(state: &mut CampaignState, faction_id: usize, amount: f32) {
    if let Some(entry) = state
        .war_exhaustion
        .iter_mut()
        .find(|e| e.faction_id == faction_id)
    {
        entry.gold_spent += amount;
    }
}

/// Check exhaustion threshold crossings and apply faction effects.
fn check_thresholds(
    state: &mut CampaignState,
    fi: usize,
    prev: f32,
    new: f32,
    guild_faction_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    let name = state.factions[fi].name.clone();

    // --- 25: Minor combat penalty ---
    if prev < 25.0 && new >= 25.0 {
        // -10% combat effectiveness (reduce military strength proportionally)
        state.factions[fi].military_strength *= 0.90;
        events.push(WorldEvent::WarExhaustionMilestone {
            faction_id: fi,
            level: new,
            description: format!(
                "{} shows signs of war weariness — combat effectiveness reduced by 10%",
                name
            ),
        });
    }

    // --- 50: Morale crisis ---
    if prev < 50.0 && new >= 50.0 {
        // -20% combat effectiveness
        state.factions[fi].military_strength *= 0.80;

        // Civilian unrest +10 in all owned regions
        for region in &mut state.overworld.regions {
            if region.owner_faction_id == fi {
                region.unrest = (region.unrest + 10.0).min(100.0);
            }
        }

        events.push(WorldEvent::WarExhaustionMilestone {
            faction_id: fi,
            level: new,
            description: format!(
                "{} suffers morale crisis — combat down 20%, civilian unrest rising",
                name
            ),
        });
    }

    // --- 75: Seeks peace ---
    if prev < 75.0 && new >= 75.0 {
        events.push(WorldEvent::WarExhaustionMilestone {
            faction_id: fi,
            level: new,
            description: format!(
                "{} desperately seeks peace — willing to accept unfavorable terms",
                name
            ),
        });

        // Faction AI will use this level to drive ceasefire decisions.
        // Improve relationship slightly to make ceasefire more likely.
        state.factions[fi].relationship_to_guild =
            (state.factions[fi].relationship_to_guild + 15.0).min(100.0);
    }

    // --- 90: Collapse ---
    if prev < 90.0 && new >= 90.0 {
        // Auto-ceasefire
        state.factions[fi].diplomatic_stance = DiplomaticStance::Hostile;
        state.factions[fi].at_war_with.retain(|&id| id != guild_faction_id);

        // Massive strength loss
        state.factions[fi].military_strength *= 0.30;

        // Regions may defect — high-unrest regions flip to guild
        let regions_to_flip: Vec<usize> = state
            .overworld
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.owner_faction_id == fi && r.unrest > 50.0)
            .map(|(i, _)| i)
            .collect();

        for idx in &regions_to_flip {
            state.overworld.regions[*idx].owner_faction_id = guild_faction_id;
            state.overworld.regions[*idx].control = 30.0;
            state.overworld.regions[*idx].unrest = (state.overworld.regions[*idx].unrest - 20.0).max(0.0);
        }

        // Also end faction-to-faction wars
        for other_fi in 0..state.factions.len() {
            if other_fi == fi {
                continue;
            }
            state.factions[other_fi]
                .at_war_with
                .retain(|&id| id != fi);
        }
        state.factions[fi].at_war_with.clear();

        events.push(WorldEvent::WarExhaustionMilestone {
            faction_id: fi,
            level: new,
            description: format!(
                "{} collapses from war exhaustion — auto-ceasefire, {} regions defect",
                name,
                regions_to_flip.len()
            ),
        });
    }
}

/// Emit milestone events for guild war exhaustion (no faction effects, just events).
fn check_guild_thresholds(prev: f32, new: f32, events: &mut Vec<WorldEvent>) {
    let thresholds: &[(f32, &str)] = &[
        (25.0, "Guild members grow weary of war — morale declining"),
        (50.0, "War weariness spreads through the guild — recruitment suffers"),
        (
            75.0,
            "Guild on the brink — adventurers urge peace negotiations",
        ),
        (
            90.0,
            "Guild near collapse from war exhaustion — operations severely impaired",
        ),
    ];

    for &(threshold, desc) in thresholds {
        if prev < threshold && new >= threshold {
            events.push(WorldEvent::WarExhaustionMilestone {
                faction_id: usize::MAX, // Sentinel for guild
                level: new,
                description: desc.to_string(),
            });
        }
    }
}
