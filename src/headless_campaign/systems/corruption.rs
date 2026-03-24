//! Corruption system — guild operations degrade over time.
//!
//! Officials siphon gold, quest rewards get skimmed, recruits are lower quality.
//! Players must actively investigate, purge, or appoint overseers to fight corruption.
//! Fires every 200 ticks (~20s game time).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often corruption ticks (in ticks).
const CORRUPTION_INTERVAL: u64 = 200;

/// Tick the corruption system. Called every tick; internally gates on interval.
pub fn tick_corruption(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CORRUPTION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Natural corruption growth ---
    let mut growth = 0.3;

    // Wealth attracts corruption: +0.2 per 100 gold
    growth += (state.guild.gold / 100.0) * 0.2;

    // Large roster: +0.1 per 5 adventurers
    let alive_count = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count() as f32;
    growth += (alive_count / 5.0) * 0.1;

    // Low average morale accelerates corruption
    let morale_sum: f32 = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| a.morale)
        .sum();
    let avg_morale = if alive_count > 0.0 {
        morale_sum / alive_count
    } else {
        50.0
    };
    if avg_morale < 50.0 {
        // +0.1 per 10 points below 50
        growth += ((50.0 - avg_morale) / 10.0) * 0.1;
    }

    // Overseer reduces corruption: -1 per tick while assigned
    if let Some(overseer_id) = state.corruption.overseer_id {
        // Check the overseer is still alive and in the guild
        let overseer_alive = state
            .adventurers
            .iter()
            .any(|a| a.id == overseer_id && a.status != AdventurerStatus::Dead);
        if overseer_alive {
            growth -= 1.0;
        } else {
            // Overseer died or deserted — remove assignment
            state.corruption.overseer_id = None;
        }
    }

    state.corruption.level = (state.corruption.level + growth).clamp(0.0, 100.0);

    // --- Threshold effects ---
    apply_corruption_effects(state, events);
}

/// Apply corruption effects based on the current corruption level.
fn apply_corruption_effects(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    let level = state.corruption.level;

    // Gold siphoning (tracked for reporting, actual skimming happens at quest completion
    // via the `corruption_gold_multiplier` helper)

    // At 40+: 5% chance supplies go missing
    if level >= 40.0 {
        let roll = lcg_f32(&mut state.rng);
        if roll < 0.05 {
            let supply_loss = (state.guild.supplies * 0.1).min(15.0).max(2.0);
            state.guild.supplies = (state.guild.supplies - supply_loss).max(0.0);
            events.push(WorldEvent::RandomEvent {
                name: "Supplies Missing".to_string(),
                description: format!(
                    "Corruption in the guild — {:.0} supplies have gone missing.",
                    supply_loss
                ),
            });
            events.push(WorldEvent::SupplyChanged {
                amount: -supply_loss,
                reason: "Corruption: supplies gone missing".to_string(),
            });
        }
    }

    // At 60+: 10% chance embezzlement event
    if level >= 60.0 {
        let roll = lcg_f32(&mut state.rng);
        if roll < 0.10 {
            let gold_lost = 20.0 + (lcg_next(&mut state.rng) % 31) as f32; // 20-50
            let actual_loss = gold_lost.min(state.guild.gold);
            state.guild.gold = (state.guild.gold - actual_loss).max(0.0);
            state.corruption.gold_siphoned += actual_loss;

            events.push(WorldEvent::EmbezzlementDiscovered {
                gold_lost: actual_loss,
            });
            events.push(WorldEvent::GoldChanged {
                amount: -actual_loss,
                reason: "Embezzlement discovered".to_string(),
            });
        }

        // Morale penalty at 60+
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - 2.0).max(0.0);
            }
        }
    }

    // At 80+: faction relations damaged, desertion risk
    if level >= 80.0 {
        // Reputation hit (guild seen as corrupt)
        let rep_loss = 1.0;
        state.guild.reputation = (state.guild.reputation - rep_loss).max(0.0);

        // Scandal chance: 5%
        let roll = lcg_f32(&mut state.rng);
        if roll < 0.05 {
            let reputation_lost = 5.0 + (lcg_next(&mut state.rng) % 6) as f32; // 5-10
            state.guild.reputation = (state.guild.reputation - reputation_lost).max(0.0);

            // Damage faction relations
            for faction in &mut state.factions {
                faction.relationship_to_guild =
                    (faction.relationship_to_guild - 3.0).max(-100.0);
            }

            events.push(WorldEvent::CorruptionScandal {
                reputation_lost,
            });
        }

        // Desertion risk: 3% per living adventurer with loyalty < 40
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead && adv.loyalty < 40.0 {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.03 {
                    let id = adv.id;
                    let name = adv.name.clone();
                    adv.status = AdventurerStatus::Dead;
                    events.push(WorldEvent::AdventurerDeserted {
                        adventurer_id: id,
                        reason: format!(
                            "{} deserted — disgusted by guild corruption.",
                            name
                        ),
                    });
                }
            }
        }
    }
}

/// Returns the gold reward multiplier based on corruption level.
/// Used by quest reward systems to skim gold.
pub fn corruption_gold_multiplier(corruption_level: f32) -> f32 {
    if corruption_level >= 80.0 {
        0.70 // -30%
    } else if corruption_level >= 60.0 {
        0.80 // -20%
    } else if corruption_level >= 40.0 {
        0.90 // -10%
    } else if corruption_level >= 20.0 {
        0.95 // -5%
    } else {
        1.0
    }
}

/// Returns the recruitment quality penalty based on corruption level (0.0 to 0.15).
pub fn corruption_recruitment_penalty(corruption_level: f32) -> f32 {
    if corruption_level >= 60.0 {
        0.15
    } else if corruption_level >= 40.0 {
        0.10
    } else {
        0.0
    }
}
