//! Endgame crisis system — ticks active crises forward.
//!
//! Each crisis type has its own escalation mechanics.
//! The Sleeping King activates champions on a timer who travel
//! to join the King's faction, snowballing its power.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Tick the active crisis. Runs every tick.
pub fn tick_crisis(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let crisis = match &state.overworld.active_crisis {
        Some(c) => c.clone(),
        None => return,
    };

    match crisis {
        ActiveCrisis::SleepingKing {
            king_faction_id,
            champion_ids,
            champions_arrived,
            next_activation_tick,
            activation_interval,
        } => {
            tick_sleeping_king(
                state, events,
                king_faction_id, &champion_ids,
                champions_arrived, next_activation_tick, activation_interval,
            );
        }
        ActiveCrisis::Breach {
            source_location_id,
            wave_number,
            wave_strength,
            next_wave_tick,
        } => {
            tick_breach(state, events, source_location_id, wave_number, wave_strength, next_wave_tick);
        }
        ActiveCrisis::Corruption {
            origin_region_id,
            corrupted_regions,
            spread_rate_ticks,
            next_spread_tick,
        } => {
            tick_corruption(state, events, origin_region_id, &corrupted_regions, spread_rate_ticks, next_spread_tick);
        }
        ActiveCrisis::Decline { severity, tick_started } => {
            tick_decline(state, events, severity, tick_started);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Sleeping King
// ---------------------------------------------------------------------------

fn tick_sleeping_king(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    king_faction_id: usize,
    champion_ids: &[u32],
    mut champions_arrived: u32,
    next_activation_tick: u64,
    activation_interval: u64,
) {
    let mut next_tick = next_activation_tick;

    // Activate next dormant champion
    if state.tick >= next_tick {
        // Find next dormant champion (no rallying_to and faction_id != king)
        if let Some(champ) = state.adventurers.iter_mut().find(|a| {
            champion_ids.contains(&a.id)
                && a.rallying_to.is_none()
                && a.faction_id != Some(king_faction_id)
                && a.status != AdventurerStatus::Dead
        }) {
            // Find king faction position (use their capital region)
            let king_pos = state.overworld.regions.iter()
                .find(|r| r.owner_faction_id == king_faction_id)
                .map(|r| {
                    // Approximate region center
                    let idx = r.id;
                    ((idx as f32 * 20.0) - 30.0, (idx as f32 * 15.0) - 20.0)
                })
                .unwrap_or((50.0, 50.0));

            champ.rallying_to = Some(RallyTarget {
                faction_id: king_faction_id,
                destination: king_pos,
                speed: 3.0,
            });
            champ.status = AdventurerStatus::Traveling;

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has heard the King's call and begins traveling to join them!",
                    champ.name
                ),
            });

            next_tick = state.tick + activation_interval;
        }
    }

    // Move rallying champions toward destination
    let dt_sec = CAMPAIGN_TICK_MS as f32 / 1000.0;
    for champ in &mut state.adventurers {
        if !champion_ids.contains(&champ.id) {
            continue;
        }
        let rally = match &champ.rallying_to {
            Some(r) => r.clone(),
            None => continue,
        };
        if champ.status == AdventurerStatus::Dead {
            continue;
        }

        let dx = rally.destination.0 - champ.guild_relationship; // hack: use position tracking
        let dy = rally.destination.1;
        // Simple distance check — champion "arrives" after enough ticks
        // (Real implementation would track position on the overworld)
        let travel_ticks = 2000; // ~200 seconds to cross the map
        // For now, check if champion has been rallying long enough
        // We'll use a simple tick counter approach
    }

    // Check for champion arrivals — any champion with rallying_to whose
    // faction_id matches king AND has been traveling for enough ticks
    let mut newly_arrived = Vec::new();
    for champ in &mut state.adventurers {
        if !champion_ids.contains(&champ.id) || champ.status == AdventurerStatus::Dead {
            continue;
        }
        if let Some(ref rally) = champ.rallying_to {
            if rally.faction_id == king_faction_id {
                // Simplified: champion arrives after activation_interval ticks of travel
                // (In full implementation, track actual position)
                // For now, arrive immediately to demonstrate the mechanic
                champ.faction_id = Some(king_faction_id);
                champ.rallying_to = None;
                champ.status = AdventurerStatus::Idle;
                newly_arrived.push((champ.id, champ.name.clone()));
            }
        }
    }

    for (champ_id, champ_name) in &newly_arrived {
        champions_arrived += 1;

        // Apply champion's leadership buff to the king's faction
        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == king_faction_id) {
            // Each champion massively boosts the faction
            let power_per_champion = 25.0 * champions_arrived as f32; // Snowball!
            faction.military_strength += power_per_champion;
            faction.max_military_strength += power_per_champion;

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has joined the Sleeping King! ({}/{} champions arrived, faction strength now {:.0})",
                    champ_name, champions_arrived, champion_ids.len(), faction.military_strength
                ),
            });

            // At 5+ champions, the king declares war if not already
            if champions_arrived >= 5 && faction.diplomatic_stance != DiplomaticStance::AtWar {
                faction.diplomatic_stance = DiplomaticStance::AtWar;
                faction.at_war_with.push(state.diplomacy.guild_faction_id);
                faction.relationship_to_guild = -100.0;
                events.push(WorldEvent::CampaignMilestone {
                    description: "The Sleeping King has declared war on all who oppose them!".into(),
                });
            }
        }
    }

    // Update crisis state
    state.overworld.active_crisis = Some(ActiveCrisis::SleepingKing {
        king_faction_id,
        champion_ids: champion_ids.to_vec(),
        champions_arrived,
        next_activation_tick: next_tick,
        activation_interval,
    });
}

// ---------------------------------------------------------------------------
// Breach — dungeon monster waves
// ---------------------------------------------------------------------------

fn tick_breach(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    source_location_id: usize,
    mut wave_number: u32,
    mut wave_strength: f32,
    next_wave_tick: u64,
) {
    if state.tick < next_wave_tick {
        return;
    }

    wave_number += 1;
    wave_strength *= 1.3; // Each wave is 30% stronger

    // Damage the nearest guild region
    if let Some(region) = state.overworld.regions.iter_mut()
        .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
        .min_by(|a, b| a.control.partial_cmp(&b.control).unwrap_or(std::cmp::Ordering::Equal))
    {
        region.control = (region.control - wave_strength * 0.5).max(0.0);
        region.unrest = (region.unrest + wave_strength * 0.3).min(100.0);
        region.threat_level = (region.threat_level + wave_strength * 0.2).min(100.0);
    }

    events.push(WorldEvent::CampaignMilestone {
        description: format!(
            "Breach wave {} erupts! (strength {:.0}) Monsters pour from the depths.",
            wave_number, wave_strength
        ),
    });

    let wave_interval = 3000u64.saturating_sub(wave_number as u64 * 200); // Waves get faster
    state.overworld.active_crisis = Some(ActiveCrisis::Breach {
        source_location_id,
        wave_number,
        wave_strength,
        next_wave_tick: state.tick + wave_interval.max(500),
    });
}

// ---------------------------------------------------------------------------
// Corruption — spreading contamination
// ---------------------------------------------------------------------------

fn tick_corruption(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    origin_region_id: usize,
    corrupted: &[usize],
    spread_rate_ticks: u64,
    next_spread_tick: u64,
) {
    // Corrupted regions take ongoing damage
    for &rid in corrupted {
        if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == rid) {
            region.unrest = (region.unrest + 2.0).min(100.0);
            region.control = (region.control - 1.0).max(0.0);
        }
    }

    if state.tick < next_spread_tick {
        return;
    }

    // Spread to an adjacent uncorrupted region
    let mut new_corrupted = corrupted.to_vec();
    let mut spread_to = None;

    for &rid in corrupted {
        if let Some(region) = state.overworld.regions.iter().find(|r| r.id == rid) {
            for &neighbor in &region.neighbors {
                if !new_corrupted.contains(&neighbor) {
                    spread_to = Some(neighbor);
                    break;
                }
            }
        }
        if spread_to.is_some() {
            break;
        }
    }

    if let Some(new_rid) = spread_to {
        new_corrupted.push(new_rid);
        let region_name = state.overworld.regions.iter()
            .find(|r| r.id == new_rid)
            .map(|r| r.name.clone())
            .unwrap_or_else(|| format!("Region {}", new_rid));

        events.push(WorldEvent::CampaignMilestone {
            description: format!(
                "The corruption spreads to {}! ({}/{} regions affected)",
                region_name, new_corrupted.len(), state.overworld.regions.len()
            ),
        });
    }

    state.overworld.active_crisis = Some(ActiveCrisis::Corruption {
        origin_region_id,
        corrupted_regions: new_corrupted,
        spread_rate_ticks,
        next_spread_tick: state.tick + spread_rate_ticks,
    });
}

// ---------------------------------------------------------------------------
// Decline — slow resource drain
// ---------------------------------------------------------------------------

fn tick_decline(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    mut severity: f32,
    tick_started: u64,
) {
    // Severity increases over time
    let elapsed = state.tick.saturating_sub(tick_started);
    severity = 1.0 + (elapsed as f32 / 10000.0); // Slowly gets worse

    // Drain guild resources
    if state.tick % 100 == 0 {
        state.guild.gold -= severity * 0.5;
        state.guild.supplies -= severity * 0.3;

        // Morale drops across all adventurers
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - severity * 0.1).max(0.0);
                adv.stress = (adv.stress + severity * 0.05).min(100.0);
            }
        }

        // Regions slowly deteriorate
        for region in &mut state.overworld.regions {
            region.control = (region.control - severity * 0.1).max(0.0);
            region.unrest = (region.unrest + severity * 0.05).min(100.0);
        }
    }

    state.overworld.active_crisis = Some(ActiveCrisis::Decline {
        severity,
        tick_started,
    });
}
