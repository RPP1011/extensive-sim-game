//! Population and civilian morale system — every 100 ticks (~10s).
//!
//! Tracks regional population growth, civilian morale drift, tax income,
//! and population-driven events (boom towns, depopulation, unrest, prosperity).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{lcg_f32, CampaignState};

/// Tick population dynamics for all regions. Fires every 100 ticks.
pub fn tick_population(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 3 != 0 {
        return;
    }

    let guild_fid = state.diplomacy.guild_faction_id;

    // Check if any region has active combat (battles in progress).
    let regions_with_combat: Vec<usize> = state
        .active_battles
        .iter()
        .filter_map(|b| {
            // Find the quest's target region (approximate: use party position)
            state
                .active_quests
                .iter()
                .find(|q| q.id == b.quest_id)
                .and_then(|q| q.dispatched_party_id)
                .and_then(|pid| state.parties.iter().find(|p| p.id == pid))
                .and_then(|_party| {
                    // Map party to region by checking which region it's in
                    // Simplification: use quest's target location's region
                    None::<usize>
                })
        })
        .collect();

    // Average building tier for morale bonus (guild buildings benefit guild regions).
    let avg_building_tier = {
        let b = &state.guild_buildings;
        (b.training_grounds as f32
            + b.watchtower as f32
            + b.trade_post as f32
            + b.barracks as f32
            + b.infirmary as f32
            + b.war_room as f32)
            / 6.0
    };

    // Collect active crisis count for growth penalty.
    let has_crisis = !state.overworld.active_crises.is_empty();

    let num_regions = state.overworld.regions.len();
    for i in 0..num_regions {
        let region = &state.overworld.regions[i];
        let is_guild = region.owner_faction_id == guild_fid;
        let region_id = region.id;
        let region_name = region.name.clone();

        // --- Morale drift ---
        let mut morale_delta: f32 = 0.0;

        // No combat bonus
        if !regions_with_combat.contains(&region_id) {
            morale_delta += 0.5;
        }

        // High tax penalty
        if region.tax_rate > 0.3 {
            morale_delta -= 1.0;
        }

        // High unrest penalty
        if region.unrest > 50.0 {
            morale_delta -= 2.0;
        }

        // Building tier bonus (guild regions only)
        if is_guild {
            morale_delta += 0.2 * avg_building_tier;
        }

        let old_morale = region.civilian_morale;
        let new_morale = (old_morale + morale_delta).clamp(0.0, 100.0);

        // --- Population growth ---
        let mut growth_pct: f32 = 0.001; // base 0.1%

        if new_morale > 60.0 {
            growth_pct *= 1.5; // +50% if happy
        }
        if new_morale < 30.0 {
            growth_pct *= 0.5; // -50% if miserable
        }
        if has_crisis {
            growth_pct -= 0.01; // -1% during crisis
        }
        if region.threat_level > 50.0 {
            growth_pct -= 0.005; // -0.5% high threat
        }

        let old_pop = region.population;
        let pop_change = (old_pop as f32 * growth_pct).round() as i32;
        let new_pop = (old_pop as i32 + pop_change).max(0) as u32;

        // --- Tax income (guild-controlled regions only) ---
        if is_guild {
            let tax_income = new_pop as f32 * region.tax_rate * 0.001;
            if tax_income > 0.0 {
                state.guild.gold += tax_income;
                events.push(WorldEvent::TaxCollected {
                    region_id,
                    amount: tax_income,
                });
            }
        }

        // Apply changes
        let region = &mut state.overworld.regions[i];
        region.civilian_morale = new_morale;
        region.population = new_pop;
        region.growth_rate = growth_pct;

        // --- Population events ---
        // BoomTown: population exceeds 800
        if new_pop > 800 && old_pop <= 800 {
            events.push(WorldEvent::PopulationEvent {
                region_id,
                event_type: "BoomTown".into(),
                description: format!(
                    "{} is booming! Population has grown to {}.",
                    region_name, new_pop
                ),
            });
        }

        // Depopulated: population drops below 50
        if new_pop < 50 && old_pop >= 50 {
            events.push(WorldEvent::PopulationEvent {
                region_id,
                event_type: "Depopulated".into(),
                description: format!(
                    "{} has been nearly abandoned. Only {} civilians remain.",
                    region_name, new_pop
                ),
            });
        }

        // Unrest / rebellion: morale drops below 20
        if new_morale < 20.0 && old_morale >= 20.0 {
            // Small random chance of rebellion escalation
            let roll = lcg_f32(&mut state.rng);
            let event_type = if roll < 0.3 {
                "Rebellion"
            } else {
                "Unrest"
            };
            events.push(WorldEvent::PopulationEvent {
                region_id,
                event_type: event_type.into(),
                description: format!(
                    "Civilian morale in {} has collapsed to {:.0}. {} spreads!",
                    region_name, new_morale, event_type
                ),
            });
            // Rebellion increases region unrest
            if event_type == "Rebellion" {
                state.overworld.regions[i].unrest =
                    (state.overworld.regions[i].unrest + 15.0).min(100.0);
            }
        }

        // Prosperous: morale exceeds 80
        if new_morale > 80.0 && old_morale <= 80.0 {
            events.push(WorldEvent::PopulationEvent {
                region_id,
                event_type: "Prosperous".into(),
                description: format!(
                    "{} is thriving! Civilian morale has reached {:.0}.",
                    region_name, new_morale
                ),
            });
        }
    }
}
