//! Plague vector propagation system — fires every 400 ticks.
//!
//! Disease spreads between population centers via trade routes and migration,
//! creating epidemic decision points. Active caravans and migration waves
//! accelerate spread. Guilds can quarantine trade routes (reducing income)
//! and deploy healers for medical response.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often plague vector propagation fires (in ticks).
const PLAGUE_VECTOR_INTERVAL: u64 = 13;

/// Base chance per tick of a new plague spawning.
const PLAGUE_SPAWN_CHANCE: f32 = 0.02;

/// Active caravan bonus to spread probability.
const CARAVAN_SPREAD_BONUS: f32 = 0.20;

/// Active migration bonus to spread probability.
const MIGRATION_SPREAD_BONUS: f32 = 0.30;

/// Disease names for random plague generation.
const PLAGUE_NAMES: &[&str] = &[
    "Crimson Pox",
    "Ashlung",
    "Weeping Rot",
    "Blightvein",
    "Hollow Fever",
    "Marrow Wilt",
    "Pale Consumption",
    "Ironblood Plague",
];

/// Tick the plague vector system. Called every tick, but only processes every
/// `PLAGUE_VECTOR_INTERVAL` ticks.
pub fn tick_plague_vectors(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % PLAGUE_VECTOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Possibly spawn a new plague ---
    try_spawn_plague(state, events);

    // --- Phase 2: Propagate existing plagues to adjacent regions ---
    propagate_plagues(state, events);

    // --- Phase 3: Apply mortality (population loss in infected regions) ---
    apply_mortality(state, events);

    // --- Phase 4: Apply containment from healers in regions ---
    apply_medical_response(state);

    // --- Phase 5: Resolve contained plagues ---
    resolve_plagues(state, events);
}

/// Roll for a new plague spawning somewhere in the world.
fn try_spawn_plague(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let roll = lcg_f32(&mut state.rng);
    if roll >= PLAGUE_SPAWN_CHANCE {
        return;
    }

    // Pick a random region as origin
    let num_regions = state.overworld.regions.len();
    if num_regions == 0 {
        return;
    }
    let origin_idx = (lcg_next(&mut state.rng) as usize) % num_regions;
    let origin_id = state.overworld.regions[origin_idx].id as u32;

    // Don't spawn if this region already has an active plague vector
    if state
        .plague_vectors
        .iter()
        .any(|pv| pv.infected_regions.contains(&origin_id))
    {
        return;
    }

    // Random virulence [0.1, 0.8] and mortality [0.05, 0.3]
    let virulence = 0.1 + lcg_f32(&mut state.rng) * 0.7;
    let mortality = 0.05 + lcg_f32(&mut state.rng) * 0.25;

    let name_idx = (lcg_next(&mut state.rng) as usize) % PLAGUE_NAMES.len();
    let disease_name = PLAGUE_NAMES[name_idx].to_string();

    let pv = PlagueVector {
        disease_name: disease_name.clone(),
        origin_region: origin_id,
        infected_regions: vec![origin_id],
        virulence,
        mortality,
        tick_started: state.tick as u32,
    };
    state.plague_vectors.push(pv);

    events.push(WorldEvent::PlagueSpread {
        disease_name,
        from_region: origin_id,
        to_region: origin_id,
    });
}

/// Propagate each plague to adjacent regions based on trade volume and migration.
fn propagate_plagues(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Pre-compute: which region pairs have active caravans
    let caravan_region_pairs: Vec<(usize, usize)> = state
        .trade_routes
        .iter()
        .filter(|r| r.active)
        .filter(|r| state.caravans.iter().any(|c| c.route_id == r.id))
        .map(|r| (r.start_location_id, r.end_location_id))
        .collect();

    // Pre-compute: which region pairs have active migrations
    let migration_region_pairs: Vec<(usize, usize)> = state
        .migrations
        .iter()
        .filter(|m| m.progress < 1.0)
        .map(|m| (m.origin_region_id, m.destination_region_id))
        .collect();

    // Count total active trade routes for trade_volume_ratio denominator
    let total_active_routes = state.trade_routes.iter().filter(|r| r.active).count().max(1);

    // Collect spread targets to avoid borrow issues
    let mut spreads: Vec<(usize, u32, u32)> = Vec::new(); // (plague_idx, from_region, to_region)

    for (pidx, pv) in state.plague_vectors.iter().enumerate() {
        let current_regions = pv.infected_regions.clone();
        for &infected_region_id in &current_regions {
            let infected_idx = infected_region_id as usize;
            if infected_idx >= state.overworld.regions.len() {
                continue;
            }
            let neighbors = state.overworld.regions[infected_idx].neighbors.clone();

            // Count trade routes touching this region for trade_volume_ratio
            let routes_touching = state
                .trade_routes
                .iter()
                .filter(|r| {
                    r.active
                        && (r.start_location_id == infected_idx
                            || r.end_location_id == infected_idx)
                })
                .count();
            let trade_volume_ratio = routes_touching as f32 / total_active_routes as f32;

            for &neighbor_id in &neighbors {
                let neighbor_u32 = neighbor_id as u32;
                if pv.infected_regions.contains(&neighbor_u32) {
                    continue; // Already infected
                }
                if neighbor_id >= state.overworld.regions.len() {
                    continue;
                }

                // Sanitation approximation: higher morale + lower unrest = better sanitation
                let neighbor = &state.overworld.regions[neighbor_id];
                let sanitation_level =
                    (neighbor.civilian_morale / 100.0 * 0.7 + (1.0 - neighbor.unrest / 100.0) * 0.3)
                        .clamp(0.0, 1.0);

                // Base spread probability
                let mut spread_prob =
                    pv.virulence * trade_volume_ratio * (1.0 - sanitation_level);

                // Caravan bonus: active caravan between these regions
                let has_caravan = caravan_region_pairs.iter().any(|&(a, b)| {
                    (a == infected_idx && b == neighbor_id)
                        || (b == infected_idx && a == neighbor_id)
                });
                if has_caravan {
                    spread_prob += CARAVAN_SPREAD_BONUS;
                }

                // Migration bonus: active migration between these regions
                let has_migration = migration_region_pairs.iter().any(|&(a, b)| {
                    (a == infected_idx && b == neighbor_id)
                        || (b == infected_idx && a == neighbor_id)
                });
                if has_migration {
                    spread_prob += MIGRATION_SPREAD_BONUS;
                }

                let roll = lcg_f32(&mut state.rng);
                if roll < spread_prob.clamp(0.0, 1.0) {
                    spreads.push((pidx, infected_region_id, neighbor_u32));
                }
            }
        }
    }

    // Apply spreads
    for (pidx, from_region, to_region) in spreads {
        if pidx < state.plague_vectors.len() {
            let pv = &mut state.plague_vectors[pidx];
            if !pv.infected_regions.contains(&to_region) {
                pv.infected_regions.push(to_region);
                events.push(WorldEvent::PlagueSpread {
                    disease_name: pv.disease_name.clone(),
                    from_region,
                    to_region,
                });
            }
        }
    }
}

/// Apply mortality: each infected region loses population proportional to
/// mortality * (1.0 - medical_capacity).
fn apply_mortality(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect mortality data first
    let mut region_mortality: Vec<(usize, f32)> = Vec::new(); // (region_id_as_idx, mortality)

    for pv in &state.plague_vectors {
        for &region_id in &pv.infected_regions {
            let idx = region_id as usize;
            if idx < state.overworld.regions.len() {
                region_mortality.push((idx, pv.mortality));
            }
        }
    }

    // Medical capacity approximation: infirmary tier provides base medical capacity.
    let infirmary_capacity = state.guild_buildings.infirmary as f32 * 0.15;

    // Check for healer adventurers in each region (approximate via party positions)
    let healer_archetypes = ["cleric", "healer", "priest", "druid", "shaman"];

    for (region_idx, mortality) in region_mortality {
        let region = &state.overworld.regions[region_idx];
        let region_id = region.id;

        // Count healers in this region (use same heuristic as disease.rs)
        let healers_present = state.parties.iter().filter(|p| {
            let party_region = find_closest_region(p.position, &state.overworld.regions);
            party_region == Some(region_id)
        }).flat_map(|p| &p.member_ids).filter(|&&mid| {
            state.adventurers.iter().find(|a| a.id == mid).map(|a| {
                healer_archetypes.iter().any(|h| a.archetype.to_lowercase().contains(h))
            }).unwrap_or(false)
        }).count();

        // Medical capacity = infirmary base + 0.1 per healer present, capped at 0.8
        let medical_capacity = (infirmary_capacity + healers_present as f32 * 0.1).min(0.8);

        let effective_mortality = mortality * (1.0 - medical_capacity);
        let pop = state.overworld.regions[region_idx].population;
        let deaths = (pop as f32 * effective_mortality * 0.01).round() as u32;

        if deaths > 0 {
            state.overworld.regions[region_idx].population =
                state.overworld.regions[region_idx].population.saturating_sub(deaths).max(10);

            // Morale penalty from plague deaths
            state.overworld.regions[region_idx].civilian_morale =
                (state.overworld.regions[region_idx].civilian_morale - 2.0).max(0.0);

            events.push(WorldEvent::PlagueDeaths {
                region_id: region_id as u32,
                deaths,
            });
        }
    }
}

/// Adventurers with healing skills in infected regions reduce mortality
/// (already factored in `apply_mortality`). Additionally, slowly increase
/// containment by reducing virulence of plagues in regions with healers.
fn apply_medical_response(state: &mut CampaignState) {
    let healer_archetypes = ["cleric", "healer", "priest", "druid", "shaman"];

    for pv in &mut state.plague_vectors {
        let mut total_healer_power: f32 = 0.0;

        for &region_id in &pv.infected_regions {
            let idx = region_id as usize;
            if idx >= state.overworld.regions.len() {
                continue;
            }

            // Count healers present in this region
            let healers = state.parties.iter().filter(|p| {
                let party_region = find_closest_region(p.position, &state.overworld.regions);
                party_region == Some(idx)
            }).flat_map(|p| &p.member_ids).filter(|&&mid| {
                state.adventurers.iter().find(|a| a.id == mid).map(|a| {
                    healer_archetypes.iter().any(|h| a.archetype.to_lowercase().contains(h))
                }).unwrap_or(false)
            }).count();

            total_healer_power += healers as f32 * 0.02;
        }

        // Healers slowly reduce virulence
        if total_healer_power > 0.0 {
            pv.virulence = (pv.virulence - total_healer_power).max(0.0);
        }
    }
}

/// Resolve plagues that have been contained (virulence dropped to 0) or
/// that have burned out (all infected regions depopulated below threshold).
fn resolve_plagues(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut contained_names: Vec<String> = Vec::new();

    state.plague_vectors.retain(|pv| {
        // Contained: virulence reached 0
        if pv.virulence <= 0.0 {
            contained_names.push(pv.disease_name.clone());
            return false;
        }

        // Burned out: ran for over 4000 ticks
        let elapsed = state.tick as u32 - pv.tick_started;
        if elapsed > 4000 {
            contained_names.push(pv.disease_name.clone());
            return false;
        }

        true
    });

    for disease_name in contained_names {
        events.push(WorldEvent::PlagueContained { disease_name });
    }
}

/// Find the closest region to a position (reuses same heuristic as disease.rs).
fn find_closest_region(pos: (f32, f32), regions: &[Region]) -> Option<usize> {
    if regions.is_empty() {
        return None;
    }
    let hash = ((pos.0 * 7.0 + pos.1 * 13.0).abs() as usize) % regions.len();
    Some(regions[hash].id)
}
