//! Disease and plague system — fires every 200 ticks.
//!
//! Diseases can outbreak in densely populated regions with low civilian morale,
//! spread along trade routes (region neighbors), reduce population and morale,
//! and infect adventurers in affected regions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use super::class_system::effective_noncombat_stats;

/// How often to tick the disease system (in ticks).
const DISEASE_INTERVAL: u64 = 200;

/// Base outbreak chance per qualifying region per tick.
const OUTBREAK_CHANCE: f32 = 0.02;

/// Population threshold above which outbreaks can occur.
const POPULATION_OUTBREAK_THRESHOLD: f32 = 500.0;

/// Morale threshold below which outbreaks can occur.
const MORALE_OUTBREAK_THRESHOLD: f32 = 40.0;

/// Per-adventurer infection chance when in an infected region.
const ADVENTURER_INFECTION_CHANCE: f32 = 0.10;

/// Healer archetypes reduce infection spread chance by this factor.
const HEALER_SPREAD_REDUCTION: f32 = 0.50;

/// How many ticks before a disease ends naturally (if not contained).
const DISEASE_NATURAL_DURATION: u64 = 3000;

/// Disease names for random generation.
const DISEASE_NAMES: &[&str] = &[
    "Red Pox",
    "Swamp Fever",
    "Bone Rot",
    "Iron Lung",
    "Shadow Blight",
    "Weeping Sores",
    "Grey Wasting",
    "Rattlecough",
];

/// Tick the disease system. Called every tick, but only processes every
/// `DISEASE_INTERVAL` ticks.
pub fn tick_disease(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % DISEASE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Check for new outbreaks ---
    try_outbreak(state, events);

    // --- Phase 2: Spread existing diseases to neighboring regions ---
    spread_diseases(state, events);

    // --- Phase 3: Apply disease effects on regions ---
    apply_region_effects(state);

    // --- Phase 4: Infect adventurers in diseased regions ---
    infect_adventurers(state, events);

    // --- Phase 4b: Medicine bonus increases containment ---
    // Guild adventurers with medicine stats help contain diseases faster
    let guild_medicine: f32 = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.faction_id.is_none())
        .map(|a| effective_noncombat_stats(a).3) // medicine component
        .sum();
    if guild_medicine > 0.0 {
        for disease in &mut state.diseases {
            disease.containment += guild_medicine * 0.03;
        }
    }

    // --- Phase 5: Check for containment / natural end ---
    resolve_diseases(state, events);
}

/// Roll for new disease outbreaks in qualifying regions.
fn try_outbreak(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect region IDs that qualify for outbreak
    let qualifying: Vec<usize> = state
        .overworld
        .regions
        .iter()
        .filter(|r| {
            (r.population as f32) > POPULATION_OUTBREAK_THRESHOLD
                && r.civilian_morale < MORALE_OUTBREAK_THRESHOLD
        })
        .filter(|r| {
            // Don't spawn another outbreak in a region already diseased
            !state
                .diseases
                .iter()
                .any(|d| d.affected_regions.contains(&r.id))
        })
        .map(|r| r.id)
        .collect();

    for region_id in qualifying {
        let roll = lcg_f32(&mut state.rng);
        if roll < OUTBREAK_CHANCE {
            let disease_id = state.next_disease_id;
            state.next_disease_id += 1;

            let name_idx = (lcg_next(&mut state.rng) as usize) % DISEASE_NAMES.len();
            let name = DISEASE_NAMES[name_idx].to_string();

            // Severity scales with population density and low morale
            let region = &state.overworld.regions[region_id];
            let severity_base = 20.0 + lcg_f32(&mut state.rng) * 40.0;
            let morale_factor = 1.0 + (MORALE_OUTBREAK_THRESHOLD - region.civilian_morale) / 100.0;
            let severity = (severity_base * morale_factor).clamp(10.0, 100.0);

            let spread_rate = 0.1 + lcg_f32(&mut state.rng) * 0.3;

            let disease = Disease {
                id: disease_id,
                name: name.clone(),
                severity,
                spread_rate,
                affected_regions: vec![region_id],
                started_tick: state.tick,
                containment: 0.0,
            };
            state.diseases.push(disease);

            events.push(WorldEvent::DiseaseOutbreak {
                disease_id,
                disease_name: name,
                region_id,
                severity,
            });
        }
    }
}

/// Spread diseases to neighboring regions along trade routes.
fn spread_diseases(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect spread targets first to avoid borrow issues
    let mut spreads: Vec<(usize, usize)> = Vec::new(); // (disease_idx, target_region_id)

    for (didx, disease) in state.diseases.iter().enumerate() {
        let current_regions = disease.affected_regions.clone();
        for &region_id in &current_regions {
            if region_id >= state.overworld.regions.len() {
                continue;
            }
            let neighbors = state.overworld.regions[region_id].neighbors.clone();
            for &neighbor_id in &neighbors {
                if disease.affected_regions.contains(&neighbor_id) {
                    continue; // Already infected
                }
                if neighbor_id >= state.overworld.regions.len() {
                    continue;
                }

                // Spread chance scales with spread_rate, population, and inverse containment
                let neighbor_pop = state.overworld.regions[neighbor_id].population;
                let pop_factor = (neighbor_pop as f32 / 500.0).clamp(0.1, 2.0);
                let containment_factor = 1.0 - (disease.containment / 100.0).clamp(0.0, 0.9);
                let spread_chance = disease.spread_rate * pop_factor * containment_factor;

                let roll = lcg_f32(&mut state.rng);
                if roll < spread_chance {
                    spreads.push((didx, neighbor_id));
                }
            }
        }
    }

    // Apply spreads
    for (didx, target_region) in spreads {
        if didx < state.diseases.len() {
            let disease = &mut state.diseases[didx];
            if !disease.affected_regions.contains(&target_region) {
                let from_region = disease.affected_regions[0]; // Use origin as "from"
                disease.affected_regions.push(target_region);
                events.push(WorldEvent::DiseaseSpread {
                    disease_id: disease.id,
                    from_region,
                    to_region: target_region,
                });
            }
        }
    }
}

/// Apply disease effects on affected regions: population loss, morale drop, trade penalty.
fn apply_region_effects(state: &mut CampaignState) {
    // Collect which regions are affected and at what severity
    let mut region_severity: Vec<(usize, f32)> = Vec::new();
    for disease in &state.diseases {
        for &region_id in &disease.affected_regions {
            region_severity.push((region_id, disease.severity));
        }
    }

    for (region_id, severity) in region_severity {
        if region_id >= state.overworld.regions.len() {
            continue;
        }
        let region = &mut state.overworld.regions[region_id];

        // Population loss: severity * 0.1% per disease tick
        let pop_loss = (region.population as f32 * severity * 0.001) as u32;
        region.population = region.population.saturating_sub(pop_loss).max(10);

        // Civilian morale penalty
        let morale_loss = severity * 0.05;
        region.civilian_morale = (region.civilian_morale - morale_loss).max(0.0);

        // Unrest increase from disease
        region.unrest = (region.unrest + severity * 0.02).min(100.0);
    }

    // Trade income penalty: reduce total_trade_income proportionally to diseased regions
    let total_regions = state.overworld.regions.len().max(1) as f32;
    let diseased_region_count = state
        .diseases
        .iter()
        .flat_map(|d| &d.affected_regions)
        .collect::<std::collections::HashSet<_>>()
        .len() as f32;
    let trade_penalty = diseased_region_count / total_regions * 0.1;
    state.guild.total_trade_income *= 1.0 - trade_penalty;
}

/// Infect adventurers who are in diseased regions.
fn infect_adventurers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Build a set of diseased region positions (approximate: use region index to match
    // parties by checking which region they're closest to)
    let diseased_regions: Vec<(u32, usize)> = state
        .diseases
        .iter()
        .flat_map(|d| d.affected_regions.iter().map(move |&r| (d.id, r)))
        .collect();

    if diseased_regions.is_empty() {
        return;
    }

    // Check which parties are in diseased regions.
    // We approximate by checking if the party is in a region that has a disease.
    // Parties on quests have target_positions that correspond to regions.
    let party_regions: Vec<(u32, Vec<u32>)> = state
        .parties
        .iter()
        .map(|p| (p.id, p.member_ids.clone()))
        .collect();

    // Check if any party member is a healer (reduces spread)
    let healer_archetypes = ["cleric", "healer", "priest", "druid", "shaman"];

    for (party_id, member_ids) in &party_regions {
        // Determine which region this party is in based on position
        let party = match state.parties.iter().find(|p| p.id == *party_id) {
            Some(p) => p,
            None => continue,
        };

        // Find closest region to party position
        let party_region = find_closest_region(party.position, &state.overworld.regions);
        let party_region = match party_region {
            Some(r) => r,
            None => continue,
        };

        // Check if this region has a disease
        let disease_id = diseased_regions
            .iter()
            .find(|(_, r)| *r == party_region)
            .map(|(d, _)| *d);

        let disease_id = match disease_id {
            Some(d) => d,
            None => continue,
        };

        // Check if party has a healer
        let has_healer = member_ids.iter().any(|&mid| {
            state
                .adventurers
                .iter()
                .find(|a| a.id == mid)
                .map(|a| {
                    healer_archetypes
                        .iter()
                        .any(|h| a.archetype.to_lowercase().contains(h))
                })
                .unwrap_or(false)
        });

        let infection_chance = if has_healer {
            ADVENTURER_INFECTION_CHANCE * HEALER_SPREAD_REDUCTION
        } else {
            ADVENTURER_INFECTION_CHANCE
        };

        // Roll per adventurer
        for &mid in member_ids {
            let adv = match state.adventurers.iter().find(|a| a.id == mid) {
                Some(a) => a,
                None => continue,
            };
            if adv.status == AdventurerStatus::Dead {
                continue;
            }
            if adv.disease_status != DiseaseStatus::Healthy {
                continue;
            }

            // Medicine stat reduces infection chance
            let medicine_bonus = effective_noncombat_stats(adv).3;
            let adjusted_chance = infection_chance / (1.0 + medicine_bonus * 0.03);

            let roll = lcg_f32(&mut state.rng);
            if roll < adjusted_chance {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                    let severity = state
                        .diseases
                        .iter()
                        .find(|d| d.id == disease_id)
                        .map(|d| d.severity * 0.5)
                        .unwrap_or(20.0);
                    adv.disease_status = DiseaseStatus::Infected {
                        disease_id,
                        severity,
                    };
                    // Effects: -20% combat effectiveness via fatigue
                    adv.fatigue = (adv.fatigue + 20.0).min(100.0);

                    events.push(WorldEvent::AdventurerInfected {
                        adventurer_id: mid,
                        disease_id,
                    });
                }
            }
        }
    }

    // Infected adventurers can spread to party members (intra-party contagion)
    spread_within_parties(state, events);
}

/// Infected adventurers spread disease to healthy party members.
fn spread_within_parties(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut new_infections: Vec<(u32, u32, f32)> = Vec::new(); // (adv_id, disease_id, severity)

    for party in &state.parties {
        // Find infected members in this party
        let infected_disease: Option<(u32, f32)> = party.member_ids.iter().find_map(|&mid| {
            state.adventurers.iter().find(|a| a.id == mid).and_then(|a| {
                if let DiseaseStatus::Infected {
                    disease_id,
                    severity,
                } = &a.disease_status
                {
                    Some((*disease_id, *severity))
                } else {
                    None
                }
            })
        });

        let (disease_id, severity) = match infected_disease {
            Some(d) => d,
            None => continue,
        };

        // Check for healer in party
        let healer_archetypes = ["cleric", "healer", "priest", "druid", "shaman"];
        let has_healer = party.member_ids.iter().any(|&mid| {
            state
                .adventurers
                .iter()
                .find(|a| a.id == mid)
                .map(|a| {
                    healer_archetypes
                        .iter()
                        .any(|h| a.archetype.to_lowercase().contains(h))
                })
                .unwrap_or(false)
        });

        let spread_chance = if has_healer {
            0.05 * HEALER_SPREAD_REDUCTION
        } else {
            0.05
        };

        for &mid in &party.member_ids {
            let adv = match state.adventurers.iter().find(|a| a.id == mid) {
                Some(a) => a,
                None => continue,
            };
            if adv.disease_status != DiseaseStatus::Healthy || adv.status == AdventurerStatus::Dead
            {
                continue;
            }

            let roll = lcg_f32(&mut state.rng);
            if roll < spread_chance {
                new_infections.push((mid, disease_id, severity));
            }
        }
    }

    for (adv_id, disease_id, severity) in new_infections {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            adv.disease_status = DiseaseStatus::Infected {
                disease_id,
                severity: severity * 0.5,
            };
            adv.fatigue = (adv.fatigue + 20.0).min(100.0);

            events.push(WorldEvent::AdventurerInfected {
                adventurer_id: adv_id,
                disease_id,
            });
        }
    }
}

/// Resolve diseases that have been contained or expired.
fn resolve_diseases(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut contained_names: Vec<(u32, String)> = Vec::new();

    state.diseases.retain(|d| {
        let expired = tick - d.started_tick >= DISEASE_NATURAL_DURATION;
        let contained = d.containment > d.severity;
        if expired || contained {
            contained_names.push((d.id, d.name.clone()));
            false
        } else {
            true
        }
    });

    for (disease_id, disease_name) in contained_names {
        // Move infected adventurers to Recovering
        for adv in &mut state.adventurers {
            if let DiseaseStatus::Infected {
                disease_id: did, ..
            } = &adv.disease_status
            {
                if *did == disease_id {
                    adv.disease_status = DiseaseStatus::Recovering;
                }
            }
        }

        events.push(WorldEvent::DiseaseContained {
            disease_id,
            disease_name,
        });
    }

    // Recovering adventurers heal back to Healthy over time
    for adv in &mut state.adventurers {
        if adv.disease_status == DiseaseStatus::Recovering {
            // Recover after a short period (just set back to Healthy each disease tick)
            adv.disease_status = DiseaseStatus::Healthy;
        }
    }
}

/// Find the closest region to a position (simple distance check using region index).
fn find_closest_region(pos: (f32, f32), regions: &[Region]) -> Option<usize> {
    if regions.is_empty() {
        return None;
    }
    // Use a deterministic mapping: region index based on position quadrant
    // Since regions don't have positions, use a simple hash of position to region
    let hash = ((pos.0 * 7.0 + pos.1 * 13.0).abs() as usize) % regions.len();
    Some(regions[hash].id)
}
