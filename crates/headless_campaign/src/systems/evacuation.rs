//! Emergency evacuation system — fires every 200 ticks.
//!
//! When regions face catastrophic threats (monster swarms, severe disease,
//! enemy conquest, extreme weather), the guild can evacuate civilians,
//! adventurers, and resources to a neighboring region.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to tick the evacuation system (in ticks).
const EVACUATION_INTERVAL: u64 = 7;

/// Cost in gold to order an evacuation.
pub const EVACUATION_COST: f32 = 30.0;

/// Duration of an evacuation in ticks.
const EVACUATION_DURATION: u64 = 10;

/// Monster population threshold that triggers a swarm warning.
const MONSTER_SWARM_THRESHOLD: f32 = 90.0;

/// Disease severity threshold for evacuation trigger.
const DISEASE_SEVERITY_THRESHOLD: f32 = 70.0;

/// Region control threshold below which enemy conquest is imminent.
const CONTROL_COLLAPSE_THRESHOLD: f32 = 20.0;

/// Weather severity threshold for evacuation trigger.
const WEATHER_SEVERITY_THRESHOLD: f32 = 0.8;

/// Minimum civilian population saved (fraction).
const MIN_EVACUEE_FRACTION: f32 = 0.50;
/// Maximum civilian population saved (fraction).
const MAX_EVACUEE_FRACTION: f32 = 0.80;

/// Minimum supplies saved (fraction of trade goods/resources).
const MIN_SUPPLIES_FRACTION: f32 = 0.30;
/// Maximum supplies saved (fraction).
const MAX_SUPPLIES_FRACTION: f32 = 0.50;

/// Population loss when NOT evacuating a triggered region.
const NO_EVAC_POPULATION_LOSS_FRACTION: f32 = 0.30;
/// Morale penalty when NOT evacuating.
const NO_EVAC_MORALE_PENALTY: f32 = 15.0;

/// Reputation gain for successful evacuation.
const EVAC_REPUTATION_GAIN: f32 = 5.0;
/// Morale boost at destination from receiving refugees.
const REFUGEE_MORALE_BOOST: f32 = 3.0;

/// Per-party bonus: each guild party in the region increases evacuation fraction.
const PARTY_PRESENCE_BONUS: f32 = 0.05;

/// Tick the evacuation system. Called every tick, but only processes every
/// `EVACUATION_INTERVAL` ticks.
pub fn tick_evacuations(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % EVACUATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Phase 1: Detect new evacuation triggers and present choice events
    detect_triggers(state, events);

    // Phase 2: Progress active evacuations
    progress_evacuations(state, events);
}

/// Detect catastrophic threats in regions and emit evacuation choice events.
fn detect_triggers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let regions_len = state.overworld.regions.len();
    if regions_len == 0 {
        return;
    }

    // Collect regions already being evacuated
    let evacuating_regions: Vec<usize> = state
        .evacuations
        .iter()
        .filter(|e| !e.completed)
        .map(|e| e.source_region_id)
        .collect();

    // Check pending choices for existing evacuation choices
    let pending_evac_regions: Vec<usize> = state
        .pending_choices
        .iter()
        .filter_map(|c| match &c.source {
            ChoiceSource::Evacuation { source_region_id } => Some(*source_region_id),
            _ => None,
        })
        .collect();

    // Scan each region for catastrophic threats
    let mut triggered: Vec<(usize, String)> = Vec::new();

    for region in &state.overworld.regions {
        let rid = region.id;

        // Skip if already evacuating or has pending choice
        if evacuating_regions.contains(&rid) || pending_evac_regions.contains(&rid) {
            continue;
        }

        // Trigger: Monster swarm (population > 90)
        let monster_swarm = state
            .monster_populations
            .iter()
            .any(|m| m.region_id == rid && m.population > MONSTER_SWARM_THRESHOLD);
        if monster_swarm {
            triggered.push((rid, "Monster swarm overwhelming the region".into()));
            continue;
        }

        // Trigger: Active disease with severity > 70
        let severe_disease = state
            .diseases
            .iter()
            .any(|d| d.affected_regions.contains(&rid) && d.severity > DISEASE_SEVERITY_THRESHOLD);
        if severe_disease {
            triggered.push((rid, "Deadly plague ravaging the population".into()));
            continue;
        }

        // Trigger: Enemy faction about to conquer (region control < 20)
        if region.control < CONTROL_COLLAPSE_THRESHOLD {
            triggered.push((rid, "Enemy forces overrunning the region".into()));
            continue;
        }

        // Trigger: Severe weather (flood/blizzard/storm with high severity)
        let severe_weather = state
            .overworld
            .active_weather
            .iter()
            .any(|w| {
                w.affected_regions.contains(&rid)
                    && w.severity > WEATHER_SEVERITY_THRESHOLD
                    && matches!(
                        w.weather_type,
                        WeatherType::Flood | WeatherType::Blizzard | WeatherType::Storm
                    )
            });
        if severe_weather {
            triggered.push((rid, "Catastrophic weather threatening the region".into()));
            continue;
        }
    }

    // Emit choice events for triggered regions
    for (region_id, reason) in triggered {
        let region_name = state
            .overworld
            .regions
            .get(region_id)
            .map(|r| r.name.clone())
            .unwrap_or_else(|| format!("Region {}", region_id));

        // Find best neighboring destination (highest morale, not also in crisis)
        let neighbors: Vec<usize> = state
            .overworld
            .regions
            .get(region_id)
            .map(|r| r.neighbors.clone())
            .unwrap_or_default();

        let best_dest = neighbors
            .iter()
            .filter(|&&nid| nid < regions_len)
            .filter(|&&nid| !evacuating_regions.contains(&nid))
            .max_by(|&&a, &&b| {
                let ma = state
                    .overworld
                    .regions
                    .get(a)
                    .map(|r| r.civilian_morale)
                    .unwrap_or(0.0);
                let mb = state
                    .overworld
                    .regions
                    .get(b)
                    .map(|r| r.civilian_morale)
                    .unwrap_or(0.0);
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0);

        let dest_name = state
            .overworld
            .regions
            .get(best_dest)
            .map(|r| r.name.clone())
            .unwrap_or_else(|| format!("Region {}", best_dest));

        let choice_id = state.next_event_id;
        state.next_event_id += 1;

        let prompt = format!(
            "EMERGENCY: {} in {}! Evacuate civilians to {} (costs {:.0} gold) or hold position and defend?",
            reason, region_name, dest_name, EVACUATION_COST
        );

        let deadline_ms = state.elapsed_ms + (13 * CAMPAIGN_TURN_SECS as u64 * 1000);

        state.pending_choices.push(ChoiceEvent {
            id: choice_id,
            source: ChoiceSource::Evacuation {
                source_region_id: region_id,
            },
            prompt: prompt.clone(),
            options: vec![
                ChoiceOption {
                    label: format!("Evacuate to {}", dest_name),
                    description: format!(
                        "Order the evacuation. Saves 50-80% of civilians and 30-50% of supplies. Costs {:.0} gold.",
                        EVACUATION_COST
                    ),
                    effects: vec![
                        ChoiceEffect::Gold(-EVACUATION_COST),
                        ChoiceEffect::BeginEvacuation {
                            source_region_id: region_id,
                            destination_region_id: best_dest,
                        },
                    ],
                },
                ChoiceOption {
                    label: "Defend and hold".into(),
                    description:
                        "Stay and fight. Risk population loss and morale collapse.".into(),
                    effects: vec![ChoiceEffect::NoEvacuationPenalty {
                        region_id,
                    }],
                },
            ],
            default_option: 1, // Default: defend (no cost)
            deadline_ms: Some(deadline_ms),
            created_at_ms: state.elapsed_ms,
        });

        events.push(WorldEvent::ChoicePresented {
            choice_id,
            prompt,
            num_options: 2,
        });
    }
}

/// Progress active evacuations and complete them after EVACUATION_DURATION ticks.
fn progress_evacuations(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let current_tick = state.tick;
    let mut completed_indices = Vec::new();

    for (i, evac) in state.evacuations.iter().enumerate() {
        if !evac.completed && current_tick >= evac.started_tick + EVACUATION_DURATION {
            completed_indices.push(i);
        }
    }

    for idx in completed_indices {
        let evac = &mut state.evacuations[idx];
        evac.completed = true;

        let evacuees = evac.evacuees;
        let supplies_saved = evac.supplies_saved;
        let dest_id = evac.destination_region_id;
        let source_id = evac.source_region_id;

        // Add refugees to destination
        if let Some(dest) = state.overworld.regions.iter_mut().find(|r| r.id == dest_id) {
            dest.population = dest.population.saturating_add(evacuees);
            dest.civilian_morale = (dest.civilian_morale + REFUGEE_MORALE_BOOST).min(100.0);
        }

        // Grant reputation
        state.guild.reputation += EVAC_REPUTATION_GAIN;
        // Return some supplies
        state.guild.supplies += supplies_saved;

        events.push(WorldEvent::EvacuationCompleted {
            source_region_id: source_id,
            destination_region_id: dest_id,
            evacuees,
            supplies_saved,
        });

        events.push(WorldEvent::CiviliansRescued {
            region_id: source_id,
            count: evacuees,
        });

        // Update trackers
        state.system_trackers.evacuations_completed += 1;
        state.system_trackers.total_civilians_rescued += evacuees;
        state.system_trackers.active_evacuations =
            state.system_trackers.active_evacuations.saturating_sub(1);
    }
}

/// Begin an evacuation from a source region to a destination region.
/// Called from the choice resolution or from `OrderEvacuation` action.
pub fn begin_evacuation(
    state: &mut CampaignState,
    source_region_id: usize,
    destination_region_id: usize,
    events: &mut Vec<WorldEvent>,
) -> bool {
    let regions_len = state.overworld.regions.len();
    if source_region_id >= regions_len || destination_region_id >= regions_len {
        return false;
    }

    // Calculate evacuee count based on population and party presence
    let source_pop = state
        .overworld
        .regions
        .get(source_region_id)
        .map(|r| r.population)
        .unwrap_or(0);

    // Count guild parties present (simple heuristic: parties on active quests)
    let parties_in_region = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
            )
        })
        .count();

    let party_bonus = (parties_in_region as f32 * PARTY_PRESENCE_BONUS).min(0.15);

    // Deterministic fraction based on RNG
    let r = lcg_f32(&mut state.rng);
    let base_fraction = MIN_EVACUEE_FRACTION + r * (MAX_EVACUEE_FRACTION - MIN_EVACUEE_FRACTION);
    let fraction = (base_fraction + party_bonus).min(MAX_EVACUEE_FRACTION);
    let evacuees = (source_pop as f32 * fraction) as u32;

    // Calculate supplies saved
    let r2 = lcg_f32(&mut state.rng);
    let supply_fraction =
        MIN_SUPPLIES_FRACTION + r2 * (MAX_SUPPLIES_FRACTION - MIN_SUPPLIES_FRACTION);
    // Save a fraction of guild supplies as the region contribution
    let supplies_saved = state.guild.supplies * supply_fraction * 0.1;

    // Reduce source population
    if let Some(source) = state
        .overworld
        .regions
        .iter_mut()
        .find(|r| r.id == source_region_id)
    {
        source.population = source.population.saturating_sub(evacuees);
        source.civilian_morale = (source.civilian_morale - 10.0).max(0.0);
    }

    let evac_id = state.next_evacuation_id;
    state.next_evacuation_id += 1;

    state.evacuations.push(Evacuation {
        id: evac_id,
        source_region_id,
        destination_region_id,
        evacuees,
        supplies_saved,
        started_tick: state.tick,
        completed: false,
        cost: EVACUATION_COST,
    });

    // Note: gold deduction is handled by the ChoiceEffect::Gold or by the
    // OrderEvacuation action handler. We only deduct here if called directly.
    // The caller is responsible for gold deduction.

    // Update trackers
    state.system_trackers.active_evacuations += 1;

    events.push(WorldEvent::EvacuationOrdered {
        source_region_id,
        destination_region_id,
        evacuees,
        cost: EVACUATION_COST,
    });

    true
}

/// Apply penalties when the player chooses NOT to evacuate a triggered region.
pub fn apply_no_evacuation_penalty(
    state: &mut CampaignState,
    region_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    if let Some(region) = state
        .overworld
        .regions
        .iter_mut()
        .find(|r| r.id == region_id)
    {
        let pop_loss = (region.population as f32 * NO_EVAC_POPULATION_LOSS_FRACTION) as u32;
        region.population = region.population.saturating_sub(pop_loss);
        region.civilian_morale = (region.civilian_morale - NO_EVAC_MORALE_PENALTY).max(0.0);

        events.push(WorldEvent::EvacuationFailed {
            region_id,
            population_lost: pop_loss,
            reason: "Guild chose to hold position".into(),
        });
    }
}
