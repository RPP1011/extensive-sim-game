//! Dead zone system — every 500 ticks.
//!
//! Over-exploited regions accumulate extraction pressure from questing and
//! harvesting. When pressure exceeds a threshold, dead zones form and spread
//! corruption to neighbors, forcing geographic strategy and resource rotation.
//!
//! Effects on dead-zone regions:
//! - -30% quest rewards
//! - -50% resource yields
//! - +20% monster spawn rate (via threat_level increase)
//! - Morale penalty for parties and civilians

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 500 ticks.
const DEAD_ZONE_INTERVAL: u64 = 500;

/// Extraction pressure added per completed quest in a region.
const QUEST_PRESSURE: f32 = 0.02;
/// Extraction pressure added per crafting harvest in a region.
const HARVEST_PRESSURE: f32 = 0.03;

/// Pressure threshold above which dead zone level starts rising.
const PRESSURE_THRESHOLD: f32 = 0.5;
/// Rate at which dead zone level rises when pressure exceeds threshold.
const DEAD_ZONE_RISE_RATE: f32 = 0.01;
/// Dead zone level above which corruption spreads to neighbors.
const SPREAD_THRESHOLD: f32 = 0.6;
/// Rate at which dead zones spread to adjacent regions.
const SPREAD_RATE: f32 = 0.005;
/// Pressure threshold below which dead zones start recovering.
const RECOVERY_PRESSURE_THRESHOLD: f32 = 0.2;
/// Rate at which dead zone level decreases during recovery.
const RECOVERY_RATE: f32 = 0.005;
/// Natural decay of extraction pressure per tick.
const PRESSURE_DECAY: f32 = 0.01;

// ---------------------------------------------------------------------------
// Public API for other systems to record extraction events
// ---------------------------------------------------------------------------

/// Record extraction pressure from a completed quest in a region.
pub fn record_quest_pressure(state: &mut CampaignState, region_id: usize) {
    if region_id < state.dead_zone_level.len() {
        state.extraction_pressure[region_id] += QUEST_PRESSURE;
    }
}

/// Record extraction pressure from a crafting harvest in a region.
pub fn record_harvest_pressure(state: &mut CampaignState, region_id: usize) {
    if region_id < state.dead_zone_level.len() {
        state.extraction_pressure[region_id] += HARVEST_PRESSURE;
    }
}

/// Get the dead zone level for a region (0.0 = healthy, 1.0 = fully dead).
/// Returns 0.0 for out-of-bounds region IDs.
pub fn dead_zone_level(state: &CampaignState, region_id: usize) -> f32 {
    state.dead_zone_level.get(region_id).copied().unwrap_or(0.0)
}

/// Quest reward multiplier for a region (1.0 = normal, 0.7 = dead zone penalty).
pub fn quest_reward_multiplier(state: &CampaignState, region_id: usize) -> f32 {
    let level = dead_zone_level(state, region_id);
    if level > 0.0 {
        1.0 - 0.3 * level
    } else {
        1.0
    }
}

/// Resource yield multiplier for a region (1.0 = normal, 0.5 = dead zone penalty).
pub fn resource_yield_multiplier(state: &CampaignState, region_id: usize) -> f32 {
    let level = dead_zone_level(state, region_id);
    if level > 0.0 {
        1.0 - 0.5 * level
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Main tick entry point
// ---------------------------------------------------------------------------

/// Advance dead zones: decay pressure, expand/recover dead zones, spread
/// corruption to neighbors.
pub fn tick_dead_zones(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % DEAD_ZONE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let num_regions = state.overworld.regions.len();
    if num_regions == 0 {
        return;
    }

    // Lazy initialization: ensure vectors match region count.
    ensure_vectors(state);

    // --- Natural pressure decay ---
    for pressure in &mut state.extraction_pressure {
        *pressure = (*pressure - PRESSURE_DECAY).max(0.0);
    }

    // --- Dead zone expansion / recovery ---
    // Snapshot levels to avoid feedback within a single tick.
    let prev_levels: Vec<f32> = state.dead_zone_level.clone();
    let pressures: Vec<f32> = state.extraction_pressure.clone();

    for i in 0..num_regions {
        if pressures[i] > PRESSURE_THRESHOLD {
            // Pressure is high — dead zone rises.
            state.dead_zone_level[i] = (prev_levels[i] + DEAD_ZONE_RISE_RATE).min(1.0);

            if state.dead_zone_level[i] > prev_levels[i] {
                events.push(WorldEvent::DeadZoneExpanding {
                    region_id: i,
                    level: state.dead_zone_level[i],
                });

                // Apply effects: increase threat (monster spawn rate boost).
                if let Some(region) = state.overworld.regions.get_mut(i) {
                    region.threat_level =
                        (region.threat_level + state.dead_zone_level[i] * 2.0).min(100.0);
                    region.civilian_morale =
                        (region.civilian_morale - state.dead_zone_level[i] * 3.0).max(0.0);
                }
            }
        } else if pressures[i] < RECOVERY_PRESSURE_THRESHOLD && prev_levels[i] > 0.0 {
            // Pressure is low — dead zone recovers.
            state.dead_zone_level[i] = (prev_levels[i] - RECOVERY_RATE).max(0.0);

            if state.dead_zone_level[i] < prev_levels[i] {
                events.push(WorldEvent::DeadZoneRecovering {
                    region_id: i,
                    new_level: state.dead_zone_level[i],
                });
            }
        }
    }

    // --- Spread to neighbors ---
    // Use the snapshot (prev_levels) so spreading is simultaneous.
    let neighbor_data: Vec<Vec<usize>> = state
        .overworld
        .regions
        .iter()
        .map(|r| r.neighbors.clone())
        .collect();

    for i in 0..num_regions {
        if prev_levels[i] > SPREAD_THRESHOLD {
            if let Some(neighbors) = neighbor_data.get(i) {
                for &neighbor_id in neighbors {
                    if neighbor_id < num_regions {
                        let old = state.dead_zone_level[neighbor_id];
                        state.dead_zone_level[neighbor_id] =
                            (old + SPREAD_RATE).min(1.0);

                        if state.dead_zone_level[neighbor_id] > old && old < SPREAD_THRESHOLD {
                            events.push(WorldEvent::DeadZoneSpreading {
                                from_region: i,
                                to_region: neighbor_id,
                            });
                        }
                    }
                }
            }
        }
    }

    // --- Apply morale penalty to parties in dead-zone regions ---
    // Find which region each party is in by checking closest region position.
    // (Simplified: use party position to find matching region index.)
    for party in &mut state.parties {
        // Approximate region: use the quest's region or fallback.
        // Since parties don't have a direct region_id, we check all regions
        // with dead zones and apply penalties to parties on quests there.
        if let Some(quest_id) = party.quest_id {
            if let Some(quest) = state.active_quests.iter().find(|q| q.id == quest_id) {
                let region_id = quest.request.source_area_id.unwrap_or(0);
                if region_id < state.dead_zone_level.len()
                    && state.dead_zone_level[region_id] > 0.3
                {
                    party.morale =
                        (party.morale - state.dead_zone_level[region_id] * 5.0).max(0.0);
                }
            }
        }
    }
}

/// Ensure dead zone vectors match the number of regions.
fn ensure_vectors(state: &mut CampaignState) {
    let num_regions = state.overworld.regions.len();
    while state.dead_zone_level.len() < num_regions {
        state.dead_zone_level.push(0.0);
    }
    while state.extraction_pressure.len() < num_regions {
        state.extraction_pressure.push(0.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    fn test_state_with_regions(num_regions: usize) -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.overworld.regions = (0..num_regions)
            .map(|i| Region {
                id: i,
                name: format!("Region {}", i),
                owner_faction_id: 0,
                neighbors: if i + 1 < num_regions {
                    vec![i + 1]
                } else {
                    vec![]
                },
                unrest: 10.0,
                control: 80.0,
                threat_level: 20.0,
                visibility: 0.5,
                population: 500,
                civilian_morale: 50.0,
                tax_rate: 0.1,
                growth_rate: 0.0,
            })
            .collect();
        // Also set up neighbor back-links.
        if num_regions > 1 {
            state.overworld.regions[1].neighbors.push(0);
        }
        state.dead_zone_level = vec![0.0; num_regions];
        state.extraction_pressure = vec![0.0; num_regions];
        state
    }

    #[test]
    fn does_not_fire_at_tick_zero() {
        let mut state = test_state_with_regions(3);
        state.tick = 0;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert!(events.is_empty());
    }

    #[test]
    fn does_not_fire_off_interval() {
        let mut state = test_state_with_regions(3);
        state.tick = 250;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert!(events.is_empty());
    }

    #[test]
    fn pressure_decays_naturally() {
        let mut state = test_state_with_regions(3);
        state.extraction_pressure[0] = 0.1;
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert!(
            state.extraction_pressure[0] < 0.1,
            "Pressure should decay: got {}",
            state.extraction_pressure[0]
        );
    }

    #[test]
    fn dead_zone_rises_when_pressure_high() {
        let mut state = test_state_with_regions(3);
        state.extraction_pressure[0] = 0.7; // Above threshold of 0.5
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert!(
            state.dead_zone_level[0] > 0.0,
            "Dead zone should rise when pressure exceeds threshold"
        );
        assert!(events.iter().any(|e| matches!(
            e,
            WorldEvent::DeadZoneExpanding { region_id: 0, .. }
        )));
    }

    #[test]
    fn dead_zone_recovers_when_pressure_low() {
        let mut state = test_state_with_regions(3);
        state.dead_zone_level[0] = 0.5;
        state.extraction_pressure[0] = 0.05; // Below recovery threshold 0.2
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert!(
            state.dead_zone_level[0] < 0.5,
            "Dead zone should recover when pressure is low"
        );
        assert!(events.iter().any(|e| matches!(
            e,
            WorldEvent::DeadZoneRecovering { region_id: 0, .. }
        )));
    }

    #[test]
    fn dead_zone_spreads_to_neighbors() {
        let mut state = test_state_with_regions(3);
        state.dead_zone_level[0] = 0.8; // Above spread threshold of 0.6
        state.extraction_pressure[0] = 0.6; // Keep pressure high to prevent recovery
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert!(
            state.dead_zone_level[1] > 0.0,
            "Dead zone should spread to neighbor: got {}",
            state.dead_zone_level[1]
        );
        assert!(events.iter().any(|e| matches!(
            e,
            WorldEvent::DeadZoneSpreading {
                from_region: 0,
                to_region: 1,
            }
        )));
    }

    #[test]
    fn record_quest_pressure_accumulates() {
        let mut state = test_state_with_regions(3);
        record_quest_pressure(&mut state, 1);
        record_quest_pressure(&mut state, 1);
        assert!(
            (state.extraction_pressure[1] - 0.04).abs() < 0.001,
            "Two quests should add 0.04 pressure: got {}",
            state.extraction_pressure[1]
        );
    }

    #[test]
    fn record_harvest_pressure_accumulates() {
        let mut state = test_state_with_regions(3);
        record_harvest_pressure(&mut state, 2);
        assert!(
            (state.extraction_pressure[2] - 0.03).abs() < 0.001,
            "One harvest should add 0.03 pressure: got {}",
            state.extraction_pressure[2]
        );
    }

    #[test]
    fn reward_multipliers_scale_with_level() {
        let mut state = test_state_with_regions(3);
        state.dead_zone_level[0] = 1.0;
        state.dead_zone_level[1] = 0.0;

        let quest_mult = quest_reward_multiplier(&state, 0);
        assert!(
            (quest_mult - 0.7).abs() < 0.01,
            "Full dead zone should give 0.7 quest multiplier: got {}",
            quest_mult
        );

        let resource_mult = resource_yield_multiplier(&state, 0);
        assert!(
            (resource_mult - 0.5).abs() < 0.01,
            "Full dead zone should give 0.5 resource multiplier: got {}",
            resource_mult
        );

        assert!(
            (quest_reward_multiplier(&state, 1) - 1.0).abs() < 0.01,
            "Healthy region should have 1.0 quest multiplier"
        );
        assert!(
            (resource_yield_multiplier(&state, 1) - 1.0).abs() < 0.01,
            "Healthy region should have 1.0 resource multiplier"
        );
    }

    #[test]
    fn vectors_lazily_initialized() {
        let mut state = test_state_with_regions(5);
        state.dead_zone_level.clear();
        state.extraction_pressure.clear();
        state.extraction_pressure = vec![]; // Force empty
        state.dead_zone_level = vec![];
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_dead_zones(&mut state, &mut deltas, &mut events);
        assert_eq!(
            state.dead_zone_level.len(),
            5,
            "Should lazily initialize to match region count"
        );
        assert_eq!(state.extraction_pressure.len(), 5);
    }
}
