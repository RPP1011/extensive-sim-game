//! Dynamic world geography system — fires every 1000 ticks.
//!
//! The world map slowly changes over time: forests grow, rivers flood, deserts
//! expand, settlements grow into cities, roads degrade, and wilderness reclaims
//! abandoned regions. Each change is triggered by region conditions and advances
//! 5-10 progress per geography tick until completing at 100%.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to evaluate geography changes (in ticks).
const GEOGRAPHY_INTERVAL: u64 = 33;

/// Maximum concurrent geography changes across the world.
const MAX_ACTIVE_CHANGES: usize = 8;

// ---------------------------------------------------------------------------
// Main tick entry point
// ---------------------------------------------------------------------------

/// Advance geography: evaluate conditions for new changes, progress existing
/// ones, and apply completion effects.
pub fn tick_geography(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % GEOGRAPHY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Progress existing changes ---
    progress_changes(state, events);

    // --- Evaluate conditions for new changes ---
    if state.geography_changes.len() < MAX_ACTIVE_CHANGES {
        evaluate_new_changes(state, events);
    }
}

// ---------------------------------------------------------------------------
// Progress existing changes
// ---------------------------------------------------------------------------

fn progress_changes(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Advance progress on all active changes and collect completed ones.
    let mut completed = Vec::new();

    for change in &mut state.geography_changes {
        // Advance 5-10 per tick (deterministic from RNG).
        let advance = 5.0 + lcg_f32(&mut state.rng) * 5.0;
        change.progress = (change.progress + advance).min(100.0);

        if change.progress >= 100.0 {
            completed.push((change.id, change.change_type, change.region_id));
        } else {
            events.push(WorldEvent::GeographyChanging {
                region_id: change.region_id,
                change_type: change.change_type,
                progress: change.progress,
            });
        }
    }

    // Apply completion effects and remove completed changes.
    for (_id, change_type, region_id) in &completed {
        let effect = apply_completion(state, *change_type, *region_id);
        events.push(WorldEvent::GeographyComplete {
            region_id: *region_id,
            change_type: *change_type,
            effects: vec![effect],
        });
    }

    let completed_ids: Vec<u32> = completed.iter().map(|(id, _, _)| *id).collect();
    state
        .geography_changes
        .retain(|c| !completed_ids.contains(&c.id));
}

// ---------------------------------------------------------------------------
// Evaluate conditions for new changes
// ---------------------------------------------------------------------------

fn evaluate_new_changes(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let num_regions = state.overworld.regions.len();
    if num_regions == 0 {
        return;
    }

    // Check each region for conditions that trigger geography changes.
    // We snapshot the region data first to avoid borrow issues.
    let region_data: Vec<(usize, f32, f32, u32, f32, f32)> = state
        .overworld
        .regions
        .iter()
        .map(|r| {
            (
                r.id,
                r.control,
                r.unrest,
                r.population,
                r.threat_level,
                r.civilian_morale,
            )
        })
        .collect();

    let season = state.overworld.season;
    let active_quest_count = state.active_quests.len();
    let party_count = state.parties.len();

    for (rid, control, unrest, population, threat_level, morale) in &region_data {
        // Skip if this region already has an active geography change.
        if state
            .geography_changes
            .iter()
            .any(|c| c.region_id == *rid)
        {
            continue;
        }

        // Cap concurrent changes.
        if state.geography_changes.len() >= MAX_ACTIVE_CHANGES {
            break;
        }

        // Evaluate each change type's trigger conditions.
        if let Some(change_type) = evaluate_region_conditions(
            &mut state.rng,
            *rid,
            *control,
            *unrest,
            *population,
            *threat_level,
            *morale,
            season,
            active_quest_count,
            party_count,
        ) {
            let id = state.next_geo_change_id;
            state.next_geo_change_id += 1;

            state.geography_changes.push(GeographyChange {
                id,
                change_type,
                region_id: *rid,
                progress: 0.0,
                started_tick: state.tick,
            });

            events.push(WorldEvent::GeographyChanging {
                region_id: *rid,
                change_type,
                progress: 0.0,
            });
        }
    }
}

/// Check conditions for a single region and return a change type if triggered.
#[allow(clippy::too_many_arguments)]
fn evaluate_region_conditions(
    rng: &mut u64,
    _rid: usize,
    control: f32,
    unrest: f32,
    population: u32,
    threat_level: f32,
    morale: f32,
    season: Season,
    active_quest_count: usize,
    party_count: usize,
) -> Option<GeoChangeType> {
    let roll = lcg_f32(rng);

    // ForestGrowth: low population + low threat (no deforestation pressure)
    // Probability ~15% when conditions met.
    if population < 300 && threat_level < 30.0 && roll < 0.15 {
        return Some(GeoChangeType::ForestGrowth);
    }

    let roll = lcg_f32(rng);
    // DesertExpansion: drought conditions (low morale proxy) + summer/autumn
    if morale < 30.0
        && (season == Season::Summer || season == Season::Autumn)
        && roll < 0.10
    {
        return Some(GeoChangeType::DesertExpansion);
    }

    let roll = lcg_f32(rng);
    // RiverFlood: spring + high population (pressure on floodplains)
    if season == Season::Spring && population > 400 && roll < 0.12 {
        return Some(GeoChangeType::RiverFlood);
    }

    let roll = lcg_f32(rng);
    // SettlementGrowth: high population + high morale + low unrest
    if population > 500 && morale > 60.0 && unrest < 20.0 && roll < 0.10 {
        return Some(GeoChangeType::SettlementGrowth);
    }

    let roll = lcg_f32(rng);
    // RoadDegradation: no active quests using roads + low control (no maintenance)
    if active_quest_count == 0 && control < 40.0 && roll < 0.12 {
        return Some(GeoChangeType::RoadDegradation);
    }

    let roll = lcg_f32(rng);
    // WildernessReclaim: low control + no parties traveling + high threat
    if control < 25.0 && party_count == 0 && threat_level > 50.0 && roll < 0.15 {
        return Some(GeoChangeType::WildernessReclaim);
    }

    None
}

// ---------------------------------------------------------------------------
// Completion effects — permanently modify region properties
// ---------------------------------------------------------------------------

fn apply_completion(
    state: &mut CampaignState,
    change_type: GeoChangeType,
    region_id: usize,
) -> String {
    let region = match state.overworld.regions.get_mut(region_id) {
        Some(r) => r,
        None => return "Region not found".into(),
    };

    match change_type {
        GeoChangeType::ForestGrowth => {
            // Travel slower (higher threat from dense terrain) but richer resources.
            region.threat_level = (region.threat_level + 5.0).min(100.0);
            region.growth_rate += 0.02;
            "Forest has grown thick — travel is slower but herbs and wood are plentiful".into()
        }
        GeoChangeType::DesertExpansion => {
            // Population declines, morale drops, but rare gems appear.
            region.population = region.population.saturating_sub(50);
            region.civilian_morale = (region.civilian_morale - 10.0).max(0.0);
            region.growth_rate = (region.growth_rate - 0.03).max(-0.1);
            "Desert sands have swallowed fertile land — population declining but gem deposits uncovered".into()
        }
        GeoChangeType::RiverFlood => {
            // Fertile aftermath but infrastructure damage.
            region.growth_rate += 0.05;
            region.control = (region.control - 10.0).max(0.0);
            region.unrest = (region.unrest + 8.0).min(100.0);
            "River has flooded — rich soil deposited but infrastructure damaged".into()
        }
        GeoChangeType::SettlementGrowth => {
            // Population boom, more trade, higher tax base.
            region.population = (region.population + 100).min(2000);
            region.civilian_morale = (region.civilian_morale + 5.0).min(100.0);
            region.tax_rate = (region.tax_rate + 0.02).min(0.5);
            "Settlement has grown — more population and trade opportunities".into()
        }
        GeoChangeType::RoadDegradation => {
            // Travel slower, trade disrupted, control drops further.
            region.control = (region.control - 8.0).max(0.0);
            region.threat_level = (region.threat_level + 10.0).min(100.0);
            "Roads have fallen into disrepair — travel is dangerous and trade disrupted".into()
        }
        GeoChangeType::WildernessReclaim => {
            // More monsters but more resources. Permanent shift.
            region.threat_level = (region.threat_level + 15.0).min(100.0);
            region.control = (region.control - 15.0).max(0.0);
            region.growth_rate += 0.03;
            region.population = region.population.saturating_sub(30);
            "Wilderness has reclaimed the region — dangerous but resource-rich".into()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::StepDeltas;

    fn test_state_with_regions(num_regions: usize) -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.overworld.regions = (0..num_regions)
            .map(|i| Region {
                id: i,
                name: format!("Region {}", i),
                owner_faction_id: 0,
                neighbors: vec![],
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
        state
    }

    #[test]
    fn geography_does_not_fire_at_tick_zero() {
        let mut state = test_state_with_regions(5);
        state.tick = 0;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_geography(&mut state, &mut deltas, &mut events);
        assert!(state.geography_changes.is_empty());
        assert!(events.is_empty());
    }

    #[test]
    fn geography_does_not_fire_off_interval() {
        let mut state = test_state_with_regions(5);
        state.tick = 500; // Not a multiple of 1000.
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_geography(&mut state, &mut deltas, &mut events);
        assert!(state.geography_changes.is_empty());
    }

    #[test]
    fn geography_change_progresses_and_completes() {
        let mut state = test_state_with_regions(3);
        // Manually insert a change near completion.
        state.geography_changes.push(GeographyChange {
            id: 0,
            change_type: GeoChangeType::ForestGrowth,
            region_id: 0,
            progress: 92.0,
            started_tick: 1000,
        });
        state.next_geo_change_id = 1;

        state.tick = 2000;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_geography(&mut state, &mut deltas, &mut events);

        // Change should have completed (92 + 5-10 >= 100).
        assert!(
            state.geography_changes.is_empty(),
            "Change should have completed and been removed"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, WorldEvent::GeographyComplete { .. })),
            "Should emit GeographyComplete event"
        );
    }

    #[test]
    fn settlement_growth_increases_population() {
        let mut state = test_state_with_regions(3);
        let pop_before = state.overworld.regions[1].population;

        // Manually apply completion.
        let _effects = apply_completion(&mut state, GeoChangeType::SettlementGrowth, 1);

        assert!(
            state.overworld.regions[1].population > pop_before,
            "Settlement growth should increase population"
        );
    }

    #[test]
    fn desert_expansion_decreases_population() {
        let mut state = test_state_with_regions(3);
        let pop_before = state.overworld.regions[2].population;

        let _effects = apply_completion(&mut state, GeoChangeType::DesertExpansion, 2);

        assert!(
            state.overworld.regions[2].population < pop_before,
            "Desert expansion should decrease population"
        );
    }

    #[test]
    fn no_duplicate_changes_per_region() {
        let mut state = test_state_with_regions(3);
        // Pre-seed a change in region 0.
        state.geography_changes.push(GeographyChange {
            id: 0,
            change_type: GeoChangeType::ForestGrowth,
            region_id: 0,
            progress: 50.0,
            started_tick: 1000,
        });
        state.next_geo_change_id = 1;

        // Run many ticks; region 0 should never get a second change.
        for i in 2..=20 {
            state.tick = GEOGRAPHY_INTERVAL * i;
            let mut deltas = StepDeltas::default();
            let mut events = Vec::new();
            // Keep the existing change alive by resetting progress.
            if let Some(c) = state.geography_changes.iter_mut().find(|c| c.region_id == 0) {
                c.progress = 50.0;
            }
            tick_geography(&mut state, &mut deltas, &mut events);
        }

        let region0_count = state
            .geography_changes
            .iter()
            .filter(|c| c.region_id == 0)
            .count();
        assert_eq!(region0_count, 1, "Region should have at most one active change");
    }

    #[test]
    fn max_active_changes_enforced() {
        let mut state = test_state_with_regions(20);
        // Pre-seed MAX_ACTIVE_CHANGES changes.
        for i in 0..MAX_ACTIVE_CHANGES {
            state.geography_changes.push(GeographyChange {
                id: i as u32,
                change_type: GeoChangeType::RoadDegradation,
                region_id: i,
                progress: 10.0,
                started_tick: 1000,
            });
        }
        state.next_geo_change_id = MAX_ACTIVE_CHANGES as u32;

        state.tick = 2000;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_geography(&mut state, &mut deltas, &mut events);

        assert!(
            state.geography_changes.len() <= MAX_ACTIVE_CHANGES,
            "Should not exceed MAX_ACTIVE_CHANGES"
        );
    }
}
