//! Scouting and fog-of-war system — every 50 ticks (~5s).
//!
//! Manages region visibility that decays over time. High visibility
//! improves threat accuracy, enables champion detection, surfaces
//! crisis warnings, and reveals quest opportunities.

use crate::actions::{ScoutReportDetails, StepDeltas, WorldEvent};
use crate::state::*;

/// Visibility threshold above which a region is considered "scouted".
const SCOUTED_THRESHOLD: f32 = 0.5;

/// Minimum visibility for guild-controlled regions.
const GUILD_REGION_MIN_VISIBILITY: f32 = 0.6;

/// Visibility decay per scouting tick.
const VISIBILITY_DECAY: f32 = 0.005;

/// Visibility boost when a party travels through a region.
const PARTY_TRAVEL_BOOST: f32 = 0.1;

/// Visibility boost from HireScout action.
pub const HIRE_SCOUT_BOOST: f32 = 0.4;

/// Cadence: runs every 50 ticks.
const SCOUTING_INTERVAL: u64 = 1;

pub fn tick_scouting(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SCOUTING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.overworld.regions.is_empty() {
        return;
    }

    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Snapshot previous visibility to detect threshold crossings.
    let prev_visibility: Vec<f32> = state
        .overworld
        .regions
        .iter()
        .map(|r| r.visibility)
        .collect();

    // 1. Decay all region visibility.
    for region in &mut state.overworld.regions {
        region.visibility = (region.visibility - VISIBILITY_DECAY).max(0.0);
    }

    // 2. Guild-controlled regions maintain minimum visibility.
    for region in &mut state.overworld.regions {
        if region.owner_faction_id == guild_faction_id {
            region.visibility = region.visibility.max(GUILD_REGION_MIN_VISIBILITY);
        }
    }

    // 3. Parties traveling through regions boost visibility.
    //    We map party position to the nearest region by checking which
    //    region's locations are closest.
    let party_positions: Vec<(f32, f32)> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission
            )
        })
        .map(|p| p.position)
        .collect();

    if !party_positions.is_empty() {
        // Build a simple region→center mapping from locations owned by each faction.
        // Fall back to boosting all regions that share the location's faction owner.
        for pos in &party_positions {
            // Find the closest location to this party.
            let closest_loc = state
                .overworld
                .locations
                .iter()
                .min_by(|a, b| {
                    let da = dist_sq(a.position, *pos);
                    let db = dist_sq(b.position, *pos);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(loc) = closest_loc {
                // Boost the region that owns this location's faction, or the
                // first region whose faction matches.
                if let Some(faction_owner) = loc.faction_owner {
                    for region in &mut state.overworld.regions {
                        if region.owner_faction_id == faction_owner {
                            region.visibility =
                                (region.visibility + PARTY_TRAVEL_BOOST).min(1.0);
                        }
                    }
                }
            }
        }
    }

    // 4. Generate scout reports for regions that just crossed the threshold.
    for (i, region) in state.overworld.regions.iter().enumerate() {
        let was_below = prev_visibility.get(i).copied().unwrap_or(0.0) < SCOUTED_THRESHOLD;
        let now_above = region.visibility >= SCOUTED_THRESHOLD;

        if was_below && now_above {
            let report = build_scout_report(state, region);
            events.push(WorldEvent::RegionScoutReport {
                region_id: region.id,
                details: report,
            });
        }
    }
}

/// Boost a specific region's visibility (called from the HireScout action handler).
pub fn boost_region_visibility(
    state: &mut CampaignState,
    region_id: usize,
    amount: f32,
    events: &mut Vec<WorldEvent>,
) {
    let prev = state
        .overworld
        .regions
        .iter()
        .find(|r| r.id == region_id)
        .map(|r| r.visibility)
        .unwrap_or(0.0);

    if let Some(region) = state
        .overworld
        .regions
        .iter_mut()
        .find(|r| r.id == region_id)
    {
        region.visibility = (region.visibility + amount).min(1.0);
    }

    // Check if we crossed the threshold.
    if let Some(region) = state.overworld.regions.iter().find(|r| r.id == region_id) {
        if prev < SCOUTED_THRESHOLD && region.visibility >= SCOUTED_THRESHOLD {
            let report = build_scout_report(state, region);
            events.push(WorldEvent::RegionScoutReport {
                region_id,
                details: report,
            });
        }
    }
}

/// Apply visibility-based error to a threat value.
/// At low visibility (0.0), threat has +/-30% error.
/// At high visibility (1.0), threat has +/-5% error.
/// Returns (min_threat, max_threat) range.
pub fn threat_accuracy_range(threat: f32, visibility: f32) -> (f32, f32) {
    let error_pct = 0.30 - 0.25 * visibility.clamp(0.0, 1.0); // 0.30 → 0.05
    let delta = threat * error_pct;
    ((threat - delta).max(0.0), threat + delta)
}

/// Whether a champion traveling in a region should be visible to the guild.
pub fn is_champion_visible(region_visibility: f32) -> bool {
    region_visibility > SCOUTED_THRESHOLD
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_scout_report(state: &CampaignState, region: &Region) -> ScoutReportDetails {
    // Faction military strength
    let faction_strength = state
        .factions
        .iter()
        .find(|f| f.id == region.owner_faction_id)
        .map(|f| f.military_strength)
        .unwrap_or(0.0);

    // Count quest opportunities in this region.
    // We approximate by counting quests whose source_area matches locations
    // owned by this region's faction.
    let region_location_ids: Vec<usize> = state
        .overworld
        .locations
        .iter()
        .filter(|l| l.faction_owner == Some(region.owner_faction_id))
        .map(|l| l.id)
        .collect();

    let quest_opportunities = state
        .request_board
        .iter()
        .filter(|q| {
            q.source_area_id
                .map(|aid| region_location_ids.contains(&aid))
                .unwrap_or(false)
        })
        .count();

    // Champion sightings: adventurers with rallying_to set whose party is in
    // a location owned by this region's faction.
    let champion_sightings: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.rallying_to.is_some())
        .filter(|a| {
            // Find their party and check if it's in a region-owned location.
            state.parties.iter().any(|p| {
                p.member_ids.contains(&a.id)
                    && matches!(p.status, PartyStatus::Traveling)
                    && is_party_in_region_locations(p, &region_location_ids, state)
            })
        })
        .map(|a| a.id)
        .collect();

    ScoutReportDetails {
        region_name: region.name.clone(),
        faction_military_strength: faction_strength,
        unrest: region.unrest,
        threat_level: region.threat_level,
        quest_opportunities,
        champion_sightings,
    }
}

/// Check if a party is near any of the given location IDs.
fn is_party_in_region_locations(
    party: &Party,
    location_ids: &[usize],
    state: &CampaignState,
) -> bool {
    for &lid in location_ids {
        if let Some(loc) = state.overworld.locations.iter().find(|l| l.id == lid) {
            if dist_sq(party.position, loc.position) < 100.0 {
                return true;
            }
        }
    }
    false
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}
