//! Site preparation system — every 100 ticks (~10s).
//!
//! Players invest resources into preparing locations for encounters:
//! - **Defensive** — walls, traps, watchtowers → combat defense bonus when defending
//! - **Offensive** — ambush positions, hidden caches, escape routes → attack bonus
//! - **Logistical** — supply depots, forward camps → reduces supply drain, extends range
//!
//! Active preparations with an assigned party grow by 2.0 per tick.
//! Unassigned preparations decay by 0.5 per tick (unmaintained).
//! Enemy faction capture resets all preparations in that region.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{CampaignState, PrepType, SitePreparation};

/// Cadence: runs every 100 ticks.
const SITE_PREP_INTERVAL: u64 = 3;

/// Level increase per tick when a party is assigned.
const GROWTH_PER_TICK: f32 = 2.0;

/// Level decay per tick when no party is assigned.
const DECAY_PER_TICK: f32 = 0.5;

/// Maximum preparation level.
const MAX_LEVEL: f32 = 100.0;

/// Gold cost to establish a new site preparation.
pub const SITE_PREP_COST: f32 = 30.0;

/// Returns the defensive combat modifier for a region based on its defensive
/// site preparation level. Applied when the guild defends in this region.
pub fn defensive_modifier(level: f32) -> f32 {
    if level >= 80.0 {
        0.30
    } else if level >= 50.0 {
        0.20
    } else if level > 20.0 {
        0.10
    } else {
        0.0
    }
}

/// Returns the offensive combat modifier for a region based on its offensive
/// site preparation level. Applied when the guild attacks in this region.
pub fn offensive_modifier(level: f32) -> f32 {
    if level >= 80.0 {
        0.30
    } else if level >= 50.0 {
        0.20
    } else if level > 20.0 {
        0.10
    } else {
        0.0
    }
}

/// Returns the supply drain multiplier for a region based on its logistical
/// site preparation level. Lower is better (less drain).
pub fn logistical_drain_multiplier(level: f32) -> f32 {
    if level >= 80.0 {
        0.0 // party base — no drain
    } else if level >= 50.0 {
        0.50
    } else if level > 20.0 {
        0.75
    } else {
        1.0
    }
}

/// Aggregate the best preparation level of a given type for a region.
pub fn best_prep_level(preps: &[SitePreparation], region_id: usize, prep_type: PrepType) -> f32 {
    preps
        .iter()
        .filter(|p| p.region_id == region_id && p.prep_type == prep_type)
        .map(|p| p.level)
        .fold(0.0_f32, f32::max)
}

pub fn tick_site_prep(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SITE_PREP_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Update preparation levels.
    for prep in &mut state.site_preparations {
        if prep.assigned_party_id.is_some() {
            prep.level = (prep.level + GROWTH_PER_TICK).min(MAX_LEVEL);
        } else {
            prep.level = (prep.level - DECAY_PER_TICK).max(0.0);
        }
    }

    // Emit SiteCompleted events for preps that just hit 100.
    for prep in &state.site_preparations {
        // Check if level just reached max (was below max last tick).
        // Since growth is 2.0 per tick, "just reached" means level == 100 and
        // it was < 100 before this tick (i.e. level - GROWTH <= 98).
        if prep.level >= MAX_LEVEL && (prep.level - GROWTH_PER_TICK) < MAX_LEVEL {
            events.push(WorldEvent::SiteCompleted {
                site_id: prep.id,
                region_id: prep.region_id,
                prep_type: format!("{:?}", prep.prep_type),
            });
        }
    }

    // Remove preparations that have decayed to zero.
    state
        .site_preparations
        .retain(|p| p.level > 0.0);
}

/// Called when a region changes ownership — destroy all site preparations
/// in the captured region.
pub fn on_region_captured(
    state: &mut CampaignState,
    region_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    let destroyed: Vec<u32> = state
        .site_preparations
        .iter()
        .filter(|p| p.region_id == region_id)
        .map(|p| p.id)
        .collect();

    for site_id in &destroyed {
        events.push(WorldEvent::SiteDestroyed {
            site_id: *site_id,
            region_id,
        });
    }

    state
        .site_preparations
        .retain(|p| p.region_id != region_id);
}
