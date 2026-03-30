//! Chronicle / narrative log system — delta architecture port.
//!
//! Records significant campaign events as narrative entries:
//! - Settlement economic milestones (treasury crossing thresholds)
//! - Population milestones (settlement population crossing thresholds)
//!
//! Uses the narrow-window technique from `legendary_deeds.rs` to fire
//! events exactly once without storing explicit "already fired" state.
//!
//! Cadence: every 3 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ChronicleCategory, ChronicleEntry, WorldState};

/// Cadence: chronicle scans every 500 ticks (rarer to avoid spam).
const CHRONICLE_INTERVAL: u64 = 500;

/// Treasury thresholds for economic milestones.
const TREASURY_THRESHOLDS: &[(f32, &str)] = &[
    (1_000.0, "crossed 1,000 gold in treasury"),
    (5_000.0, "crossed 5,000 gold in treasury"),
    (10_000.0, "crossed 10,000 gold in treasury"),
    (50_000.0, "crossed 50,000 gold in treasury"),
    (100_000.0, "crossed 100,000 gold in treasury"),
];

/// Population thresholds for population milestones.
const POPULATION_THRESHOLDS: &[(u32, &str)] = &[
    (100, "population reached 100"),
    (200, "population reached 200"),
    (500, "population reached 500"),
];

/// Narrow window size for treasury detection (same concept as legendary_deeds).
/// Treasury must be in [threshold, threshold + WINDOW) to fire.
const TREASURY_WINDOW: f32 = 500.0;

/// Narrow window size for population detection.
/// Population must be in [threshold, threshold + WINDOW) to fire.
const POPULATION_WINDOW: u32 = 10;

/// Compute chronicle deltas by scanning world state for recordable conditions.
pub fn compute_chronicle(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHRONICLE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Settlement economic milestones (deduplicated) ---
    for settlement in &state.settlements {
        for &(threshold, desc) in TREASURY_THRESHOLDS {
            if settlement.treasury >= threshold
                && settlement.treasury < threshold + TREASURY_WINDOW
            {
                let already_recorded = state.chronicle.iter().rev().take(1000).any(|e| {
                    e.category == ChronicleCategory::Economy
                        && e.text.contains(&settlement.name)
                        && e.text.contains(desc)
                });
                if already_recorded { continue; }

                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: ChronicleCategory::Economy,
                        text: format!("{} {}", settlement.name, desc),
                        entity_ids: vec![],
                    },
                });
            }
        }

        // --- Settlement population milestones (once per threshold) ---
        for &(threshold, desc) in POPULATION_THRESHOLDS {
            if settlement.population >= threshold
                && settlement.population < threshold + POPULATION_WINDOW
            {
                // Only record if not already in chronicle for this settlement + threshold.
                let already_recorded = state.chronicle.iter().rev().take(1000).any(|e| {
                    e.category == ChronicleCategory::Economy
                        && e.text.contains(&settlement.name)
                        && e.text.contains(desc)
                });
                if already_recorded { continue; }

                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: ChronicleCategory::Economy,
                        text: format!("{} {}", settlement.name, desc),
                        entity_ids: vec![],
                    },
                });
            }
        }
    }
}
