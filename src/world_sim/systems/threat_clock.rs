#![allow(unused)]
//! Global threat clock — escalating pressure that drives the world forward.
//!
//! The threat clock is a global counter that grows each interval. As it rises,
//! monster density increases across all regions, forcing settlements to invest
//! in defense or risk being overrun. Adventuring NPCs who clear threats push
//! the clock back temporarily.
//!
//! This creates the core DF-style pressure loop:
//! - Clock rises → monsters spawn → settlements threatened
//! - NPCs respond by specializing in combat → clear threats → clock pauses
//! - If NPCs don't respond → settlements fall → population drops → pressure mounts
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const CLOCK_INTERVAL: u64 = 100;

/// Base growth rate per interval. The clock always ticks up unless actively pushed back.
const BASE_GROWTH: f32 = 0.05;

/// Growth acceleration per 1000 ticks elapsed (threat ramps up over time).
const GROWTH_ACCELERATION: f32 = 0.0001;

/// Monster density increase per clock level per region, per interval.
const DENSITY_PER_CLOCK: f32 = 0.002;

/// Clock thresholds for chronicle events.
const THRESHOLD_WARNING: f32 = 25.0;
const THRESHOLD_RISING: f32 = 50.0;
const THRESHOLD_CRITICAL: f32 = 75.0;
const THRESHOLD_DOOM: f32 = 90.0;

/// Window size for threshold crossing detection (only fire once).
const WINDOW: f32 = 2.0;

pub fn compute_threat_clock(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CLOCK_INTERVAL != 0 || state.tick == 0 { return; }

    // Current global threat clock is the average threat across all regions.
    let avg_threat = if state.regions.is_empty() {
        0.0
    } else {
        state.regions.iter().map(|r| r.threat_level).sum::<f32>() / state.regions.len() as f32
    };

    // Growth rate accelerates over time.
    let growth = BASE_GROWTH + (state.tick as f32 * GROWTH_ACCELERATION / 1000.0);

    // Counterbalance: combat-specialized NPCs reduce growth.
    // Count NPCs with significant combat tags.
    let combat_npcs = state.entities.iter().filter(|e| {
        e.alive && e.kind == EntityKind::Npc
            && e.npc.as_ref().map_or(false, |n| n.behavior_value(tags::COMBAT) > 100.0)
    }).count() as f32;

    // Each combat NPC reduces growth. Population health also matters.
    let alive_npcs = state.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count() as f32;
    let pop_factor = (alive_npcs / 1000.0).clamp(0.2, 2.0); // healthy pop = more suppression
    let suppression = combat_npcs * 0.05 * pop_factor;

    let net_growth = (growth - suppression).max(0.0); // can be fully suppressed

    // Apply density increase to regions that aren't already at cap.
    const MAX_MONSTER_DENSITY: f32 = 1.5;
    for region in &state.regions {
        if region.monster_density >= MAX_MONSTER_DENSITY { continue; }
        let density_increase = net_growth * DENSITY_PER_CLOCK;
        if density_increase > 0.001 {
            out.push(WorldDelta::UpdateRegion {
                region_id: region.id,
                field: RegionField::MonsterDensity,
                value: density_increase,
            });
        }
    }

    // Chronicle entries at threshold crossings.
    let new_avg = avg_threat + net_growth;
    for &(threshold, category, msg) in &[
        (THRESHOLD_WARNING, ChronicleCategory::Narrative, "Dark stirrings are felt across the land. Threat is rising."),
        (THRESHOLD_RISING, ChronicleCategory::Crisis, "The threat grows serious. Monsters roam freely between settlements."),
        (THRESHOLD_CRITICAL, ChronicleCategory::Crisis, "A darkness grips the world. Settlements are under siege from all sides."),
        (THRESHOLD_DOOM, ChronicleCategory::Crisis, "The world teeters on the edge of ruin. Only heroes can turn the tide."),
    ] {
        if avg_threat < threshold && new_avg >= threshold && new_avg < threshold + WINDOW {
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category,
                    text: msg.to_string(),
                    entity_ids: vec![],
                },
            });
        }
    }
}
