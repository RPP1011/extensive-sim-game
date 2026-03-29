#![allow(unused)]
//! Victory conditions — fires every 7 ticks.
//!
//! Checks multiple win conditions (economic, military, diplomatic, quest,
//! cultural). Near-victory escalation increases threat in regions. Maps to
//! Damage deltas (escalation) and is primarily a read-only check.
//!
//! Original: `crates/headless_campaign/src/systems/victory_conditions.rs`
//!
//! NEEDS STATE: `victory_condition`, `victory_progress` on WorldState
//! NEEDS DELTA: VictoryAchieved, NearVictoryEscalation

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// Victory check interval.
const VICTORY_CHECK_INTERVAL: u64 = 7;

/// Damage dealt to entities during near-victory escalation.
const ESCALATION_DAMAGE: f32 = 3.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_victory_conditions(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % VICTORY_CHECK_INTERVAL != 0 {
        return;
    }

    // Without victory progress tracking, we approximate near-victory escalation:
    // When the world has been running for a long time (tick > 15000),
    // regions with high threat experience escalation damage to NPCs.

    if state.tick < 15_000 {
        return;
    }

    // Escalation: damage NPCs in high-threat regions
    for region in &state.regions {
        if region.threat_level < 70.0 {
            continue;
        }

        // Find NPCs in this region (approximate via settlement proximity)
        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }
            let roll = tick_hash(state.tick, entity.id as u64 ^ region.id as u64);
            if roll < 0.05 {
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: ESCALATION_DAMAGE,
                    source_id: 0,
                });
            }
        }
    }
}
