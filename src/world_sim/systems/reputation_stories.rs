#![allow(unused)]
//! Reputation stories — fires every 17 ticks.
//!
//! Tales of deeds spread across regions, affecting settlement treasury and
//! NPC morale. Positive stories grant treasury bonuses; negative ones drain
//! treasury. Story accuracy degrades as it spreads (Chinese telephone).
//!
//! Original: `crates/headless_campaign/src/systems/reputation_stories.rs`
//!
//! NEEDS STATE: `reputation_stories: Vec<ReputationStory>` on WorldState
//! NEEDS DELTA: CreateStory, SpreadStory, FadeStory

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Story tick interval.
const STORY_INTERVAL: u64 = 17;

/// Treasury bonus from positive stories.
const POSITIVE_STORY_GOLD: f32 = 3.0;

/// Treasury drain from negative stories.
const NEGATIVE_STORY_DRAIN: f32 = 2.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_reputation_stories(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % STORY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without story tracking state, we approximate:
    // Settlements with high treasury (good reputation) gain a small bonus.
    // Settlements with very low treasury (bad reputation) suffer a drain.

    for settlement in &state.settlements {
        let roll = tick_hash(state.tick, settlement.id as u64 ^ 0x570E1);
        if settlement.treasury > 60.0 && roll < 0.15 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: POSITIVE_STORY_GOLD,
            });
        } else if settlement.treasury < 15.0 && settlement.treasury > -100.0 && roll < 0.20 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -NEGATIVE_STORY_DRAIN,
            });
        }
    }
}
