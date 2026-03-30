//! Guild charter / legitimacy — fires every 17 ticks.
//!
//! Charter legitimacy affects morale and desertion. High legitimacy grants
//! treasury bonuses; low legitimacy drains gold and damages NPCs (desertion
//! proxy). Maps to UpdateTreasury, Damage, and Heal deltas.
//!
//! Original: `crates/headless_campaign/src/systems/charter.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Charter check interval.
const CHARTER_INTERVAL: u64 = 17;

/// Treasury bonus for high legitimacy.
const HIGH_LEGITIMACY_GOLD_BONUS: f32 = 5.0;

/// Treasury drain for low legitimacy.
const LOW_LEGITIMACY_GOLD_DRAIN: f32 = 3.0;

pub fn compute_charter(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHARTER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without charter state, we use settlement treasury levels as a proxy
    // for legitimacy. Well-funded settlements gain a bonus; poorly-funded
    // settlements suffer drain.

    for settlement in &state.settlements {
        if settlement.treasury > 80.0 {
            // High legitimacy: small treasury bonus (virtuous cycle)
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: settlement.id,
                delta: HIGH_LEGITIMACY_GOLD_BONUS,
            });
        } else if settlement.treasury < 20.0 && settlement.treasury > -100.0 {
            // Low legitimacy: drain treasury further and damage NPCs (desertion)
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: settlement.id,
                delta: -LOW_LEGITIMACY_GOLD_DRAIN,
            });
        }
    }
}
