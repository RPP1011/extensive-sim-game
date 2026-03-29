#![allow(unused)]
//! Choice event generation — fires every 10 ticks.
//!
//! Generates branching decisions from templates. In the delta architecture,
//! choices are primarily narrative/UI events. The only mechanical effect is
//! gold or treasury changes from choice outcomes.
//!
//! Original: `crates/headless_campaign/src/systems/choices.rs`
//!
//! NEEDS STATE: `pending_choices: Vec<ChoiceEvent>` on WorldState
//! NEEDS STATE: `choice_templates` registry
//! NEEDS DELTA: PresentChoice, ResolveChoice

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// World event choices check cadence.
const WORLD_EVENT_INTERVAL: u64 = 10;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_choices(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick == 0 || state.tick % WORLD_EVENT_INTERVAL != 0 {
        return;
    }

    // Without choice templates and pending_choices on WorldState, we approximate:
    // Occasionally generate small gold windfalls or costs to settlements as
    // "resolved choice outcomes".

    for settlement in &state.settlements {
        let roll = tick_hash(state.tick, settlement.id as u64 ^ 0xC401CE);
        if roll < 0.05 {
            // Random event: small gold change (positive or negative)
            let amount_roll = tick_hash(state.tick, settlement.id as u64 ^ 0xA3B7);
            let amount = if amount_roll < 0.5 { 10.0 } else { -5.0 };
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: amount,
            });
        }
    }
}
