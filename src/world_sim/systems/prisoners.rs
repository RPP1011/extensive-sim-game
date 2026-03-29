#![allow(unused)]
//! Prisoner system — delta architecture port.
//!
//! Manages captured enemy prisoners: upkeep costs, escape attempts,
//! and captured adventurer ransom events.
//!
//! Original: `crates/headless_campaign/src/systems/prisoners.rs`
//! Cadence: every 7 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: prisoners: Vec<Prisoner> on WorldState
//   Prisoner { id, name, faction_id, strength, captured_tick, ransom_value, escape_chance, loyalty_shift }
// NEEDS STATE: captured_adventurers: Vec<u32> on WorldState
// NEEDS STATE: guild gold, reputation
// NEEDS DELTA: RemovePrisoner { prisoner_id }
// NEEDS DELTA: AdjustReputation { delta }
// NEEDS DELTA: CreateChoiceEvent { ... } (for ransom offers)

/// Cadence gate.
const PRISONER_TICK_INTERVAL: u64 = 7;

/// Upkeep cost per prisoner per tick.
const PRISONER_UPKEEP: f32 = 1.0;

/// Base escape chance per tick (5%).
const BASE_ESCAPE_CHANCE: f32 = 0.05;

/// Escape chance increase per 200-tick cycle held.
const ESCAPE_CHANCE_PER_TICK: f32 = 0.02;

/// Reputation penalty on prisoner escape.
const ESCAPE_REPUTATION_PENALTY: f32 = 5.0;

/// Compute prisoner deltas: upkeep costs, escape attempts, ransom offers.
///
/// Prisoner upkeep can be expressed as UpdateTreasury on the guild's
/// settlement. Escape attempts need prisoner state. Ransom offers need
/// choice events.
pub fn compute_prisoners(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PRISONER_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Prisoner upkeep ---
    // NEEDS STATE: prisoner_count = state.prisoners.len()
    // Express as treasury cost on the guild's home settlement:
    // if let Some(guild_settlement) = state.settlements.first() {
    //     let upkeep = prisoner_count as f32 * PRISONER_UPKEEP;
    //     out.push(WorldDelta::UpdateTreasury {
    //         location_id: guild_settlement.id,
    //         delta: -upkeep,
    //     });
    // }

    // --- Escape attempts ---
    // NEEDS STATE: for each prisoner:
    //   ticks_held = tick - captured_tick
    //   effective_chance = (base + cycles * per_tick).min(0.8)
    //   deterministic roll < effective_chance:
    //     out.push(WorldDelta::RemovePrisoner { prisoner_id })
    //     out.push(WorldDelta::AdjustReputation { delta: -ESCAPE_REPUTATION_PENALTY })

    // --- Captured adventurer ransom ---
    // NEEDS STATE: for each captured adventurer (10% chance):
    //   Present choice: pay ransom or refuse
    //   out.push(WorldDelta::CreateChoiceEvent { ... })

    // Structural: use settlement treasury as upkeep proxy
    // (no prisoners tracked yet, so nothing to emit)
}
