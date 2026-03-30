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

//   Prisoner { id, name, faction_id, strength, captured_tick, ransom_value, escape_chance, loyalty_shift }

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
pub fn compute_prisoners(_state: &WorldState, _out: &mut Vec<WorldDelta>) {
    // Stub: prisoner state not yet tracked. See git history for planned design.
}
