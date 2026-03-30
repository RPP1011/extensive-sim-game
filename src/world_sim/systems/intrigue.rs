//! Political intrigue system — delta architecture port.
//!
//! Generates court intrigues within factions experiencing instability.
//! The guild can choose to support claimants, expose scandals, exploit
//! chaos, or stay neutral via choice events.
//!
//! Original: `crates/headless_campaign/src/systems/intrigue.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::entity_hash_f32;

//   PoliticalIntrigue { id, faction_id, intrigue_type, participants, guild_involvement, resolution_tick, resolved }
//   IntrigueType: SuccessionDispute, NobleRivalry, CourtScandal, PowerGrab, SecretAlliance, Assassination

/// Cadence gate.
const INTRIGUE_TICK_INTERVAL: u64 = 17;

/// Base chance per qualifying faction per tick of spawning an intrigue.
const BASE_INTRIGUE_CHANCE: f32 = 0.05;

/// Max concurrent active intrigues.
const MAX_ACTIVE_INTRIGUES: usize = 5;

/// Min/max ticks before intrigue resolves.
const MIN_RESOLUTION_TICKS: u64 = 33;
const MAX_RESOLUTION_TICKS: u64 = 67;

/// Compute intrigue deltas: resolve matured intrigues, generate new ones.
///
/// Since WorldState lacks intrigue/faction storage, this is a structural
/// placeholder. Gold/reputation effects from intrigue resolution could be
/// expressed via TransferGold and UpdateTreasury when the state is available.
pub fn compute_intrigue(state: &WorldState, _out: &mut Vec<WorldDelta>) {
    if state.tick % INTRIGUE_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Resolve matured intrigues ---
    //   Determine outcome based on guild_involvement level
    //   Guild-backed (involvement > 20): +20 faction relation, +50 gold
    //     out.push(WorldDelta::TransferGold { from_entity: faction, to_entity: guild, amount: 50.0 })
    //   Guild-exploited (involvement < -20): 40% caught → -20 faction relation
    //   Moderate involvement: 50% backed side wins
    //   Neutral: no effect
    //   out.push(WorldDelta::ResolveIntrigue { intrigue_id })

    // --- Generate new intrigues ---
    //   Skip if already has active intrigue, or at max capacity
    //   Deterministic roll < 0.05:
    //   Pick intrigue type weighted by faction situation
    //   out.push(WorldDelta::CreateIntrigue { faction_id, intrigue_type, resolution_tick })
    //   Present choice event to player

    // Structural: identify qualifying factions via region threat as proxy
    for region in &state.regions {
        if region.threat_level > 50.0 {
            let _roll = entity_hash_f32(region.id, state.tick, 0);
        }
    }
}

