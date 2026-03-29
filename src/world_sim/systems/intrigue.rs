#![allow(unused)]
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

// NEEDS STATE: intrigues: Vec<PoliticalIntrigue> on WorldState
//   PoliticalIntrigue { id, faction_id, intrigue_type, participants, guild_involvement, resolution_tick, resolved }
//   IntrigueType: SuccessionDispute, NobleRivalry, CourtScandal, PowerGrab, SecretAlliance, Assassination
// NEEDS STATE: factions with diplomatic_stance, military_strength, at_war_with, relationship_to_guild
// NEEDS STATE: guild gold, reputation
// NEEDS DELTA: CreateIntrigue { faction_id, intrigue_type, resolution_tick }
// NEEDS DELTA: ResolveIntrigue { intrigue_id }
// NEEDS DELTA: UpdateRelation { faction_id, delta }
// NEEDS DELTA: AdjustReputation { delta }

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
pub fn compute_intrigue(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % INTRIGUE_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Resolve matured intrigues ---
    // NEEDS STATE: for each unresolved intrigue where tick >= resolution_tick:
    //   Determine outcome based on guild_involvement level
    //   Guild-backed (involvement > 20): +20 faction relation, +50 gold
    //     out.push(WorldDelta::TransferGold { from_id: faction, to_id: guild, amount: 50.0 })
    //   Guild-exploited (involvement < -20): 40% caught → -20 faction relation
    //   Moderate involvement: 50% backed side wins
    //   Neutral: no effect
    //   out.push(WorldDelta::ResolveIntrigue { intrigue_id })

    // --- Generate new intrigues ---
    // NEEDS STATE: for each faction with instability (at war, low strength, hostile stance):
    //   Skip if already has active intrigue, or at max capacity
    //   Deterministic roll < 0.05:
    //   Pick intrigue type weighted by faction situation
    //   out.push(WorldDelta::CreateIntrigue { faction_id, intrigue_type, resolution_tick })
    //   Present choice event to player

    // Structural: identify qualifying factions via region threat as proxy
    for region in &state.regions {
        if region.threat_level > 50.0 {
            let _roll = deterministic_roll(state.tick, region.id);
            // NEEDS DELTA: CreateIntrigue
        }
    }
}

fn deterministic_roll(tick: u64, id: u32) -> f32 {
    let h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(id as u64);
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xff51afd7ed558ccd);
    let h = h ^ (h >> 33);
    (h & 0xFFFF) as f32 / 65536.0
}
