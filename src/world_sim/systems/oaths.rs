#![allow(unused)]
//! Adventurer oath system — delta architecture port.
//!
//! NPCs can swear binding oaths granting powerful bonuses with strict
//! constraints. Breaking an oath carries loyalty, morale, and reputation
//! penalties plus an "oathbreaker" tag. Fulfilling grants a permanent
//! bonus and "oathkeeper" tag.
//!
//! Original: `crates/headless_campaign/src/systems/oaths.rs`
//! Cadence: every 10 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: oaths: Vec<Oath> on WorldState or Entity
//   Oath { id, adventurer_id, oath_type: OathType, sworn_tick, fulfilled, broken, bonus_active }
//   OathType: OathOfVengeance, OathOfProtection, OathOfPoverty, OathOfSilence, OathOfService, OathOfExploration
// NEEDS STATE: adventurer loyalty, morale, history_tags, level, status on Entity/NpcData
// NEEDS STATE: guild reputation
// NEEDS DELTA: BreakOath { oath_id }
// NEEDS DELTA: FulfillOath { oath_id }
// NEEDS DELTA: SwearOath { entity_id, oath_type }
// NEEDS DELTA: AdjustMorale { entity_id, delta }
// NEEDS DELTA: AdjustLoyalty { entity_id, delta }
// NEEDS DELTA: AdjustReputation { delta }

/// Cadence gate.
const OATH_TICK_INTERVAL: u64 = 10;

/// Loyalty threshold for voluntary oath proposals.
const OATH_PROPOSAL_LOYALTY: f32 = 70.0;

/// Chance per eligible NPC per tick to propose an oath (10%).
const OATH_PROPOSAL_CHANCE: f32 = 0.10;

/// Exploration deadline (ticks).
const EXPLORATION_DEADLINE_TICKS: u64 = 67;

/// Compute oath deltas: check violations, fulfillment, proposals.
///
/// Since WorldState lacks oath storage, this is a structural placeholder.
pub fn compute_oaths(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % OATH_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Check active oaths for violations and fulfillment ---
    // NEEDS STATE: iterate state.oaths where !broken && !fulfilled
    //
    // OathOfVengeance: violation if NPC idle while target faction has active quest
    // OathOfProtection: violation if ward has died
    // OathOfPoverty: violation if equipped item quality > common threshold
    // OathOfSilence: violation if assigned to diplomatic quest
    // OathOfService: fulfilled after 3+ completed quests since sworn
    // OathOfExploration: violation if idle too long without exploring

    // --- Phase 2: Broken oath penalties ---
    // out.push(WorldDelta::AdjustLoyalty { entity_id, delta: -20.0 })
    // out.push(WorldDelta::AdjustMorale { entity_id, delta: -15.0 })
    // out.push(WorldDelta::AdjustReputation { delta: -5.0 })
    // Add "oathbreaker" history tag

    // --- Phase 3: Fulfilled oath bonuses ---
    // out.push(WorldDelta::AdjustReputation { delta: 10.0 })
    // Add "oathkeeper" history tag

    // --- Phase 4: Oath proposals from high-loyalty NPCs ---
    // For each alive NPC with loyalty > 70 and no active oath:
    //   Deterministic roll < 0.10:
    //   out.push(WorldDelta::SwearOath { entity_id, oath_type })

    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        // NEEDS STATE: check if entity has an active oath
        // NEEDS STATE: check loyalty level
        // Deterministic proposal roll
        let _roll = deterministic_roll(state.tick, entity.id);
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
