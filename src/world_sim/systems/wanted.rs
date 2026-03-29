#![allow(unused)]
//! Wanted poster system — delta architecture port.
//!
//! Factions issue bounties on guild NPCs who've wronged them. Wanted NPCs
//! face increased ambush chance in hostile territory. Posters expire after
//! 167 ticks or can be resolved through payment, combat, or diplomacy.
//!
//! Original: `crates/headless_campaign/src/systems/wanted.rs`
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: wanted_posters: Vec<WantedPoster> on WorldState
//   WantedPoster { id, adventurer_id, faction_id, bounty_amount, reason, posted_tick, hunters_dispatched }
// NEEDS STATE: factions with relationship_to_guild, military_strength
// NEEDS STATE: guild reputation
// NEEDS DELTA: IssuePoster { entity_id, faction_id, bounty }
// NEEDS DELTA: RemovePoster { poster_id }
// NEEDS DELTA: AdjustReputation { delta }
// NEEDS DELTA: CreateBattle { ... } (for hunter encounters)

/// Cadence gate.
const WANTED_TICK_INTERVAL: u64 = 10;

/// Poster expiry (ticks).
const POSTER_EXPIRY_TICKS: u64 = 167;

/// Faction relation threshold for diplomatic poster removal.
const DIPLOMACY_REMOVAL_THRESHOLD: f32 = 30.0;

/// Ambush chance increase for wanted NPCs in hostile territory.
const WANTED_AMBUSH_BONUS: f32 = 0.20;

/// Battle poster issuance chance.
const BATTLE_POSTER_CHANCE: f32 = 0.15;

/// Compute wanted poster deltas.
///
/// Since WorldState lacks wanted poster and faction relation storage,
/// this is a structural placeholder. Hunter encounter battles could be
/// expressed via Damage/Die deltas when the combat state is available.
pub fn compute_wanted(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % WANTED_TICK_INTERVAL != 0 {
        return;
    }

    // --- Phase 1: Check for new poster triggers ---
    // NEEDS STATE: recently completed quests/battles against hostile factions
    // Deterministic roll < BATTLE_POSTER_CHANCE:
    //   out.push(WorldDelta::IssuePoster { entity_id, faction_id, bounty })

    // --- Phase 2: Diplomatic removal ---
    // NEEDS STATE: for each poster where faction relation > DIPLOMACY_REMOVAL_THRESHOLD:
    //   out.push(WorldDelta::RemovePoster { poster_id })

    // --- Phase 3: Expire old posters ---
    // NEEDS STATE: for each poster where tick - posted_tick >= POSTER_EXPIRY_TICKS:
    //   out.push(WorldDelta::RemovePoster { poster_id })

    // --- Phase 4: Dispatch bounty hunters ---
    // NEEDS STATE: for each undispatched poster, if faction has enough strength:
    //   Mark dispatched, reduce faction military strength
    //   No delta needed — dispatching is a state flag

    // --- Phase 5: Hunter encounters ---
    // NEEDS STATE: wanted NPCs traveling through hostile territory
    //   Deterministic roll for encounter:
    //   Create battle encounter via Damage deltas or EscalateFidelity
    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        // NEEDS STATE: check if entity has an active wanted poster
        // NEEDS STATE: check if entity is in hostile faction territory
    }
}
