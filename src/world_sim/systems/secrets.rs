#![allow(unused)]
//! Adventurer secret past system — delta architecture port.
//!
//! Some NPCs have hidden histories revealed over time. Suspicion grows
//! when they act inconsistently, and reveal triggers at suspicion > 80
//! or by random event. Reveal effects vary by secret type.
//!
//! Original: `crates/headless_campaign/src/systems/secrets.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: secret_past on Entity/NpcData
//   SecretPast { secret_type, revealed, reveal_tick, suspicion }
//   SecretType: ExiledNoble, FormerAssassin, RunawayHeir, CursedBloodline,
//               DeepCoverSpy, FallenPaladin, WitnessToCrime, HiddenMage
// NEEDS STATE: adventurer morale, stress, loyalty, stats on Entity/NpcData
// NEEDS STATE: guild gold, reputation
// NEEDS DELTA: RevealSecret { entity_id, secret_type }
// NEEDS DELTA: UpdateSuspicion { entity_id, delta }
// NEEDS DELTA: AdjustMorale { entity_id, delta }
// NEEDS DELTA: Die { entity_id } (for spy betrayal)

/// Cadence gate.
const SECRETS_TICK_INTERVAL: u64 = 17;

/// Chance that an NPC gets a secret past (15%).
const SECRET_ASSIGNMENT_CHANCE: f32 = 0.15;

/// Suspicion growth per tick when acting inconsistently.
const SUSPICION_GROWTH: f32 = 2.0;

/// Suspicion threshold that triggers reveal.
const REVEAL_THRESHOLD: f32 = 80.0;

/// Random reveal chance per tick (2%).
const RANDOM_REVEAL_CHANCE: f32 = 0.02;

/// Compute secret past deltas: assign secrets, grow suspicion, trigger reveals.
///
/// Since WorldState lacks secret storage on entities, this is a structural
/// placeholder. Reveals that involve gold theft (DeepCoverSpy betrayal)
/// can be expressed via TransferGold once entity gold is tracked.
pub fn compute_secrets(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SECRETS_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }

        // NEEDS STATE: check entity.npc.secret_past

        // --- Secret assignment (15% of NPCs without a secret) ---
        // Deterministic roll < SECRET_ASSIGNMENT_CHANCE:
        //   Assign random secret type based on hash(tick, entity_id)
        //   out.push(WorldDelta::UpdateSuspicion { entity_id, delta: 0.0 }) // init

        // --- Suspicion growth ---
        // Context-dependent growth rate:
        //   FormerAssassin: 2x during diplomatic quests
        //   ExiledNoble: 1.5x near home faction
        //   DeepCoverSpy: 1x steady
        //   CursedBloodline: 2x when stressed
        //   HiddenMage: 1.5x during combat
        // out.push(WorldDelta::UpdateSuspicion { entity_id, delta: bump })

        // --- Reveal trigger ---
        // suspicion > REVEAL_THRESHOLD || random_roll < RANDOM_REVEAL_CHANCE:
        //   out.push(WorldDelta::RevealSecret { entity_id, secret_type })
        //
        //   Effects by type:
        //   ExiledNoble: +20 faction relation, +10 loyalty
        //   FormerAssassin: combat stat boost, -10 morale
        //   DeepCoverSpy: 40% betrayal (steal gold + die), 60% loyal (+25 loyalty)
        //     Betrayal: out.push(WorldDelta::TransferGold { from: guild, to: spy, amount })
        //               out.push(WorldDelta::Die { entity_id })
        //   CursedBloodline: stat boost + stress, party morale penalty
        //   RunawayHeir: +10 reputation, +15 loyalty
        //   FallenPaladin: +20 resolve, +10 morale
        //   WitnessToCrime: +15 stress, -15 faction relation
        //   HiddenMage: combat stat boost

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
