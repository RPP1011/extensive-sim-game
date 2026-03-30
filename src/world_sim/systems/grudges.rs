#![allow(unused)]
//! Adventurer grudge/vendetta system — delta architecture port.
//!
//! NPCs develop grudges against factions, nemeses, or regions from traumatic
//! events. Grudges drive bonus combat power against targets, stress in
//! traumatic regions, and reckless behavior at high intensity. Grudges decay
//! over time and resolve when the target is defeated.
//!
//! Original: `crates/headless_campaign/src/systems/grudges.rs`
//! Cadence: every 10 ticks, with decay sub-gate at every 17 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::entity_hash_f32;

//   Grudge { id, adventurer_id, target: GrudgeTarget, intensity, cause, formed_tick, resolved }
//   GrudgeTarget { Faction(id), Region(id), Nemesis(id) }

/// Combat power bonus multiplier when fighting a grudge target.
const GRUDGE_COMBAT_BONUS: f32 = 0.20;

/// Stress added per tick when entity is in a grudge region.
const GRUDGE_REGION_STRESS: f32 = 3.0;

/// Intensity threshold for reckless behavior.
const RECKLESS_THRESHOLD: f32 = 70.0;

/// Ticks between decay steps.
const DECAY_INTERVAL: u64 = 17;

/// Intensity lost per decay step.
const DECAY_AMOUNT: f32 = 1.0;

/// Ticks after which unresolved grudge causes bitterness.
const BITTERNESS_THRESHOLD: u64 = 167;

/// Morale penalty for bitterness.
const BITTERNESS_MORALE_PENALTY: f32 = 5.0;

/// Morale bonus when grudge resolved via vengeance.
const VENGEANCE_MORALE_BONUS: f32 = 20.0;

/// Max grudges per entity.
const MAX_GRUDGES_PER_ENTITY: usize = 5;

/// Cadence gate.
const GRUDGE_TICK_INTERVAL: u64 = 10;

/// Compute grudge deltas: detection, region stress, decay, bitterness, fulfillment.
///
/// Since WorldState lacks grudge/nemesis storage, this is a structural
/// placeholder. The mapped delta emissions are documented below.
pub fn compute_grudges(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GRUDGE_TICK_INTERVAL != 0 {
        return;
    }

    // --- 1. Near-death grudge detection ---
    // For each NPC entity with hp < 20% of max_hp on an active grid:
    //   15% deterministic chance → CreateGrudge(Region, intensity=40)
    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        let hp_ratio = entity.hp / entity.max_hp.max(1.0);
        if hp_ratio < 0.2 && entity.grid_id.is_some() {
            let _roll = entity_hash_f32(entity.id, state.tick, 0);
        }
    }

    // --- 2. Region stress ---
    //   out.push(WorldDelta::AdjustStress { entity_id, delta: GRUDGE_REGION_STRESS })

    // --- 3. Grudge decay (every DECAY_INTERVAL ticks) ---
    if state.tick % DECAY_INTERVAL == 0 {
        //   out.push(WorldDelta::UpdateGrudge { grudge_id, intensity_delta: -DECAY_AMOUNT })
        //   if intensity <= 0: out.push(WorldDelta::ResolveGrudge { grudge_id })
    }

    // --- 4. Bitterness from old unresolved grudges ---
    //   out.push(WorldDelta::AdjustMorale { entity_id, delta: -BITTERNESS_MORALE_PENALTY })

    // --- 5. Vendetta fulfillment ---
    //   out.push(WorldDelta::ResolveGrudge { grudge_id })
    //   out.push(WorldDelta::AdjustMorale { entity_id, delta: VENGEANCE_MORALE_BONUS })
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Combat power bonus for entity against a given faction.
/// Requires grudge state.
pub fn grudge_combat_bonus(_entity_id: u32, _enemy_faction_id: u32) -> f32 {
    0.0
}

/// Whether entity acts recklessly due to high-intensity grudge.
pub fn acts_recklessly(_entity_id: u32, _enemy_faction_id: u32) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Deterministic RNG helper
// ---------------------------------------------------------------------------

