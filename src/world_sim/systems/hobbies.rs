#![allow(unused)]
//! Adventurer hobbies and downtime system — delta architecture port.
//!
//! Idle NPCs develop interests over time that provide passive bonuses.
//! Hobby selection is weighted by class tags; NPCs can have at most 2
//! hobbies. Gambling has special gold gain/loss mechanics.
//!
//! Original: `crates/headless_campaign/src/systems/hobbies.rs`
//! Cadence: every 7 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: hobbies: Vec<HobbyProgress> on Entity/NpcData
//   HobbyProgress { hobby: Hobby, skill_level, started_tick }
//   Hobby: Cooking, Cartography, Gambling, Herbalism, Smithing, Training, Meditation, Storytelling
// NEEDS STATE: class_tags on NpcData (used as archetype proxy)
// NEEDS DELTA: AssignHobby { entity_id, hobby }
// NEEDS DELTA: UpdateHobbySkill { entity_id, hobby, delta }

/// Cadence gate.
const HOBBY_TICK_INTERVAL: u64 = 7;

/// Ticks an NPC must be idle before picking a hobby.
const IDLE_THRESHOLD: u64 = 17;

/// Skill gain per hobby tick for idle NPCs.
const SKILL_GAIN_PER_TICK: f32 = 2.0;

/// Maximum hobbies per NPC.
const MAX_HOBBIES: usize = 2;

/// Compute hobby deltas: hobby selection, skill progression, gambling.
///
/// Since WorldState lacks hobby storage on entities, this is a structural
/// placeholder. Gambling gold gain/loss can be expressed via TransferGold.
pub fn compute_hobbies(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % HOBBY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        let npc = entity.npc.as_ref().unwrap();

        // Determine if NPC is "idle" (not on a High-fidelity combat grid)
        let in_combat = entity.grid_id.map_or(false, |gid| {
            state.grid(gid)
                .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
                .unwrap_or(false)
        });
        let at_settlement = !in_combat;

        if !at_settlement {
            continue;
        }

        // --- Phase 1: Hobby selection ---
        // NEEDS STATE: entity hobbies list
        // If hobbies.len() < MAX_HOBBIES && idle long enough:
        //   Build weighted pool from class_tags (warrior → Training/Smithing, etc.)
        //   Deterministic pick
        //   out.push(WorldDelta::AssignHobby { entity_id, hobby })

        // --- Phase 2: Skill progression ---
        // For idle NPCs with hobbies:
        //   out.push(WorldDelta::UpdateHobbySkill { entity_id, hobby, delta: SKILL_GAIN_PER_TICK })

        // --- Phase 3: Gambling effects ---
        // For NPCs with Gambling hobby:
        //   20% chance: +5 gold via TransferGold
        //   10% chance: -10 gold via TransferGold
        //   5% chance: create rivalry with another gambler (bond reduction)
        //
        // Gold changes expressible via TransferGold:
        //   out.push(WorldDelta::TransferGold {
        //       from_id: entity.id,       // or guild settlement
        //       to_id: settlement_id,     // or entity.id
        //       amount: 5.0,
        //   });

        let _class_tags = &npc.class_tags;
    }
}

/// Compute the passive bonus scale factor for a hobby.
/// Returns 0.0 at skill 0, ramps linearly to 1.0 at skill 50+.
pub fn hobby_bonus_scale(skill_level: f32) -> f32 {
    (skill_level / 50.0).min(1.0)
}
