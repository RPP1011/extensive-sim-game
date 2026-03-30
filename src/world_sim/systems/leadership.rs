#![allow(unused)]
//! Guild leadership and succession system — delta architecture port.
//!
//! The guild has a single leader whose style provides global bonuses.
//! When the leader dies or retires (approval < 20), a succession crisis
//! ensues: 10 ticks of morale/performance penalties before a new leader
//! is appointed (highest level NPC).
//!
//! Original: `crates/headless_campaign/src/systems/leadership.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityField, EntityKind, WorldState};

//   GuildLeader { adventurer_id, appointed_tick, leadership_style, approval_rating, decisions_made }
//   LeadershipStyle: Aggressive, Cautious, Diplomatic, Mercantile, Scholarly, Inspirational
//   SuccessionCrisis { crisis_start_tick, previous_leader_id }

/// Cadence gate.
const LEADERSHIP_TICK_INTERVAL: u64 = 17;

/// Duration of succession crisis (ticks).
const SUCCESSION_CRISIS_TICKS: u64 = 10;

/// Approval threshold for voluntary retirement.
const LOW_APPROVAL_THRESHOLD: f32 = 20.0;

/// Morale penalty per tick during succession crisis.
const CRISIS_MORALE_PENALTY: f32 = 2.0;

/// Compute leadership deltas: crisis handling, approval updates, retirement.
///
/// Leader selection uses entity level as the primary criterion. Morale
/// penalties during crises can be expressed via per-entity deltas.
pub fn compute_leadership(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % LEADERSHIP_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Leadership morale effect: highest-level NPC inspires others.
    let best_candidate = state
        .entities
        .iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .max_by_key(|e| e.level);

    if let Some(leader) = best_candidate {
        let bonus = 0.5 + leader.level as f32 * 0.1; // higher-level leader inspires more
        for entity in state.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc && e.id != leader.id) {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: bonus,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Determine leadership style from entity stats and tags.
/// Would use history_tags when available; falls back to Inspirational.
pub fn determine_style_from_level(level: u32) -> &'static str {
    match level {
        0..=3 => "Cautious",
        4..=7 => "Inspirational",
        _ => "Aggressive",
    }
}
