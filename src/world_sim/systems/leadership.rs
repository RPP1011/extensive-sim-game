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
use crate::world_sim::state::{EntityKind, WorldState};

// NEEDS STATE: guild_leader: Option<GuildLeader> on WorldState
//   GuildLeader { adventurer_id, appointed_tick, leadership_style, approval_rating, decisions_made }
//   LeadershipStyle: Aggressive, Cautious, Diplomatic, Mercantile, Scholarly, Inspirational
// NEEDS STATE: succession_crisis: Option<SuccessionCrisis> on WorldState
//   SuccessionCrisis { crisis_start_tick, previous_leader_id }
// NEEDS STATE: adventurer morale, level, loyalty, history_tags, status
// NEEDS DELTA: AppointLeader { entity_id, style }
// NEEDS DELTA: TriggerSuccessionCrisis { previous_leader_id }
// NEEDS DELTA: ResolveSuccessionCrisis {}
// NEEDS DELTA: UpdateApproval { delta }
// NEEDS DELTA: AdjustMorale { entity_id, delta }

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

    // --- Handle succession crisis ---
    // NEEDS STATE: if succession_crisis is Some:
    //   if tick >= crisis_start + SUCCESSION_CRISIS_TICKS:
    //     Select new leader (highest level alive NPC)
    //     out.push(WorldDelta::AppointLeader { entity_id, style })
    //     out.push(WorldDelta::ResolveSuccessionCrisis)
    //   else:
    //     Apply morale penalty to all alive NPCs:
    //     for entity in alive_npcs:
    //       out.push(WorldDelta::AdjustMorale { entity_id, delta: -CRISIS_MORALE_PENALTY })

    // --- Check if current leader is alive ---
    // NEEDS STATE: guild_leader.adventurer_id
    // If leader entity is dead:
    //   out.push(WorldDelta::TriggerSuccessionCrisis { previous_leader_id })

    // --- Auto-appoint if no leader ---
    // Select highest-level alive NPC, break ties by loyalty (not available on Entity yet)
    let best_candidate = state
        .entities
        .iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .max_by_key(|e| e.level);

    // NEEDS STATE: check if guild_leader is None
    // if let Some(candidate) = best_candidate {
    //     out.push(WorldDelta::AppointLeader { entity_id: candidate.id, style: determined_from_tags })
    // }

    // --- Update approval rating ---
    // NEEDS STATE: guild_leader.approval_rating
    // Approval changes based on:
    //   Recent quest victories: +2 per win (last 5)
    //   Recent losses: -3 per loss (last 5)
    //   High guild treasury: +1
    //   Low guild treasury: -2
    //   Dead NPC count: -0.5 per dead
    //   Mean drift toward 50: (50 - approval) * 0.05
    // out.push(WorldDelta::UpdateApproval { delta: total })

    // --- Low approval retirement ---
    // approval < LOW_APPROVAL_THRESHOLD:
    //   Council succession (>= 3 NPCs at level 5+): short crisis
    //   No council: immediate replacement
    //   out.push(WorldDelta::TriggerSuccessionCrisis { previous_leader_id })

    let _ = best_candidate;
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
