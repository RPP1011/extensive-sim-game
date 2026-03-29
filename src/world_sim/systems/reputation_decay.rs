#![allow(unused)]
//! Reputation decay system — delta architecture port.
//!
//! Guild/faction reputation decays over time unless actively maintained.
//! Decay is accelerated by inactivity, corruption, losses, and negative
//! rumors, and reduced by recent victories, propaganda, and guild tier.
//!
//! Original: `crates/headless_campaign/src/systems/reputation_decay.rs`
//! Cadence: every 7 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: guild_reputation: f32 on WorldState (or on a dedicated GuildState)
// NEEDS STATE: reputation_history: ReputationHistory on WorldState
//   where ReputationHistory { recent_deeds: Vec<ReputationDeed>, trend: ReputationTrend, maintenance_cost: f32 }
// NEEDS STATE: completed_quests: Vec<CompletedQuest> on WorldState (for inactivity/loss checks)
// NEEDS STATE: active_crises: Vec<ActiveCrisis> on WorldState (for corruption check)
// NEEDS STATE: rumors: Vec<Rumor> on WorldState (for negative rumor check)
// NEEDS STATE: propaganda_campaigns: Vec<PropagandaCampaign> on WorldState
// NEEDS STATE: active_festivals: Vec<Festival> on WorldState
// NEEDS STATE: guild_tier: GuildTier on WorldState
// NEEDS DELTA: AdjustReputation { faction_id: u32, delta: f32, reason: String }

/// Cadence: reputation decay every 7 ticks.
const TICK_CADENCE: u64 = 7;

/// Base reputation decay per cycle.
const BASE_DECAY: f32 = 0.1;

/// Reputation floor — won't decay below this.
const REPUTATION_FLOOR: f32 = 10.0;

/// Ticks without quest completion before decay accelerates.
const INACTIVITY_THRESHOLD: u64 = 67;

/// Corruption threshold above which scandal penalty applies.
const CORRUPTION_THRESHOLD: f32 = 50.0;

/// Recent ticks to look back for battle losses.
const RECENT_LOSS_WINDOW: u64 = 33;

/// Recent ticks to look back for quest completions (reducer).
const RECENT_QUEST_WINDOW: u64 = 17;

/// Compute reputation decay deltas.
///
/// Reads guild reputation, quest history, corruption, rumors, propaganda,
/// festivals, and guild tier to calculate net reputation decay. Emits a
/// single delta for the net change (if any).
///
/// Since WorldState lacks guild/reputation fields, this is a structural
/// placeholder. The settlement treasury can serve as a partial proxy:
/// settlements with high treasury lose less reputation (they're prosperous).
pub fn compute_reputation_decay(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    // NEEDS STATE: guild_reputation on WorldState
    // if current_rep <= REPUTATION_FLOOR { return; }

    // --- Calculate base decay ---
    // let mut decay = BASE_DECAY;

    // --- Decay accelerators ---
    // 1. No quest completed in INACTIVITY_THRESHOLD ticks → decay doubles
    // NEEDS STATE: completed_quests
    //   let ticks_since_quest = state.tick - last_quest_tick;
    //   if ticks_since_quest > INACTIVITY_THRESHOLD { decay += BASE_DECAY; }

    // 2. Corruption severity > CORRUPTION_THRESHOLD → +0.3
    // NEEDS STATE: active_crises (corruption check)

    // 3. Recent battle losses → +0.2 per loss
    // NEEDS STATE: completed_quests with defeat results

    // 4. Negative rumors → +0.1 per negative rumor
    // NEEDS STATE: rumors vec

    // --- Decay reducers ---
    // let mut reduction_mult = 1.0f32;

    // 1. Recent quest completions → ×0.5
    // 2. Active propaganda → ×0.7
    // 3. War memorial festivals → ×0.9 each
    // 4. Guild tier bonus: Silver ×0.9, Gold ×0.8, Legendary ×0.7

    // --- Apply net decay ---
    // let net_decay = decay * reduction_mult;
    // let actual_decay = net_decay.min(current_rep - REPUTATION_FLOOR);
    // if actual_decay > 0.001:
    //   out.push(WorldDelta::AdjustReputation {
    //     faction_id: guild_faction_id,
    //     delta: -actual_decay,
    //     reason: reasons.join(", "),
    //   });

    // --- Propaganda maintenance cost ---
    // If active propaganda, deduct gold from guild treasury.
    // NEEDS STATE: propaganda_campaigns, guild gold
    // Can use UpdateTreasury for the guild's settlement:
    //   out.push(WorldDelta::UpdateTreasury {
    //     location_id: guild_settlement_id,
    //     delta: -maintenance_cost,
    //   });
}

// ---------------------------------------------------------------------------
// Reputation trend tracking (pure query)
// ---------------------------------------------------------------------------

/// Reputation trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReputationTrend {
    Rising,
    Stable,
    Falling,
}

/// Thresholds for trend classification.
const TRAJECTORY_RISING_THRESHOLD: f32 = 0.5;
const TRAJECTORY_FALLING_THRESHOLD: f32 = -0.5;

/// Classify reputation trend from a net change over a window.
pub fn classify_trend(net_change: f32) -> ReputationTrend {
    if net_change > TRAJECTORY_RISING_THRESHOLD {
        ReputationTrend::Rising
    } else if net_change < TRAJECTORY_FALLING_THRESHOLD {
        ReputationTrend::Falling
    } else {
        ReputationTrend::Stable
    }
}
