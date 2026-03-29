#![allow(unused)]
//! Guild identity/specialization system — delta architecture port.
//!
//! The guild develops a distinct identity across five axes: martial,
//! mercantile, scholarly, diplomatic, shadowy. Each ranges 0-100 and
//! decays toward 10 (neutral) when unexercised. When the highest axis
//! exceeds 50, the guild acquires a dominant identity type that unlocks
//! unique bonuses.
//!
//! Original: `crates/headless_campaign/src/systems/guild_identity.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: guild_identity on WorldState
//   GuildIdentity { martial, mercantile, scholarly, diplomatic, shadowy, dominant: Option<IdentityType> }
//   IdentityType: WarriorsGuild, MerchantCompany, ScholarOrder, DiplomaticCorps, ShadowNetwork
// NEEDS STATE: completed quests with types, active battles, trade routes, spies, festivals
// NEEDS DELTA: UpdateIdentityAxis { axis, delta }
// NEEDS DELTA: SetDominantIdentity { identity_type }

/// Cadence gate.
const IDENTITY_TICK_INTERVAL: u64 = 17;

/// Influence gain per matching action.
const ACTION_INFLUENCE_GAIN: f32 = 2.0;

/// Passive decay rate toward neutral each tick.
const DECAY_RATE: f32 = 0.5;

/// Neutral resting value.
const NEUTRAL: f32 = 10.0;

/// Threshold for dominant identity.
const DOMINANT_THRESHOLD: f32 = 50.0;

/// Tension suppression for opposing identity pairs.
const TENSION_SUPPRESSION: f32 = 0.3;

/// Compute guild identity deltas: tally actions, apply gains/decay/tension.
///
/// Identity axis changes would be expressed via UpdateIdentityAxis deltas.
/// The economic effects of identity (trade income for MerchantCompany,
/// gold boost) can use UpdateTreasury.
pub fn compute_guild_identity(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % IDENTITY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Tally identity-relevant actions ---
    // NEEDS STATE: completed quests, active battles, trade routes, etc.
    //
    // Combat quests/battles → martial
    // Escort/Gather quests, trade routes → mercantile
    // Exploration quests, dungeons → scholarly
    // Diplomatic quests, festivals → diplomatic
    // Spies, black market → shadowy

    // Proxy: use region threat levels and settlement counts as identity signals
    let mut martial_signal = 0.0f32;
    let mut mercantile_signal = 0.0f32;

    for region in &state.regions {
        if region.threat_level > 50.0 {
            martial_signal += ACTION_INFLUENCE_GAIN; // high threat = martial focus
        }
    }
    for settlement in &state.settlements {
        if settlement.treasury > 100.0 {
            mercantile_signal += ACTION_INFLUENCE_GAIN * 0.5; // wealth = mercantile
        }
    }

    // --- Phase 2: Apply gains, decay, and tension ---
    // out.push(WorldDelta::UpdateIdentityAxis { axis: "martial", delta: martial_signal })
    // For axes with no activity: decay toward NEUTRAL
    // Opposing tension: martial↔diplomatic, mercantile↔shadowy

    // --- Phase 3: Determine dominant identity ---
    // If best axis > DOMINANT_THRESHOLD:
    //   out.push(WorldDelta::SetDominantIdentity { identity_type })

    // --- Phase 4: Apply identity bonuses ---
    // MerchantCompany: +20% trade income → UpdateTreasury per trade route
    //   out.push(WorldDelta::UpdateTreasury { location_id, delta: route_bonus })
    // ScholarOrder: +20% XP (applied by consuming systems)
    // DiplomaticCorps: +15% relation gains (NEEDS DELTA: UpdateRelation)
    // ShadowNetwork: faster heat decay (NEEDS STATE: black_market)
    // WarriorsGuild: combat bonus (applied passively by combat systems)

    let _ = martial_signal;
    let _ = mercantile_signal;
}

/// Decay a value toward `target` by `rate`.
fn decay_toward(value: f32, target: f32, rate: f32) -> f32 {
    if value > target {
        (value - rate).max(target)
    } else if value < target {
        (value + rate).min(target)
    } else {
        value
    }
}

/// Public helper: returns the combat power multiplier from guild identity.
pub fn identity_combat_bonus(_dominant_is_warriors: bool) -> f32 {
    // NEEDS STATE: guild_identity.dominant
    if _dominant_is_warriors {
        0.15
    } else {
        0.0
    }
}
