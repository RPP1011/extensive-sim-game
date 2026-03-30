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

//   GuildIdentity { martial, mercantile, scholarly, diplomatic, shadowy, dominant: Option<IdentityType> }
//   IdentityType: WarriorsGuild, MerchantCompany, ScholarOrder, DiplomaticCorps, ShadowNetwork

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
pub fn compute_guild_identity(_state: &WorldState, _out: &mut Vec<WorldDelta>) {
    // Stub: identity axis state not yet tracked. See git history for planned design.
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
    if _dominant_is_warriors {
        0.15
    } else {
        0.0
    }
}
