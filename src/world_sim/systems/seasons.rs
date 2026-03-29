#![allow(unused)]
//! Seasonal cycle — every tick.
//!
//! Derives the current season from `state.tick` and applies per-tick
//! morale effects. Other systems can call `current_season()` /
//! `season_modifiers()` to read seasonal multipliers.
//!
//! Original: `crates/headless_campaign/src/systems/seasons.rs`
//!
//! NEEDS STATE: `season` field on WorldState (or derived from tick)
//! NEEDS STATE: `morale` field on Entity / NpcData for morale drift

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Ticks per season cycle (Spring -> Summer -> Autumn -> Winter).
/// At the world-sim tick rate this gives a reasonable in-game year.
pub const TICKS_PER_SEASON: u64 = 1200;

/// The four seasons, derived from tick count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

impl Season {
    pub fn next(self) -> Season {
        match self {
            Season::Spring => Season::Summer,
            Season::Summer => Season::Autumn,
            Season::Autumn => Season::Winter,
            Season::Winter => Season::Spring,
        }
    }
}

/// Multipliers and per-tick offsets for the current season.
#[derive(Clone, Copy, Debug)]
pub struct SeasonModifiers {
    /// Multiplier on travel speed (>1 = faster).
    pub travel_speed: f32,
    /// Multiplier on supply drain rate (>1 = more expensive).
    pub supply_drain: f32,
    /// Multiplier on threat calculations (>1 = more dangerous).
    pub threat: f32,
    /// Multiplier on recruitment chance (>1 = easier to recruit).
    pub recruit_chance: f32,
    /// Flat morale change applied per tick (positive = improving).
    pub morale_per_tick: f32,
}

/// Derive the current season from the world tick.
pub fn current_season(tick: u64) -> Season {
    let season_index = (tick / TICKS_PER_SEASON) % 4;
    match season_index {
        0 => Season::Spring,
        1 => Season::Summer,
        2 => Season::Autumn,
        3 => Season::Winter,
        _ => unreachable!(),
    }
}

/// Return the seasonal modifiers for the given season.
pub fn season_modifiers(season: Season) -> SeasonModifiers {
    match season {
        Season::Spring => SeasonModifiers {
            travel_speed: 1.0,
            supply_drain: 1.0,
            threat: 0.9,
            recruit_chance: 1.2,
            morale_per_tick: 0.1,
        },
        Season::Summer => SeasonModifiers {
            travel_speed: 1.1,
            supply_drain: 0.8,
            threat: 1.0,
            recruit_chance: 1.0,
            morale_per_tick: 0.05,
        },
        Season::Autumn => SeasonModifiers {
            travel_speed: 0.9,
            supply_drain: 1.1,
            threat: 1.1,
            recruit_chance: 0.9,
            morale_per_tick: -0.05,
        },
        Season::Winter => SeasonModifiers {
            travel_speed: 0.7,
            supply_drain: 1.5,
            threat: 1.3,
            recruit_chance: 0.6,
            morale_per_tick: -0.1,
        },
    }
}

/// Compute seasonal deltas. Currently a no-op in terms of deltas because:
/// - Season is derived from tick (pure function, no state mutation needed).
/// - Morale drift would require a morale field on NpcData or a Heal-like delta.
///
/// Other systems should call `current_season(state.tick)` and
/// `season_modifiers(season)` to apply seasonal effects in their own deltas.
pub fn compute_seasons(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Season is derived from state.tick — no delta needed for season transition.
    //
    // Morale drift: the original system modified `adv.morale` per tick.
    // In the delta architecture, morale would be expressed as a status effect
    // or a dedicated delta variant. For now we note the need and let downstream
    // systems (weather, threat) incorporate season_modifiers() directly.
    //
    // NEEDS DELTA: MoraleDrift { entity_id, amount } — if we want per-NPC morale
}
