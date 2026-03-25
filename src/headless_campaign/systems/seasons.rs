//! Seasonal weather system — cycles every 1200 turns (~1 hour game time).
//!
//! Affects travel speed, supply consumption, threat level, recruitment
//! chance, and adventurer morale via `SeasonModifiers`.
//! Other systems read modifiers via `season_modifiers()` rather than
//! being modified directly.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Turns per season cycle (Spring → Summer → Autumn → Winter).
/// At 3s/turn: 1200 turns = 1 hour of game time per season, 4 hours per year.
pub const TICKS_PER_SEASON: u64 = 1200;

/// Multipliers and per-tick offsets for the current season.
#[derive(Clone, Copy, Debug)]
pub struct SeasonModifiers {
    /// Multiplier on party travel speed (>1 = faster).
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

/// Advance the season clock and apply per-tick morale drift.
pub fn tick_seasons(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // Advance the season tick counter
    state.overworld.season_tick += 1;

    // Check for season transition
    if state.overworld.season_tick >= TICKS_PER_SEASON {
        state.overworld.season_tick = 0;
        let new_season = state.overworld.season.next();
        state.overworld.season = new_season;
        events.push(WorldEvent::SeasonChanged { new_season });
    }

    // Apply morale drift to all adventurers
    let mods = season_modifiers(state.overworld.season);
    for adv in &mut state.adventurers {
        adv.morale = (adv.morale + mods.morale_per_tick).clamp(0.0, 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn season_cycles_correctly() {
        assert_eq!(Season::Spring.next(), Season::Summer);
        assert_eq!(Season::Summer.next(), Season::Autumn);
        assert_eq!(Season::Autumn.next(), Season::Winter);
        assert_eq!(Season::Winter.next(), Season::Spring);
    }

    #[test]
    fn default_season_is_spring() {
        assert_eq!(Season::default(), Season::Spring);
    }

    #[test]
    fn modifiers_are_reasonable() {
        for season in [Season::Spring, Season::Summer, Season::Autumn, Season::Winter] {
            let m = season_modifiers(season);
            assert!(m.travel_speed > 0.0 && m.travel_speed <= 2.0);
            assert!(m.supply_drain > 0.0 && m.supply_drain <= 2.0);
            assert!(m.threat > 0.0 && m.threat <= 2.0);
            assert!(m.recruit_chance > 0.0 && m.recruit_chance <= 2.0);
        }
    }
}
