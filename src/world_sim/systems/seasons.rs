//! Seasonal cycle — every tick.
//!
//! Derives the current season from `state.tick` and applies per-tick
//! morale effects. Other systems can call `current_season()` /
//! `season_modifiers()` to read seasonal multipliers.
//!
//! Original: `crates/headless_campaign/src/systems/seasons.rs`
//!

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
            morale_per_tick: -0.01, // slight autumn melancholy
        },
        Season::Winter => SeasonModifiers {
            travel_speed: 0.7,
            supply_drain: 1.5,
            threat: 1.3,
            recruit_chance: 0.6,
            morale_per_tick: -0.02, // gentler winter morale drain
        },
    }
}

/// Food production multiplier by season.
pub fn food_production_mult(season: Season) -> f32 {
    match season {
        Season::Spring => 1.5, // planting & early harvest
        Season::Summer => 1.2, // good growing weather
        Season::Autumn => 1.8, // main harvest
        Season::Winter => 0.2, // almost no production
    }
}

/// Food consumption multiplier by season (people eat more in cold).
pub fn food_consumption_mult(season: Season) -> f32 {
    match season {
        Season::Spring => 1.0,
        Season::Summer => 0.9,
        Season::Autumn => 1.0,
        Season::Winter => 1.4, // more calories needed in cold
    }
}

/// Price pressure multiplier by season (scarcity drives prices up).
pub fn price_pressure(season: Season) -> [f32; 8] {
    let mut p = [1.0f32; 8];
    match season {
        Season::Spring => {
            p[0] = 1.2; // food slightly scarce (reserves depleted from winter)
        }
        Season::Summer => {
            p[0] = 0.9; // food abundant
        }
        Season::Autumn => {
            p[0] = 0.7; // food very cheap (harvest)
            p[2] = 0.8; // wood cheap (logging season)
        }
        Season::Winter => {
            p[0] = 2.0; // food very expensive (scarcity)
            p[2] = 1.5; // wood expensive (heating fuel)
            p[7] = 1.5; // medicine expensive (illness season)
        }
    }
    p
}

/// Compute seasonal deltas — economic effects, morale drift, chronicle events.
pub fn compute_seasons(state: &WorldState, out: &mut Vec<WorldDelta>) {
    let season = current_season(state.tick);
    let mods = season_modifiers(season);

    // Record season transition in chronicle.
    if state.tick > 0 && state.tick % TICKS_PER_SEASON == 0 {
        let (name, flavor) = match season {
            Season::Spring => ("Spring", "The snows melt and new life stirs across the land."),
            Season::Summer => ("Summer", "The long days bring warmth, prosperity, and easier travel."),
            Season::Autumn => ("Autumn", "Leaves turn gold as the world braces for darker times ahead."),
            Season::Winter => ("Winter", "Bitter cold grips the world. Supplies dwindle and monsters grow bold."),
        };
        out.push(WorldDelta::RecordChronicle {
            entry: crate::world_sim::state::ChronicleEntry {
                tick: state.tick,
                category: crate::world_sim::state::ChronicleCategory::Narrative,
                text: format!("{} arrives. {}", name, flavor),
                entity_ids: vec![],
            },
        });

        out.push(WorldDelta::RecordEvent {
            event: crate::world_sim::state::WorldEvent::SeasonChanged {
                new_season: season as u8,
            },
        });

        // Season transition: adjust food stockpiles for winter scarcity.
        if season == Season::Winter {
            // Winter onset: food decays faster (spoilage, increased consumption).
            for settlement in &state.settlements {
                let food_loss = settlement.stockpile[0] * 0.1; // 10% spoilage
                if food_loss > 0.0 {
                    out.push(WorldDelta::ConsumeCommodity {
                        settlement_id: settlement.id,
                        commodity: 0, // FOOD
                        amount: food_loss,
                    });
                }
            }
        }

        // Autumn harvest bonus: settlements with farms get food boost.
        if season == Season::Autumn {
            for settlement in &state.settlements {
                // Harvest bonus proportional to existing food production.
                let harvest_bonus = settlement.stockpile[0] * 0.3;
                if harvest_bonus > 0.0 {
                    out.push(WorldDelta::ProduceCommodity {
                        settlement_id: settlement.id,
                        commodity: 0, // FOOD
                        amount: harvest_bonus.min(50.0),
                    });
                }
            }
        }
    }

    // --- Per-tick seasonal price pressure (every 50 ticks) ---
    if state.tick % 50 == 0 && state.tick > 0 {
        let pressure = price_pressure(season);
        for settlement in &state.settlements {
            let mut new_prices = settlement.prices;
            for c in 0..8 {
                // Drift prices toward seasonal pressure.
                let target = pressure[c];
                let current = new_prices[c];
                // Move 5% toward target each cycle.
                new_prices[c] = current + (target - current) * 0.05;
                new_prices[c] = new_prices[c].max(0.1); // floor
            }
            out.push(WorldDelta::UpdatePrices {
                settlement_id: settlement.id,
                prices: new_prices,
            });
        }
    }

    // --- Morale drift from season (every 10 ticks) ---
    if state.tick % 10 == 0 {
        let morale_delta = mods.morale_per_tick * 10.0; // accumulated over 10 ticks
        if morale_delta.abs() > 0.01 {
            for entity in &state.entities {
                if !entity.alive || entity.kind != crate::world_sim::state::EntityKind::Npc { continue; }
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: crate::world_sim::state::EntityField::Morale,
                    value: morale_delta,
                });
            }
        }
    }

    // --- Festival: mid-season celebration (once per season at tick 600) ---
    let season_tick = state.tick % TICKS_PER_SEASON;
    if season_tick == TICKS_PER_SEASON / 2 && state.tick > 0 {
        let festival_name = match season {
            Season::Spring => "Blossom Festival",
            Season::Summer => "Solstice Feast",
            Season::Autumn => "Harvest Fair",
            Season::Winter => "Midwinter Vigil",
        };
        out.push(WorldDelta::RecordChronicle {
            entry: crate::world_sim::state::ChronicleEntry {
                tick: state.tick,
                category: crate::world_sim::state::ChronicleCategory::Achievement,
                text: format!("The {} is celebrated across the realm.", festival_name),
                entity_ids: vec![],
            },
        });

        // Festival morale boost to all NPCs.
        for entity in &state.entities {
            if !entity.alive || entity.kind != crate::world_sim::state::EntityKind::Npc { continue; }
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: crate::world_sim::state::EntityField::Morale,
                value: 10.0, // festival cheer
            });
        }
    }
}
