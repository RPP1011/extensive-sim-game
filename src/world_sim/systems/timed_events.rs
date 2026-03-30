#![allow(unused)]
//! Time-limited events system — fires every 7 ticks.
//!
//! Special opportunities that briefly affect the world: trade windfalls,
//! meteor showers (commodity boosts), eclipses (entity buffs), etc.
//! Events have short durations and are modeled as temporary commodity
//! production boosts or entity buffs via existing delta variants.
//!
//! Original: `crates/headless_campaign/src/systems/timed_events.rs`
//!
//! NEEDS STATE: `timed_events: Vec<TimedEvent>` on WorldState
//! NEEDS DELTA: SpawnTimedEvent { ... }
//! NEEDS DELTA: ExpireTimedEvent { event_id }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState, WorldTeam};
use crate::world_sim::state::{entity_hash_f32};
use crate::world_sim::NUM_COMMODITIES;

use super::seasons::{current_season, Season};

/// How often to roll for timed events (in ticks).
const EVENT_INTERVAL: u64 = 7;

/// Base probability of an event firing each roll (5%).
const BASE_CHANCE: f32 = 0.05;

/// Maximum concurrent active timed events.
const MAX_ACTIVE: usize = 2;


/// Count of active timed events (approximated by checking recent buff statuses).
fn estimate_active_events(state: &WorldState) -> usize {
    // Without timed_events on WorldState, we use a counter derived from tick.
    // Events last ~50-100 ticks, so we check how many event-producing ticks
    // have fired recently.
    let mut count = 0;
    for offset in 0..10u64 {
        let past_tick = state.tick.saturating_sub(offset * EVENT_INTERVAL);
        if past_tick == 0 {
            break;
        }
        let past_roll = entity_hash_f32(0, past_tick, 0xE4E4_7001u64.wrapping_add(0xEE01));
        if past_roll < BASE_CHANCE {
            count += 1;
        }
    }
    count
}

pub fn compute_timed_events(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EVENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Cap concurrent events.
    if estimate_active_events(state) >= MAX_ACTIVE {
        return;
    }

    let season = current_season(state.tick);
    let season_mult = match season {
        Season::Spring => 1.2,
        Season::Summer => 1.0,
        Season::Autumn => 1.3,
        Season::Winter => 0.7,
    };
    let chance = BASE_CHANCE * season_mult;

    let roll = entity_hash_f32(0, state.tick, 0xEE01_7001);
    if roll > chance {
        return;
    }

    // Pick event type.
    let event_roll = entity_hash_f32(0, state.tick, 0xEE02_7002);

    if event_roll < 0.15 {
        // Trade Winds: gold bonus to all settlements.
        for settlement in &state.settlements {
            let bonus = 5.0 + entity_hash_f32(settlement.id, state.tick, 0) * 15.0;
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: bonus,
            });
        }
    } else if event_roll < 0.30 {
        // Meteor Shower: rare commodity production boost.
        let commodity = (state.tick as usize / EVENT_INTERVAL as usize) % NUM_COMMODITIES;
        for settlement in &state.settlements {
            out.push(WorldDelta::ProduceCommodity {
                location_id: settlement.id,
                commodity,
                amount: 3.0 + entity_hash_f32(settlement.id, state.tick, 0xCCC) * 7.0,
            });
        }
    } else if event_roll < 0.45 {
        // Eclipse: buff friendly NPCs with a temporary HoT (heal over time).
        for entity in &state.entities {
            if entity.alive && entity.kind == EntityKind::Npc && entity.team == WorldTeam::Friendly
            {
                out.push(WorldDelta::ApplyStatus {
                    target_id: entity.id,
                    status: StatusEffect {
                        kind: StatusEffectKind::Hot {
                            heal_per_tick: 2.0,
                            tick_interval_ms: 1000,
                            tick_elapsed_ms: 0,
                        },
                        source_id: 0,
                        remaining_ms: 5000,
                    },
                });
            }
        }
    } else if event_roll < 0.60 {
        // Harvest Moon: food commodity boost.
        for settlement in &state.settlements {
            out.push(WorldDelta::ProduceCommodity {
                location_id: settlement.id,
                commodity: crate::world_sim::commodity::FOOD, // food
                amount: 5.0 + entity_hash_f32(settlement.id, state.tick, 0xF00D) * 10.0,
            });
        }
    } else if event_roll < 0.75 {
        // Faction Summit: heal friendly NPCs (morale boost proxy).
        for entity in &state.entities {
            if entity.alive
                && entity.kind == EntityKind::Npc
                && entity.team == WorldTeam::Friendly
                && entity.hp < entity.max_hp
            {
                let heal = (entity.max_hp - entity.hp) * 0.1;
                out.push(WorldDelta::Heal {
                    target_id: entity.id,
                    amount: heal,
                    source_id: 0,
                });
            }
        }
    } else {
        // Ancient Portal: brief commodity production at a random settlement.
        if let Some(settlement) = state.settlements.first() {
            let idx = (state.tick as usize) % state.settlements.len().max(1);
            let s = &state.settlements[idx];
            for c in 0..NUM_COMMODITIES {
                out.push(WorldDelta::ProduceCommodity {
                    location_id: s.id,
                    commodity: c,
                    amount: 2.0,
                });
            }
        }
    }
}
