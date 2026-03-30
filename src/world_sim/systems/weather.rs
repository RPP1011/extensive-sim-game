//! Weather hazards — fires every 7 ticks.
//!
//! Applies ongoing weather effects to entities and settlements:
//! - Storms: slow travel, damage parties
//! - Blizzards: severe travel slowdown, building damage
//! - Floods: trade disruption, settlement damage
//! - Droughts: increased supply costs, unrest
//! - Fog: threat increase in affected regions
//! - Heatwaves: entity damage (fatigue)
//!
//! Original: `crates/headless_campaign/src/systems/weather.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

use super::seasons::{current_season, season_modifiers, Season};

/// How often to apply weather effects (in ticks).
const WEATHER_INTERVAL: u64 = 7;

/// Weather types that can affect the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeatherType {
    Storm,
    Blizzard,
    Flood,
    Drought,
    Fog,
    Heatwave,
}

/// An active weather event (would live on WorldState).
#[derive(Clone, Debug)]
pub struct WeatherEvent {
    pub id: u32,
    pub weather_type: WeatherType,
    pub affected_region_ids: Vec<u32>,
    pub severity: f32,
    pub started_tick: u64,
    pub duration: u64,
}

impl WeatherEvent {
    pub fn is_active(&self, tick: u64) -> bool {
        tick < self.started_tick + self.duration
    }

    pub fn scale(&self) -> f32 {
        self.severity / 100.0
    }
}


pub fn compute_weather(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % WEATHER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without active_weather on WorldState, we can only apply season-derived
    // effects. When WeatherEvent storage is added, this function will iterate
    // active events and dispatch per-type effects below.
    //
    // For now, apply mild seasonal weather effects to demonstrate the pattern.

    let season = current_season(state.tick);
    let mods = season_modifiers(season);

    // Winter: slow all traveling NPCs (via reduced move force).
    // This is a simplified stand-in for blizzard/storm effects.
    if season == Season::Winter {
        apply_winter_travel_penalty(state, out, mods.travel_speed);
    }

    // All seasons: if threat modifier > 1.0, apply mild damage to entities
    // in high-threat regions (simulates ambient monster encounters).
    if mods.threat > 1.0 {
        apply_ambient_threat_damage(state, out, mods.threat);
    }
}

/// Slow traveling NPCs during winter by emitting counter-Move deltas.
/// The seasonal travel_speed modifier (0.7 in winter) means we need to
/// counteract 30% of the movement that the travel system will emit.
fn apply_winter_travel_penalty(
    state: &WorldState,
    out: &mut Vec<WorldDelta>,
    travel_speed_mod: f32,
) {
    if travel_speed_mod >= 1.0 {
        return; // No penalty needed.
    }

    let penalty_fraction = 1.0 - travel_speed_mod; // e.g. 0.3 in winter

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Only penalize entities that are actively traveling.
        let is_traveling = matches!(
            npc.economic_intent,
            crate::world_sim::state::EconomicIntent::Travel { .. }
                | crate::world_sim::state::EconomicIntent::Trade { .. }
        );
        if !is_traveling {
            continue;
        }

        // Emit a counter-force proportional to the entity's current velocity.
        // Since we don't store velocity, we estimate from the travel system's
        // expected output: move_speed * 0.1 in the travel direction.
        // The penalty reduces effective speed by `penalty_fraction`.
        //
        // We approximate by looking at the direction to destination and applying
        // a braking force. This is an approximation — the merge phase sums forces.
        let dest = match &npc.economic_intent {
            crate::world_sim::state::EconomicIntent::Travel { destination } => Some(*destination),
            crate::world_sim::state::EconomicIntent::Trade {
                destination_settlement_id,
            } => state.settlement(*destination_settlement_id).map(|s| s.pos),
            _ => None,
        };

        if let Some(dest) = dest {
            let dx = dest.0 - entity.pos.0;
            let dy = dest.1 - entity.pos.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 0.5 {
                let base_speed = entity.move_speed * 0.1;
                let brake = base_speed * penalty_fraction;
                let nx = dx / dist;
                let ny = dy / dist;
                // Counter-force opposes travel direction.
                out.push(WorldDelta::Move {
                    entity_id: entity.id,
                    force: (-nx * brake, -ny * brake),
                });
            }
        }
    }
}

/// Apply mild damage to NPCs in regions with threat_level above a threshold,
/// scaled by the seasonal threat modifier.
fn apply_ambient_threat_damage(state: &WorldState, out: &mut Vec<WorldDelta>, threat_mod: f32) {
    let threat_threshold = 50.0;

    for region in &state.regions {
        let effective_threat = region.threat_level * threat_mod;
        if effective_threat <= threat_threshold {
            continue;
        }

        // Find NPCs near this region's settlements.
        // Without a region_id on Entity, we use spatial proximity to settlements
        // in the region as a heuristic.
        let region_settlements: Vec<(f32, f32)> = state
            .settlements
            .iter()
            .filter(|_s| {
                // Heuristic: settlement belongs to region if names overlap or
                // IDs are close. For now, use all settlements as a simplification.
                true
            })
            .map(|s| s.pos)
            .collect();

        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }

            // Check if entity is near any settlement in this region.
            let near_region = region_settlements.iter().any(|&spos| {
                let dx = entity.pos.0 - spos.0;
                let dy = entity.pos.1 - spos.1;
                dx * dx + dy * dy < 100.0 // within 10 units
            });

            if !near_region {
                continue;
            }

            // Probabilistic damage based on tick hash.
            let roll = entity_hash_f32(entity.id, state.tick, 0xEA7E_8001);
            let damage_chance = (effective_threat - threat_threshold) * 0.001;
            if roll < damage_chance {
                let damage = (effective_threat - threat_threshold) * 0.1;
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: damage,
                    source_id: 0, // environmental damage
                });
            }
        }
    }
}
