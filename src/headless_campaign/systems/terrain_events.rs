//! Terrain events system — natural disasters and geographic changes.
//!
//! Fires every 500 ticks with a 3% chance per roll. Season influences which
//! events can occur (floods in spring, wildfires in summer, earthquakes any
//! time). Active events apply ongoing effects and expire after their duration.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to roll for a terrain event (in ticks).
const TERRAIN_EVENT_INTERVAL: u64 = 500;

/// Base probability of a terrain event firing each roll.
const TERRAIN_EVENT_CHANCE: f32 = 0.03;

/// Advance terrain events: expire finished ones, roll for new ones.
pub fn tick_terrain_events(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TERRAIN_EVENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire completed terrain events and apply recovery ---
    expire_terrain_events(state, events);

    // --- Roll for new terrain event ---
    let roll = lcg_f32(&mut state.rng);
    if roll > TERRAIN_EVENT_CHANCE {
        return;
    }

    if state.overworld.regions.is_empty() {
        return;
    }

    // Pick event type based on season
    let event_type = pick_event_type(state);

    // Pick affected regions
    let affected_regions = pick_affected_regions(state, event_type);
    if affected_regions.is_empty() {
        return;
    }

    // Determine severity (0.3 – 1.0)
    let severity = 0.3 + lcg_f32(&mut state.rng) * 0.7;

    // Determine duration based on type
    let duration = pick_duration(state, event_type);

    let event_id = state.next_terrain_event_id;
    state.next_terrain_event_id += 1;

    let terrain_event = TerrainEvent {
        id: event_id,
        event_type,
        affected_regions: affected_regions.clone(),
        started_tick: state.tick,
        duration,
        severity,
    };

    // Apply immediate effects
    apply_immediate_effects(state, &terrain_event, events);

    // Emit start event
    events.push(WorldEvent::TerrainEventStarted {
        event_id,
        event_type: format!("{:?}", event_type),
        affected_regions: affected_regions.clone(),
        severity,
        duration,
    });

    state.terrain_events.push(terrain_event);
}

// ---------------------------------------------------------------------------
// Event type selection
// ---------------------------------------------------------------------------

fn pick_event_type(state: &mut CampaignState) -> TerrainEventType {
    // Build weighted pool based on season
    let season = state.overworld.season;
    let mut pool: Vec<(TerrainEventType, f32)> = Vec::new();

    // Earthquakes happen any time
    pool.push((TerrainEventType::Earthquake, 10.0));

    // Floods more likely in spring
    let flood_weight = match season {
        Season::Spring => 15.0,
        Season::Winter => 8.0, // snowmelt
        _ => 3.0,
    };
    pool.push((TerrainEventType::Flood, flood_weight));

    // Wildfires more likely in summer
    let fire_weight = match season {
        Season::Summer => 15.0,
        Season::Autumn => 8.0, // dry conditions
        _ => 2.0,
    };
    pool.push((TerrainEventType::Wildfire, fire_weight));

    // Landslides more likely in spring (rain) and winter (freeze-thaw)
    let slide_weight = match season {
        Season::Spring | Season::Winter => 10.0,
        _ => 5.0,
    };
    pool.push((TerrainEventType::Landslide, slide_weight));

    // Volcanic eruptions are rare
    pool.push((TerrainEventType::VolcanicEruption, 3.0));

    // Sinkholes are uncommon
    pool.push((TerrainEventType::Sinkhole, 5.0));

    // Weighted selection
    let total: f32 = pool.iter().map(|(_, w)| w).sum();
    let pick = lcg_f32(&mut state.rng) * total;
    let mut cumulative = 0.0;
    for (evt, w) in &pool {
        cumulative += w;
        if pick < cumulative {
            return *evt;
        }
    }
    pool.last().unwrap().0
}

// ---------------------------------------------------------------------------
// Region selection
// ---------------------------------------------------------------------------

fn pick_affected_regions(state: &mut CampaignState, event_type: TerrainEventType) -> Vec<usize> {
    let num_regions = state.overworld.regions.len();
    if num_regions == 0 {
        return Vec::new();
    }

    let primary = (lcg_next(&mut state.rng) as usize) % num_regions;

    match event_type {
        // Earthquake can spread to neighbors
        TerrainEventType::Earthquake => {
            let mut affected = vec![primary];
            // 40% chance to spread to each neighbor
            let neighbors = state.overworld.regions[primary].neighbors.clone();
            for n in neighbors {
                if lcg_f32(&mut state.rng) < 0.4 {
                    affected.push(n);
                }
            }
            affected
        }
        // Flood spreads along connected regions
        TerrainEventType::Flood => {
            let mut affected = vec![primary];
            let neighbors = state.overworld.regions[primary].neighbors.clone();
            // 50% chance each neighbor also floods
            for n in neighbors {
                if lcg_f32(&mut state.rng) < 0.5 {
                    affected.push(n);
                }
            }
            affected
        }
        // Wildfire: primary + potential spread
        TerrainEventType::Wildfire => {
            let mut affected = vec![primary];
            let neighbors = state.overworld.regions[primary].neighbors.clone();
            for n in neighbors {
                if lcg_f32(&mut state.rng) < 0.3 {
                    affected.push(n);
                }
            }
            affected
        }
        // Landslide: between primary and one neighbor
        TerrainEventType::Landslide => {
            vec![primary]
        }
        // Volcanic eruption: localized
        TerrainEventType::VolcanicEruption => {
            vec![primary]
        }
        // Sinkhole: single region
        TerrainEventType::Sinkhole => {
            vec![primary]
        }
    }
}

// ---------------------------------------------------------------------------
// Duration selection (300-1500 ticks)
// ---------------------------------------------------------------------------

fn pick_duration(state: &mut CampaignState, event_type: TerrainEventType) -> u64 {
    let (min, max) = match event_type {
        TerrainEventType::Earthquake => (300, 600),
        TerrainEventType::Flood => (500, 1500),
        TerrainEventType::Wildfire => (400, 1000),
        TerrainEventType::Landslide => (600, 1200),
        TerrainEventType::VolcanicEruption => (800, 1500),
        TerrainEventType::Sinkhole => (300, 500), // quick collapse, permanent effect
    };
    let range = max - min;
    min + (lcg_next(&mut state.rng) as u64 % (range + 1))
}

// ---------------------------------------------------------------------------
// Immediate effects (applied once when event starts)
// ---------------------------------------------------------------------------

fn apply_immediate_effects(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    match terrain_event.event_type {
        TerrainEventType::Earthquake => apply_earthquake(state, terrain_event, events),
        TerrainEventType::Flood => apply_flood(state, terrain_event, events),
        TerrainEventType::Wildfire => apply_wildfire(state, terrain_event, events),
        TerrainEventType::Landslide => apply_landslide(state, terrain_event, events),
        TerrainEventType::VolcanicEruption => apply_volcanic_eruption(state, terrain_event, events),
        TerrainEventType::Sinkhole => apply_sinkhole(state, terrain_event, events),
    }
}

fn apply_earthquake(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    let severity = terrain_event.severity;

    // Damage buildings: -1 tier in affected regions (applied to guild buildings
    // since buildings are guild-level, not region-level)
    let building_damage = (severity * 1.5).floor() as u8; // 0 or 1
    if building_damage > 0 {
        let buildings = &mut state.guild_buildings;
        // Damage the lowest-tier non-zero building
        let tiers = [
            &mut buildings.training_grounds,
            &mut buildings.watchtower,
            &mut buildings.trade_post,
            &mut buildings.barracks,
            &mut buildings.infirmary,
            &mut buildings.war_room,
        ];
        let building_names = [
            "Training Grounds",
            "Watchtower",
            "Trade Post",
            "Barracks",
            "Infirmary",
            "War Room",
        ];
        // Pick a random non-zero building to damage
        let nonzero: Vec<usize> = tiers
            .iter()
            .enumerate()
            .filter(|(_, t)| ***t > 0)
            .map(|(i, _)| i)
            .collect();
        if !nonzero.is_empty() {
            let idx = (lcg_next(&mut state.rng) as usize) % nonzero.len();
            let building_idx = nonzero[idx];
            // Re-borrow mutably for the specific field
            match building_idx {
                0 => state.guild_buildings.training_grounds = state.guild_buildings.training_grounds.saturating_sub(1),
                1 => state.guild_buildings.watchtower = state.guild_buildings.watchtower.saturating_sub(1),
                2 => state.guild_buildings.trade_post = state.guild_buildings.trade_post.saturating_sub(1),
                3 => state.guild_buildings.barracks = state.guild_buildings.barracks.saturating_sub(1),
                4 => state.guild_buildings.infirmary = state.guild_buildings.infirmary.saturating_sub(1),
                5 => state.guild_buildings.war_room = state.guild_buildings.war_room.saturating_sub(1),
                _ => {}
            }
            for &region_id in &terrain_event.affected_regions {
                events.push(WorldEvent::TerrainDamage {
                    event_id: terrain_event.id,
                    region_id,
                    description: format!(
                        "Earthquake damages {} (-1 tier)",
                        building_names[building_idx]
                    ),
                });
            }
        }
    }

    // Increase unrest in affected regions
    for &region_id in &terrain_event.affected_regions {
        if let Some(region) = state.overworld.regions.get_mut(region_id) {
            region.unrest = (region.unrest + 15.0 * severity).min(100.0);
        }
    }

    // 20% chance per affected region to discover a new dungeon
    for &region_id in &terrain_event.affected_regions {
        if lcg_f32(&mut state.rng) < 0.2 {
            let loc_id = state.overworld.locations.len();
            let region_center = if region_id < state.overworld.regions.len() {
                // Approximate position from region index
                ((region_id as f32 * 20.0) % 100.0, (region_id as f32 * 15.0 + 10.0) % 100.0)
            } else {
                (50.0, 50.0)
            };
            let x = region_center.0 + (lcg_next(&mut state.rng) % 10) as f32;
            let y = region_center.1 + (lcg_next(&mut state.rng) % 10) as f32;

            state.overworld.locations.push(Location {
                id: loc_id,
                name: format!("Earthquake Rift #{}", terrain_event.id),
                position: (x, y),
                location_type: LocationType::Dungeon,
                threat_level: 30.0 + (lcg_next(&mut state.rng) % 40) as f32,
                resource_availability: 40.0 + (lcg_next(&mut state.rng) % 30) as f32,
                faction_owner: None,
                scouted: true,
            });

            events.push(WorldEvent::TerrainDiscovery {
                event_id: terrain_event.id,
                region_id,
                description: format!(
                    "Earthquake opens a rift revealing a new dungeon in region {}",
                    region_id
                ),
            });
        }
    }
}

fn apply_flood(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    let severity = terrain_event.severity;

    for &region_id in &terrain_event.affected_regions {
        // Damage faction military strength (proxy for population damage)
        // -5% of faction strength per affected region
        if let Some(region) = state.overworld.regions.get(region_id) {
            let owner = region.owner_faction_id;
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == owner) {
                let loss = faction.military_strength * 0.05 * severity;
                faction.military_strength = (faction.military_strength - loss).max(0.0);
            }
        }

        // Increase unrest (civilian morale proxy)
        if let Some(region) = state.overworld.regions.get_mut(region_id) {
            region.unrest = (region.unrest + 10.0 * severity).min(100.0);
            // Reduce control (represents disrupted governance)
            region.control = (region.control - 5.0 * severity).max(0.0);
        }

        // Reduce resource availability at locations in the region
        for loc in &mut state.overworld.locations {
            // Approximate: locations near the region center are affected
            let region_x = (region_id as f32 * 20.0) % 100.0;
            let region_y = (region_id as f32 * 15.0 + 10.0) % 100.0;
            let dx = loc.position.0 - region_x;
            let dy = loc.position.1 - region_y;
            if dx * dx + dy * dy < 400.0 {
                loc.resource_availability =
                    (loc.resource_availability - 10.0 * severity).max(0.0);
            }
        }

        events.push(WorldEvent::TerrainDamage {
            event_id: terrain_event.id,
            region_id,
            description: format!(
                "Flooding damages region {}: travel slowed, resources depleted",
                region_id
            ),
        });
    }

    // Morale hit to all adventurers
    for adv in &mut state.adventurers {
        if adv.status != AdventurerStatus::Dead {
            adv.morale = (adv.morale - 10.0 * severity).max(0.0);
        }
    }
}

fn apply_wildfire(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    let severity = terrain_event.severity;

    for &region_id in &terrain_event.affected_regions {
        // Destroy 50% of resource nodes (scaled by severity)
        for loc in &mut state.overworld.locations {
            let region_x = (region_id as f32 * 20.0) % 100.0;
            let region_y = (region_id as f32 * 15.0 + 10.0) % 100.0;
            let dx = loc.position.0 - region_x;
            let dy = loc.position.1 - region_y;
            if dx * dx + dy * dy < 400.0 {
                let resource_loss = loc.resource_availability * 0.5 * severity;
                loc.resource_availability =
                    (loc.resource_availability - resource_loss).max(0.0);
            }
        }

        // Kill 30% of monster population (reduce threat)
        if let Some(region) = state.overworld.regions.get_mut(region_id) {
            let threat_reduction = region.threat_level * 0.3 * severity;
            region.threat_level = (region.threat_level - threat_reduction).max(0.0);
        }

        events.push(WorldEvent::TerrainDamage {
            event_id: terrain_event.id,
            region_id,
            description: format!(
                "Wildfire sweeps region {}: resources destroyed, monsters cleared",
                region_id
            ),
        });
    }
}

fn apply_landslide(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    // Block travel between the primary region and one of its neighbors.
    // We model this by increasing unrest (travel disruption proxy).
    for &region_id in &terrain_event.affected_regions {
        if let Some(region) = state.overworld.regions.get_mut(region_id) {
            region.unrest = (region.unrest + 20.0 * terrain_event.severity).min(100.0);
        }

        events.push(WorldEvent::TerrainDamage {
            event_id: terrain_event.id,
            region_id,
            description: format!(
                "Landslide blocks routes in region {} for {} ticks",
                region_id, terrain_event.duration
            ),
        });
    }
}

fn apply_volcanic_eruption(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    let severity = terrain_event.severity;

    for &region_id in &terrain_event.affected_regions {
        // Massive damage: faction strength -20%
        if let Some(region) = state.overworld.regions.get(region_id) {
            let owner = region.owner_faction_id;
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == owner) {
                let loss = faction.military_strength * 0.2 * severity;
                faction.military_strength = (faction.military_strength - loss).max(0.0);
            }
        }

        // Destroy buildings (all -1 tier if severity > 0.7)
        if severity > 0.7 {
            state.guild_buildings.training_grounds =
                state.guild_buildings.training_grounds.saturating_sub(1);
            state.guild_buildings.watchtower =
                state.guild_buildings.watchtower.saturating_sub(1);
            state.guild_buildings.trade_post =
                state.guild_buildings.trade_post.saturating_sub(1);
        }

        // Massive unrest
        if let Some(region) = state.overworld.regions.get_mut(region_id) {
            region.unrest = (region.unrest + 30.0 * severity).min(100.0);
            region.control = (region.control - 15.0 * severity).max(0.0);
        }

        events.push(WorldEvent::TerrainDamage {
            event_id: terrain_event.id,
            region_id,
            description: format!(
                "Volcanic eruption devastates region {}: massive destruction",
                region_id
            ),
        });

        // But: spawn Obsidian resource nodes
        let loc_id = state.overworld.locations.len();
        let x = (region_id as f32 * 20.0) % 100.0 + (lcg_next(&mut state.rng) % 15) as f32;
        let y = (region_id as f32 * 15.0 + 10.0) % 100.0 + (lcg_next(&mut state.rng) % 15) as f32;

        state.overworld.locations.push(Location {
            id: loc_id,
            name: format!("Obsidian Deposit #{}", terrain_event.id),
            position: (x, y),
            location_type: LocationType::Ruin, // repurpose as resource site
            threat_level: 50.0 + (lcg_next(&mut state.rng) % 20) as f32,
            resource_availability: 80.0 + (lcg_next(&mut state.rng) % 21) as f32,
            faction_owner: None,
            scouted: true,
        });

        events.push(WorldEvent::TerrainDiscovery {
            event_id: terrain_event.id,
            region_id,
            description: format!(
                "Volcanic eruption reveals obsidian deposits in region {}",
                region_id
            ),
        });
    }
}

fn apply_sinkhole(
    state: &mut CampaignState,
    terrain_event: &TerrainEvent,
    events: &mut Vec<WorldEvent>,
) {
    for &region_id in &terrain_event.affected_regions {
        // Create a new dungeon entry
        let loc_id = state.overworld.locations.len();
        let x = (region_id as f32 * 20.0) % 100.0 + (lcg_next(&mut state.rng) % 20) as f32;
        let y = (region_id as f32 * 15.0 + 10.0) % 100.0 + (lcg_next(&mut state.rng) % 20) as f32;

        let dungeon_names = [
            "Collapsed Cavern",
            "Sunken Vault",
            "Underground Passage",
            "Buried Temple",
            "Deep Hollow",
        ];
        let name_idx = (lcg_next(&mut state.rng) as usize) % dungeon_names.len();

        state.overworld.locations.push(Location {
            id: loc_id,
            name: format!("{} #{}", dungeon_names[name_idx], terrain_event.id),
            position: (x, y),
            location_type: LocationType::Dungeon,
            threat_level: 40.0 + (lcg_next(&mut state.rng) % 30) as f32,
            resource_availability: 50.0 + (lcg_next(&mut state.rng) % 30) as f32,
            faction_owner: None,
            scouted: true,
        });

        // Minor unrest from the ground literally collapsing
        if let Some(region) = state.overworld.regions.get_mut(region_id) {
            region.unrest = (region.unrest + 5.0 * terrain_event.severity).min(100.0);
        }

        events.push(WorldEvent::TerrainDiscovery {
            event_id: terrain_event.id,
            region_id,
            description: format!(
                "A sinkhole opens in region {}, revealing a new dungeon",
                region_id
            ),
        });
    }
}

// ---------------------------------------------------------------------------
// Expiration and recovery
// ---------------------------------------------------------------------------

fn expire_terrain_events(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut expired = Vec::new();

    // Find expired events
    for (i, te) in state.terrain_events.iter().enumerate() {
        if tick >= te.started_tick + te.duration {
            expired.push(i);
        }
    }

    // Process expired events in reverse order (to preserve indices)
    for &i in expired.iter().rev() {
        let te = state.terrain_events.remove(i);

        // Recovery: regions slowly rebuild after event ends
        for &region_id in &te.affected_regions {
            if let Some(region) = state.overworld.regions.get_mut(region_id) {
                // Partial unrest recovery
                region.unrest = (region.unrest - 5.0).max(0.0);
                // Partial control recovery
                region.control = (region.control + 3.0).min(100.0);
            }
        }

        events.push(WorldEvent::TerrainEventEnded {
            event_id: te.id,
            event_type: format!("{:?}", te.event_type),
            affected_regions: te.affected_regions,
        });
    }
}
