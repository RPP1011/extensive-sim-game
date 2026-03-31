//! Physical resource nodes — terrain-based spawning and regrowth.
//!
//! Resource nodes are physical entities scattered across regions based on
//! terrain type. NPCs harvest them via the work system (Farm→FOOD needs
//! nearby BerryBush/FishingSpot, Sawmill→WOOD needs Trees, Mine→IRON needs
//! OreVein).
//!
//! - Renewable resources (trees, herbs, berries, fish) regenerate over time.
//! - Non-renewable resources (ore, stone) deplete permanently and are marked dead.
//!
//! `spawn_initial_resources` is called once at world init.
//! `tick_resource_regrowth` runs every 50 ticks in the post-apply phase.

use crate::world_sim::state::{
    Entity, EntityKind, ResourceData, ResourceType, Terrain, WorldState,
    entity_hash_f32,
};

/// How often (in ticks) to run resource regrowth.
const REGROWTH_INTERVAL: u64 = 50;

/// Spawn resource nodes for all regions based on terrain.
///
/// Called once after world init. Uses `entity_hash_f32` for deterministic
/// placement (no external RNG).
pub fn spawn_initial_resources(state: &mut WorldState) {
    // Phase 1: collect spawn plans from regions (immutable borrow).
    // Each plan: (region_id, region_pos, resource_type, count, remaining, max_cap, regrow_rate).
    let mut plans: Vec<(u32, (f32, f32), ResourceType, u32, f32, f32, f32)> = Vec::new();

    for region in &state.regions {
        let rid = region.id;
        let base_pos = region.pos;

        // Determine which resource types to spawn and how many, by terrain.
        let spawn_specs: &[(ResourceType, u32, u32, f32, f32, f32)] = match region.terrain {
            // (type, min_count, max_count, remaining, max_capacity, regrow_rate)
            Terrain::Forest => &[
                (ResourceType::Tree, 3, 6, 10.0, 10.0, 0.003),   // slow regrow — wood is scarce
                (ResourceType::HerbPatch, 1, 2, 4.0, 4.0, 0.002),
                (ResourceType::BerryBush, 2, 3, 5.0, 5.0, 0.003), // supports ~10 NPCs
            ],
            Terrain::Mountains => &[
                (ResourceType::OreVein, 3, 5, 50.0, 50.0, 0.0),
                (ResourceType::StoneOutcrop, 1, 3, 30.0, 30.0, 0.0),
            ],
            Terrain::Plains => &[
                (ResourceType::BerryBush, 2, 4, 10.0, 10.0, 0.02),
                (ResourceType::Tree, 1, 2, 15.0, 15.0, 0.008),
            ],
            Terrain::Jungle => &[
                (ResourceType::HerbPatch, 3, 6, 12.0, 12.0, 0.02),
                (ResourceType::Tree, 2, 4, 25.0, 25.0, 0.015),
            ],
            Terrain::Swamp => &[
                (ResourceType::HerbPatch, 3, 5, 10.0, 10.0, 0.018),
                (ResourceType::FishingSpot, 1, 2, 8.0, 8.0, 0.01),
            ],
            Terrain::Coast => &[
                (ResourceType::FishingSpot, 2, 3, 15.0, 15.0, 0.015),
                (ResourceType::BerryBush, 1, 2, 8.0, 8.0, 0.015),
            ],
            Terrain::Desert => &[
                (ResourceType::StoneOutcrop, 1, 2, 20.0, 20.0, 0.0),
            ],
            Terrain::Tundra => &[
                (ResourceType::OreVein, 1, 2, 30.0, 30.0, 0.0),
            ],
            Terrain::Volcano => &[
                (ResourceType::OreVein, 2, 4, 60.0, 60.0, 0.0),
            ],
            Terrain::Caverns => &[
                (ResourceType::OreVein, 3, 5, 40.0, 40.0, 0.0),
                (ResourceType::StoneOutcrop, 2, 3, 25.0, 25.0, 0.0),
            ],
            Terrain::Glacier => &[
                (ResourceType::StoneOutcrop, 1, 2, 15.0, 15.0, 0.0),
            ],
            Terrain::Badlands => &[
                (ResourceType::StoneOutcrop, 1, 2, 15.0, 15.0, 0.0),
            ],
            // DeepOcean, FlyingIslands, Corruption — no ground resources.
            _ => &[],
        };

        for (spec_idx, &(rtype, min, max, remaining, max_cap, regrow)) in spawn_specs.iter().enumerate() {
            // Deterministic count: min + hash-fraction * (max - min + 1)
            let count_hash = entity_hash_f32(rid, spec_idx as u64, 0xFEED);
            let count = min + (count_hash * (max - min + 1) as f32) as u32;
            let count = count.min(max);
            plans.push((rid, base_pos, rtype, count, remaining, max_cap, regrow));
        }
    }

    // Phase 2: create entities from plans (mutable borrow for next_entity_id).
    let mut new_entities: Vec<Entity> = Vec::new();
    for (rid, base_pos, rtype, count, remaining, max_cap, regrow) in plans {
        for i in 0..count {
            let id = state.next_entity_id();
            // Position: region center + deterministic scatter (±60 units).
            let jx = entity_hash_f32(id, i as u64, 0xA1) * 120.0 - 60.0;
            let jy = entity_hash_f32(id, i as u64, 0xA2) * 120.0 - 60.0;
            let pos = (base_pos.0 + jx, base_pos.1 + jy);

            let data = ResourceData {
                resource_type: rtype,
                remaining,
                max_capacity: max_cap,
                regrow_rate: regrow,
                region_id: rid,
            };

            new_entities.push(Entity::new_resource(id, pos, data));
        }
    }

    state.entities.extend(new_entities);
}

/// Regenerate renewable resources and mark depleted non-renewables as dead.
///
/// Called post-apply from runtime every `REGROWTH_INTERVAL` ticks.
pub fn tick_resource_regrowth(state: &mut WorldState) {
    if state.tick % REGROWTH_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Resource {
            continue;
        }

        let resource = match &mut entity.resource {
            Some(r) => r,
            None => continue,
        };

        if resource.regrow_rate > 0.0 {
            // Renewable: regenerate toward max capacity.
            resource.remaining = (resource.remaining + resource.regrow_rate * REGROWTH_INTERVAL as f32)
                .min(resource.max_capacity);
        } else if resource.remaining <= 0.0 {
            // Non-renewable depleted: mark entity as dead.
            entity.alive = false;
        }
    }
}

/// Find the nearest alive resource entity of a given type within `max_dist`
/// world units of `pos`. Returns `(entity_index, distance_squared)`.
pub fn find_nearest_resource(
    state: &WorldState,
    pos: (f32, f32),
    resource_type: ResourceType,
    max_dist: f32,
) -> Option<(usize, f32)> {
    let max_dist_sq = max_dist * max_dist;
    let mut best: Option<(usize, f32)> = None;

    for (idx, entity) in state.entities.iter().enumerate() {
        if !entity.alive || entity.kind != EntityKind::Resource {
            continue;
        }
        let r = match &entity.resource {
            Some(r) => r,
            None => continue,
        };
        if r.resource_type != resource_type || r.remaining <= 0.0 {
            continue;
        }
        let dx = entity.pos.0 - pos.0;
        let dy = entity.pos.1 - pos.1;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq > max_dist_sq {
            continue;
        }
        if best.is_none() || dist_sq < best.unwrap().1 {
            best = Some((idx, dist_sq));
        }
    }

    best
}
