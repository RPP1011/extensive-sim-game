//! Terrain events — natural disasters tied to terrain type, sub-biome, and season.
//!
//! Events are terrain-specific and season-gated:
//! - Volcanic eruptions (Volcano): destroy buildings, kill NPCs, boost minerals
//! - Floods (Coast/river in Spring): damage stockpiles, wash away goods
//! - Avalanches (Mountains/Glacier elevation 3+ in Winter): damage NPCs, block travel
//! - Forest fires (Forest/Jungle in Summer, dense>light): destroy wood/herbs, kill monsters
//! - Sinkholes (Caverns): expose ores, damage buildings
//! - Sandstorms (Desert/Badlands in Summer): slow travel, damage goods
//!
//! Events create major chronicle entries that shape world history.
//!
//! Cadence: every 200 ticks (~per season quarter).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;
use crate::world_sim::NUM_COMMODITIES;

use super::seasons::{current_season, Season};

const EVENT_CHECK_INTERVAL: u64 = 200;

pub fn compute_terrain_events(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EVENT_CHECK_INTERVAL != 0 || state.tick == 0 { return; }
    if state.regions.is_empty() { return; }

    let season = current_season(state.tick);

    for region in &state.regions {
        let roll = entity_hash_f32(region.id, state.tick, 0xD15A);

        match region.terrain {
            Terrain::Volcano => {
                if roll < 0.02 {
                    eruption(state, region, out);
                }
            }
            Terrain::Coast | Terrain::Swamp => {
                if season == Season::Spring && (region.has_river || region.has_lake) && roll < 0.03 {
                    flood(state, region, out);
                }
            }
            Terrain::Mountains | Terrain::Glacier => {
                if season == Season::Winter && region.elevation >= 3 && roll < 0.03 {
                    avalanche(state, region, out);
                }
            }
            Terrain::Forest | Terrain::Jungle => {
                let chance = match (season, region.sub_biome) {
                    (Season::Summer, SubBiome::DenseForest) => 0.04,
                    (Season::Summer, SubBiome::AncientForest) => 0.01, // ancient trees resist fire
                    (Season::Summer, _) => 0.02,
                    (Season::Autumn, SubBiome::DenseForest) => 0.01,
                    _ => 0.0,
                };
                if roll < chance {
                    forest_fire(state, region, out);
                }
            }
            Terrain::Caverns => {
                if roll < 0.02 {
                    cave_collapse(state, region, out);
                }
            }
            Terrain::Desert | Terrain::Badlands => {
                if season == Season::Summer && roll < 0.03 {
                    sandstorm(state, region, out);
                }
            }
            Terrain::DeathZone => {
                if roll < 0.05 { // death zones are inherently unstable
                    corruption_pulse(state, region, out);
                }
            }
            _ => {}
        }
    }
}

fn settlements_near_region(state: &WorldState, region: &RegionState) -> Vec<u32> {
    // Settlements whose region matches or is adjacent.
    let mut sids = Vec::new();
    let region_ids: Vec<u32> = std::iter::once(region.id)
        .chain(region.neighbors.iter().copied())
        .collect();
    for s in &state.settlements {
        // Rough heuristic: check if settlement shares a faction/region.
        // Since settlements don't store region_id directly, use position proximity.
        for &rid in &region_ids {
            if let Some(r) = state.regions.iter().find(|r| r.id == rid) {
                // Approximate region center from grid position.
                let rx = (r.id as f32 * 137.5).sin() * 150.0;
                let ry = (r.id as f32 * 73.1).cos() * 150.0;
                let dx = s.pos.0 - rx;
                let dy = s.pos.1 - ry;
                if dx * dx + dy * dy < 10000.0 { // 100 unit radius
                    sids.push(s.id);
                    break;
                }
            }
        }
    }
    sids
}

fn eruption(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("The volcano in {} erupts! Lava and ash rain destruction.", region.name),
            entity_ids: vec![],
        },
    });

    for sid in settlements_near_region(state, region) {
        if let Some(s) = state.settlement(sid) {
            out.push(WorldDelta::UpdateTreasury { settlement_id: sid, delta: -50.0 });
            let food_loss = s.stockpile[crate::world_sim::commodity::FOOD] * 0.3;
            if food_loss > 0.0 {
                out.push(WorldDelta::ConsumeCommodity {
                    settlement_id: sid, commodity: crate::world_sim::commodity::FOOD, amount: food_loss,
                });
            }
        }
    }

    // Damage NPCs near the volcano.
    let rx = (region.id as f32 * 137.5).sin() * 150.0;
    let ry = (region.id as f32 * 73.1).cos() * 150.0;
    for entity in &state.entities {
        if !entity.alive { continue; }
        let dx = entity.pos.0 - rx;
        let dy = entity.pos.1 - ry;
        if dx * dx + dy * dy < 3600.0 {
            out.push(WorldDelta::Damage {
                target_id: entity.id, amount: 30.0 + entity.max_hp * 0.2, source_id: 0,
            });
        }
    }

    // Eruption clears local monsters.
    out.push(WorldDelta::UpdateRegion {
        region_id: region.id, field: RegionField::MonsterDensity, value: -10.0,
    });
}

fn flood(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("Spring floods devastate {} as waters rise.", region.name),
            entity_ids: vec![],
        },
    });

    for sid in settlements_near_region(state, region) {
        if let Some(s) = state.settlement(sid) {
            for &c in &[crate::world_sim::commodity::FOOD, crate::world_sim::commodity::WOOD] {
                let loss = s.stockpile[c] * 0.2;
                if loss > 0.0 {
                    out.push(WorldDelta::ConsumeCommodity { settlement_id: sid, commodity: c, amount: loss });
                }
            }
            out.push(WorldDelta::UpdateTreasury { settlement_id: sid, delta: -20.0 });
        }
    }
}

fn avalanche(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    let elev_name = match region.elevation {
        3 => "peaks", 4 => "summit", _ => "slopes",
    };
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("An avalanche roars down the {} of {}!", elev_name, region.name),
            entity_ids: vec![],
        },
    });

    let rx = (region.id as f32 * 137.5).sin() * 150.0;
    let ry = (region.id as f32 * 73.1).cos() * 150.0;
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let dx = entity.pos.0 - rx;
        let dy = entity.pos.1 - ry;
        if dx * dx + dy * dy < 2500.0 {
            out.push(WorldDelta::Damage {
                target_id: entity.id, amount: 15.0 + entity.max_hp * 0.1, source_id: 0,
            });
        }
    }

    out.push(WorldDelta::UpdateRegion {
        region_id: region.id, field: RegionField::ThreatLevel, value: 0.2,
    });
}

fn forest_fire(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    let biome = match region.sub_biome {
        SubBiome::AncientForest => "ancient forest",
        SubBiome::DenseForest => "dense forest",
        SubBiome::LightForest => "light woodland",
        SubBiome::TempleJungle => "temple jungle",
        _ => "forest",
    };
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("Wildfire rages through the {} of {}!", biome, region.name),
            entity_ids: vec![],
        },
    });

    for sid in settlements_near_region(state, region) {
        if let Some(s) = state.settlement(sid) {
            let wood_loss = s.stockpile[crate::world_sim::commodity::WOOD] * 0.5;
            if wood_loss > 0.0 {
                out.push(WorldDelta::ConsumeCommodity {
                    settlement_id: sid, commodity: crate::world_sim::commodity::WOOD, amount: wood_loss,
                });
            }
            let herb_loss = s.stockpile[crate::world_sim::commodity::HERBS] * 0.3;
            if herb_loss > 0.0 {
                out.push(WorldDelta::ConsumeCommodity {
                    settlement_id: sid, commodity: crate::world_sim::commodity::HERBS, amount: herb_loss,
                });
            }
        }
    }

    // Fire clears dense forest monsters.
    if region.sub_biome.monster_stealth() > 0.0 {
        out.push(WorldDelta::UpdateRegion {
            region_id: region.id, field: RegionField::MonsterDensity, value: -15.0,
        });
    }
}

fn cave_collapse(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("A cave-in rumbles through the caverns of {}, exposing new ore veins.", region.name),
            entity_ids: vec![],
        },
    });

    // Expose ore — produce iron/crystal at nearby settlements.
    for sid in settlements_near_region(state, region) {
        out.push(WorldDelta::ProduceCommodity {
            settlement_id: sid, commodity: crate::world_sim::commodity::IRON, amount: 10.0,
        });
        out.push(WorldDelta::ProduceCommodity {
            settlement_id: sid, commodity: crate::world_sim::commodity::CRYSTAL, amount: 5.0,
        });
    }

    // Damage dungeon sites — reset explored depth.
    // (Would need mutable access; skip for now — chronicle is the main effect.)
}

fn sandstorm(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    let variant = match region.sub_biome {
        SubBiome::SandDunes => "sand dunes",
        SubBiome::RockyDesert => "rocky wastes",
        _ => "desert",
    };
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("A blinding sandstorm sweeps across the {} of {}.", variant, region.name),
            entity_ids: vec![],
        },
    });

    // Damage goods in transit (reduce stockpiles slightly).
    for sid in settlements_near_region(state, region) {
        if let Some(s) = state.settlement(sid) {
            for c in 0..NUM_COMMODITIES {
                let loss = s.stockpile[c] * 0.1;
                if loss > 0.5 {
                    out.push(WorldDelta::ConsumeCommodity { settlement_id: sid, commodity: c, amount: loss });
                }
            }
        }
    }

    // Temporarily increase threat.
    out.push(WorldDelta::UpdateRegion {
        region_id: region.id, field: RegionField::ThreatLevel, value: 0.15,
    });
}

fn corruption_pulse(state: &WorldState, region: &RegionState, out: &mut Vec<WorldDelta>) {
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Narrative,
            text: format!("A wave of corruption pulses outward from the death zone in {}, tainting the land.", region.name),
            entity_ids: vec![],
        },
    });

    // Increase monster density in the death zone and neighbors.
    out.push(WorldDelta::UpdateRegion {
        region_id: region.id, field: RegionField::MonsterDensity, value: 5.0,
    });
    for &nid in &region.neighbors {
        out.push(WorldDelta::UpdateRegion {
            region_id: nid, field: RegionField::MonsterDensity, value: 2.0,
        });
    }

    // Damage all living things nearby.
    let rx = (region.id as f32 * 137.5).sin() * 150.0;
    let ry = (region.id as f32 * 73.1).cos() * 150.0;
    for entity in &state.entities {
        if !entity.alive { continue; }
        let dx = entity.pos.0 - rx;
        let dy = entity.pos.1 - ry;
        if dx * dx + dy * dy < 5000.0 {
            out.push(WorldDelta::Damage {
                target_id: entity.id, amount: 10.0, source_id: 0,
            });
        }
    }
}
