//! Compositional mass scenario generator for building AI BC dataset generation.
//!
//! Two-layer architecture:
//! - **Layer 1: World State Generators** — 5 independent axes producing diverse settlement states
//! - **Layer 2: Pressure Injectors** — 24 event types that force specific oracle decisions
//!
//! Combinatorial explosion (8 terrain × 5 maturity × 5 resource × 6 roster × 5 quality =
//! 6,000 seed configs × 24+ pressures) generates massive dataset variety.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::features::compute_spatial_features;
use super::oracle::{strategic_oracle, structural_oracle};
use super::scenario_gen::{build_observation, populate_memory};
use super::types::*;
use crate::world_sim::city_grid::{CellState, CellTerrain, CityGrid, InfluenceMap};
use crate::world_sim::state::{
    tag, BuildingData, BuildingType, EnemyCapabilities, Entity, EntityKind, SettlementState,
    WorldState,
};
use crate::world_sim::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// SimpleRng — deterministic RNG (same as scenario_gen.rs)
// ---------------------------------------------------------------------------

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) ^ self.0) as u32
    }
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }
    fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }
    /// Random integer in [lo, hi] inclusive.
    fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        if lo >= hi {
            return lo;
        }
        lo + (self.next_u32() % (hi - lo + 1))
    }
    /// Random float in [lo, hi].
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
    /// Pick a random index from a weighted distribution. Returns index.
    fn weighted_pick(&mut self, weights: &[f32]) -> usize {
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return self.range_u32(0, weights.len().saturating_sub(1) as u32) as usize;
        }
        let mut r = self.next_f32() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

// ===========================================================================
// Axis enums
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TerrainType {
    FlatOpen = 0,
    RiverBisect = 1,
    Hillside = 2,
    CliffEdge = 3,
    Coastal = 4,
    Swamp = 5,
    ForestClearing = 6,
    MountainPass = 7,
}

const ALL_TERRAIN: [TerrainType; 8] = [
    TerrainType::FlatOpen,
    TerrainType::RiverBisect,
    TerrainType::Hillside,
    TerrainType::CliffEdge,
    TerrainType::Coastal,
    TerrainType::Swamp,
    TerrainType::ForestClearing,
    TerrainType::MountainPass,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MaturityLevel {
    Empty = 0,
    Sparse = 1,
    Moderate = 2,
    Dense = 3,
    Overgrown = 4,
}

const ALL_MATURITY: [MaturityLevel; 5] = [
    MaturityLevel::Empty,
    MaturityLevel::Sparse,
    MaturityLevel::Moderate,
    MaturityLevel::Dense,
    MaturityLevel::Overgrown,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ResourceProfile {
    Abundant = 0,
    Mixed = 1,
    Scarce = 2,
    Specialized = 3,
    Depleting = 4,
}

const ALL_RESOURCE: [ResourceProfile; 5] = [
    ResourceProfile::Abundant,
    ResourceProfile::Mixed,
    ResourceProfile::Scarce,
    ResourceProfile::Specialized,
    ResourceProfile::Depleting,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum NpcComposition {
    MilitaryHeavy = 0,
    CivilianHeavy = 1,
    Balanced = 2,
    EliteFew = 3,
    LargeLowLevel = 4,
    Specialist = 5,
}

const ALL_NPC: [NpcComposition; 6] = [
    NpcComposition::MilitaryHeavy,
    NpcComposition::CivilianHeavy,
    NpcComposition::Balanced,
    NpcComposition::EliteFew,
    NpcComposition::LargeLowLevel,
    NpcComposition::Specialist,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BuildingQuality {
    WellPlanned = 0,
    OrganicGrowth = 1,
    BattleDamaged = 2,
    UnderConstruction = 3,
    AbandonedDecayed = 4,
}

const ALL_QUALITY: [BuildingQuality; 5] = [
    BuildingQuality::WellPlanned,
    BuildingQuality::OrganicGrowth,
    BuildingQuality::BattleDamaged,
    BuildingQuality::UnderConstruction,
    BuildingQuality::AbandonedDecayed,
];

// ===========================================================================
// Pressure type enum — all 24 pressure types
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PressureType {
    // Military (0-7)
    InfantryRaid = 0,
    SiegeAssault = 1,
    WallJumpers = 2,
    Climbers = 3,
    Tunnelers = 4,
    Flyers = 5,
    MultiVector = 6,
    Infiltrators = 7,
    // Environmental (8-12)
    Flood = 8,
    FireOutbreak = 9,
    Earthquake = 10,
    Landslide = 11,
    Storm = 12,
    // Economic (13-16)
    ResourceDepletion = 13,
    TradeBoom = 14,
    SupplyDisruption = 15,
    ResourceDiscovery = 16,
    // Population (17-20)
    RefugeeWave = 17,
    PopulationDecline = 18,
    ClassTension = 19,
    SpecialistArrival = 20,
    // Temporal (21-23)
    WinterDeadline = 21,
    HarvestSurge = 22,
    BuildingDecay = 23,
}

const ALL_PRESSURES: [PressureType; 24] = [
    PressureType::InfantryRaid,
    PressureType::SiegeAssault,
    PressureType::WallJumpers,
    PressureType::Climbers,
    PressureType::Tunnelers,
    PressureType::Flyers,
    PressureType::MultiVector,
    PressureType::Infiltrators,
    PressureType::Flood,
    PressureType::FireOutbreak,
    PressureType::Earthquake,
    PressureType::Landslide,
    PressureType::Storm,
    PressureType::ResourceDepletion,
    PressureType::TradeBoom,
    PressureType::SupplyDisruption,
    PressureType::ResourceDiscovery,
    PressureType::RefugeeWave,
    PressureType::PopulationDecline,
    PressureType::ClassTension,
    PressureType::SpecialistArrival,
    PressureType::WinterDeadline,
    PressureType::HarvestSurge,
    PressureType::BuildingDecay,
];

impl PressureType {
    fn category(self) -> ChallengeCategory {
        match self as u8 {
            0..=7 => ChallengeCategory::Military,
            8..=12 => ChallengeCategory::Environmental,
            13..=16 => ChallengeCategory::Economic,
            17..=20 => ChallengeCategory::Population,
            21..=23 => ChallengeCategory::Temporal,
            _ => ChallengeCategory::Military,
        }
    }

    fn name(self) -> &'static str {
        match self {
            PressureType::InfantryRaid => "infantry_raid",
            PressureType::SiegeAssault => "siege_assault",
            PressureType::WallJumpers => "wall_jumpers",
            PressureType::Climbers => "climbers",
            PressureType::Tunnelers => "tunnelers",
            PressureType::Flyers => "flyers",
            PressureType::MultiVector => "multi_vector",
            PressureType::Infiltrators => "infiltrators",
            PressureType::Flood => "flood",
            PressureType::FireOutbreak => "fire",
            PressureType::Earthquake => "earthquake",
            PressureType::Landslide => "landslide",
            PressureType::Storm => "storm",
            PressureType::ResourceDepletion => "resource_depletion",
            PressureType::TradeBoom => "trade_boom",
            PressureType::SupplyDisruption => "supply_disruption",
            PressureType::ResourceDiscovery => "resource_discovery",
            PressureType::RefugeeWave => "refugee_wave",
            PressureType::PopulationDecline => "population_decline",
            PressureType::ClassTension => "class_tension",
            PressureType::SpecialistArrival => "specialist_arrival",
            PressureType::WinterDeadline => "winter_deadline",
            PressureType::HarvestSurge => "harvest_surge",
            PressureType::BuildingDecay => "building_decay",
        }
    }
}

// ===========================================================================
// Layer 1: World State Generators
// ===========================================================================

/// Generate terrain on the city grid. Modifies the grid's cell terrain, water cells,
/// and returns a resource modifier array [food, iron, wood, herbs, hide, crystal, equipment, medicine].
fn generate_terrain(
    terrain_type: TerrainType,
    grid: &mut CityGrid,
    rng: &mut SimpleRng,
) -> [f32; NUM_COMMODITIES] {
    let cols = grid.cols;
    let rows = grid.rows;
    let mut resource_mods = [1.0_f32; NUM_COMMODITIES];

    match terrain_type {
        TerrainType::FlatOpen => {
            // ~90% buildable, all flat. Baseline.
            for r in 0..rows {
                for c in 0..cols {
                    grid.cell_mut(c, r).terrain = CellTerrain::Flat;
                }
            }
            resource_mods[0] = 1.5; // food bonus on open plains
        }
        TerrainType::RiverBisect => {
            // ~70% buildable. River runs through the middle.
            let river_col = cols / 2;
            let river_width = 3 + (rng.next_u32() % 3) as usize;
            for r in 0..rows {
                for c in 0..cols {
                    let cell = grid.cell_mut(c, r);
                    // Meander the river slightly.
                    let offset = ((r as f32 / 10.0).sin() * 3.0) as i32;
                    let rc = river_col as i32 + offset;
                    let dist = (c as i32 - rc).unsigned_abs() as usize;
                    if dist < river_width / 2 {
                        cell.terrain = CellTerrain::Water;
                        cell.state = CellState::Water;
                    } else if dist < river_width {
                        cell.terrain = CellTerrain::Flat; // river bank, flood risk
                    }
                }
            }
            resource_mods[0] = 1.3; // food from fishing
        }
        TerrainType::Hillside => {
            // ~75% buildable. Elevation gradient from south to north.
            for r in 0..rows {
                let row_frac = r as f32 / rows as f32;
                for c in 0..cols {
                    let cell = grid.cell_mut(c, r);
                    let noise = rng.next_f32() * 0.15;
                    let elev = row_frac + noise;
                    if elev > 0.85 {
                        cell.terrain = CellTerrain::Steep;
                    } else if elev > 0.5 {
                        cell.terrain = CellTerrain::Slope;
                    }
                }
            }
            resource_mods[1] = 1.3; // iron in hills
            resource_mods[5] = 1.2; // crystal
        }
        TerrainType::CliffEdge => {
            // ~60% buildable. One side is cliff/impassable.
            let cliff_start = (cols as f32 * 0.6) as usize;
            for r in 0..rows {
                for c in 0..cols {
                    let cell = grid.cell_mut(c, r);
                    if c >= cliff_start {
                        let noise = rng.next_f32() * 0.1;
                        let frac = (c - cliff_start) as f32 / (cols - cliff_start) as f32;
                        if frac + noise > 0.3 {
                            cell.terrain = CellTerrain::Cliff;
                        } else {
                            cell.terrain = CellTerrain::Steep;
                        }
                    }
                }
            }
            resource_mods[1] = 1.5; // iron from cliff face
        }
        TerrainType::Coastal => {
            // ~65% buildable. Water on one side, dock access.
            let water_depth = (cols as f32 * 0.25) as usize;
            for r in 0..rows {
                for c in 0..cols {
                    let cell = grid.cell_mut(c, r);
                    let noise = ((rng.next_f32() - 0.5) * 4.0) as i32;
                    let adjusted = c as i32 + noise;
                    if adjusted < water_depth as i32 {
                        cell.terrain = CellTerrain::Water;
                        cell.state = CellState::Water;
                    } else if adjusted < (water_depth + 3) as i32 {
                        cell.terrain = CellTerrain::Flat; // beach/dock area
                    }
                }
            }
            resource_mods[0] = 1.2; // fishing
        }
        TerrainType::Swamp => {
            // ~50% buildable. Scattered water and difficult ground.
            for r in 0..rows {
                for c in 0..cols {
                    let cell = grid.cell_mut(c, r);
                    let noise = rng.next_f32();
                    if noise > 0.7 {
                        cell.terrain = CellTerrain::Water;
                        cell.state = CellState::Water;
                    }
                }
            }
            resource_mods[3] = 2.0; // herbs abundant in swamp
            resource_mods[7] = 1.5; // medicine
        }
        TerrainType::ForestClearing => {
            // ~55% buildable. Dense trees on edges, clearing in center.
            let cx = cols / 2;
            let cy = rows / 2;
            let clearing_r = (cols.min(rows) as f32 * 0.25) as usize;
            for r in 0..rows {
                for c in 0..cols {
                    let dx = (c as i32 - cx as i32).unsigned_abs() as usize;
                    let dy = (r as i32 - cy as i32).unsigned_abs() as usize;
                    let dist = ((dx * dx + dy * dy) as f32).sqrt() as usize;
                    let cell = grid.cell_mut(c, r);
                    if dist > clearing_r {
                        // Forest = slope terrain (heuristic for "tree-covered")
                        let noise = rng.next_f32();
                        if noise > 0.3 {
                            cell.terrain = CellTerrain::Slope;
                        }
                    }
                }
            }
            resource_mods[2] = 3.0; // wood abundant
            resource_mods[3] = 1.5; // herbs
        }
        TerrainType::MountainPass => {
            // ~40% buildable. Natural chokepoint between two mountain walls.
            let pass_center = rows / 2;
            let pass_width = (rows as f32 * 0.2) as usize;
            for r in 0..rows {
                for c in 0..cols {
                    let cell = grid.cell_mut(c, r);
                    let dist_from_pass = (r as i32 - pass_center as i32).unsigned_abs() as usize;
                    let noise = (rng.next_f32() * 5.0) as usize;
                    if dist_from_pass > pass_width + noise {
                        cell.terrain = CellTerrain::Cliff;
                    } else if dist_from_pass > pass_width / 2 + noise / 2 {
                        cell.terrain = CellTerrain::Steep;
                    } else {
                        cell.terrain = CellTerrain::Slope;
                    }
                }
            }
            resource_mods[1] = 2.0; // iron/stone in mountains
            resource_mods[5] = 1.5; // crystal
        }
    }

    resource_mods
}

/// Check if a cell is buildable (not water, not cliff).
fn is_buildable(grid: &CityGrid, col: usize, row: usize) -> bool {
    if !grid.in_bounds(col, row) {
        return false;
    }
    let cell = grid.cell(col, row);
    cell.state != CellState::Water
        && cell.terrain != CellTerrain::Water
        && cell.terrain != CellTerrain::Cliff
}

/// Place buildings for a given maturity level. Returns building entity list (not yet in state).
fn generate_maturity(
    level: MaturityLevel,
    state: &mut WorldState,
    rng: &mut SimpleRng,
) {
    let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));
    let gi = match state.settlements.first().and_then(|s| s.city_grid_idx) {
        Some(idx) => idx,
        None => return,
    };

    let (building_count, wall_coverage, settlement_level, organized) = match level {
        MaturityLevel::Empty => (rng.range_u32(0, 2) as usize, 0.0_f32, 1, false),
        MaturityLevel::Sparse => (rng.range_u32(3, 8) as usize, 0.1, 1, false),
        MaturityLevel::Moderate => (rng.range_u32(8, 20) as usize, 0.4, 2, true),
        MaturityLevel::Dense => (rng.range_u32(20, 40) as usize, 0.8, 3, true),
        MaturityLevel::Overgrown => (rng.range_u32(40, 55) as usize, 0.9, 4, false),
    };

    state.settlements[0].infrastructure_level = settlement_level as f32;

    // Set settlement age.
    state.tick = match level {
        MaturityLevel::Empty => rng.range_u32(0, 100) as u64,
        MaturityLevel::Sparse => rng.range_u32(100, 500) as u64,
        MaturityLevel::Moderate => rng.range_u32(500, 2000) as u64,
        MaturityLevel::Dense => rng.range_u32(2000, 5000) as u64,
        MaturityLevel::Overgrown => rng.range_u32(5000, 10000) as u64,
    };

    let grid_center = 64_usize;

    // Building type distributions by maturity.
    let building_types: &[BuildingType] = match level {
        MaturityLevel::Empty => &[BuildingType::Tent, BuildingType::Camp],
        MaturityLevel::Sparse => &[
            BuildingType::House,
            BuildingType::House,
            BuildingType::Farm,
            BuildingType::Well,
        ],
        MaturityLevel::Moderate => &[
            BuildingType::House,
            BuildingType::House,
            BuildingType::House,
            BuildingType::Farm,
            BuildingType::Forge,
            BuildingType::Market,
            BuildingType::Barracks,
            BuildingType::Watchtower,
            BuildingType::Temple,
        ],
        MaturityLevel::Dense => &[
            BuildingType::House,
            BuildingType::House,
            BuildingType::Longhouse,
            BuildingType::Manor,
            BuildingType::Farm,
            BuildingType::Forge,
            BuildingType::Market,
            BuildingType::Warehouse,
            BuildingType::Barracks,
            BuildingType::Watchtower,
            BuildingType::Temple,
            BuildingType::GuildHall,
            BuildingType::Library,
            BuildingType::Inn,
        ],
        MaturityLevel::Overgrown => &[
            BuildingType::House,
            BuildingType::House,
            BuildingType::House,
            BuildingType::House,
            BuildingType::Longhouse,
            BuildingType::Manor,
            BuildingType::Farm,
            BuildingType::Farm,
            BuildingType::Forge,
            BuildingType::Market,
            BuildingType::Warehouse,
            BuildingType::Barracks,
            BuildingType::Watchtower,
            BuildingType::Temple,
            BuildingType::GuildHall,
            BuildingType::Library,
            BuildingType::Inn,
            BuildingType::TradePost,
            BuildingType::CourtHouse,
        ],
    };

    // Place buildings.
    for i in 0..building_count {
        let btype = building_types[i % building_types.len()];
        let tier = match level {
            MaturityLevel::Empty => 1,
            MaturityLevel::Sparse => 1,
            MaturityLevel::Moderate => rng.range_u32(1, 2) as u8,
            MaturityLevel::Dense => rng.range_u32(1, 3) as u8,
            MaturityLevel::Overgrown => rng.range_u32(1, 3) as u8,
        };

        // Placement: grid-aligned for organized, random walk for organic.
        let (col, row) = if organized {
            // Grid-aligned: rows and columns with spacing.
            let grid_spacing = 4;
            let row_idx = i / 8;
            let col_idx = i % 8;
            let c = grid_center + col_idx * grid_spacing;
            let r = grid_center + row_idx * grid_spacing;
            (c.min(126), r.min(126))
        } else {
            // Random walk from center.
            let angle = rng.next_f32() * std::f32::consts::TAU;
            let dist = rng.range_f32(3.0, 30.0 + i as f32 * 0.5);
            let c = (grid_center as f32 + angle.cos() * dist) as usize;
            let r = (grid_center as f32 + angle.sin() * dist) as usize;
            (c.clamp(1, 126), r.clamp(1, 126))
        };

        // Skip if not buildable.
        if !is_buildable(&state.city_grids[gi], col, row) {
            continue;
        }

        let eid = state.next_entity_id();
        let mut entity = Entity::new_building(eid, settlement_pos);
        entity.building = Some(BuildingData {
            building_type: btype,
            settlement_id: Some(settlement_id),
            grid_col: col as u16,
            grid_row: row as u16,
            footprint_w: 2,
            footprint_h: 2,
            tier,
            room_seed: eid as u64 ^ rng.next_u32() as u64,
            rooms: btype.default_rooms(),
            residential_capacity: btype.residential_capacity(),
            work_capacity: btype.work_capacity(),
            resident_ids: Vec::new(),
            worker_ids: Vec::new(),
            construction_progress: 1.0,
            built_tick: 0,
            builder_id: None,
            temporary: matches!(btype, BuildingType::Tent | BuildingType::Camp),
            ttl_ticks: None,
            name: format!("{:?} #{}", btype, i),
            storage: [0.0; NUM_COMMODITIES],
            storage_capacity: btype.storage_capacity(),
            owner_id: None,
            builder_modifiers: Vec::new(),
            owner_modifiers: Vec::new(),
            worker_class_ticks: Vec::new(),
            specialization_tag: None,
            specialization_strength: 0.0,
            specialization_name: String::new(),
            structural: None,
        });
        state.entities.push(entity);

        // Mark grid cell.
        let g = &mut state.city_grids[gi];
        if g.in_bounds(col, row) {
            let c = g.cell_mut(col, row);
            c.state = CellState::Building;
            c.building_id = Some(eid);
        }
    }

    // Place wall segments according to wall_coverage fraction.
    if wall_coverage > 0.0 {
        let perimeter_cells = compute_perimeter_positions(grid_center, 20);
        let wall_count = (perimeter_cells.len() as f32 * wall_coverage) as usize;

        for (wi, &(wc, wr)) in perimeter_cells.iter().take(wall_count).enumerate() {
            if !is_buildable(&state.city_grids[gi], wc, wr) {
                continue;
            }

            let eid = state.next_entity_id();
            let btype = if wi % 15 == 0 {
                BuildingType::Gate
            } else {
                BuildingType::Wall
            };
            let mut entity = Entity::new_building(eid, settlement_pos);
            entity.building = Some(BuildingData {
                building_type: btype,
                settlement_id: Some(settlement_id),
                grid_col: wc as u16,
                grid_row: wr as u16,
                footprint_w: 1,
                footprint_h: 1,
                tier: settlement_level.min(3),
                room_seed: eid as u64,
                rooms: btype.default_rooms(),
                residential_capacity: 0,
                work_capacity: 0,
                resident_ids: Vec::new(),
                worker_ids: Vec::new(),
                construction_progress: 1.0,
                built_tick: 0,
                builder_id: None,
                temporary: false,
                ttl_ticks: None,
                name: format!("{:?}", btype),
                storage: [0.0; NUM_COMMODITIES],
                storage_capacity: 0.0,
                owner_id: None,
                builder_modifiers: Vec::new(),
                owner_modifiers: Vec::new(),
                worker_class_ticks: Vec::new(),
                specialization_tag: None,
                specialization_strength: 0.0,
                specialization_name: String::new(),
                structural: None,
            });
            state.entities.push(entity);

            let g = &mut state.city_grids[gi];
            if g.in_bounds(wc, wr) {
                let c = g.cell_mut(wc, wr);
                c.state = CellState::Wall;
                c.building_id = Some(eid);
            }
        }
    }
}

/// Compute perimeter cell positions around a center at a given radius.
fn compute_perimeter_positions(center: usize, radius: usize) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    let lo = center.saturating_sub(radius);
    let hi = (center + radius).min(126);

    // Top and bottom edges.
    for c in lo..=hi {
        positions.push((c, lo));
        positions.push((c, hi));
    }
    // Left and right edges (excluding corners).
    for r in (lo + 1)..hi {
        positions.push((lo, r));
        positions.push((hi, r));
    }
    positions
}

/// Apply a resource profile to the settlement stockpiles.
fn apply_resource_profile(
    profile: ResourceProfile,
    state: &mut WorldState,
    resource_mods: &[f32; NUM_COMMODITIES],
    rng: &mut SimpleRng,
) {
    use crate::world_sim::commodity::*;

    let settlement = match state.settlements.first_mut() {
        Some(s) => s,
        None => return,
    };

    // Base stockpile amounts by profile.
    let (food, iron, wood, herbs) = match profile {
        ResourceProfile::Abundant => (500.0, 400.0, 400.0, 200.0),
        ResourceProfile::Mixed => (400.0, 50.0, 300.0, 100.0),
        ResourceProfile::Scarce => (100.0, 50.0, 80.0, 30.0),
        ResourceProfile::Specialized => (200.0, 0.0, 800.0, 50.0),
        ResourceProfile::Depleting => (150.0, 100.0, 120.0, 60.0),
    };

    // Apply resource modifiers from terrain and add noise.
    settlement.stockpile[FOOD] = food * resource_mods[FOOD] * rng.range_f32(0.8, 1.2);
    settlement.stockpile[IRON] = iron * resource_mods[IRON] * rng.range_f32(0.8, 1.2);
    settlement.stockpile[WOOD] = wood * resource_mods[WOOD] * rng.range_f32(0.8, 1.2);
    settlement.stockpile[HERBS] = herbs * resource_mods[HERBS] * rng.range_f32(0.8, 1.2);
    settlement.stockpile[HIDE] = rng.range_f32(20.0, 100.0) * resource_mods[HIDE];
    settlement.stockpile[CRYSTAL] = rng.range_f32(0.0, 50.0) * resource_mods[CRYSTAL];
    settlement.stockpile[EQUIPMENT] = rng.range_f32(10.0, 80.0);
    settlement.stockpile[MEDICINE] = rng.range_f32(5.0, 40.0) * resource_mods[MEDICINE];

    // Treasury gold.
    settlement.treasury = match profile {
        ResourceProfile::Abundant => rng.range_f32(200.0, 500.0),
        ResourceProfile::Mixed => rng.range_f32(100.0, 300.0),
        ResourceProfile::Scarce => rng.range_f32(10.0, 80.0),
        ResourceProfile::Specialized => rng.range_f32(50.0, 200.0),
        ResourceProfile::Depleting => rng.range_f32(30.0, 150.0),
    };
}

/// Generate NPC roster with appropriate class/level distributions.
fn generate_npc_roster(
    composition: NpcComposition,
    state: &mut WorldState,
    rng: &mut SimpleRng,
) {
    let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));

    let (total, combat_frac, hv_count, level_range) = match composition {
        NpcComposition::MilitaryHeavy => (
            rng.range_u32(20, 40) as usize,
            0.6,
            rng.range_u32(1, 2) as usize,
            (2, 5),
        ),
        NpcComposition::CivilianHeavy => (
            rng.range_u32(25, 50) as usize,
            0.1,
            1,
            (1, 3),
        ),
        NpcComposition::Balanced => (
            rng.range_u32(20, 35) as usize,
            0.3,
            rng.range_u32(1, 2) as usize,
            (1, 4),
        ),
        NpcComposition::EliteFew => (
            rng.range_u32(8, 15) as usize,
            0.4,
            1,
            (4, 8),
        ),
        NpcComposition::LargeLowLevel => (
            rng.range_u32(40, 70) as usize,
            0.2,
            rng.range_u32(0, 1) as usize,
            (1, 2),
        ),
        NpcComposition::Specialist => (
            rng.range_u32(15, 30) as usize,
            0.2,
            rng.range_u32(3, 4) as usize,
            (2, 5),
        ),
    };

    let combat_count = (total as f32 * combat_frac) as usize;
    let worker_count = total - combat_count;

    // Spawn combat NPCs.
    let combat_classes = ["warrior", "archer", "defender"];
    for i in 0..combat_count {
        let eid = state.next_entity_id();
        let level = rng.range_u32(level_range.0, level_range.1);
        let class = combat_classes[i % combat_classes.len()];
        let mut entity = Entity::new_npc(eid, settlement_pos);
        entity.level = level;
        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
            npc.class_tags = vec![class.to_string()];
            npc.archetype = "garrison".into();
        }
        state.entities.push(entity);
    }

    // Spawn worker NPCs.
    let worker_classes = ["farmer", "builder", "miner", "craftsman", "merchant"];
    for i in 0..worker_count {
        let eid = state.next_entity_id();
        let level = rng.range_u32(1, level_range.1.min(3));
        let class = worker_classes[i % worker_classes.len()];
        let mut entity = Entity::new_npc(eid, settlement_pos);
        entity.level = level;
        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
            npc.class_tags = vec![class.to_string()];
        }
        state.entities.push(entity);
    }

    // Spawn high-value NPCs.
    let hv_roles = [
        "leader",
        "master_smith",
        "archmage",
        "high_priest",
        "commander",
        "guild_master",
    ];
    for i in 0..hv_count {
        let eid = state.next_entity_id();
        let level = rng.range_u32(level_range.1, level_range.1 + 3);
        let role = hv_roles[i % hv_roles.len()];
        let mut entity = Entity::new_npc(eid, settlement_pos);
        entity.level = level;
        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
            npc.archetype = role.to_string();
        }
        state.entities.push(entity);
    }

    // Update population count.
    if let Some(s) = state.settlements.first_mut() {
        s.population = (combat_count + worker_count + hv_count) as u32;
    }
}

/// Apply building quality modifications to existing buildings in the state.
fn apply_building_quality(
    quality: BuildingQuality,
    state: &mut WorldState,
    rng: &mut SimpleRng,
) {
    let gi = match state.settlements.first().and_then(|s| s.city_grid_idx) {
        Some(idx) => idx,
        None => return,
    };

    let building_ids: Vec<u32> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .map(|e| e.id)
        .collect();

    for &bid in &building_ids {
        let entity = match state.entities.iter_mut().find(|e| e.id == bid) {
            Some(e) => e,
            None => continue,
        };
        let bd = match entity.building.as_mut() {
            Some(b) => b,
            None => continue,
        };

        match quality {
            BuildingQuality::WellPlanned => {
                // Good condition, proper zones.
                bd.construction_progress = 1.0;
            }
            BuildingQuality::OrganicGrowth => {
                // Patchy condition, mixed zoning.
                bd.construction_progress = rng.range_f32(0.7, 1.0);
            }
            BuildingQuality::BattleDamaged => {
                // Damaged buildings, broken walls.
                bd.construction_progress = rng.range_f32(0.2, 0.7);
                entity.hp *= rng.range_f32(0.3, 0.8);
            }
            BuildingQuality::UnderConstruction => {
                // Partial completion.
                bd.construction_progress = rng.range_f32(0.1, 0.6);
            }
            BuildingQuality::AbandonedDecayed => {
                // Crumbling.
                bd.construction_progress = rng.range_f32(0.1, 0.4);
                entity.hp *= rng.range_f32(0.1, 0.5);
            }
        }
    }

    // Set zone types on grid cells for quality types that affect zoning.
    match quality {
        BuildingQuality::WellPlanned => {
            // Proper zones: residential in center, military on perimeter.
            apply_grid_zones(
                &mut state.city_grids[gi],
                &[
                    crate::world_sim::city_grid::ZoneType::Residential,
                    crate::world_sim::city_grid::ZoneType::Commercial,
                ],
                rng,
            );
        }
        BuildingQuality::OrganicGrowth => {
            // Mixed zones everywhere.
            apply_grid_zones(
                &mut state.city_grids[gi],
                &[
                    crate::world_sim::city_grid::ZoneType::Residential,
                    crate::world_sim::city_grid::ZoneType::Commercial,
                    crate::world_sim::city_grid::ZoneType::Industrial,
                ],
                rng,
            );
        }
        _ => {} // Other qualities don't need zone changes.
    }
}

fn apply_grid_zones(
    grid: &mut CityGrid,
    zones: &[crate::world_sim::city_grid::ZoneType],
    rng: &mut SimpleRng,
) {
    if zones.is_empty() {
        return;
    }
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let cell = grid.cell_mut(c, r);
            if cell.state == CellState::Building {
                let idx = rng.range_u32(0, zones.len().saturating_sub(1) as u32) as usize;
                cell.zone = zones[idx];
            }
        }
    }
}

// ===========================================================================
// Layer 2: Pressure Injectors
// ===========================================================================

/// Metadata about what decision pressures a pressure injector creates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureMetadata {
    pub pressure_type: PressureType,
    /// Which (category_row, decision_col) cells in the coverage matrix this activates.
    pub activated_cells: Vec<(usize, usize)>,
}

/// Inject a pressure into the world state and return a Challenge + metadata.
fn inject_pressure(
    pressure: PressureType,
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    match pressure {
        PressureType::InfantryRaid => inject_infantry_raid(state, severity, rng),
        PressureType::SiegeAssault => inject_siege_assault(state, severity, rng),
        PressureType::WallJumpers => inject_wall_jumpers(state, severity, rng),
        PressureType::Climbers => inject_climbers(state, severity, rng),
        PressureType::Tunnelers => inject_tunnelers(state, severity, rng),
        PressureType::Flyers => inject_flyers(state, severity, rng),
        PressureType::MultiVector => inject_multi_vector(state, severity, rng),
        PressureType::Infiltrators => inject_infiltrators(state, severity, rng),
        PressureType::Flood => inject_flood(state, severity, rng),
        PressureType::FireOutbreak => inject_fire_outbreak(state, severity, rng),
        PressureType::Earthquake => inject_earthquake(state, severity, rng),
        PressureType::Landslide => inject_landslide(state, severity, rng),
        PressureType::Storm => inject_storm(state, severity, rng),
        PressureType::ResourceDepletion => inject_resource_depletion(state, severity, rng),
        PressureType::TradeBoom => inject_trade_boom(state, severity, rng),
        PressureType::SupplyDisruption => inject_supply_disruption(state, severity, rng),
        PressureType::ResourceDiscovery => inject_resource_discovery(state, severity, rng),
        PressureType::RefugeeWave => inject_refugee_wave(state, severity, rng),
        PressureType::PopulationDecline => inject_population_decline(state, severity, rng),
        PressureType::ClassTension => inject_class_tension(state, severity, rng),
        PressureType::SpecialistArrival => inject_specialist_arrival(state, severity, rng),
        PressureType::WinterDeadline => inject_winter_deadline(state, severity, rng),
        PressureType::HarvestSurge => inject_harvest_surge(state, severity, rng),
        PressureType::BuildingDecay => inject_building_decay(state, severity, rng),
    }
}

// ---------------------------------------------------------------------------
// Military pressure injectors
// ---------------------------------------------------------------------------

fn random_direction(rng: &mut SimpleRng) -> (f32, f32) {
    let dirs: [(f32, f32); 4] = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)];
    dirs[rng.range_u32(0, 3) as usize]
}

fn spawn_enemies(
    state: &mut WorldState,
    count: u16,
    level_range: (u8, u8),
    direction: (f32, f32),
    capabilities: EnemyCapabilities,
    type_name: &str,
    severity: f32,
    rng: &mut SimpleRng,
) -> Vec<EnemyProfile> {
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));
    let perimeter_dist = 80.0_f32;
    let base_x = settlement_pos.0 + direction.0 * perimeter_dist;
    let base_y = settlement_pos.1 + direction.1 * perimeter_dist;

    for i in 0..count {
        let eid = state.next_entity_id();
        let spread = (i as f32 - count as f32 / 2.0) * 3.0;
        let spawn_pos = (
            base_x + direction.1.abs() * spread,
            base_y + direction.0.abs() * spread,
        );
        let level = level_range.0 as u32
            + rng.range_u32(0, (level_range.1 - level_range.0) as u32);
        let mut entity = Entity::new_monster(eid, spawn_pos, level);
        entity.hp *= 1.0 + severity * 0.5;
        entity.max_hp = entity.hp;
        entity.attack_damage *= 1.0 + severity * 0.3;
        entity.move_target = Some(settlement_pos);
        entity.enemy_capabilities = Some(capabilities);
        if let Some(npc) = entity.npc.as_mut() {
            npc.archetype = type_name.to_string();
        }
        state.entities.push(entity);
    }

    if let Some(s) = state.settlements.first_mut() {
        s.threat_level = (s.threat_level + severity).min(1.0);
    }

    vec![EnemyProfile {
        type_tag: tag(type_name.as_bytes()),
        type_name: type_name.to_string(),
        level_range,
        count,
        can_jump: capabilities.can_jump,
        jump_height: capabilities.jump_height,
        can_climb: capabilities.can_climb,
        can_tunnel: capabilities.can_tunnel,
        can_fly: capabilities.can_fly,
        has_siege: capabilities.has_siege,
        siege_damage: capabilities.siege_damage,
    }]
}

fn make_challenge(
    category: ChallengeCategory,
    sub_type_name: &str,
    severity: f32,
    direction: Option<(f32, f32)>,
    deadline_tick: Option<u64>,
    enemy_profiles: Vec<EnemyProfile>,
) -> Challenge {
    Challenge {
        category,
        sub_type: tag(sub_type_name.as_bytes()),
        sub_type_name: sub_type_name.to_string(),
        severity,
        direction,
        deadline_tick,
        enemy_profiles,
    }
}

fn inject_infantry_raid(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(5, 15 + (severity * 10.0) as u32) as u16;
    let profiles = spawn_enemies(
        state,
        count,
        (1, 3),
        dir,
        EnemyCapabilities::default(),
        "infantry",
        severity,
        rng,
    );
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "infantry_raid",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::InfantryRaid,
        activated_cells: vec![
            (0, 0),  // Military × Placement
            (0, 7),  // Military × WallComposition
            (0, 14), // Military × DefensiveIntegration
        ],
    };
    (challenge, meta)
}

fn inject_siege_assault(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(3, 8) as u16;
    let caps = EnemyCapabilities {
        has_siege: true,
        siege_damage: severity * 50.0,
        ..Default::default()
    };
    let profiles = spawn_enemies(state, count, (2, 5), dir, caps, "siege_engine", severity, rng);
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "siege_assault",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::SiegeAssault,
        activated_cells: vec![
            (0, 0),  // Military × Placement
            (0, 7),  // Military × WallComposition
            (0, 13), // Military × MaterialSelection
        ],
    };
    (challenge, meta)
}

fn inject_wall_jumpers(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(4, 10) as u16;
    let jump_h = rng.range_u32(3, 5) as u8;
    let caps = EnemyCapabilities {
        can_jump: true,
        jump_height: jump_h,
        ..Default::default()
    };
    let profiles = spawn_enemies(state, count, (2, 4), dir, caps, "wall_jumper", severity, rng);
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "wall_jumpers",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::WallJumpers,
        activated_cells: vec![
            (0, 6),  // Military × VerticalDesign (wall height)
            (0, 7),  // Military × WallComposition
            (0, 14), // Military × DefensiveIntegration
        ],
    };
    (challenge, meta)
}

fn inject_climbers(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(4, 10) as u16;
    let caps = EnemyCapabilities {
        can_climb: true,
        ..Default::default()
    };
    let profiles = spawn_enemies(state, count, (2, 4), dir, caps, "climber", severity, rng);
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "climbers",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Climbers,
        activated_cells: vec![
            (0, 7),  // Military × WallComposition (smooth walls)
            (0, 13), // Military × MaterialSelection
        ],
    };
    (challenge, meta)
}

fn inject_tunnelers(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(3, 8) as u16;
    let caps = EnemyCapabilities {
        can_tunnel: true,
        ..Default::default()
    };
    let profiles = spawn_enemies(state, count, (2, 5), dir, caps, "tunneler", severity, rng);
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "tunnelers",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Tunnelers,
        activated_cells: vec![
            (0, 9),  // Military × Foundation (deep foundations)
            (0, 14), // Military × DefensiveIntegration
        ],
    };
    (challenge, meta)
}

fn inject_flyers(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(3, 8) as u16;
    let caps = EnemyCapabilities {
        can_fly: true,
        ..Default::default()
    };
    let profiles = spawn_enemies(state, count, (2, 5), dir, caps, "flyer", severity, rng);
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "flyers",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Flyers,
        activated_cells: vec![
            (0, 8),  // Military × RoofDesign
            (0, 12), // Military × RoomSpecialization (safe rooms)
        ],
    };
    (challenge, meta)
}

fn inject_multi_vector(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // 2-3 directions, mixed types.
    let dir1 = random_direction(rng);
    let dir2 = (-dir1.0, -dir1.1); // Opposite direction.

    let count1 = rng.range_u32(3, 8) as u16;
    let count2 = rng.range_u32(3, 8) as u16;

    let mut profiles = spawn_enemies(
        state,
        count1,
        (1, 3),
        dir1,
        EnemyCapabilities::default(),
        "infantry",
        severity,
        rng,
    );
    let caps2 = EnemyCapabilities {
        can_jump: true,
        jump_height: 3,
        ..Default::default()
    };
    profiles.extend(spawn_enemies(
        state, count2, (2, 4), dir2, caps2, "wall_jumper", severity, rng,
    ));

    let challenge = make_challenge(
        ChallengeCategory::Military,
        "multi_vector",
        severity,
        Some(dir1),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::MultiVector,
        activated_cells: vec![
            (0, 0),  // Military × Placement (perimeter coverage)
            (0, 6),  // Military × VerticalDesign
            (0, 14), // Military × DefensiveIntegration
        ],
    };
    (challenge, meta)
}

fn inject_infiltrators(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let dir = random_direction(rng);
    let count = rng.range_u32(2, 5) as u16;
    let caps = EnemyCapabilities::default(); // Stealth is implicit.
    let profiles = spawn_enemies(state, count, (3, 6), dir, caps, "infiltrator", severity, rng);
    let challenge = make_challenge(
        ChallengeCategory::Military,
        "infiltrators",
        severity,
        Some(dir),
        None,
        profiles,
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Infiltrators,
        activated_cells: vec![
            (0, 10), // Military × Openings (chokepoints)
            (0, 11), // Military × InteriorFlow
            (0, 12), // Military × RoomSpecialization (HV NPC protection)
        ],
    };
    (challenge, meta)
}

// ---------------------------------------------------------------------------
// Environmental pressure injectors
// ---------------------------------------------------------------------------

fn inject_flood(
    state: &mut WorldState,
    severity: f32,
    _rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    if let Some(gi) = state.settlements.first().and_then(|s| s.city_grid_idx) {
        let grid = &mut state.city_grids[gi];
        let rows_to_flood = (grid.rows as f32 * severity * 0.3) as usize;
        for row in 0..rows_to_flood.min(grid.rows) {
            for col in 0..grid.cols {
                let cell = grid.cell_mut(col, row);
                if cell.state == CellState::Empty {
                    cell.terrain = CellTerrain::Water;
                    cell.state = CellState::Water;
                }
            }
        }
    }

    // Add flood damage memory.
    if let Some(s) = state.settlements.first_mut() {
        s.construction_memory.short_term.push(ConstructionEvent {
            tick: state.tick,
            kind: ConstructionEventKind::FloodDamage,
            severity,
            location: (64, 10),
            source_entity: None,
        });
    }

    let challenge = make_challenge(
        ChallengeCategory::Environmental,
        "flood",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Flood,
        activated_cells: vec![
            (1, 2),  // Environmental × Routing (drainage)
            (1, 9),  // Environmental × Foundation (raised)
            (1, 15), // Environmental × EnvironmentalAdaptation
        ],
    };
    (challenge, meta)
}

fn inject_fire_outbreak(
    state: &mut WorldState,
    severity: f32,
    _rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    if let Some(gi) = state.settlements.first().and_then(|s| s.city_grid_idx) {
        let grid = &mut state.city_grids[gi];
        let edge = if severity > 0.5 { 10 } else { 5 };
        for col in 0..edge.min(grid.cols) {
            for row in 0..grid.rows {
                let cell = grid.cell_mut(col, row);
                if cell.state == CellState::Empty {
                    cell.density = 1; // wood-density marker
                }
            }
        }
    }

    if let Some(s) = state.settlements.first_mut() {
        s.construction_memory.short_term.push(ConstructionEvent {
            tick: state.tick,
            kind: ConstructionEventKind::FireSpread,
            severity,
            location: (5, 64),
            source_entity: None,
        });
    }

    let challenge = make_challenge(
        ChallengeCategory::Environmental,
        "fire",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::FireOutbreak,
        activated_cells: vec![
            (1, 3),  // Environmental × ZoneComposition (fire breaks)
            (1, 13), // Environmental × MaterialSelection (non-wood)
            (1, 15), // Environmental × EnvironmentalAdaptation
        ],
    };
    (challenge, meta)
}

fn inject_earthquake(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Damage some buildings.
    let damage_frac = severity * 0.4;
    for entity in state.entities.iter_mut() {
        if entity.kind == EntityKind::Building && entity.alive {
            if rng.next_f32() < damage_frac {
                entity.hp *= rng.range_f32(0.3, 0.8);
                if let Some(bd) = entity.building.as_mut() {
                    bd.construction_progress *= rng.range_f32(0.5, 0.9);
                }
            }
        }
    }

    if let Some(s) = state.settlements.first_mut() {
        s.construction_memory.short_term.push(ConstructionEvent {
            tick: state.tick,
            kind: ConstructionEventKind::EarthquakeDamage,
            severity,
            location: (64, 64),
            source_entity: None,
        });
    }

    let challenge = make_challenge(
        ChallengeCategory::Environmental,
        "earthquake",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Earthquake,
        activated_cells: vec![
            (1, 5),  // Environmental × FootprintGeometry (wide)
            (1, 9),  // Environmental × Foundation (deep, wide)
            (1, 17), // Environmental × RenovationUpgrade
        ],
    };
    (challenge, meta)
}

fn inject_landslide(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Mark steep terrain cells as collapsed.
    if let Some(gi) = state.settlements.first().and_then(|s| s.city_grid_idx) {
        let grid = &mut state.city_grids[gi];
        let affected_cols = (grid.cols as f32 * severity * 0.2) as usize;
        for col in 0..affected_cols.min(grid.cols) {
            for row in 0..grid.rows {
                let cell = grid.cell_mut(col, row);
                if cell.terrain == CellTerrain::Slope || cell.terrain == CellTerrain::Steep {
                    if rng.next_f32() < severity {
                        cell.terrain = CellTerrain::Cliff; // collapsed
                    }
                }
            }
        }
    }

    let challenge = make_challenge(
        ChallengeCategory::Environmental,
        "landslide",
        severity,
        Some((-1.0, 0.0)), // From west.
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Landslide,
        activated_cells: vec![
            (1, 0),  // Environmental × Placement (relocation)
            (1, 9),  // Environmental × Foundation (retaining)
            (1, 15), // Environmental × EnvironmentalAdaptation
        ],
    };
    (challenge, meta)
}

fn inject_storm(
    state: &mut WorldState,
    severity: f32,
    _rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Storms damage roofs, add deadline pressure.
    let deadline = state.tick + (500.0 * (1.0 - severity * 0.5)) as u64;

    let challenge = make_challenge(
        ChallengeCategory::Environmental,
        "storm",
        severity,
        None,
        Some(deadline),
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::Storm,
        activated_cells: vec![
            (1, 8),  // Environmental × RoofDesign (reinforced)
            (1, 13), // Environmental × MaterialSelection
            (1, 15), // Environmental × EnvironmentalAdaptation
        ],
    };
    (challenge, meta)
}

// ---------------------------------------------------------------------------
// Economic pressure injectors
// ---------------------------------------------------------------------------

fn inject_resource_depletion(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Deplete a random resource.
    let resource_idx = rng.range_u32(0, 3) as usize; // food, iron, wood, or herbs
    if let Some(s) = state.settlements.first_mut() {
        s.stockpile[resource_idx] *= 1.0 - severity * 0.7;
    }

    if let Some(s) = state.settlements.first_mut() {
        s.construction_memory.short_term.push(ConstructionEvent {
            tick: state.tick,
            kind: ConstructionEventKind::ResourceDepleted,
            severity,
            location: (64, 64),
            source_entity: None,
        });
    }

    let challenge = make_challenge(
        ChallengeCategory::Economic,
        "resource_depletion",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::ResourceDepletion,
        activated_cells: vec![
            (2, 1),  // Economic × Prioritization
            (2, 13), // Economic × MaterialSelection (substitution)
        ],
    };
    (challenge, meta)
}

fn inject_trade_boom(
    state: &mut WorldState,
    severity: f32,
    _rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Increase treasury, create demand for market/warehouse.
    if let Some(s) = state.settlements.first_mut() {
        s.treasury *= 1.0 + severity * 2.0;
    }

    let challenge = make_challenge(
        ChallengeCategory::Economic,
        "trade_boom",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::TradeBoom,
        activated_cells: vec![
            (2, 0), // Economic × Placement (market/warehouse)
            (2, 2), // Economic × Routing (road throughput)
        ],
    };
    (challenge, meta)
}

fn inject_supply_disruption(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Damage roads/paths.
    if let Some(gi) = state.settlements.first().and_then(|s| s.city_grid_idx) {
        let grid = &mut state.city_grids[gi];
        for r in 0..grid.rows {
            for c in 0..grid.cols {
                let cell = grid.cell_mut(c, r);
                if cell.state == CellState::Road && rng.next_f32() < severity * 0.3 {
                    cell.state = CellState::Ruin;
                    cell.road_tier = 0;
                }
            }
        }
    }

    let challenge = make_challenge(
        ChallengeCategory::Economic,
        "supply_disruption",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::SupplyDisruption,
        activated_cells: vec![
            (2, 2),  // Economic × Routing (rerouting)
            (2, 17), // Economic × RenovationUpgrade (road repair)
        ],
    };
    (challenge, meta)
}

fn inject_resource_discovery(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Add a resource bonus to one side of the map.
    let resource_idx = rng.range_u32(1, 5) as usize; // iron, wood, herbs, hide, crystal
    if let Some(s) = state.settlements.first_mut() {
        s.stockpile[resource_idx] += severity * 300.0;
    }

    let challenge = make_challenge(
        ChallengeCategory::Economic,
        "resource_discovery",
        severity,
        Some(random_direction(rng)),
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::ResourceDiscovery,
        activated_cells: vec![
            (2, 0), // Economic × Placement (processing building)
            (2, 2), // Economic × Routing (access path)
        ],
    };
    (challenge, meta)
}

// ---------------------------------------------------------------------------
// Population pressure injectors
// ---------------------------------------------------------------------------

fn inject_refugee_wave(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let refugee_count = rng.range_u32(10, (30.0 * severity) as u32 + 10);
    let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));

    for _ in 0..refugee_count {
        let eid = state.next_entity_id();
        let mut entity = Entity::new_npc(eid, settlement_pos);
        entity.level = 1;
        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
            npc.class_tags = vec!["refugee".to_string()];
        }
        state.entities.push(entity);
    }

    if let Some(s) = state.settlements.first_mut() {
        s.population += refugee_count;
        s.construction_memory.short_term.push(ConstructionEvent {
            tick: state.tick,
            kind: ConstructionEventKind::PopulationOverflow,
            severity,
            location: (64, 64),
            source_entity: None,
        });
    }

    let challenge = make_challenge(
        ChallengeCategory::Population,
        "refugee_wave",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::RefugeeWave,
        activated_cells: vec![
            (3, 0),  // Population × Placement (housing)
            (3, 1),  // Population × Prioritization
            (3, 16), // Population × ExpansionProvision
        ],
    };
    (challenge, meta)
}

fn inject_population_decline(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Kill off some NPCs.
    let kill_frac = severity * 0.3;
    for entity in state.entities.iter_mut() {
        if entity.kind == EntityKind::Npc && entity.alive {
            if rng.next_f32() < kill_frac {
                entity.alive = false;
            }
        }
    }

    if let Some(s) = state.settlements.first_mut() {
        s.population = (s.population as f32 * (1.0 - kill_frac)) as u32;
    }

    let challenge = make_challenge(
        ChallengeCategory::Population,
        "population_decline",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::PopulationDecline,
        activated_cells: vec![
            (3, 4), // Population × Demolition (consolidate)
            (3, 1), // Population × Prioritization
        ],
    };
    (challenge, meta)
}

fn inject_class_tension(
    _state: &mut WorldState,
    severity: f32,
    _rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // No physical world modification — the challenge metadata is enough
    // for the oracle to reason about zone separation.
    let challenge = make_challenge(
        ChallengeCategory::Population,
        "class_tension",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::ClassTension,
        activated_cells: vec![
            (3, 3),  // Population × ZoneComposition (separation)
            (3, 11), // Population × InteriorFlow
        ],
    };
    (challenge, meta)
}

fn inject_specialist_arrival(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));

    let specialist_roles = ["master_smith", "archmage", "sage", "court_wizard"];
    let role = specialist_roles[rng.range_u32(0, specialist_roles.len() as u32 - 1) as usize];
    let level = rng.range_u32(5, 8);

    let eid = state.next_entity_id();
    let mut entity = Entity::new_npc(eid, settlement_pos);
    entity.level = level;
    if let Some(npc) = entity.npc.as_mut() {
        npc.home_settlement_id = Some(settlement_id);
        npc.archetype = role.to_string();
    }
    state.entities.push(entity);

    let challenge = make_challenge(
        ChallengeCategory::Population,
        "specialist_arrival",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::SpecialistArrival,
        activated_cells: vec![
            (3, 0),  // Population × Placement (specialist housing)
            (3, 12), // Population × RoomSpecialization
        ],
    };
    (challenge, meta)
}

// ---------------------------------------------------------------------------
// Temporal pressure injectors
// ---------------------------------------------------------------------------

fn inject_winter_deadline(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    let ticks_remaining = rng.range_u32(100, (500.0 * (1.0 - severity * 0.5)) as u32);
    let deadline = state.tick + ticks_remaining as u64;

    let challenge = make_challenge(
        ChallengeCategory::Temporal,
        "winter_deadline",
        severity,
        None,
        Some(deadline),
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::WinterDeadline,
        activated_cells: vec![
            (4, 1),  // Temporal × Prioritization
            (4, 15), // Temporal × EnvironmentalAdaptation (insulation)
        ],
    };
    (challenge, meta)
}

fn inject_harvest_surge(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Lots of food coming, need storage.
    if let Some(s) = state.settlements.first_mut() {
        s.stockpile[0] += severity * 500.0; // food
    }

    let deadline = state.tick + rng.range_u32(200, 600) as u64;

    let challenge = make_challenge(
        ChallengeCategory::Temporal,
        "harvest_surge",
        severity,
        None,
        Some(deadline),
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::HarvestSurge,
        activated_cells: vec![
            (4, 0), // Temporal × Placement (warehouse/storage)
            (4, 1), // Temporal × Prioritization
        ],
    };
    (challenge, meta)
}

fn inject_building_decay(
    state: &mut WorldState,
    severity: f32,
    rng: &mut SimpleRng,
) -> (Challenge, PressureMetadata) {
    // Decay existing buildings.
    for entity in state.entities.iter_mut() {
        if entity.kind == EntityKind::Building && entity.alive {
            if rng.next_f32() < severity * 0.5 {
                entity.hp *= rng.range_f32(0.3, 0.7);
                if let Some(bd) = entity.building.as_mut() {
                    bd.construction_progress *= rng.range_f32(0.4, 0.8);
                }
            }
        }
    }

    let challenge = make_challenge(
        ChallengeCategory::Temporal,
        "building_decay",
        severity,
        None,
        None,
        Vec::new(),
    );
    let meta = PressureMetadata {
        pressure_type: PressureType::BuildingDecay,
        activated_cells: vec![
            (4, 4),  // Temporal × Demolition
            (4, 13), // Temporal × MaterialSelection (upgrade)
            (4, 17), // Temporal × RenovationUpgrade
        ],
    };
    (challenge, meta)
}

// ===========================================================================
// Composition Engine
// ===========================================================================

/// Compose a WorldState from the 5 axes.
pub fn compose_world_state(
    terrain: TerrainType,
    maturity: MaturityLevel,
    resources: ResourceProfile,
    npcs: NpcComposition,
    quality: BuildingQuality,
    seed: u64,
) -> WorldState {
    let mut rng = SimpleRng::new(seed);
    let mut state = WorldState::new(seed);

    let settlement_id: u32 = 1;
    let settlement_pos = (0.0_f32, 0.0_f32);

    // Create settlement.
    let settlement = SettlementState::new(settlement_id, "Generated Settlement".into(), settlement_pos);
    state.settlements.push(settlement);

    // Create city grid with generic terrain (we'll overwrite it).
    let terrain_str = match terrain {
        TerrainType::Hillside | TerrainType::MountainPass => "Mountains",
        TerrainType::Coastal => "Coast",
        TerrainType::Swamp => "Swamp",
        TerrainType::ForestClearing => "Forest",
        _ => "Plains",
    };
    let grid = CityGrid::new(128, 128, settlement_id, terrain_str, seed);
    let influence = InfluenceMap::new(128, 128);
    state.settlements[0].city_grid_idx = Some(state.city_grids.len());
    state.city_grids.push(grid);
    state.influence_maps.push(influence);

    // Layer 1: Apply the 5 axes.
    let gi = state.settlements[0].city_grid_idx.unwrap();
    let resource_mods = generate_terrain(terrain, &mut state.city_grids[gi], &mut rng);
    generate_maturity(maturity, &mut state, &mut rng);
    apply_resource_profile(resources, &mut state, &resource_mods, &mut rng);
    generate_npc_roster(npcs, &mut state, &mut rng);
    apply_building_quality(quality, &mut state, &mut rng);

    // Rebuild indices.
    state.rebuild_all_indices();

    state
}

/// Check compatibility between terrain, maturity, and a set of pressures.
pub fn check_compatibility(
    terrain: TerrainType,
    maturity: MaturityLevel,
    pressures: &[PressureType],
) -> bool {
    for &p in pressures {
        match p {
            // Flood needs water-adjacent terrain or flat terrain.
            PressureType::Flood => {
                if matches!(terrain, TerrainType::MountainPass | TerrainType::CliffEdge) {
                    return false;
                }
            }
            // Landslide needs elevation.
            PressureType::Landslide => {
                if matches!(
                    terrain,
                    TerrainType::FlatOpen | TerrainType::Swamp | TerrainType::Coastal
                ) {
                    return false;
                }
            }
            // Building decay needs existing buildings.
            PressureType::BuildingDecay => {
                if matches!(maturity, MaturityLevel::Empty) {
                    return false;
                }
            }
            // Supply disruption needs existing roads (i.e. built-up settlement).
            PressureType::SupplyDisruption => {
                if matches!(maturity, MaturityLevel::Empty | MaturityLevel::Sparse) {
                    return false;
                }
            }
            // Population decline needs population.
            PressureType::PopulationDecline => {
                if matches!(maturity, MaturityLevel::Empty) {
                    return false;
                }
            }
            // Demolition-oriented pressures need existing buildings.
            PressureType::Earthquake => {
                if matches!(maturity, MaturityLevel::Empty) {
                    return false;
                }
            }
            _ => {}
        }
    }

    // Check multi-pressure compatibility.
    if pressures.len() >= 2 {
        let categories: Vec<ChallengeCategory> =
            pressures.iter().map(|p| p.category()).collect();

        // Two military pressures in same direction = incompatible (handled by resampling direction).
        // Two environmental of same type = incompatible.
        for i in 0..pressures.len() {
            for j in (i + 1)..pressures.len() {
                if categories[i] == ChallengeCategory::Environmental
                    && categories[j] == ChallengeCategory::Environmental
                    && pressures[i] == pressures[j]
                {
                    return false;
                }
            }
        }
    }

    true
}

// ===========================================================================
// Coverage Tracker
// ===========================================================================

/// 10 challenge categories × 18 decision types coverage matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageTracker {
    /// matrix[category][decision_type] = count.
    pub matrix: [[u32; 18]; 10],
    pub target_per_cell: u32,
    pub dead_cells: HashSet<(usize, usize)>,
    pub total_pairs: u64,
    scenarios_run: u64,
}

impl CoverageTracker {
    pub fn new(target: u32) -> Self {
        Self {
            matrix: [[0; 18]; 10],
            target_per_cell: target,
            dead_cells: HashSet::new(),
            total_pairs: 0,
            scenarios_run: 0,
        }
    }

    /// Record actions into the coverage matrix.
    pub fn record_actions(&mut self, actions: &[BuildingAction], challenges: &[Challenge]) {
        for action in actions {
            let decision_col = action.decision_type as usize;
            // Map each action to all active challenge categories.
            if challenges.is_empty() {
                // No challenges = terrain-only decisions.
                self.matrix[5][decision_col] += 1; // Terrain category
                self.total_pairs += 1;
            } else {
                for ch in challenges {
                    let cat_row = ch.category as usize;
                    if cat_row < 10 && decision_col < 18 {
                        self.matrix[cat_row][decision_col] += 1;
                        self.total_pairs += 1;
                    }
                }
            }
        }
        self.scenarios_run += 1;
    }

    /// Detect dead cells: 0 hits after `threshold` scenarios.
    pub fn update_dead_cells(&mut self, threshold: u64) {
        if self.scenarios_run < threshold {
            return;
        }
        for row in 0..10 {
            for col in 0..18 {
                if self.matrix[row][col] == 0 {
                    self.dead_cells.insert((row, col));
                }
            }
        }
    }

    /// Compute sampling weights that bias toward under-represented cells.
    /// Returns weights for each of the 24 pressure types.
    pub fn pressure_weights(&self) -> [f32; 24] {
        let mut weights = [1.0_f32; 24];

        // For each pressure, compute how under-represented its activated cells are.
        // Lower coverage = higher weight.
        for (pi, &pressure) in ALL_PRESSURES.iter().enumerate() {
            let (_, meta) = make_dummy_metadata(pressure);
            let mut deficit = 0.0_f32;
            for &(row, col) in &meta.activated_cells {
                if row < 10 && col < 18 && !self.dead_cells.contains(&(row, col)) {
                    let current = self.matrix[row][col];
                    let gap = self.target_per_cell.saturating_sub(current) as f32;
                    deficit += gap / self.target_per_cell.max(1) as f32;
                }
            }
            weights[pi] = 1.0 + deficit;
        }

        weights
    }

    /// Compute axis weights. Returns (terrain[8], maturity[5], resource[5], npc[6], quality[5]).
    pub fn axis_weights(&self) -> ([f32; 8], [f32; 5], [f32; 5], [f32; 6], [f32; 5]) {
        // Uniform for now — could bias based on coverage patterns.
        ([1.0; 8], [1.0; 5], [1.0; 5], [1.0; 6], [1.0; 5])
    }

    /// Minimum count among active (non-dead) cells.
    pub fn min_active_cell(&self) -> u32 {
        let mut min = u32::MAX;
        for row in 0..10 {
            for col in 0..18 {
                if !self.dead_cells.contains(&(row, col)) {
                    min = min.min(self.matrix[row][col]);
                }
            }
        }
        if min == u32::MAX { 0 } else { min }
    }
}

/// Create a dummy metadata for coverage weight computation (no state modification).
fn make_dummy_metadata(pressure: PressureType) -> (ChallengeCategory, PressureMetadata) {
    let mut rng = SimpleRng::new(0);
    let mut dummy_state = WorldState::new(0);
    let settlement = SettlementState::new(1, "dummy".into(), (0.0, 0.0));
    dummy_state.settlements.push(settlement);
    let grid = CityGrid::new(4, 4, 1, "Plains", 0);
    let influence = InfluenceMap::new(4, 4);
    dummy_state.settlements[0].city_grid_idx = Some(0);
    dummy_state.city_grids.push(grid);
    dummy_state.influence_maps.push(influence);

    let (_, meta) = inject_pressure(pressure, &mut dummy_state, 0.5, &mut rng);
    (pressure.category(), meta)
}

// ===========================================================================
// Sampling weights for the composition engine
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingWeights {
    pub terrain: [f32; 8],
    pub maturity: [f32; 5],
    pub resources: [f32; 5],
    pub npcs: [f32; 6],
    pub quality: [f32; 5],
    pub pressures: [f32; 24],
}

impl Default for SamplingWeights {
    fn default() -> Self {
        Self {
            terrain: [1.0; 8],
            maturity: [1.0; 5],
            resources: [1.0; 5],
            npcs: [1.0; 6],
            quality: [1.0; 5],
            pressures: [1.0; 24],
        }
    }
}

// ===========================================================================
// Generated scenario output
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPair {
    pub obs: BuildingObservation,
    pub action: BuildingAction,
    pub meta: PairMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairMetadata {
    pub confidence: f32,
    pub utility_score: f32,
    pub category: String,
    pub decision_type: String,
    pub scenario_id: u64,
    pub pressures: Vec<String>,
}

// ===========================================================================
// Generation config
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenConfig {
    pub target_pairs: u64,
    pub min_cell: u32,
    pub seed: u64,
    pub output_path: String,
    pub coverage_path: Option<String>,
}

// ===========================================================================
// Coverage report output
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub total_scenarios: u64,
    pub total_pairs: u64,
    pub matrix: [[u32; 18]; 10],
    pub dead_cells: Vec<(usize, usize)>,
    pub min_active_cell: u32,
}

// ===========================================================================
// Core generation function
// ===========================================================================

/// Generate a single scenario: compose world state, inject pressures, run oracle.
fn generate_scenario(
    weights: &SamplingWeights,
    scenario_id: u64,
    rng: &mut SimpleRng,
) -> Vec<GeneratedPair> {
    // 1. Sample axes.
    let terrain_idx = rng.weighted_pick(&weights.terrain);
    let maturity_idx = rng.weighted_pick(&weights.maturity);
    let resource_idx = rng.weighted_pick(&weights.resources);
    let npc_idx = rng.weighted_pick(&weights.npcs);
    let quality_idx = rng.weighted_pick(&weights.quality);

    let terrain = ALL_TERRAIN[terrain_idx];
    let maturity = ALL_MATURITY[maturity_idx];
    let resources = ALL_RESOURCE[resource_idx];
    let npcs = ALL_NPC[npc_idx];
    let quality = ALL_QUALITY[quality_idx];

    // 2. Compose world state.
    let axis_seed = rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32);
    let mut state = compose_world_state(terrain, maturity, resources, npcs, quality, axis_seed);

    // 3. Sample 1-3 pressures.
    let pressure_count = rng.range_u32(1, 3) as usize;
    let mut pressures: Vec<PressureType> = Vec::new();
    let mut challenges: Vec<Challenge> = Vec::new();
    let mut all_metadata: Vec<PressureMetadata> = Vec::new();
    let mut attempts = 0;

    while pressures.len() < pressure_count && attempts < 50 {
        attempts += 1;
        let pidx = rng.weighted_pick(&weights.pressures);
        let candidate = ALL_PRESSURES[pidx];

        // Skip duplicates.
        if pressures.contains(&candidate) {
            continue;
        }

        // Check compatibility.
        let mut test_set = pressures.clone();
        test_set.push(candidate);
        if !check_compatibility(terrain, maturity, &test_set) {
            continue;
        }

        // Inject pressure.
        let severity = rng.range_f32(0.3, 1.0);
        let (challenge, meta) = inject_pressure(candidate, &mut state, severity, rng);
        pressures.push(candidate);
        challenges.push(challenge);
        all_metadata.push(meta);
    }

    // Rebuild indices after all modifications.
    state.rebuild_all_indices();

    let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);

    // 4. Populate memory.
    let memory = if let Some(first_challenge) = challenges.first() {
        populate_memory(&state, first_challenge, settlement_id)
    } else {
        ConstructionMemory::new()
    };

    // 5. Compute spatial features.
    let spatial = compute_spatial_features(&state, settlement_id);

    // 6. Build observations and run oracle.
    let strategic_obs = build_observation(
        &state,
        settlement_id,
        &challenges,
        &memory,
        &spatial,
        DecisionTier::Strategic,
    );
    let strategic_actions = strategic_oracle(&strategic_obs);

    let structural_obs = build_observation(
        &state,
        settlement_id,
        &challenges,
        &memory,
        &spatial,
        DecisionTier::Structural,
    );
    let structural_actions = structural_oracle(&structural_obs, &strategic_actions);

    // 7. Assemble (obs, action) pairs.
    let pressure_names: Vec<String> = pressures.iter().map(|p| p.name().to_string()).collect();
    let mut pairs = Vec::new();

    // Strategic pairs.
    for action in &strategic_actions {
        let category = if challenges.is_empty() {
            "terrain".to_string()
        } else {
            format!("{:?}", challenges[0].category)
        };
        pairs.push(GeneratedPair {
            obs: strategic_obs.clone(),
            action: action.clone(),
            meta: PairMetadata {
                confidence: action.priority.min(1.0),
                utility_score: action.priority,
                category,
                decision_type: format!("{:?}", action.decision_type),
                scenario_id,
                pressures: pressure_names.clone(),
            },
        });
    }

    // Structural pairs.
    for action in &structural_actions {
        let category = if challenges.is_empty() {
            "terrain".to_string()
        } else {
            format!("{:?}", challenges[0].category)
        };
        pairs.push(GeneratedPair {
            obs: structural_obs.clone(),
            action: action.clone(),
            meta: PairMetadata {
                confidence: action.priority.min(1.0),
                utility_score: action.priority,
                category,
                decision_type: format!("{:?}", action.decision_type),
                scenario_id,
                pressures: pressure_names.clone(),
            },
        });
    }

    pairs
}

/// Generate the full dataset. Returns coverage report.
pub fn generate_dataset(config: &GenConfig) -> CoverageReport {
    let mut rng = SimpleRng::new(config.seed);
    let mut tracker = CoverageTracker::new(config.min_cell);
    let mut weights = SamplingWeights::default();

    // Open output file.
    let output_path = Path::new(&config.output_path);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut file = std::io::BufWriter::new(
        std::fs::File::create(output_path).expect("Failed to create output file"),
    );

    let mut total_pairs: u64 = 0;
    let mut scenario_id: u64 = 0;
    let reweight_interval: u64 = 100;

    eprintln!("Generating building AI BC dataset...");
    eprintln!(
        "  Target pairs: {}, min cell: {}",
        config.target_pairs, config.min_cell
    );

    while total_pairs < config.target_pairs {
        scenario_id += 1;

        // Generate scenario.
        let pairs = generate_scenario(&weights, scenario_id, &mut rng);

        // Record coverage.
        let actions: Vec<BuildingAction> = pairs.iter().map(|p| p.action.clone()).collect();
        let challenges: Vec<Challenge> = if pairs.is_empty() {
            Vec::new()
        } else {
            pairs[0].obs.challenges.clone()
        };
        tracker.record_actions(&actions, &challenges);

        // Write pairs to JSONL.
        for pair in &pairs {
            let json = serde_json::to_string(pair).unwrap_or_default();
            writeln!(file, "{}", json).ok();
            total_pairs += 1;
        }

        // Reweight every N scenarios.
        if scenario_id % reweight_interval == 0 {
            tracker.update_dead_cells(500);
            weights.pressures = tracker.pressure_weights();
            let (tw, mw, rw, nw, qw) = tracker.axis_weights();
            weights.terrain = tw;
            weights.maturity = mw;
            weights.resources = rw;
            weights.npcs = nw;
            weights.quality = qw;

            if scenario_id % 500 == 0 {
                eprintln!(
                    "  [{} scenarios] {} pairs, min active cell: {}",
                    scenario_id,
                    total_pairs,
                    tracker.min_active_cell()
                );
            }
        }

        // Early exit if min_cell target reached.
        if tracker.min_active_cell() >= config.min_cell && total_pairs >= config.target_pairs / 2 {
            break;
        }
    }

    file.flush().ok();

    let min_active = tracker.min_active_cell();
    let report = CoverageReport {
        total_scenarios: scenario_id,
        total_pairs,
        matrix: tracker.matrix,
        dead_cells: tracker.dead_cells.into_iter().collect(),
        min_active_cell: min_active,
    };

    // Write coverage report if path specified.
    if let Some(ref cov_path) = config.coverage_path {
        let cov_path = Path::new(cov_path);
        if let Some(parent) = cov_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(&report) {
            std::fs::write(cov_path, json).ok();
        }
    }

    eprintln!(
        "Done. {} scenarios, {} pairs, min active cell: {}",
        report.total_scenarios, report.total_pairs, report.min_active_cell
    );

    report
}

/// Fill gaps in an existing dataset by targeting under-represented cells.
pub fn fill_gaps(
    existing_path: &Path,
    min_cell: u32,
    output_path: &Path,
    seed: u64,
) -> CoverageReport {
    // Load existing coverage from the dataset.
    let mut tracker = CoverageTracker::new(min_cell);

    // Scan existing file for coverage.
    if let Ok(contents) = std::fs::read_to_string(existing_path) {
        for line in contents.lines() {
            if let Ok(pair) = serde_json::from_str::<GeneratedPair>(line) {
                tracker.record_actions(&[pair.action], &pair.obs.challenges);
            }
        }
    }
    eprintln!(
        "Loaded existing coverage: {} pairs, min active cell: {}",
        tracker.total_pairs,
        tracker.min_active_cell()
    );

    tracker.update_dead_cells(500);

    // Generate supplemental data targeting gaps.
    let mut rng = SimpleRng::new(seed);
    let mut weights = SamplingWeights::default();
    weights.pressures = tracker.pressure_weights();

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut file = std::io::BufWriter::new(
        std::fs::File::create(output_path).expect("Failed to create output file"),
    );

    let mut scenario_id = 100000_u64; // Offset to avoid ID collisions.
    let max_scenarios = 50000_u64;

    while tracker.min_active_cell() < min_cell && scenario_id < 100000 + max_scenarios {
        scenario_id += 1;

        let pairs = generate_scenario(&weights, scenario_id, &mut rng);

        let actions: Vec<BuildingAction> = pairs.iter().map(|p| p.action.clone()).collect();
        let challenges: Vec<Challenge> = if pairs.is_empty() {
            Vec::new()
        } else {
            pairs[0].obs.challenges.clone()
        };
        tracker.record_actions(&actions, &challenges);

        for pair in &pairs {
            let json = serde_json::to_string(pair).unwrap_or_default();
            writeln!(file, "{}", json).ok();
        }

        if scenario_id % 100 == 0 {
            tracker.update_dead_cells(500);
            weights.pressures = tracker.pressure_weights();
        }
    }

    file.flush().ok();

    let min_active = tracker.min_active_cell();
    let report = CoverageReport {
        total_scenarios: scenario_id - 100000,
        total_pairs: tracker.total_pairs,
        matrix: tracker.matrix,
        dead_cells: tracker.dead_cells.into_iter().collect(),
        min_active_cell: min_active,
    };

    if let Ok(json) = serde_json::to_string_pretty(&report) {
        let cov_path = output_path.with_extension("coverage.json");
        std::fs::write(cov_path, json).ok();
    }

    eprintln!(
        "Fill-gaps done. {} supplemental scenarios, min active cell: {}",
        report.total_scenarios, report.min_active_cell
    );

    report
}

/// Compute coverage from an existing JSONL dataset file.
pub fn compute_coverage(dataset_path: &Path) -> CoverageReport {
    let mut tracker = CoverageTracker::new(100);

    if let Ok(contents) = std::fs::read_to_string(dataset_path) {
        for line in contents.lines() {
            if let Ok(pair) = serde_json::from_str::<GeneratedPair>(line) {
                tracker.record_actions(&[pair.action], &pair.obs.challenges);
            }
        }
    }

    tracker.update_dead_cells(500);

    let min_active = tracker.min_active_cell();
    CoverageReport {
        total_scenarios: tracker.scenarios_run,
        total_pairs: tracker.total_pairs,
        matrix: tracker.matrix,
        dead_cells: tracker.dead_cells.into_iter().collect(),
        min_active_cell: min_active,
    }
}
