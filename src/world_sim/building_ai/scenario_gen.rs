//! Scenario seed generation, challenge injection, memory population, and
//! observation assembly for the building intelligence dataset pipeline.
//!
//! Each function is deliberately self-contained — no global state, no side
//! effects beyond what the signature says. This lets the testing harness call
//! them in isolation or chain them via `run_scenario_pipeline`.

use std::path::Path;

use super::features::{compute_spatial_features, SpatialFeatures};
use super::scenario_config::*;
use super::types::*;
use crate::world_sim::state::{
    tag, BuildingData, BuildingType, Entity, EntityKind, SettlementState, WorldState,
};
use crate::world_sim::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple deterministic RNG closure from a WorldState.
fn make_rng(state: &mut WorldState) -> impl FnMut() -> f64 + '_ {
    move || {
        let v = state.next_rand_u32();
        (v as f64) / (u32::MAX as f64)
    }
}

/// Standalone RNG from a seed (doesn't need WorldState).
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) ^ self.0) as u32
    }
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }
}

fn parse_building_type(name: &str) -> BuildingType {
    match name.to_lowercase().as_str() {
        "house" => BuildingType::House,
        "longhouse" => BuildingType::Longhouse,
        "manor" => BuildingType::Manor,
        "farm" => BuildingType::Farm,
        "mine" => BuildingType::Mine,
        "sawmill" => BuildingType::Sawmill,
        "forge" => BuildingType::Forge,
        "workshop" => BuildingType::Workshop,
        "apothecary" => BuildingType::Apothecary,
        "market" => BuildingType::Market,
        "warehouse" => BuildingType::Warehouse,
        "inn" => BuildingType::Inn,
        "tradepost" | "trade_post" => BuildingType::TradePost,
        "guildhall" | "guild_hall" => BuildingType::GuildHall,
        "temple" => BuildingType::Temple,
        "barracks" => BuildingType::Barracks,
        "watchtower" => BuildingType::Watchtower,
        "library" => BuildingType::Library,
        "courthouse" | "court_house" => BuildingType::CourtHouse,
        "wall" => BuildingType::Wall,
        "gate" => BuildingType::Gate,
        "well" => BuildingType::Well,
        "tent" => BuildingType::Tent,
        "camp" => BuildingType::Camp,
        "shrine" => BuildingType::Shrine,
        "treasury" => BuildingType::Treasury,
        _ => BuildingType::House, // fallback
    }
}

/// Map a commodity name from TOML to an index into the stockpile array.
fn commodity_index(name: &str) -> Option<usize> {
    use crate::world_sim::commodity::*;
    match name.to_lowercase().as_str() {
        "food" => Some(FOOD),
        "iron" => Some(IRON),
        "wood" => Some(WOOD),
        "herbs" => Some(HERBS),
        "hide" => Some(HIDE),
        "crystal" => Some(CRYSTAL),
        "equipment" => Some(EQUIPMENT),
        "medicine" => Some(MEDICINE),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// generate_from_seed
// ---------------------------------------------------------------------------

/// Generate a WorldState from a seed configuration.
///
/// Creates a minimal world with one settlement, a city grid, placed buildings,
/// spawned NPCs, and set stockpile levels. Suitable for scenario evaluation
/// without running the full world-sim init pipeline.
pub fn generate_from_seed(seed: &SeedConfig, rng_seed: u64) -> WorldState {
    let effective_seed = seed.rng_seed.unwrap_or(rng_seed);
    let mut state = WorldState::new(effective_seed);
    state.skip_resource_init = true;

    let settlement_id: u32 = 1;
    let settlement_pos = (0.0_f32, 0.0_f32);

    // --- Settlement ---
    let mut settlement = SettlementState::new(settlement_id, "Scenario Settlement".into(), settlement_pos);
    let level = seed.settlement_level.unwrap_or(1);
    settlement.infrastructure_level = level as f32;

    // Age ticks (sets the world tick to represent settlement age).
    if let Some(ref age) = seed.age_ticks {
        let mut rng = SimpleRng::new(effective_seed.wrapping_add(1));
        let ticks = age.resolve(&mut || rng.next_f64()) as u64;
        state.tick = ticks;
    }

    state.settlements.push(settlement);

    // --- Buildings (from layout if present, else from buildings list) ---
    let _rng = SimpleRng::new(effective_seed.wrapping_add(2));
    let grid_center = 64_u16;

    if let Some(ref layout) = seed.layout {
        // Center the layout on the 128x128 grid.
        let (lw, lh) = layout.grid_size;
        let offset_col = 64_u16.saturating_sub(lw / 2);
        let offset_row = 64_u16.saturating_sub(lh / 2);

        // Layout-based building creation: positioned buildings with footprints.
        for (bi, lb) in layout.buildings.iter().enumerate() {
            let btype = parse_building_type(&lb.building_type);
            let tier = lb.tier.unwrap_or(1);
            let eid = state.next_entity_id();
            let (col, row) = (lb.cell.0 + offset_col, lb.cell.1 + offset_row);
            let (fw, fh) = lb.footprint;

            // Compute world position from grid cell (1:1 voxel mapping).
            let world_pos = (
                settlement_pos.0 + col as f32,
                settlement_pos.1 + row as f32,
            );
            let mut entity = Entity::new_building(eid, world_pos);
            entity.building = Some(BuildingData {
                building_type: btype,
                settlement_id: Some(settlement_id),
                grid_col: col,
                grid_row: row,
                footprint_w: fw as u8,
                footprint_h: fh as u8,
                tier,
                room_seed: eid as u64 ^ effective_seed,
                rooms: btype.default_rooms(),
                residential_capacity: btype.residential_capacity(),
                work_capacity: btype.work_capacity(),
                resident_ids: Vec::new(),
                worker_ids: Vec::new(),
                construction_progress: 1.0,
                built_tick: 0,
                builder_id: None,
                temporary: false,
                ttl_ticks: None,
                name: format!("{:?} #{}", btype, bi),
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
        }

        // Create Wall entities from wall circuits.
        for wc in &layout.wall_circuits {
            let pts = &wc.waypoints;
            for i in 0..pts.len() {
                let (x0, y0) = (pts[i].0 + offset_col, pts[i].1 + offset_row);
                let wall_eid = state.next_entity_id();
                let wall_world_pos = (settlement_pos.0 + x0 as f32, settlement_pos.1 + y0 as f32);
                let mut wall_entity = Entity::new_building(wall_eid, wall_world_pos);
                wall_entity.building = Some(BuildingData {
                    building_type: BuildingType::Wall,
                    settlement_id: Some(settlement_id),
                    grid_col: x0,
                    grid_row: y0,
                    footprint_w: 1,
                    footprint_h: 1,
                    tier: 1,
                    room_seed: wall_eid as u64,
                    rooms: Vec::new(),
                    residential_capacity: 0,
                    work_capacity: 0,
                    resident_ids: Vec::new(),
                    worker_ids: Vec::new(),
                    construction_progress: 1.0,
                    built_tick: 0,
                    builder_id: None,
                    temporary: false,
                    ttl_ticks: None,
                    name: format!("Wall segment #{}", i),
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
                state.entities.push(wall_entity);
            }
        }
    } else {
        // Legacy: building list without explicit layout.
        for (bi, bcfg) in seed.buildings.iter().enumerate() {
            let btype = parse_building_type(&bcfg.building_type);
            let tier = bcfg.tier.unwrap_or(1);
            let count = bcfg.count.unwrap_or(1);

            for ci in 0..count {
                let eid = state.next_entity_id();
                let (col, row) = if let Some((c, r)) = bcfg.grid_cell {
                    (c, r)
                } else {
                    let offset = (bi as u16 * count + ci).wrapping_mul(7);
                    let col = grid_center.wrapping_add(offset % 32).saturating_sub(16);
                    let row = grid_center.wrapping_add(offset / 32 * 3).saturating_sub(8);
                    (col.min(126), row.min(126))
                };

                let mut entity = Entity::new_building(eid, settlement_pos);
                entity.building = Some(BuildingData {
                    building_type: btype,
                    settlement_id: Some(settlement_id),
                    grid_col: col,
                    grid_row: row,
                    footprint_w: 2,
                    footprint_h: 2,
                    tier,
                    room_seed: eid as u64 ^ effective_seed,
                    rooms: btype.default_rooms(),
                    residential_capacity: btype.residential_capacity(),
                    work_capacity: btype.work_capacity(),
                    resident_ids: Vec::new(),
                    worker_ids: Vec::new(),
                    construction_progress: 1.0,
                    built_tick: 0,
                    builder_id: None,
                    temporary: false,
                    ttl_ticks: None,
                    name: format!("{:?} #{}", btype, ci),
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
            }
        }
    }

    // --- NPCs ---
    // Collect building world positions for NPC spawning near buildings.
    let building_positions: Vec<(f32, f32)> = state.entities.iter()
        .filter(|e| e.building.is_some())
        .map(|e| e.pos)
        .collect();

    for ncfg in &seed.npcs {
        let count = ncfg.count;
        for _ in 0..count {
            let eid = state.next_entity_id();
            let level = {
                let mut r = SimpleRng::new(effective_seed.wrapping_add(eid as u64));
                ncfg.level.resolve(&mut || r.next_f64()) as u32
            };
            // Spawn near a building (deterministic pick based on entity id).
            let spawn_pos = if !building_positions.is_empty() {
                let idx = eid as usize % building_positions.len();
                let bp = building_positions[idx];
                let jitter = ((eid as f32) * 0.7).sin() * 2.0;
                (bp.0 + jitter, bp.1 + jitter)
            } else {
                settlement_pos
            };
            let mut entity = Entity::new_npc(eid, spawn_pos);
            entity.level = level;
            if let Some(npc) = entity.npc.as_mut() {
                npc.name = crate::world_sim::naming::generate_personal_name(eid, effective_seed);
                npc.home_settlement_id = Some(settlement_id);
                npc.class_tags = vec![ncfg.class.clone()];
                if ncfg.is_garrison {
                    npc.archetype = "garrison".into();
                }
            }
            state.entities.push(entity);
        }
    }

    // --- High-value NPCs ---
    for hv in &seed.high_value_npcs {
        let eid = state.next_entity_id();
        let level = {
            let mut r = SimpleRng::new(effective_seed.wrapping_add(eid as u64));
            hv.level.resolve(&mut || r.next_f64()) as u32
        };
        let spawn_pos = if !building_positions.is_empty() {
            building_positions[eid as usize % building_positions.len()]
        } else {
            settlement_pos
        };
        let mut entity = Entity::new_npc(eid, spawn_pos);
        entity.level = level;
        if let Some(npc) = entity.npc.as_mut() {
            npc.name = crate::world_sim::naming::generate_personal_name(eid, effective_seed);
            npc.home_settlement_id = Some(settlement_id);
            npc.archetype = hv.role.clone();
            if let Some(ref cls) = hv.class {
                npc.class_tags = vec![cls.clone()];
            }
        }
        state.entities.push(entity);
    }

    // --- Assign NPCs to buildings (home + work) ---
    {
        let residential: Vec<(u32, u8)> = state.entities.iter()
            .filter(|e| e.building.as_ref().map_or(false, |b| b.residential_capacity > 0 && b.settlement_id == Some(settlement_id)))
            .map(|e| (e.id, e.building.as_ref().unwrap().residential_capacity))
            .collect();
        let workplaces: Vec<u32> = state.entities.iter()
            .filter(|e| e.building.as_ref().map_or(false, |b| b.work_capacity > 0 && b.settlement_id == Some(settlement_id)))
            .map(|e| e.id)
            .collect();

        // Round-robin home assignment by capacity.
        let mut home_slots: Vec<u32> = Vec::new();
        for (bid, cap) in &residential {
            for _ in 0..*cap {
                home_slots.push(*bid);
            }
        }
        let mut home_idx = 0;
        let mut work_idx = 0;
        for entity in state.entities.iter_mut() {
            if entity.kind != EntityKind::Npc || !entity.alive { continue; }
            if let Some(npc) = entity.npc.as_mut() {
                if npc.home_settlement_id != Some(settlement_id) { continue; }
                if !home_slots.is_empty() && npc.home_building_id.is_none() {
                    npc.home_building_id = Some(home_slots[home_idx % home_slots.len()]);
                    home_idx += 1;
                }
                if !workplaces.is_empty() && npc.work_building_id.is_none() {
                    npc.work_building_id = Some(workplaces[work_idx % workplaces.len()]);
                    work_idx += 1;
                }
            }
        }
    }

    // --- Stockpiles ---
    for (name, amount) in &seed.stockpiles {
        if let Some(idx) = commodity_index(name) {
            let mut r = SimpleRng::new(effective_seed.wrapping_add(idx as u64 + 100));
            let val = amount.resolve(&mut || r.next_f64()) as f32;
            state.settlements[0].stockpile[idx] = val;
        }
    }

    // --- Population count ---
    if let Some(ref pop) = seed.population {
        let mut r = SimpleRng::new(effective_seed.wrapping_add(200));
        state.settlements[0].population = pop.resolve(&mut || r.next_f64()) as u32;
    } else {
        // Default to NPC count.
        let npc_count = state
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive)
            .count();
        state.settlements[0].population = npc_count as u32;
    }

    // Rebuild indices so group_index is usable.
    state.rebuild_all_indices();

    state
}

/// Bresenham line rasterization for wall/road stamping.
fn bresenham_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        if x >= 0 && y >= 0 {
            out.push((x as usize, y as usize));
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// inject_challenge
// ---------------------------------------------------------------------------

/// Inject a challenge into the world state.
///
/// For military challenges: spawn enemy entities at the perimeter heading
/// toward the settlement from the threat direction.
/// For environmental: set up conditions (lower terrain for flood, wood clusters for fire).
/// For temporal: set the tick deadline on the state.
pub fn inject_challenge(state: &mut WorldState, challenge: &Challenge) {
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));

    match challenge.category {
        ChallengeCategory::Military | ChallengeCategory::UnitCapability => {
            inject_military(state, challenge, settlement_pos);
        }
        ChallengeCategory::Environmental => {
            inject_environmental(state, challenge, settlement_pos);
        }
        ChallengeCategory::Temporal => {
            if let Some(deadline) = challenge.deadline_tick {
                // Store deadline as a world event or via tick offset.
                // For scenario purposes we record it in the chronicle.
                state.chronicle.push(crate::world_sim::state::ChronicleEntry {
                    tick: state.tick,
                    category: crate::world_sim::state::ChronicleCategory::Narrative,
                    text: format!("Deadline challenge: {} ticks remaining.", deadline),
                    entity_ids: Vec::new(),
                });
            }
        }
        // Other categories are primarily observational — the challenge metadata
        // is enough for the oracle to reason about. We keep this arm extensible.
        _ => {}
    }
}

fn inject_military(state: &mut WorldState, challenge: &Challenge, settlement_pos: (f32, f32)) {
    let (dx, dy) = challenge.direction.unwrap_or((0.0, -1.0));
    let perimeter_dist = 80.0_f32; // world units from settlement center

    for profile in &challenge.enemy_profiles {
        let base_x = settlement_pos.0 + dx * perimeter_dist;
        let base_y = settlement_pos.1 + dy * perimeter_dist;

        for i in 0..profile.count {
            let eid = state.next_entity_id();
            let spread = (i as f32 - profile.count as f32 / 2.0) * 3.0;
            let spawn_pos = (base_x + dy.abs() * spread, base_y + dx.abs() * spread);

            let level = profile.level_range.0 as u32
                + (i as u32 % (profile.level_range.1.saturating_sub(profile.level_range.0) as u32 + 1));

            let mut entity = Entity::new_monster(eid, spawn_pos, level);
            entity.hp *= 1.0 + challenge.severity * 0.5;
            entity.max_hp = entity.hp;
            entity.attack_damage *= 1.0 + challenge.severity * 0.3;

            // Set move target toward settlement.
            entity.move_target = Some(settlement_pos);

            if let Some(npc) = entity.npc.as_mut() {
                npc.archetype = profile.type_name.clone();
            }

            state.entities.push(entity);
        }
    }

    // Update threat level on the settlement.
    if let Some(s) = state.settlements.first_mut() {
        s.threat_level = (s.threat_level + challenge.severity).min(1.0);
    }

    state.rebuild_all_indices();
}

fn inject_environmental(
    _state: &mut WorldState,
    _challenge: &Challenge,
    _settlement_pos: (f32, f32),
) {
    // VoxelWorld terrain injection not performed by scenario_gen.
    // Environmental challenges are represented through memory events and challenge structs.
}

// ---------------------------------------------------------------------------
// populate_memory
// ---------------------------------------------------------------------------

/// Populate construction memory buffers from settlement history and challenge context.
///
/// Generates synthetic construction events that match the challenge type and
/// settlement age. This provides the model with relevant history context.
pub fn populate_memory(
    state: &WorldState,
    challenge: &Challenge,
    settlement_id: u32,
) -> ConstructionMemory {
    let mut memory = ConstructionMemory::new();
    let tick = state.tick;

    // Settlement age determines memory richness.
    let age_ticks = tick;
    let is_old = age_ticks > 2000;

    match challenge.category {
        ChallengeCategory::Military | ChallengeCategory::UnitCapability => {
            populate_military_memory(&mut memory, challenge, tick, is_old);
        }
        ChallengeCategory::Environmental => {
            populate_environmental_memory(&mut memory, challenge, tick, is_old);
        }
        _ => {
            // Minimal baseline memory for non-combat scenarios.
            if is_old {
                memory.long_term.push(StructuralLesson {
                    lesson_tag: bi_tags::UPGRADE_PATH,
                    confidence: 0.5,
                    source_patterns: vec![ConstructionEventKind::ConstructionCompleted],
                    learned_tick: tick / 2,
                });
            }
        }
    }

    // All settlements get construction-completed events proportional to building count.
    let building_count = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Building
                && e.building
                    .as_ref()
                    .map(|b| b.settlement_id == Some(settlement_id))
                    .unwrap_or(false)
        })
        .count();

    for i in 0..building_count.min(memory.short_term.capacity) {
        memory.short_term.push(ConstructionEvent {
            tick: tick.saturating_sub((building_count - i) as u64 * 50),
            kind: ConstructionEventKind::ConstructionCompleted,
            severity: 0.0,
            location: (64, 64),
            source_entity: None,
        });
    }

    memory
}

fn populate_military_memory(
    memory: &mut ConstructionMemory,
    challenge: &Challenge,
    tick: u64,
    is_old: bool,
) {
    // Medium-term: breach/damage patterns from prior attacks.
    let has_jumpers = challenge
        .enemy_profiles
        .iter()
        .any(|p| p.can_jump && p.jump_height > 0);
    let has_siege = challenge.enemy_profiles.iter().any(|p| p.has_siege);

    if is_old {
        // Older settlements have experienced prior attacks.
        memory.medium_term.push(AggregatedPattern {
            kind: ConstructionEventKind::WallBreach,
            count: 3,
            mean_severity: 0.6,
            location_centroid: (64.0, 20.0),
            first_tick: tick.saturating_sub(1000),
            last_tick: tick.saturating_sub(200),
            importance: 0.7,
        });

        memory.medium_term.push(AggregatedPattern {
            kind: ConstructionEventKind::WallDamage,
            count: 8,
            mean_severity: 0.4,
            location_centroid: (64.0, 20.0),
            first_tick: tick.saturating_sub(1500),
            last_tick: tick.saturating_sub(100),
            importance: 0.6,
        });

        memory.medium_term.push(AggregatedPattern {
            kind: ConstructionEventKind::GarrisonEngaged,
            count: 5,
            mean_severity: 0.5,
            location_centroid: (64.0, 30.0),
            first_tick: tick.saturating_sub(800),
            last_tick: tick.saturating_sub(50),
            importance: 0.5,
        });
    }

    // Long-term structural lessons.
    if has_jumpers {
        memory.long_term.push(StructuralLesson {
            lesson_tag: bi_tags::JUMP_COUNTER,
            confidence: 0.8,
            source_patterns: vec![ConstructionEventKind::WallBreach],
            learned_tick: tick.saturating_sub(500),
        });
        memory.long_term.push(StructuralLesson {
            lesson_tag: bi_tags::WALL_TOO_LOW,
            confidence: 0.9,
            source_patterns: vec![
                ConstructionEventKind::WallBreach,
                ConstructionEventKind::EnemySighted,
            ],
            learned_tick: tick.saturating_sub(400),
        });
    }

    if has_siege {
        memory.long_term.push(StructuralLesson {
            lesson_tag: bi_tags::SIEGE_COUNTER,
            confidence: 0.7,
            source_patterns: vec![
                ConstructionEventKind::WallDamage,
                ConstructionEventKind::BuildingDamaged,
            ],
            learned_tick: tick.saturating_sub(600),
        });
        memory.long_term.push(StructuralLesson {
            lesson_tag: bi_tags::WALL_TOO_THIN,
            confidence: 0.85,
            source_patterns: vec![ConstructionEventKind::WallBreach],
            learned_tick: tick.saturating_sub(500),
        });
    }

    if is_old {
        memory.long_term.push(StructuralLesson {
            lesson_tag: bi_tags::GARRISON_SYNERGY,
            confidence: 0.6,
            source_patterns: vec![ConstructionEventKind::GarrisonEngaged],
            learned_tick: tick.saturating_sub(300),
        });
    }

    // Short-term: enemy sightings.
    for (i, profile) in challenge.enemy_profiles.iter().enumerate() {
        memory.short_term.push(ConstructionEvent {
            tick,
            kind: ConstructionEventKind::EnemySighted,
            severity: challenge.severity,
            location: (64, 10 + i as u16 * 5),
            source_entity: None,
        });
        let _ = profile; // profile info is captured via the lesson tags above
    }
}

fn populate_environmental_memory(
    memory: &mut ConstructionMemory,
    challenge: &Challenge,
    tick: u64,
    is_old: bool,
) {
    let sub = challenge.sub_type_name.as_str();
    match sub {
        "flood" | "river_flood" => {
            if is_old {
                memory.medium_term.push(AggregatedPattern {
                    kind: ConstructionEventKind::FloodDamage,
                    count: 4,
                    mean_severity: 0.5,
                    location_centroid: (64.0, 10.0),
                    first_tick: tick.saturating_sub(2000),
                    last_tick: tick.saturating_sub(300),
                    importance: 0.65,
                });
            }
            memory.long_term.push(StructuralLesson {
                lesson_tag: bi_tags::FLOOD_LOW_GROUND,
                confidence: if is_old { 0.9 } else { 0.4 },
                source_patterns: vec![ConstructionEventKind::FloodDamage],
                learned_tick: tick.saturating_sub(if is_old { 1000 } else { 0 }),
            });
            memory.long_term.push(StructuralLesson {
                lesson_tag: bi_tags::FLOOD_PREVENTION,
                confidence: if is_old { 0.8 } else { 0.3 },
                source_patterns: vec![ConstructionEventKind::FloodDamage],
                learned_tick: tick.saturating_sub(if is_old { 800 } else { 0 }),
            });
        }
        "fire" | "wildfire" => {
            if is_old {
                memory.medium_term.push(AggregatedPattern {
                    kind: ConstructionEventKind::FireSpread,
                    count: 6,
                    mean_severity: 0.7,
                    location_centroid: (30.0, 64.0),
                    first_tick: tick.saturating_sub(1500),
                    last_tick: tick.saturating_sub(200),
                    importance: 0.75,
                });
            }
            memory.long_term.push(StructuralLesson {
                lesson_tag: bi_tags::WOOD_BURNS,
                confidence: if is_old { 0.95 } else { 0.5 },
                source_patterns: vec![
                    ConstructionEventKind::FireSpread,
                    ConstructionEventKind::BuildingDestroyed,
                ],
                learned_tick: tick.saturating_sub(if is_old { 700 } else { 0 }),
            });
            memory.long_term.push(StructuralLesson {
                lesson_tag: bi_tags::FIRE_RECOVERY,
                confidence: if is_old { 0.7 } else { 0.2 },
                source_patterns: vec![ConstructionEventKind::FireSpread],
                learned_tick: tick.saturating_sub(if is_old { 500 } else { 0 }),
            });
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// build_observation
// ---------------------------------------------------------------------------

/// Build a complete observation for the oracle.
///
/// Extracts unit summaries and high-value NPC info from the world state,
/// combines with pre-computed spatial features and memory, and packages
/// everything into a `BuildingObservation`.
pub fn build_observation(
    state: &WorldState,
    settlement_id: u32,
    challenges: &[Challenge],
    memory: &ConstructionMemory,
    spatial: &SpatialFeatures,
    tier: DecisionTier,
) -> BuildingObservation {
    // --- Extract friendly roster ---
    let friendly_roster: Vec<UnitSummary> = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Npc
                && e.settlement_id() == Some(settlement_id)
        })
        .map(|e| {
            let is_garrison = e
                .npc
                .as_ref()
                .map(|n| n.archetype == "garrison")
                .unwrap_or(false);
            let class_tag = e
                .npc
                .as_ref()
                .and_then(|n| n.class_tags.first())
                .map(|s| tag(s.as_bytes()))
                .unwrap_or(0);
            let combat_effectiveness = e.attack_damage * (e.hp / e.max_hp.max(1.0));

            UnitSummary {
                entity_id: e.id,
                level: e.level.min(255) as u8,
                class_tag,
                combat_effectiveness,
                position: e.pos,
                is_garrison,
            }
        })
        .collect();

    // --- Extract high-value NPCs ---
    let high_value_npcs: Vec<HighValueNpc> = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Npc
                && e.settlement_id() == Some(settlement_id)
                && e.npc
                    .as_ref()
                    .map(|n| is_high_value_role(&n.archetype))
                    .unwrap_or(false)
        })
        .map(|e| {
            let npc = e.npc.as_ref().unwrap();
            HighValueNpc {
                entity_id: e.id,
                role_tag: tag(npc.archetype.as_bytes()),
                role_name: npc.archetype.clone(),
                level: e.level.min(255) as u8,
                protection_priority: protection_priority_for_role(&npc.archetype),
                position: e.pos,
            }
        })
        .collect();

    // --- Settlement metadata ---
    let settlement = state
        .settlements
        .iter()
        .find(|s| s.id == settlement_id);
    let settlement_level = settlement
        .map(|s| s.infrastructure_level as u8)
        .unwrap_or(1);
    let tech_tier = settlement
        .map(|s| (s.infrastructure_level / 2.0).ceil() as u8)
        .unwrap_or(1);

    BuildingObservation {
        settlement_id,
        tick: state.tick,
        challenges: challenges.to_vec(),
        memory: memory.clone(),
        spatial: spatial.clone(),
        friendly_roster,
        high_value_npcs,
        settlement_level,
        tech_tier,
        decision_tier: tier,
    }
}

fn is_high_value_role(archetype: &str) -> bool {
    matches!(
        archetype,
        "leader" | "master_smith" | "archmage" | "high_priest" | "commander"
            | "guild_master" | "sage" | "court_wizard"
    )
}

fn protection_priority_for_role(archetype: &str) -> f32 {
    match archetype {
        "leader" | "commander" => 0.9,
        "master_smith" | "archmage" | "court_wizard" => 0.7,
        "high_priest" | "sage" => 0.6,
        "guild_master" => 0.5,
        _ => 0.3,
    }
}

// ---------------------------------------------------------------------------
// run_scenario_pipeline
// ---------------------------------------------------------------------------

/// Load a ScenarioFile, resolve template references, generate the world state,
/// inject challenges, populate memory, compute spatial features, and build
/// observations for both Strategic and Structural tiers.
pub fn run_scenario_pipeline(
    scenario: &ScenarioFile,
    base_dir: &Path,
) -> Vec<BuildingObservation> {
    // 1. Resolve seed (load template if referenced).
    let seed = resolve_seed(&scenario.seed, base_dir);

    // 2. Generate world state.
    let mut state = generate_from_seed(&seed, 42);

    // 3. Convert and inject challenges.
    let mut challenges = Vec::new();
    for ccfg in &scenario.challenges {
        let challenge = resolve_challenge(ccfg, base_dir);
        inject_challenge(&mut state, &challenge);
        challenges.push(challenge);
    }

    let settlement_id = state
        .settlements
        .first()
        .map(|s| s.id)
        .unwrap_or(1);

    // 4. Populate memory.
    let memory = if let Some(first_challenge) = challenges.first() {
        populate_memory(&state, first_challenge, settlement_id)
    } else {
        ConstructionMemory::new()
    };

    // 5. Compute spatial features.
    let spatial = compute_spatial_features(&state, settlement_id);

    // 6. Build observations for both tiers.
    let mut observations = Vec::new();

    let strategic = build_observation(
        &state,
        settlement_id,
        &challenges,
        &memory,
        &spatial,
        DecisionTier::Strategic,
    );
    observations.push(strategic);

    let structural = build_observation(
        &state,
        settlement_id,
        &challenges,
        &memory,
        &spatial,
        DecisionTier::Structural,
    );
    observations.push(structural);

    observations
}

// ---------------------------------------------------------------------------
// Template resolution helpers
// ---------------------------------------------------------------------------

pub fn resolve_seed(seed: &SeedConfig, base_dir: &Path) -> SeedConfig {
    if let Some(ref template_path) = seed.template {
        let path = base_dir.join("seeds").join(template_path);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(mut template) = toml::from_str::<SeedConfig>(&content) {
                // Overlay inline overrides onto the template.
                if let Some(level) = seed.settlement_level {
                    template.settlement_level = Some(level);
                }
                if seed.population.is_some() {
                    template.population = seed.population.clone();
                }
                if seed.tech_tier.is_some() {
                    template.tech_tier = seed.tech_tier;
                }
                if seed.terrain.is_some() {
                    template.terrain = seed.terrain.clone();
                }
                if seed.age_ticks.is_some() {
                    template.age_ticks = seed.age_ticks.clone();
                }
                if !seed.stockpiles.is_empty() {
                    for (k, v) in &seed.stockpiles {
                        template.stockpiles.insert(k.clone(), v.clone());
                    }
                }
                if !seed.buildings.is_empty() {
                    template.buildings = seed.buildings.clone();
                }
                if !seed.npcs.is_empty() {
                    template.npcs = seed.npcs.clone();
                }
                if !seed.high_value_npcs.is_empty() {
                    template.high_value_npcs = seed.high_value_npcs.clone();
                }
                if seed.rng_seed.is_some() {
                    template.rng_seed = seed.rng_seed;
                }
                return template;
            }
        }
    }
    seed.clone()
}

pub fn resolve_challenge(ccfg: &ChallengeConfig, base_dir: &Path) -> Challenge {
    // If a template is referenced, load it and merge.
    let effective = if let Some(ref template_path) = ccfg.template {
        let path = base_dir.join("challenges").join(template_path);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(template) = toml::from_str::<ChallengeConfig>(&content) {
                merge_challenge_config(&template, ccfg)
            } else {
                ccfg.clone()
            }
        } else {
            ccfg.clone()
        }
    } else {
        ccfg.clone()
    };

    let category = parse_challenge_category(effective.category.as_deref().unwrap_or("military"));
    let sub_type_name = effective.sub_type.clone().unwrap_or_default();
    let sub_type = tag(sub_type_name.as_bytes());

    let severity = effective
        .severity
        .as_ref()
        .map(|s| {
            let mut rng = SimpleRng::new(sub_type as u64);
            s.resolve(&mut || rng.next_f64()) as f32
        })
        .unwrap_or(0.5);

    let direction = effective.direction.as_ref().map(|d| d.to_vector());
    let deadline_tick = effective.deadline_ticks;

    let enemy_profiles: Vec<EnemyProfile> = effective
        .enemies
        .iter()
        .map(|ecfg| resolve_enemy_profile(ecfg, base_dir))
        .collect();

    Challenge {
        category,
        sub_type,
        sub_type_name,
        severity,
        direction,
        deadline_tick,
        enemy_profiles,
    }
}

fn merge_challenge_config(template: &ChallengeConfig, overlay: &ChallengeConfig) -> ChallengeConfig {
    ChallengeConfig {
        template: None,
        category: overlay.category.clone().or_else(|| template.category.clone()),
        sub_type: overlay.sub_type.clone().or_else(|| template.sub_type.clone()),
        severity: overlay.severity.clone().or_else(|| template.severity.clone()),
        direction: overlay.direction.clone().or_else(|| template.direction.clone()),
        delay_ticks: overlay.delay_ticks.or(template.delay_ticks),
        deadline_ticks: overlay.deadline_ticks.or(template.deadline_ticks),
        enemies: if overlay.enemies.is_empty() {
            template.enemies.clone()
        } else {
            overlay.enemies.clone()
        },
        env_params: overlay.env_params.clone().or_else(|| template.env_params.clone()),
    }
}

fn parse_challenge_category(s: &str) -> ChallengeCategory {
    match s.to_lowercase().as_str() {
        "military" => ChallengeCategory::Military,
        "environmental" => ChallengeCategory::Environmental,
        "economic" => ChallengeCategory::Economic,
        "population" => ChallengeCategory::Population,
        "temporal" => ChallengeCategory::Temporal,
        "terrain" => ChallengeCategory::Terrain,
        "multi_settlement" | "multisettlement" => ChallengeCategory::MultiSettlement,
        "unit_capability" | "unitcapability" => ChallengeCategory::UnitCapability,
        "high_value_npc" | "highvaluenpc" => ChallengeCategory::HighValueNpc,
        "level_scaled" | "levelscaled" => ChallengeCategory::LevelScaled,
        _ => ChallengeCategory::Military,
    }
}

fn resolve_enemy_profile(ecfg: &EnemyConfig, base_dir: &Path) -> EnemyProfile {
    // Load from profile template if referenced.
    if let Some(ref profile_path) = ecfg.profile {
        let path = base_dir.join("enemy_profiles").join(profile_path);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(template) = toml::from_str::<EnemyConfig>(&content) {
                return enemy_config_to_profile(&merge_enemy_config(&template, ecfg));
            }
        }
    }
    enemy_config_to_profile(ecfg)
}

fn merge_enemy_config(template: &EnemyConfig, overlay: &EnemyConfig) -> EnemyConfig {
    EnemyConfig {
        profile: None,
        type_name: overlay.type_name.clone().or_else(|| template.type_name.clone()),
        level_range: overlay.level_range.or(template.level_range),
        count: overlay.count.clone().or_else(|| template.count.clone()),
        can_jump: overlay.can_jump.or(template.can_jump),
        jump_height: overlay.jump_height.or(template.jump_height),
        can_climb: overlay.can_climb.or(template.can_climb),
        can_tunnel: overlay.can_tunnel.or(template.can_tunnel),
        can_fly: overlay.can_fly.or(template.can_fly),
        has_siege: overlay.has_siege.or(template.has_siege),
        siege_damage: overlay.siege_damage.or(template.siege_damage),
    }
}

fn enemy_config_to_profile(ecfg: &EnemyConfig) -> EnemyProfile {
    let type_name = ecfg.type_name.clone().unwrap_or_else(|| "unknown".into());
    let type_tag = tag(type_name.as_bytes());
    let level_range = ecfg.level_range.unwrap_or([1, 3]);
    let count = ecfg
        .count
        .as_ref()
        .map(|c| {
            let mut rng = SimpleRng::new(type_tag as u64);
            c.resolve(&mut || rng.next_f64()) as u16
        })
        .unwrap_or(5);

    EnemyProfile {
        type_tag,
        type_name,
        level_range: (level_range[0], level_range[1]),
        count,
        can_jump: ecfg.can_jump.unwrap_or(false),
        jump_height: ecfg.jump_height.unwrap_or(0),
        can_climb: ecfg.can_climb.unwrap_or(false),
        can_tunnel: ecfg.can_tunnel.unwrap_or(false),
        can_fly: ecfg.can_fly.unwrap_or(false),
        has_siege: ecfg.has_siege.unwrap_or(false),
        siege_damage: ecfg.siege_damage.unwrap_or(0.0),
    }
}
