//! Backend-agnostic visualizer types and playback controller.
//!
//! Defines the `TraceFrame` intermediate representation that renderers consume,
//! the `VisualizerBackend` trait for pluggable rendering, and the
//! `PlaybackController` that manages tick progression and frame generation.

use serde::{Deserialize, Serialize};

use super::state::{ChronicleEntry, ChronicleCategory, EntityKind, WorldState};
use super::systems::seasons::{current_season, Season, TICKS_PER_SEASON};
use super::trace::WorldSimTrace;

// ---------------------------------------------------------------------------
// View types — what renderers see
// ---------------------------------------------------------------------------

/// Settlement as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SettlementView {
    pub id: u32,
    pub name: String,
    pub pos: (f32, f32),
    pub faction_color: [u8; 3],
    pub population: u32,
    pub treasury: f32,
    pub specialty: String,
    pub threat_level: f32,
    /// Active service contracts at this settlement.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub contracts: Vec<ContractView>,
}

/// Service contract as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractView {
    pub service: String,
    pub requester: String,
    pub max_payment: f32,
    pub num_bids: usize,
    pub accepted: bool,
    pub provider: Option<String>,
    pub ticks_open: u64,
}

/// Entity as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityView {
    pub id: u32,
    pub pos: (f32, f32),
    pub kind: u8,
    pub team: u8,
    pub alive: bool,
    pub level: u32,
    pub name: Option<String>,
    /// For buildings: building type name (e.g. "Farm", "Mine"). None for non-buildings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub building_type: Option<String>,
    /// For resources: resource type name (e.g. "Tree", "Ore Vein"). None for non-resources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_type: Option<String>,
    /// For resources: fraction remaining (0.0–1.0). None for non-resources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_remaining: Option<f32>,
}

/// Inventory item as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InventoryItemView {
    pub name: String,
    pub slot: String,
    pub quality: f32,
    pub durability: f32,
}

/// Faction as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionView {
    pub id: u32,
    pub name: String,
    pub color: [u8; 3],
    pub territory_count: u32,
    pub military: f32,
    pub stance: String,
    pub treasury: f32,
}

/// Chronicle/event entry as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventView {
    pub tick: u64,
    pub category: String,
    pub text: String,
}

/// Region as seen by a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionView {
    pub id: u32,
    pub name: String,
    pub terrain: String,
    pub pos: (f32, f32),
    pub elevation: u8,
    pub is_floating: bool,
    pub faction_color: Option<[u8; 3]>,
    pub threat_level: f32,
    pub monster_density: f32,
    pub unrest: f32,
    pub control: f32,
}

/// Trade route between two settlements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeRouteView {
    pub from_pos: (f32, f32),
    pub to_pos: (f32, f32),
}

/// Detailed NPC state for a selected/tracked entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpcDetailView {
    pub entity_id: u32,
    pub name: String,
    pub level: u32,
    pub hp: f32,
    pub max_hp: f32,
    pub gold: f32,
    pub archetype: String,
    // Needs
    pub hunger: f32,
    pub shelter: f32,
    pub safety: f32,
    pub social: f32,
    pub purpose: f32,
    pub esteem: f32,
    // Emotions
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub pride: f32,
    pub anxiety: f32,
    // State
    pub morale: f32,
    pub stress: f32,
    pub economic_intent: String,
    pub work_state: String,
    // Goals
    pub goals: Vec<String>,
    // Classes
    pub classes: Vec<String>,
    // Top behavior tags
    pub top_tags: Vec<(String, f32)>,
    // Recent memory events
    pub recent_memories: Vec<String>,
    // Biography (full text)
    pub biography: String,
    // Position
    pub pos: (f32, f32),
    pub home_settlement: Option<String>,
    // Inventory
    pub equipped_items: Vec<InventoryItemView>,
    pub carried_gold: f32,
    pub inventory_commodities: Vec<(String, f32)>,
}

/// WFC building interior layout for 3D rendering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildingInteriorView {
    pub building_id: u32,
    pub world_x: f32,
    pub world_z: f32,
    pub interior_w: u8,
    pub interior_h: u8,
    pub num_floors: u8,
    /// Tile as u8 (see interior_gen::tiles::Tile repr), length = w * h * floors.
    pub tiles: Vec<u8>,
}

/// Aggregate stats for a single frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameSummary {
    pub alive_npcs: usize,
    pub alive_monsters: usize,
    pub total_population: u32,
    pub avg_threat: f32,
    pub season: String,
    pub year: u32,
}

/// Complete frame data for a renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceFrame {
    pub tick: u64,
    pub total_ticks: u64,
    pub settlements: Vec<SettlementView>,
    pub entities: Vec<EntityView>,
    pub factions: Vec<FactionView>,
    pub regions: Vec<RegionView>,
    pub trade_routes: Vec<TradeRouteView>,
    pub events: Vec<EventView>,
    pub summary: FrameSummary,
    /// Detailed state for a selected/tracked NPC (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_npc: Option<NpcDetailView>,
    /// WFC building interiors near the selected entity (if any).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub building_interiors: Vec<BuildingInteriorView>,
    /// Voxel surface data for 3D terrain rendering (legacy instanced mesh path).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub voxel_surfaces: Vec<VoxelSurfaceEntry>,
    /// Raw voxel chunk data for WebGPU SDF renderer.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub voxel_chunks: Vec<VoxelChunkView>,
    /// Number of loaded voxel chunks (for HUD display).
    #[serde(skip_serializing_if = "is_zero")]
    pub voxel_chunk_count: usize,
}

fn is_zero(v: &usize) -> bool { *v == 0 }

/// A chunk's material data for the WebGPU SDF renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelChunkView {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    pub materials: Vec<u8>, // 4096 bytes: material enum as u8 per voxel
}

/// A visible surface voxel for the 3D renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelSurfaceEntry {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub material: u8,       // VoxelMaterial as u8
    pub normal: [f32; 3],   // surface normal for lighting
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a faction ID to a display color. Six preset colors, gray fallback.
fn faction_color(faction_id: u32) -> [u8; 3] {
    match faction_id {
        0 => [66, 133, 244],   // blue
        1 => [219, 68, 55],    // red
        2 => [244, 180, 0],    // yellow
        3 => [15, 157, 88],    // green
        4 => [171, 71, 188],   // purple
        5 => [255, 112, 67],   // orange
        _ => [158, 158, 158],  // gray
    }
}

/// Format a `ChronicleCategory` as a human-readable string.
fn category_name(cat: ChronicleCategory) -> &'static str {
    match cat {
        ChronicleCategory::Battle => "Battle",
        ChronicleCategory::Quest => "Quest",
        ChronicleCategory::Diplomacy => "Diplomacy",
        ChronicleCategory::Economy => "Economy",
        ChronicleCategory::Death => "Death",
        ChronicleCategory::Discovery => "Discovery",
        ChronicleCategory::Crisis => "Crisis",
        ChronicleCategory::Achievement => "Achievement",
        ChronicleCategory::Narrative => "Narrative",
    }
}

/// Format a `Season` as a human-readable string.
fn season_name(season: Season) -> &'static str {
    match season {
        Season::Spring => "Spring",
        Season::Summer => "Summer",
        Season::Autumn => "Autumn",
        Season::Winter => "Winter",
    }
}

// ---------------------------------------------------------------------------
// generate_frame — pure projection from WorldState
// ---------------------------------------------------------------------------

/// Build a `TraceFrame` from the current world state and chronicle log.
///
/// `event_window` controls how many ticks back to include chronicle entries
/// (e.g. 200 means show events from `[tick - 200, tick]`).
pub fn generate_frame(
    state: &WorldState,
    chronicle: &[ChronicleEntry],
    event_window: u64,
    total_ticks: u64,
) -> TraceFrame {
    generate_frame_with_selection(state, chronicle, event_window, total_ticks, None)
}

/// Build a `TraceFrame` with optional NPC detail for a selected entity.
pub fn generate_frame_with_selection(
    state: &WorldState,
    chronicle: &[ChronicleEntry],
    event_window: u64,
    total_ticks: u64,
    selected_entity_id: Option<u32>,
) -> TraceFrame {
    let tick = state.tick;

    // --- Settlements ---
    let settlements: Vec<SettlementView> = state.settlements.iter().map(|s| {
        let color = s.faction_id
            .map(|fid| faction_color(fid))
            .unwrap_or([158, 158, 158]);
        let contracts: Vec<ContractView> = s.service_contracts.iter().map(|c| {
            let service = format!("{:?}", c.service);
            let requester = state.entity(c.requester_id)
                .map(|e| super::naming::entity_display_name(e))
                .unwrap_or_else(|| format!("#{}", c.requester_id));
            let provider = c.provider_id.and_then(|pid| {
                state.entity(pid).map(|e| super::naming::entity_display_name(e))
            });
            ContractView {
                service,
                requester,
                max_payment: c.max_payment.estimated_value(),
                num_bids: c.bids.len(),
                accepted: c.accepted_bid.is_some(),
                provider,
                ticks_open: tick.saturating_sub(c.posted_tick),
            }
        }).collect();

        SettlementView {
            id: s.id,
            name: s.name.clone(),
            pos: s.pos,
            faction_color: color,
            population: s.population,
            treasury: s.treasury,
            specialty: s.specialty.name().to_string(),
            threat_level: s.threat_level,
            contracts,
        }
    }).collect();

    // --- Entities ---
    let entities: Vec<EntityView> = state.entities.iter().map(|e| {
        let name = if e.alive {
            Some(super::naming::entity_display_name(e))
        } else {
            None
        };
        EntityView {
            id: e.id,
            pos: e.pos,
            kind: e.kind as u8,
            team: e.team as u8,
            alive: e.alive,
            level: e.level,
            name,
            building_type: e.building.as_ref().map(|b| format!("{:?}", b.building_type)),
            resource_type: e.resource.as_ref().map(|r| r.resource_type.display_name().to_string()),
            resource_remaining: e.resource.as_ref().map(|r| {
                if r.max_capacity > 0.0 { r.remaining / r.max_capacity } else { 0.0 }
            }),
        }
    }).collect();

    // --- Factions ---
    let factions: Vec<FactionView> = state.factions.iter().map(|f| {
        let stance = format!("{:?}", f.diplomatic_stance);
        FactionView {
            id: f.id,
            name: f.name.clone(),
            color: faction_color(f.id),
            territory_count: f.territory_size,
            military: f.military_strength,
            stance,
            treasury: f.treasury,
        }
    }).collect();

    // --- Events (filtered by window) ---
    let window_start = tick.saturating_sub(event_window);
    let events: Vec<EventView> = chronicle.iter()
        .filter(|e| e.tick >= window_start && e.tick <= tick)
        .map(|e| EventView {
            tick: e.tick,
            category: category_name(e.category).to_string(),
            text: e.text.clone(),
        })
        .collect();

    // --- Summary ---
    let alive_npcs = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count();
    let alive_monsters = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Monster)
        .count();
    let total_population: u32 = state.settlements.iter().map(|s| s.population).sum();

    let avg_threat = if state.regions.is_empty() {
        0.0
    } else {
        let sum: f32 = state.regions.iter().map(|r| r.threat_level).sum();
        sum / state.regions.len() as f32
    };

    let season = current_season(tick);
    let year = (tick / (TICKS_PER_SEASON * 4)) as u32 + 1;

    let summary = FrameSummary {
        alive_npcs,
        alive_monsters,
        total_population,
        avg_threat,
        season: season_name(season).to_string(),
        year,
    };

    // --- Regions ---
    // Regions don't store positions — derive from settlement positions.
    // Each settlement maps to its region (settlement_idx → region_idx in worldgen).
    let regions: Vec<RegionView> = state.regions.iter().map(|r| {
        RegionView {
            id: r.id,
            name: r.name.clone(),
            terrain: r.terrain.name().to_string(),
            pos: r.pos,
            elevation: r.elevation,
            is_floating: r.is_floating,
            faction_color: r.faction_id.map(|fid| faction_color(fid)),
            threat_level: r.threat_level,
            monster_density: r.monster_density,
            unrest: r.unrest,
            control: r.control,
        }
    }).collect();

    // --- Trade routes ---
    let trade_routes: Vec<TradeRouteView> = state.trade_routes.iter().filter_map(|route| {
        let sa = state.settlement(route.settlement_a)?;
        let sb = state.settlement(route.settlement_b)?;
        Some(TradeRouteView { from_pos: sa.pos, to_pos: sb.pos })
    }).collect();

    // --- Building interiors (near selected entity) ---
    let building_interiors: Vec<BuildingInteriorView> = selected_entity_id
        .and_then(|eid| {
            let entity = state.entity(eid)?;
            let settlement_id = entity.settlement_id()?;

            let building_range = state.group_index.settlement_buildings(settlement_id);
            let mut interiors = Vec::new();

            for idx in building_range {
                if idx >= state.entities.len() { continue; }
                let bld_entity = &state.entities[idx];
                let bld = match &bld_entity.building {
                    Some(b) => b,
                    None => continue,
                };

                let (fp_w, fp_h) = super::interior_gen::footprint_size(bld.building_type, bld.tier);

                // Use the building's own world position.
                let world_x = bld_entity.pos.0;
                let world_z = bld_entity.pos.1;

                // Limit to ~20 buildings near the selected entity
                if interiors.len() >= 20 { break; }

                if let Some(layout) = super::interior_gen::generate_interior(
                    bld.building_type,
                    bld.tier,
                    bld.room_seed,
                    fp_w,
                    fp_h,
                ) {
                    let mut tiles: Vec<u8> = Vec::with_capacity(
                        layout.width * layout.height * layout.floors.len(),
                    );
                    for floor in &layout.floors {
                        for tile in floor {
                            tiles.push(*tile as u8);
                        }
                    }
                    interiors.push(BuildingInteriorView {
                        building_id: bld_entity.id,
                        world_x,
                        world_z,
                        interior_w: layout.width as u8,
                        interior_h: layout.height as u8,
                        num_floors: layout.floors.len() as u8,
                        tiles,
                    });
                }
            }
            Some(interiors)
        })
        .unwrap_or_default();

    // --- Selected entity detail (NPCs and monsters) ---
    let selected_npc = selected_entity_id.and_then(|eid| {
        let entity = state.entity(eid)?;
        if !entity.alive { return None; }
        let display_name = super::naming::entity_display_name(entity);

        // NPC-specific fields (default for monsters)
        let npc = entity.npc.as_ref();

        let home_settlement = npc
            .and_then(|n| n.home_settlement_id)
            .and_then(|sid| state.settlement(sid))
            .map(|s| s.name.clone());

        let goals: Vec<String> = npc.map(|n| {
            n.goal_stack.goals.iter().map(|g| {
                format!("{:?} (prio {:.1}, progress {:.0}%)", g.kind, g.priority, g.progress * 100.0)
            }).collect()
        }).unwrap_or_default();

        let classes: Vec<String> = npc.map(|n| {
            n.classes.iter().map(|c| {
                let name = if c.display_name.is_empty() {
                    format!("Class({})", c.class_name_hash)
                } else {
                    c.display_name.clone()
                };
                format!("{} lv{}", name, c.level)
            }).collect()
        }).unwrap_or_default();

        let top_tags: Vec<(String, f32)> = npc.map(|n| {
            let mut tags: Vec<_> = n.behavior_profile.iter()
                .map(|&(hash, val)| {
                    let name = super::systems::biography::tag_display_name(hash);
                    (name.to_string(), val)
                })
                .collect();
            tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            tags.truncate(8);
            tags
        }).unwrap_or_default();

        let recent_memories: Vec<String> = npc.map(|n| {
            n.memory.events.iter().rev().take(10).map(|e| {
                let year = e.tick / 4800;
                let season = ["Spring", "Summer", "Autumn", "Winter"][(e.tick / 1200 % 4) as usize];
                format!("Y{} {} — {:?}", year, season, e.event_type)
            }).collect()
        }).unwrap_or_default();

        let biography = super::systems::biography::generate_biography(entity, state);

        let work_state = npc.map(|n| format!("{:?}", n.work_state)).unwrap_or_default();
        let economic_intent = npc.map(|n| format!("{:?}", n.economic_intent)).unwrap_or_default();
        let archetype = npc.map(|n| n.archetype.clone()).unwrap_or_else(|| {
            if entity.kind == super::state::EntityKind::Monster { "Monster".to_string() }
            else { "Unknown".to_string() }
        });

        // Equipped items
        let equipped_items: Vec<InventoryItemView> = npc.map(|n| {
            let mut items = Vec::new();
            for slot_id in [n.equipped_items.weapon_id, n.equipped_items.armor_id, n.equipped_items.accessory_id] {
                if let Some(iid) = slot_id {
                    if let Some(item_entity) = state.entity(iid) {
                        if let Some(item) = &item_entity.item {
                            items.push(InventoryItemView {
                                name: item_entity.npc.as_ref()
                                    .map(|n| n.name.clone())
                                    .unwrap_or_else(|| format!("Item #{}", iid)),
                                slot: format!("{:?}", item.slot),
                                quality: item.effective_quality(),
                                durability: item.durability,
                            });
                        }
                    }
                }
            }
            items
        }).unwrap_or_default();

        let carried_gold = npc.map(|n| n.gold).unwrap_or(0.0);

        // Inventory commodities
        let commodity_names = ["Food", "Iron", "Wood", "Equipment", "Medicine", "Leather", "Stone", "Cloth"];
        let inventory_commodities: Vec<(String, f32)> = entity.inventory.as_ref()
            .map(|inv| {
                inv.commodities.iter().enumerate()
                    .filter(|(_, &v)| v > 0.01)
                    .map(|(i, &v)| (commodity_names.get(i).unwrap_or(&"?").to_string(), v))
                    .collect()
            })
            .unwrap_or_default();

        Some(NpcDetailView {
            entity_id: eid,
            name: display_name,
            level: entity.level,
            hp: entity.hp,
            max_hp: entity.max_hp,
            gold: carried_gold,
            archetype,
            hunger: npc.map(|n| n.needs.hunger).unwrap_or(100.0),
            shelter: npc.map(|n| n.needs.shelter).unwrap_or(100.0),
            safety: npc.map(|n| n.needs.safety).unwrap_or(100.0),
            social: npc.map(|n| n.needs.social).unwrap_or(100.0),
            purpose: npc.map(|n| n.needs.purpose).unwrap_or(100.0),
            esteem: npc.map(|n| n.needs.esteem).unwrap_or(100.0),
            joy: npc.map(|n| n.emotions.joy).unwrap_or(0.0),
            sadness: npc.map(|n| n.emotions.grief).unwrap_or(0.0),
            anger: npc.map(|n| n.emotions.anger).unwrap_or(0.0),
            fear: npc.map(|n| n.emotions.fear).unwrap_or(0.0),
            pride: npc.map(|n| n.emotions.pride).unwrap_or(0.0),
            anxiety: npc.map(|n| n.emotions.anxiety).unwrap_or(0.0),
            morale: npc.map(|n| n.morale).unwrap_or(50.0),
            stress: npc.map(|n| n.stress).unwrap_or(0.0),
            economic_intent,
            work_state,
            goals,
            classes,
            top_tags,
            recent_memories,
            biography,
            pos: entity.pos,
            home_settlement,
            equipped_items,
            carried_gold,
            inventory_commodities,
        })
    });

    TraceFrame {
        tick,
        total_ticks,
        settlements,
        entities,
        factions,
        regions,
        trade_routes,
        events,
        summary,
        selected_npc,
        building_interiors,
        voxel_surfaces: extract_voxel_surfaces(state),
        voxel_chunks: extract_voxel_chunks(state),
        voxel_chunk_count: state.voxel_world.chunk_count(),
    }
}

/// Extract visible surface voxels from the voxel world for 3D rendering.
/// Only sends surface voxels (solid adjacent to air), capped for bandwidth.
fn extract_voxel_surfaces(state: &WorldState) -> Vec<VoxelSurfaceEntry> {
    use super::voxel::*;

    let vw = &state.voxel_world;
    if vw.chunks.is_empty() { return Vec::new(); }

    let mut surfaces = Vec::new();
    let max_surfaces = 8000; // bandwidth cap

    for chunk in vw.chunks.values() {
        if surfaces.len() >= max_surfaces { break; }

        let base_x = chunk.pos.x * CHUNK_SIZE as i32;
        let base_y = chunk.pos.y * CHUNK_SIZE as i32;
        let base_z = chunk.pos.z * CHUNK_SIZE as i32;

        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    if surfaces.len() >= max_surfaces { break; }

                    let voxel = chunk.get(lx, ly, lz);
                    if !voxel.material.is_solid() { continue; }

                    // Check if any face-neighbor is air (surface detection).
                    let gx = base_x + lx as i32;
                    let gy = base_y + ly as i32;
                    let gz = base_z + lz as i32;

                    let is_surface = [(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                        .iter()
                        .any(|&(dx, dy, dz)| {
                            !vw.get_voxel(gx + dx, gy + dy, gz + dz).material.is_solid()
                        });

                    if !is_surface { continue; }

                    // Compute normal from exposed faces.
                    let mut nx = 0.0f32;
                    let mut ny = 0.0f32;
                    let mut nz = 0.0f32;
                    for &(dx, dy, dz) in &[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                        if !vw.get_voxel(gx + dx, gy + dy, gz + dz).material.is_solid() {
                            nx += dx as f32;
                            ny += dy as f32;
                            nz += dz as f32;
                        }
                    }
                    let mag = (nx * nx + ny * ny + nz * nz).sqrt().max(0.001);

                    let (wx, wy, wz) = voxel_to_world(gx, gy, gz);
                    surfaces.push(VoxelSurfaceEntry {
                        x: wx,
                        y: wy,
                        z: wz,
                        material: voxel.material as u8,
                        normal: [nx / mag, ny / mag, nz / mag],
                    });
                }
            }
        }
    }

    surfaces
}

/// Extract raw chunk material data for the WebGPU SDF renderer.
/// Only sends chunks once (when first loaded) by checking dirty flag.
fn extract_voxel_chunks(state: &WorldState) -> Vec<VoxelChunkView> {
    let vw = &state.voxel_world;
    let mut chunks = Vec::new();
    let max_chunks = 125;

    // Also send volume metadata so client can size the volume correctly.
    for chunk in vw.chunks.values() {
        if chunks.len() >= max_chunks { break; }
        let materials: Vec<u8> = chunk.voxels.iter().map(|v| v.material as u8).collect();
        chunks.push(VoxelChunkView {
            cx: chunk.pos.x,
            cy: chunk.pos.y,
            cz: chunk.pos.z,
            materials,
        });
    }

    chunks
}

// ---------------------------------------------------------------------------
// PlaybackCommand + VisualizerBackend trait
// ---------------------------------------------------------------------------

/// Commands the playback controller can process from user input.
#[derive(Clone, Debug)]
pub enum PlaybackCommand {
    None,
    TogglePause,
    SeekForward(u64),
    SeekBackward(u64),
    SpeedUp,
    SlowDown,
    Quit,
}

/// Pluggable rendering backend. Implementations only consume `TraceFrame`
/// and never touch `WorldState` directly.
pub trait VisualizerBackend {
    /// Called once before the first frame with run metadata.
    fn init(&mut self, total_ticks: u64, seed: u64);
    /// Render a single frame.
    fn render_frame(&mut self, frame: &TraceFrame);
    /// Poll for user input and return a command.
    fn handle_input(&mut self) -> PlaybackCommand;
    /// Whether the backend is still running (false = exit).
    fn is_running(&self) -> bool;
    /// Called once when the playback loop ends.
    fn cleanup(&mut self);
}

// ---------------------------------------------------------------------------
// PlaybackController — drives trace playback
// ---------------------------------------------------------------------------

/// Manages playback state over a `WorldSimTrace`, producing `TraceFrame`s
/// on demand without the renderer needing to know about `WorldState`.
pub struct PlaybackController {
    trace: WorldSimTrace,
    current_tick: u64,
    speed: f32,
    paused: bool,
    cached_state: Option<WorldState>,
    event_window: u64,
}

impl PlaybackController {
    /// Create a new controller from a recorded trace.
    pub fn new(trace: WorldSimTrace, event_window: u64) -> Self {
        PlaybackController {
            current_tick: 0,
            speed: 1.0,
            paused: false,
            cached_state: None,
            event_window,
            trace,
        }
    }

    /// Advance playback by `dt_secs` of real time (scaled by speed).
    /// Does nothing while paused.
    pub fn advance(&mut self, dt_secs: f32) {
        if self.paused {
            return;
        }
        let ticks_to_advance = (dt_secs * self.speed * 10.0) as u64; // 10 ticks/sec base
        let new_tick = (self.current_tick + ticks_to_advance).min(self.trace.total_ticks);
        if new_tick != self.current_tick {
            self.current_tick = new_tick;
            self.cached_state = None; // invalidate cache
        }
    }

    /// Jump to a specific tick (clamped to valid range).
    pub fn seek_to(&mut self, tick: u64) {
        let clamped = tick.min(self.trace.total_ticks);
        if clamped != self.current_tick {
            self.current_tick = clamped;
            self.cached_state = None;
        }
    }

    /// Process a `PlaybackCommand`.
    pub fn handle_command(&mut self, cmd: PlaybackCommand) {
        match cmd {
            PlaybackCommand::None => {}
            PlaybackCommand::TogglePause => {
                self.paused = !self.paused;
            }
            PlaybackCommand::SeekForward(ticks) => {
                self.seek_to(self.current_tick.saturating_add(ticks));
            }
            PlaybackCommand::SeekBackward(ticks) => {
                self.seek_to(self.current_tick.saturating_sub(ticks));
            }
            PlaybackCommand::SpeedUp => {
                self.speed = (self.speed * 2.0).min(1024.0);
            }
            PlaybackCommand::SlowDown => {
                self.speed = (self.speed / 2.0).max(0.125);
            }
            PlaybackCommand::Quit => {
                // Backends handle quit via is_running(); nothing to do here.
            }
        }
    }

    /// Generate the current frame, reconstructing state if needed.
    pub fn current_frame(&mut self) -> TraceFrame {
        // Reconstruct cached state if invalidated.
        if self.cached_state.is_none() {
            self.cached_state = self.trace.state_at_tick(self.current_tick);
        }

        if let Some(state) = &self.cached_state {
            generate_frame(
                state,
                &self.trace.chronicle_log,
                self.event_window,
                self.trace.total_ticks,
            )
        } else {
            // Fallback: empty frame (shouldn't happen with valid trace).
            TraceFrame {
                tick: self.current_tick,
                total_ticks: self.trace.total_ticks,
                settlements: vec![],
                entities: vec![],
                factions: vec![],
                regions: vec![],
                trade_routes: vec![],
                events: vec![],
                selected_npc: None,
                building_interiors: vec![],
                voxel_surfaces: vec![],
                voxel_chunks: vec![],
                voxel_chunk_count: 0,
                summary: FrameSummary {
                    alive_npcs: 0,
                    alive_monsters: 0,
                    total_population: 0,
                    avg_threat: 0.0,
                    season: "Unknown".to_string(),
                    year: 0,
                },
            }
        }
    }

    /// Current tick position.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Current playback speed multiplier.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Whether playback is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Total ticks in the trace.
    pub fn total_ticks(&self) -> u64 {
        self.trace.total_ticks
    }

    /// The trace seed.
    pub fn seed(&self) -> u64 {
        self.trace.seed
    }
}
