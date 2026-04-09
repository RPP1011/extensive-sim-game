//! VoxelBridge — syncs world sim VoxelWorld state to the voxel engine Scene.
//!
//! Pull-based: after each world sim tick, `sync_chunks` scans for dirty chunks
//! and pushes VoxelGrid copies to the Scene. Entity markers are synced separately.

use std::collections::HashMap;

use voxel_engine::scene::{Scene, SceneConfig, Transform};
use voxel_engine::scene::handle::{EntityHandle, ChunkHandle};
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::material::{MaterialPalette, PaletteEntry, MaterialType};

use super::voxel::{VoxelWorld, VoxelMaterial, ChunkPos, CHUNK_SIZE};
use super::state::{WorldState, EntityKind};

// ---------------------------------------------------------------------------
// Material palette — maps VoxelMaterial repr(u8) to engine PaletteEntry
// ---------------------------------------------------------------------------

fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();

    let set = |p: &mut MaterialPalette, mat: VoxelMaterial, r: u8, g: u8, b: u8, mt: MaterialType| {
        p.set(mat as u8, PaletteEntry {
            r, g, b,
            roughness: 200,
            emissive: 0,
            material_type: mt,
        });
    };

    // Natural terrain
    set(&mut p, VoxelMaterial::Dirt,       139, 90,  43,  MaterialType::Dirt);
    set(&mut p, VoxelMaterial::Stone,      136, 140, 141, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Granite,    100, 100, 105, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Sand,       210, 190, 140, MaterialType::Sand);
    set(&mut p, VoxelMaterial::Clay,       170, 130, 90,  MaterialType::Dirt);
    set(&mut p, VoxelMaterial::Gravel,     150, 150, 145, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Grass,      86,  152, 59,  MaterialType::Foliage);

    // Fluids
    set(&mut p, VoxelMaterial::Water,      64,  128, 200, MaterialType::Ice);
    set(&mut p, VoxelMaterial::Lava,       220, 80,  20,  MaterialType::Stone);
    set(&mut p, VoxelMaterial::Ice,        180, 220, 240, MaterialType::Ice);
    set(&mut p, VoxelMaterial::Snow,       240, 245, 250, MaterialType::Ice);

    // Ores
    set(&mut p, VoxelMaterial::IronOre,    140, 120, 110, MaterialType::Metal);
    set(&mut p, VoxelMaterial::CopperOre,  180, 120, 80,  MaterialType::Metal);
    set(&mut p, VoxelMaterial::GoldOre,    220, 190, 80,  MaterialType::Metal);
    set(&mut p, VoxelMaterial::Coal,       50,  50,  55,  MaterialType::Stone);
    set(&mut p, VoxelMaterial::Crystal,    180, 130, 220, MaterialType::Glass);

    // Placed by NPCs
    set(&mut p, VoxelMaterial::WoodLog,    130, 90,  50,  MaterialType::Wood);
    set(&mut p, VoxelMaterial::WoodPlanks, 190, 150, 90,  MaterialType::Wood);
    set(&mut p, VoxelMaterial::StoneBlock, 155, 155, 150, MaterialType::Stone);
    set(&mut p, VoxelMaterial::StoneBrick, 160, 160, 160, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Thatch,     200, 180, 100, MaterialType::Wood);
    set(&mut p, VoxelMaterial::Iron,       180, 180, 190, MaterialType::Metal);
    set(&mut p, VoxelMaterial::Glass,      200, 220, 230, MaterialType::Glass);

    // Agricultural
    set(&mut p, VoxelMaterial::Farmland,   120, 80,  40,  MaterialType::Dirt);
    set(&mut p, VoxelMaterial::Crop,       100, 170, 50,  MaterialType::Foliage);

    // New materials
    set(&mut p, VoxelMaterial::Basalt,     60,  60,  65,  MaterialType::Stone);
    set(&mut p, VoxelMaterial::Sandstone,  200, 170, 120, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Marble,     230, 225, 220, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Bone,       220, 210, 190, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Brick,      170, 90,  70,  MaterialType::Brick);
    set(&mut p, VoxelMaterial::CutStone,   170, 170, 165, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Concrete,   180, 180, 175, MaterialType::Concrete);
    set(&mut p, VoxelMaterial::Ceramic,    210, 190, 160, MaterialType::Stone);
    set(&mut p, VoxelMaterial::Steel,      190, 195, 200, MaterialType::Metal);
    set(&mut p, VoxelMaterial::Bronze,     180, 140, 80,  MaterialType::Metal);
    set(&mut p, VoxelMaterial::Obsidian,   30,  25,  35,  MaterialType::Glass);

    p
}

/// Pre-computed RGBA palette for GPU upload.
fn palette_rgba() -> [[u8; 4]; 256] {
    build_palette().to_rgba()
}

// ---------------------------------------------------------------------------
// Entity marker colors
// ---------------------------------------------------------------------------

/// Palette index for entity marker cubes. We use indices 200+ to avoid collision
/// with VoxelMaterial variants (which go up to ~33).
const MARKER_NPC_BLUE: u8 = 200;
const MARKER_MONSTER_RED: u8 = 201;
const MARKER_BUILDING_GRAY: u8 = 202;
const MARKER_RESOURCE_GREEN: u8 = 203;

fn build_marker_palette_entries(p: &mut MaterialPalette) {
    p.set(MARKER_NPC_BLUE, PaletteEntry {
        r: 66, g: 133, b: 244, roughness: 100, emissive: 40, material_type: MaterialType::Plastic,
    });
    p.set(MARKER_MONSTER_RED, PaletteEntry {
        r: 219, g: 68, b: 55, roughness: 100, emissive: 40, material_type: MaterialType::Plastic,
    });
    p.set(MARKER_BUILDING_GRAY, PaletteEntry {
        r: 158, g: 158, b: 158, roughness: 200, emissive: 0, material_type: MaterialType::Stone,
    });
    p.set(MARKER_RESOURCE_GREEN, PaletteEntry {
        r: 15, g: 157, b: 88, roughness: 150, emissive: 20, material_type: MaterialType::Foliage,
    });
}

fn marker_index_for_kind(kind: EntityKind) -> u8 {
    match kind {
        EntityKind::Npc => MARKER_NPC_BLUE,
        EntityKind::Monster => MARKER_MONSTER_RED,
        EntityKind::Building => MARKER_BUILDING_GRAY,
        EntityKind::Resource => MARKER_RESOURCE_GREEN,
        _ => MARKER_NPC_BLUE,
    }
}

// ---------------------------------------------------------------------------
// VoxelBridge
// ---------------------------------------------------------------------------

pub struct VoxelBridge {
    scene: Scene,
    palette: MaterialPalette,
    palette_rgba: [[u8; 4]; 256],
    chunk_handles: HashMap<ChunkPos, ChunkHandle>,
    entity_handles: HashMap<u32, EntityHandle>,
    /// Cached entity positions — skip set_transform when unchanged.
    entity_positions: HashMap<u32, (f32, f32, f32)>,
}

impl VoxelBridge {
    pub fn new() -> Self {
        let mut palette = build_palette();
        build_marker_palette_entries(&mut palette);
        let rgba = palette.to_rgba();

        Self {
            scene: Scene::new_headless(SceneConfig::default()),
            palette,
            palette_rgba: rgba,
            chunk_handles: HashMap::new(),
            entity_handles: HashMap::new(),
            entity_positions: HashMap::new(),
        }
    }

    /// Access the palette RGBA (for GPU upload).
    pub fn palette_rgba(&self) -> &[[u8; 4]; 256] {
        &self.palette_rgba
    }

    /// Access the palette.
    pub fn palette(&self) -> &MaterialPalette {
        &self.palette
    }

    /// Access the scene.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Mutable access to the scene.
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    // -----------------------------------------------------------------------
    // Chunk sync
    // -----------------------------------------------------------------------

    /// Convert a world sim Chunk to a voxel engine VoxelGrid.
    /// Swaps Y↔Z: world sim uses Z-up, voxel engine uses Y-up.
    pub fn chunk_to_grid(chunk: &super::voxel::Chunk) -> VoxelGrid {
        let mut grid = VoxelGrid::new(CHUNK_SIZE as u32, CHUNK_SIZE as u32, CHUNK_SIZE as u32);
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let voxel = chunk.get(x, y, z);
                    let mat_idx = voxel.material as u8;
                    if mat_idx != 0 {
                        // sim (x, y, z) → engine (x, z, y)
                        grid.set(x as u32, z as u32, y as u32, mat_idx);
                    }
                }
            }
        }
        grid
    }

    /// Bulk-load all chunks from VoxelWorld into the Scene.
    pub fn load_all_chunks(&mut self, voxel_world: &mut VoxelWorld) {
        let positions: Vec<ChunkPos> = voxel_world.chunks.keys().copied().collect();
        for cp in positions {
            if let Some(chunk) = voxel_world.chunks.get(&cp) {
                let grid = Self::chunk_to_grid(chunk);
                let pos = glam::IVec3::new(cp.x, cp.y, cp.z);
                let handle = self.scene.load_chunk(pos, &grid);
                self.chunk_handles.insert(cp, handle);
            }
        }
        // Clear dirty flags.
        for chunk in voxel_world.chunks.values_mut() {
            chunk.dirty = false;
        }
    }

    /// Scan for dirty chunks and push updates to the Scene.
    pub fn sync_chunks(&mut self, voxel_world: &mut VoxelWorld) {
        for chunk in voxel_world.chunks.values_mut() {
            if !chunk.dirty { continue; }

            let grid = Self::chunk_to_grid(chunk);
            let cp = chunk.pos;
            let pos = glam::IVec3::new(cp.x, cp.y, cp.z);

            // Unload old chunk if it exists, then reload.
            if let Some(old_handle) = self.chunk_handles.remove(&cp) {
                self.scene.unload_chunk(old_handle);
            }
            let handle = self.scene.load_chunk(pos, &grid);
            self.chunk_handles.insert(cp, handle);
            chunk.dirty = false;
        }
    }

    /// Number of chunks loaded in the Scene.
    pub fn chunk_count(&self) -> usize {
        self.chunk_handles.len()
    }

    // -----------------------------------------------------------------------
    // Entity sync
    // -----------------------------------------------------------------------

    /// Build a 2×2×2 marker grid filled with the given palette index.
    fn make_marker_grid(palette_idx: u8) -> VoxelGrid {
        let mut grid = VoxelGrid::new(2, 2, 2);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    grid.set(x, y, z, palette_idx);
                }
            }
        }
        grid
    }

    /// Sync all alive entities from world state into the scene.
    /// Spawns new markers, updates moved positions, despawns dead entities.
    pub fn sync_entities(&mut self, state: &WorldState) {
        // Track which entities we've seen this tick.
        let mut seen = std::collections::HashSet::new();

        for entity in &state.entities {
            if !entity.alive { continue; }
            seen.insert(entity.id);

            let z = state.voxel_world.surface_height(
                entity.pos.0 as i32,
                entity.pos.1 as i32,
            ) as f32;
            let pos = (entity.pos.0, entity.pos.1, z);

            // Check if position changed.
            if let Some(old_pos) = self.entity_positions.get(&entity.id) {
                if (old_pos.0 - pos.0).abs() < 0.01
                    && (old_pos.1 - pos.1).abs() < 0.01
                    && (old_pos.2 - pos.2).abs() < 0.01
                {
                    continue; // No movement.
                }
            }

            let transform = Transform {
                position: glam::Vec3::new(pos.0, pos.2, pos.1),
                ..Default::default()
            };

            if let Some(&handle) = self.entity_handles.get(&entity.id) {
                // Update position.
                self.scene.set_transform(handle, transform);
            } else {
                // Spawn new marker.
                let idx = marker_index_for_kind(entity.kind);
                let grid = Self::make_marker_grid(idx);
                let handle = self.scene.spawn(&grid, transform, &self.palette);
                self.entity_handles.insert(entity.id, handle);
            }
            self.entity_positions.insert(entity.id, pos);
        }

        // Despawn dead/removed entities.
        let dead_ids: Vec<u32> = self.entity_handles.keys()
            .filter(|id| !seen.contains(id))
            .copied()
            .collect();
        for id in dead_ids {
            if let Some(handle) = self.entity_handles.remove(&id) {
                self.scene.despawn(handle);
            }
            self.entity_positions.remove(&id);
        }
    }

    /// Full sync: chunks + entities.
    pub fn sync_all(&mut self, state: &mut WorldState) {
        self.sync_chunks(&mut state.voxel_world);
        self.sync_entities(state);
    }
}
