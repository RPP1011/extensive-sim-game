//! 3D voxel world with chunked storage.
//!
//! The voxel grid is the physical source of truth for the world. Each voxel
//! stores material, light level, damage state, and flags. Chunks are 16³
//! (4096 voxels, 16KB each) and loaded sparsely around active entities.
//!
//! Terrain generation fills chunks from region terrain type with layered
//! materials: bedrock → stone (with ore veins) → subsoil → surface.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096

/// World units per voxel (1 voxel = 1.0 world unit).
pub const VOXEL_SCALE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Coordinates
// ---------------------------------------------------------------------------

/// Chunk coordinate in chunk-space. Each chunk covers 16×16×16 voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self { Self { x, y, z } }

    /// Convert a voxel-space position to chunk position.
    pub fn from_voxel(vx: i32, vy: i32, vz: i32) -> Self {
        Self {
            x: vx.div_euclid(CHUNK_SIZE as i32),
            y: vy.div_euclid(CHUNK_SIZE as i32),
            z: vz.div_euclid(CHUNK_SIZE as i32),
        }
    }
}

/// Convert world-space (f32) to voxel-space (i32).
pub fn world_to_voxel(wx: f32, wy: f32, wz: f32) -> (i32, i32, i32) {
    (
        (wx / VOXEL_SCALE).floor() as i32,
        (wy / VOXEL_SCALE).floor() as i32,
        (wz / VOXEL_SCALE).floor() as i32,
    )
}

/// Convert voxel-space to world-space center.
pub fn voxel_to_world(vx: i32, vy: i32, vz: i32) -> (f32, f32, f32) {
    (
        vx as f32 * VOXEL_SCALE + VOXEL_SCALE * 0.5,
        vy as f32 * VOXEL_SCALE + VOXEL_SCALE * 0.5,
        vz as f32 * VOXEL_SCALE + VOXEL_SCALE * 0.5,
    )
}

/// Local index within a chunk from local coordinates (0..15 each).
#[inline]
pub fn local_index(lx: usize, ly: usize, lz: usize) -> usize {
    debug_assert!(lx < CHUNK_SIZE && ly < CHUNK_SIZE && lz < CHUNK_SIZE);
    (lz * CHUNK_SIZE + ly) * CHUNK_SIZE + lx
}

/// Decompose a voxel-space position into (ChunkPos, local_index).
pub fn voxel_to_chunk_local(vx: i32, vy: i32, vz: i32) -> (ChunkPos, usize) {
    let cp = ChunkPos::from_voxel(vx, vy, vz);
    let lx = vx.rem_euclid(CHUNK_SIZE as i32) as usize;
    let ly = vy.rem_euclid(CHUNK_SIZE as i32) as usize;
    let lz = vz.rem_euclid(CHUNK_SIZE as i32) as usize;
    (cp, local_index(lx, ly, lz))
}

// ---------------------------------------------------------------------------
// Voxel materials
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VoxelMaterial {
    Air = 0,
    // Natural terrain
    Dirt,
    Stone,
    Granite,    // bedrock, unmineable
    Sand,
    Clay,
    Gravel,
    Grass,      // surface layer on dirt
    // Fluids
    Water,
    Lava,
    Ice,
    Snow,
    // Ores
    IronOre,
    CopperOre,
    GoldOre,
    Coal,
    Crystal,
    // Placed by NPCs
    WoodLog,
    WoodPlanks,
    StoneBlock,
    StoneBrick,
    Thatch,
    Iron,
    Glass,
    // Agricultural
    Farmland,
    Crop,
    // New materials (appended to preserve repr(u8) ordering)
    Basalt,
    Sandstone,
    Marble,
    Bone,
    Brick,
    CutStone,
    Concrete,
    Ceramic,
    Steel,
    Bronze,
    Obsidian,
}

impl Default for VoxelMaterial {
    fn default() -> Self { VoxelMaterial::Air }
}

impl VoxelMaterial {
    /// Whether this material is solid (blocks movement and light).
    pub fn is_solid(self) -> bool {
        !matches!(self, VoxelMaterial::Air | VoxelMaterial::Water | VoxelMaterial::Lava)
    }

    /// Whether this material is a fluid.
    pub fn is_fluid(self) -> bool {
        matches!(self, VoxelMaterial::Water | VoxelMaterial::Lava)
    }

    /// Whether this material is transparent (light passes through).
    pub fn is_transparent(self) -> bool {
        matches!(self, VoxelMaterial::Air | VoxelMaterial::Water | VoxelMaterial::Glass | VoxelMaterial::Ice)
    }

    /// Mining hardness (ticks to break). 0 = instant, u32::MAX = unbreakable.
    pub fn hardness(self) -> u32 {
        match self {
            VoxelMaterial::Air | VoxelMaterial::Water | VoxelMaterial::Lava => 0,
            VoxelMaterial::Dirt | VoxelMaterial::Grass | VoxelMaterial::Sand
            | VoxelMaterial::Farmland | VoxelMaterial::Crop | VoxelMaterial::Snow => 5,
            VoxelMaterial::Clay | VoxelMaterial::Gravel | VoxelMaterial::Thatch => 8,
            VoxelMaterial::WoodLog | VoxelMaterial::WoodPlanks => 15,
            VoxelMaterial::Stone | VoxelMaterial::StoneBrick | VoxelMaterial::StoneBlock => 30,
            VoxelMaterial::Coal => 20,
            VoxelMaterial::IronOre | VoxelMaterial::CopperOre => 35,
            VoxelMaterial::GoldOre | VoxelMaterial::Crystal => 40,
            VoxelMaterial::Iron | VoxelMaterial::Glass | VoxelMaterial::Ice => 25,
            VoxelMaterial::Granite => u32::MAX, // bedrock
            // New materials
            VoxelMaterial::Bone => 10,
            VoxelMaterial::Sandstone => 20,
            VoxelMaterial::Brick | VoxelMaterial::Ceramic => 25,
            VoxelMaterial::Marble | VoxelMaterial::Basalt => 35,
            VoxelMaterial::Bronze => 40,
            VoxelMaterial::Steel => 50,
            VoxelMaterial::Obsidian => 60,
            VoxelMaterial::CutStone | VoxelMaterial::Concrete => 30,
        }
    }

    /// What commodity this material yields when mined (if any).
    pub fn mine_yield(self) -> Option<(usize, f32)> {
        use crate::world_sim::commodity;
        match self {
            VoxelMaterial::Stone | VoxelMaterial::StoneBlock | VoxelMaterial::StoneBrick => {
                Some((commodity::CRYSTAL, 0.5)) // stone → crystal/stone commodity
            }
            VoxelMaterial::IronOre => Some((commodity::IRON, 1.0)),
            VoxelMaterial::Coal => Some((commodity::IRON, 0.3)), // fuel
            VoxelMaterial::WoodLog => Some((commodity::WOOD, 1.0)),
            VoxelMaterial::WoodPlanks => Some((commodity::WOOD, 0.5)),
            VoxelMaterial::Dirt | VoxelMaterial::Grass => Some((commodity::FOOD, 0.1)), // earthworms
            VoxelMaterial::Crystal | VoxelMaterial::GoldOre => Some((commodity::CRYSTAL, 1.0)),
            VoxelMaterial::CopperOre => Some((commodity::IRON, 0.7)),
            _ => None,
        }
    }

    /// Physical and construction properties of this material.
    pub fn properties(self) -> MaterialProperties {
        match self {
            VoxelMaterial::Air => MaterialProperties { hp_multiplier: 0.0, fire_resistance: 0.0, load_bearing: false, weight: 0.0, rubble_move_cost: 0.0, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::Stone => MaterialProperties { hp_multiplier: 100.0, fire_resistance: 1.0, load_bearing: true, weight: 2.5, rubble_move_cost: 3.0, construction_cost: 5.0, blast_resistance: 0.6 },
            VoxelMaterial::Granite => MaterialProperties { hp_multiplier: 200.0, fire_resistance: 1.0, load_bearing: true, weight: 2.7, rubble_move_cost: 4.0, construction_cost: 10.0, blast_resistance: 0.9 },
            VoxelMaterial::Basalt => MaterialProperties { hp_multiplier: 150.0, fire_resistance: 1.0, load_bearing: true, weight: 2.8, rubble_move_cost: 3.5, construction_cost: 8.0, blast_resistance: 0.8 },
            VoxelMaterial::Sandstone => MaterialProperties { hp_multiplier: 60.0, fire_resistance: 0.9, load_bearing: true, weight: 2.2, rubble_move_cost: 2.0, construction_cost: 3.0, blast_resistance: 0.3 },
            VoxelMaterial::Marble => MaterialProperties { hp_multiplier: 80.0, fire_resistance: 1.0, load_bearing: true, weight: 2.6, rubble_move_cost: 3.0, construction_cost: 12.0, blast_resistance: 0.5 },
            VoxelMaterial::Dirt => MaterialProperties { hp_multiplier: 20.0, fire_resistance: 0.8, load_bearing: true, weight: 1.5, rubble_move_cost: 1.5, construction_cost: 1.0, blast_resistance: 0.1 },
            VoxelMaterial::Clay => MaterialProperties { hp_multiplier: 25.0, fire_resistance: 0.9, load_bearing: true, weight: 1.8, rubble_move_cost: 2.0, construction_cost: 2.0, blast_resistance: 0.15 },
            VoxelMaterial::Sand => MaterialProperties { hp_multiplier: 10.0, fire_resistance: 1.0, load_bearing: false, weight: 1.6, rubble_move_cost: 2.5, construction_cost: 1.0, blast_resistance: 0.05 },
            VoxelMaterial::Gravel => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 1.0, load_bearing: false, weight: 1.7, rubble_move_cost: 2.0, construction_cost: 1.0, blast_resistance: 0.1 },
            VoxelMaterial::Ice => MaterialProperties { hp_multiplier: 30.0, fire_resistance: 0.0, load_bearing: true, weight: 0.9, rubble_move_cost: 1.0, construction_cost: 0.0, blast_resistance: 0.2 },
            VoxelMaterial::Snow => MaterialProperties { hp_multiplier: 5.0, fire_resistance: 0.0, load_bearing: false, weight: 0.3, rubble_move_cost: 1.5, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::Grass => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 0.3, load_bearing: true, weight: 1.4, rubble_move_cost: 1.0, construction_cost: 0.0, blast_resistance: 0.05 },
            VoxelMaterial::Water => MaterialProperties { hp_multiplier: 0.0, fire_resistance: 1.0, load_bearing: false, weight: 1.0, rubble_move_cost: 0.0, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::Lava => MaterialProperties { hp_multiplier: 0.0, fire_resistance: 1.0, load_bearing: false, weight: 3.0, rubble_move_cost: 0.0, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::IronOre => MaterialProperties { hp_multiplier: 120.0, fire_resistance: 1.0, load_bearing: true, weight: 3.5, rubble_move_cost: 3.0, construction_cost: 0.0, blast_resistance: 0.7 },
            VoxelMaterial::CopperOre => MaterialProperties { hp_multiplier: 100.0, fire_resistance: 1.0, load_bearing: true, weight: 3.2, rubble_move_cost: 3.0, construction_cost: 0.0, blast_resistance: 0.6 },
            VoxelMaterial::GoldOre => MaterialProperties { hp_multiplier: 80.0, fire_resistance: 1.0, load_bearing: true, weight: 4.0, rubble_move_cost: 3.0, construction_cost: 0.0, blast_resistance: 0.5 },
            VoxelMaterial::Coal => MaterialProperties { hp_multiplier: 40.0, fire_resistance: 0.2, load_bearing: true, weight: 1.4, rubble_move_cost: 2.0, construction_cost: 0.0, blast_resistance: 0.2 },
            VoxelMaterial::Crystal => MaterialProperties { hp_multiplier: 50.0, fire_resistance: 0.8, load_bearing: false, weight: 2.3, rubble_move_cost: 2.5, construction_cost: 15.0, blast_resistance: 0.3 },
            VoxelMaterial::WoodLog => MaterialProperties { hp_multiplier: 40.0, fire_resistance: 0.2, load_bearing: true, weight: 0.7, rubble_move_cost: 2.0, construction_cost: 3.0, blast_resistance: 0.2 },
            VoxelMaterial::WoodPlanks => MaterialProperties { hp_multiplier: 35.0, fire_resistance: 0.2, load_bearing: true, weight: 0.5, rubble_move_cost: 1.5, construction_cost: 2.0, blast_resistance: 0.15 },
            VoxelMaterial::Thatch => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 0.1, load_bearing: false, weight: 0.2, rubble_move_cost: 1.0, construction_cost: 1.0, blast_resistance: 0.05 },
            VoxelMaterial::Bone => MaterialProperties { hp_multiplier: 30.0, fire_resistance: 0.5, load_bearing: true, weight: 1.0, rubble_move_cost: 2.0, construction_cost: 2.0, blast_resistance: 0.2 },
            VoxelMaterial::StoneBlock => MaterialProperties { hp_multiplier: 110.0, fire_resistance: 1.0, load_bearing: true, weight: 2.6, rubble_move_cost: 3.0, construction_cost: 7.0, blast_resistance: 0.7 },
            VoxelMaterial::StoneBrick => MaterialProperties { hp_multiplier: 120.0, fire_resistance: 1.0, load_bearing: true, weight: 2.5, rubble_move_cost: 3.0, construction_cost: 8.0, blast_resistance: 0.75 },
            VoxelMaterial::Brick => MaterialProperties { hp_multiplier: 90.0, fire_resistance: 1.0, load_bearing: true, weight: 2.0, rubble_move_cost: 2.5, construction_cost: 5.0, blast_resistance: 0.5 },
            VoxelMaterial::CutStone => MaterialProperties { hp_multiplier: 130.0, fire_resistance: 1.0, load_bearing: true, weight: 2.7, rubble_move_cost: 3.5, construction_cost: 10.0, blast_resistance: 0.8 },
            VoxelMaterial::Concrete => MaterialProperties { hp_multiplier: 140.0, fire_resistance: 1.0, load_bearing: true, weight: 2.4, rubble_move_cost: 4.0, construction_cost: 6.0, blast_resistance: 0.85 },
            VoxelMaterial::Glass => MaterialProperties { hp_multiplier: 10.0, fire_resistance: 0.7, load_bearing: false, weight: 2.5, rubble_move_cost: 3.0, construction_cost: 8.0, blast_resistance: 0.05 },
            VoxelMaterial::Ceramic => MaterialProperties { hp_multiplier: 50.0, fire_resistance: 1.0, load_bearing: false, weight: 2.0, rubble_move_cost: 2.5, construction_cost: 6.0, blast_resistance: 0.3 },
            VoxelMaterial::Iron => MaterialProperties { hp_multiplier: 150.0, fire_resistance: 0.8, load_bearing: true, weight: 7.8, rubble_move_cost: 4.0, construction_cost: 10.0, blast_resistance: 0.8 },
            VoxelMaterial::Steel => MaterialProperties { hp_multiplier: 200.0, fire_resistance: 0.9, load_bearing: true, weight: 7.9, rubble_move_cost: 4.5, construction_cost: 15.0, blast_resistance: 0.95 },
            VoxelMaterial::Bronze => MaterialProperties { hp_multiplier: 120.0, fire_resistance: 0.85, load_bearing: true, weight: 8.5, rubble_move_cost: 4.0, construction_cost: 12.0, blast_resistance: 0.7 },
            VoxelMaterial::Obsidian => MaterialProperties { hp_multiplier: 70.0, fire_resistance: 1.0, load_bearing: true, weight: 2.4, rubble_move_cost: 3.5, construction_cost: 20.0, blast_resistance: 0.4 },
            VoxelMaterial::Farmland => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 0.5, load_bearing: true, weight: 1.3, rubble_move_cost: 1.0, construction_cost: 1.0, blast_resistance: 0.05 },
            VoxelMaterial::Crop => MaterialProperties { hp_multiplier: 5.0, fire_resistance: 0.1, load_bearing: false, weight: 0.1, rubble_move_cost: 0.5, construction_cost: 0.0, blast_resistance: 0.0 },
        }
    }
}

// ---------------------------------------------------------------------------
// MaterialProperties
// ---------------------------------------------------------------------------

/// Physical and structural properties of a voxel material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialProperties {
    /// HP multiplier relative to base (0.0 = no HP, e.g. Air/fluids).
    pub hp_multiplier: f32,
    /// Fire resistance in [0, 1]: 0 = burns instantly, 1 = immune.
    pub fire_resistance: f32,
    /// Whether this material can bear structural load above it.
    pub load_bearing: bool,
    /// Mass per unit volume (tonnes/m³ equivalent).
    pub weight: f32,
    /// Movement cost multiplier when navigating through rubble.
    pub rubble_move_cost: f32,
    /// Resource cost to place one voxel of this material.
    pub construction_cost: f32,
    /// Fraction of explosion energy absorbed (0 = none, 1 = perfect).
    pub blast_resistance: f32,
}

// ---------------------------------------------------------------------------
// VoxelZone
// ---------------------------------------------------------------------------

/// Functional designation of a voxel — used for building zone tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum VoxelZone {
    #[default]
    None = 0,
    Residential = 1,
    Commercial = 2,
    Industrial = 3,
    Military = 4,
    Agricultural = 5,
    Sacred = 6,
    Underground = 7,
}

// ---------------------------------------------------------------------------
// Voxel
// ---------------------------------------------------------------------------

/// Per-voxel data. Stores material, light, damage, flags, and building metadata.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct Voxel {
    pub material: VoxelMaterial,
    /// Light level 0-15.
    pub light: u8,
    /// Mining damage accumulated (0-255). Breaks when >= hardness.
    pub damage: u8,
    /// Packed flags:
    /// - bits 0-3: water level (0-15, for fluid voxels)
    /// - bits 4-5: flow direction (0=none, 1=N, 2=E, 3=S... encoded)
    /// - bit 6: is_source (spring/sea boundary)
    /// - bit 7: is_support (load-bearing)
    pub flags: u8,
    /// Structural health in [0.0, 1.0]. 1.0 = intact, 0.0 = collapsed.
    pub integrity: f32,
    /// Building entity ID this voxel belongs to, if any.
    pub building_id: Option<u32>,
    /// Functional zone designation.
    pub zone: VoxelZone,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            material: VoxelMaterial::Air,
            light: 0,
            damage: 0,
            flags: 0,
            integrity: 1.0,
            building_id: None,
            zone: VoxelZone::None,
        }
    }
}

impl Voxel {
    pub fn new(material: VoxelMaterial) -> Self {
        Self { material, light: 0, damage: 0, flags: 0, integrity: 1.0, building_id: None, zone: VoxelZone::None }
    }

    pub fn water_level(self) -> u8 { self.flags & 0x0F }
    pub fn set_water_level(&mut self, level: u8) {
        self.flags = (self.flags & 0xF0) | (level & 0x0F);
    }
    pub fn is_source(self) -> bool { self.flags & 0x40 != 0 }
    pub fn set_source(&mut self, v: bool) {
        if v { self.flags |= 0x40; } else { self.flags &= !0x40; }
    }

    /// Effective HP: structural integrity scaled by material HP multiplier.
    pub fn effective_hp(&self) -> f32 {
        self.integrity * self.material.properties().hp_multiplier
    }
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A 16×16×16 block of voxels.
#[derive(Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub voxels: Vec<Voxel>, // CHUNK_VOLUME elements (use Vec for serde, but always len=4096)
    pub pos: ChunkPos,
    /// Set when any voxel changes — signals SDF/mesh regeneration needed.
    pub dirty: bool,
}

impl Chunk {
    pub fn new_air(pos: ChunkPos) -> Self {
        Self {
            voxels: vec![Voxel::default(); CHUNK_VOLUME],
            pos,
            dirty: true,
        }
    }

    pub fn new_filled(pos: ChunkPos, material: VoxelMaterial) -> Self {
        Self {
            voxels: vec![Voxel::new(material); CHUNK_VOLUME],
            pos,
            dirty: true,
        }
    }

    #[inline]
    pub fn get(&self, lx: usize, ly: usize, lz: usize) -> Voxel {
        self.voxels[local_index(lx, ly, lz)]
    }

    #[inline]
    pub fn set(&mut self, lx: usize, ly: usize, lz: usize, voxel: Voxel) {
        self.voxels[local_index(lx, ly, lz)] = voxel;
        self.dirty = true;
    }

    /// Count of non-air voxels.
    pub fn solid_count(&self) -> usize {
        self.voxels.iter().filter(|v| v.material.is_solid()).count()
    }

    /// True if the entire chunk is air (can be unloaded).
    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|v| v.material == VoxelMaterial::Air)
    }
}

impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Chunk({},{},{} solid={})", self.pos.x, self.pos.y, self.pos.z, self.solid_count())
    }
}

// ---------------------------------------------------------------------------
// VoxelWorld
// ---------------------------------------------------------------------------

/// Sparse chunk storage — the physical world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelWorld {
    pub chunks: HashMap<ChunkPos, Chunk>,
    /// Global water level (z coordinate). Sea/lake surfaces.
    pub sea_level: i32,
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self { chunks: HashMap::new(), sea_level: 28 }
    }
}

impl VoxelWorld {
    /// Get a voxel at world voxel coordinates. Returns Air for unloaded chunks.
    pub fn get_voxel(&self, vx: i32, vy: i32, vz: i32) -> Voxel {
        let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
        self.chunks.get(&cp)
            .map(|c| c.voxels[idx])
            .unwrap_or_default()
    }

    /// Set a voxel at world voxel coordinates. Creates chunk if needed.
    pub fn set_voxel(&mut self, vx: i32, vy: i32, vz: i32, voxel: Voxel) {
        let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
        let chunk = self.chunks.entry(cp).or_insert_with(|| Chunk::new_air(cp));
        chunk.voxels[idx] = voxel;
        chunk.dirty = true;
    }

    /// Get the surface height at (x, y) — highest solid voxel z.
    /// Scans downward from max loaded height. Returns sea_level if no solid found.
    pub fn surface_height(&self, vx: i32, vy: i32) -> i32 {
        // Scan from z=63 down to z=0 (4 chunks of height)
        for vz in (0..64).rev() {
            if self.get_voxel(vx, vy, vz).material.is_solid() {
                return vz + 1; // surface is one above the solid
            }
        }
        self.sea_level
    }

    /// Generate terrain for a chunk based on world position.
    /// Layered: bedrock (0-2), stone (2-20), subsoil (20-28), surface (28-30+).
    pub fn generate_chunk(&mut self, cp: ChunkPos, seed: u64) {
        if self.chunks.contains_key(&cp) { return; }

        let mut chunk = Chunk::new_air(cp);
        let base_x = cp.x * CHUNK_SIZE as i32;
        let base_y = cp.y * CHUNK_SIZE as i32;
        let base_z = cp.z * CHUNK_SIZE as i32;

        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let vx = base_x + lx as i32;
                    let vy = base_y + ly as i32;
                    let vz = base_z + lz as i32;

                    // Surface height varies by position (simple hash-based noise).
                    let surface = 30 + terrain_height_offset(vx, vy, seed);

                    let material = if vz < 0 {
                        VoxelMaterial::Granite // deep bedrock
                    } else if vz < 2 {
                        VoxelMaterial::Granite // bedrock layer
                    } else if vz < 15 {
                        // Deep stone with ore veins
                        ore_at(vx, vy, vz, seed).unwrap_or(VoxelMaterial::Stone)
                    } else if vz < 25 {
                        // Subsoil: mix of stone and dirt
                        if terrain_noise(vx, vy, vz, seed, 0x1234) > 0.4 {
                            VoxelMaterial::Stone
                        } else {
                            VoxelMaterial::Dirt
                        }
                    } else if vz < surface {
                        VoxelMaterial::Dirt
                    } else if vz == surface {
                        VoxelMaterial::Grass
                    } else if vz <= self.sea_level {
                        // Below sea level but above surface = water
                        VoxelMaterial::Water
                    } else {
                        VoxelMaterial::Air
                    };

                    if material != VoxelMaterial::Air {
                        chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(material);
                    }
                }
            }
        }

        chunk.dirty = true;
        self.chunks.insert(cp, chunk);
    }

    /// Ensure chunks are loaded around a world position (loading radius in chunks).
    pub fn ensure_loaded_around(&mut self, wx: f32, wy: f32, wz: f32, radius: i32, seed: u64) {
        let (vx, vy, vz) = world_to_voxel(wx, wy, wz);
        let center = ChunkPos::from_voxel(vx, vy, vz);
        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let cp = ChunkPos::new(center.x + dx, center.y + dy, center.z + dz);
                    self.generate_chunk(cp, seed);
                }
            }
        }
    }

    /// Apply mining damage to a voxel. Returns Some((material, yield)) if broken.
    pub fn mine_voxel(&mut self, vx: i32, vy: i32, vz: i32, damage: u8) -> Option<(VoxelMaterial, Option<(usize, f32)>)> {
        let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
        let chunk = self.chunks.get_mut(&cp)?;
        let voxel = &mut chunk.voxels[idx];

        if !voxel.material.is_solid() { return None; }
        let hardness = voxel.material.hardness();
        if hardness == u32::MAX { return None; } // unbreakable

        voxel.damage = voxel.damage.saturating_add(damage);
        if (voxel.damage as u32) >= hardness {
            let mat = voxel.material;
            let yield_info = mat.mine_yield();
            voxel.material = VoxelMaterial::Air;
            voxel.damage = 0;
            chunk.dirty = true;
            Some((mat, yield_info))
        } else {
            None // still mining
        }
    }

    /// Get the 6 face-adjacent neighbors of a voxel position.
    pub fn neighbors6(vx: i32, vy: i32, vz: i32) -> [(i32, i32, i32); 6] {
        [
            (vx - 1, vy, vz), (vx + 1, vy, vz),
            (vx, vy - 1, vz), (vx, vy + 1, vz),
            (vx, vy, vz - 1), (vx, vy, vz + 1),
        ]
    }

    /// Check if a voxel is at a chunk boundary (any local coord is 0 or 15).
    pub fn is_chunk_boundary(vx: i32, vy: i32, vz: i32) -> bool {
        let lx = vx.rem_euclid(CHUNK_SIZE as i32);
        let ly = vy.rem_euclid(CHUNK_SIZE as i32);
        let lz = vz.rem_euclid(CHUNK_SIZE as i32);
        lx == 0 || lx == 15 || ly == 0 || ly == 15 || lz == 0 || lz == 15
    }

    // -----------------------------------------------------------------------
    // Area operations
    // -----------------------------------------------------------------------

    /// Remove all solid voxels in an axis-aligned box. Returns list of (material, yield) for each broken voxel.
    pub fn remove_box(&mut self, min: (i32, i32, i32), max: (i32, i32, i32)) -> Vec<(VoxelMaterial, Option<(usize, f32)>)> {
        let mut results = Vec::new();
        for vz in min.2..=max.2 {
            for vy in min.1..=max.1 {
                for vx in min.0..=max.0 {
                    let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
                    if let Some(chunk) = self.chunks.get_mut(&cp) {
                        let voxel = &mut chunk.voxels[idx];
                        if voxel.material.is_solid() && voxel.material.hardness() < u32::MAX {
                            let mat = voxel.material;
                            let yield_info = mat.mine_yield();
                            results.push((mat, yield_info));
                            voxel.material = VoxelMaterial::Air;
                            voxel.damage = 0;
                            chunk.dirty = true;
                        }
                    }
                }
            }
        }
        results
    }

    /// Remove all solid voxels within a sphere. Returns list of (material, yield).
    pub fn remove_sphere(&mut self, center: (i32, i32, i32), radius: i32) -> Vec<(VoxelMaterial, Option<(usize, f32)>)> {
        let mut results = Vec::new();
        let r2 = (radius * radius) as i64;
        for vz in (center.2 - radius)..=(center.2 + radius) {
            for vy in (center.1 - radius)..=(center.1 + radius) {
                for vx in (center.0 - radius)..=(center.0 + radius) {
                    let dx = (vx - center.0) as i64;
                    let dy = (vy - center.1) as i64;
                    let dz = (vz - center.2) as i64;
                    if dx * dx + dy * dy + dz * dz > r2 { continue; }

                    let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
                    if let Some(chunk) = self.chunks.get_mut(&cp) {
                        let voxel = &mut chunk.voxels[idx];
                        if voxel.material.is_solid() && voxel.material.hardness() < u32::MAX {
                            let mat = voxel.material;
                            let yield_info = mat.mine_yield();
                            results.push((mat, yield_info));
                            voxel.material = VoxelMaterial::Air;
                            voxel.damage = 0;
                            chunk.dirty = true;
                        }
                    }
                }
            }
        }
        results
    }

    /// Fill an axis-aligned box with a material. Overwrites existing voxels.
    pub fn fill_box(&mut self, min: (i32, i32, i32), max: (i32, i32, i32), material: VoxelMaterial) {
        for vz in min.2..=max.2 {
            for vy in min.1..=max.1 {
                for vx in min.0..=max.0 {
                    self.set_voxel(vx, vy, vz, Voxel::new(material));
                }
            }
        }
    }

    /// Fill a sphere with a material.
    pub fn fill_sphere(&mut self, center: (i32, i32, i32), radius: i32, material: VoxelMaterial) {
        let r2 = (radius * radius) as i64;
        for vz in (center.2 - radius)..=(center.2 + radius) {
            for vy in (center.1 - radius)..=(center.1 + radius) {
                for vx in (center.0 - radius)..=(center.0 + radius) {
                    let dx = (vx - center.0) as i64;
                    let dy = (vy - center.1) as i64;
                    let dz = (vz - center.2) as i64;
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        self.set_voxel(vx, vy, vz, Voxel::new(material));
                    }
                }
            }
        }
    }

    /// Replace all voxels of one material with another in a box region.
    pub fn replace_in_box(&mut self, min: (i32, i32, i32), max: (i32, i32, i32),
                          from: VoxelMaterial, to: VoxelMaterial) -> u32 {
        let mut count = 0u32;
        for vz in min.2..=max.2 {
            for vy in min.1..=max.1 {
                for vx in min.0..=max.0 {
                    let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
                    if let Some(chunk) = self.chunks.get_mut(&cp) {
                        if chunk.voxels[idx].material == from {
                            chunk.voxels[idx].material = to;
                            chunk.dirty = true;
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Count loaded chunks.
    pub fn chunk_count(&self) -> usize { self.chunks.len() }

    /// Count total solid voxels across all loaded chunks.
    pub fn total_solid(&self) -> usize {
        self.chunks.values().map(|c| c.solid_count()).sum()
    }

    // -----------------------------------------------------------------------
    // Destructible terrain
    // -----------------------------------------------------------------------

    /// Apply structural damage to a voxel. Returns true if the voxel was destroyed.
    /// Triggers cascading collapse for unsupported voxels above.
    pub fn damage_voxel(&mut self, vx: i32, vy: i32, vz: i32, damage: f32) -> bool {
        let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
        let chunk = match self.chunks.get_mut(&cp) {
            Some(c) => c,
            None => return false,
        };
        let voxel = &mut chunk.voxels[idx];
        if !voxel.material.is_solid() { return false; }

        let props = voxel.material.properties();
        let effective_hp = voxel.integrity * props.hp_multiplier;
        let new_hp = effective_hp - damage;

        if new_hp <= 0.0 {
            if props.load_bearing {
                voxel.integrity = 0.0;
                voxel.zone = VoxelZone::None;
                voxel.building_id = None;
            } else {
                *voxel = Voxel::default();
            }
            chunk.dirty = true;
            self.cascade_collapse(vx, vy, vz + 1);
            true
        } else {
            voxel.integrity = new_hp / props.hp_multiplier;
            chunk.dirty = true;
            false
        }
    }

    /// Check structural support for voxel at (vx, vy, vz) and collapse if unsupported.
    fn cascade_collapse(&mut self, vx: i32, vy: i32, vz: i32) {
        let voxel = self.get_voxel(vx, vy, vz);
        if !voxel.material.is_solid() || voxel.integrity == 0.0 { return; }

        if self.is_supported(vx, vy, vz) { return; }

        let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
        if let Some(chunk) = self.chunks.get_mut(&cp) {
            let v = &mut chunk.voxels[idx];
            if v.material.properties().load_bearing {
                v.integrity = 0.0;
                v.zone = VoxelZone::None;
                v.building_id = None;
            } else {
                *v = Voxel::default();
            }
            chunk.dirty = true;
        }

        self.cascade_collapse(vx, vy, vz + 1);
    }

    /// Check if a voxel position is structurally supported.
    fn is_supported(&self, vx: i32, vy: i32, vz: i32) -> bool {
        if vz <= 0 { return true; }

        let below = self.get_voxel(vx, vy, vz - 1);
        if below.material.is_solid() && below.integrity > 0.0 && below.material.properties().load_bearing {
            return true;
        }

        let mut solid_neighbors = 0u8;
        for &(dx, dy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let n = self.get_voxel(vx + dx, vy + dy, vz);
            if n.material.is_solid() && n.integrity > 0.0 {
                solid_neighbors += 1;
            }
        }
        solid_neighbors >= 2
    }
}

// ---------------------------------------------------------------------------
// Terrain generation helpers
// ---------------------------------------------------------------------------

/// Smooth terrain height variation via bilinear interpolation of hash noise.
/// Grid spacing of 8 voxels ensures adjacent columns have similar heights.
fn terrain_height_offset(vx: i32, vy: i32, seed: u64) -> i32 {
    let scale = 8; // grid spacing — larger = smoother hills
    // Grid corners
    let gx0 = vx.div_euclid(scale) * scale;
    let gy0 = vy.div_euclid(scale) * scale;
    let gx1 = gx0 + scale;
    let gy1 = gy0 + scale;
    // Fractional position within grid cell [0, 1)
    let fx = (vx - gx0) as f32 / scale as f32;
    let fy = (vy - gy0) as f32 / scale as f32;
    // Smooth interpolation (smoothstep)
    let fx = fx * fx * (3.0 - 2.0 * fx);
    let fy = fy * fy * (3.0 - 2.0 * fy);
    // Hash at four corners → heights in [-5, 5]
    let h00 = (hash_3d(gx0, gy0, 0, seed) % 11) as f32 - 5.0;
    let h10 = (hash_3d(gx1, gy0, 0, seed) % 11) as f32 - 5.0;
    let h01 = (hash_3d(gx0, gy1, 0, seed) % 11) as f32 - 5.0;
    let h11 = (hash_3d(gx1, gy1, 0, seed) % 11) as f32 - 5.0;
    // Bilinear interpolation
    let h0 = h00 + (h10 - h00) * fx;
    let h1 = h01 + (h11 - h01) * fx;
    let h = h0 + (h1 - h0) * fy;
    h.round() as i32
}

/// Determine if an ore vein exists at this position.
fn ore_at(vx: i32, vy: i32, vz: i32, seed: u64) -> Option<VoxelMaterial> {
    let n = terrain_noise(vx, vy, vz, seed, 0x0EE1);
    if n > 0.92 {
        // Ore type based on depth
        if vz < 8 { Some(VoxelMaterial::GoldOre) }
        else if vz < 12 { Some(VoxelMaterial::IronOre) }
        else { Some(VoxelMaterial::Coal) }
    } else if n > 0.88 && vz < 10 {
        Some(VoxelMaterial::CopperOre)
    } else if n > 0.95 && vz < 6 {
        Some(VoxelMaterial::Crystal)
    } else {
        None
    }
}

/// Simple 3D noise in [0, 1] range from integer coordinates.
fn terrain_noise(x: i32, y: i32, z: i32, seed: u64, salt: u64) -> f32 {
    let h = hash_3d(x, y, z, seed.wrapping_add(salt));
    (h as f32) / (u32::MAX as f32)
}

/// Deterministic hash from 3D integer coordinates.
fn hash_3d(x: i32, y: i32, z: i32, seed: u64) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(x as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(y as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(z as u64);
    h = h ^ (h >> 33);
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h = h ^ (h >> 33);
    (h >> 32) as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_building_metadata() {
        let mut v = Voxel::new(VoxelMaterial::StoneBrick);
        assert_eq!(v.zone, VoxelZone::None);
        assert_eq!(v.building_id, None);
        assert_eq!(v.integrity, 1.0);

        v.building_id = Some(42);
        v.zone = VoxelZone::Residential;
        v.integrity = 0.75;

        assert_eq!(v.building_id, Some(42));
        assert_eq!(v.zone, VoxelZone::Residential);
        assert!((v.integrity - 0.75).abs() < f32::EPSILON);

        let ehp = v.effective_hp();
        let expected = 0.75 * VoxelMaterial::StoneBrick.properties().hp_multiplier;
        assert!((ehp - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn chunk_pos_from_voxel() {
        assert_eq!(ChunkPos::from_voxel(0, 0, 0), ChunkPos::new(0, 0, 0));
        assert_eq!(ChunkPos::from_voxel(15, 15, 15), ChunkPos::new(0, 0, 0));
        assert_eq!(ChunkPos::from_voxel(16, 0, 0), ChunkPos::new(1, 0, 0));
        assert_eq!(ChunkPos::from_voxel(-1, 0, 0), ChunkPos::new(-1, 0, 0));
        assert_eq!(ChunkPos::from_voxel(-16, 0, 0), ChunkPos::new(-1, 0, 0));
        assert_eq!(ChunkPos::from_voxel(-17, 0, 0), ChunkPos::new(-2, 0, 0));
    }

    #[test]
    fn voxel_roundtrip() {
        let (vx, vy, vz) = (10, -5, 30);
        let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
        assert_eq!(cp, ChunkPos::new(0, -1, 1));
        // Local coords: (10, 11, 14)
        let lx = vx.rem_euclid(16) as usize;
        let ly = vy.rem_euclid(16) as usize;
        let lz = vz.rem_euclid(16) as usize;
        assert_eq!(idx, local_index(lx, ly, lz));
    }

    #[test]
    fn generate_and_query() {
        let mut world = VoxelWorld::default();
        world.generate_chunk(ChunkPos::new(0, 0, 0), 42);
        world.generate_chunk(ChunkPos::new(0, 0, 1), 42);
        world.generate_chunk(ChunkPos::new(0, 0, 2), 42);

        // Bedrock at z=0
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Granite);
        // Stone at z=10
        let v = world.get_voxel(0, 0, 10);
        assert!(v.material == VoxelMaterial::Stone || v.material.is_solid());
        // Air well above surface
        assert_eq!(world.get_voxel(0, 0, 40).material, VoxelMaterial::Air);
    }

    #[test]
    fn set_and_get_voxel() {
        let mut world = VoxelWorld::default();
        world.set_voxel(5, 5, 5, Voxel::new(VoxelMaterial::IronOre));
        assert_eq!(world.get_voxel(5, 5, 5).material, VoxelMaterial::IronOre);
        assert_eq!(world.get_voxel(5, 5, 6).material, VoxelMaterial::Air);
    }

    #[test]
    fn voxel_flags() {
        let mut v = Voxel::new(VoxelMaterial::Water);
        v.set_water_level(12);
        assert_eq!(v.water_level(), 12);
        v.set_source(true);
        assert!(v.is_source());
        assert_eq!(v.water_level(), 12); // didn't clobber
    }

    #[test]
    fn surface_height() {
        let mut world = VoxelWorld::default();
        world.generate_chunk(ChunkPos::new(0, 0, 0), 42);
        world.generate_chunk(ChunkPos::new(0, 0, 1), 42);
        world.generate_chunk(ChunkPos::new(0, 0, 2), 42);

        let h = world.surface_height(8, 8);
        // Should be around 25-35 (surface ≈ 30 ± 5)
        assert!(h >= 20 && h <= 40, "surface height {} out of expected range", h);
    }

    // -----------------------------------------------------------------------
    // Destruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn mine_dirt_breaks_in_one_hit() {
        let mut world = VoxelWorld::default();
        world.set_voxel(0, 0, 0, Voxel::new(VoxelMaterial::Dirt));
        assert!(world.get_voxel(0, 0, 0).material.is_solid());

        // Dirt hardness = 5. One hit with damage=10 should break it.
        let result = world.mine_voxel(0, 0, 0, 10);
        assert!(result.is_some(), "dirt should break with damage 10 >= hardness 5");
        let (mat, yield_info) = result.unwrap();
        assert_eq!(mat, VoxelMaterial::Dirt);
        assert!(yield_info.is_some(), "dirt should yield something when mined");

        // Voxel should now be air.
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Air);
    }

    #[test]
    fn mine_stone_requires_multiple_hits() {
        let mut world = VoxelWorld::default();
        world.set_voxel(5, 5, 5, Voxel::new(VoxelMaterial::Stone));

        // Stone hardness = 30. Hit with 10 damage three times.
        let r1 = world.mine_voxel(5, 5, 5, 10);
        assert!(r1.is_none(), "stone should NOT break after 10/30 damage");
        assert_eq!(world.get_voxel(5, 5, 5).damage, 10);

        let r2 = world.mine_voxel(5, 5, 5, 10);
        assert!(r2.is_none(), "stone should NOT break after 20/30 damage");
        assert_eq!(world.get_voxel(5, 5, 5).damage, 20);

        let r3 = world.mine_voxel(5, 5, 5, 10);
        assert!(r3.is_some(), "stone SHOULD break after 30/30 damage");
        assert_eq!(world.get_voxel(5, 5, 5).material, VoxelMaterial::Air);
    }

    #[test]
    fn mine_granite_is_unbreakable() {
        let mut world = VoxelWorld::default();
        world.set_voxel(0, 0, 0, Voxel::new(VoxelMaterial::Granite));

        let result = world.mine_voxel(0, 0, 0, 255);
        assert!(result.is_none(), "granite should be unbreakable");
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Granite);
    }

    #[test]
    fn mine_air_returns_none() {
        let mut world = VoxelWorld::default();
        // Default is air — mining air should do nothing.
        world.set_voxel(0, 0, 0, Voxel::new(VoxelMaterial::Air));
        let result = world.mine_voxel(0, 0, 0, 10);
        assert!(result.is_none(), "mining air should return None");
    }

    #[test]
    fn mine_ore_yields_correct_commodity() {
        let mut world = VoxelWorld::default();
        world.set_voxel(0, 0, 0, Voxel::new(VoxelMaterial::IronOre));

        // IronOre hardness = 35
        let result = world.mine_voxel(0, 0, 0, 255); // one-shot
        let (mat, yield_info) = result.unwrap();
        assert_eq!(mat, VoxelMaterial::IronOre);
        let (commodity, amount) = yield_info.unwrap();
        assert_eq!(commodity, crate::world_sim::commodity::IRON);
        assert!(amount > 0.0);
    }

    #[test]
    fn destruction_marks_chunk_dirty() {
        let mut world = VoxelWorld::default();
        world.set_voxel(0, 0, 0, Voxel::new(VoxelMaterial::Dirt));

        // Clear dirty flag manually.
        let cp = ChunkPos::from_voxel(0, 0, 0);
        world.chunks.get_mut(&cp).unwrap().dirty = false;

        world.mine_voxel(0, 0, 0, 255);
        assert!(world.chunks.get(&cp).unwrap().dirty, "chunk should be dirty after mining");
    }

    // -----------------------------------------------------------------------
    // Cross-chunk boundary tests
    // -----------------------------------------------------------------------

    #[test]
    fn cross_chunk_set_and_get() {
        let mut world = VoxelWorld::default();

        // Place voxels straddling a chunk boundary at x=15 and x=16.
        world.set_voxel(15, 0, 0, Voxel::new(VoxelMaterial::Stone));
        world.set_voxel(16, 0, 0, Voxel::new(VoxelMaterial::IronOre));

        // They should be in different chunks.
        let cp1 = ChunkPos::from_voxel(15, 0, 0);
        let cp2 = ChunkPos::from_voxel(16, 0, 0);
        assert_ne!(cp1, cp2, "voxels at x=15 and x=16 must be in different chunks");

        // Both should be readable.
        assert_eq!(world.get_voxel(15, 0, 0).material, VoxelMaterial::Stone);
        assert_eq!(world.get_voxel(16, 0, 0).material, VoxelMaterial::IronOre);
    }

    #[test]
    fn cross_chunk_negative_coords() {
        let mut world = VoxelWorld::default();

        // Negative coordinates should work correctly.
        world.set_voxel(-1, -1, -1, Voxel::new(VoxelMaterial::Clay));
        world.set_voxel(0, 0, 0, Voxel::new(VoxelMaterial::Sand));

        assert_eq!(world.get_voxel(-1, -1, -1).material, VoxelMaterial::Clay);
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Sand);

        // Should be in different chunks.
        let cp_neg = ChunkPos::from_voxel(-1, -1, -1);
        let cp_pos = ChunkPos::from_voxel(0, 0, 0);
        assert_ne!(cp_neg, cp_pos);
    }

    #[test]
    fn neighbors6_across_chunk_boundary() {
        let mut world = VoxelWorld::default();

        // Place stone at the boundary between two chunks.
        world.set_voxel(15, 8, 8, Voxel::new(VoxelMaterial::Stone));
        world.set_voxel(16, 8, 8, Voxel::new(VoxelMaterial::Stone));

        // Check that neighbors6 of x=15 includes x=16 (cross-chunk).
        let neighbors = VoxelWorld::neighbors6(15, 8, 8);
        assert!(neighbors.contains(&(16, 8, 8)));
        assert!(neighbors.contains(&(14, 8, 8)));

        // Both neighbors should be queryable across chunk boundaries.
        for (nx, ny, nz) in &neighbors {
            let _v = world.get_voxel(*nx, *ny, *nz); // should not panic
        }
    }

    #[test]
    fn mine_at_chunk_boundary() {
        let mut world = VoxelWorld::default();

        // Place stone at x=15 (last voxel in chunk 0).
        world.set_voxel(15, 0, 0, Voxel::new(VoxelMaterial::Dirt));

        // Mine it — should work fine across boundary math.
        let result = world.mine_voxel(15, 0, 0, 255);
        assert!(result.is_some());
        assert_eq!(world.get_voxel(15, 0, 0).material, VoxelMaterial::Air);

        // Adjacent voxel in next chunk should be unaffected.
        assert_eq!(world.get_voxel(16, 0, 0).material, VoxelMaterial::Air);
    }

    // -----------------------------------------------------------------------
    // Water at chunk boundaries
    // -----------------------------------------------------------------------

    #[test]
    fn water_visible_across_chunk_boundary() {
        let mut world = VoxelWorld::default();

        // Place water at the very edge of chunk (0,0,0) — local x=15.
        let mut water = Voxel::new(VoxelMaterial::Water);
        water.set_water_level(8);
        water.set_source(true);
        world.set_voxel(15, 0, 0, water);

        // Query from the perspective of the neighboring chunk.
        let v = world.get_voxel(15, 0, 0);
        assert_eq!(v.material, VoxelMaterial::Water);
        assert_eq!(v.water_level(), 8);
        assert!(v.is_source());

        // The next-door voxel (x=16, in chunk 1,0,0) should be air.
        assert_eq!(world.get_voxel(16, 0, 0).material, VoxelMaterial::Air);
    }

    #[test]
    fn water_column_spans_chunks_vertically() {
        let mut world = VoxelWorld::default();

        // Fill a column of water from z=10 to z=20, crossing the chunk boundary at z=16.
        for z in 10..=20 {
            let mut w = Voxel::new(VoxelMaterial::Water);
            w.set_water_level(15);
            world.set_voxel(0, 0, z, w);
        }

        // Verify all are water, including across the z=15/16 chunk boundary.
        for z in 10..=20 {
            let v = world.get_voxel(0, 0, z);
            assert_eq!(v.material, VoxelMaterial::Water,
                "water at z={} should be present (crosses chunk at z=16)", z);
            assert_eq!(v.water_level(), 15);
        }

        // Verify it's in two different chunks.
        let cp_low = ChunkPos::from_voxel(0, 0, 10);
        let cp_high = ChunkPos::from_voxel(0, 0, 20);
        assert_ne!(cp_low, cp_high, "z=10 and z=20 should be in different chunks");
    }

    // -----------------------------------------------------------------------
    // Terrain generation consistency at chunk boundaries
    // -----------------------------------------------------------------------

    #[test]
    fn terrain_consistent_across_chunk_boundary() {
        let mut world = VoxelWorld::default();
        let seed = 42;

        // Generate two horizontally adjacent chunks.
        world.generate_chunk(ChunkPos::new(0, 0, 0), seed);
        world.generate_chunk(ChunkPos::new(1, 0, 0), seed);

        // The voxel at x=15 (end of chunk 0) and x=16 (start of chunk 1)
        // should have consistent terrain — no seams.
        // At z=0-2, both should be bedrock (granite).
        assert_eq!(world.get_voxel(15, 8, 0).material, VoxelMaterial::Granite);
        assert_eq!(world.get_voxel(16, 8, 0).material, VoxelMaterial::Granite);

        // At z=10 (deep stone), both should be solid.
        assert!(world.get_voxel(15, 8, 10).material.is_solid(),
            "x=15 at z=10 should be solid stone layer");
        assert!(world.get_voxel(16, 8, 10).material.is_solid(),
            "x=16 at z=10 should be solid stone layer");

        // Surface heights should be close (within 1-2 voxels) since they're
        // adjacent columns.
        let h_left = world.surface_height(15, 8);
        let h_right = world.surface_height(16, 8);
        let diff = (h_left - h_right).abs();
        // Not guaranteed to be equal (noise), but shouldn't be wildly different.
        // With ±5 noise range, adjacent columns could differ by up to 10 in worst case.
        assert!(diff <= 10, "surface heights at x=15 ({}) and x=16 ({}) differ by {} — terrain may have seams",
            h_left, h_right, diff);
    }

    #[test]
    fn terrain_deterministic_across_generation_order() {
        let seed = 99;

        // Generate chunks in order A, B.
        let mut world1 = VoxelWorld::default();
        world1.generate_chunk(ChunkPos::new(0, 0, 0), seed);
        world1.generate_chunk(ChunkPos::new(1, 0, 0), seed);

        // Generate chunks in order B, A.
        let mut world2 = VoxelWorld::default();
        world2.generate_chunk(ChunkPos::new(1, 0, 0), seed);
        world2.generate_chunk(ChunkPos::new(0, 0, 0), seed);

        // Voxels should be identical regardless of generation order.
        for vx in 0..32 {
            for vy in 0..16 {
                for vz in 0..16 {
                    let v1 = world1.get_voxel(vx, vy, vz);
                    let v2 = world2.get_voxel(vx, vy, vz);
                    assert_eq!(v1.material, v2.material,
                        "voxel at ({},{},{}) differs based on chunk generation order", vx, vy, vz);
                }
            }
        }
    }

    #[test]
    fn is_chunk_boundary_detects_edges() {
        assert!(VoxelWorld::is_chunk_boundary(0, 5, 5));   // x=0 → local x=0
        assert!(VoxelWorld::is_chunk_boundary(15, 5, 5));  // x=15 → local x=15
        assert!(VoxelWorld::is_chunk_boundary(16, 5, 5));  // x=16 → local x=0 (next chunk)
        assert!(!VoxelWorld::is_chunk_boundary(8, 8, 8));  // center of chunk
        assert!(VoxelWorld::is_chunk_boundary(-1, 0, 0));  // x=-1 → local x=15
    }

    // -----------------------------------------------------------------------
    // Area operation tests
    // -----------------------------------------------------------------------

    #[test]
    fn remove_box_clears_region() {
        let mut world = VoxelWorld::default();
        // Fill a 4x4x4 solid cube.
        world.fill_box((0, 0, 0), (3, 3, 3), VoxelMaterial::Stone);
        for vx in 0..=3 { for vy in 0..=3 { for vz in 0..=3 {
            assert!(world.get_voxel(vx, vy, vz).material.is_solid());
        }}}

        // Remove a 2x2x2 subregion.
        let results = world.remove_box((1, 1, 1), (2, 2, 2));
        assert_eq!(results.len(), 8, "should have removed 2x2x2 = 8 voxels");
        for (mat, _) in &results {
            assert_eq!(*mat, VoxelMaterial::Stone);
        }

        // Inner region should be air.
        for vx in 1..=2 { for vy in 1..=2 { for vz in 1..=2 {
            assert_eq!(world.get_voxel(vx, vy, vz).material, VoxelMaterial::Air);
        }}}
        // Outer shell should still be stone.
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Stone);
        assert_eq!(world.get_voxel(3, 3, 3).material, VoxelMaterial::Stone);
    }

    #[test]
    fn remove_box_across_chunk_boundary() {
        let mut world = VoxelWorld::default();
        // Fill stone across the chunk boundary at x=14..17.
        world.fill_box((14, 0, 0), (17, 0, 0), VoxelMaterial::Stone);

        let results = world.remove_box((14, 0, 0), (17, 0, 0));
        assert_eq!(results.len(), 4);

        // All four should be air now — even across chunks.
        for vx in 14..=17 {
            assert_eq!(world.get_voxel(vx, 0, 0).material, VoxelMaterial::Air,
                "voxel at x={} should be air after cross-chunk remove_box", vx);
        }
    }

    #[test]
    fn remove_box_skips_granite() {
        let mut world = VoxelWorld::default();
        world.fill_box((0, 0, 0), (2, 2, 2), VoxelMaterial::Granite);

        let results = world.remove_box((0, 0, 0), (2, 2, 2));
        assert_eq!(results.len(), 0, "granite is unbreakable, nothing should be removed");

        // All should still be granite.
        assert_eq!(world.get_voxel(1, 1, 1).material, VoxelMaterial::Granite);
    }

    #[test]
    fn remove_sphere_shape() {
        let mut world = VoxelWorld::default();
        // Fill a big cube.
        world.fill_box((-5, -5, -5), (5, 5, 5), VoxelMaterial::Dirt);

        // Remove a sphere of radius 3 at the center.
        let results = world.remove_sphere((0, 0, 0), 3);
        assert!(results.len() > 0, "sphere removal should clear some voxels");

        // Center should be air.
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Air);
        // Just outside radius should still be solid.
        assert_eq!(world.get_voxel(4, 0, 0).material, VoxelMaterial::Dirt);
        assert_eq!(world.get_voxel(0, 4, 0).material, VoxelMaterial::Dirt);
        // Corner of sphere (3,0,0) should be air (3^2 = 9 <= 9).
        assert_eq!(world.get_voxel(3, 0, 0).material, VoxelMaterial::Air);
        // Diagonal (2,2,2) distance^2 = 12 > 9, should still be solid.
        assert_eq!(world.get_voxel(2, 2, 2).material, VoxelMaterial::Dirt);
    }

    #[test]
    fn remove_sphere_across_chunks() {
        let mut world = VoxelWorld::default();
        // Sphere centered at chunk boundary.
        world.fill_box((12, 12, 12), (20, 20, 20), VoxelMaterial::Stone);
        let results = world.remove_sphere((16, 16, 16), 3);
        assert!(results.len() > 0);
        // Center (16,16,16 — chunk boundary) should be air.
        assert_eq!(world.get_voxel(16, 16, 16).material, VoxelMaterial::Air);
        // Far corner should still be stone.
        assert_eq!(world.get_voxel(20, 20, 20).material, VoxelMaterial::Stone);
    }

    #[test]
    fn fill_box_and_replace() {
        let mut world = VoxelWorld::default();
        world.fill_box((0, 0, 0), (9, 9, 9), VoxelMaterial::Dirt);

        // Replace dirt with stone in a subregion.
        let count = world.replace_in_box((2, 2, 2), (7, 7, 7), VoxelMaterial::Dirt, VoxelMaterial::Stone);
        assert_eq!(count, 6 * 6 * 6); // 216 replaced

        // Inner should be stone, outer should still be dirt.
        assert_eq!(world.get_voxel(5, 5, 5).material, VoxelMaterial::Stone);
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Dirt);
    }

    #[test]
    fn replace_only_matches_target_material() {
        let mut world = VoxelWorld::default();
        world.fill_box((0, 0, 0), (3, 3, 3), VoxelMaterial::Stone);
        world.set_voxel(1, 1, 1, Voxel::new(VoxelMaterial::IronOre));

        // Replace Stone→Dirt, should not touch the IronOre voxel.
        let count = world.replace_in_box((0, 0, 0), (3, 3, 3), VoxelMaterial::Stone, VoxelMaterial::Dirt);
        assert_eq!(count, 4 * 4 * 4 - 1); // all except the ore
        assert_eq!(world.get_voxel(1, 1, 1).material, VoxelMaterial::IronOre);
        assert_eq!(world.get_voxel(0, 0, 0).material, VoxelMaterial::Dirt);
    }

    #[test]
    fn remove_box_returns_yields() {
        let mut world = VoxelWorld::default();
        world.fill_box((0, 0, 0), (1, 1, 1), VoxelMaterial::IronOre);

        let results = world.remove_box((0, 0, 0), (1, 1, 1));
        assert_eq!(results.len(), 8);
        for (mat, yield_info) in &results {
            assert_eq!(*mat, VoxelMaterial::IronOre);
            let (commodity, amount) = yield_info.unwrap();
            assert_eq!(commodity, crate::world_sim::commodity::IRON);
            assert!(amount > 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // MaterialProperties tests
    // -----------------------------------------------------------------------

    #[test]
    fn material_properties_all_variants() {
        let all = [
            VoxelMaterial::Air, VoxelMaterial::Stone, VoxelMaterial::Granite,
            VoxelMaterial::Basalt, VoxelMaterial::Sandstone, VoxelMaterial::Marble,
            VoxelMaterial::Dirt, VoxelMaterial::Clay, VoxelMaterial::Sand,
            VoxelMaterial::Gravel, VoxelMaterial::Ice, VoxelMaterial::Snow,
            VoxelMaterial::Water, VoxelMaterial::Lava,
            VoxelMaterial::WoodLog, VoxelMaterial::WoodPlanks, VoxelMaterial::Thatch,
            VoxelMaterial::Bone,
            VoxelMaterial::Brick, VoxelMaterial::CutStone, VoxelMaterial::Concrete,
            VoxelMaterial::Glass, VoxelMaterial::Ceramic,
            VoxelMaterial::Iron, VoxelMaterial::Steel, VoxelMaterial::Bronze,
            VoxelMaterial::CopperOre, VoxelMaterial::GoldOre,
            VoxelMaterial::Obsidian, VoxelMaterial::Crystal,
            VoxelMaterial::IronOre, VoxelMaterial::Coal,
            VoxelMaterial::Grass, VoxelMaterial::Farmland, VoxelMaterial::Crop,
            VoxelMaterial::StoneBlock, VoxelMaterial::StoneBrick,
        ];
        for mat in &all {
            let props = mat.properties();
            assert!(props.hp_multiplier > 0.0 || mat.is_fluid() || *mat == VoxelMaterial::Air);
            if mat.is_solid() {
                assert!(props.weight > 0.0, "{:?} should have positive weight", mat);
            }
        }
    }

    #[test]
    fn material_properties_values() {
        let steel = VoxelMaterial::Steel.properties();
        assert!(steel.hp_multiplier > VoxelMaterial::WoodLog.properties().hp_multiplier);
        assert!(steel.blast_resistance > VoxelMaterial::Glass.properties().blast_resistance);
        assert!(VoxelMaterial::Glass.properties().load_bearing == false);
        assert!(VoxelMaterial::Stone.properties().load_bearing == true);
    }

    // -----------------------------------------------------------------------
    // Destructible terrain tests
    // -----------------------------------------------------------------------

    #[test]
    fn voxel_world_damage_destroys() {
        let mut world = VoxelWorld::default();
        let cp = ChunkPos::new(0, 0, 0);
        world.generate_chunk(cp, 42);

        // Find a solid voxel that isn't bedrock
        let vx = 8;
        let vy = 8;
        let vz = 10; // stone layer
        let mat = world.get_voxel(vx, vy, vz).material;
        assert!(mat.is_solid(), "expected solid at z=10");

        let hp = mat.properties().hp_multiplier;
        // Damage it to destruction
        let destroyed = world.damage_voxel(vx, vy, vz, hp + 1.0);
        assert!(destroyed);

        let after = world.get_voxel(vx, vy, vz);
        // Load-bearing materials become rubble (integrity 0), non-load-bearing become Air
        if mat.properties().load_bearing {
            assert_eq!(after.integrity, 0.0);
            assert_eq!(after.zone, VoxelZone::None);
        } else {
            assert_eq!(after.material, VoxelMaterial::Air);
        }
    }

    #[test]
    fn voxel_world_damage_partial() {
        let mut world = VoxelWorld::default();
        let cp = ChunkPos::new(0, 0, 0);
        world.generate_chunk(cp, 42);

        let vx = 8;
        let vy = 8;
        let vz = 10;
        let mat = world.get_voxel(vx, vy, vz).material;
        let hp = mat.properties().hp_multiplier;

        // Partial damage
        let destroyed = world.damage_voxel(vx, vy, vz, hp * 0.3);
        assert!(!destroyed);

        let after = world.get_voxel(vx, vy, vz);
        assert!(after.integrity > 0.0 && after.integrity < 1.0);
    }

    #[test]
    fn cascading_collapse() {
        // Column at z=10,11,12 — z=10 supported by solid below
        let mut world = VoxelWorld::default();
        // Fill z=0..10 with stone as foundation
        for z in 0..=10 {
            world.set_voxel(5, 5, z, Voxel::new(VoxelMaterial::Stone));
        }
        // Add z=11, 12 on top
        world.set_voxel(5, 5, 11, Voxel::new(VoxelMaterial::Stone));
        world.set_voxel(5, 5, 12, Voxel::new(VoxelMaterial::Stone));

        // Destroy z=10 (the top of the foundation)
        let hp = VoxelMaterial::Stone.properties().hp_multiplier;
        world.damage_voxel(5, 5, 10, hp + 1.0);

        // z=10 is now rubble (integrity 0, load_bearing → stays as stone rubble)
        let v10 = world.get_voxel(5, 5, 10);
        assert_eq!(v10.integrity, 0.0);

        // z=11 should collapse — z=10 below is rubble (integrity 0), no horizontal support
        let v11 = world.get_voxel(5, 5, 11);
        assert_eq!(v11.integrity, 0.0, "z=11 should collapse without support");

        // z=12 should also cascade
        let v12 = world.get_voxel(5, 5, 12);
        assert_eq!(v12.integrity, 0.0, "z=12 should cascade");
    }
}
