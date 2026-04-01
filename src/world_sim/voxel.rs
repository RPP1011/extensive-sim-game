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
}

// ---------------------------------------------------------------------------
// Voxel
// ---------------------------------------------------------------------------

/// Per-voxel data. 4 bytes — millions of these exist.
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
}

impl Default for Voxel {
    fn default() -> Self {
        Self { material: VoxelMaterial::Air, light: 0, damage: 0, flags: 0 }
    }
}

impl Voxel {
    pub fn new(material: VoxelMaterial) -> Self {
        Self { material, light: 0, damage: 0, flags: 0 }
    }

    pub fn water_level(self) -> u8 { self.flags & 0x0F }
    pub fn set_water_level(&mut self, level: u8) {
        self.flags = (self.flags & 0xF0) | (level & 0x0F);
    }
    pub fn is_source(self) -> bool { self.flags & 0x40 != 0 }
    pub fn set_source(&mut self, v: bool) {
        if v { self.flags |= 0x40; } else { self.flags &= !0x40; }
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

    /// Count loaded chunks.
    pub fn chunk_count(&self) -> usize { self.chunks.len() }

    /// Count total solid voxels across all loaded chunks.
    pub fn total_solid(&self) -> usize {
        self.chunks.values().map(|c| c.solid_count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Terrain generation helpers
// ---------------------------------------------------------------------------

/// Simple hash-based noise for terrain height variation (±5 voxels).
fn terrain_height_offset(vx: i32, vy: i32, seed: u64) -> i32 {
    let h = hash_3d(vx, vy, 0, seed);
    ((h % 11) as i32) - 5 // range: -5 to +5
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
}
