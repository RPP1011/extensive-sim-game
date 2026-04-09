//! Spatial constants for the world sim voxel system.
//!
//! All scale-dependent values live here so a voxel size change is a single edit.
//! Current scale: 1 voxel ≈ 5cm, 1 NPC ≈ 40 voxels tall (~2m).

// ---------------------------------------------------------------------------
// Voxel grid
// ---------------------------------------------------------------------------

/// Voxels per chunk edge. Chunks are cubic: CHUNK_SIZE³ voxels.
pub const CHUNK_SIZE: usize = 16;

/// Total voxels per chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// World units per voxel. At 0.05 (5cm), a 40-voxel NPC is 2m tall.
pub const VOXEL_SCALE: f32 = 1.0; // TODO: change to 0.05 for 5cm voxels

// ---------------------------------------------------------------------------
// Terrain generation
// ---------------------------------------------------------------------------

/// Voxels per region plan cell (horizontal). One plan cell = one biome region.
pub const CELL_SIZE: i32 = 4096;

/// Maximum surface height in voxels. Mountains peak near this value.
pub const MAX_SURFACE_Z: i32 = 400;

/// Sea level in voxels. Water fills below this in ocean/coast/swamp biomes.
pub const SEA_LEVEL: i32 = 80;

/// Base Z level where flying islands start generating.
pub const SKY_BASE_Z: i32 = 300;

// ---------------------------------------------------------------------------
// Sectors (3D spatial partitioning)
// ---------------------------------------------------------------------------

/// Voxels per sector edge. Sectors group chunks for LOD and activation.
pub const SECTOR_SIZE: i32 = 4096;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// How many sim chunks per mega-chunk axis. MEGA × CHUNK_SIZE = voxels per mega edge.
pub const MEGA: i32 = 4;

/// Voxels per mega-chunk edge.
pub const MEGA_VOXELS: u32 = (MEGA as u32) * (CHUNK_SIZE as u32);

/// Maximum distance (world units) from camera to load/render a mega-chunk.
pub const LOAD_RADIUS: f32 = 2048.0;

/// Window dimensions for the voxel renderer.
pub const RENDER_WIDTH: u32 = 1280;
pub const RENDER_HEIGHT: u32 = 720;
