pub mod noise;
pub mod region_plan;
pub mod biome;

pub use region_plan::{RegionPlan, RegionCell, generate_continent, CELL_SIZE, SEA_LEVEL, MAX_SURFACE_Z};
pub use biome::{BiomeVolume, resolve_biome, surface_materials};
