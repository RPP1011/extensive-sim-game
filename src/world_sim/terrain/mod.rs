pub mod noise;
pub mod region_plan;
pub mod biome;
pub mod materialize;
pub mod caves;
pub mod rivers;
pub mod features;
pub mod sky;
pub mod dungeons;

pub use region_plan::{RegionPlan, RegionCell, generate_continent, CELL_SIZE, SEA_LEVEL, MAX_SURFACE_Z};
pub use biome::{BiomeVolume, resolve_biome, surface_materials};
pub use materialize::materialize_chunk;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{VoxelWorld, ChunkPos, VoxelMaterial, CHUNK_SIZE};

    #[test]
    fn end_to_end_terrain_generation() {
        let plan = generate_continent(10, 10, 42);
        let mut voxel_world = VoxelWorld::default();
        voxel_world.region_plan = Some(plan);

        // Deep underground chunk (z=0 → voxels 0-15): always below any surface,
        // so depth > 80 → deep stone (solid).
        voxel_world.generate_chunk(ChunkPos::new(5, 5, 0), 42);
        // Deep bedrock (z=-10 → voxels -160 to -145): below -120 → always Granite.
        voxel_world.generate_chunk(ChunkPos::new(5, 5, -10), 42);
        // Sky chunk (z=30 → voxels 480-495, well above MAX_SURFACE_Z=400): always air.
        voxel_world.generate_chunk(ChunkPos::new(5, 5, 30), 42);

        let underground = voxel_world.chunks.get(&ChunkPos::new(5, 5, 0)).unwrap();
        let solid = underground.voxels.iter().filter(|v| v.material.is_solid()).count();
        assert!(solid > 0, "underground chunk has no solid voxels");

        let bedrock = voxel_world.chunks.get(&ChunkPos::new(5, 5, -10)).unwrap();
        let granite_count = bedrock.voxels.iter()
            .filter(|v| v.material == VoxelMaterial::Granite).count();
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert_eq!(granite_count, total, "bedrock chunk should be all Granite");

        let sky = voxel_world.chunks.get(&ChunkPos::new(5, 5, 30)).unwrap();
        let air = sky.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        assert!(air > total / 2, "sky chunk not mostly air");
    }
}
