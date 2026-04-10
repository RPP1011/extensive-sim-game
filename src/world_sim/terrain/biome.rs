use crate::world_sim::state::{Terrain, SubBiome};
use crate::world_sim::voxel::VoxelMaterial;

// ---------------------------------------------------------------------------
// BiomeVolume — resolved 3D biome at a given depth
// ---------------------------------------------------------------------------

/// Fully-resolved biome for a 3D location: surface context + underground variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiomeVolume {
    /// The surface terrain type above this location.
    pub surface: Terrain,
    /// The surface sub-biome variant above this location.
    pub surface_sub: SubBiome,
    /// The underground biome at this depth.
    pub underground: SubBiome,
}

/// Resolve a `BiomeVolume` for a 3D location.
///
/// - `depth_below_surface`: positive = above surface, negative = underground (voxels).
/// - `seed`: location seed for future randomized variation (unused for now).
pub fn resolve_biome(
    surface_terrain: Terrain,
    surface_sub: SubBiome,
    depth_below_surface: i32,
    _seed: u64,
) -> BiomeVolume {
    let underground = if depth_below_surface >= -40 {
        // Near-surface: minimal underground, only Caverns terrain gets a proper cave
        match surface_terrain {
            Terrain::Caverns => SubBiome::NaturalCave,
            _ => SubBiome::Standard,
        }
    } else if depth_below_surface >= -250 {
        // Shallow underground: strong surface influence
        match surface_terrain {
            Terrain::Volcano => SubBiome::LavaTubes,
            Terrain::Mountains => SubBiome::NaturalCave,
            Terrain::Tundra | Terrain::Glacier => SubBiome::FrozenCavern,
            Terrain::Forest | Terrain::Jungle => SubBiome::MushroomGrove,
            Terrain::Swamp | Terrain::Coast => SubBiome::Aquifer,
            Terrain::DeathZone | Terrain::AncientRuins => SubBiome::BoneOssuary,
            Terrain::Caverns => SubBiome::CrystalVein,
            _ => SubBiome::NaturalCave,
        }
    } else if depth_below_surface >= -350 {
        // Deep underground: surface influence fades
        match surface_terrain {
            Terrain::Volcano => SubBiome::LavaTubes,
            Terrain::Tundra | Terrain::Glacier => SubBiome::FrozenCavern,
            _ => SubBiome::NaturalCave,
        }
    } else {
        // Abyss: always lava tubes
        SubBiome::LavaTubes
    };

    BiomeVolume { surface: surface_terrain, surface_sub, underground }
}

// ---------------------------------------------------------------------------
// SurfaceMaterials — material layers for the top of a terrain type
// ---------------------------------------------------------------------------

/// The three material layers at a terrain surface (top → bottom).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceMaterials {
    /// Top voxel layer (what you walk on).
    pub surface: VoxelMaterial,
    /// Subsoil beneath the surface layer.
    pub subsoil: VoxelMaterial,
    /// Deep stone beneath the subsoil.
    pub deep_stone: VoxelMaterial,
}

/// Map a surface `Terrain` to its characteristic material layers.
pub fn surface_materials(terrain: Terrain) -> SurfaceMaterials {
    use VoxelMaterial::*;
    let (surface, subsoil, deep_stone) = match terrain {
        Terrain::Plains      => (TallGrass, Dirt, Stone),
        Terrain::Forest      => (Grass, Dirt, Stone),
        Terrain::Jungle      => (JungleMoss, Clay, Stone),
        Terrain::Desert      => (Sand, Sandstone, Stone),
        Terrain::Badlands    => (RedSand, Sandstone, Granite),
        Terrain::Mountains   => (Stone, Stone, Granite),
        Terrain::Tundra      => (Peat, Gravel, Stone),
        Terrain::Glacier     => (Snow, Ice, Stone),
        Terrain::Swamp       => (MudGrass, Clay, Stone),
        Terrain::Coast       => (Sand, Sand, Stone),
        Terrain::CoralReef   => (Sand, Sand, Stone),
        Terrain::Volcano     => (Basalt, Basalt, Obsidian),
        Terrain::Caverns     => (Stone, Stone, Stone),
        Terrain::DeepOcean   => (Sand, Clay, Stone),
        Terrain::DeathZone   => (Bone, Dirt, Stone),
        Terrain::AncientRuins => (Grass, CutStone, Stone),
        Terrain::FlyingIslands => (Grass, Dirt, Stone),
    };
    SurfaceMaterials { surface, subsoil, deep_stone }
}

// ---------------------------------------------------------------------------
// Cave material helpers
// ---------------------------------------------------------------------------

/// Wall material for a given underground biome.
pub fn cave_materials(biome: SubBiome) -> VoxelMaterial {
    match biome {
        SubBiome::LavaTubes    => VoxelMaterial::Basalt,
        SubBiome::FrozenCavern => VoxelMaterial::Ice,
        SubBiome::MushroomGrove => VoxelMaterial::Dirt,
        SubBiome::CrystalVein  => VoxelMaterial::Crystal,
        SubBiome::Aquifer      => VoxelMaterial::Clay,
        SubBiome::BoneOssuary  => VoxelMaterial::Bone,
        _ => VoxelMaterial::Stone,
    }
}

/// Fill material (fluid or air) that floods cave chambers of a given biome.
pub fn cave_fill(biome: SubBiome) -> VoxelMaterial {
    match biome {
        SubBiome::LavaTubes    => VoxelMaterial::Lava,
        SubBiome::Aquifer      => VoxelMaterial::Water,
        SubBiome::FrozenCavern => VoxelMaterial::Ice,
        _ => VoxelMaterial::Air,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::{Terrain, SubBiome};

    #[test]
    fn surface_returns_surface_biome() {
        let vol = resolve_biome(Terrain::Forest, SubBiome::DenseForest, 100, 42);
        assert_eq!(vol.surface, Terrain::Forest);
        assert_eq!(vol.surface_sub, SubBiome::DenseForest);
    }

    #[test]
    fn deep_underground_varies_by_surface() {
        let under_volcano = resolve_biome(Terrain::Volcano, SubBiome::Standard, -200, 42);
        assert_eq!(under_volcano.underground, SubBiome::LavaTubes);

        let under_tundra = resolve_biome(Terrain::Tundra, SubBiome::Standard, -200, 42);
        assert_eq!(under_tundra.underground, SubBiome::FrozenCavern);

        let under_forest = resolve_biome(Terrain::Forest, SubBiome::Standard, -200, 42);
        assert_eq!(under_forest.underground, SubBiome::MushroomGrove);
    }

    #[test]
    fn abyss_is_always_lava() {
        let vol = resolve_biome(Terrain::Plains, SubBiome::Standard, -500, 42);
        assert_eq!(vol.underground, SubBiome::LavaTubes);
    }

    #[test]
    fn surface_materials_differ_by_biome() {
        let desert = surface_materials(Terrain::Desert);
        let forest = surface_materials(Terrain::Forest);
        assert_ne!(desert.surface, forest.surface);
        assert_ne!(desert.subsoil, forest.subsoil);
    }
}
