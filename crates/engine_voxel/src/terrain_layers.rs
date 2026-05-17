//! Hand-written layer helpers called by emitter-generated
//! `generate_terrain(seed)` functions. Library code (not rule
//! logic) per spec — see AIS justification.

use crate::VoxelTerrain;

/// Uniform-material fill across the cubic extent. Material id `0`
/// clears cells to air. Determinism: deterministic by construction —
/// no RNG consumed.
pub fn layer_fill(terrain: &mut VoxelTerrain, material_id: u8) {
    let n = terrain.extent();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                terrain.set_cell(x, y, z, material_id);
            }
        }
    }
}
