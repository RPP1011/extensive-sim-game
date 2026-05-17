//! Lower a parsed [`TerrainBlock`] into a [`TerrainIr`] suitable for
//! the WGSL emitter (Task 8).
//!
//! The only non-trivial work at this stage is resolving material *names*
//! (string identifiers from the DSL source) to their declared numeric
//! `id` values, and computing the FNV-1a hash of any `biome_tag` strings.

use dsl_ast::terrain::{LayerKind, TerrainBlock};
use std::collections::HashMap;

/// Flattened, name-resolved representation of a `terrain { … }` block.
#[derive(Debug, Clone)]
pub struct TerrainIr {
    pub extent: u32,
    pub cell_size: f32,
    pub seed_purpose: u32,
    pub materials: Vec<MaterialIr>,
    pub layers: Vec<TerrainLayerIr>,
}

/// Name-resolved material entry.
#[derive(Debug, Clone)]
pub struct MaterialIr {
    pub id: u8,
    pub walkable: bool,
    pub hardness: u8,
    /// FNV-1a of the `biome_tag` string, or `0` when absent.
    pub biome_tag_hash: u32,
    pub color: u32,
    pub movement_cost: f32,
}

/// Name-resolved terrain layer.
#[derive(Debug, Clone)]
pub enum TerrainLayerIr {
    Fill { material_id: u8 },
}

/// Errors produced during terrain lowering.
#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("unknown material: {0}")]
    UnknownMaterial(String),
}

/// Lower a parsed [`TerrainBlock`] to a [`TerrainIr`].
///
/// Returns [`LowerError::UnknownMaterial`] if any layer references a
/// material name that was not declared in the `materials { … }` block.
pub fn lower_terrain(t: &TerrainBlock) -> Result<TerrainIr, LowerError> {
    // Build name → id lookup for the resolution pass.
    let by_name: HashMap<&str, u8> = t
        .materials
        .iter()
        .map(|m| (m.name.as_str(), m.id))
        .collect();

    let materials = t
        .materials
        .iter()
        .map(|m| MaterialIr {
            id: m.id,
            walkable: m.walkable,
            hardness: m.hardness,
            biome_tag_hash: m.biome_tag.as_deref().map(fnv1a_u32).unwrap_or(0),
            color: m.color,
            movement_cost: m.movement_cost,
        })
        .collect();

    let mut layers = Vec::with_capacity(t.layers.len());
    for l in &t.layers {
        match &l.kind {
            LayerKind::Fill { material } => {
                let mid = by_name
                    .get(material.as_str())
                    .copied()
                    .ok_or_else(|| LowerError::UnknownMaterial(material.clone()))?;
                layers.push(TerrainLayerIr::Fill { material_id: mid });
            }
        }
    }

    Ok(TerrainIr {
        extent: t.extent,
        cell_size: t.cell_size,
        seed_purpose: t.seed_purpose,
        materials,
        layers,
    })
}

/// FNV-1a 32-bit hash — deterministic (P5), same string → same output
/// regardless of toolchain version.
fn fnv1a_u32(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}
