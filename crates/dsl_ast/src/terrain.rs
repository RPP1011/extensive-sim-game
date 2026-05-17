//! AST for the `terrain { ... }` block. See
//! `docs/superpowers/specs/2026-05-17-terrain-dsl-spec-design.md`.

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TerrainBlock {
    pub extent: u32,
    pub cell_size: f32,
    pub seed_purpose: u32,
    pub materials: Vec<MaterialDecl>,
    pub layers: Vec<LayerDecl>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MaterialDecl {
    pub name: String,
    pub id: u8,
    pub walkable: bool,
    pub hardness: u8,
    pub biome_tag: Option<String>,
    pub color: u32,
    pub movement_cost: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerDecl {
    pub index: u32,
    pub kind: LayerKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum LayerKind {
    Fill { material: String },
}
