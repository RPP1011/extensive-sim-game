//! Const-friendly per-material property table. Stored alongside the
//! `VoxelGrid` inside `VoxelTerrain`; rows are produced by the DSL
//! emitter. See spec section "Material properties".

#[derive(Copy, Clone, Debug)]
pub struct MaterialRow {
    pub id: u8,
    pub walkable: bool,
    pub hardness: u8,
    pub biome_tag_hash: u32,
    pub color: u32,
    pub movement_cost: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct MaterialTable {
    rows: &'static [MaterialRow],
}

impl MaterialTable {
    pub const fn new(rows: &'static [MaterialRow]) -> Self {
        Self { rows }
    }

    pub fn get(&self, id: u8) -> Option<&MaterialRow> {
        if id == 0 { return None; }
        self.rows.iter().find(|r| r.id == id)
    }

    pub fn rows(&self) -> &'static [MaterialRow] { self.rows }
}

/// Empty table used by `VoxelTerrain::new()` / `with_extent()` callers
/// that have not gone through the DSL terrain pipeline.
pub const EMPTY: MaterialTable = MaterialTable::new(&[]);
