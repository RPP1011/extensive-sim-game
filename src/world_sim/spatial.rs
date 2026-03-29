//! Spatial hash index for O(1) entity lookups by position.
//!
//! Rebuilt each tick from the frozen snapshot. Entities are hashed into
//! fixed-size cells. Grid queries become cell lookups instead of linear scans.

use std::collections::HashMap;

use super::state::{Entity, WorldTeam};

/// Cell size in world units. Entities within the same cell are co-located.
const CELL_SIZE: f32 = 32.0;

/// A cell coordinate in the spatial hash.
type CellKey = (i32, i32);

/// Per-cell data: lists of entity indices (into WorldState.entities) by category.
#[derive(Default)]
struct Cell {
    /// All entity indices in this cell.
    all: Vec<usize>,
    /// Hostile (monster) entity indices.
    hostile: Vec<usize>,
    /// Friendly (NPC) entity indices.
    friendly: Vec<usize>,
}

/// Spatial hash index built from a snapshot of entities.
pub struct SpatialIndex {
    cells: HashMap<CellKey, Cell>,
    /// entity_id → index into WorldState.entities.
    id_to_idx: HashMap<u32, usize>,
}

impl SpatialIndex {
    /// Create an empty index. Call `rebuild()` to populate.
    pub fn new() -> Self {
        SpatialIndex {
            cells: HashMap::new(),
            id_to_idx: HashMap::new(),
        }
    }

    /// Build the spatial index from all alive entities (allocates on first call).
    pub fn build(entities: &[Entity]) -> Self {
        let mut si = Self::new();
        si.rebuild(entities);
        si
    }

    /// Clear and rebuild in-place. Reuses all existing HashMap capacity.
    pub fn rebuild(&mut self, entities: &[Entity]) {
        // Clear cell contents without deallocating the cell vecs.
        for cell in self.cells.values_mut() {
            cell.all.clear();
            cell.hostile.clear();
            cell.friendly.clear();
        }
        self.id_to_idx.clear();

        for (idx, entity) in entities.iter().enumerate() {
            self.id_to_idx.insert(entity.id, idx);
            if !entity.alive { continue; }

            let key = pos_to_cell(entity.pos);
            let cell = self.cells.entry(key).or_default();
            cell.all.push(idx);

            match entity.team {
                WorldTeam::Hostile => cell.hostile.push(idx),
                WorldTeam::Friendly => cell.friendly.push(idx),
                WorldTeam::Neutral => {}
            }
        }
    }

    /// Fast entity lookup by ID → index.
    #[inline]
    pub fn entity_idx(&self, id: u32) -> Option<usize> {
        self.id_to_idx.get(&id).copied()
    }

    /// Check if any hostile (monster) entities exist within a circle.
    pub fn has_hostiles_in_radius(&self, center: (f32, f32), radius: f32) -> bool {
        for key in cells_in_radius(center, radius) {
            if let Some(cell) = self.cells.get(&key) {
                if !cell.hostile.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    /// Check if any friendly (NPC) entities exist within a circle.
    pub fn has_friendlies_in_radius(&self, center: (f32, f32), radius: f32) -> bool {
        for key in cells_in_radius(center, radius) {
            if let Some(cell) = self.cells.get(&key) {
                if !cell.friendly.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    /// Get all entity indices within a circle.
    pub fn entities_in_radius<'a>(
        &'a self,
        entities: &'a [Entity],
        center: (f32, f32),
        radius: f32,
    ) -> impl Iterator<Item = &'a Entity> + 'a {
        let r2 = radius * radius;
        cells_in_radius(center, radius)
            .into_iter()
            .filter_map(move |key| self.cells.get(&key))
            .flat_map(|cell| cell.all.iter())
            .filter_map(move |&idx| {
                let e = &entities[idx];
                let dx = e.pos.0 - center.0;
                let dy = e.pos.1 - center.1;
                if dx * dx + dy * dy <= r2 { Some(e) } else { None }
            })
    }

    /// Get all hostile entity indices within a circle.
    pub fn hostiles_in_radius<'a>(
        &'a self,
        entities: &'a [Entity],
        center: (f32, f32),
        radius: f32,
    ) -> impl Iterator<Item = &'a Entity> + 'a {
        let r2 = radius * radius;
        cells_in_radius(center, radius)
            .into_iter()
            .filter_map(move |key| self.cells.get(&key))
            .flat_map(|cell| cell.hostile.iter())
            .filter_map(move |&idx| {
                let e = &entities[idx];
                let dx = e.pos.0 - center.0;
                let dy = e.pos.1 - center.1;
                if dx * dx + dy * dy <= r2 { Some(e) } else { None }
            })
    }

    /// Get all friendly entity indices within a circle.
    pub fn friendlies_in_radius<'a>(
        &'a self,
        entities: &'a [Entity],
        center: (f32, f32),
        radius: f32,
    ) -> impl Iterator<Item = &'a Entity> + 'a {
        let r2 = radius * radius;
        cells_in_radius(center, radius)
            .into_iter()
            .filter_map(move |key| self.cells.get(&key))
            .flat_map(|cell| cell.friendly.iter())
            .filter_map(move |&idx| {
                let e = &entities[idx];
                let dx = e.pos.0 - center.0;
                let dy = e.pos.1 - center.1;
                if dx * dx + dy * dy <= r2 { Some(e) } else { None }
            })
    }
}

#[inline]
fn pos_to_cell(pos: (f32, f32)) -> CellKey {
    (
        (pos.0 / CELL_SIZE).floor() as i32,
        (pos.1 / CELL_SIZE).floor() as i32,
    )
}

/// Return all cell keys that overlap with a circle.
fn cells_in_radius(center: (f32, f32), radius: f32) -> Vec<CellKey> {
    let min_cx = ((center.0 - radius) / CELL_SIZE).floor() as i32;
    let max_cx = ((center.0 + radius) / CELL_SIZE).floor() as i32;
    let min_cy = ((center.1 - radius) / CELL_SIZE).floor() as i32;
    let max_cy = ((center.1 + radius) / CELL_SIZE).floor() as i32;

    let mut keys = Vec::with_capacity(((max_cx - min_cx + 1) * (max_cy - min_cy + 1)) as usize);
    for cx in min_cx..=max_cx {
        for cy in min_cy..=max_cy {
            keys.push((cx, cy));
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::Entity;

    #[test]
    fn basic_spatial_lookup() {
        let entities = vec![
            Entity::new_npc(1, (10.0, 10.0)),
            Entity::new_monster(2, (15.0, 10.0), 1),
            Entity::new_npc(3, (200.0, 200.0)),
        ];
        let idx = SpatialIndex::build(&entities);

        // Entity 1 and 2 are close, entity 3 is far.
        assert!(idx.has_friendlies_in_radius((10.0, 10.0), 20.0));
        assert!(idx.has_hostiles_in_radius((10.0, 10.0), 20.0));
        assert!(!idx.has_hostiles_in_radius((200.0, 200.0), 20.0));
        assert!(idx.has_friendlies_in_radius((200.0, 200.0), 20.0));
    }

    #[test]
    fn id_lookup() {
        let entities = vec![
            Entity::new_npc(42, (0.0, 0.0)),
            Entity::new_monster(99, (100.0, 0.0), 1),
        ];
        let idx = SpatialIndex::build(&entities);
        assert_eq!(idx.entity_idx(42), Some(0));
        assert_eq!(idx.entity_idx(99), Some(1));
        assert_eq!(idx.entity_idx(999), None);
    }
}
