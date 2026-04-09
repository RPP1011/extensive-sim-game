use serde::{Serialize, Deserialize};
use std::collections::HashSet;

/// Voxels per sector edge.
pub const SECTOR_SIZE: i32 = 4096;

/// 3D sector coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SectorPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl SectorPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self { Self { x, y, z } }

    pub fn from_voxel(vx: i32, vy: i32, vz: i32) -> Self {
        Self {
            x: vx.div_euclid(SECTOR_SIZE),
            y: vy.div_euclid(SECTOR_SIZE),
            z: vz.div_euclid(SECTOR_SIZE),
        }
    }

    pub fn sectors_around_voxel(vx: i32, vy: i32, vz: i32, radius: i32) -> Vec<SectorPos> {
        let center = Self::from_voxel(vx, vy, vz);
        let mut result = Vec::new();
        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    result.push(SectorPos::new(center.x + dx, center.y + dy, center.z + dz));
                }
            }
        }
        result
    }
}

/// Manages which 3D sectors are active.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SectorGrid {
    active: HashSet<SectorPos>,
}

impl SectorGrid {
    pub fn new() -> Self { Self { active: HashSet::new() } }
    pub fn is_active(&self, sp: &SectorPos) -> bool { self.active.contains(sp) }
    pub fn activate(&mut self, sp: SectorPos) { self.active.insert(sp); }
    pub fn deactivate(&mut self, sp: &SectorPos) { self.active.remove(sp); }
    pub fn active_sectors(&self) -> impl Iterator<Item = &SectorPos> { self.active.iter() }
    pub fn active_count(&self) -> usize { self.active.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sector_from_voxel() {
        let sp = SectorPos::from_voxel(100, 200, 300);
        assert_eq!(sp, SectorPos::new(0, 0, 0));
        let sp2 = SectorPos::from_voxel(5000, 5000, 5000);
        assert_eq!(sp2, SectorPos::new(1, 1, 1));
    }

    #[test]
    fn sector_activation() {
        let mut grid = SectorGrid::new();
        let sp = SectorPos::new(0, 0, 0);
        assert!(!grid.is_active(&sp));
        grid.activate(sp);
        assert!(grid.is_active(&sp));
        grid.deactivate(&sp);
        assert!(!grid.is_active(&sp));
    }

    #[test]
    fn sectors_around() {
        let sectors = SectorPos::sectors_around_voxel(2048, 2048, 50, 1);
        assert_eq!(sectors.len(), 27);
    }
}
