//! Overworld grid rendering pipeline — 150x80+ ASCII landscape.
//!
//! Five rendering layers: faction tint, terrain glyphs, faction borders,
//! entities (settlements + parties), and UI overlays. Supports hot-reloadable
//! terrain visual config and smooth pan/zoom camera at 60fps.

pub mod terrain_gen;
pub mod border;
pub mod camera;
pub mod renderer;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Grid dimensions (150x80+)
// ---------------------------------------------------------------------------

pub const GRID_W: u16 = 150;
pub const GRID_H: u16 = 80;

// ---------------------------------------------------------------------------
// Terrain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainType {
    DeepWater,
    ShallowWater,
    Coast,
    Plains,
    Grassland,
    Marsh,
    Forest,
    DenseForest,
    Hills,
    Foothills,
    Mountain,
    Peak,
    Road,
    Settlement,
}

impl TerrainType {
    pub fn is_water(self) -> bool {
        matches!(self, Self::DeepWater | Self::ShallowWater | Self::Coast)
    }

    pub fn movement_cost(self) -> f32 {
        match self {
            Self::Road => 0.5,
            Self::Plains | Self::Grassland => 1.0,
            Self::Forest => 1.5,
            Self::DenseForest => 2.0,
            Self::Hills | Self::Foothills => 1.8,
            Self::Mountain => 3.0,
            Self::Peak => 5.0,
            Self::Marsh => 2.5,
            Self::Coast => 1.2,
            Self::ShallowWater => 4.0,
            Self::DeepWater => 10.0,
            Self::Settlement => 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain visuals config (hot-reloadable from TOML)
// ---------------------------------------------------------------------------

/// Per-terrain-type visual properties loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainVisual {
    pub glyph: String, // Single character (as string for TOML compat)
    pub color: [u8; 3], // RGB
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainVisualsFile {
    #[serde(default)]
    pub terrain: HashMap<String, TerrainVisual>,
}

/// Registry of terrain visual properties, keyed by terrain name.
#[derive(Debug, Clone)]
pub struct TerrainVisualRegistry {
    visuals: HashMap<TerrainType, TerrainVisual>,
}

impl Default for TerrainVisualRegistry {
    fn default() -> Self {
        let mut visuals = HashMap::new();
        let defaults = [
            (TerrainType::Plains,      ".", [0x8A, 0x9A, 0x7A]),
            (TerrainType::Grassland,   ".", [0x78, 0x96, 0x64]),
            (TerrainType::Forest,      "♣", [0x5A, 0x7A, 0x4A]),
            (TerrainType::DenseForest, "♣", [0x3C, 0x5F, 0x32]),
            (TerrainType::Hills,       "~", [0x9A, 0x8A, 0x6A]),
            (TerrainType::Foothills,   "~", [0x8A, 0x80, 0x60]),
            (TerrainType::Mountain,    "^", [0x8A, 0x7A, 0x5A]),
            (TerrainType::Peak,        "▲", [0xAA, 0xA0, 0x96]),
            (TerrainType::ShallowWater,"≈", [0x7A, 0x9A, 0xBB]),
            (TerrainType::DeepWater,   "≈", [0x50, 0x78, 0xAA]),
            (TerrainType::Coast,       ",", [0xB4, 0xAA, 0x8C]),
            (TerrainType::Marsh,       "~", [0x64, 0x8C, 0x6E]),
            (TerrainType::Road,        "═", [0xA0, 0x9B, 0x91]),
            (TerrainType::Settlement,  "⌂", [0xDD, 0xDD, 0xCC]),
        ];
        for (terrain, glyph, color) in defaults {
            visuals.insert(terrain, TerrainVisual {
                glyph: glyph.to_string(),
                color,
            });
        }
        TerrainVisualRegistry { visuals }
    }
}

impl TerrainVisualRegistry {
    /// Load from a TOML file, falling back to defaults for missing entries.
    pub fn load_from_str(&mut self, toml_str: &str) -> Result<usize, String> {
        let file: TerrainVisualsFile =
            toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {}", e))?;
        let mut count = 0;
        for (name, visual) in &file.terrain {
            if let Some(terrain_type) = terrain_from_name(name) {
                self.visuals.insert(terrain_type, visual.clone());
                count += 1;
            }
        }
        Ok(count)
    }

    pub fn load_from_file(&mut self, path: &Path) -> Result<usize, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        self.load_from_str(&contents)
    }

    pub fn get(&self, terrain: TerrainType) -> (&str, [u8; 3]) {
        self.visuals
            .get(&terrain)
            .map(|v| (v.glyph.as_str(), v.color))
            .unwrap_or(("?", [0x80, 0x80, 0x80]))
    }
}

fn terrain_from_name(name: &str) -> Option<TerrainType> {
    match name.to_lowercase().as_str() {
        "deep_water" | "deepwater" => Some(TerrainType::DeepWater),
        "shallow_water" | "shallowwater" => Some(TerrainType::ShallowWater),
        "coast" => Some(TerrainType::Coast),
        "plains" => Some(TerrainType::Plains),
        "grassland" => Some(TerrainType::Grassland),
        "marsh" => Some(TerrainType::Marsh),
        "forest" => Some(TerrainType::Forest),
        "dense_forest" | "denseforest" => Some(TerrainType::DenseForest),
        "hills" => Some(TerrainType::Hills),
        "foothills" => Some(TerrainType::Foothills),
        "mountain" => Some(TerrainType::Mountain),
        "peak" => Some(TerrainType::Peak),
        "road" => Some(TerrainType::Road),
        "settlement" => Some(TerrainType::Settlement),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Map cell
// ---------------------------------------------------------------------------

/// A single cell in the overworld grid.
#[derive(Debug, Clone)]
pub struct MapCell {
    pub region_id: u16,
    pub faction_id: u8,
    pub terrain: TerrainType,
    pub height: f32,
    pub moisture: f32,
    pub glyph: char,
    pub fg_color: [u8; 3],
    pub is_border: bool,
    pub border_glyph: char,
}

impl Default for MapCell {
    fn default() -> Self {
        Self {
            region_id: 0,
            faction_id: 0,
            terrain: TerrainType::Plains,
            height: 0.0,
            moisture: 0.0,
            glyph: '.',
            fg_color: [0x8A, 0x9A, 0x7A],
            is_border: false,
            border_glyph: ' ',
        }
    }
}

// ---------------------------------------------------------------------------
// Settlement and road types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Settlement {
    pub name: String,
    pub region_id: u16,
    pub x: u16,
    pub y: u16,
    pub kind: SettlementKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlementKind {
    Town,
    Castle,
    Camp,
    Ruin,
}

impl SettlementKind {
    pub fn glyph(self) -> char {
        match self {
            Self::Town => '⌂',
            Self::Castle => '■',
            Self::Camp => '▲',
            Self::Ruin => '†',
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoadSegment {
    pub from: (u16, u16),
    pub to: (u16, u16),
}

// ---------------------------------------------------------------------------
// Overworld grid
// ---------------------------------------------------------------------------

/// The full overworld grid — generated once per campaign, cached.
#[derive(Debug, Clone)]
pub struct OverworldGrid {
    pub width: u16,
    pub height: u16,
    pub cells: Vec<MapCell>,
    pub settlements: Vec<Settlement>,
    pub roads: Vec<RoadSegment>,
    /// Incremented when faction territories change, signaling mesh rebuild.
    pub faction_version: u64,
}

impl OverworldGrid {
    pub fn new(width: u16, height: u16) -> Self {
        let size = width as usize * height as usize;
        Self {
            width,
            height,
            cells: vec![MapCell::default(); size],
            settlements: Vec::new(),
            roads: Vec::new(),
            faction_version: 0,
        }
    }

    #[inline]
    pub fn cell(&self, x: u16, y: u16) -> &MapCell {
        &self.cells[y as usize * self.width as usize + x as usize]
    }

    #[inline]
    pub fn cell_mut(&mut self, x: u16, y: u16) -> &mut MapCell {
        let w = self.width as usize;
        &mut self.cells[y as usize * w + x as usize]
    }

    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32
    }

    /// Apply visual properties from the terrain visual registry.
    pub fn apply_visuals(&mut self, visuals: &TerrainVisualRegistry) {
        for cell in &mut self.cells {
            if !cell.is_border {
                let (glyph_str, color) = visuals.get(cell.terrain);
                cell.glyph = glyph_str.chars().next().unwrap_or('?');
                cell.fg_color = color;
            }
        }
    }

    /// Bump faction version to trigger cached mesh rebuild.
    pub fn invalidate(&mut self) {
        self.faction_version += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_visual_registry_defaults() {
        let reg = TerrainVisualRegistry::default();
        let (glyph, color) = reg.get(TerrainType::Plains);
        assert_eq!(glyph, ".");
        assert_eq!(color, [0x8A, 0x9A, 0x7A]);
    }

    #[test]
    fn test_terrain_visual_toml_override() {
        let toml = r#"
[terrain.plains]
glyph = ","
color = [100, 200, 100]

[terrain.mountain]
glyph = "M"
color = [150, 120, 80]
"#;
        let mut reg = TerrainVisualRegistry::default();
        let count = reg.load_from_str(toml).unwrap();
        assert_eq!(count, 2);

        let (glyph, color) = reg.get(TerrainType::Plains);
        assert_eq!(glyph, ",");
        assert_eq!(color, [100, 200, 100]);
    }

    #[test]
    fn test_grid_creation() {
        let grid = OverworldGrid::new(GRID_W, GRID_H);
        assert_eq!(grid.cells.len(), GRID_W as usize * GRID_H as usize);
    }

    #[test]
    fn test_grid_bounds_check() {
        let grid = OverworldGrid::new(GRID_W, GRID_H);
        assert!(grid.in_bounds(0, 0));
        assert!(grid.in_bounds(149, 79));
        assert!(!grid.in_bounds(150, 0));
        assert!(!grid.in_bounds(0, 80));
        assert!(!grid.in_bounds(-1, 0));
    }

    #[test]
    fn test_movement_costs() {
        assert!(TerrainType::Road.movement_cost() < TerrainType::Plains.movement_cost());
        assert!(TerrainType::Plains.movement_cost() < TerrainType::Mountain.movement_cost());
        assert!(TerrainType::Forest.movement_cost() < TerrainType::DenseForest.movement_cost());
    }

    #[test]
    fn test_terrain_from_name() {
        assert_eq!(terrain_from_name("plains"), Some(TerrainType::Plains));
        assert_eq!(terrain_from_name("deep_water"), Some(TerrainType::DeepWater));
        assert_eq!(terrain_from_name("DeepWater"), Some(TerrainType::DeepWater));
        assert_eq!(terrain_from_name("bogus"), None);
    }
}
