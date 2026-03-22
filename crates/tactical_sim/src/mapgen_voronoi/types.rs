use serde::Serialize;

#[derive(Debug, Clone)]
pub struct OverworldData {
    pub map_seed: u64,
    pub regions: Vec<RegionInput>,
    pub factions: Vec<FactionInput>,
}

#[derive(Debug, Clone)]
pub struct RegionInput {
    pub id: usize,
    pub name: String,
    pub owner_faction_id: usize,
    pub unrest: f64,
    pub control: f64,
    pub intel_level: f64,
    pub neighbors: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FactionInput {
    pub id: usize,
    pub name: String,
    pub strength: f64,
    pub cohesion: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Site {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegionMeta {
    pub id: usize,
    pub name: String,
    pub owner_faction_id: usize,
    pub site: Site,
    pub weight: f64,
    pub area_pct: f64,
    pub organic_jitter: f64,
    pub boundary_complexity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FactionMeta {
    pub id: usize,
    pub name: String,
    pub strength: f64,
    pub cohesion: f64,
    pub territory_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Road {
    pub a: usize,
    pub b: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Settlement {
    pub faction_id: usize,
    pub region_id: usize,
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoronoiSpec {
    pub map_seed: u64,
    pub grid: GridSize,
    pub strength_scale: f64,
    pub organic_jitter: f64,
    pub regions: Vec<RegionMeta>,
    pub factions: Vec<FactionMeta>,
    pub road_count: usize,
    pub roads: Vec<Road>,
    pub settlements: Vec<Settlement>,
    pub terrain_notes: String,
    pub frontier_notes: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GridSize {
    pub w: usize,
    pub h: usize,
}
