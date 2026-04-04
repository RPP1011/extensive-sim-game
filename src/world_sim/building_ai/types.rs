//! Core types shared across all building intelligence workstreams.

use serde::{Deserialize, Serialize};

use crate::world_sim::state::{tag, BuildingType};

// ---------------------------------------------------------------------------
// Tag constants for building intelligence
// ---------------------------------------------------------------------------

pub mod bi_tags {
    use super::tag;

    // Structural lessons
    pub const WALL_TOO_LOW: u32 = tag(b"wall_too_low");
    pub const WALL_TOO_THIN: u32 = tag(b"wall_too_thin");
    pub const WOOD_BURNS: u32 = tag(b"wood_burns");
    pub const FLOOD_LOW_GROUND: u32 = tag(b"flood_low_ground");
    pub const CHOKEPOINT_EXPLOITABLE: u32 = tag(b"chokepoint_exploitable");
    pub const GARRISON_COMPENSATES: u32 = tag(b"garrison_compensates");

    // Reasoning tags (for dataset inspection)
    pub const THREAT_PROXIMITY: u32 = tag(b"threat_proximity");
    pub const GARRISON_SYNERGY: u32 = tag(b"garrison_synergy");
    pub const RESOURCE_SCARCITY: u32 = tag(b"resource_scarcity");
    pub const HOUSING_PRESSURE: u32 = tag(b"housing_pressure");
    pub const FIRE_RECOVERY: u32 = tag(b"fire_recovery");
    pub const FLOOD_PREVENTION: u32 = tag(b"flood_prevention");
    pub const JUMP_COUNTER: u32 = tag(b"jump_counter");
    pub const SIEGE_COUNTER: u32 = tag(b"siege_counter");
    pub const SPECIALIST_ACCESS: u32 = tag(b"specialist_access");
    pub const LEADER_PROTECTION: u32 = tag(b"leader_protection");
    pub const UPGRADE_PATH: u32 = tag(b"upgrade_path");
    pub const SEASONAL_PREP: u32 = tag(b"seasonal_prep");
    pub const TERRAIN_ADAPT: u32 = tag(b"terrain_adapt");
}

// ---------------------------------------------------------------------------
// Memory system — 3-tier importance-filtered ring buffers
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer backed by a Vec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingBuffer<T> {
    pub items: Vec<T>,
    pub capacity: usize,
    pub head: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
            head: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.items.len() < self.capacity {
            self.items.push(item);
        } else {
            self.items[self.head] = item;
        }
        self.head = (self.head + 1) % self.capacity;
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Per-settlement construction memory with importance-filtered tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructionMemory {
    /// Raw events, circular overwrite, all events land here.
    pub short_term: RingBuffer<ConstructionEvent>,
    /// Aggregated patterns, importance > 0.3, decay halves every 500 ticks.
    pub medium_term: RingBuffer<AggregatedPattern>,
    /// Structural lessons, importance > 0.7, permanent until contradicted.
    pub long_term: RingBuffer<StructuralLesson>,
}

impl ConstructionMemory {
    pub fn new() -> Self {
        Self {
            short_term: RingBuffer::new(64),
            medium_term: RingBuffer::new(256),
            long_term: RingBuffer::new(64),
        }
    }
}

impl Default for ConstructionMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructionEvent {
    pub tick: u64,
    pub kind: ConstructionEventKind,
    /// 0.0–1.0 normalized severity.
    pub severity: f32,
    /// Grid cell where this happened.
    pub location: (u16, u16),
    /// Entity that caused this (attacker, builder, etc).
    pub source_entity: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConstructionEventKind {
    WallBreach = 0,
    WallDamage = 1,
    BuildingDestroyed = 2,
    BuildingDamaged = 3,
    FloodDamage = 4,
    FireSpread = 5,
    EarthquakeDamage = 6,
    PathBlocked = 7,
    ResourceDepleted = 8,
    PopulationOverflow = 9,
    EnemySighted = 10,
    GarrisonEngaged = 11,
    ConstructionCompleted = 12,
    UpgradeCompleted = 13,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPattern {
    pub kind: ConstructionEventKind,
    pub count: u16,
    pub mean_severity: f32,
    pub location_centroid: (f32, f32),
    pub first_tick: u64,
    pub last_tick: u64,
    /// `severity × recency × novelty`
    pub importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralLesson {
    /// FNV hash tag, e.g. `bi_tags::WALL_TOO_LOW`.
    pub lesson_tag: u32,
    /// 0.0–1.0, increases with repeated confirmation.
    pub confidence: f32,
    /// Which event kinds contributed to this lesson.
    pub source_patterns: Vec<ConstructionEventKind>,
    pub learned_tick: u64,
}

// ---------------------------------------------------------------------------
// Building material & structural enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BuildMaterial {
    Wood = 0,
    Stone = 1,
    Iron = 2,
    Brick = 3,
    Thatch = 4,
}

impl Default for BuildMaterial {
    fn default() -> Self { Self::Wood }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoofType {
    Flat = 0,
    Pitched = 1,
    Reinforced = 2,
    Walkable = 3,
}

impl Default for RoofType {
    fn default() -> Self { Self::Pitched }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FoundationType {
    Slab = 0,
    Raised = 1,
    Pilings = 2,
    Terraced = 3,
    Deep = 4,
}

impl Default for FoundationType {
    fn default() -> Self { Self::Slab }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FootprintShape {
    Rectangular = 0,
    LShape = 1,
    UShape = 2,
    Circular = 3,
    Irregular = 4,
}

impl Default for FootprintShape {
    fn default() -> Self { Self::Rectangular }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpeningType {
    Door = 0,
    Window = 1,
    ArrowSlit = 2,
    MurderHole = 3,
    SallyPort = 4,
    ShutteredWindow = 5,
}

// ---------------------------------------------------------------------------
// Wall features — per-segment structural detail
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WallFeatures {
    pub crenellations: bool,
    pub buttressed: bool,
    pub overhang: bool,
    pub smooth_face: bool,
}

impl Default for WallFeatures {
    fn default() -> Self {
        Self {
            crenellations: false,
            buttressed: false,
            overhang: false,
            smooth_face: false,
        }
    }
}

/// Per-wall-face structural specification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WallComponentSpec {
    pub height: u8,
    pub thickness: u8,
    pub material: BuildMaterial,
    pub features: WallFeatures,
}

impl Default for WallComponentSpec {
    fn default() -> Self {
        Self {
            height: 3,
            thickness: 1,
            material: BuildMaterial::Wood,
            features: WallFeatures::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeningSpec {
    pub opening_type: OpeningType,
    pub wall_facing: Direction,
    pub count: u8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentUpgrade {
    pub component: BuildingComponent,
    pub new_material: Option<BuildMaterial>,
    pub new_thickness: Option<u8>,
    pub new_height: Option<u8>,
    pub add_features: Option<WallFeatures>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BuildingComponent {
    NorthWall = 0,
    EastWall = 1,
    SouthWall = 2,
    WestWall = 3,
    Roof = 4,
    Foundation = 5,
    Interior = 6,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomPlacement {
    pub kind: RoomKindBi,
    pub offset: (f32, f32),
    pub size: (f32, f32),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoomKindBi {
    Bedroom = 0,
    Kitchen = 1,
    Hearth = 2,
    Workshop = 3,
    Storeroom = 4,
    Entrance = 5,
    Study = 6,
    Armory = 7,
    ArcherPlatform = 8,
    MageTower = 9,
    TrainingYard = 10,
    Vault = 11,
    SafeRoom = 12,
    EscapeRoute = 13,
}

// ---------------------------------------------------------------------------
// Challenge description
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChallengeCategory {
    Military = 0,
    Environmental = 1,
    Economic = 2,
    Population = 3,
    Temporal = 4,
    Terrain = 5,
    MultiSettlement = 6,
    UnitCapability = 7,
    HighValueNpc = 8,
    LevelScaled = 9,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    pub category: ChallengeCategory,
    /// FNV hash for sub-type (e.g. tag(b"raid_wall_jumpers")).
    pub sub_type: u32,
    /// Human-readable sub-type name (from TOML).
    pub sub_type_name: String,
    pub severity: f32,
    /// Threat direction, None for non-directional challenges.
    pub direction: Option<(f32, f32)>,
    /// Tick deadline for temporal challenges.
    pub deadline_tick: Option<u64>,
    pub enemy_profiles: Vec<EnemyProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyProfile {
    pub type_tag: u32,
    pub type_name: String,
    pub level_range: (u8, u8),
    pub count: u16,
    pub can_jump: bool,
    pub jump_height: u8,
    pub can_climb: bool,
    pub can_tunnel: bool,
    pub can_fly: bool,
    pub has_siege: bool,
    pub siege_damage: f32,
}

// ---------------------------------------------------------------------------
// Unit summaries (pre-extracted from WorldState)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitSummary {
    pub entity_id: u32,
    pub level: u8,
    pub class_tag: u32,
    pub combat_effectiveness: f32,
    pub position: (f32, f32),
    pub is_garrison: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighValueNpc {
    pub entity_id: u32,
    /// Role tag: tag(b"leader"), tag(b"master_smith"), etc.
    pub role_tag: u32,
    pub role_name: String,
    pub level: u8,
    pub protection_priority: f32,
    pub position: (f32, f32),
}

// ---------------------------------------------------------------------------
// Decision tier & types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DecisionTier {
    Strategic = 0,
    Structural = 1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DecisionType {
    // Strategic (grid-level)
    Placement = 0,
    Prioritization = 1,
    Routing = 2,
    ZoneComposition = 3,
    Demolition = 4,
    // Structural (tile-level)
    FootprintGeometry = 5,
    VerticalDesign = 6,
    WallComposition = 7,
    RoofDesign = 8,
    Foundation = 9,
    Openings = 10,
    InteriorFlow = 11,
    RoomSpecialization = 12,
    MaterialSelection = 13,
    DefensiveIntegration = 14,
    EnvironmentalAdaptation = 15,
    ExpansionProvision = 16,
    RenovationUpgrade = 17,
}

impl DecisionType {
    pub fn tier(self) -> DecisionTier {
        match self as u8 {
            0..=4 => DecisionTier::Strategic,
            _ => DecisionTier::Structural,
        }
    }
}

// ---------------------------------------------------------------------------
// Action payloads (BC label output)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingAction {
    pub decision_type: DecisionType,
    pub tier: DecisionTier,
    pub action: ActionPayload,
    pub priority: f32,
    /// For dataset inspection, not consumed by the model.
    pub reasoning_tag: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionPayload {
    PlaceBuilding {
        building_type: BuildingType,
        grid_cell: (u16, u16),
    },
    SetBuildPriority {
        building_id: u32,
        priority: f32,
    },
    RouteRoad {
        waypoints: Vec<(u16, u16)>,
    },
    SetZone {
        grid_cell: (u16, u16),
        zone: String,
    },
    Demolish {
        building_id: u32,
    },
    SetFootprint {
        building_id: u32,
        shape: FootprintShape,
        dimensions: (u8, u8),
    },
    SetVertical {
        building_id: u32,
        stories: u8,
        has_basement: bool,
        elevation: u8,
    },
    SetWallSpec {
        segment_id: u32,
        height: u8,
        thickness: u8,
        material: BuildMaterial,
        features: WallFeatures,
    },
    SetRoofSpec {
        building_id: u32,
        roof_type: RoofType,
        material: BuildMaterial,
    },
    SetFoundation {
        building_id: u32,
        foundation_type: FoundationType,
        depth: u8,
    },
    SetOpenings {
        building_id: u32,
        openings: Vec<OpeningSpec>,
    },
    SetInteriorLayout {
        building_id: u32,
        rooms: Vec<RoomPlacement>,
    },
    SetMaterial {
        building_id: u32,
        component: BuildingComponent,
        material: BuildMaterial,
    },
    Renovate {
        building_id: u32,
        upgrades: Vec<ComponentUpgrade>,
    },
}

// ---------------------------------------------------------------------------
// Observation (BC model input)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingObservation {
    pub settlement_id: u32,
    pub tick: u64,
    pub challenges: Vec<Challenge>,
    pub memory: ConstructionMemory,
    pub spatial: super::features::SpatialFeatures,
    pub friendly_roster: Vec<UnitSummary>,
    pub high_value_npcs: Vec<HighValueNpc>,
    pub settlement_level: u8,
    pub tech_tier: u8,
    pub decision_tier: DecisionTier,
}
