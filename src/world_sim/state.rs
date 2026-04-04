use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::fidelity::Fidelity;
use super::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// Tag hashing — compile-time FNV-1a for tag interning
// ---------------------------------------------------------------------------

/// Compile-time FNV-1a hash for tag interning.
pub const fn tag(name: &[u8]) -> u32 {
    let mut hash: u32 = 2166136261;
    let mut i = 0;
    while i < name.len() {
        hash ^= name[i] as u32;
        hash = hash.wrapping_mul(16777619);
        i += 1;
    }
    hash
}

pub mod tags {
    use super::tag;
    // Combat
    pub const MELEE: u32 = tag(b"melee");
    pub const RANGED: u32 = tag(b"ranged");
    pub const COMBAT: u32 = tag(b"combat");
    pub const DEFENSE: u32 = tag(b"defense");
    pub const TACTICS: u32 = tag(b"tactics");
    // Craft
    pub const MINING: u32 = tag(b"mining");
    pub const SMITHING: u32 = tag(b"smithing");
    pub const CRAFTING: u32 = tag(b"crafting");
    pub const ENCHANTMENT: u32 = tag(b"enchantment");
    pub const ALCHEMY: u32 = tag(b"alchemy");
    // Social
    pub const TRADE: u32 = tag(b"trade");
    pub const DIPLOMACY: u32 = tag(b"diplomacy");
    pub const LEADERSHIP: u32 = tag(b"leadership");
    pub const NEGOTIATION: u32 = tag(b"negotiation");
    pub const DECEPTION: u32 = tag(b"deception");
    // Knowledge
    pub const RESEARCH: u32 = tag(b"research");
    pub const LORE: u32 = tag(b"lore");
    pub const MEDICINE: u32 = tag(b"medicine");
    pub const HERBALISM: u32 = tag(b"herbalism");
    pub const NAVIGATION: u32 = tag(b"navigation");
    // Survival
    pub const ENDURANCE: u32 = tag(b"endurance");
    pub const RESILIENCE: u32 = tag(b"resilience");
    pub const STEALTH: u32 = tag(b"stealth");
    pub const SURVIVAL: u32 = tag(b"survival");
    pub const AWARENESS: u32 = tag(b"awareness");
    // Spiritual
    pub const FAITH: u32 = tag(b"faith");
    pub const RITUAL: u32 = tag(b"ritual");
    // Labor
    pub const LABOR: u32 = tag(b"labor");
    pub const TEACHING: u32 = tag(b"teaching");
    pub const DISCIPLINE: u32 = tag(b"discipline");
    // Construction
    pub const CONSTRUCTION: u32 = tag(b"construction");
    pub const ARCHITECTURE: u32 = tag(b"architecture");
    pub const MASONRY: u32 = tag(b"masonry");
    // Commodity-specific
    pub const FARMING: u32 = tag(b"farming");
    pub const WOODWORK: u32 = tag(b"woodwork");
    pub const EXPLORATION: u32 = tag(b"exploration");
    // Experience-driven (from life events, not work)
    pub const COMPASSION_TAG: u32 = tag(b"compassion");
    // Seafaring / terrain-specific
    pub const SEAFARING: u32 = tag(b"seafaring");
    pub const DUNGEONEERING: u32 = tag(b"dungeoneering");
    pub const MASTERY: u32 = tag(b"mastery");
}

// ---------------------------------------------------------------------------
// ActionTags — stack-allocated tag bundle for actions
// ---------------------------------------------------------------------------

/// Stack-allocated action tag bundle. Max 8 tag-weight pairs per action.
#[derive(Debug, Clone, Copy)]
pub struct ActionTags {
    pub tags: [(u32, f32); 8],
    pub count: u8,
}

impl ActionTags {
    pub const fn empty() -> Self {
        Self { tags: [(0, 0.0); 8], count: 0 }
    }

    pub fn add(&mut self, tag_hash: u32, weight: f32) {
        if (self.count as usize) < 8 {
            self.tags[self.count as usize] = (tag_hash, weight);
            self.count += 1;
        }
    }

    pub fn merge(&mut self, other: &ActionTags) {
        for i in 0..other.count as usize {
            self.add(other.tags[i].0, other.tags[i].1);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic per-entity hash — used for randomness in systems that iterate
// entities in potentially varying order. Same (entity_id, tick, salt) always
// gives the same result regardless of iteration order.
// ---------------------------------------------------------------------------

/// Deterministic per-entity hash. Use for randomness in systems that iterate
/// entities in potentially varying order. Same (entity_id, tick, salt) always
/// gives the same result. Returns a raw u32 — use [`entity_hash_f32`] for a
/// float in [0, 1).
#[inline]
pub fn entity_hash(entity_id: u32, tick: u64, salt: u64) -> u32 {
    let h = (entity_id as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(tick)
        .wrapping_add(salt.wrapping_mul(1442695040888963407));
    (h >> 33) as u32
}

/// Convenience wrapper returning a float in [0, 1).
#[inline]
pub fn entity_hash_f32(entity_id: u32, tick: u64, salt: u64) -> f32 {
    entity_hash(entity_id, tick, salt) as f32 / u32::MAX as f32
}

/// Two-key variant for pairwise decisions (e.g. diplomacy between two factions).
/// Mixes both IDs symmetrically so (a,b) == (b,a) when desired — callers should
/// use `a.min(b)` / `a.max(b)` if symmetry matters, or pass them in a fixed order.
#[inline]
pub fn pair_hash_f32(id_a: u32, id_b: u32, tick: u64, salt: u64) -> f32 {
    let h = (id_a as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(id_b as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(tick)
        .wrapping_add(salt.wrapping_mul(1442695040888963407));
    let h = h ^ (h >> 33);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

// ---------------------------------------------------------------------------
// Tile system — sparse world-space terrain modifications
// ---------------------------------------------------------------------------

/// Integer tile position. 2.0 world units per tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TilePos {
    pub x: i32,
    pub y: i32,
}

impl TilePos {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }

    /// Convert world-space coordinates to tile position.
    pub fn from_world(wx: f32, wy: f32) -> Self {
        Self { x: (wx / 2.0).floor() as i32, y: (wy / 2.0).floor() as i32 }
    }

    /// Convert tile position to world-space center.
    pub fn to_world(self) -> (f32, f32) {
        (self.x as f32 * 2.0 + 1.0, self.y as f32 * 2.0 + 1.0)
    }

    /// 8-connected neighbors.
    pub fn neighbors8(self) -> [TilePos; 8] {
        [
            TilePos::new(self.x - 1, self.y - 1), TilePos::new(self.x, self.y - 1), TilePos::new(self.x + 1, self.y - 1),
            TilePos::new(self.x - 1, self.y),                                         TilePos::new(self.x + 1, self.y),
            TilePos::new(self.x - 1, self.y + 1), TilePos::new(self.x, self.y + 1), TilePos::new(self.x + 1, self.y + 1),
        ]
    }

    /// 4-connected neighbors (cardinal).
    pub fn neighbors4(self) -> [TilePos; 4] {
        [
            TilePos::new(self.x, self.y - 1), TilePos::new(self.x - 1, self.y),
            TilePos::new(self.x + 1, self.y), TilePos::new(self.x, self.y + 1),
        ]
    }
}

/// What type of tile is placed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileType {
    // Terrain
    Dirt,
    Stone,
    Water,

    // Structural
    Floor(TileMaterial),
    Wall(TileMaterial),
    Door,
    Window,

    // Infrastructure
    Path,
    Bridge,
    Fence,

    // Agricultural
    Farmland,

    // Functional furniture (placed inside rooms)
    Workspace(WorkspaceType),
    Bed,
    Altar,
    Bookshelf,
    StorageContainer,
    MarketStall,
    WeaponRack,
    TrainingDummy,
    Hearth,

    // Defensive
    Moat,
    TowerBase,
    GateHouse,
    ArcherPosition,
    Trap,
}

/// Construction material for walls/floors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileMaterial {
    Wood,
    Stone,
    Iron,
}

/// Type of workstation placed as a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceType {
    Forge,
    Anvil,
    Loom,
    AlchemyBench,
    Kitchen,
    Sawbench,
}

/// A placed tile in the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    pub tile_type: TileType,
    pub placed_by: Option<u32>,
    pub placed_tick: u64,
}

impl TileType {
    /// Movement speed multiplier. <1.0 = slower, >1.0 = faster, f32::MAX = impassable.
    pub fn movement_cost(self) -> f32 {
        match self {
            TileType::Path | TileType::Floor(_) | TileType::Bridge => 0.7,
            TileType::Dirt | TileType::Door => 1.0,
            TileType::Farmland | TileType::Bed | TileType::Hearth => 1.3,
            TileType::Stone | TileType::Fence => 2.0,
            TileType::Moat => 3.0,
            TileType::Wall(_) | TileType::Water | TileType::Window
            | TileType::TowerBase | TileType::GateHouse => f32::MAX,
            _ => 1.0, // furniture walkable at normal speed
        }
    }

    /// Whether this tile blocks monster movement but not NPC movement.
    pub fn blocks_monsters_only(self) -> bool {
        matches!(self, TileType::Fence)
    }

    /// Whether this tile is solid (blocks all movement and building placement).
    pub fn is_solid(self) -> bool {
        self.movement_cost() == f32::MAX
    }

    /// Whether this tile is a wall (for room enclosure detection).
    pub fn is_wall(self) -> bool {
        matches!(self, TileType::Wall(_) | TileType::Window | TileType::TowerBase)
    }

    /// Whether this tile is a floor (interior of a room).
    pub fn is_floor(self) -> bool {
        matches!(self, TileType::Floor(_))
    }

    /// Whether this tile is furniture (placed inside rooms).
    pub fn is_furniture(self) -> bool {
        matches!(self, TileType::Workspace(_) | TileType::Bed | TileType::Altar
            | TileType::Bookshelf | TileType::StorageContainer | TileType::MarketStall
            | TileType::WeaponRack | TileType::TrainingDummy | TileType::Hearth)
    }
}

/// Building function — what need a room addresses. Not a building type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingFunction {
    Shelter,
    Production,
    Worship,
    Knowledge,
    Defense,
    Trade,
    Storage,
}

impl BuildingFunction {
    /// Minimum interior tile count for this function.
    pub fn minimum_interior(self) -> u32 {
        match self {
            BuildingFunction::Shelter | BuildingFunction::Storage => 4,
            BuildingFunction::Production | BuildingFunction::Knowledge
            | BuildingFunction::Trade => 9,
            BuildingFunction::Worship => 12,
            BuildingFunction::Defense => 6,
        }
    }
}

/// A seed for room growth — placed by an NPC, grown by the automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildSeed {
    pub pos: TilePos,
    pub intended_function: BuildingFunction,
    pub minimum_interior: u32,
    pub placed_by: u32,
    pub tick: u64,
    pub complete: bool,
}

// ---------------------------------------------------------------------------
// WorldState — the complete snapshot, immutable during COMPUTE phase
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub tick: u64,
    pub rng_state: u64,

    /// All entities — backward-compatible combined view.
    /// Kept in sync with hot/cold arrays. Use `hot`/`cold` for perf-critical paths.
    pub entities: Vec<Entity>,

    /// Cache-line-friendly entity data for high-frequency iteration.
    /// Parallel array: `hot[i]` corresponds to `entities[i]`.
    #[serde(skip)]
    pub hot: Vec<HotEntity>,
    /// Heap-heavy entity data accessed only when needed.
    /// Parallel array: `cold[i]` corresponds to `entities[i]`.
    #[serde(skip)]
    pub cold: Vec<ColdEntity>,

    /// Secondary index: entity_id → index into entities/hot/cold.
    /// Rebuilt by `rebuild_entity_cache()`. Enables O(1) lookup by ID.
    #[serde(skip)]
    pub entity_index: Vec<u32>,
    /// Monotonic entity ID counter. Use `next_entity_id()` for deterministic IDs.
    pub next_id: u32,

    /// Max entity ID seen (entity_index is sized to max_id+1).
    #[serde(skip)]
    pub max_entity_id: u32,

    /// Group index: contiguous entity ranges by settlement/party.
    /// Entities are sorted by (settlement_id, party_id) so that all entities
    /// at the same settlement are adjacent. Systems iterate a slice instead of
    /// scanning all entities.
    #[serde(skip)]
    pub group_index: GroupIndex,

    /// Active fidelity zones (settlements, encounter zones).
    pub fidelity_zones: Vec<FidelityZone>,

    /// Regional data (monster populations, faction control).
    pub regions: Vec<RegionState>,

    /// Per-settlement economy (stockpiles, prices, treasury).
    pub settlements: Vec<SettlementState>,

    /// Secondary index: settlement_id → index into settlements vec.
    /// Rebuilt by `rebuild_settlement_index()`.
    #[serde(skip)]
    pub settlement_index: Vec<u32>,

    /// Sparse tile map: world-space tile modifications (walls, floors, ditches, farmland, etc.).
    /// Key: integer tile position (2.0 world units per tile). Only modified tiles are stored.
    pub tiles: std::collections::HashMap<TilePos, Tile>,

    /// Active build seeds waiting for room growth automaton to process.
    pub build_seeds: Vec<BuildSeed>,

    /// 3D voxel world — chunked terrain with materials, the physical source of truth.
    pub voxel_world: super::voxel::VoxelWorld,

    /// Baked 2D walkable surfaces derived from voxel_world. One per settlement area.
    pub nav_grids: Vec<super::nav_grid::NavGrid>,

    /// Global economy (total gold supply, trade routes).
    pub economy: EconomyState,

    /// Emergent trade routes between settlements, established by repeated profitable trading.
    pub trade_routes: Vec<TradeRoute>,

    // --- Campaign-level collections (migrated from headless_campaign) ---

    /// Faction states (governments, guilds, cults, etc.).
    pub factions: Vec<FactionState>,

    /// Active quests currently being pursued.
    pub quests: Vec<Quest>,

    /// Quest board — available quests not yet accepted.
    pub quest_board: Vec<QuestPosting>,

    /// Bond graph between entities. Key = (min_id, max_id), value = strength 0–100.
    pub adventurer_bonds: HashMap<(u32, u32), f32>,

    /// Player guild state.
    pub guild: GuildState,

    /// Narrative chronicle log (bounded ring buffer).
    pub chronicle: Vec<ChronicleEntry>,
    /// World prophecies — generated at init, fulfilled by world events.
    pub prophecies: Vec<super::systems::prophecy::Prophecy>,

    /// Relation graph between entities. Key = (entity_a, entity_b, kind), value = strength.
    pub relations: HashMap<(u32, u32, u8), f32>,

    /// World events log (recent events for system queries, bounded).
    pub world_events: Vec<WorldEvent>,

    /// When true, WorldSim::new skips resource node spawning (for building AI scenarios).
    #[serde(default)]
    pub skip_resource_init: bool,
}

impl WorldState {
    pub fn new(seed: u64) -> Self {
        Self {
            tick: 0,
            rng_state: seed,
            next_id: 0,
            entities: Vec::new(),
            hot: Vec::new(),
            cold: Vec::new(),
            entity_index: Vec::new(),
            max_entity_id: 0,
            group_index: GroupIndex::default(),
            fidelity_zones: Vec::new(),
            regions: Vec::new(),
            settlements: Vec::new(),
            settlement_index: Vec::new(),
            tiles: std::collections::HashMap::new(),
            build_seeds: Vec::new(),
            voxel_world: super::voxel::VoxelWorld::default(),
            nav_grids: Vec::new(),
            economy: EconomyState::default(),
            trade_routes: Vec::new(),
            factions: Vec::new(),
            quests: Vec::new(),
            quest_board: Vec::new(),
            adventurer_bonds: HashMap::new(),
            guild: GuildState::default(),
            chronicle: Vec::new(),
            prophecies: super::systems::prophecy::generate_prophecies(seed),
            relations: HashMap::new(),
            world_events: Vec::new(),
            skip_resource_init: false,
        }
    }

    /// Full rebuild: sort entities by group, then rebuild hot/cold/index.
    /// Call after structural changes (push, remove).
    pub fn rebuild_all_indices(&mut self) {
        self.rebuild_group_index();
        self.rebuild_entity_cache();
        // settlement_index is cheap and rarely stale — rebuild alongside entity cache.
        self.rebuild_settlement_index();
    }

    /// Ensure every settlement has a Treasury building. Call once after init.
    /// Creates a Treasury entity at the settlement center, with a large inventory
    /// pre-loaded from the settlement's stockpile and treasury gold.
    pub fn ensure_treasury_buildings(&mut self) {
        for si in 0..self.settlements.len() {
            if self.settlements[si].treasury_building_id.is_some() { continue; }

            let sid = self.settlements[si].id;
            let pos = self.settlements[si].pos;
            let stockpile = self.settlements[si].stockpile;
            let gold = self.settlements[si].treasury;

            let new_id = self.next_entity_id();
            let mut entity = Entity::new_building(new_id, pos);
            entity.building = Some(BuildingData {
                building_type: BuildingType::Treasury,
                settlement_id: Some(sid),
                grid_col: 64, // center of 128x128 grid
                grid_row: 64,
                footprint_w: 2, // Treasury is 2x2 at tier 1
                footprint_h: 2,
                tier: 1,
                room_seed: new_id as u64,
                rooms: BuildingType::Treasury.default_rooms(),
                residential_capacity: 0,
                work_capacity: BuildingType::Treasury.work_capacity(),
                resident_ids: Vec::new(),
                worker_ids: Vec::new(),
                construction_progress: 1.0,
                built_tick: 0,
                builder_id: None,
                temporary: false,
                ttl_ticks: None,
                name: format!("{} Treasury", self.settlements[si].name),
                storage: stockpile, // seed from existing stockpile
                storage_capacity: BuildingType::Treasury.storage_capacity(),
                owner_id: None,
                builder_modifiers: Vec::new(),
                owner_modifiers: Vec::new(),
                worker_class_ticks: Vec::new(),
                specialization_tag: None,
                specialization_strength: 0.0,
                specialization_name: String::new(),
                structural: None,
            });
            // Treasury inventory mirrors settlement stockpile + gold.
            let mut inv = Inventory::with_capacity(500.0);
            inv.commodities = stockpile;
            inv.gold = gold;
            entity.inventory = Some(inv);
            self.entities.push(entity);
            self.settlements[si].treasury_building_id = Some(new_id);
        }
    }

    /// Rebuild hot/cold arrays and entity_index from `entities`.
    /// Does NOT re-sort. Call `rebuild_all_indices` if order may have changed.
    pub fn rebuild_entity_cache(&mut self) {
        let len = self.entities.len();
        self.hot.resize(len, HotEntity {
            id: 0, kind: EntityKind::Npc, team: WorldTeam::Neutral, alive: false,
            level: 0, pos: (0.0, 0.0), hp: 0.0, max_hp: 0.0,
            attack_damage: 0.0, attack_range: 0.0, move_speed: 0.0, grid_id: None,
        });
        self.cold.resize_with(len, ColdEntity::default);

        self.max_entity_id = self.entities.iter().map(|e| e.id).max().unwrap_or(0);
        let idx_len = self.max_entity_id as usize + 1;
        self.entity_index.resize(idx_len, u32::MAX);
        for v in &mut self.entity_index { *v = u32::MAX; }

        for (i, e) in self.entities.iter().enumerate() {
            self.hot[i] = HotEntity {
                id: e.id, kind: e.kind, team: e.team, alive: e.alive,
                level: e.level, pos: e.pos, hp: e.hp, max_hp: e.max_hp,
                attack_damage: e.attack_damage, attack_range: e.attack_range,
                move_speed: e.move_speed, grid_id: e.grid_id,
            };
            self.cold[i] = ColdEntity {
                shield_hp: e.shield_hp, armor: e.armor, magic_resist: e.magic_resist,
                local_pos: e.local_pos,
                status_effects: std::mem::take(&mut self.cold[i].status_effects),
                npc: std::mem::take(&mut self.cold[i].npc),
            };
            // Re-populate cold from entity (only on full rebuild).
            self.cold[i].status_effects.clear();
            self.cold[i].status_effects.extend_from_slice(&e.status_effects);
            self.cold[i].npc = e.npc.clone();

            if (e.id as usize) < idx_len {
                self.entity_index[e.id as usize] = i as u32;
            }
        }
    }

    /// Fast sync: update hot array from entities without touching cold or index.
    /// O(n) flat copy of scalar fields only. No allocations.
    pub fn sync_hot_from_entities(&mut self) {
        for i in 0..self.entities.len().min(self.hot.len()) {
            let e = &self.entities[i];
            let h = &mut self.hot[i];
            h.alive = e.alive;
            h.hp = e.hp;
            h.max_hp = e.max_hp;
            h.pos = e.pos;
            h.attack_damage = e.attack_damage;
            h.attack_range = e.attack_range;
            h.move_speed = e.move_speed;
            h.grid_id = e.grid_id;
            h.level = e.level;
            h.team = e.team;
        }
    }

    /// Sync `entities[i]` back from `hot[i]` and `cold[i]`.
    /// Call after mutating hot/cold arrays directly.
    pub fn sync_entity(&mut self, i: usize) {
        let h = &self.hot[i];
        let c = &self.cold[i];
        let e = &mut self.entities[i];
        e.id = h.id; e.kind = h.kind; e.team = h.team; e.alive = h.alive;
        e.level = h.level; e.pos = h.pos; e.hp = h.hp; e.max_hp = h.max_hp;
        e.attack_damage = h.attack_damage; e.attack_range = h.attack_range;
        e.move_speed = h.move_speed; e.grid_id = h.grid_id;
        e.shield_hp = c.shield_hp; e.armor = c.armor; e.magic_resist = c.magic_resist;
        e.local_pos = c.local_pos;
        // status_effects and npc are reference types — entities[i] keeps its own copy.
        // Only sync scalar fields.
    }

    /// O(1) entity lookup by ID using the secondary index.
    pub fn entity(&self, id: u32) -> Option<&Entity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.entities.len() {
                return Some(&self.entities[idx]);
            }
        }
        None
    }

    pub fn entity_mut(&mut self, id: u32) -> Option<&mut Entity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.entities.len() {
                return Some(&mut self.entities[idx]);
            }
        }
        None
    }

    /// O(1) lookup of entity's position in the `entities` vec by ID.
    pub fn entity_idx(&self, id: u32) -> Option<usize> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.entities.len() {
                return Some(idx);
            }
        }
        None
    }

    /// O(1) hot entity lookup by ID.
    pub fn hot_entity(&self, id: u32) -> Option<&HotEntity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.hot.len() {
                return Some(&self.hot[idx]);
            }
        }
        None
    }

    /// O(1) cold entity lookup by ID.
    pub fn cold_entity(&self, id: u32) -> Option<&ColdEntity> {
        let i = id as usize;
        if i < self.entity_index.len() {
            let idx = self.entity_index[i] as usize;
            if idx < self.cold.len() {
                return Some(&self.cold[idx]);
            }
        }
        None
    }

    /// Get the next unique entity ID (monotonic, deterministic).
    pub fn next_entity_id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }

    /// Advance the RNG and return a u32. Deterministic given the same seed + call sequence.
    pub fn next_rand_u32(&mut self) -> u32 {
        // PCG-style: multiply, add, shift.
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 33) as u32
    }

    /// Advance the RNG and return a float in [0, 1).
    pub fn next_rand(&mut self) -> f32 {
        self.next_rand_u32() as f32 / u32::MAX as f32
    }

    /// Sync next_id from current max entity ID (call after loading or external entity creation).
    pub fn sync_next_id(&mut self) {
        let max = self.entities.iter().map(|e| e.id).max().unwrap_or(0);
        if self.next_id <= max {
            self.next_id = max + 1;
        }
    }

    /// Rebuild the settlement_index for O(1) lookups by settlement ID.
    pub fn rebuild_settlement_index(&mut self) {
        let max_sid = self.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;
        self.settlement_index.clear();
        self.settlement_index.resize(max_sid, u32::MAX);
        for (i, s) in self.settlements.iter().enumerate() {
            let sid = s.id as usize;
            if sid < self.settlement_index.len() {
                self.settlement_index[sid] = i as u32;
            }
        }
    }

    /// O(1) settlement lookup by ID using the secondary index.
    /// Falls back to linear scan if index not built.
    pub fn settlement(&self, id: u32) -> Option<&SettlementState> {
        let i = id as usize;
        if i < self.settlement_index.len() {
            let idx = self.settlement_index[i] as usize;
            if idx < self.settlements.len() {
                return Some(&self.settlements[idx]);
            }
        }
        // Fallback for when index hasn't been built yet (tests, init).
        self.settlements.iter().find(|s| s.id == id)
    }

    /// O(1) mutable settlement lookup by ID.
    /// Falls back to linear scan if index not built.
    pub fn settlement_mut(&mut self, id: u32) -> Option<&mut SettlementState> {
        let i = id as usize;
        if i < self.settlement_index.len() {
            let idx = self.settlement_index[i] as usize;
            if idx < self.settlements.len() {
                return Some(&mut self.settlements[idx]);
            }
        }
        self.settlements.iter_mut().find(|s| s.id == id)
    }

    /// O(1) lookup of settlement's position in the `settlements` vec by ID.
    pub fn settlement_idx(&self, id: u32) -> Option<usize> {
        let i = id as usize;
        if i < self.settlement_index.len() {
            let idx = self.settlement_index[i] as usize;
            if idx < self.settlements.len() {
                return Some(idx);
            }
        }
        self.settlements.iter().position(|s| s.id == id)
    }

    pub fn fidelity_zone_mut(&mut self, id: u32) -> Option<&mut FidelityZone> {
        self.fidelity_zones.iter_mut().find(|g| g.id == id)
    }

    /// Get the treasury building entity ID for a settlement.
    pub fn treasury_entity_id(&self, settlement_id: u32) -> Option<u32> {
        self.settlement(settlement_id)
            .and_then(|s| s.treasury_building_id)
    }

    /// Find shortest path between two regions via BFS on the region graph.
    /// Returns the sequence of region IDs to traverse (excluding start, including end).
    pub fn find_region_path(&self, from_region: u32, to_region: u32) -> Option<Vec<u32>> {
        if from_region == to_region { return Some(vec![]); }

        let mut visited = vec![false; self.regions.len()];
        let mut parent = vec![u32::MAX; self.regions.len()];
        let mut queue = std::collections::VecDeque::new();

        let start_idx = self.regions.iter().position(|r| r.id == from_region)?;
        visited[start_idx] = true;
        queue.push_back(from_region);

        while let Some(current) = queue.pop_front() {
            let region = self.regions.iter().find(|r| r.id == current)?;
            for &neighbor_id in &region.neighbors {
                let ni = match self.regions.iter().position(|r| r.id == neighbor_id) {
                    Some(i) => i,
                    None => continue,
                };
                if visited[ni] { continue; }

                // Can't pass through impassable terrain.
                let neighbor = &self.regions[ni];
                if neighbor.terrain.travel_speed() <= 0.0 && neighbor_id != to_region { continue; }

                visited[ni] = true;
                parent[ni] = current;
                queue.push_back(neighbor_id);

                if neighbor_id == to_region {
                    // Reconstruct path.
                    let mut path = vec![to_region];
                    let mut c = to_region;
                    loop {
                        let ci = match self.regions.iter().position(|r| r.id == c) {
                            Some(i) => i,
                            None => break,
                        };
                        let p = parent[ci];
                        if p == u32::MAX || p == from_region { break; }
                        path.push(p);
                        c = p;
                    }
                    path.reverse();
                    return Some(path);
                }
            }
        }
        None // no path
    }

    pub fn fidelity_zone(&self, id: u32) -> Option<&FidelityZone> {
        self.fidelity_zones.iter().find(|g| g.id == id)
    }

    pub fn faction(&self, id: u32) -> Option<&FactionState> {
        self.factions.iter().find(|f| f.id == id)
    }

    pub fn faction_mut(&mut self, id: u32) -> Option<&mut FactionState> {
        self.factions.iter_mut().find(|f| f.id == id)
    }

    pub fn quest(&self, id: u32) -> Option<&Quest> {
        self.quests.iter().find(|q| q.id == id)
    }

    pub fn quest_mut(&mut self, id: u32) -> Option<&mut Quest> {
        self.quests.iter_mut().find(|q| q.id == id)
    }

    pub fn region_mut(&mut self, id: u32) -> Option<&mut RegionState> {
        self.regions.iter_mut().find(|r| r.id == id)
    }

    /// Look up bond strength between two entities.
    pub fn bond_strength(&self, a: u32, b: u32) -> f32 {
        if a == b { return 0.0; }
        let key = (a.min(b), a.max(b));
        self.adventurer_bonds.get(&key).copied().unwrap_or(0.0)
    }

    /// Look up a relation value between two entities.
    pub fn relation(&self, a: u32, b: u32, kind: RelationKind) -> f32 {
        self.relations.get(&(a, b, kind as u8)).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// GroupIndex — contiguous ranges for per-settlement / per-party iteration
// ---------------------------------------------------------------------------

/// Stores (start, end) ranges into the entity arrays for each group.
/// After `rebuild_group_index()`, entities are sorted by (settlement, kind, party).
/// `settlement_ranges[settlement_id] = (start, end)` means
/// `entities[start..end]` are all entities at that settlement.
///
/// Within each settlement range, entities are sub-grouped by kind:
/// NPCs first, then Buildings, then Monsters, then Items/Projectiles.
/// Per-kind sub-ranges are available via `settlement_npcs(sid)` etc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroupIndex {
    /// (start, end) index range per settlement_id — all entity kinds.
    pub settlement_ranges: Vec<(u32, u32)>,
    /// Per-settlement NPC sub-range. `settlement_npc_ranges[sid] = (start, end)`.
    pub settlement_npc_ranges: Vec<(u32, u32)>,
    /// Per-settlement Building sub-range.
    pub settlement_building_ranges: Vec<(u32, u32)>,
    /// Per-settlement Monster sub-range.
    pub settlement_monster_ranges: Vec<(u32, u32)>,
    /// (start, end) index range per party_id. 0 = no party.
    pub party_ranges: Vec<(u32, u32)>,
    /// Entities not assigned to any settlement (monsters, travelers).
    pub unaffiliated_range: (u32, u32),
}

impl GroupIndex {
    /// Iterate entity indices for a given settlement (all kinds).
    pub fn settlement_entities(&self, settlement_id: u32) -> std::ops::Range<usize> {
        let i = settlement_id as usize;
        if i < self.settlement_ranges.len() {
            let (start, end) = self.settlement_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate NPC entity indices for a given settlement.
    pub fn settlement_npcs(&self, settlement_id: u32) -> std::ops::Range<usize> {
        let i = settlement_id as usize;
        if i < self.settlement_npc_ranges.len() {
            let (start, end) = self.settlement_npc_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate Building entity indices for a given settlement.
    pub fn settlement_buildings(&self, settlement_id: u32) -> std::ops::Range<usize> {
        let i = settlement_id as usize;
        if i < self.settlement_building_ranges.len() {
            let (start, end) = self.settlement_building_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate Monster entity indices for a given settlement.
    pub fn settlement_monsters(&self, settlement_id: u32) -> std::ops::Range<usize> {
        let i = settlement_id as usize;
        if i < self.settlement_monster_ranges.len() {
            let (start, end) = self.settlement_monster_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate entity indices for a given party.
    pub fn party_entities(&self, party_id: u32) -> std::ops::Range<usize> {
        let i = party_id as usize;
        if i < self.party_ranges.len() {
            let (start, end) = self.party_ranges[i];
            start as usize..end as usize
        } else {
            0..0
        }
    }

    /// Iterate entity indices not belonging to any settlement.
    pub fn unaffiliated_entities(&self) -> std::ops::Range<usize> {
        self.unaffiliated_range.0 as usize..self.unaffiliated_range.1 as usize
    }
}

impl WorldState {
    /// Sort entities by (settlement_id, party_id) and rebuild all indices.
    ///
    /// After this call:
    /// - `entities`, `hot`, `cold` are sorted so settlement members are contiguous
    /// - `group_index.settlement_ranges` gives slice ranges per settlement
    /// - `group_index.settlement_npc_ranges` etc. give per-kind sub-ranges (post-scan)
    /// - `entity_index` is rebuilt for O(1) ID lookup into the new order
    ///
    /// Call at init and after structural changes (spawn/despawn).
    pub fn rebuild_group_index(&mut self) {
        let n = self.entities.len();
        if n == 0 { return; }

        // Build sort keys: (settlement_id, party_id, original_index).
        // Entities without a settlement sort to the end.
        let mut order: Vec<(u32, u32, usize)> = Vec::with_capacity(n);
        for (i, e) in self.entities.iter().enumerate() {
            let sid = e.settlement_id().unwrap_or(u32::MAX);
            let pid = e.npc.as_ref()
                .and_then(|npc| npc.party_id)
                .unwrap_or(u32::MAX);
            order.push((sid, pid, i));
        }
        order.sort_unstable();

        // Apply the permutation to entities, hot, cold.
        let mut new_entities = Vec::with_capacity(n);
        let mut new_hot = Vec::with_capacity(n);
        let mut new_cold = Vec::with_capacity(n);
        for &(_, _, old_i) in &order {
            new_entities.push(self.entities[old_i].clone());
            if old_i < self.hot.len() {
                new_hot.push(self.hot[old_i]);
            }
            if old_i < self.cold.len() {
                new_cold.push(self.cold[old_i].clone());
            }
        }
        self.entities = new_entities;
        self.hot = new_hot;
        self.cold = new_cold;

        // Rebuild entity_index.
        self.max_entity_id = self.entities.iter().map(|e| e.id).max().unwrap_or(0);
        let idx_len = self.max_entity_id as usize + 1;
        self.entity_index.resize(idx_len, u32::MAX);
        for v in &mut self.entity_index { *v = u32::MAX; }
        for (i, e) in self.entities.iter().enumerate() {
            if (e.id as usize) < idx_len {
                self.entity_index[e.id as usize] = i as u32;
            }
        }

        // Build settlement ranges and per-kind sub-ranges.
        let max_sid = self.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;
        self.group_index.settlement_ranges.clear();
        self.group_index.settlement_ranges.resize(max_sid, (0, 0));
        // Per-kind ranges resize deferred — not populated at current entity counts.

        let mut i = 0;
        while i < n {
            let sid = order[i].0;
            if sid == u32::MAX { break; } // rest are unaffiliated
            let start = i;
            while i < n && order[i].0 == sid { i += 1; }
            let sid_idx = sid as usize;
            if sid_idx < max_sid {
                self.group_index.settlement_ranges[sid_idx] = (start as u32, i as u32);
            }
        }
        self.group_index.unaffiliated_range = (i as u32, n as u32);

        // Build party ranges.
        // Scan entities for distinct party_ids and record ranges.
        let max_pid = self.entities.iter()
            .filter_map(|e| e.npc.as_ref().and_then(|n| n.party_id))
            .max()
            .unwrap_or(0) as usize + 1;
        self.group_index.party_ranges.clear();
        self.group_index.party_ranges.resize(max_pid, (0, 0));

        // Party ranges require a secondary sort within each settlement group.
        // Since we sorted by (sid, pid), entities with the same pid are contiguous
        // within each settlement. But the same pid could appear in multiple settlements
        // if parties span settlements. For now, just record first/last occurrence.
        let mut party_first = vec![u32::MAX; max_pid];
        let mut party_last = vec![0u32; max_pid];
        for (i, e) in self.entities.iter().enumerate() {
            if let Some(pid) = e.npc.as_ref().and_then(|n| n.party_id) {
                let p = pid as usize;
                if p < max_pid {
                    if party_first[p] == u32::MAX { party_first[p] = i as u32; }
                    party_last[p] = (i + 1) as u32;
                }
            }
        }
        for p in 0..max_pid {
            if party_first[p] != u32::MAX {
                self.group_index.party_ranges[p] = (party_first[p], party_last[p]);
            }
        }

        // Per-kind sub-ranges (settlement_npc_ranges etc.) are available in the API
        // but not populated by default — the kind-guard pattern in system loops is
        // faster at current entity counts (~2K). Enable per-kind sorting when entity
        // counts exceed ~5K where branch misprediction outweighs sort overhead.
    }

    /// Remove long-dead entities from the entity pool.
    /// Compacts dead Items, Buildings, and Projectiles whose references have been
    /// cleared. Dead NPCs and Monsters are kept for recycling by recruitment/monster_ecology.
    /// Call periodically (e.g., every 500 ticks) — the caller must call
    /// `rebuild_all_indices()` afterwards since entity indices change.
    pub fn compact_dead_entities(&mut self) {
        // Collect IDs of dead items and buildings so we can clear references first.
        let dead_item_ids: Vec<u32> = self.entities.iter()
            .filter(|e| !e.alive && matches!(e.kind, EntityKind::Item))
            .map(|e| e.id)
            .collect();
        let dead_building_ids: Vec<u32> = self.entities.iter()
            .filter(|e| !e.alive && matches!(e.kind, EntityKind::Building))
            .map(|e| e.id)
            .collect();

        // Clear NPC references to dead items and buildings.
        for entity in &mut self.entities {
            if let Some(npc) = &mut entity.npc {
                if let Some(wid) = npc.equipped_items.weapon_id {
                    if dead_item_ids.contains(&wid) { npc.equipped_items.weapon_id = None; }
                }
                if let Some(aid) = npc.equipped_items.armor_id {
                    if dead_item_ids.contains(&aid) { npc.equipped_items.armor_id = None; }
                }
                if let Some(aid) = npc.equipped_items.accessory_id {
                    if dead_item_ids.contains(&aid) { npc.equipped_items.accessory_id = None; }
                }
                if let Some(bid) = npc.work_building_id {
                    if dead_building_ids.contains(&bid) { npc.work_building_id = None; }
                }
                if let Some(bid) = npc.home_building_id {
                    if dead_building_ids.contains(&bid) { npc.home_building_id = None; }
                }
            }
        }

        // Clear settlement treasury refs.
        for settlement in &mut self.settlements {
            if let Some(tid) = settlement.treasury_building_id {
                if dead_building_ids.contains(&tid) { settlement.treasury_building_id = None; }
            }
        }

        // Remove dead Items, Buildings, and Projectiles.
        self.entities.retain(|e| {
            if e.alive { return true; }
            // Keep dead NPCs and Monsters — they get recycled by recruitment/monster_ecology.
            // Only compact dead Items, Buildings, and Projectiles.
            !matches!(e.kind, EntityKind::Item | EntityKind::Building | EntityKind::Projectile | EntityKind::Resource)
        });
        // Note: after compaction, rebuild_all_indices() must be called by the caller.
    }
}

// ---------------------------------------------------------------------------
// Entity — unified representation for NPCs, monsters, buildings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    Npc,
    Monster,
    Building,
    Projectile,
    Item,
    Resource,
}

// ---------------------------------------------------------------------------
// BuildingType — typed building kinds with capacity tables
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuildingType {
    House,
    Longhouse,
    Manor,
    Farm,
    Mine,
    Sawmill,
    Forge,
    Workshop,
    Apothecary,
    Market,
    Warehouse,
    Inn,
    TradePost,
    GuildHall,
    Temple,
    Barracks,
    Watchtower,
    Library,
    CourtHouse,
    Wall,
    Gate,
    Well,
    Tent,
    Camp,
    Shrine,
    /// Settlement treasury — holds settlement gold and commodity reserves.
    /// One per settlement. Target for rogue sabotage, heists.
    Treasury,
}

impl BuildingType {
    /// Construction cost: (wood, iron) required from settlement stockpile.
    pub fn build_cost(&self) -> (f32, f32) {
        match self {
            Self::Tent | Self::Camp => (1.0, 0.0),
            Self::House => (5.0, 0.0), // wood-only cabin
            Self::Longhouse | Self::Manor => (10.0, 2.0),
            Self::Farm | Self::Well => (3.0, 0.0),
            Self::Mine => (4.0, 3.0),
            Self::Sawmill => (6.0, 2.0),
            Self::Forge => (5.0, 5.0),
            Self::Workshop | Self::Apothecary => (4.0, 2.0),
            Self::Market | Self::TradePost => (8.0, 3.0),
            Self::Warehouse => (10.0, 4.0),
            Self::Inn => (8.0, 2.0),
            Self::GuildHall | Self::CourtHouse => (15.0, 8.0),
            Self::Temple | Self::Library => (12.0, 5.0),
            Self::Barracks => (10.0, 6.0),
            Self::Watchtower => (6.0, 4.0),
            Self::Wall | Self::Gate => (3.0, 2.0),
            Self::Shrine => (4.0, 1.0),
            Self::Treasury => (15.0, 10.0),
        }
    }

    /// Base residential capacity (how many people can LIVE here).
    pub fn residential_capacity(&self) -> u8 {
        match self {
            Self::House => 4,
            Self::Longhouse => 8,
            Self::Manor => 2,
            Self::Farm => 0,
            Self::Mine => 0,
            Self::Sawmill => 0,
            Self::Forge => 0,
            Self::Workshop => 0,
            Self::Apothecary => 1,
            Self::Market => 0,
            Self::Warehouse => 0,
            Self::Inn => 6,
            Self::TradePost => 0,
            Self::GuildHall => 2,
            Self::Temple => 2,
            Self::Barracks => 12,
            Self::Watchtower => 2,
            Self::Library => 0,
            Self::CourtHouse => 0,
            Self::Wall => 0,
            Self::Gate => 0,
            Self::Well => 0,
            Self::Tent => 4,
            Self::Camp => 8,
            Self::Shrine => 0,
            Self::Treasury => 0,
        }
    }

    /// Base work capacity (how many people can WORK here).
    pub fn work_capacity(&self) -> u8 {
        match self {
            Self::House => 0,
            Self::Longhouse => 0,
            Self::Manor => 4,
            Self::Farm => 6,
            Self::Mine => 4,
            Self::Sawmill => 3,
            Self::Forge => 3,
            Self::Workshop => 2,
            Self::Apothecary => 2,
            Self::Market => 4,
            Self::Warehouse => 2,
            Self::Inn => 2,
            Self::TradePost => 2,
            Self::GuildHall => 6,
            Self::Temple => 3,
            Self::Barracks => 0,
            Self::Watchtower => 1,
            Self::Library => 3,
            Self::CourtHouse => 4,
            Self::Wall => 0,
            Self::Gate => 1,
            Self::Well => 0,
            Self::Tent => 0,
            Self::Camp => 2,
            Self::Shrine => 1,
            Self::Treasury => 4,
        }
    }

    /// Base storage capacity for this building type.
    pub fn storage_capacity(&self) -> f32 {
        match self {
            Self::Warehouse => 200.0,
            Self::Market => 100.0,
            Self::Inn => 50.0,
            Self::TradePost => 80.0,
            Self::GuildHall => 60.0,
            Self::Farm => 30.0,     // small barn storage
            Self::Mine => 30.0,     // ore pile
            Self::Sawmill => 30.0,  // lumber yard
            Self::Forge => 20.0,    // material rack
            Self::Apothecary => 20.0,
            Self::Treasury => 500.0,  // main settlement reserve
            _ => 0.0, // houses, temples, etc. don't store commodities
        }
    }

    /// Tag hashes relevant to production at this building type.
    /// Used for class→building affinity scoring (Phase 6 production quality).
    pub fn production_tags(&self) -> &'static [(u32, f32)] {
        match self {
            Self::Farm => &[(tags::FARMING, 1.0), (tags::LABOR, 0.3)],
            Self::Mine => &[(tags::MINING, 1.0), (tags::ENDURANCE, 0.3)],
            Self::Sawmill => &[(tags::WOODWORK, 1.0), (tags::LABOR, 0.3)],
            Self::Forge | Self::Workshop => &[(tags::SMITHING, 0.5), (tags::CRAFTING, 0.5), (tags::LABOR, 0.2)],
            Self::Apothecary => &[(tags::HERBALISM, 0.6), (tags::MEDICINE, 0.4)],
            Self::Market | Self::TradePost => &[(tags::TRADE, 0.6), (tags::NEGOTIATION, 0.4)],
            Self::Library => &[(tags::RESEARCH, 0.5), (tags::LORE, 0.5)],
            Self::Temple | Self::Shrine => &[(tags::FAITH, 0.6), (tags::MEDICINE, 0.2), (tags::DISCIPLINE, 0.2)],
            Self::Barracks => &[(tags::COMBAT, 0.4), (tags::DISCIPLINE, 0.3), (tags::TACTICS, 0.3)],
            _ => &[], // non-production buildings
        }
    }
}

// ---------------------------------------------------------------------------
// BuildingEffect / BuildingModifier / BuildingData
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildingEffect {
    BonusHp(f32),
    BonusArmor(f32),
    BonusResCapacity(u8),
    BonusWorkCapacity(u8),
    BonusTier(u8),
    ThreatReduction(f32),
    ConstructionSpeed(f32),
}

/// A modifier applied to a building by a builder's or owner's ability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingModifier {
    pub source_ability: String,
    pub source_entity_id: u32,
    pub effect: BuildingEffect,
}

/// A functional room within a building interior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    /// What this room is used for.
    pub kind: RoomKind,
    /// Local position offset within the building (relative to building pos).
    pub offset: (f32, f32),
    /// Who is currently occupying this room (entity ID), if anyone.
    pub occupant_id: Option<u32>,
}

/// Functional room types within buildings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomKind {
    /// Sleeping quarters.
    Bedroom,
    /// Food preparation / eating area.
    Kitchen,
    /// Central gathering / warming area.
    Hearth,
    /// Crafting / manufacturing station.
    Workshop,
    /// Storage room for goods.
    Storeroom,
    /// Study / research area.
    Study,
    /// Prayer / meditation space.
    Shrine,
    /// Training / sparring area.
    TrainingYard,
    /// Market stall / counter.
    Counter,
    /// Entrance / common area.
    Entrance,
}

impl BuildingType {
    /// Generate the default room layout for this building type.
    /// Rooms have local position offsets for NPC placement within the building.
    pub fn default_rooms(&self) -> Vec<Room> {
        match self {
            BuildingType::House => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.5), occupant_id: None },
                Room { kind: RoomKind::Hearth, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (0.5, -0.5), occupant_id: None },
            ],
            BuildingType::Longhouse => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 1.0), occupant_id: None },
                Room { kind: RoomKind::Hearth, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (-0.5, -0.5), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (0.5, -0.5), occupant_id: None },
                Room { kind: RoomKind::Kitchen, offset: (0.0, -1.0), occupant_id: None },
            ],
            BuildingType::Inn => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 1.0), occupant_id: None },
                Room { kind: RoomKind::Hearth, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Kitchen, offset: (-0.5, -0.5), occupant_id: None },
                Room { kind: RoomKind::Counter, offset: (0.5, 0.5), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (0.5, -0.5), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (-0.5, -0.5), occupant_id: None },
            ],
            BuildingType::Forge => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.5), occupant_id: None },
                Room { kind: RoomKind::Workshop, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Storeroom, offset: (0.5, -0.5), occupant_id: None },
            ],
            BuildingType::Farm => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Storeroom, offset: (0.5, 0.0), occupant_id: None },
            ],
            BuildingType::Mine => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.5), occupant_id: None },
                Room { kind: RoomKind::Workshop, offset: (0.0, -0.5), occupant_id: None },
            ],
            BuildingType::Temple => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 1.0), occupant_id: None },
                Room { kind: RoomKind::Shrine, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Study, offset: (0.5, -0.5), occupant_id: None },
            ],
            BuildingType::Library => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.5), occupant_id: None },
                Room { kind: RoomKind::Study, offset: (-0.5, 0.0), occupant_id: None },
                Room { kind: RoomKind::Study, offset: (0.5, 0.0), occupant_id: None },
            ],
            BuildingType::Barracks => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 1.0), occupant_id: None },
                Room { kind: RoomKind::TrainingYard, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (-0.5, -0.5), occupant_id: None },
                Room { kind: RoomKind::Bedroom, offset: (0.5, -0.5), occupant_id: None },
            ],
            BuildingType::Market => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.5), occupant_id: None },
                Room { kind: RoomKind::Counter, offset: (-0.5, 0.0), occupant_id: None },
                Room { kind: RoomKind::Counter, offset: (0.5, 0.0), occupant_id: None },
                Room { kind: RoomKind::Storeroom, offset: (0.0, -0.5), occupant_id: None },
            ],
            BuildingType::GuildHall => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 1.0), occupant_id: None },
                Room { kind: RoomKind::Hearth, offset: (0.0, 0.0), occupant_id: None },
                Room { kind: RoomKind::Study, offset: (-0.5, -0.5), occupant_id: None },
                Room { kind: RoomKind::Workshop, offset: (0.5, -0.5), occupant_id: None },
            ],
            _ => vec![
                Room { kind: RoomKind::Entrance, offset: (0.0, 0.0), occupant_id: None },
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Building structural detail (integrates with building_ai::types)
// ---------------------------------------------------------------------------

use super::building_ai::types::{
    BuildMaterial, ConstructionMemory, Direction, FootprintShape,
    FoundationType, OpeningSpec, RoofType, WallComponentSpec,
};

/// Detailed structural specification for a building.
/// Added as an optional field on BuildingData so existing serialized data is unaffected.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildingStructural {
    pub shape: FootprintShape,
    pub material: BuildMaterial,
    pub roof_type: RoofType,
    pub roof_material: BuildMaterial,
    pub foundation_type: FoundationType,
    pub foundation_depth: u8,
    pub stories: u8,
    pub has_basement: bool,
    pub wall_specs: [WallComponentSpec; 4], // N, E, S, W
    pub openings: Vec<OpeningSpec>,
    pub defensive_integration: bool,
    pub expansion_stubs: Vec<Direction>,
}

/// Capability flags for enemy entities (jump, climb, tunnel, fly, siege).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct EnemyCapabilities {
    pub can_jump: bool,
    pub jump_height: u8,
    pub can_climb: bool,
    pub can_tunnel: bool,
    pub can_fly: bool,
    pub has_siege: bool,
    pub siege_damage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingData {
    pub building_type: BuildingType,
    /// None for wilderness buildings.
    pub settlement_id: Option<u32>,
    /// Position on settlement grid (top-left corner of footprint).
    pub grid_col: u16,
    pub grid_row: u16,
    /// Footprint width in grid cells (1 = single cell).
    pub footprint_w: u8,
    /// Footprint height in grid cells (1 = single cell).
    pub footprint_h: u8,
    /// 0-3 quality/upgrade level.
    pub tier: u8,
    /// Deterministic room_gen interior seed.
    pub room_seed: u64,
    /// Interior rooms — functional zones within this building.
    /// Derived from building type. NPCs enter specific rooms when working/eating/resting.
    pub rooms: Vec<Room>,

    // Capacity
    /// How many people can LIVE here.
    pub residential_capacity: u8,
    /// How many people can WORK here.
    pub work_capacity: u8,
    /// Entity IDs of NPCs living here.
    pub resident_ids: Vec<u32>,
    /// Entity IDs of NPCs working here.
    pub worker_ids: Vec<u32>,

    // State
    /// 0.0-1.0, 1.0 = complete.
    pub construction_progress: f32,
    pub built_tick: u64,
    /// Who built it.
    pub builder_id: Option<u32>,
    /// Tent, camp — decays after TTL.
    pub temporary: bool,
    /// Auto-destroy after this many ticks.
    pub ttl_ticks: Option<u64>,
    /// Procedural name ("The Iron Forge", "Korrin's House").
    pub name: String,

    // Storage
    /// Per-commodity inventory stored in this building.
    pub storage: [f32; super::NUM_COMMODITIES],
    /// Maximum total commodity units this building can hold.
    pub storage_capacity: f32,

    // Ability integration
    /// NPC who owns/operates this building.
    pub owner_id: Option<u32>,
    /// Permanent: applied at construction by builder's abilities.
    pub builder_modifiers: Vec<BuildingModifier>,
    /// Active: applied by current owner, fade if owner leaves.
    pub owner_modifiers: Vec<BuildingModifier>,

    // Specialization — emerges from worker class composition over time.
    /// Accumulated (class_name_hash, tick_count) pairs for workers at this building.
    pub worker_class_ticks: Vec<(u32, u64)>,
    /// Dominant class hash once specialization triggers (at 1000 accumulated ticks).
    pub specialization_tag: Option<u32>,
    /// 0.0-1.0 strength of specialization (grows with dominance ratio).
    pub specialization_strength: f32,
    /// Human-readable specialization name (e.g., "Miners' Forge").
    pub specialization_name: String,

    // Structural detail (building intelligence system)
    /// Detailed structural specification — wall composition, roof, foundation, etc.
    #[serde(default)]
    pub structural: Option<BuildingStructural>,
}

impl Default for BuildingData {
    fn default() -> Self {
        Self {
            building_type: BuildingType::House,
            settlement_id: None,
            grid_col: 0,
            grid_row: 0,
            footprint_w: 1,
            footprint_h: 1,
            tier: 0,
            room_seed: 0,
            rooms: BuildingType::House.default_rooms(),
            residential_capacity: BuildingType::House.residential_capacity(),
            work_capacity: BuildingType::House.work_capacity(),
            resident_ids: Vec::new(),
            worker_ids: Vec::new(),
            construction_progress: 0.0,
            built_tick: 0,
            builder_id: None,
            temporary: false,
            ttl_ticks: None,
            name: String::new(),
            storage: [0.0; super::NUM_COMMODITIES],
            storage_capacity: BuildingType::House.storage_capacity(),
            owner_id: None,
            builder_modifiers: Vec::new(),
            owner_modifiers: Vec::new(),
            worker_class_ticks: Vec::new(),
            specialization_tag: None,
            specialization_strength: 0.0,
            specialization_name: String::new(),
            structural: None,
        }
    }
}

impl BuildingData {
    /// Total commodity units currently stored.
    pub fn storage_used(&self) -> f32 {
        self.storage.iter().sum()
    }

    /// Available storage space remaining.
    pub fn storage_free(&self) -> f32 {
        (self.storage_capacity - self.storage_used()).max(0.0)
    }

    /// Try to deposit a commodity. Returns amount actually stored.
    pub fn deposit(&mut self, commodity: usize, amount: f32) -> f32 {
        if commodity >= self.storage.len() { return 0.0; }
        let space = self.storage_free();
        let stored = amount.min(space);
        self.storage[commodity] += stored;
        stored
    }

    /// Try to withdraw a commodity. Returns amount actually withdrawn.
    pub fn withdraw(&mut self, commodity: usize, amount: f32) -> f32 {
        if commodity >= self.storage.len() { return 0.0; }
        let available = self.storage[commodity];
        let taken = amount.min(available);
        self.storage[commodity] -= taken;
        taken
    }
}

/// Team affiliation for combat. NPCs on the same team don't attack each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldTeam {
    /// Player-aligned faction.
    Friendly,
    /// Hostile monsters / enemy faction.
    Hostile,
    /// Neutral (won't attack, can be attacked).
    Neutral,
}

impl Default for WorldTeam {
    fn default() -> Self {
        WorldTeam::Neutral
    }
}

/// Hot entity data — 48 bytes, cache-line friendly.
/// Iterated by every system every tick. Keep this small.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct HotEntity {
    pub id: u32,
    pub kind: EntityKind,
    pub team: WorldTeam,
    pub alive: bool,
    pub level: u32,
    pub pos: (f32, f32),
    pub hp: f32,
    pub max_hp: f32,
    pub attack_damage: f32,
    pub attack_range: f32,
    pub move_speed: f32,
    pub grid_id: Option<u32>,
}

/// Cold entity data — large, heap-heavy. Only accessed when a system
/// needs NPC-specific or detailed combat data. Indexed in parallel with HotEntity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColdEntity {
    pub shield_hp: f32,
    pub armor: f32,
    pub magic_resist: f32,
    pub local_pos: Option<(f32, f32)>,
    pub status_effects: Vec<StatusEffect>,
    pub npc: Option<NpcData>,
}

// ---------------------------------------------------------------------------
// Inventory — unified commodity + gold container on any entity
// ---------------------------------------------------------------------------

/// Generic inventory for any entity (NPC backpack, building warehouse, treasury).
/// Any entity with `inventory: Some(Inventory)` can participate in commodity transfers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    /// Per-commodity stock. Index by commodity constant (FOOD=0, IRON=1, etc.).
    pub commodities: [f32; super::NUM_COMMODITIES],
    /// Gold/currency held in this inventory.
    pub gold: f32,
    /// Maximum total commodity units. 0.0 = unlimited.
    pub capacity: f32,
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            commodities: [0.0; super::NUM_COMMODITIES],
            gold: 0.0,
            capacity: 0.0,
        }
    }
}

impl Inventory {
    /// Create an inventory with a specific capacity.
    pub fn with_capacity(capacity: f32) -> Self {
        Self { capacity, ..Default::default() }
    }

    /// Total commodity units currently stored.
    pub fn total_stored(&self) -> f32 {
        self.commodities.iter().sum()
    }

    /// Free space remaining. Returns f32::MAX if unlimited.
    pub fn free_space(&self) -> f32 {
        if self.capacity <= 0.0 { return f32::MAX; }
        (self.capacity - self.total_stored()).max(0.0)
    }

    /// Deposit a commodity. Returns amount actually stored (may be less if at capacity).
    pub fn deposit(&mut self, commodity: usize, amount: f32) -> f32 {
        if commodity >= self.commodities.len() || amount <= 0.0 { return 0.0; }
        let space = self.free_space();
        let stored = amount.min(space);
        self.commodities[commodity] += stored;
        stored
    }

    /// Withdraw a commodity. Returns amount actually withdrawn.
    pub fn withdraw(&mut self, commodity: usize, amount: f32) -> f32 {
        if commodity >= self.commodities.len() || amount <= 0.0 { return 0.0; }
        let available = self.commodities[commodity];
        let taken = amount.min(available);
        self.commodities[commodity] -= taken;
        taken
    }

    /// Transfer commodity from this inventory to another. Returns amount transferred.
    pub fn transfer_to(&mut self, other: &mut Inventory, commodity: usize, amount: f32) -> f32 {
        let withdrawn = self.withdraw(commodity, amount);
        if withdrawn <= 0.0 { return 0.0; }
        let deposited = other.deposit(commodity, withdrawn);
        // Return any excess that couldn't be deposited.
        if deposited < withdrawn {
            self.commodities[commodity] += withdrawn - deposited;
        }
        deposited
    }

    /// Transfer gold from this inventory to another. Returns amount transferred.
    pub fn transfer_gold_to(&mut self, other: &mut Inventory, amount: f32) -> f32 {
        let taken = amount.min(self.gold).max(0.0);
        self.gold -= taken;
        other.gold += taken;
        taken
    }
}

/// Combined view of an entity for backward-compatible access.
/// This is NOT stored — it's constructed on demand from hot + cold refs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    pub team: WorldTeam,
    pub pos: (f32, f32),
    pub grid_id: Option<u32>,
    pub local_pos: Option<(f32, f32)>,
    pub alive: bool,
    pub hp: f32,
    pub max_hp: f32,
    pub shield_hp: f32,
    pub armor: f32,
    pub magic_resist: f32,
    pub attack_damage: f32,
    pub attack_range: f32,
    pub move_speed: f32,
    pub level: u32,
    pub status_effects: Vec<StatusEffect>,
    pub npc: Option<NpcData>,
    pub building: Option<BuildingData>,
    pub item: Option<ItemData>,
    /// Physical resource node data (trees, ore veins, herb patches, etc.).
    pub resource: Option<ResourceData>,
    /// Unified inventory — commodities + gold. Any entity can have one.
    pub inventory: Option<Inventory>,
    /// Current movement target — cleared on arrival. Set by other systems;
    /// only the unified movement system actually moves entities.
    pub move_target: Option<(f32, f32)>,
    /// Speed multiplier (default 1.0). Stacks with `move_speed`.
    pub move_speed_mult: f32,

    // Building intelligence integration
    /// Enemy capability flags (jump, climb, tunnel, fly, siege).
    #[serde(default)]
    pub enemy_capabilities: Option<EnemyCapabilities>,
}

impl Entity {
    /// Returns the settlement this entity belongs to, checking NPC data first,
    /// then building data, then item data.
    pub fn settlement_id(&self) -> Option<u32> {
        self.npc.as_ref().and_then(|n| n.home_settlement_id)
            .or_else(|| self.building.as_ref().and_then(|b| b.settlement_id))
            .or_else(|| self.item.as_ref().and_then(|i| i.settlement_id))
    }

    /// Split into hot + cold components.
    pub fn split(&self) -> (HotEntity, ColdEntity) {
        (
            HotEntity {
                id: self.id, kind: self.kind, team: self.team, alive: self.alive,
                level: self.level, pos: self.pos, hp: self.hp, max_hp: self.max_hp,
                attack_damage: self.attack_damage, attack_range: self.attack_range,
                move_speed: self.move_speed, grid_id: self.grid_id,
            },
            ColdEntity {
                shield_hp: self.shield_hp, armor: self.armor, magic_resist: self.magic_resist,
                local_pos: self.local_pos, status_effects: self.status_effects.clone(),
                npc: self.npc.clone(),
            },
        )
    }

    /// Reconstruct from hot + cold.
    pub fn from_parts(hot: &HotEntity, cold: &ColdEntity) -> Self {
        Entity {
            id: hot.id, kind: hot.kind, team: hot.team, pos: hot.pos,
            grid_id: hot.grid_id, local_pos: cold.local_pos, alive: hot.alive,
            hp: hot.hp, max_hp: hot.max_hp, shield_hp: cold.shield_hp,
            armor: cold.armor, magic_resist: cold.magic_resist,
            attack_damage: hot.attack_damage, attack_range: hot.attack_range,
            move_speed: hot.move_speed, level: hot.level,
            status_effects: cold.status_effects.clone(), npc: cold.npc.clone(),
            building: None, item: None, resource: None, inventory: None,
            move_target: None, move_speed_mult: 1.0, enemy_capabilities: None,
        }
    }

    pub fn new_npc(id: u32, pos: (f32, f32)) -> Self {
        Self {
            id,
            kind: EntityKind::Npc,
            team: WorldTeam::Friendly,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 100.0,
            max_hp: 100.0,
            shield_hp: 0.0,
            armor: 0.0,
            magic_resist: 0.0,
            attack_damage: 10.0,
            attack_range: 1.5,
            move_speed: 3.0,
            level: 1,
            status_effects: Vec::new(),
            npc: Some(NpcData::default()),
            building: None,
            item: None,
            resource: None,
            inventory: Some(Inventory::with_capacity(50.0)), // NPC backpack
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities: None,
        }
    }

    pub fn new_monster(id: u32, pos: (f32, f32), level: u32) -> Self {
        Self::new_monster_typed(id, pos, level, CreatureType::Territorial)
    }

    pub fn new_monster_typed(id: u32, pos: (f32, f32), level: u32, creature_type: CreatureType) -> Self {
        let mut npc_data = NpcData::default();
        npc_data.creature_type = creature_type;
        npc_data.personality = creature_type.personality();
        npc_data.home_den = Some(pos); // spawn position is home den
        Self {
            id,
            kind: EntityKind::Monster,
            team: WorldTeam::Hostile,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 50.0 + level as f32 * 20.0,
            max_hp: 50.0 + level as f32 * 20.0,
            shield_hp: 0.0,
            armor: 5.0,
            magic_resist: 0.0,
            attack_damage: 5.0 + level as f32 * 3.0,
            attack_range: 1.5,
            move_speed: 2.0,
            level,
            status_effects: Vec::new(),
            npc: Some(npc_data),
            building: None,
            item: None,
            resource: None,
            inventory: None,
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities: None,
        }
    }

    pub fn new_building(id: u32, pos: (f32, f32)) -> Self {
        Self {
            id,
            kind: EntityKind::Building,
            team: WorldTeam::Neutral,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 500.0,
            max_hp: 500.0,
            shield_hp: 0.0,
            armor: 20.0,
            magic_resist: 0.0,
            attack_damage: 0.0,
            attack_range: 0.0,
            move_speed: 0.0,
            level: 0,
            status_effects: Vec::new(),
            npc: None,
            building: None,
            item: None,
            resource: None,
            inventory: None, // set by caller based on building type
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities: None,
        }
    }

    pub fn new_item(id: u32, pos: (f32, f32), item_data: ItemData) -> Self {
        Self {
            id,
            kind: EntityKind::Item,
            team: WorldTeam::Neutral,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 1.0,
            max_hp: 1.0,
            shield_hp: 0.0,
            armor: 0.0,
            magic_resist: 0.0,
            attack_damage: 0.0,
            attack_range: 0.0,
            move_speed: 0.0,
            level: 0,
            status_effects: Vec::new(),
            npc: None,
            building: None,
            item: Some(item_data),
            resource: None,
            inventory: None,
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities: None,
        }
    }

    /// Create a new resource node entity at the given position.
    pub fn new_resource(id: u32, pos: (f32, f32), resource_data: ResourceData) -> Self {
        Self {
            id,
            kind: EntityKind::Resource,
            team: WorldTeam::Neutral,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 1.0,
            max_hp: 1.0,
            shield_hp: 0.0,
            armor: 0.0,
            magic_resist: 0.0,
            attack_damage: 0.0,
            attack_range: 0.0,
            move_speed: 0.0,
            level: 0,
            status_effects: Vec::new(),
            npc: None,
            building: None,
            item: None,
            resource: Some(resource_data),
            inventory: None,
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities: None,
        }
    }

    // -----------------------------------------------------------------------
    // Inventory accessors — read from entity.inventory (canonical source)
    // -----------------------------------------------------------------------

    /// Read a single commodity from the entity's inventory.
    /// Returns 0.0 if the entity has no inventory.
    #[inline]
    pub fn inv_commodity(&self, commodity: usize) -> f32 {
        self.inventory
            .as_ref()
            .map(|inv| inv.commodities[commodity])
            .unwrap_or(0.0)
    }

    /// Sum of all commodities in the entity's inventory.
    #[inline]
    pub fn inv_total_commodities(&self) -> f32 {
        self.inventory
            .as_ref()
            .map(|inv| inv.commodities.iter().sum())
            .unwrap_or(0.0)
    }

    /// Whether the entity carries any commodity above the given threshold.
    #[inline]
    pub fn inv_has_any_commodity(&self, threshold: f32) -> bool {
        self.inventory
            .as_ref()
            .map(|inv| inv.commodities.iter().any(|&g| g > threshold))
            .unwrap_or(false)
    }

    /// Sync building.storage FROM entity.inventory.
    pub fn sync_building_storage_from_inventory(&mut self) {
        if let (Some(bld), Some(inv)) = (&mut self.building, &self.inventory) {
            bld.storage = inv.commodities;
        }
    }

    /// Sync entity.inventory FROM building.storage.
    pub fn sync_inventory_from_building_storage(&mut self) {
        if let (Some(bld), Some(inv)) = (&self.building, &mut self.inventory) {
            inv.commodities = bld.storage;
        }
    }
}

// ---------------------------------------------------------------------------
// StatusEffect — live status effect on an entity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusEffect {
    pub kind: StatusEffectKind,
    pub source_id: u32,
    pub remaining_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatusEffectKind {
    Stun,
    Slow { factor: f32 },
    Root,
    Silence,
    Dot { damage_per_tick: f32, tick_interval_ms: u32, tick_elapsed_ms: u32 },
    Hot { heal_per_tick: f32, tick_interval_ms: u32, tick_elapsed_ms: u32 },
    Buff { stat: String, factor: f32 },
    Debuff { stat: String, factor: f32 },
}

impl StatusEffectKind {
    /// Discriminant for dedup: two effects with the same discriminant are the "same kind".
    pub fn discriminant(&self) -> u8 {
        match self {
            StatusEffectKind::Stun => 0,
            StatusEffectKind::Slow { .. } => 1,
            StatusEffectKind::Root => 2,
            StatusEffectKind::Silence => 3,
            StatusEffectKind::Dot { .. } => 4,
            StatusEffectKind::Hot { .. } => 5,
            StatusEffectKind::Buff { .. } => 6,
            StatusEffectKind::Debuff { .. } => 7,
        }
    }
}

// ---------------------------------------------------------------------------
// Needs — Maslow-inspired need levels for NPCs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Needs {
    pub hunger: f32,     // 0-100, rises over time
    pub safety: f32,     // 0-100, threat proximity
    pub shelter: f32,    // 0-100, housing quality
    pub social: f32,     // 0-100, recent interactions
    pub purpose: f32,    // 0-100, meaningful work
    pub esteem: f32,     // 0-100, achievements
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            hunger: 50.0,
            safety: 50.0,
            shelter: 50.0,
            social: 50.0,
            purpose: 50.0,
            esteem: 50.0,
        }
    }
}

impl Needs {
    /// Returns the name and urgency (0.0-1.0) of the most urgent need.
    /// Urgency = (100 - value) / 100.
    pub fn most_urgent(&self) -> (&str, f32) {
        let needs: [(&str, f32); 6] = [
            ("hunger", self.hunger),
            ("safety", self.safety),
            ("shelter", self.shelter),
            ("social", self.social),
            ("purpose", self.purpose),
            ("esteem", self.esteem),
        ];
        let (name, value) = needs.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        (name, (100.0 - value) / 100.0)
    }
}

// ---------------------------------------------------------------------------
// Memory — event log + semantic beliefs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemEventType {
    WasAttacked,
    AttackedEnemy,
    FriendDied(u32),
    WonFight,
    CompletedQuest,
    LearnedSkill,
    WasHealed,
    WasBetrayedBy(u32),
    TradedWith(u32),
    BuiltSomething,
    LostHome,
    WasRescuedBy(u32),
    Starved,
    FoundShelter,
    MadeNewFriend(u32),
    BecameApprentice(u32),
    CompletedApprenticeship(u32),
    TrainedApprentice(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub tick: u64,
    pub event_type: MemEventType,
    pub location: (f32, f32),
    pub entity_ids: Vec<u32>,
    pub emotional_impact: f32,  // -1.0 to +1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeliefType {
    LocationDangerous(u32),    // settlement_id or region hash
    LocationSafe(u32),
    EntityTrustworthy(u32),
    EntityDangerous(u32),
    SettlementProsperous(u32),
    SettlementPoor(u32),
    SkillValuable(u32),        // tag hash
    FactionFriendly(u32),
    FactionHostile(u32),
    /// Personal grudge against an entity who killed a friend.
    Grudge(u32),
    /// Heard a story about an entity's deed.
    HeardStory { about: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub belief_type: BeliefType,
    pub confidence: f32,       // 0.0-1.0
    pub formed_tick: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Memory {
    pub events: std::collections::VecDeque<MemoryEvent>,
    pub beliefs: Vec<Belief>,
}

impl Memory {
    /// Record a new event, capping the ring buffer at 20 entries.
    pub fn record_event(&mut self, event: MemoryEvent) {
        self.events.push_back(event);
        while self.events.len() > 20 {
            self.events.pop_front();
        }
    }

    /// Check if a belief of the given type exists, returning its confidence if so.
    pub fn has_belief(&self, bt: &BeliefType) -> Option<f32> {
        self.beliefs.iter().find(|b| belief_type_matches(&b.belief_type, bt)).map(|b| b.confidence)
    }
}

/// Structural equality for BeliefType (same variant + same inner value).
fn belief_type_matches(a: &BeliefType, b: &BeliefType) -> bool {
    match (a, b) {
        (BeliefType::LocationDangerous(x), BeliefType::LocationDangerous(y)) => x == y,
        (BeliefType::LocationSafe(x), BeliefType::LocationSafe(y)) => x == y,
        (BeliefType::EntityTrustworthy(x), BeliefType::EntityTrustworthy(y)) => x == y,
        (BeliefType::EntityDangerous(x), BeliefType::EntityDangerous(y)) => x == y,
        (BeliefType::SettlementProsperous(x), BeliefType::SettlementProsperous(y)) => x == y,
        (BeliefType::SettlementPoor(x), BeliefType::SettlementPoor(y)) => x == y,
        (BeliefType::SkillValuable(x), BeliefType::SkillValuable(y)) => x == y,
        (BeliefType::FactionFriendly(x), BeliefType::FactionFriendly(y)) => x == y,
        (BeliefType::FactionHostile(x), BeliefType::FactionHostile(y)) => x == y,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Personality — experience-shaped behavioral tendencies
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Personality {
    pub risk_tolerance: f32,    // 0-1, starts 0.5
    pub social_drive: f32,      // 0-1, starts 0.5
    pub ambition: f32,          // 0-1, starts 0.5
    pub compassion: f32,        // 0-1, starts 0.5
    pub curiosity: f32,         // 0-1, starts 0.5
}

impl Default for Personality {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            social_drive: 0.5,
            ambition: 0.5,
            compassion: 0.5,
            curiosity: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// CreatureType — differentiates agent archetypes for unified AI
// ---------------------------------------------------------------------------

/// Creature archetype controlling personality presets and allowed action sets.
/// NPCs default to `Citizen`. Monsters get a type at spawn time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreatureType {
    /// Standard NPC — full action space.
    Citizen,
    /// Pack predator (wolves) — high social drive, clusters with pack leader.
    PackPredator,
    /// Territorial predator (bears) — defends den, solitary.
    Territorial,
    /// Raider (goblins) — high ambition, can loot.
    Raider,
    /// Megabeast (dragons) — near-zero personality, never flees.
    Megabeast,
}

impl Default for CreatureType {
    fn default() -> Self { CreatureType::Citizen }
}

impl CreatureType {
    /// Default personality for this creature type.
    pub fn personality(self) -> Personality {
        match self {
            CreatureType::Citizen => Personality::default(),
            CreatureType::PackPredator => Personality {
                ambition: 0.3, risk_tolerance: 0.6, compassion: 0.0,
                social_drive: 0.9, curiosity: 0.4,
            },
            CreatureType::Territorial => Personality {
                ambition: 0.2, risk_tolerance: 0.7, compassion: 0.0,
                social_drive: 0.1, curiosity: 0.2,
            },
            CreatureType::Raider => Personality {
                ambition: 0.8, risk_tolerance: 0.5, compassion: 0.1,
                social_drive: 0.4, curiosity: 0.3,
            },
            CreatureType::Megabeast => Personality {
                ambition: 0.1, risk_tolerance: 0.95, compassion: 0.0,
                social_drive: 0.0, curiosity: 0.1,
            },
        }
    }

    /// Hunger drain multiplier for this creature type.
    pub fn hunger_drain_mult(self) -> f32 {
        match self {
            CreatureType::Citizen => 1.0,
            CreatureType::PackPredator => 3.0,
            CreatureType::Territorial => 2.0,
            CreatureType::Raider => 1.5,
            CreatureType::Megabeast => 0.5,
        }
    }

    /// Whether this creature type can flee (megabeasts never flee).
    pub fn can_flee(self) -> bool {
        !matches!(self, CreatureType::Megabeast)
    }

    /// HP ratio threshold at which this creature flees. Returns None if never flees.
    pub fn flee_hp_threshold(self) -> Option<f32> {
        match self {
            CreatureType::Citizen => Some(0.3),
            CreatureType::PackPredator => Some(0.3),
            CreatureType::Territorial => Some(0.25),
            CreatureType::Raider => Some(0.25),
            CreatureType::Megabeast => None,
        }
    }

    /// Whether this creature type can use economy actions (trade, build, work, harvest).
    pub fn has_economy_actions(self) -> bool {
        matches!(self, CreatureType::Citizen | CreatureType::Raider)
    }

    /// Whether this creature type can build/place blueprints.
    pub fn can_build(self) -> bool {
        matches!(self, CreatureType::Citizen)
    }
}

// ---------------------------------------------------------------------------
// Emotions — transient emotional state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Emotions {
    pub joy: f32,       // 0-1
    pub anger: f32,     // 0-1
    pub fear: f32,      // 0-1
    pub grief: f32,     // 0-1
    pub pride: f32,     // 0-1
    pub anxiety: f32,   // 0-1
}

impl Default for Emotions {
    fn default() -> Self {
        Self {
            joy: 0.0,
            anger: 0.0,
            fear: 0.0,
            grief: 0.0,
            pride: 0.0,
            anxiety: 0.0,
        }
    }
}

impl Emotions {
    /// Decay all emotions toward 0 by the given rate.
    pub fn decay(&mut self, rate: f32) {
        self.joy = (self.joy - rate).max(0.0);
        self.anger = (self.anger - rate).max(0.0);
        self.fear = (self.fear - rate).max(0.0);
        self.grief = (self.grief - rate).max(0.0);
        self.pride = (self.pride - rate).max(0.0);
        self.anxiety = (self.anxiety - rate).max(0.0);
    }

    /// Returns the name and value of the strongest (dominant) emotion.
    pub fn dominant(&self) -> (&str, f32) {
        let emotions: [(&str, f32); 6] = [
            ("joy", self.joy),
            ("anger", self.anger),
            ("fear", self.fear),
            ("grief", self.grief),
            ("pride", self.pride),
            ("anxiety", self.anxiety),
        ];
        emotions.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
    }
}

// ---------------------------------------------------------------------------
// Aspiration — medium-term behavioral orientation (Phase B)
// ---------------------------------------------------------------------------

/// Number of need dimensions for aspiration vectors.
pub const NUM_NEEDS: usize = 6; // hunger, safety, shelter, social, purpose, esteem

/// Persistent medium-term orientation biasing NPC behavior over hundreds of ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aspiration {
    /// Personality-weighted need gap vector, normalized. Recomputed every 500 ticks.
    pub need_vector: [f32; NUM_NEEDS],
    /// Tick when need_vector was last recomputed.
    pub vector_formed_at: u64,
    /// Optional concrete target crystallized from a salient event.
    pub crystal: Option<Crystal>,
    /// Progress toward crystal completion [0.0, 1.0].
    pub crystal_progress: f32,
    /// Tick when crystal progress last increased.
    pub crystal_last_advanced: u64,
}

impl Default for Aspiration {
    fn default() -> Self {
        Self {
            need_vector: [0.0; NUM_NEEDS],
            vector_formed_at: 0,
            crystal: None,
            crystal_progress: 0.0,
            crystal_last_advanced: 0,
        }
    }
}

/// A concrete target bound to an aspiration need dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crystal {
    /// Which need dimension this crystal addresses (index into need_vector).
    pub need_idx: usize,
    /// What the NPC is concretely pursuing.
    pub target: CrystalTarget,
    /// Tick when the crystal formed.
    pub formed_at: u64,
}

/// Target types for aspiration crystals — enum over ID types, not goal types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrystalTarget {
    /// Pursue a relationship with or emulate this entity.
    Entity(u32),
    /// Pursue this class (by class name hash).
    Class(u32),
    /// Build or work at this building.
    Building(u32),
    /// Reach this location.
    Location((f32, f32), f32),
    /// Harvest/find this resource type.
    Resource(ResourceType),
}

// ---------------------------------------------------------------------------
// Relationship — per-pair NPC relationship tracking
// ---------------------------------------------------------------------------

/// Directional relationship from one NPC toward another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Trust level: [-1.0, 1.0]. Starts at 0.0, positive = friendly, negative = hostile.
    pub trust: f32,
    /// Familiarity: [0.0, 1.0]. Grows with proximity and interaction.
    pub familiarity: f32,
    /// Tick of last meaningful interaction (trade, co-construction, combat, etc.).
    pub last_interaction: u64,
    /// Mental model of the other NPC's personality (Phase D: theory of mind).
    pub perceived_personality: PerceivedPersonality,
}

impl Relationship {
    pub fn new() -> Self {
        Self {
            trust: 0.0,
            familiarity: 0.0,
            last_interaction: 0,
            perceived_personality: PerceivedPersonality::default(),
        }
    }
}

/// Number of personality traits tracked in perceived personality.
pub const NUM_TRAITS: usize = 5; // risk_tolerance, social_drive, ambition, compassion, curiosity

/// An NPC's mental model of another NPC's personality, built from observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceivedPersonality {
    /// Estimated trait values [0.0, 1.0]. Indices: 0=risk_tolerance, 1=social_drive,
    /// 2=ambition, 3=compassion, 4=curiosity.
    pub traits: [f32; NUM_TRAITS],
    /// Per-trait confidence [0.0, 1.0]. Higher = more observations.
    pub confidence: [f32; NUM_TRAITS],
    /// Total number of action observations.
    pub observation_count: u32,
}

impl Default for PerceivedPersonality {
    fn default() -> Self {
        Self {
            traits: [0.5; NUM_TRAITS], // neutral assumption
            confidence: [0.0; NUM_TRAITS],
            observation_count: 0,
        }
    }
}

impl PerceivedPersonality {
    /// Update perception from an observed action. Authority figures are
    /// observed more closely (higher alpha).
    pub fn observe_action(&mut self, action_type: u8, authority: f32) {
        let attention = 1.0 + authority;
        // Action→trait signal mapping.
        let signals: &[(usize, f32)] = match action_type {
            3 => &[(2, 0.7)],                  // Working → ambition up
            8 => &[(3, 0.6), (2, 0.4)],        // Building → compassion + ambition
            5 => &[(0, 0.8)],                   // Fighting → risk tolerance up
            9 => &[(0, 0.2)],                   // Fleeing → risk tolerance down
            10 => &[(1, 0.7)],                  // Trading → social drive up
            0 => &[(2, 0.3)],                   // Idle → ambition down
            11 => &[(4, 0.5)],                  // Harvesting → curiosity (varied resources)
            _ => &[],
        };
        for &(trait_idx, signal) in signals {
            let base_alpha = 0.05 / (1.0 + self.confidence[trait_idx]);
            let alpha = base_alpha * attention;
            self.traits[trait_idx] += alpha * (signal - self.traits[trait_idx]);
            self.confidence[trait_idx] = (self.confidence[trait_idx] + 0.01 * attention).min(1.0);
        }
        self.observation_count += 1;
    }

    /// Perceived compatibility with own personality (0.0 = opposite, 1.0 = identical).
    pub fn compatibility(&self, own: &Personality) -> f32 {
        let own_arr = [own.risk_tolerance, own.social_drive, own.ambition, own.compassion, own.curiosity];
        let mut sum_diff = 0.0f32;
        for i in 0..NUM_TRAITS {
            sum_diff += (own_arr[i] - self.traits[i]).abs();
        }
        1.0 - sum_diff / NUM_TRAITS as f32
    }
}

// ---------------------------------------------------------------------------
// WorldAbility — parsed ability for active use in world sim
// ---------------------------------------------------------------------------

/// A parsed ability that can be actively used by NPCs in the world simulation
/// outside of combat. Extracted from class_tags DSL at level-up time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldAbility {
    /// Display name.
    pub name: String,
    /// What this ability does in the world sim.
    pub effect: WorldAbilityEffect,
    /// Cooldown in ticks between uses.
    pub cooldown_ticks: u32,
    /// Tick when last activated.
    pub last_used: u64,
    /// Duration of ongoing effect (0 = instant).
    pub duration_ticks: u32,
    /// Is this currently active (for duration-based effects)?
    pub active_until: u64,
}

/// Concrete world sim effects that abilities can produce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldAbilityEffect {
    /// Boost morale of nearby NPCs (rally, inspire, rallying_cry).
    Rally { morale_restore: f32, radius: f32 },
    /// Boost production yield (golden_touch, trade_empire).
    GoldenTouch { yield_mult: f32 },
    /// Reduce threat in area (fortify, sanctuary).
    Fortify { safety_bonus: f32, radius: f32 },
    /// Reveal resources in large radius (reveal, prophetic_vision, all_seeing_eye).
    Reveal { radius: f32, count: u32 },
    /// Manipulate local trade prices (corner_market, trade_embargo).
    CornerMarket { price_mult: f32 },
    /// Boost nearby NPCs' work speed (field_command, war_cry).
    FieldCommand { work_speed_mult: f32, radius: f32 },
    /// Generate passive income (forge_trade_route, passive_income).
    PassiveIncome { gold_per_tick: f32 },
    /// Hide from threats (ghost_walk, hidden_camp, vanish).
    Stealth { duration_ticks: u32 },
    /// Boost diplomacy/trust (ceasefire, broker_alliance).
    Diplomacy { trust_bonus: f32 },
}

// ---------------------------------------------------------------------------
// PassiveEffects — ability-derived simulation bonuses (Phase F)
// ---------------------------------------------------------------------------

/// Passive bonuses from class abilities that affect the world simulation
/// outside of combat. Derived from behavior profile tag composition.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PassiveEffects {
    /// Production yield multiplier (stacks with Phase 6 quality). From economy tags.
    pub production_mult: f32,
    /// Trade ratio bonus. From TRADE/NEGOTIATION tags.
    pub trade_bonus: f32,
    /// Morale aura radius (0 = no aura). From LEADERSHIP tags.
    pub morale_aura_radius: f32,
    /// Safety aura radius (0 = no aura). From DEFENSE/TACTICS tags.
    pub safety_aura_radius: f32,
    /// Perception radius multiplier. From AWARENESS/STEALTH tags.
    pub perception_mult: f32,
}

impl PassiveEffects {
    /// Recompute from behavior profile. Called on class level-up.
    pub fn compute(behavior_profile: &[(u32, f32)]) -> Self {
        let lookup = |tag: u32| -> f32 {
            match behavior_profile.binary_search_by_key(&tag, |&(t, _)| t) {
                Ok(idx) => behavior_profile[idx].1,
                Err(_) => 0.0,
            }
        };

        let trade_val = lookup(tags::TRADE) + lookup(tags::NEGOTIATION) * 0.5;
        let leadership_val = lookup(tags::LEADERSHIP) + lookup(tags::DISCIPLINE) * 0.3;
        let defense_val = lookup(tags::DEFENSE) + lookup(tags::TACTICS) * 0.5;
        let awareness_val = lookup(tags::AWARENESS) + lookup(tags::STEALTH) * 0.3;
        let economy_val = lookup(tags::CRAFTING) + lookup(tags::SMITHING) + lookup(tags::FARMING);

        Self {
            production_mult: 1.0 + (economy_val / 200.0).min(0.5), // max +50%
            trade_bonus: (trade_val / 100.0).min(0.3),              // max +30%
            morale_aura_radius: if leadership_val > 50.0 { (leadership_val / 10.0).min(20.0) } else { 0.0 },
            safety_aura_radius: if defense_val > 50.0 { (defense_val / 10.0).min(15.0) } else { 0.0 },
            perception_mult: 1.0 + (awareness_val / 200.0).min(0.5), // max +50%
        }
    }
}

// ---------------------------------------------------------------------------
// PriceBelief — per-commodity value estimate for information asymmetry (Phase E)
// ---------------------------------------------------------------------------

/// An NPC's belief about the value of a commodity, formed from experience.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PriceBelief {
    /// Estimated value (abstract units). Higher = more valuable/scarce.
    pub estimated_value: f32,
    /// Confidence [0.0, 1.0]. Decays without updates.
    pub confidence: f32,
    /// Tick when last updated.
    pub last_updated: u64,
}

impl Default for PriceBelief {
    fn default() -> Self {
        Self { estimated_value: 1.0, confidence: 0.0, last_updated: 0 }
    }
}

impl PriceBelief {
    /// Update from direct observation (production or consumption).
    pub fn update_direct(&mut self, observed_value: f32, tick: u64) {
        let alpha = 0.3;
        self.estimated_value = self.estimated_value * (1.0 - alpha) + observed_value * alpha;
        self.confidence = 1.0;
        self.last_updated = tick;
    }

    /// Update from trade experience.
    pub fn update_trade(&mut self, trade_value: f32, tick: u64) {
        let alpha = 0.2;
        self.estimated_value = self.estimated_value * (1.0 - alpha) + trade_value * alpha;
        self.confidence = (self.confidence + 0.3).min(1.0);
        self.last_updated = tick;
    }

    /// Update from secondhand information (gossip during trade).
    pub fn update_secondhand(&mut self, value: f32, teller_confidence: f32, tick: u64) {
        let alpha = 0.1 * teller_confidence;
        self.estimated_value = self.estimated_value * (1.0 - alpha) + value * alpha;
        self.confidence = (self.confidence + 0.05 * teller_confidence).min(1.0);
        self.last_updated = tick;
    }

    /// Decay confidence when not updated.
    pub fn decay(&mut self) {
        self.confidence *= 0.95;
    }
}

// ---------------------------------------------------------------------------
// OutcomeEMA — per-action-type exponential moving average of outcomes
// ---------------------------------------------------------------------------

/// Tracks success/failure history for an action type, enabling adaptive behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeEMA {
    /// Running average of outcome valence, [-1.0, 1.0].
    pub value: f32,
    /// Number of samples (first 10 learn faster).
    pub sample_count: u32,
}

impl OutcomeEMA {
    pub fn new() -> Self { Self { value: 0.0, sample_count: 0 } }

    /// Update with a new outcome valence (1.0=success, -1.0=failure, 0.0=neutral).
    pub fn update(&mut self, valence: f32) {
        let alpha = if self.sample_count < 10 { 0.3 } else { 0.1 };
        self.value = self.value * (1.0 - alpha) + valence * alpha;
        self.sample_count += 1;
    }

    /// Utility multiplier: maps [-1, 1] → [floor, 1.0].
    pub fn utility_mod(&self, floor: f32) -> f32 {
        floor + (1.0 - floor) * (self.value + 1.0) / 2.0
    }
}

// ---------------------------------------------------------------------------
// ResourceKnowledge — per-NPC discovered resource tracking
// ---------------------------------------------------------------------------

/// What an NPC knows about a resource node's location and type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceKnowledge {
    /// Last-known position.
    pub pos: (f32, f32),
    /// Resource type.
    pub resource_type: ResourceType,
    /// Tick when this NPC last observed the resource.
    pub observed_tick: u64,
}

// ---------------------------------------------------------------------------
// WorkState — spatial work state machine
// ---------------------------------------------------------------------------

/// State machine for NPC spatial work loop.
///
/// NPCs cycle through: Idle → TravelingToWork → Working → Idle.
/// On work completion, produced commodities go directly into the NPC's inventory.
/// (CarryingToStorage is a legacy variant kept for backward compatibility.)
/// The compute function emits movement/production deltas; the advance function
/// handles state transitions (called post-apply since it needs `&mut WorldState`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkState {
    Idle,
    TravelingToWork { target_pos: (f32, f32) },
    Working { building_id: u32, ticks_remaining: u16 },
    CarryingToStorage { commodity: u8, amount: f32, target_pos: (f32, f32) },
}

impl Default for WorkState {
    fn default() -> Self { WorkState::Idle }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Multi-step plan types for goal execution
// ---------------------------------------------------------------------------

/// A concrete action within a multi-step plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlannedAction {
    /// Walk to a specific position.
    MoveTo { target: (f32, f32) },
    /// Sell carried commodity at current location.
    Sell { commodity: usize },
    /// Buy commodity at current location.
    Buy { commodity: usize },
    /// Wait idle for N ticks.
    Wait { ticks: u16 },
    /// Engage in combat (fight hostiles in range).
    Fight,
    /// Perform a timed action at current location.
    Perform { activity: String, ticks: u16 },
    /// Pick up commodity from nearby building.
    Acquire { commodity: u8, amount: f32 },
    /// Deposit commodity at nearby building.
    Deposit { commodity: u8, amount: f32 },
    /// Pay gold.
    PayGold { amount: f32 },
    /// Gather a resource: walk to nearest resource node and harvest into inventory.
    Gather { commodity: u8, amount: f32 },
    /// Place a building on the city grid (requires resources in inventory).
    PlaceBuilding { building_type: u8 },
}

/// Execution status of a single plan step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    InProgress,
    Complete,
    Failed,
}

/// A single step in a multi-step plan attached to a Goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedStep {
    pub action: PlannedAction,
    pub status: StepStatus,
}

// Goal stack — prioritized NPC goal system with interruption/resumption
// ---------------------------------------------------------------------------

/// A single goal on an NPC's goal stack. Higher priority goals preempt lower ones.
/// When a goal completes or is abandoned, it pops and the next goal resumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// What the NPC is trying to do.
    pub kind: GoalKind,
    /// Priority (higher = more urgent). Hunger=90, Combat=80, Work=50, Social=30.
    pub priority: u8,
    /// Tick when this goal was pushed.
    pub started_tick: u64,
    /// Progress toward completion (0.0–1.0). Meaning depends on GoalKind.
    pub progress: f32,
    /// Optional target position the NPC is moving toward for this goal.
    pub target_pos: Option<(f32, f32)>,
    /// Optional target entity ID (the building, NPC, or item this goal targets).
    pub target_entity: Option<u32>,
    /// Multi-step plan: ordered steps the NPC executes to fulfill this goal.
    pub plan: Vec<PlannedStep>,
    /// Index of the currently active step in `plan`.
    pub plan_index: u16,
}

/// What an NPC is trying to accomplish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GoalKind {
    /// No active goal — waiting for the decision system to assign one.
    Idle,
    /// Work at assigned building (farming, mining, smithing, etc.).
    Work,
    /// Walk to food source and eat.
    Eat,
    /// Trade: carry goods to another settlement.
    Trade { destination_settlement_id: u32 },
    /// Mobilize for combat (move to threat, fight hostiles).
    Fight,
    /// Seek social interaction at a gathering spot.
    Socialize,
    /// Return home to rest (shelter need).
    Rest,
    /// Go on a quest/adventure.
    Quest { quest_id: u32, destination: (f32, f32) },
    /// Flee from danger.
    Flee { from: (f32, f32) },
    /// Construct a building.
    Build { building_id: u32 },
    /// Haul commodity from one location to another.
    Haul { commodity: u8, amount: f32, destination: (f32, f32) },
    /// Relocate to a different settlement.
    Relocate { destination_settlement_id: u32 },
    /// Travel to a resource source, gather commodity into inventory.
    Gather { commodity: u8, amount: f32 },
    /// Fulfill a service contract posted at a settlement.
    FulfillContract { contract_idx: usize },
}

// ---------------------------------------------------------------------------
// Service contracts — NPCs post and fulfill work requests
// ---------------------------------------------------------------------------

/// Payment in gold, commodities, or both. Supports barter (no gold)
/// and mixed payments (3 gold + 5 wood).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Payment {
    pub gold: f32,
    pub commodities: Vec<(u8, f32)>,
}

impl Payment {
    pub fn gold_only(amount: f32) -> Self {
        Self { gold: amount, commodities: Vec::new() }
    }
    pub fn commodity(commodity: u8, amount: f32) -> Self {
        Self { gold: 0.0, commodities: vec![(commodity, amount)] }
    }
    pub fn estimated_value(&self) -> f32 {
        self.gold + self.commodities.iter().map(|(_, a)| a).sum::<f32>()
    }
    pub fn is_empty(&self) -> bool {
        self.gold <= 0.0 && self.commodities.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceContract {
    pub requester_id: u32,
    pub service: ServiceType,
    /// Maximum the requester will pay (gold and/or commodities).
    pub max_payment: Payment,
    /// Actual agreed payment (set to winning bid amount on resolution).
    pub payment: f32,
    pub provider_id: Option<u32>,
    pub posted_tick: u64,
    pub completed: bool,
    /// Bidding deadline — urgency-driven:
    /// critical (need < 15): +5 ticks, high (< 30): +15, medium (< 50): +30, low: +100.
    pub bidding_deadline: u64,
    pub bids: Vec<ContractBid>,
    pub accepted_bid: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractBid {
    pub bidder_id: u32,
    /// Gold amount the bidder is willing to accept for the work.
    pub bid_amount: f32,
    /// Bidder's relevant skill value at time of bid.
    pub skill_value: f32,
    /// Bidder's credit history at time of bid.
    pub credit_history: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceType {
    Build(BuildingType),
    Gather(u8, f32),
    Craft,
    Heal,
    Guard(u32),
    Haul(u8, f32, (f32, f32)),
    Teach(u32),
    /// Direct commodity exchange — no service, just swap goods.
    Barter { offer: (u8, f32), want: (u8, f32) },
}

/// Standard priority levels for goals.
pub mod goal_priority {
    /// Fleeing from immediate lethal danger.
    pub const FLEE: u8 = 95;
    /// Eating when critically hungry (hunger < 15).
    pub const EAT_CRITICAL: u8 = 90;
    /// Active combat (in a fight, can't walk away).
    pub const FIGHT: u8 = 80;
    /// Eating when moderately hungry (hunger < 30).
    pub const EAT: u8 = 70;
    /// Work at assigned building.
    pub const WORK: u8 = 50;
    /// Hauling produced goods to storage.
    pub const HAUL: u8 = 50;
    /// Construction.
    pub const BUILD: u8 = 45;
    /// Trading — carrying goods between settlements.
    pub const TRADE: u8 = 40;
    /// Questing / adventuring.
    pub const QUEST: u8 = 40;
    /// Socializing at a gathering spot.
    pub const SOCIALIZE: u8 = 30;
    /// Resting at home.
    pub const REST: u8 = 20;
    /// Relocating to a new settlement.
    pub const RELOCATE: u8 = 15;
    /// No goal — waiting for assignment.
    pub const IDLE: u8 = 0;
}

impl Goal {
    pub fn new(kind: GoalKind, priority: u8, tick: u64) -> Self {
        Self {
            kind,
            priority,
            started_tick: tick,
            progress: 0.0,
            target_pos: None,
            target_entity: None,
            plan: Vec::new(),
            plan_index: 0,
        }
    }

    /// Create a goal with a target position.
    pub fn with_target_pos(mut self, pos: (f32, f32)) -> Self {
        self.target_pos = Some(pos);
        self
    }

    /// Create a goal with a target entity.
    pub fn with_target_entity(mut self, id: u32) -> Self {
        self.target_entity = Some(id);
        self
    }

    /// Attach a multi-step plan to this goal.
    pub fn with_plan(mut self, steps: Vec<PlannedStep>) -> Self {
        self.plan = steps;
        self.plan_index = 0;
        self
    }

    /// Get the currently active plan step, if any.
    pub fn current_step(&self) -> Option<&PlannedStep> {
        self.plan.get(self.plan_index as usize)
    }

    /// Get the currently active plan step mutably.
    pub fn current_step_mut(&mut self) -> Option<&mut PlannedStep> {
        let idx = self.plan_index as usize;
        self.plan.get_mut(idx)
    }

    /// Whether this goal has a plan with remaining steps.
    pub fn has_active_plan(&self) -> bool {
        !self.plan.is_empty() && (self.plan_index as usize) < self.plan.len()
    }

    /// Advance to the next plan step. Returns true if there are more steps.
    pub fn advance_plan(&mut self) -> bool {
        if let Some(step) = self.plan.get_mut(self.plan_index as usize) {
            step.status = StepStatus::Complete;
        }
        self.plan_index += 1;
        (self.plan_index as usize) < self.plan.len()
    }

    /// Whether the entire plan has been completed.
    pub fn plan_complete(&self) -> bool {
        !self.plan.is_empty() && (self.plan_index as usize) >= self.plan.len()
    }
}

/// Goal stack on an NPC. The top goal (highest priority) drives behavior.
/// When an interrupt (eating, combat) fires, a higher-priority goal is pushed.
/// When it completes, it pops and the previous goal resumes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalStack {
    /// Goals sorted by priority descending. `goals[0]` is the active goal.
    pub goals: Vec<Goal>,
}

impl GoalStack {
    /// Push a goal. If a goal of the same kind already exists, replace it
    /// if the new priority is higher. Maintains priority-sorted order.
    pub fn push(&mut self, goal: Goal) {
        // Remove existing goal of same kind (dedup).
        self.goals.retain(|g| g.kind != goal.kind);
        self.goals.push(goal);
        self.goals.sort_by(|a, b| b.priority.cmp(&a.priority));
        // Cap stack depth to prevent unbounded growth.
        if self.goals.len() > 8 {
            self.goals.truncate(8);
        }
    }

    /// Pop the current (highest-priority) goal. Returns it if one existed.
    pub fn pop(&mut self) -> Option<Goal> {
        if self.goals.is_empty() { return None; }
        Some(self.goals.remove(0))
    }

    /// Get the current active goal (highest priority), if any.
    pub fn current(&self) -> Option<&Goal> {
        self.goals.first()
    }

    /// Get the current active goal mutably.
    pub fn current_mut(&mut self) -> Option<&mut Goal> {
        self.goals.first_mut()
    }

    /// Get the current goal kind, or Idle if stack is empty.
    pub fn current_kind(&self) -> &GoalKind {
        self.goals.first().map(|g| &g.kind).unwrap_or(&GoalKind::Idle)
    }

    /// Remove all goals of a specific kind.
    pub fn remove_kind(&mut self, kind: &GoalKind) {
        self.goals.retain(|g| &g.kind != kind);
    }

    /// Check if the stack has a goal of the given kind.
    pub fn has(&self, kind: &GoalKind) -> bool {
        self.goals.iter().any(|g| &g.kind == kind)
    }

    /// Is the stack empty (NPC has nothing to do)?
    pub fn is_empty(&self) -> bool {
        self.goals.is_empty()
    }
}

// ---------------------------------------------------------------------------
// NpcAction — observable multi-tick action state
// ---------------------------------------------------------------------------

/// What an NPC is visibly doing right now. Each action has a duration and progress.
/// This is the "animation state" — what a viewer/UI would display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NpcAction {
    /// Standing idle, no visible activity.
    Idle,
    /// Walking along a path to a destination.
    Walking { destination: (f32, f32) },
    /// Eating at a food source (inn/market).
    Eating { ticks_remaining: u8, building_id: u32 },
    /// Working at a workstation (farming, mining, smithing, etc.).
    Working { ticks_remaining: u16, building_id: u32, activity: WorkActivity },
    /// Carrying goods to storage.
    Hauling { commodity: u8, amount: f32 },
    /// Fighting a hostile entity.
    Fighting { target_id: u32 },
    /// Talking with another NPC at a social location.
    Socializing { partner_id: u32, ticks_remaining: u8 },
    /// Resting at home.
    Resting { ticks_remaining: u8 },
    /// Building/constructing a structure.
    Building { building_id: u32, ticks_remaining: u16 },
    /// Fleeing from danger.
    Fleeing,
    /// Trading at a market (buying/selling goods).
    Trading { ticks_remaining: u8 },
    /// Harvesting a resource node (tree, ore vein, herb patch, etc.).
    Harvesting { resource_id: u32 },
}

/// Specific work activity for display purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkActivity {
    Farming,
    Mining,
    Logging,
    Smithing,
    Brewing,
    Crafting,
    Researching,
    Cooking,
}

impl Default for NpcAction {
    fn default() -> Self { NpcAction::Idle }
}

impl NpcAction {
    /// Human-readable description of the action.
    pub fn description(&self) -> String {
        match self {
            NpcAction::Idle => "idle".into(),
            NpcAction::Walking { .. } => "walking".into(),
            NpcAction::Eating { ticks_remaining, .. } => format!("eating ({} ticks left)", ticks_remaining),
            NpcAction::Working { activity, ticks_remaining, .. } => {
                let name = match activity {
                    WorkActivity::Farming => "farming",
                    WorkActivity::Mining => "mining",
                    WorkActivity::Logging => "logging",
                    WorkActivity::Smithing => "smithing",
                    WorkActivity::Brewing => "brewing",
                    WorkActivity::Crafting => "crafting",
                    WorkActivity::Researching => "researching",
                    WorkActivity::Cooking => "cooking",
                };
                format!("{} ({} ticks left)", name, ticks_remaining)
            }
            NpcAction::Hauling { commodity: _, amount } => format!("hauling {:.1} units", amount),
            NpcAction::Fighting { target_id } => format!("fighting #{}", target_id),
            NpcAction::Socializing { ticks_remaining, .. } => format!("socializing ({} ticks left)", ticks_remaining),
            NpcAction::Resting { ticks_remaining } => format!("resting ({} ticks left)", ticks_remaining),
            NpcAction::Building { ticks_remaining, .. } => format!("building ({} ticks left)", ticks_remaining),
            NpcAction::Fleeing => "fleeing".into(),
            NpcAction::Trading { ticks_remaining } => format!("trading ({} ticks left)", ticks_remaining),
            NpcAction::Harvesting { resource_id } => format!("harvesting resource #{}", resource_id),
        }
    }

    /// Is this a non-idle action (NPC is visibly doing something)?
    pub fn is_active(&self) -> bool {
        !matches!(self, NpcAction::Idle)
    }

    /// Action type discriminant for outcome tracking (0-11).
    pub fn action_type_id(&self) -> u8 {
        match self {
            NpcAction::Idle => 0,
            NpcAction::Walking { .. } => 1,
            NpcAction::Eating { .. } => 2,
            NpcAction::Working { .. } => 3,
            NpcAction::Hauling { .. } => 4,
            NpcAction::Fighting { .. } => 5,
            NpcAction::Socializing { .. } => 6,
            NpcAction::Resting { .. } => 7,
            NpcAction::Building { .. } => 8,
            NpcAction::Fleeing => 9,
            NpcAction::Trading { .. } => 10,
            NpcAction::Harvesting { .. } => 11,
        }
    }
}

// ---------------------------------------------------------------------------
// NpcData — NPC-specific economic/social data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcData {
    /// Personal name (e.g. "Korrin", "Thessa"). Generated deterministically from entity ID + seed.
    pub name: String,
    /// Links to campaign Adventurer id.
    pub adventurer_id: u32,
    pub gold: f32,
    /// Gold owed to another entity (settlement or NPC creditor).
    pub debt: f32,
    /// Entity ID of creditor (who this NPC owes gold to).
    pub creditor_id: Option<u32>,
    /// Rolling average gold earned per work cycle (EMA, alpha=0.1).
    pub income_rate: f32,
    /// Credit history score: 0-255, higher = more reliable contractor.
    pub credit_history: u8,
    pub home_settlement_id: Option<u32>,
    pub home_building_id: Option<u32>,
    pub work_building_id: Option<u32>,
    /// Spatial work loop state machine (Idle → Travel → Work → Carry → Idle).
    pub work_state: WorkState,
    pub economic_intent: EconomicIntent,
    /// What the NPC is visibly doing right now. Observable by the viewer/UI.
    pub action: NpcAction,
    /// Sworn oaths (loyalty, vengeance, protection). Fulfillment → Oathkeeper class.
    pub oaths: Vec<crate::world_sim::systems::oaths::Oath>,
    /// Spouse entity ID, if married.
    pub spouse_id: Option<u32>,
    /// Children entity IDs.
    pub children: Vec<u32>,
    /// Parent entity IDs (mother, father). Empty for first-gen NPCs.
    pub parents: Vec<u32>,
    /// Tick when born (0 for initial NPCs).
    pub born_tick: u64,
    /// Chain of mentor entity IDs (most recent first). Tracks skill lineage.
    pub mentor_lineage: Vec<u32>,
    pub apprentice_of: Option<u32>,
    pub apprentices: Vec<u32>,
    pub apprenticeship_start_tick: u64,
    /// Which building the NPC is currently inside (entity ID), if any.
    pub inside_building_id: Option<u32>,
    /// Which room within the building the NPC is using (index into building.rooms).
    pub current_room: Option<u8>,
    /// Priority goal stack. Top goal drives behavior; interrupts push/pop.
    pub goal_stack: GoalStack,
    /// Cached A* path on the city grid: list of (col, row) waypoints to follow.
    /// Cleared when goal changes or NPC leaves settlement.
    pub cached_path: Vec<(u16, u16)>,
    /// Index into cached_path — which waypoint we're currently walking toward.
    pub path_index: u16,
    pub price_knowledge: Vec<PriceReport>,
    /// Index into `WorldState.trade_routes` if this NPC is assigned to a route.
    pub trade_route_id: Option<usize>,
    /// History of profitable trade completions: (destination_settlement_id, profit_count).
    pub trade_history: Vec<(u32, u32)>,
    pub class_tags: Vec<String>,
    /// What commodities this NPC produces (commodity_index, rate_per_tick).
    pub behavior_production: Vec<(usize, f32)>,

    // --- Campaign system fields (migrated from headless_campaign::Adventurer) ---

    /// 0–100. Combat morale.
    pub morale: f32,
    /// 0–100. Psychological stress.
    pub stress: f32,
    /// 0–100. Physical fatigue.
    pub fatigue: f32,
    /// 0–100. Loyalty to guild/faction.
    pub loyalty: f32,
    /// 0–100. Injury severity (>=90 incapacitated).
    pub injury: f32,
    /// Hero archetype name (e.g. "knight", "ranger", "mage").
    pub archetype: String,
    /// Party membership, if any.
    pub party_id: Option<u32>,
    /// Faction membership, if any.
    pub faction_id: Option<u32>,
    /// Current mood (index into a mood enum). 0 = Neutral.
    pub mood: u8,
    /// Active fear type indices.
    pub fears: Vec<u8>,
    /// Legendary deed type indices.
    pub deeds: Vec<u8>,
    /// 0–100. Resolve under pressure.
    pub resolve: f32,
    /// Guild relationship score (-100 to 100).
    pub guild_relationship: f32,

    // --- Agent inner state (needs-driven behavior) ---

    /// Creature archetype — controls personality presets and allowed actions.
    pub creature_type: CreatureType,
    /// Home den position for territorial creatures (returns home when safe).
    pub home_den: Option<(f32, f32)>,
    /// Pack leader entity ID for pack creatures (regroups when social need is low).
    pub pack_leader_id: Option<u32>,

    /// Known resource locations discovered through perception or trade.
    /// Key: resource entity ID. Value: (position, resource_type, tick_observed).
    /// Stale entries (>2000 ticks old) become unreliable.
    pub known_resources: std::collections::HashMap<u32, ResourceKnowledge>,

    /// Maslow-inspired need levels driving goal selection.
    pub needs: Needs,
    /// Event log + semantic beliefs formed from experience.
    pub memory: Memory,
    /// Experience-shaped behavioral tendencies.
    pub personality: Personality,
    /// Transient emotional reactions.
    pub emotions: Emotions,

    /// Lifetime behavior tag accumulator. Sorted by tag hash for O(log n) lookup.
    pub behavior_profile: Vec<(u32, f32)>,

    /// Per-pair relationships with other NPCs. Capped at 20 entries.
    pub relationships: std::collections::HashMap<u32, Relationship>,

    /// Per-action-type outcome tracking for adaptive behavior (Phase A).
    /// Key: (action_type_discriminant, target_type_hash). Value: EMA of outcomes.
    pub action_outcomes: std::collections::HashMap<(u8, u32), OutcomeEMA>,

    /// Medium-term behavioral orientation (Phase B).
    pub aspiration: Aspiration,

    /// Per-commodity price beliefs (Phase E: information asymmetry).
    /// Key: commodity index. Value: belief about value/scarcity.
    pub price_beliefs: [PriceBelief; super::NUM_COMMODITIES],

    /// Passive ability effects on the simulation (Phase F: ability divergence).
    /// Derived from class composition and behavior profile.
    pub passive_effects: PassiveEffects,

    /// Active abilities usable in the world sim (parsed from class_tags DSL).
    pub world_abilities: Vec<WorldAbility>,

    /// Per-action-type cultural conformity bias (Phase C). Range [-0.3, 0.3].
    /// Index matches NpcAction::action_type_id() (0=idle..11=harvesting).
    pub cultural_bias: [f32; 12],

    // --- Action commitment (hysteresis) ---

    /// Current committed action and its utility score at time of commitment.
    pub current_intention: Option<(NpcAction, f32)>,
    /// How many ticks the NPC has been executing the current intention.
    pub intention_ticks: u32,

    /// Acquired classes from behavior profile matching.
    pub classes: Vec<ClassSlot>,

    /// Equipment quality per slot (legacy — stat bonuses now come from equipped item entities).
    pub equipment: Equipment,

    /// Item entity IDs equipped in each slot.
    pub equipped_items: EquippedItems,
}

/// Simple equipment with quality levels per slot (legacy — use EquippedItems for item entities).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Equipment {
    /// Weapon quality. +1 attack_damage per quality point.
    pub weapon: f32,
    /// Armor quality. +0.5 armor + 3 max_hp per quality point.
    pub armor: f32,
    /// Accessory quality. +0.02 move_speed per quality point.
    pub accessory: f32,
}

impl Equipment {
    pub fn attack_bonus(&self) -> f32 { self.weapon }
    pub fn armor_bonus(&self) -> f32 { self.armor * 0.5 }
    pub fn hp_bonus(&self) -> f32 { self.armor * 3.0 }
    pub fn speed_bonus(&self) -> f32 { self.accessory * 0.02 }
    pub fn total_quality(&self) -> f32 { self.weapon + self.armor + self.accessory }
}

// ---------------------------------------------------------------------------
// Item system — physical items as entities
// ---------------------------------------------------------------------------

/// Equipment slot for an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ItemSlot {
    Weapon = 0,
    Armor = 1,
    Accessory = 2,
}

/// Item rarity tier, affects stat multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ItemRarity {
    Common = 0,
    Uncommon = 1,
    Rare = 2,
    Epic = 3,
    Legendary = 4,
}

impl ItemRarity {
    /// Stat multiplier for this rarity tier.
    pub fn stat_multiplier(&self) -> f32 {
        match self {
            ItemRarity::Common => 1.0,
            ItemRarity::Uncommon => 1.3,
            ItemRarity::Rare => 1.7,
            ItemRarity::Epic => 2.2,
            ItemRarity::Legendary => 3.0,
        }
    }

    /// Determine rarity from crafter skill level (SMITHING tag value).
    pub fn from_skill(skill: f32) -> Self {
        if skill >= 2000.0 { ItemRarity::Legendary }
        else if skill >= 1000.0 { ItemRarity::Epic }
        else if skill >= 400.0 { ItemRarity::Rare }
        else if skill >= 100.0 { ItemRarity::Uncommon }
        else { ItemRarity::Common }
    }
}

/// Data for an item entity. Items are full entities with EntityKind::Item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemData {
    /// Equipment slot this item fills.
    pub slot: ItemSlot,
    /// Rarity tier.
    pub rarity: ItemRarity,
    /// Base quality (1.0–20.0). Combined with rarity multiplier for stat bonuses.
    pub quality: f32,
    /// Current durability (0.0–100.0). Item breaks at 0.
    pub durability: f32,
    /// Maximum durability.
    pub max_durability: f32,
    /// Entity ID of the NPC who has this equipped, or None if unowned/in storage.
    pub owner_id: Option<u32>,
    /// Settlement where this item is located (for unowned items).
    pub settlement_id: Option<u32>,
    /// Display name (e.g. "Iron Longsword", "Mithril Platemail").
    pub name: String,
    /// Entity ID of the crafter, if any.
    pub crafter_id: Option<u32>,
    /// Tick when crafted.
    pub crafted_tick: u64,
    /// History of significant events for this item.
    pub history: Vec<ItemEvent>,
    /// Whether this item has earned a legendary name from its deeds.
    pub is_legendary: bool,
    /// Whether this is a relic — special dungeon/ruin artifact with building bonuses.
    pub is_relic: bool,
    /// Relic effect when installed in a building (tag hash → bonus value).
    pub relic_bonus: Option<(u32, f32)>,
}

/// A significant event in an item's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemEvent {
    pub tick: u64,
    pub kind: ItemEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemEventKind {
    /// Crafted by an NPC.
    Crafted { crafter_name: String },
    /// Wielded by a new owner.
    Wielded { owner_name: String },
    /// Used to kill an entity.
    Kill { victim_name: String, victim_level: u32 },
    /// Survived a battle (owner took damage but lived).
    SurvivedBattle,
    /// Dropped on death of owner.
    DroppedOnDeath { owner_name: String },
}

impl ItemData {
    /// Effective quality = base quality × rarity multiplier.
    pub fn effective_quality(&self) -> f32 {
        self.quality * self.rarity.stat_multiplier()
    }

    /// Durability fraction (0.0–1.0). Stats scale linearly with durability.
    pub fn durability_fraction(&self) -> f32 {
        if self.max_durability <= 0.0 { return 1.0; }
        (self.durability / self.max_durability).clamp(0.0, 1.0)
    }

    /// Attack damage bonus (weapon only). Scales with quality, rarity, and durability.
    pub fn attack_bonus(&self) -> f32 {
        if self.slot != ItemSlot::Weapon { return 0.0; }
        self.effective_quality() * self.durability_fraction()
    }

    /// Armor bonus (armor only).
    pub fn armor_bonus(&self) -> f32 {
        if self.slot != ItemSlot::Armor { return 0.0; }
        self.effective_quality() * 0.5 * self.durability_fraction()
    }

    /// Max HP bonus (armor only).
    pub fn hp_bonus(&self) -> f32 {
        if self.slot != ItemSlot::Armor { return 0.0; }
        self.effective_quality() * 3.0 * self.durability_fraction()
    }

    /// Move speed bonus (accessory only).
    pub fn speed_bonus(&self) -> f32 {
        if self.slot != ItemSlot::Accessory { return 0.0; }
        self.effective_quality() * 0.02 * self.durability_fraction()
    }
}

// ---------------------------------------------------------------------------
// ResourceType / ResourceData — physical resource nodes in the world
// ---------------------------------------------------------------------------

/// Type of harvestable resource node in the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Produces WOOD. Renewable (regrows).
    Tree,
    /// Produces IRON. Non-renewable (depletes permanently).
    OreVein,
    /// Produces HERBS. Renewable.
    HerbPatch,
    /// Produces FOOD. Renewable (seasonal).
    BerryBush,
    /// Produces CRYSTAL (stone). Non-renewable.
    StoneOutcrop,
    /// Produces FOOD (coastal). Renewable.
    FishingSpot,
}

/// Data for a physical resource node entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceData {
    /// What kind of resource this node provides.
    pub resource_type: ResourceType,
    /// Units currently available for harvesting. Depletes when harvested.
    pub remaining: f32,
    /// Maximum capacity when fully regenerated.
    pub max_capacity: f32,
    /// Units per tick that regenerate (0.0 for non-renewable like ore).
    pub regrow_rate: f32,
    /// Which region this resource belongs to.
    pub region_id: u32,
}

impl ResourceType {
    /// Which commodity index this resource type produces.
    pub fn commodity(&self) -> usize {
        use super::commodity;
        match self {
            ResourceType::Tree => commodity::WOOD,
            ResourceType::OreVein => commodity::IRON,
            ResourceType::HerbPatch => commodity::HERBS,
            ResourceType::BerryBush => commodity::FOOD,
            ResourceType::StoneOutcrop => commodity::CRYSTAL,
            ResourceType::FishingSpot => commodity::FOOD,
        }
    }

    /// Display name for the visualizer.
    pub fn display_name(&self) -> &'static str {
        match self {
            ResourceType::Tree => "Tree",
            ResourceType::OreVein => "Ore Vein",
            ResourceType::HerbPatch => "Herb Patch",
            ResourceType::BerryBush => "Berry Bush",
            ResourceType::StoneOutcrop => "Stone Outcrop",
            ResourceType::FishingSpot => "Fishing Spot",
        }
    }
}

/// Equipment slots on an NPC, pointing to item entity IDs.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EquippedItems {
    /// Entity ID of equipped weapon, if any.
    pub weapon_id: Option<u32>,
    /// Entity ID of equipped armor, if any.
    pub armor_id: Option<u32>,
    /// Entity ID of equipped accessory, if any.
    pub accessory_id: Option<u32>,
}

impl EquippedItems {
    /// Get the item ID for a given slot.
    pub fn slot_id(&self, slot: ItemSlot) -> Option<u32> {
        match slot {
            ItemSlot::Weapon => self.weapon_id,
            ItemSlot::Armor => self.armor_id,
            ItemSlot::Accessory => self.accessory_id,
        }
    }

    /// Set the item ID for a given slot.
    pub fn set_slot(&mut self, slot: ItemSlot, id: Option<u32>) {
        match slot {
            ItemSlot::Weapon => self.weapon_id = id,
            ItemSlot::Armor => self.armor_id = id,
            ItemSlot::Accessory => self.accessory_id = id,
        }
    }
}

/// A class granted to an NPC from behavior profile matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassSlot {
    /// FNV-1a hash of the base class template name.
    pub class_name_hash: u32,
    /// Current level in this class.
    pub level: u16,
    /// Accumulated XP toward next level.
    pub xp: f32,
    /// Procedural display name (e.g., "Stoneheart Miner", "Frontier Herbalist").
    /// Empty until naming runs. LFM model can override later.
    pub display_name: String,
}

impl NpcData {
    /// Accumulate tags into the sorted behavior profile.
    pub fn accumulate_tags(&mut self, action: &ActionTags) {
        for i in 0..action.count as usize {
            let (tag_hash, weight) = action.tags[i];
            if weight <= 0.0 { continue; }
            match self.behavior_profile.binary_search_by_key(&tag_hash, |&(t, _)| t) {
                Ok(idx) => self.behavior_profile[idx].1 += weight,
                Err(idx) => {
                    self.behavior_profile.insert(idx, (tag_hash, weight));
                }
            }
        }
    }

    /// Look up accumulated value for a tag. O(log n).
    pub fn behavior_value(&self, tag_hash: u32) -> f32 {
        match self.behavior_profile.binary_search_by_key(&tag_hash, |&(t, _)| t) {
            Ok(idx) => self.behavior_profile[idx].1,
            Err(_) => 0.0,
        }
    }

    /// Focus score derived from average need satisfaction.
    /// Returns a multiplier in `[0.5, 1.5]` — miserable NPCs are unfocused,
    /// content NPCs are productive.
    pub fn focus(&self) -> f32 {
        let avg = (self.needs.hunger + self.needs.safety + self.needs.shelter
            + self.needs.social + self.needs.purpose + self.needs.esteem) / 6.0;
        (avg / 100.0).clamp(0.5, 1.5)
    }

    /// Modify relationship trust and familiarity with another entity.
    /// Creates the relationship entry if it doesn't exist. Enforces cap of 20.
    pub fn modify_relationship(&mut self, other_id: u32, trust_delta: f32, familiarity_delta: f32, tick: u64) {
        let rel = self.relationships.entry(other_id).or_insert_with(Relationship::new);
        rel.trust = (rel.trust + trust_delta).clamp(-1.0, 1.0);
        rel.familiarity = (rel.familiarity + familiarity_delta).clamp(0.0, 1.0);
        rel.last_interaction = tick;

        // Enforce cap: evict lowest familiarity if over 20.
        if self.relationships.len() > 20 {
            let evict = self.relationships.iter()
                .filter(|(&id, _)| id != other_id)
                .min_by(|a, b| a.1.familiarity.partial_cmp(&b.1.familiarity).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&id, _)| id);
            if let Some(id) = evict {
                self.relationships.remove(&id);
            }
        }
    }

    /// Get trust toward another entity. Returns 0.0 if no relationship exists.
    pub fn trust_toward(&self, other_id: u32) -> f32 {
        self.relationships.get(&other_id).map(|r| r.trust).unwrap_or(0.0)
    }
}

impl Default for NpcData {
    fn default() -> Self {
        Self {
            adventurer_id: 0,
            name: String::new(),
            gold: 0.0,
            debt: 0.0,
            creditor_id: None,
            income_rate: 0.0,
            credit_history: 128,
            home_settlement_id: None,
            home_building_id: None,
            work_building_id: None,
            work_state: WorkState::Idle,
            economic_intent: EconomicIntent::Idle,
            action: NpcAction::Idle,
            oaths: Vec::new(),
            spouse_id: None,
            children: Vec::new(),
            parents: Vec::new(),
            born_tick: 0,
            mentor_lineage: Vec::new(),
            apprentice_of: None,
            apprentices: Vec::new(),
            apprenticeship_start_tick: 0,
            inside_building_id: None,
            current_room: None,
            goal_stack: GoalStack::default(),
            cached_path: Vec::new(),
            path_index: 0,
            price_knowledge: Vec::new(),
            trade_route_id: None,
            trade_history: Vec::new(),
            class_tags: Vec::new(),
            behavior_production: Vec::new(),
            morale: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            loyalty: 50.0,
            injury: 0.0,
            archetype: String::new(),
            party_id: None,
            faction_id: None,
            mood: 0,
            fears: Vec::new(),
            deeds: Vec::new(),
            resolve: 50.0,
            guild_relationship: 0.0,
            creature_type: CreatureType::Citizen,
            home_den: None,
            pack_leader_id: None,
            known_resources: std::collections::HashMap::new(),
            needs: Needs::default(),
            memory: Memory::default(),
            personality: Personality::default(),
            emotions: Emotions::default(),
            behavior_profile: Vec::new(),
            relationships: std::collections::HashMap::new(),
            action_outcomes: std::collections::HashMap::new(),
            aspiration: Aspiration::default(),
            passive_effects: PassiveEffects::default(),
            world_abilities: Vec::new(),
            price_beliefs: [PriceBelief::default(); crate::world_sim::NUM_COMMODITIES],
            cultural_bias: [0.0; 12],
            current_intention: None,
            intention_ticks: 0,
            classes: Vec::new(),
            equipment: Equipment::default(),
            equipped_items: EquippedItems::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EconomicIntent {
    Idle,
    Produce,
    Trade { destination_settlement_id: u32 },
    Buy { commodity: usize },
    Sell { commodity: usize },
    Travel { destination: (f32, f32) },
    /// On a quest — traveling to quest destination to fight/explore.
    Adventuring { quest_id: u32, destination: (f32, f32) },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceReport {
    pub settlement_id: u32,
    pub prices: [f32; NUM_COMMODITIES],
    pub tick_observed: u64,
}

// ---------------------------------------------------------------------------
// FidelityZone — a proximity bubble that controls simulation fidelity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityZone {
    pub id: u32,
    pub fidelity: Fidelity,
    pub center: (f32, f32),
    pub radius: f32,
    /// Entity IDs currently on this grid.
    pub entity_ids: Vec<u32>,
}

impl FidelityZone {
    pub fn has_hostiles(&self, state: &super::WorldState) -> bool {
        self.entity_ids.iter().any(|id| {
            state.entity(*id)
                .map(|e| e.kind == EntityKind::Monster && e.alive)
                .unwrap_or(false)
        })
    }

    pub fn has_friendlies(&self, state: &super::WorldState) -> bool {
        self.entity_ids.iter().any(|id| {
            state.entity(*id)
                .map(|e| e.kind == EntityKind::Npc && e.alive)
                .unwrap_or(false)
        })
    }
}

// ---------------------------------------------------------------------------
// SettlementSpecialty — settlement economic specialization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SettlementSpecialty {
    #[default]
    General,
    /// Iron, crystal production bonus.
    MiningTown,
    /// Price discovery, merchant NPCs.
    TradeHub,
    /// Warriors, patrols reduce threat.
    MilitaryOutpost,
    /// Food, hide surplus.
    FarmingVillage,
    /// Research, XP bonus.
    ScholarCity,
    /// Coastal trade, fish (food).
    PortTown,
    /// Equipment, medicine production.
    CraftingGuild,
}

impl SettlementSpecialty {
    pub const ALL: [SettlementSpecialty; 8] = [
        SettlementSpecialty::General,
        SettlementSpecialty::MiningTown,
        SettlementSpecialty::TradeHub,
        SettlementSpecialty::MilitaryOutpost,
        SettlementSpecialty::FarmingVillage,
        SettlementSpecialty::ScholarCity,
        SettlementSpecialty::PortTown,
        SettlementSpecialty::CraftingGuild,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SettlementSpecialty::General => "General",
            SettlementSpecialty::MiningTown => "Mining Town",
            SettlementSpecialty::TradeHub => "Trade Hub",
            SettlementSpecialty::MilitaryOutpost => "Military Outpost",
            SettlementSpecialty::FarmingVillage => "Farming Village",
            SettlementSpecialty::ScholarCity => "Scholar City",
            SettlementSpecialty::PortTown => "Port Town",
            SettlementSpecialty::CraftingGuild => "Crafting Guild",
        }
    }

    /// Commodity production bonuses for this specialty.
    /// Returns (commodity_index, multiplier) pairs.
    pub fn production_bonuses(self) -> &'static [(usize, f32)] {
        use crate::world_sim::commodity::*;
        match self {
            SettlementSpecialty::General => &[],
            SettlementSpecialty::MiningTown => &[(IRON, 2.0), (CRYSTAL, 1.5)],
            SettlementSpecialty::TradeHub => &[(FOOD, 1.0)],  // trade hubs don't produce, they move goods
            SettlementSpecialty::MilitaryOutpost => &[(IRON, 1.3), (EQUIPMENT, 0.5)],
            SettlementSpecialty::FarmingVillage => &[(FOOD, 2.0), (HIDE, 1.5)],
            SettlementSpecialty::ScholarCity => &[(HERBS, 1.0), (MEDICINE, 0.5)],
            SettlementSpecialty::PortTown => &[(FOOD, 1.5), (WOOD, 1.0)],
            SettlementSpecialty::CraftingGuild => &[(EQUIPMENT, 2.0), (MEDICINE, 1.5)],
        }
    }

    /// Preferred NPC archetypes for this specialty (legacy uniform distribution).
    pub fn preferred_archetypes(self) -> &'static [&'static str] {
        match self {
            SettlementSpecialty::General => &["farmer", "knight", "ranger", "merchant"],
            SettlementSpecialty::MiningTown => &["smith", "miner", "knight"],
            SettlementSpecialty::TradeHub => &["merchant", "rogue", "cleric"],
            SettlementSpecialty::MilitaryOutpost => &["knight", "ranger", "mage"],
            SettlementSpecialty::FarmingVillage => &["farmer", "cleric", "ranger"],
            SettlementSpecialty::ScholarCity => &["mage", "cleric", "rogue"],
            SettlementSpecialty::PortTown => &["merchant", "ranger", "rogue"],
            SettlementSpecialty::CraftingGuild => &["smith", "mage", "merchant"],
        }
    }

    /// Weighted archetype distribution for this specialty.
    ///
    /// Returns `(archetype, cumulative_weight)` pairs where weights are out of 100.
    /// The "mixed" pool (`warrior, ranger, mage, cleric, rogue, merchant, farmer, smith`)
    /// fills the remaining weight (always the last 20%).
    pub fn weighted_archetypes(self) -> &'static [(&'static str, u32)] {
        match self {
            // Mining Town: 40% miners, 20% smiths, 20% merchants, 20% mixed
            SettlementSpecialty::MiningTown => &[
                ("miner", 40), ("smith", 60), ("merchant", 80),
            ],
            // Trade Hub: 30% merchants, 20% rogues, 20% diplomats, 30% mixed
            SettlementSpecialty::TradeHub => &[
                ("merchant", 30), ("rogue", 50), ("diplomat", 70),
            ],
            // Military Outpost: 40% warriors/knights, 20% rangers, 20% clerics, 20% mixed
            SettlementSpecialty::MilitaryOutpost => &[
                ("knight", 40), ("ranger", 60), ("cleric", 80),
            ],
            // Farming Village: 40% farmers, 20% herbalists, 20% clerics, 20% mixed
            SettlementSpecialty::FarmingVillage => &[
                ("farmer", 40), ("herbalist", 60), ("cleric", 80),
            ],
            // Scholar City: 30% mages, 30% scholars, 20% alchemists, 20% mixed
            SettlementSpecialty::ScholarCity => &[
                ("mage", 30), ("scholar", 60), ("alchemist", 80),
            ],
            // Crafting Guild: 40% smiths, 20% alchemists, 20% artificers, 20% mixed
            SettlementSpecialty::CraftingGuild => &[
                ("smith", 40), ("alchemist", 60), ("artificer", 80),
            ],
            // Port Town: 30% merchants, 20% rangers, 20% rogues, 30% mixed
            SettlementSpecialty::PortTown => &[
                ("merchant", 30), ("ranger", 50), ("rogue", 70),
            ],
            // General: 100% mixed
            SettlementSpecialty::General => &[],
        }
    }
}

impl std::fmt::Display for SettlementSpecialty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// SettlementState — per-settlement economy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementState {
    pub id: u32,
    pub name: String,
    pub pos: (f32, f32),
    pub grid_id: Option<u32>,
    pub stockpile: [f32; NUM_COMMODITIES],
    pub prices: [f32; NUM_COMMODITIES],
    pub treasury: f32,
    pub population: u32,
    pub specialty: SettlementSpecialty,

    // --- Campaign system fields ---

    /// Owning faction, if any.
    pub faction_id: Option<u32>,
    /// 0–1. Threat from nearby monsters/enemies.
    pub threat_level: f32,
    /// 0–5. Building/upgrade level.
    pub infrastructure_level: f32,

    /// Context tags applied to all actions at this settlement.
    pub context_tags: Vec<(u32, f32)>,

    /// Entity ID of the treasury building. Holds settlement gold and commodity reserves.
    /// All economic transfers should route through this building's inventory.
    pub treasury_building_id: Option<u32>,

    /// Active service contracts posted by NPCs in this settlement.
    pub service_contracts: Vec<ServiceContract>,

    // Building intelligence integration
    /// Per-settlement construction memory (short/medium/long-term event history).
    #[serde(default)]
    pub construction_memory: ConstructionMemory,
}

impl SettlementState {
    pub fn new(id: u32, name: String, pos: (f32, f32)) -> Self {
        Self {
            id,
            name,
            pos,
            grid_id: None,
            stockpile: [0.0; NUM_COMMODITIES],
            prices: [1.0; NUM_COMMODITIES],
            treasury: 0.0,
            population: 0,
            specialty: SettlementSpecialty::default(),
            faction_id: None,
            threat_level: 0.0,
            infrastructure_level: 0.0,
            context_tags: Vec::new(),
            treasury_building_id: None,
            service_contracts: Vec::new(),
            construction_memory: ConstructionMemory::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain — biome type for a region
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Terrain {
    /// Food, hide. Gentle farmland.
    Plains,
    /// Wood, herbs. Dense canopy, hidden dangers.
    Forest,
    /// Iron, crystal. High elevation, harsh conditions.
    Mountains,
    /// Food, trade bonus. Access to sea travel.
    Coast,
    /// Herbs, medicine. Slow travel, disease risk.
    Swamp,
    /// Crystal, sparse. Extreme heat, mirages.
    Desert,
    /// Hide, sparse. Extreme cold, blizzards.
    Tundra,
    // --- New terrain types ---
    /// Lava, obsidian. Extreme danger, rare minerals. Eruption risk.
    Volcano,
    /// Open water. Impassable on foot. Sea monsters. Ship travel only.
    DeepOcean,
    /// Dense tropical. High food + herbs, disease, hidden temples.
    Jungle,
    /// Frozen peaks. Rare crystal, extreme cold, impassable in winter.
    Glacier,
    /// Underground networks. Ore-rich, pitch dark, cave monsters.
    Caverns,
    /// Eroded wasteland. Sparse resources, bandits, harsh winds.
    Badlands,
    /// Floating landmasses. Extreme rarity, crystal-rich, wind access only.
    FlyingIslands,
    /// Corrupted land. 5× threat, toxic air, undead, legendary loot.
    DeathZone,
    /// Crumbling civilization. Relic sites, traps, dungeon entrances.
    AncientRuins,
    /// Shallow tropical water. Food + crystal, sea creatures, underwater access.
    CoralReef,
}

impl Terrain {
    pub const ALL: [Terrain; 17] = [
        Terrain::Plains, Terrain::Forest, Terrain::Mountains,
        Terrain::Coast, Terrain::Swamp, Terrain::Desert, Terrain::Tundra,
        Terrain::Volcano, Terrain::DeepOcean, Terrain::Jungle,
        Terrain::Glacier, Terrain::Caverns, Terrain::Badlands,
        Terrain::FlyingIslands, Terrain::DeathZone, Terrain::AncientRuins,
        Terrain::CoralReef,
    ];

    /// Name for display.
    pub fn name(self) -> &'static str {
        match self {
            Terrain::Plains => "Plains",
            Terrain::Forest => "Forest",
            Terrain::Mountains => "Mountains",
            Terrain::Coast => "Coast",
            Terrain::Swamp => "Swamp",
            Terrain::Desert => "Desert",
            Terrain::Tundra => "Tundra",
            Terrain::Volcano => "Volcano",
            Terrain::DeepOcean => "Deep Ocean",
            Terrain::Jungle => "Jungle",
            Terrain::Glacier => "Glacier",
            Terrain::Caverns => "Caverns",
            Terrain::Badlands => "Badlands",
            Terrain::FlyingIslands => "Flying Islands",
            Terrain::DeathZone => "Death Zone",
            Terrain::AncientRuins => "Ancient Ruins",
            Terrain::CoralReef => "Coral Reef",
        }
    }

    /// Threat multiplier for this terrain. Higher = more dangerous.
    pub fn threat_multiplier(self) -> f32 {
        match self {
            Terrain::Plains => 1.0,
            Terrain::Forest => 1.2,
            Terrain::Mountains => 1.5,
            Terrain::Coast => 0.8,
            Terrain::Swamp => 1.3,
            Terrain::Desert => 1.4,
            Terrain::Tundra => 1.3,
            Terrain::Volcano => 3.0,
            Terrain::DeepOcean => 2.5,
            Terrain::Jungle => 1.8,
            Terrain::Glacier => 2.0,
            Terrain::Caverns => 2.0,
            Terrain::Badlands => 1.6,
            Terrain::FlyingIslands => 1.5,
            Terrain::DeathZone => 5.0,
            Terrain::AncientRuins => 2.5,
            Terrain::CoralReef => 1.0,
        }
    }

    /// Travel speed multiplier. Lower = slower passage.
    pub fn travel_speed(self) -> f32 {
        match self {
            Terrain::Plains => 1.0,
            Terrain::Forest => 0.7,
            Terrain::Mountains => 0.5,
            Terrain::Coast => 1.0,
            Terrain::Swamp => 0.4,
            Terrain::Desert => 0.6,
            Terrain::Tundra => 0.6,
            Terrain::Volcano => 0.3,
            Terrain::DeepOcean => 0.0, // impassable on foot
            Terrain::Jungle => 0.4,
            Terrain::Glacier => 0.3,
            Terrain::Caverns => 0.5,
            Terrain::Badlands => 0.7,
            Terrain::FlyingIslands => 0.0, // requires special access
            Terrain::DeathZone => 0.5,
            Terrain::AncientRuins => 0.6,
            Terrain::CoralReef => 0.0, // underwater
        }
    }

    /// Whether settlements can be founded here.
    pub fn is_settleable(self) -> bool {
        match self {
            Terrain::DeepOcean | Terrain::FlyingIslands | Terrain::CoralReef => false,
            Terrain::Volcano | Terrain::DeathZone => false, // too dangerous
            _ => true,
        }
    }

    /// Base elevation tier for this terrain type.
    pub fn base_elevation(self) -> u8 {
        match self {
            Terrain::DeepOcean | Terrain::CoralReef => 0,
            Terrain::Coast | Terrain::Swamp => 0,
            Terrain::Plains | Terrain::Desert | Terrain::Jungle => 1,
            Terrain::Forest | Terrain::Badlands | Terrain::AncientRuins => 1,
            Terrain::Tundra | Terrain::Caverns => 2,
            Terrain::Mountains | Terrain::Volcano => 3,
            Terrain::Glacier => 3,
            Terrain::DeathZone => 2,
            Terrain::FlyingIslands => 4,
        }
    }

    /// Resource multiplier from elevation. Higher = rarer but more valuable ores.
    pub fn elevation_resource_mult(elevation: u8) -> f32 {
        match elevation {
            0 => 1.0,       // sea level — standard
            1 => 1.0,       // foothills — standard
            2 => 1.3,       // highlands — modest bonus
            3 => 1.8,       // peaks — good ores
            4 => 3.0,       // summit/sky — extremely rare materials
            _ => 1.0,
        }
    }

    /// Threat multiplier from elevation. Higher = more dangerous.
    pub fn elevation_threat_mult(elevation: u8) -> f32 {
        match elevation {
            0 => 0.8,       // sea level — relatively safe
            1 => 1.0,       // foothills — standard
            2 => 1.3,       // highlands — tougher monsters
            3 => 2.0,       // peaks — dangerous
            4 => 3.0,       // summit/sky — extreme
            _ => 1.0,
        }
    }

    /// Travel speed penalty from elevation. Higher = slower.
    pub fn elevation_travel_mult(elevation: u8) -> f32 {
        match elevation {
            0 => 1.0,
            1 => 1.0,
            2 => 0.8,
            3 => 0.5,       // peaks — slow climbing
            4 => 0.3,       // summit/sky — very difficult
            _ => 1.0,
        }
    }

    /// Whether this terrain has dungeon/ruin entrances for adventuring.
    pub fn has_dungeons(self) -> bool {
        matches!(self, Terrain::Caverns | Terrain::AncientRuins | Terrain::Volcano | Terrain::DeathZone)
    }

    /// Commodity indices that this terrain naturally produces.
    pub fn primary_commodities(self) -> &'static [(usize, f32)] {
        use crate::world_sim::commodity::*;
        match self {
            Terrain::Plains       => &[(FOOD, 1.5), (HIDE, 1.0)],
            Terrain::Forest       => &[(WOOD, 1.5), (HERBS, 1.0)],
            Terrain::Mountains    => &[(IRON, 1.5), (CRYSTAL, 1.0)],
            Terrain::Coast        => &[(FOOD, 1.2), (WOOD, 0.8)],
            Terrain::Swamp        => &[(HERBS, 1.5), (MEDICINE, 1.0)],
            Terrain::Desert       => &[(CRYSTAL, 1.2)],
            Terrain::Tundra       => &[(HIDE, 1.2)],
            Terrain::Volcano      => &[(IRON, 2.0), (CRYSTAL, 2.0)],  // rare minerals in lava
            Terrain::DeepOcean    => &[(FOOD, 0.5)],                   // fishing only
            Terrain::Jungle       => &[(FOOD, 1.8), (HERBS, 1.5), (WOOD, 1.0)],
            Terrain::Glacier      => &[(CRYSTAL, 1.5)],                // frozen crystal veins
            Terrain::Caverns      => &[(IRON, 2.0), (CRYSTAL, 1.5)],  // underground ore
            Terrain::Badlands     => &[(IRON, 0.8)],                   // sparse
            Terrain::FlyingIslands => &[(CRYSTAL, 3.0)],               // sky crystal
            Terrain::DeathZone    => &[(CRYSTAL, 2.0), (MEDICINE, 1.5)], // cursed reagents
            Terrain::AncientRuins => &[(CRYSTAL, 1.0)],                // artifacts
            Terrain::CoralReef    => &[(FOOD, 1.5), (CRYSTAL, 0.8)],   // pearls + fish
        }
    }
}

impl std::fmt::Display for Terrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// DungeonSite — explorable dungeon entrance in a region
// ---------------------------------------------------------------------------

/// A dungeon entrance within a region. Adventuring parties can enter for
/// loot, XP, and dungeoneering tags. Dungeons have depth levels — deeper
/// floors have better loot but more danger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DungeonSite {
    /// World-space position of the entrance.
    pub pos: (f32, f32),
    /// Procedural name (e.g., "The Sunken Halls", "Cavern of Echoes").
    pub name: String,
    /// Current explored depth (0 = entrance only, higher = deeper).
    pub explored_depth: u8,
    /// Maximum depth (determines loot tier).
    pub max_depth: u8,
    /// Whether this dungeon has been fully cleared.
    pub is_cleared: bool,
    /// Tick when last explored (for respawn timing).
    pub last_explored_tick: u64,
    /// Threat level modifier (from terrain).
    pub threat_mult: f32,
}

// ---------------------------------------------------------------------------
// SubBiome — terrain sub-variants for richer geography
// ---------------------------------------------------------------------------

/// Sub-variant of a terrain type. Adds detail within a biome category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubBiome {
    /// Default — no special variant.
    Standard,
    // Forest variants
    /// Sparse trees, easy travel, low wood yield.
    LightForest,
    /// Thick canopy, slow travel, high wood yield, monsters hide.
    DenseForest,
    /// Massive ancient trees, rare herbs, sacred groves, mystical encounters.
    AncientForest,
    // Desert variants
    /// Sand dunes, very slow travel.
    SandDunes,
    /// Rocky desert with exposed ore veins.
    RockyDesert,
    // Mountain variants
    /// Volcanic hot springs, herbs grow near heat.
    HotSprings,
    // Swamp variants
    /// Bioluminescent fungi, rare alchemy ingredients.
    GlowingMarsh,
    // Jungle variants
    /// Overgrown ruins, hidden temples.
    TempleJungle,
}

impl Default for SubBiome {
    fn default() -> Self { SubBiome::Standard }
}

impl SubBiome {
    /// Display suffix for region names.
    pub fn suffix(self) -> &'static str {
        match self {
            SubBiome::Standard => "",
            SubBiome::LightForest => " (light)",
            SubBiome::DenseForest => " (dense)",
            SubBiome::AncientForest => " (ancient)",
            SubBiome::SandDunes => " (dunes)",
            SubBiome::RockyDesert => " (rocky)",
            SubBiome::HotSprings => " (hot springs)",
            SubBiome::GlowingMarsh => " (glowing)",
            SubBiome::TempleJungle => " (temple)",
        }
    }

    /// Wood yield multiplier for forest variants.
    pub fn wood_mult(self) -> f32 {
        match self {
            SubBiome::LightForest => 0.6,
            SubBiome::DenseForest => 1.8,
            SubBiome::AncientForest => 1.2,
            _ => 1.0,
        }
    }

    /// Herb yield multiplier.
    pub fn herb_mult(self) -> f32 {
        match self {
            SubBiome::AncientForest => 2.0,
            SubBiome::GlowingMarsh => 2.5,
            SubBiome::TempleJungle => 1.5,
            SubBiome::HotSprings => 1.8,
            _ => 1.0,
        }
    }

    /// Travel speed multiplier.
    pub fn travel_mult(self) -> f32 {
        match self {
            SubBiome::LightForest => 1.2,   // easier than standard forest
            SubBiome::DenseForest => 0.5,   // very slow
            SubBiome::AncientForest => 0.7, // moderate
            SubBiome::SandDunes => 0.4,     // exhausting
            SubBiome::GlowingMarsh => 0.6,
            SubBiome::TempleJungle => 0.5,
            _ => 1.0,
        }
    }

    /// Monster stealth bonus — monsters in dense terrain are harder to detect.
    pub fn monster_stealth(self) -> f32 {
        match self {
            SubBiome::DenseForest => 0.3,   // monsters ambush
            SubBiome::AncientForest => 0.2, // mystical concealment
            SubBiome::TempleJungle => 0.3,  // hidden traps
            _ => 0.0,
        }
    }

    /// Threat multiplier from sub-biome.
    pub fn threat_mult(self) -> f32 {
        match self {
            SubBiome::LightForest => 0.7,
            SubBiome::DenseForest => 1.5,
            SubBiome::AncientForest => 1.3,
            SubBiome::SandDunes => 1.2,
            SubBiome::GlowingMarsh => 1.4,
            SubBiome::TempleJungle => 1.6,
            _ => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RegionState — regional monster population and faction control
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionState {
    pub id: u32,
    pub name: String,
    pub terrain: Terrain,
    /// World-space center position of this region.
    pub pos: (f32, f32),
    pub monster_density: f32,
    pub faction_id: Option<u32>,
    pub threat_level: f32,

    // --- Water features ---

    /// Whether this region has a river flowing through it.
    pub has_river: bool,
    /// Whether this region has a lake.
    pub has_lake: bool,
    /// Whether this region borders the ocean (edge of map or adjacent to DeepOcean).
    pub is_coastal: bool,
    /// IDs of regions this region's river connects to (for river travel).
    pub river_connections: Vec<u32>,
    /// Dungeon sites in this region: (world_pos, depth_level, is_cleared).
    pub dungeon_sites: Vec<DungeonSite>,

    // --- Vertical terrain ---

    /// Sub-biome variant for terrain detail (e.g., Forest → Light/Dense/Ancient).
    pub sub_biome: SubBiome,
    /// Adjacent region IDs (4-connected grid neighbors). Forms the travel graph.
    pub neighbors: Vec<u32>,
    /// Whether this region is a chokepoint (only 1-2 passable neighbors).
    pub is_chokepoint: bool,
    /// Elevation tier (0-4). 0=sea level, 1=foothills, 2=highlands, 3=peaks, 4=summit/sky.
    pub elevation: u8,
    /// Whether this region is a floating landmass (FlyingIslands). Requires special access.
    pub is_floating: bool,

    // --- Campaign system fields ---

    /// 0–1. Civil unrest level.
    pub unrest: f32,
    /// 0–1. Faction control strength.
    pub control: f32,
}

// ---------------------------------------------------------------------------
// EconomyState — global economy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EconomyState {
    pub total_gold_supply: f32,
    pub total_commodities: [f32; NUM_COMMODITIES],
}

// ---------------------------------------------------------------------------
// TradeRoute -- emergent trade route between settlements
// ---------------------------------------------------------------------------

/// A trade route established by NPCs who repeatedly profit from trading
/// between two settlements. Strength decays without activity; routes are
/// abandoned when strength drops below a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRoute {
    pub settlement_a: u32,
    pub settlement_b: u32,
    pub established_tick: u64,
    pub total_profit: f32,
    pub trade_count: u32,
    /// 0.0-1.0. Decays without activity; route abandoned below 0.1.
    pub strength: f32,
}

// ---------------------------------------------------------------------------
// FactionState — political/military faction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionState {
    pub id: u32,
    pub name: String,
    /// Relationship to the player guild (-100 to 100).
    pub relationship_to_guild: f32,
    pub military_strength: f32,
    pub max_military_strength: f32,
    pub territory_size: u32,
    pub diplomatic_stance: DiplomaticStance,
    /// Treasury (faction gold).
    pub treasury: f32,
    /// Faction-to-faction war targets.
    pub at_war_with: Vec<u32>,
    /// Running coup-risk score (0.0–1.0).
    pub coup_risk: f32,
    /// Escalation level (0=normal, 5=maximum).
    pub escalation_level: u32,
    /// Tech/research level.
    pub tech_level: u32,
    /// Recent actions log (bounded).
    pub recent_actions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiplomaticStance {
    Friendly,
    Neutral,
    Hostile,
    AtWar,
    Coalition,
}

impl Default for DiplomaticStance {
    fn default() -> Self {
        DiplomaticStance::Neutral
    }
}

// ---------------------------------------------------------------------------
// GuildState — player guild
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuildState {
    pub gold: f32,
    pub supplies: f32,
    /// 0–100. Affects quest quality and pricing.
    pub reputation: f32,
    /// Guild tier (0–5).
    pub tier: u32,
    /// Credit rating for loans.
    pub credit_rating: f32,
    /// Active quest capacity.
    pub active_quest_capacity: u32,
}

impl Default for GuildState {
    fn default() -> Self {
        Self {
            gold: 100.0,
            supplies: 50.0,
            reputation: 10.0,
            tier: 0,
            credit_rating: 50.0,
            active_quest_capacity: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Quest — an active quest being pursued
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    pub id: u32,
    pub name: String,
    pub quest_type: QuestType,
    /// Assigned party entity IDs.
    pub party_member_ids: Vec<u32>,
    /// Destination position.
    pub destination: (f32, f32),
    /// 0–1. Progress toward completion.
    pub progress: f32,
    /// Current quest phase.
    pub status: QuestStatus,
    /// Tick when quest was accepted.
    pub accepted_tick: u64,
    /// Tick deadline (0 = no deadline).
    pub deadline_tick: u64,
    /// Threat level of the quest.
    pub threat_level: f32,
    /// Gold reward on completion.
    pub reward_gold: f32,
    /// XP reward on completion.
    pub reward_xp: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestType {
    Hunt,
    Escort,
    Deliver,
    Explore,
    Defend,
    Gather,
    Rescue,
    Assassinate,
    Diplomacy,
    Custom,
}

impl Default for QuestType {
    fn default() -> Self {
        QuestType::Hunt
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestStatus {
    /// Quest accepted, party traveling to destination.
    Traveling,
    /// Party is at the quest location, working on it.
    InProgress,
    /// Quest completed successfully.
    Completed,
    /// Quest failed.
    Failed,
    /// Party is returning to base.
    Returning,
}

impl Default for QuestStatus {
    fn default() -> Self {
        QuestStatus::Traveling
    }
}

// ---------------------------------------------------------------------------
// QuestPosting — a quest available on the board
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestPosting {
    pub id: u32,
    pub name: String,
    pub quest_type: QuestType,
    /// Settlement that posted this quest.
    pub settlement_id: u32,
    pub destination: (f32, f32),
    pub threat_level: f32,
    pub reward_gold: f32,
    pub reward_xp: u32,
    /// Tick when posting expires.
    pub expires_tick: u64,
}

// ---------------------------------------------------------------------------
// ChronicleEntry — narrative log entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleEntry {
    pub tick: u64,
    pub category: ChronicleCategory,
    pub text: String,
    /// Entity IDs involved.
    pub entity_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChronicleCategory {
    Battle,
    Quest,
    Diplomacy,
    Economy,
    Death,
    Discovery,
    Crisis,
    Achievement,
    Narrative,
}

// ---------------------------------------------------------------------------
// WorldEvent — events recorded during a tick
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldEvent {
    /// Generic event with a category tag and description.
    Generic { category: ChronicleCategory, text: String },
    /// Entity died.
    EntityDied { entity_id: u32, cause: String },
    /// Quest status changed.
    QuestChanged { quest_id: u32, new_status: QuestStatus },
    /// Faction relation changed.
    FactionRelationChanged { faction_id: u32, old: f32, new: f32 },
    /// Region ownership changed.
    RegionOwnerChanged { region_id: u32, old_owner: Option<u32>, new_owner: Option<u32> },
    /// Bond grief event.
    BondGrief { entity_id: u32, dead_id: u32, bond_strength: f32 },
    /// Season changed.
    SeasonChanged { new_season: u8 },
    /// Battle started.
    BattleStarted { grid_id: u32, participants: Vec<u32> },
    /// Battle ended.
    BattleEnded { grid_id: u32, victor_team: WorldTeam },
    /// Settlement posted a quest to reduce threat.
    QuestPosted { settlement_id: u32, threat_level: f32, reward_gold: f32 },
    /// NPC accepted a quest.
    QuestAccepted { entity_id: u32, quest_id: u32 },
    /// Quest completed.
    QuestCompleted { entity_id: u32, quest_id: u32, reward_gold: f32 },
    /// Faction conquered a settlement from another faction.
    SettlementConquered { settlement_id: u32, new_faction_id: u32 },
}

// ---------------------------------------------------------------------------
// EntityField — fields on Entity/NpcData addressable by delta
// ---------------------------------------------------------------------------

/// Addressable scalar fields on an entity's NpcData.
/// Used by `UpdateEntityField` delta for consolidated per-entity updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityField {
    Morale,
    Stress,
    Fatigue,
    Loyalty,
    Injury,
    Resolve,
    GuildRelationship,
    Gold,
    Hp,
    MaxHp,
    ShieldHp,
    Armor,
    MagicResist,
    AttackDamage,
    AttackRange,
    MoveSpeed,
    /// Level increment (value cast to u32, added to entity.level).
    Level,
}

// ---------------------------------------------------------------------------
// FactionField — addressable fields on FactionState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactionField {
    RelationshipToGuild,
    MilitaryStrength,
    Treasury,
    CoupRisk,
    EscalationLevel,
    TechLevel,
}

// ---------------------------------------------------------------------------
// RegionField — addressable fields on RegionState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionField {
    MonsterDensity,
    ThreatLevel,
    Unrest,
    Control,
}

// ---------------------------------------------------------------------------
// SettlementField — addressable fields on SettlementState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SettlementField {
    Treasury,
    Population,
    ThreatLevel,
    InfrastructureLevel,
    /// Set (not additive): value is cast to u32 as the new faction owner.
    FactionId,
}

// ---------------------------------------------------------------------------
// RelationKind — types of entity-to-entity relations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RelationKind {
    /// General relationship (-100 to 100).
    Relationship = 0,
    /// Bond strength (0–100).
    Bond = 1,
    /// Romance level (0–100).
    Romance = 2,
    /// Rivalry intensity (0–100).
    Rivalry = 3,
    /// Grudge intensity (0–100).
    Grudge = 4,
    /// Mentor-mentee (0–100).
    Mentorship = 5,
}

// ---------------------------------------------------------------------------
// QuestDelta — quest lifecycle changes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestDelta {
    /// Advance progress by a fraction.
    AdvanceProgress { amount: f32 },
    /// Change quest status.
    SetStatus { status: QuestStatus },
    /// Add a member to the quest party.
    AddMember { entity_id: u32 },
    /// Remove a member from the quest party.
    RemoveMember { entity_id: u32 },
    /// Complete the quest (triggers rewards).
    Complete,
    /// Fail the quest.
    Fail,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> WorldState {
        let mut s = WorldState::new(42);
        // Add 3 NPCs at settlement 0, 2 buildings at settlement 0, 1 monster (no settlement).
        let mut settlement = SettlementState::new(0, "TestTown".into(), (0.0, 0.0));
        settlement.treasury = 100.0;
        settlement.stockpile[0] = 50.0; // food
        s.settlements.push(settlement);

        for i in 0..3 {
            let mut e = Entity::new_npc(i, (i as f32, 0.0));
            e.npc.as_mut().unwrap().home_settlement_id = Some(0);
            s.entities.push(e);
        }
        for i in 3..5 {
            let mut e = Entity::new_building(i, (i as f32, 0.0));
            e.building = Some(BuildingData {
                settlement_id: Some(0),
                ..Default::default()
            });
            s.entities.push(e);
        }
        let monster = Entity::new_monster(5, (10.0, 10.0), 3);
        s.entities.push(monster);

        s.next_id = 6;
        s.rebuild_all_indices();
        s
    }

    #[test]
    fn entity_index_o1_lookup() {
        let s = make_test_state();
        for i in 0..6 {
            let e = s.entity(i).expect("entity should exist");
            assert_eq!(e.id, i);
        }
        assert!(s.entity(999).is_none());
    }

    #[test]
    fn entity_idx_returns_position() {
        let s = make_test_state();
        for id in 0..6 {
            let idx = s.entity_idx(id).expect("should have index");
            assert_eq!(s.entities[idx].id, id);
        }
    }

    #[test]
    fn entity_mut_modifies_in_place() {
        let mut s = make_test_state();
        s.entity_mut(0).unwrap().hp = 999.0;
        assert_eq!(s.entity(0).unwrap().hp, 999.0);
    }

    #[test]
    fn next_entity_id_is_monotonic() {
        let mut s = make_test_state();
        let a = s.next_entity_id();
        let b = s.next_entity_id();
        let c = s.next_entity_id();
        assert!(a < b && b < c);
    }

    #[test]
    fn settlement_index_o1_lookup() {
        let s = make_test_state();
        let settlement = s.settlement(0).expect("should exist");
        assert_eq!(settlement.name, "TestTown");
        assert!(s.settlement(999).is_none());
    }

    #[test]
    fn settlement_mut_modifies() {
        let mut s = make_test_state();
        s.settlement_mut(0).unwrap().treasury = 999.0;
        assert_eq!(s.settlement(0).unwrap().treasury, 999.0);
    }

    #[test]
    fn group_index_settlement_range() {
        let s = make_test_state();
        let range = s.group_index.settlement_entities(0);
        // Should contain all 5 entities at settlement 0 (3 NPCs + 2 buildings).
        assert_eq!(range.len(), 5);
        for i in range {
            assert!(s.entities[i].settlement_id() == Some(0));
        }
    }

    #[test]
    fn group_index_unaffiliated_contains_monster() {
        let s = make_test_state();
        let range = s.group_index.unaffiliated_entities();
        assert!(range.len() >= 1);
        let has_monster = range.clone().any(|i| s.entities[i].kind == EntityKind::Monster);
        assert!(has_monster);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut s1 = WorldState::new(42);
        let mut s2 = WorldState::new(42);
        let vals1: Vec<u32> = (0..10).map(|_| s1.next_rand_u32()).collect();
        let vals2: Vec<u32> = (0..10).map(|_| s2.next_rand_u32()).collect();
        assert_eq!(vals1, vals2);
    }

    #[test]
    fn inventory_deposit_withdraw() {
        let mut inv = Inventory::with_capacity(100.0);
        let stored = inv.deposit(0, 50.0);
        assert_eq!(stored, 50.0);
        assert_eq!(inv.commodities[0], 50.0);

        let taken = inv.withdraw(0, 30.0);
        assert_eq!(taken, 30.0);
        assert_eq!(inv.commodities[0], 20.0);

        // Withdraw more than available.
        let taken = inv.withdraw(0, 100.0);
        assert_eq!(taken, 20.0);
        assert_eq!(inv.commodities[0], 0.0);
    }

    #[test]
    fn inventory_capacity_limit() {
        let mut inv = Inventory::with_capacity(10.0);
        let stored = inv.deposit(0, 50.0);
        assert_eq!(stored, 10.0); // capped at capacity
        assert_eq!(inv.commodities[0], 10.0);
    }

    #[test]
    fn inventory_unlimited_capacity() {
        let mut inv = Inventory::default(); // capacity = 0.0 = unlimited
        let stored = inv.deposit(0, 1_000_000.0);
        assert_eq!(stored, 1_000_000.0);
    }
}
