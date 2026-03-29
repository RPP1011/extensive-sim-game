use serde::{Deserialize, Serialize};

use super::fidelity::Fidelity;
use super::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// WorldState — the complete snapshot, immutable during COMPUTE phase
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub tick: u64,
    pub rng_state: u64,

    /// All entities (NPCs, monsters, buildings, projectiles).
    pub entities: Vec<Entity>,

    /// Active local grids (settlements, encounter zones).
    pub grids: Vec<LocalGrid>,

    /// Regional data (monster populations, faction control).
    pub regions: Vec<RegionState>,

    /// Per-settlement economy (stockpiles, prices, treasury).
    pub settlements: Vec<SettlementState>,

    /// Global economy (total gold supply, trade routes).
    pub economy: EconomyState,
}

impl WorldState {
    pub fn new(seed: u64) -> Self {
        Self {
            tick: 0,
            rng_state: seed,
            entities: Vec::new(),
            grids: Vec::new(),
            regions: Vec::new(),
            settlements: Vec::new(),
            economy: EconomyState::default(),
        }
    }

    pub fn entity(&self, id: u32) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entity_mut(&mut self, id: u32) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn settlement(&self, id: u32) -> Option<&SettlementState> {
        self.settlements.iter().find(|s| s.id == id)
    }

    pub fn grid(&self, id: u32) -> Option<&LocalGrid> {
        self.grids.iter().find(|g| g.id == id)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    pub team: WorldTeam,
    /// World position (always valid).
    pub pos: (f32, f32),
    /// Which local grid this entity is on, if any.
    pub grid_id: Option<u32>,
    /// Position within local grid (only valid when grid_id is Some).
    pub local_pos: Option<(f32, f32)>,
    /// Whether this entity is alive/active.
    pub alive: bool,

    // --- Combat state (used at High fidelity) ---
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

    // --- NPC-specific ---
    pub npc: Option<NpcData>,
}

impl Entity {
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
        }
    }

    pub fn new_monster(id: u32, pos: (f32, f32), level: u32) -> Self {
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
            npc: None,
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
// NpcData — NPC-specific economic/social data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcData {
    /// Links to campaign Adventurer id.
    pub adventurer_id: u32,
    pub gold: f32,
    pub home_settlement_id: Option<u32>,
    pub economic_intent: EconomicIntent,
    pub price_knowledge: Vec<PriceReport>,
    pub carried_goods: [f32; NUM_COMMODITIES],
    pub class_tags: Vec<String>,
    /// What commodities this NPC produces (commodity_index, rate_per_tick).
    pub behavior_production: Vec<(usize, f32)>,
}

impl Default for NpcData {
    fn default() -> Self {
        Self {
            adventurer_id: 0,
            gold: 0.0,
            home_settlement_id: None,
            economic_intent: EconomicIntent::Idle,
            price_knowledge: Vec::new(),
            carried_goods: [0.0; NUM_COMMODITIES],
            class_tags: Vec::new(),
            behavior_production: Vec::new(),
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceReport {
    pub settlement_id: u32,
    pub prices: [f32; NUM_COMMODITIES],
    pub tick_observed: u64,
}

// ---------------------------------------------------------------------------
// LocalGrid — a spatial grid for a settlement or encounter zone
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalGrid {
    pub id: u32,
    pub fidelity: Fidelity,
    pub center: (f32, f32),
    pub radius: f32,
    /// Entity IDs currently on this grid.
    pub entity_ids: Vec<u32>,
}

impl LocalGrid {
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
    pub monster_density: f32,
    pub faction_id: Option<u32>,
    pub threat_level: f32,
}

// ---------------------------------------------------------------------------
// EconomyState — global economy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EconomyState {
    pub total_gold_supply: f32,
    pub total_commodities: [f32; NUM_COMMODITIES],
}
