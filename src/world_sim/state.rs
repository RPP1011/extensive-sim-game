use std::collections::HashMap;

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

    /// Relation graph between entities. Key = (entity_a, entity_b, kind), value = strength.
    pub relations: HashMap<(u32, u32, u8), f32>,

    /// World events log (recent events for system queries, bounded).
    pub world_events: Vec<WorldEvent>,
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
            factions: Vec::new(),
            quests: Vec::new(),
            quest_board: Vec::new(),
            adventurer_bonds: HashMap::new(),
            guild: GuildState::default(),
            chronicle: Vec::new(),
            relations: HashMap::new(),
            world_events: Vec::new(),
        }
    }

    pub fn entity(&self, id: u32) -> Option<&Entity> {
        // Fast path: if id matches index position, O(1).
        let i = id as usize;
        if i < self.entities.len() && self.entities[i].id == id {
            return Some(&self.entities[i]);
        }
        // Fallback: linear scan for non-contiguous IDs.
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entity_mut(&mut self, id: u32) -> Option<&mut Entity> {
        let i = id as usize;
        if i < self.entities.len() && self.entities[i].id == id {
            return Some(&mut self.entities[i]);
        }
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn settlement(&self, id: u32) -> Option<&SettlementState> {
        self.settlements.iter().find(|s| s.id == id)
    }

    pub fn grid(&self, id: u32) -> Option<&LocalGrid> {
        self.grids.iter().find(|g| g.id == id)
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

    pub fn settlement_mut(&mut self, id: u32) -> Option<&mut SettlementState> {
        self.settlements.iter_mut().find(|s| s.id == id)
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
    /// Experience points.
    pub xp: u32,
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
            morale: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            loyalty: 50.0,
            injury: 0.0,
            xp: 0,
            archetype: String::new(),
            party_id: None,
            faction_id: None,
            mood: 0,
            fears: Vec::new(),
            deeds: Vec::new(),
            resolve: 50.0,
            guild_relationship: 0.0,
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

    // --- Campaign system fields ---

    /// Owning faction, if any.
    pub faction_id: Option<u32>,
    /// 0–1. Threat from nearby monsters/enemies.
    pub threat_level: f32,
    /// 0–5. Building/upgrade level.
    pub infrastructure_level: f32,
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
            faction_id: None,
            threat_level: 0.0,
            infrastructure_level: 0.0,
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
