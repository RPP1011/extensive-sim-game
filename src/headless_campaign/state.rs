//! Campaign state for headless simulation.
//!
//! All types are `Clone + Debug` for MCTS tree search (cheap state cloning,
//! deterministic stepping). The game world advances via fixed 100ms ticks.

use serde::{Deserialize, Serialize};

use super::config::CampaignConfig;

/// Fixed tick duration in milliseconds, matching the combat sim.
pub const CAMPAIGN_TICK_MS: u32 = 100;

// ---------------------------------------------------------------------------
// Top-level state
// ---------------------------------------------------------------------------

/// Complete campaign state as a plain, Clone-able struct.
///
/// Designed for MCTS: cheap to clone, deterministic to step.
/// No Bevy ECS — all state is explicit fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignState {
    // --- Time ---
    /// Monotonic tick counter (increments by 1 each step).
    pub tick: u64,
    /// Total elapsed game time in milliseconds (`tick * CAMPAIGN_TICK_MS`).
    pub elapsed_ms: u64,

    // --- Guild ---
    pub guild: GuildState,

    // --- World ---
    pub overworld: OverworldState,
    pub factions: Vec<FactionState>,
    pub diplomacy: DiplomacyMatrix,

    // --- Units ---
    pub adventurers: Vec<Adventurer>,
    pub parties: Vec<Party>,

    // --- Quests ---
    /// Available requests on the guild board (not yet accepted).
    pub request_board: Vec<QuestRequest>,
    /// Accepted, in-progress quests.
    pub active_quests: Vec<ActiveQuest>,
    /// Completed quest history (bounded ring buffer in practice).
    pub completed_quests: Vec<CompletedQuest>,

    // --- Battles ---
    pub active_battles: Vec<BattleState>,

    // --- Progression ---
    /// Unlocked abilities/buffs. Effect-property vectors, NOT identity-indexed.
    pub unlocks: Vec<UnlockInstance>,
    pub progression_history: Vec<ProgressionEvent>,

    // --- Choices ---
    /// Pending choice events requiring player decision.
    /// If deadline passes, the `default_option` is auto-selected.
    pub pending_choices: Vec<ChoiceEvent>,

    // --- Hook state ---
    /// Tracks which quest hooks have fired and when.
    #[serde(default)]
    pub hook_state: super::quest_hooks::HookState,

    // --- Narrative ---
    pub event_log: Vec<CampaignEvent>,
    pub npc_relationships: Vec<NpcRelationship>,

    // --- RNG ---
    /// LCG state for deterministic randomness. Isolated from combat sim RNG.
    pub rng: u64,

    // --- ID counters ---
    pub next_quest_id: u32,
    pub next_party_id: u32,
    pub next_battle_id: u32,
    pub next_unlock_id: u32,
    pub next_event_id: u32,

    // --- Combat mode ---
    pub combat_mode: CombatMode,

    // --- Campaign phase ---
    /// Current phase of the campaign.
    pub phase: CampaignPhase,

    /// Available starting packages. Loaded from data files.
    /// Emptied after the player chooses one.
    pub available_starting_choices: Vec<StartingChoice>,

    // --- Player character ---
    /// The player's created character. Built up during character creation.
    #[serde(default)]
    pub player_character: Option<PlayerCharacter>,

    // --- Configuration ---
    /// All tunable balance parameters. Systems read from this.
    pub config: CampaignConfig,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombatMode {
    /// Predict outcomes via small MLP (microseconds). Used during MCTS.
    Oracle,
    /// Run full deterministic combat sim (milliseconds). Used for evaluation.
    FullSim,
}

// ---------------------------------------------------------------------------
// Guild
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuildState {
    pub gold: f32,
    /// Aggregate supply pool at the guild base.
    pub supplies: f32,
    /// 0–100. Affects request quality and pricing.
    pub reputation: f32,
    pub base: BaseState,
    /// How many quests can run simultaneously.
    pub active_quest_capacity: usize,
    /// Bitmask of known unlock categories.
    pub unlock_bitmask: u64,
    /// Inventory of unequipped gear.
    pub inventory: Vec<InventoryItem>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseState {
    pub base_type: BaseType,
    /// Overworld coordinates (x, y).
    pub position: (f32, f32),
    pub upgrade_slots: Vec<UpgradeSlot>,
    pub defensive_strength: f32,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseType {
    Camp,
    Fixed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpgradeSlot {
    pub slot_type: UpgradeSlotType,
    pub installed: Option<String>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpgradeSlotType {
    Defense,
    Storage,
    Training,
    Information,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InventoryItem {
    pub id: u32,
    pub name: String,
    pub slot: EquipmentSlot,
    pub quality: f32,
    pub stat_bonuses: StatBonuses,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StatBonuses {
    pub hp_bonus: f32,
    pub attack_bonus: f32,
    pub defense_bonus: f32,
    pub speed_bonus: f32,
}

// ---------------------------------------------------------------------------
// Adventurer
// ---------------------------------------------------------------------------

/// An adventurer in the guild. Maps from existing `HeroCompanion`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adventurer {
    pub id: u32,
    pub name: String,
    /// Archetype from hero_templates (e.g. "knight", "ranger", "mage").
    pub archetype: String,
    pub level: u32,
    pub xp: u32,

    // --- Combat stats summary ---
    pub stats: AdventurerStats,
    pub equipment: Equipment,
    pub traits: Vec<String>,

    // --- Condition ---
    pub status: AdventurerStatus,
    /// 0–100.
    pub loyalty: f32,
    /// 0–100.
    pub stress: f32,
    /// 0–100.
    pub fatigue: f32,
    /// 0–100. ≥90 = incapacitated.
    pub injury: f32,
    /// 0–100.
    pub resolve: f32,
    /// 0–100.
    pub morale: f32,

    // --- Assignment ---
    /// Which party this adventurer belongs to, if any.
    pub party_id: Option<u32>,
    /// NPC relationship score to the guild (−100 to 100).
    pub guild_relationship: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdventurerStats {
    pub max_hp: f32,
    pub attack: f32,
    pub defense: f32,
    pub speed: f32,
    pub ability_power: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Equipment {
    pub weapon: Option<u32>,
    pub offhand: Option<u32>,
    pub chest: Option<u32>,
    pub boots: Option<u32>,
    pub accessory: Option<u32>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdventurerStatus {
    Idle,
    /// In the candidate pool for a quest, not yet dispatched.
    Assigned,
    /// En route to quest location.
    Traveling,
    /// At quest location, executing.
    OnMission,
    /// In active battle.
    Fighting,
    /// Recovering from injury, unavailable.
    Injured,
    Dead,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquipmentSlot {
    Weapon,
    Offhand,
    Chest,
    Boots,
    Accessory,
}

// ---------------------------------------------------------------------------
// Party
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Party {
    pub id: u32,
    /// Adventurer IDs in this party.
    pub member_ids: Vec<u32>,
    /// Continuous overworld position (x, y).
    pub position: (f32, f32),
    /// Where the party is heading. `None` = stationary.
    pub destination: Option<(f32, f32)>,
    /// Tiles per second.
    pub speed: f32,
    pub status: PartyStatus,
    /// 0–100.
    pub supply_level: f32,
    /// Aggregate party morale (mean of members).
    pub morale: f32,
    /// Which quest this party is assigned to.
    pub quest_id: Option<u32>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyStatus {
    Idle,
    Traveling,
    OnMission,
    Fighting,
    /// Heading back to guild base.
    Returning,
}

// ---------------------------------------------------------------------------
// Quests
// ---------------------------------------------------------------------------

/// A request on the guild board (not yet accepted).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestRequest {
    pub id: u32,
    pub source_faction_id: Option<usize>,
    pub source_area_id: Option<usize>,
    pub quest_type: QuestType,
    /// Estimated difficulty (0–100).
    pub threat_level: f32,
    pub reward: QuestReward,
    /// Distance from guild base in tiles.
    pub distance: f32,
    /// Location where the quest takes place.
    pub target_position: (f32, f32),
    /// Expires at this elapsed_ms if not accepted.
    pub deadline_ms: u64,
    pub description: String,
    /// When it appeared on the board.
    pub arrived_at_ms: u64,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestType {
    Combat,
    Exploration,
    Diplomatic,
    Escort,
    Rescue,
    Gather,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuestReward {
    pub gold: f32,
    pub reputation: f32,
    pub relation_faction_id: Option<usize>,
    pub relation_change: f32,
    pub supply_reward: f32,
    /// Whether this quest has a chance of dropping equipment.
    pub potential_loot: bool,
}

/// An accepted, in-progress quest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveQuest {
    pub id: u32,
    /// Original request data (preserved for reward application).
    pub request: QuestRequest,
    pub status: ActiveQuestStatus,
    /// Adventurer IDs the player assigned to the candidate pool.
    pub assigned_pool: Vec<u32>,
    /// Party formed by the NPC system after dispatch.
    pub dispatched_party_id: Option<u32>,
    /// Milliseconds since the quest was accepted.
    pub elapsed_ms: u64,
    /// Events that happened during the quest.
    pub events: Vec<QuestEvent>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveQuestStatus {
    /// Pool assigned, not yet dispatched.
    Preparing,
    /// Party formed, traveling to quest location.
    Dispatched,
    /// At location, executing quest objectives.
    InProgress,
    /// Battle triggered.
    InCombat,
    /// Completed, party heading back.
    Returning,
    /// Going sideways — support actions available.
    NeedsSupport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestEvent {
    pub tick: u64,
    pub description: String,
}

/// A quest that has been completed (history record).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletedQuest {
    pub id: u32,
    pub quest_type: QuestType,
    pub result: QuestResult,
    pub reward_applied: QuestReward,
    pub completed_at_ms: u64,
    pub party_id: u32,
    pub casualties: u32,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestResult {
    Victory,
    Defeat,
    Abandoned,
}

// ---------------------------------------------------------------------------
// Battles
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BattleState {
    pub id: u32,
    /// Which quest triggered this battle.
    pub quest_id: u32,
    /// Our party.
    pub party_id: u32,
    pub location: (f32, f32),

    // --- Summarized combat state ---
    /// Aggregate HP remaining for our party (0–1).
    pub party_health_ratio: f32,
    /// Aggregate HP remaining for enemies (0–1).
    pub enemy_health_ratio: f32,
    /// Enemy composition strength estimate.
    pub enemy_strength: f32,
    pub elapsed_ticks: u64,
    /// Combat oracle prediction (−1 = certain defeat, +1 = certain victory).
    pub predicted_outcome: f32,
    pub status: BattleStatus,

    // --- Support state ---
    pub runner_sent: bool,
    pub mercenary_hired: bool,
    pub rescue_called: bool,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BattleStatus {
    Active,
    Victory,
    Defeat,
    Retreat,
}

// ---------------------------------------------------------------------------
// Overworld
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverworldState {
    pub regions: Vec<Region>,
    /// Named locations: settlements, dungeons, ruins, outposts.
    pub locations: Vec<Location>,
    /// Global threat level (0–100).
    pub global_threat_level: f32,
    /// Overall campaign progress toward endgame (0–1).
    pub campaign_progress: f32,
    /// Endgame calamity, if one has been selected.
    pub endgame_calamity: Option<CalamityType>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Region {
    pub id: usize,
    pub name: String,
    pub owner_faction_id: usize,
    pub neighbors: Vec<usize>,
    /// 0–100.
    pub unrest: f32,
    /// 0–100.
    pub control: f32,
    /// Regional threat level (0–100).
    pub threat_level: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Location {
    pub id: usize,
    pub name: String,
    pub position: (f32, f32),
    pub location_type: LocationType,
    /// 0–100.
    pub threat_level: f32,
    /// 0–100.
    pub resource_availability: f32,
    pub faction_owner: Option<usize>,
    /// Whether the guild has scouted this location.
    pub scouted: bool,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocationType {
    Settlement,
    Wilderness,
    Dungeon,
    Ruin,
    Outpost,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CalamityType {
    /// An aggressive faction with a super-ruler unit.
    AggressiveFaction { faction_id: usize },
    /// A major monster threatening the world.
    MajorMonster { name: String, strength: f32 },
    /// A flood of minor crises overwhelming the region.
    CrisisFlood,
    /// A conquest victory condition.
    Conquest,
}

// ---------------------------------------------------------------------------
// Factions & Diplomacy
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionState {
    pub id: usize,
    pub name: String,
    /// Relationship to the guild (−100 to 100).
    pub relationship_to_guild: f32,
    pub military_strength: f32,
    /// Strength cap — factions rebuild toward this over time.
    #[serde(default = "default_max_strength")]
    pub max_military_strength: f32,
    pub territory_size: usize,
    pub diplomatic_stance: DiplomaticStance,
    /// Whether this faction is in a coalition with the guild.
    #[serde(default)]
    pub coalition_member: bool,
    /// Faction-to-faction war targets (faction IDs this faction is at war with).
    #[serde(default)]
    pub at_war_with: Vec<usize>,
    /// Whether this faction has an adventurer guild (NPC guild for coalition).
    #[serde(default)]
    pub has_guild: bool,
    /// NPC guild adventurer count (if has_guild).
    #[serde(default)]
    pub guild_adventurer_count: u32,
    /// Recent actions taken by this faction (bounded).
    pub recent_actions: Vec<FactionActionRecord>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiplomaticStance {
    /// Positive relations, willing to trade and cooperate.
    Friendly,
    /// Default stance, no strong feelings.
    Neutral,
    /// Negative relations, building military, may declare war.
    Hostile,
    /// Actively attacking the guild's territory.
    AtWar,
    /// Allied — shares intel, provides military aid, joint operations.
    Coalition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionActionRecord {
    pub tick: u64,
    pub action: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiplomacyMatrix {
    /// Faction-to-faction relations. `relations[a][b]` = how faction a feels
    /// about faction b (signed).
    pub relations: Vec<Vec<i32>>,
    /// Which faction the guild is affiliated with.
    pub guild_faction_id: usize,
}

// ---------------------------------------------------------------------------
// Progression Unlocks
// ---------------------------------------------------------------------------

/// NOT identity-indexed. Described by mechanical properties so the model
/// can handle LFM-generated novel abilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnlockInstance {
    pub id: u32,
    pub category: UnlockCategory,
    pub properties: UnlockProperties,
    pub name: String,
    pub description: String,
    /// Whether this unlock is available for use.
    pub active: bool,
    /// Remaining cooldown in milliseconds (0 = ready).
    pub cooldown_remaining_ms: u64,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnlockCategory {
    /// Scout networks, alarms — reduces uncertainty.
    Information,
    /// Send potion, call favor — direct intervention.
    ActiveAbility,
    /// Improved coordination, last stands.
    PassiveBuff,
    /// Better supply chains, cheaper hiring.
    Economic,
}

/// Mechanical properties — this is what the model sees, not the name.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnlockProperties {
    /// 0 = passive (always on).
    pub cooldown_ms: u64,
    pub target_type: TargetType,
    /// Effect strength (interpretation depends on category).
    pub magnitude: f32,
    /// 0 = instant or permanent.
    pub duration_ms: u64,
    /// Gold cost to use.
    pub resource_cost: f32,
    pub is_passive: bool,
    /// Learned category features for the entity encoder.
    pub category_embedding: [f32; 8],
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetType {
    GuildSelf,
    Adventurer,
    Party,
    Quest,
    Area,
    Faction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgressionEvent {
    pub tick: u64,
    pub unlock_id: u32,
    pub description: String,
}

// ---------------------------------------------------------------------------
// NPC Relationships
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpcRelationship {
    pub npc_id: u32,
    pub npc_name: String,
    pub npc_type: NpcType,
    /// −100 to 100.
    pub relationship_score: f32,
    /// Last interaction timestamp (elapsed_ms).
    pub last_interaction_ms: u64,
    /// Whether this NPC can rescue for free (relationship-gated).
    pub rescue_available: bool,
    /// Gold cost if rescue is not free.
    pub rescue_cost: f32,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NpcType {
    Adventurer,
    FactionLeader,
    Merchant,
    Informant,
    Mercenary,
}

// ---------------------------------------------------------------------------
// Narrative events
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignEvent {
    pub id: u32,
    pub tick: u64,
    pub description: String,
}

// ---------------------------------------------------------------------------
// Campaign outcome
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignOutcome {
    /// Player defeated the endgame calamity.
    Victory,
    /// Guild collapsed (no adventurers, no gold, or calamity won).
    Defeat,
    /// Hit the maximum tick limit.
    Timeout,
}

// ---------------------------------------------------------------------------
// Campaign phase
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignPhase {
    /// Character creation — backstory events playing out.
    /// Systems don't tick. Only creation choices are valid.
    CharacterCreation,
    /// Starting package selection (final creation step).
    ChoosingStartingPackage,
    /// Normal gameplay.
    Playing,
}

impl Default for CampaignPhase {
    fn default() -> Self {
        Self::CharacterCreation
    }
}

// ---------------------------------------------------------------------------
// Player character
// ---------------------------------------------------------------------------

/// The player's named character — an adventurer with a special flag.
/// Built incrementally during character creation from backstory choices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerCharacter {
    /// The adventurer ID in the adventurers list.
    pub adventurer_id: u32,
    /// Player-chosen name.
    pub name: String,
    /// Origin backstory (e.g. "Noble Exile", "Frontier Settler").
    pub origin: String,
    /// Accumulated backstory summary from creation choices.
    pub backstory: Vec<String>,
    /// Personal goal — tracked but doesn't end the game.
    pub goal: Option<PersonalGoal>,
    /// Whether the PC is alive (if dead, leadership transfers).
    pub alive: bool,
    /// Successor adventurer ID (appointed second-in-command).
    pub successor_id: Option<u32>,
}

/// A personal goal chosen during character creation.
/// Cosmetic to the world — just tracks progress toward a player ambition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalGoal {
    pub name: String,
    pub description: String,
    /// Conditions to check against CampaignState.
    pub conditions: Vec<GoalCondition>,
    /// Has this goal been achieved? (doesn't end the game)
    pub achieved: bool,
    /// Can this goal become impossible?
    pub fail_condition: Option<String>,
    pub failed: bool,
}

/// A condition for goal completion, checked against game state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GoalCondition {
    /// Faction relationship above threshold.
    FactionRelation { faction_id: usize, min_relation: f32 },
    /// Guild reputation above threshold.
    Reputation { min_reputation: f32 },
    /// Gold above threshold.
    Gold { min_gold: f32 },
    /// Region control above threshold.
    RegionControl { region_id: usize, min_control: f32 },
    /// Quest victory count above threshold.
    QuestsCompleted { min_count: usize },
    /// Adventurer count above threshold.
    AdventurerCount { min_count: usize },
    /// PC level above threshold.
    PlayerLevel { min_level: u32 },
    /// All factions at Friendly or Coalition.
    AllFactionsAllied,
    /// Specific trait acquired by PC.
    HasTrait { trait_name: String },
}

// ---------------------------------------------------------------------------
// Choice events
// ---------------------------------------------------------------------------

/// A branching decision presented to the player.
///
/// Generated by quest outcomes, NPC encounters, progression milestones,
/// faction diplomacy, and random events. If the deadline passes without
/// the player choosing, `default_option` is auto-selected (the path of
/// least investment/commitment).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChoiceEvent {
    pub id: u32,
    /// What category generated this choice.
    pub source: ChoiceSource,
    /// Display text describing the situation.
    pub prompt: String,
    /// Available options (2–4).
    pub options: Vec<ChoiceOption>,
    /// Index into `options` that is auto-selected on deadline expiry.
    /// Should be the lowest-investment/safest/most passive option.
    pub default_option: usize,
    /// Deadline in elapsed_ms. `None` = no deadline (blocks until chosen).
    pub deadline_ms: Option<u64>,
    /// When this choice was created.
    pub created_at_ms: u64,
}

/// What system generated this choice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChoiceSource {
    /// Quest reached a branching point.
    QuestBranch { quest_id: u32 },
    /// Progression milestone — pick one of N unlocks.
    ProgressionUnlock,
    /// NPC encounter (merchant deal, faction proposal, etc).
    NpcEncounter { npc_id: u32 },
    /// Flashpoint approach selection.
    FlashpointIntent { region_id: usize },
    /// Random world event.
    WorldEvent,
}

/// A single option in a choice event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChoiceOption {
    /// Short label (e.g. "Stealth Approach", "Invest in Scout Network").
    pub label: String,
    /// Description of what happens if chosen.
    pub description: String,
    /// Effects applied when this option is selected.
    pub effects: Vec<ChoiceEffect>,
}

/// A concrete effect applied when a choice option is selected.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChoiceEffect {
    /// Add/remove gold.
    Gold(f32),
    /// Add/remove supplies.
    Supplies(f32),
    /// Change reputation.
    Reputation(f32),
    /// Change faction relationship.
    FactionRelation { faction_id: usize, delta: f32 },
    /// Grant an unlock.
    GrantUnlock(UnlockInstance),
    /// Modify quest threat level.
    ModifyQuestThreat { quest_id: u32, multiplier: f32 },
    /// Modify quest reward.
    ModifyQuestReward { quest_id: u32, gold_bonus: f32, rep_bonus: f32 },
    /// Add an adventurer to the guild.
    AddAdventurer(Adventurer),
    /// Heal/injure an adventurer.
    ModifyAdventurerInjury { adventurer_id: u32, delta: f32 },
    /// Change adventurer loyalty.
    ModifyAdventurerLoyalty { adventurer_id: u32, delta: f32 },
    /// Add item to inventory.
    AddItem(InventoryItem),
    /// Set a quest to a specific status.
    SetQuestStatus { quest_id: u32, status: ActiveQuestStatus },
    /// Narrative flavor — no mechanical effect.
    Narrative(String),
}

// ---------------------------------------------------------------------------
// Starting choice
// ---------------------------------------------------------------------------

/// A starting package the player can choose at tick 0.
///
/// Loaded from data files (TOML/JSON). Each package defines its own
/// adventurers, gold bonus, supply bonus, and flavor text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StartingChoice {
    /// Display name (e.g. "War Chest", "The Veteran", "Fresh Recruits").
    pub name: String,
    /// Description shown to the player.
    pub description: String,
    /// Adventurers granted by this package.
    pub adventurers: Vec<Adventurer>,
    /// Gold added to the guild's starting gold.
    pub gold_bonus: f32,
    /// Supplies added to the guild's starting supplies.
    pub supply_bonus: f32,
    /// Starting reputation modifier.
    pub reputation_bonus: f32,
    /// Optional starting inventory items.
    pub items: Vec<InventoryItem>,
}

impl StartingChoice {
    /// Load all starting packages from a directory of TOML files.
    pub fn load_from_dir(dir: &std::path::Path) -> Result<Vec<Self>, String> {
        let mut choices = Vec::new();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Read error: {}", e))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                let content = std::fs::read_to_string(&path)
                    .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
                let choice: StartingChoice = toml::from_str(&content)
                    .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;
                choices.push(choice);
            }
        }
        Ok(choices)
    }
}

// ---------------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------------

/// Deterministic LCG (same as combat sim). Returns value in [0, u32::MAX].
#[inline]
pub fn lcg_next(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

fn default_max_strength() -> f32 {
    100.0
}

/// Returns a float in [0, 1).
#[inline]
pub fn lcg_f32(state: &mut u64) -> f32 {
    lcg_next(state) as f32 / u32::MAX as f32
}

// ---------------------------------------------------------------------------
// Test campaign constructors
// ---------------------------------------------------------------------------

impl CampaignState {
    /// Create a campaign with world content but no adventurers.
    ///
    /// The player must issue `ChooseStartingPackage` as their first action
    /// to receive adventurers and starting resources. Until then, the sim
    /// does not tick.
    ///
    /// Includes 3 regions, 2 factions, 4 locations, 2 NPC relationships.
    pub fn default_test_campaign(seed: u64) -> Self {
        // These adventurers are NOT placed in the guild — they're templates
        // for the starting choice packages. The guild starts empty.
        let _adventurer_templates = vec![
            Adventurer {
                id: 1,
                name: "Kael the Swift".into(),
                archetype: "ranger".into(),
                level: 3,
                xp: 0,
                stats: AdventurerStats {
                    max_hp: 80.0,
                    attack: 18.0,
                    defense: 10.0,
                    speed: 14.0,
                    ability_power: 8.0,
                },
                equipment: Equipment::default(),
                traits: vec!["keen_eye".into()],
                status: AdventurerStatus::Idle,
                loyalty: 70.0,
                stress: 10.0,
                fatigue: 5.0,
                injury: 0.0,
                resolve: 60.0,
                morale: 80.0,
                party_id: None,
                guild_relationship: 50.0,
            },
            Adventurer {
                id: 2,
                name: "Sera the Stalwart".into(),
                archetype: "knight".into(),
                level: 4,
                xp: 0,
                stats: AdventurerStats {
                    max_hp: 120.0,
                    attack: 14.0,
                    defense: 20.0,
                    speed: 8.0,
                    ability_power: 5.0,
                },
                equipment: Equipment::default(),
                traits: vec!["shield_wall".into()],
                status: AdventurerStatus::Idle,
                loyalty: 80.0,
                stress: 5.0,
                fatigue: 10.0,
                injury: 0.0,
                resolve: 75.0,
                morale: 85.0,
                party_id: None,
                guild_relationship: 60.0,
            },
            Adventurer {
                id: 3,
                name: "Mira the Wise".into(),
                archetype: "mage".into(),
                level: 3,
                xp: 0,
                stats: AdventurerStats {
                    max_hp: 60.0,
                    attack: 8.0,
                    defense: 6.0,
                    speed: 10.0,
                    ability_power: 25.0,
                },
                equipment: Equipment::default(),
                traits: vec!["arcane_focus".into()],
                status: AdventurerStatus::Idle,
                loyalty: 65.0,
                stress: 15.0,
                fatigue: 8.0,
                injury: 0.0,
                resolve: 55.0,
                morale: 75.0,
                party_id: None,
                guild_relationship: 40.0,
            },
            Adventurer {
                id: 4,
                name: "Bron the Healer".into(),
                archetype: "cleric".into(),
                level: 2,
                xp: 0,
                stats: AdventurerStats {
                    max_hp: 70.0,
                    attack: 6.0,
                    defense: 12.0,
                    speed: 9.0,
                    ability_power: 20.0,
                },
                equipment: Equipment::default(),
                traits: vec!["healing_hands".into()],
                status: AdventurerStatus::Idle,
                loyalty: 85.0,
                stress: 8.0,
                fatigue: 3.0,
                injury: 0.0,
                resolve: 70.0,
                morale: 90.0,
                party_id: None,
                guild_relationship: 65.0,
            },
        ];

        // Load world template: pick one based on seed from assets/world_templates/,
        // falling back to the built-in default if no templates are found.
        let world_template = Self::load_world_template(seed);

        let regions = world_template.regions;
        let locations = world_template.locations;
        let factions = world_template.factions;
        let diplomacy = world_template.diplomacy;
        let npc_relationships = world_template.npc_relationships;
        let global_threat_level = world_template.global_threat_level;

        CampaignState {
            tick: 0,
            elapsed_ms: 0,
            guild: GuildState {
                gold: 100.0,
                supplies: 80.0,
                reputation: 25.0,
                base: BaseState {
                    base_type: BaseType::Camp,
                    position: (0.0, 0.0),
                    upgrade_slots: Vec::new(),
                    defensive_strength: 10.0,
                },
                active_quest_capacity: 3,
                unlock_bitmask: 0,
                inventory: Vec::new(),
            },
            overworld: OverworldState {
                regions,
                locations,
                global_threat_level,
                campaign_progress: 0.0,
                endgame_calamity: None,
            },
            factions,
            diplomacy,
            adventurers: Vec::new(), // Empty until ChooseStartingPackage
            parties: Vec::new(),
            request_board: Vec::new(),
            active_quests: Vec::new(),
            completed_quests: Vec::new(),
            active_battles: Vec::new(),
            pending_choices: Vec::new(),
            hook_state: super::quest_hooks::HookState::default(),
            unlocks: Vec::new(),
            progression_history: Vec::new(),
            event_log: Vec::new(),
            npc_relationships,
            rng: seed,
            next_quest_id: 1,
            next_party_id: 1,
            next_battle_id: 1,
            next_unlock_id: 1,
            next_event_id: 1,
            combat_mode: CombatMode::Oracle,
            phase: CampaignPhase::ChoosingStartingPackage,
            available_starting_choices: Self::load_or_default_starting_choices(),
            player_character: None,
            config: CampaignConfig::default(),
        }
    }

    /// Load a world template based on seed from `assets/world_templates/`,
    /// falling back to the built-in default frontier template.
    fn load_world_template(seed: u64) -> super::world_templates::WorldTemplate {
        let dir = std::path::Path::new("assets/world_templates");
        if dir.exists() {
            match super::world_templates::WorldTemplate::load_from_dir(dir) {
                Ok(templates) if !templates.is_empty() => {
                    let idx = (seed as usize) % templates.len();
                    return templates.into_iter().nth(idx).unwrap();
                }
                Ok(_) => {} // empty dir, fall through
                Err(_) => {} // parse error, fall through
            }
        }
        super::world_templates::WorldTemplate::default_frontier()
    }

    /// Load starting choices from `assets/starting_packages/` or return defaults.
    fn load_or_default_starting_choices() -> Vec<StartingChoice> {
        let dir = std::path::Path::new("assets/starting_packages");
        if dir.exists() {
            match StartingChoice::load_from_dir(dir) {
                Ok(choices) if !choices.is_empty() => return choices,
                Ok(_) => {} // empty dir, fall through
                Err(_) => {} // parse error, fall through
            }
        }

        // Fallback: three inline defaults
        vec![
            StartingChoice {
                name: "War Chest".into(),
                description: "Extra gold, two mid-level adventurers".into(),
                adventurers: vec![
                    Adventurer {
                        id: 1, name: "Finn".into(), archetype: "ranger".into(), level: 2, xp: 0,
                        stats: AdventurerStats { max_hp: 70.0, attack: 14.0, defense: 8.0, speed: 12.0, ability_power: 6.0 },
                        equipment: Equipment::default(), traits: Vec::new(),
                        status: AdventurerStatus::Idle,
                        loyalty: 60.0, stress: 10.0, fatigue: 5.0, injury: 0.0, resolve: 50.0, morale: 70.0,
                        party_id: None, guild_relationship: 40.0,
                    },
                    Adventurer {
                        id: 2, name: "Greta".into(), archetype: "knight".into(), level: 2, xp: 0,
                        stats: AdventurerStats { max_hp: 100.0, attack: 10.0, defense: 16.0, speed: 7.0, ability_power: 4.0 },
                        equipment: Equipment::default(), traits: Vec::new(),
                        status: AdventurerStatus::Idle,
                        loyalty: 65.0, stress: 5.0, fatigue: 8.0, injury: 0.0, resolve: 55.0, morale: 75.0,
                        party_id: None, guild_relationship: 45.0,
                    },
                ],
                gold_bonus: 150.0, supply_bonus: 30.0, reputation_bonus: 5.0, items: Vec::new(),
            },
            StartingChoice {
                name: "The Veteran".into(),
                description: "One overleveled adventurer with a cursed arm".into(),
                adventurers: vec![
                    Adventurer {
                        id: 1, name: "Gareth Ironhand".into(), archetype: "knight".into(), level: 6, xp: 0,
                        stats: AdventurerStats { max_hp: 150.0, attack: 22.0, defense: 25.0, speed: 6.0, ability_power: 8.0 },
                        equipment: Equipment::default(), traits: vec!["cursed_arm".into()],
                        status: AdventurerStatus::Idle,
                        loyalty: 40.0, stress: 30.0, fatigue: 20.0, injury: 15.0, resolve: 80.0, morale: 50.0,
                        party_id: None, guild_relationship: 25.0,
                    },
                ],
                gold_bonus: 0.0, supply_bonus: 0.0, reputation_bonus: 10.0, items: Vec::new(),
            },
            StartingChoice {
                name: "Fresh Recruits".into(),
                description: "Four level-1 rookies".into(),
                adventurers: vec![
                    Adventurer {
                        id: 1, name: "Alaric".into(), archetype: "ranger".into(), level: 1, xp: 0,
                        stats: AdventurerStats { max_hp: 60.0, attack: 12.0, defense: 6.0, speed: 13.0, ability_power: 5.0 },
                        equipment: Equipment::default(), traits: Vec::new(),
                        status: AdventurerStatus::Idle,
                        loyalty: 55.0, stress: 12.0, fatigue: 8.0, injury: 0.0, resolve: 45.0, morale: 65.0,
                        party_id: None, guild_relationship: 35.0,
                    },
                    Adventurer {
                        id: 2, name: "Brynn".into(), archetype: "mage".into(), level: 1, xp: 0,
                        stats: AdventurerStats { max_hp: 45.0, attack: 5.0, defense: 4.0, speed: 9.0, ability_power: 18.0 },
                        equipment: Equipment::default(), traits: Vec::new(),
                        status: AdventurerStatus::Idle,
                        loyalty: 50.0, stress: 15.0, fatigue: 5.0, injury: 0.0, resolve: 40.0, morale: 60.0,
                        party_id: None, guild_relationship: 30.0,
                    },
                    Adventurer {
                        id: 3, name: "Cira".into(), archetype: "cleric".into(), level: 1, xp: 0,
                        stats: AdventurerStats { max_hp: 55.0, attack: 4.0, defense: 8.0, speed: 8.0, ability_power: 15.0 },
                        equipment: Equipment::default(), traits: Vec::new(),
                        status: AdventurerStatus::Idle,
                        loyalty: 70.0, stress: 8.0, fatigue: 3.0, injury: 0.0, resolve: 50.0, morale: 75.0,
                        party_id: None, guild_relationship: 45.0,
                    },
                    Adventurer {
                        id: 4, name: "Daven".into(), archetype: "rogue".into(), level: 1, xp: 0,
                        stats: AdventurerStats { max_hp: 50.0, attack: 15.0, defense: 5.0, speed: 14.0, ability_power: 4.0 },
                        equipment: Equipment::default(), traits: Vec::new(),
                        status: AdventurerStatus::Idle,
                        loyalty: 45.0, stress: 18.0, fatigue: 10.0, injury: 0.0, resolve: 35.0, morale: 55.0,
                        party_id: None, guild_relationship: 25.0,
                    },
                ],
                gold_bonus: 20.0, supply_bonus: 10.0, reputation_bonus: 0.0, items: Vec::new(),
            },
        ]
    }

    /// Create a test campaign with custom config.
    pub fn with_config(seed: u64, config: CampaignConfig) -> Self {
        let mut state = Self::default_test_campaign(seed);
        // Apply starting state from config
        state.guild.gold = config.starting_state.gold;
        state.guild.supplies = config.starting_state.supplies;
        state.guild.reputation = config.starting_state.reputation;
        state.guild.base.defensive_strength = config.starting_state.base_defensive_strength;
        state.guild.active_quest_capacity = config.starting_state.active_quest_capacity;
        state.overworld.global_threat_level = config.starting_state.global_threat_level;
        state.config = config;
        state
    }
}
