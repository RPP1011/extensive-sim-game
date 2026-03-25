//! Campaign actions, step results, and action validation.

use serde::{Deserialize, Serialize};

use super::state::*;

// ---------------------------------------------------------------------------
// Player actions
// ---------------------------------------------------------------------------

/// Every action the guild manager can take. The agent selects one per
/// decision point; between decisions the world ticks with `action = None`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CampaignAction {
    /// Do nothing. CfC integrates, world ticks forward.
    Wait,

    /// Rest the guild. Triggers recovery, presents pending progression,
    /// and allows adventurers to level up / gain abilities / accept classes.
    /// No quests can be dispatched during rest.
    Rest,

    /// Accept a quest from the request board.
    AcceptQuest { request_id: u32 },

    /// Decline a quest (removes it from the board).
    DeclineQuest { request_id: u32 },

    /// Assign an adventurer to a quest's candidate pool.
    AssignToPool { adventurer_id: u32, quest_id: u32 },

    /// Remove an adventurer from a quest's candidate pool.
    UnassignFromPool { adventurer_id: u32, quest_id: u32 },

    /// Dispatch a quest — NPC system selects final party from the pool.
    DispatchQuest { quest_id: u32 },

    /// Purchase supplies for a party (preparing or in-field).
    PurchaseSupplies { party_id: u32, amount: f32 },

    /// Train an adventurer (must be idle).
    TrainAdventurer {
        adventurer_id: u32,
        training_type: TrainingType,
    },

    /// Equip gear on an adventurer (must be idle or assigned).
    EquipGear { adventurer_id: u32, item_id: u32 },

    /// Send a runner to a party in the field.
    SendRunner {
        party_id: u32,
        payload: RunnerPayload,
    },

    /// Hire a mercenary for an active quest or battle.
    HireMercenary { quest_id: u32 },

    /// Call for rescue during an active battle.
    CallRescue { battle_id: u32 },

    /// Hire a scout for a location.
    HireScout { location_id: usize },

    /// Diplomatic action toward a faction.
    DiplomaticAction {
        faction_id: usize,
        action_type: DiplomacyActionType,
    },

    /// Propose a coalition with a friendly faction.
    ProposeCoalition { faction_id: usize },

    /// Request military aid from a coalition member.
    RequestCoalitionAid { faction_id: usize },

    /// Use an active unlock ability on a target.
    UseAbility {
        unlock_id: u32,
        target: AbilityTarget,
    },

    /// Set the guild's economic spending priority.
    SetSpendPriority { priority: SpendPriority },

    /// Choose starting package. Only valid at tick 0 before the campaign has
    /// been initialized. This is the player's first decision.
    ChooseStartingPackage { choice: StartingChoice },

    /// Respond to a pending choice event by selecting an option index.
    RespondToChoice { choice_id: u32, option_index: usize },

    /// Dispatch a party to intercept a traveling champion (Sleeping King crisis).
    /// The party will travel to the champion's current position and engage in combat.
    InterceptChampion {
        /// The party to send on the interception.
        party_id: u32,
        /// The champion adventurer ID to intercept.
        champion_id: u32,
    },
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingType {
    Combat,
    Exploration,
    Leadership,
    Survival,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RunnerPayload {
    Supplies(f32),
    Message,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiplomacyActionType {
    ImproveRelations,
    TradeAgreement,
    RequestAid,
    Threaten,
    ProposeCeasefire,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AbilityTarget {
    Adventurer(u32),
    Party(u32),
    Quest(u32),
    Location(usize),
    Faction(usize),
}

#[derive(Clone, Debug, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpendPriority {
    #[default]
    Balanced,
    SaveForEmergencies,
    InvestInGrowth,
    MilitaryFocus,
}

/// Returns a human-readable name for a diplomatic agreement type.
pub fn agreement_type_name(agreement: &super::state::DiplomaticAgreement) -> &'static str {
    match agreement {
        super::state::DiplomaticAgreement::TradeAgreement { .. } => "Trade Agreement",
        super::state::DiplomaticAgreement::NonAggressionPact { .. } => "Non-Aggression Pact",
        super::state::DiplomaticAgreement::MilitaryAlliance { .. } => "Military Alliance",
    }
}

// ---------------------------------------------------------------------------
// Action costs (gold)
// ---------------------------------------------------------------------------

/// Base costs for guild management actions.
pub const RUNNER_COST: f32 = 15.0;
pub const MERCENARY_COST: f32 = 50.0;
pub const SCOUT_COST: f32 = 10.0;
pub const TRAINING_COST: f32 = 20.0;
pub const RESCUE_BRIBE_COST: f32 = 80.0;
pub const SUPPLY_COST_PER_UNIT: f32 = 0.5;

// ---------------------------------------------------------------------------
// Step result
// ---------------------------------------------------------------------------

/// Output of a single `step_campaign` call.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CampaignStepResult {
    /// World events generated during this tick.
    pub events: Vec<WorldEvent>,

    /// Invariant violations detected.
    pub violations: Vec<String>,

    /// Campaign outcome (`None` while still in progress).
    pub outcome: Option<CampaignOutcome>,

    /// Granular per-system change tracking.
    pub deltas: StepDeltas,

    /// Feedback from the player action (if any).
    pub action_result: Option<ActionResult>,

    /// State after this step.
    pub tick: u64,
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// World events
// ---------------------------------------------------------------------------

/// Events that occurred during a single tick.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorldEvent {
    // --- Quests ---
    QuestRequestArrived {
        request_id: u32,
        quest_type: QuestType,
        threat_level: f32,
    },
    QuestRequestExpired {
        request_id: u32,
    },
    QuestDispatched {
        quest_id: u32,
        party_id: u32,
        member_count: usize,
    },
    QuestCompleted {
        quest_id: u32,
        result: QuestResult,
    },

    // --- Battles ---
    BattleStarted {
        battle_id: u32,
        quest_id: u32,
        party_health: f32,
        enemy_strength: f32,
    },
    BattleUpdate {
        battle_id: u32,
        party_health_ratio: f32,
        enemy_health_ratio: f32,
    },
    BattleEnded {
        battle_id: u32,
        result: BattleStatus,
    },

    // --- Parties ---
    PartyFormed {
        party_id: u32,
        member_ids: Vec<u32>,
    },
    PartyArrived {
        party_id: u32,
        location: (f32, f32),
    },
    PartyReturned {
        party_id: u32,
    },
    PartySupplyLow {
        party_id: u32,
        supply_level: f32,
    },

    // --- Adventurers ---
    AdventurerLevelUp {
        adventurer_id: u32,
        new_level: u32,
    },
    AdventurerInjured {
        adventurer_id: u32,
        injury_level: f32,
    },
    AdventurerRecovered {
        adventurer_id: u32,
    },
    AdventurerDeserted {
        adventurer_id: u32,
        reason: String,
    },
    AdventurerDied {
        adventurer_id: u32,
        cause: String,
    },

    // --- Support ---
    RunnerSent {
        party_id: u32,
        cost: f32,
    },
    MercenaryHired {
        quest_id: u32,
        cost: f32,
    },
    RescueCalled {
        battle_id: u32,
        cost: f32,
        npc_id: Option<u32>,
    },
    ScoutReport {
        location_id: usize,
        threat_level: f32,
    },
    /// Detailed scout report generated when a region crosses the visibility
    /// threshold (0.5). Contains faction strength, threats, and opportunities.
    RegionScoutReport {
        region_id: usize,
        details: ScoutReportDetails,
    },

    // --- Factions ---
    FactionActionTaken {
        faction_id: usize,
        action: String,
    },
    FactionRelationChanged {
        faction_id: usize,
        old: f32,
        new: f32,
    },
    RegionOwnerChanged {
        region_id: usize,
        old_owner: usize,
        new_owner: usize,
    },

    // --- Progression ---
    ProgressionUnlocked {
        unlock_id: u32,
        category: UnlockCategory,
        name: String,
    },

    // --- Economy ---
    GoldChanged {
        amount: f32,
        reason: String,
    },
    SupplyChanged {
        amount: f32,
        reason: String,
    },

    // --- Choices ---
    ChoicePresented {
        choice_id: u32,
        prompt: String,
        num_options: usize,
    },
    ChoiceResolved {
        choice_id: u32,
        option_index: usize,
        label: String,
        was_default: bool,
    },

    // --- Champions ---
    /// A guild party has intercepted a traveling champion.
    ChampionIntercepted {
        /// The guild party that intercepted.
        party_id: u32,
        /// The champion's traveling party.
        champion_party_id: u32,
        /// The champion adventurer ID.
        champion_id: u32,
    },
    /// A champion's traveling party arrived at the king's territory.
    ChampionArrived {
        champion_id: u32,
        champion_name: String,
        faction_id: usize,
    },

    // --- Bonds ---
    BondGrief {
        adventurer_id: u32,
        dead_id: u32,
        bond_strength: f32,
    },

    // --- Seasons ---
    SeasonChanged {
        new_season: super::state::Season,
    },

    // --- Random World Events ---
    RandomEvent {
        name: String,
        description: String,
    },

    // --- Campaign ---
    CalamityWarning {
        description: String,
    },
    CampaignMilestone {
        description: String,
    },

    // --- Buildings ---
    BuildingUpgraded {
        building: String,
        new_tier: u8,
        cost: f32,
    },

    // --- Diplomacy ---
    AgreementFormed { faction_a: usize, faction_b: usize, agreement_type: String },
    AgreementExpired { faction_a: usize, faction_b: usize, agreement_type: String },
    WarCeasefire { faction_a: usize, faction_b: usize, reason: String },
    FactionSplit { original_id: usize, new_faction_name: String },

    // --- Population ---
    PopulationEvent { region_id: usize, event_type: String, description: String },
    TaxCollected { region_id: usize, amount: f32 },
    RefugeesArrived { region: String, count: u32 },

    // --- Caravans ---
    CaravanCompleted { route_id: u32, gold_delivered: f32 },
    CaravanRaided { route_id: u32, gold_stolen: f32, raider_faction: String },

    // --- Retirement ---
    AdventurerRetired { adventurer_id: u32, name: String, legacy_type: String, bonus_description: String },
    LegacyBonusApplied { legacy_type: String, description: String },

    // --- Mentorship ---
    MentorshipCompleted { mentor_id: u32, apprentice_id: u32, xp_gained: f32 },
    SkillTransferred { from_id: u32, to_id: u32, tag: String },

    // --- Civil War ---
    CivilWarStarted { faction_id: usize, cause: String },
    CivilWarResolved { faction_id: usize, rebels_won: bool, description: String },

    // --- Legendary Deeds ---
    LegendaryDeedEarned { adventurer_id: u32, title: String, description: String },

    // --- Rumors ---
    RumorReceived { text: String, rumor_type: String },
    RumorInvestigated { rumor_id: u32, outcome: String },

    // --- War Exhaustion ---
    WarExhaustionMilestone { faction_id: usize, level: f32, description: String },

    // --- Guild Tiers ---
    GuildTierChanged { old_tier: String, new_tier: String, description: String },

    // --- Espionage ---
    IntelGathered { spy_id: u32, faction_id: usize, total_intel: f32 },
    SpyCaught { spy_id: u32, adventurer_id: u32, faction_id: usize },

    // --- Mercenaries ---
    MercenaryContractExpired { mercenary_id: u32, name: String },
    MercenaryDeserted { mercenary_id: u32, name: String },
    MercenaryBetrayedGuild { mercenary_id: u32, name: String, strength: f32 },

    // --- Black Market ---
    BlackMarketDeal { description: String, profit: f32 },
    BlackMarketDiscovered { reputation_lost: f32 },
    GamblingOutcome { adventurer_id: u32, amount: f32 },

    // --- Crafting ---
    ItemCrafted { name: String, quality: String },
    ResourceGathered { resource: String, amount: f32 },

    // --- Migration ---
    MigrationStarted { from: String, to: String, count: u32, cause: String },

    // --- Festivals ---
    FestivalStarted { faction: String, name: String },

    // --- Rivalries ---
    RivalryFormed { a: u32, b: u32, cause: String },
    RivalryDuel { challenger: u32, challenged: u32 },
    RivalryResolved { a: u32, b: u32 },

    // --- Chronicle ---
    ChronicleRecorded { text: String, significance: f32 },

    // --- Nemesis ---
    NemesisAppeared { name: String, faction: String },
    NemesisGrew { name: String, new_strength: f32 },
    NemesisDefeated { name: String, adventurer_id: u32 },

    // --- Favors ---
    FavorRequested { request_id: u32, faction_id: usize, description: String },
    FavorCompleted { faction_id: usize, request_id: u32, reward_favor: f32 },
    FavorCalledIn { faction_id: usize, favor_type: String, cost: f32, description: String },

    // --- Site Prep ---
    SiteCompleted { site_id: u32, region_id: usize, prep_type: String },
    SiteDestroyed { site_id: u32, region_id: usize },

    // --- Council ---
    CouncilVoteProposed { topic: String },
    CouncilVoteResolved { passed: bool, topic: String },

    // --- Disease ---
    DiseaseOutbreak { disease_id: u32, disease_name: String, region_id: usize, severity: f32 },
    DiseaseSpread { disease_id: u32, from_region: usize, to_region: usize },
    DiseaseContained { disease_id: u32, disease_name: String },
    AdventurerInfected { adventurer_id: u32, disease_id: u32 },

    // --- Prisoners ---
    PrisonerCaptured { prisoner_id: u32, prisoner_name: String, faction_id: usize },
    PrisonerEscaped { prisoner_id: u32, prisoner_name: String, faction_id: usize },
    AdventurerCaptured { adventurer_id: u32, adventurer_name: String, faction_id: usize },

    // --- Propaganda ---
    PropagandaEffect { campaign_id: u32, description: String },
    PropagandaExpired { campaign_id: u32, campaign_type: String },

    // --- Monster Ecology ---
    MonsterAttack { region: String, species: String, damage: f32 },
    MonsterSwarm { region: String, species: String },
    MonsterMigration { from: String, to: String, species: String },

    // --- Visions ---
    VisionReceived { adventurer_id: u32, vision_type: String, text: String },
    VisionFulfilled { adventurer_id: u32, text: String, vision_type: String },

    // --- Hobbies ---
    HobbyDeveloped { adventurer_id: u32, hobby: String },

    // --- Loans ---
    LoanRepaid { loan_id: u32, amount: f32, remaining: f32 },
    LoanDefaulted { loan_id: u32, lender_faction_id: usize, amount_owed: f32 },
    CreditRatingChanged { old: f32, new: f32 },

    // --- Victory Conditions ---
    VictoryProgress { condition: String, percent: f32 },
    VictoryAchieved { condition: String },

    // --- Dungeons ---
    DungeonExplored { dungeon_id: u32, dungeon_name: String, explored_pct: f32 },
    DungeonLootFound { dungeon_id: u32, dungeon_name: String, gold_found: f32 },
    DungeonThreatEmerged { dungeon_id: u32, dungeon_name: String, threat_level: f32 },
    DungeonConnectionDiscovered { from_dungeon_id: u32, from_name: String, to_dungeon_id: u32, to_name: String },
    DungeonRumorHeard { dungeon_id: u32, dungeon_name: String, region_name: String },

    // --- NPC Reputation ---
    NpcReputationChanged { npc_name: String, old: f32, new: f32 },
    NpcServiceUnlocked { npc_name: String, service: String },

    // --- Weather ---
    WeatherStarted { weather_type: super::state::WeatherType, regions: Vec<usize>, severity: f32 },
    WeatherEnded { weather_type: super::state::WeatherType },
    WeatherDamage { weather_type: super::state::WeatherType, description: String },

    // --- Artifacts ---
    ArtifactCreated { name: String, origin: String },
    ArtifactEquipped { name: String, adventurer_id: u32 },

    // --- Difficulty Scaling ---
    DifficultyEscalation { description: String },
    DifficultyRelief { description: String },
    PressureChanged { old: f32, new: f32 },

    // --- Culture ---
    CultureShift { region_id: usize, dominant_culture: String },
    CulturalMilestone { region_id: usize, culture: String, level: f32 },

    // --- Near-victory escalation ---
    NearVictoryEscalation,

    // --- Archives ---
    KnowledgeGained { amount: f32, source: String },
    ResearchCompleted { topic: String },

    // --- Auction ---
    AuctionStarted { items: Vec<String> },
    AuctionBidPlaced { item: String, amount: f32 },
    AuctionWon { item: String, winner: String, price: f32 },
    AuctionLost { item: String, winner: String },

    // --- Bounties ---
    BountyPosted { description: String, reward: f32 },
    BountyCompleted { description: String, reward_gold: f32 },
    BountyExpired { description: String },

    // --- Companions ---
    CompanionAcquired { adventurer_id: u32, species: super::state::CompanionSpecies, name: String },
    CompanionLost { name: String, reason: String },
    CompanionBondMilestone { adventurer_id: u32, name: String, level: i32 },

    // --- Contracts ---
    ContractOffered { contract_id: u32, commissioner: String, reward_gold: f32 },
    ContractCompleted { contract_id: u32, reward_gold: f32, reward_reputation: f32 },
    ContractFailed { contract_id: u32, penalty_gold: f32, penalty_reputation: f32 },

    // --- Corruption ---
    EmbezzlementDiscovered { gold_lost: f32 },
    CorruptionScandal { reputation_lost: f32 },

    // --- Counter-espionage ---
    EnemyAgentDetected { agent_id: u32, faction_id: usize, infiltration_level: f32 },
    EnemyAgentCaptured { agent_id: u32, faction_id: usize, intel_gained: f32 },
    EnemyAgentExpelled { agent_id: u32, faction_id: usize },
    GuildIntelLeaked { faction_id: usize, combat_bonus_pct: f32 },
    SabotagePrevented { agent_id: u32, description: String, prevented: bool },

    // --- Economic competition ---
    TradeWarDeclared { aggressor_faction_id: usize, target_faction_id: usize, aggressor_share: f32, description: String },
    TradeWarResolved { winner_faction_id: usize, loser_faction_id: usize, rivalry_type: String, description: String },
    EmbargoImposed { imposer_faction_id: usize, target_faction_id: usize, description: String },
    PriceWarStarted { faction_a: usize, faction_b: usize, description: String },
    MarketDominanceShift { faction_id: usize, market_share: f32, description: String },

    // --- Equipment durability ---
    EquipmentDegraded { adventurer_id: u32, item: String, durability: f32 },
    EquipmentBroken { adventurer_id: u32, item: String },
    EquipmentRepaired { adventurer_id: u32, item: String },

    // --- Evacuation ---
    EvacuationOrdered { source_region_id: usize, destination_region_id: usize, evacuees: u32, cost: f32 },
    EvacuationCompleted { source_region_id: usize, destination_region_id: usize, evacuees: u32, supplies_saved: f32 },
    EvacuationFailed { region_id: usize, reason: String, population_lost: u32 },
    CiviliansRescued { count: u32, region_id: usize },

    // --- Exploration ---
    TileExplored { percentage: f32 },
    ExplorationMilestone { percentage: u32 },
    LandmarkDiscovered { name: String, reward: String },

    // --- Faction tech ---
    FactionTechAdvanced { faction: usize, tech: String, level: f32 },
    FactionTechMilestone { faction: usize, tech: String, capability: String },

    // --- Fears ---
    FearDeveloped { adventurer_id: u32, fear_type: super::state::FearType, severity: f32 },
    FearTriggered { adventurer_id: u32, fear_type: super::state::FearType, severity: f32 },
    FearOvercome { adventurer_id: u32, fear_type: super::state::FearType, new_severity: f32, times_overcome: u32 },
    FearConquered { adventurer_id: u32, fear_type: super::state::FearType },

    // --- Food ---
    PartyStarving { party_id: u32 },

    // --- Infrastructure ---
    InfrastructureCompleted { infra_id: u32, infra_type: String, region_a: usize, region_b: usize },
    InfrastructureDamaged { infra_id: u32, infra_type: String, amount: f32, cause: String },

    // --- Insurance ---
    InsurancePurchased { policy_id: u32, policy_type: String, premium_per_tick: f32 },
    InsuranceLapsed { policy_id: u32, policy_type: String },
    InsuranceClaimed { policy_id: u32, payout: f32, reason: String },
    InsuranceCanceled { policy_id: u32 },

    // --- Intel reports ---
    IntelReportGenerated { report_type: super::state::ReportType, summary: String },

    // --- Intrigue ---
    IntrigueStarted { intrigue_id: u32, faction_id: usize, intrigue_type: String, description: String },
    IntrigueResolved { intrigue_id: u32, faction_id: usize, outcome: String },

    // --- Journals ---
    JournalEntryWritten { adventurer_id: u32, entry_type: super::state::JournalType, sentiment: f32 },

    // --- Last stand ---
    LastStandTriggered { adventurer_name: String, description: String },
    LastStandResolved { outcome: super::state::LastStandOutcome, description: String },

    // --- Marriages ---
    MarriageArranged { marriage_id: u32, adventurer_id: u32, faction_id: usize, noble_name: String, dowry: f32 },
    MarriageCrisis { marriage_id: u32, adventurer_id: u32, faction_id: usize, reason: String },
    HeirBorn { marriage_id: u32, adventurer_id: u32, heir_name: String },
    Divorced { marriage_id: u32, adventurer_id: u32, faction_id: usize, relation_penalty: f32 },

    // --- Memorials ---
    FuneralHeld { adventurer_id: u32, adventurer_name: String, memorial_type: String },
    MemorialCreated { memorial_id: u32, adventurer_name: String, memorial_type: String, description: String },
    MemorialMorale { adventurer_id: u32, morale_delta: f32, source: String },

    // --- Messengers ---
    MessengerSent { target: u32 },
    MessengerArrived { target: u32 },
    MessengerLost { target: u32 },

    // --- Moods ---
    MoodChanged { adventurer_id: u32, old_mood: super::state::Mood, new_mood: super::state::Mood, cause: super::state::MoodCause },

    // --- Personal goals ---
    PersonalGoalAssigned { adventurer_id: u32, goal: super::state::GoalType },
    PersonalGoalFulfilled { adventurer_id: u32, goal: super::state::GoalType },
    PersonalGoalAbandoned { adventurer_id: u32, goal: super::state::GoalType },

    // --- Quest chains ---
    QuestChainStarted { chain_id: u32, chain_name: String, theme: String, total_steps: u32 },
    QuestChainStepCompleted { chain_id: u32, chain_name: String, step: u32, total_steps: u32, description: String },
    QuestChainCompleted { chain_id: u32, chain_name: String, total_steps: u32, gold_reward: f32, artifact_name: String },
    QuestChainFailed { chain_id: u32, chain_name: String, step: u32, reason: String },

    // --- Religion ---
    TempleDevotion { temple_name: String, change: f32 },
    BlessingExpired { temple_name: String },

    // --- Reputation decay ---
    ReputationDecayed { amount: f32, reason: String },
    ReputationMaintained { cost: f32 },
    ReputationTrajectoryChanged { trend: super::state::ReputationTrend },

    // --- Reputation stories ---
    StoryCreated { text: String, impact: f32 },
    StorySpread { story_text: String, region_name: String },
    StoryFaded { text: String },

    // --- Skill challenges ---
    SkillChallengePresented { challenge_id: u32, skill_type: super::state::SkillType, difficulty: f32, quest_id: Option<u32>, adventurer_id: u32 },
    SkillChallengeSucceeded { challenge_id: u32, adventurer_id: u32, skill_type: super::state::SkillType },
    SkillChallengeFailed { challenge_id: u32, adventurer_id: u32, skill_type: super::state::SkillType },
    CriticalSuccess { challenge_id: u32, adventurer_id: u32, skill_type: super::state::SkillType },
    CriticalFailure { challenge_id: u32, adventurer_id: u32, skill_type: super::state::SkillType },

    // --- Supply lines ---
    SupplyLineInterdicted { supply_line_id: u32, faction_id: usize, source_region_id: usize, destination_region_id: usize },
    SupplyLineRestored { supply_line_id: u32, source_region_id: usize, destination_region_id: usize },
    EnemySupplyDisrupted { faction_id: usize, supply_line_id: u32 },

    // --- Terrain events ---
    TerrainEventStarted { event_id: u32, event_type: String, affected_regions: Vec<usize>, severity: f32, duration: u64 },
    TerrainEventEnded { event_id: u32, event_type: String, affected_regions: Vec<usize> },
    TerrainDamage { event_id: u32, description: String, region_id: usize },
    TerrainDiscovery { event_id: u32, description: String, region_id: usize },

    // --- Timed events ---
    TimedEventAppeared { event_id: u32, name: String, description: String, deadline_tick: u64 },
    TimedEventExpired { name: String },
    TimedEventResponded { event_id: u32, name: String, gold_gained: f32, reputation_gained: f32, speed_bonus: f32 },
    TimedEventTrap { event_id: u32, name: String, gold_lost: f32 },

    // --- Trade goods ---
    SupplyShortage { good_type: String, region_id: usize },
    TradeProfitMade { good_type: String, profit: f32, source_region: usize, dest_region: usize },

    // --- Traveling merchants ---
    MerchantArrived { merchant_id: u32, name: String, specialty: String, num_items: usize },
    MerchantDeparted { merchant_id: u32, name: String },
    RareMerchantSpotted { merchant_id: u32, name: String, specialty: String },
    MerchantPurchase { merchant_id: u32, item_name: String, price: f32 },

    // --- Treasure hunts ---
    TreasureMapFound { map_id: u32, name: String, num_steps: u32 },
    TreasureStepCompleted { map_id: u32, step_index: u32, clue: String, reward: f32 },
    TreasureHuntCompleted { map_id: u32, total_reward: f32, artifact_name: String },

    // --- Trophies ---
    TrophyEarned { name: String, bonus_description: String },
    TrophyBonusApplied { total_bonuses: String },

    // --- Wanted ---
    WantedPosterIssued { poster_id: u32, adventurer_id: u32, faction_id: usize, bounty_amount: f32, reason: String },
    BountyPaidOff { poster_id: u32, adventurer_id: u32, faction_id: usize, amount: f32 },
    BountyHunterDispatched { poster_id: u32, adventurer_id: u32, faction_id: usize, hunter_strength: f32 },
    BountyHunterDefeated { poster_id: u32, adventurer_id: u32, new_bounty: f32 },

    // --- Alliance blocs ---
    BlocFormed { bloc_id: u32, name: String, member_factions: Vec<usize>, leader_faction_id: usize },
    BlocCohesionChanged { bloc_id: u32, old_cohesion: f32, new_cohesion: f32, reason: String },
    BlocDissolved { bloc_id: u32, name: String, reason: String },
    BlocAttack { bloc_id: u32, target_faction: usize, description: String },
    GuildJoinedBloc { bloc_id: u32, bloc_name: String },

    // --- Bloodlines ---
    BloodlineEstablished { bloodline_id: u32, founder_name: String, archetype: String, stat_bonus: f32 },
    BloodlinePrestigeGrown { bloodline_id: u32, founder_name: String, new_prestige: f32 },
    DescendantAppeared { bloodline_id: u32, adventurer_id: u32, name: String, founder_name: String, generation: u32 },

    // --- Grudges ---
    GrudgeFormed { adventurer_id: u32, target: GrudgeTarget, intensity: f32, cause: String },
    GrudgeIntensified { adventurer_id: u32, target: GrudgeTarget, old_intensity: f32, new_intensity: f32 },
    GrudgeFaded { adventurer_id: u32, target: GrudgeTarget },
    VendettaFulfilled { adventurer_id: u32, target: GrudgeTarget },

    // --- Guild identity ---
    IdentityShift { old: Option<String>, new: Option<String> },
    IdentityBonusUnlocked { identity: String, bonus: String },
    IdentityRecruitAttracted { identity: String, recruit_type: String },

    // --- Guild rooms ---
    RoomBonusApplied { room_type: String, level: u32, boosted: bool },

    // --- Leadership ---
    LeaderAppointed { adventurer_id: u32, style: String },
    LeaderDied { adventurer_id: u32 },
    LeaderRetired { adventurer_id: u32, approval_rating: f32 },
    ApprovalChanged { adventurer_id: u32, approval_rating: f32 },
    SuccessionCrisis,

    // --- Nicknames ---
    NicknameEarned { adventurer_id: u32, title: String, source: String },

    // --- Oaths ---
    OathSworn { adventurer_id: u32, oath_id: u32, oath_type: String },
    OathBroken { adventurer_id: u32, oath_id: u32, oath_type: String },
    OathFulfilled { adventurer_id: u32, oath_id: u32, oath_type: String },

    // --- Romance ---
    RomanceBegan { adventurer_a: u32, adventurer_b: u32 },
    RomanceProgressed { adventurer_a: u32, adventurer_b: u32, new_stage: RomanceStage },
    RomanceStrained { adventurer_a: u32, adventurer_b: u32 },
    RomanceBrokenUp { adventurer_a: u32, adventurer_b: u32 },
    RomanticGesture { adventurer_a: u32, adventurer_b: u32 },

    // --- Seasonal quests ---
    SeasonalQuestAvailable { quest_id: u32, name: String, season: Season },
    SeasonalQuestCompleted { quest_id: u32, name: String, reward_gold: f32, reward_special: String },
    SeasonChampionEarned { season: Season, reward: String },

    // --- Smuggling ---
    SmugglingRouteDiscovered { route_id: u32, start_region: usize, end_region: usize },
    SmugglingRouteEstablished { route_id: u32, start_region: usize, end_region: usize },
    SmugglingProfit { gold: f32 },
    SmugglingBusted { route_id: u32, penalty: f32 },
    SmugglingRouteSuspended { route_id: u32 },

    // --- Vassalage ---
    VassalageEstablished { vassal_id: usize, lord_id: usize, vassal_name: String, lord_name: String },
    TributePaid { vassal_id: usize, lord_id: usize, amount: f32 },
    VassalRebellion { vassal_id: usize, lord_id: usize, vassal_name: String, lord_name: String },

    // --- Awakening ---
    AwakeningTriggered { adventurer_id: u32, awakening_type: AwakeningType, description: String },

    // --- Charter ---
    CharterAmended { description: String },
    CharterViolation { article: String },
    LegitimacyChanged { old: f32, new: f32 },

    // --- Folk hero ---
    FolkFameChanged { region: usize, amount: f32 },
    FolkTaleCreated { tale: String, region: usize, adventurer_id: u32, fame_impact: f32 },
    FolkHeroStatus { region: usize, adventurer: u32 },

    // --- Geography ---
    GeographyChanging { region_id: usize, change_type: GeoChangeType, progress: f32 },
    GeographyComplete { region_id: usize, change_type: GeoChangeType, effects: Vec<String> },

    // --- Legacy weapons ---
    LegacyWeaponCreated { weapon_id: u32, wielder_id: u32, name: String },
    LegacyWeaponInherited { weapon_id: u32, old_wielder_id: u32, new_wielder_id: Option<u32>, weapon_name: String },
    LegacyWeaponLevelUp { weapon_id: u32, new_level: u32, ability: String },
    LegacyWeaponNamed { weapon_id: u32, old_name: String, new_name: String },

    // --- Secrets ---
    SecretRevealed { adventurer_id: u32, secret_type: SecretType, description: String },
    SuspicionRising { adventurer_id: u32 },

    // --- Commodity futures ---
    FuturesContractSettled { contract_id: u32, profit: f32 },
    FuturesContractOffered { contract_id: u32, good_type: String, strike_price: f32 },

    // --- Coup Engine ---
    CoupAttempted { faction_id: usize, success: bool, new_leader_id: usize },
    CoupRiskRising { faction_id: usize, risk_level: f32 },

    // --- Divine Favor ---
    DivineFavorChanged { order_idx: usize, new_favor: f32, reason: String },
    MiracleGranted { order_idx: usize, miracle_type: String, beneficiary: String },
    DivineDispleasure { order_idx: usize, effect: String },

    // --- Wounds ---
    WoundSustained { adventurer_id: u32, wound_type: WoundType, severity: f32 },
    WoundHealed { adventurer_id: u32, wound_type: WoundType },

    // --- Plague vectors ---
    PlagueSpread { disease_name: String, from_region: u32, to_region: u32 },
    PlagueContained { disease_name: String },
    PlagueDeaths { region_id: u32, deaths: u32 },

    // --- Price controls ---
    PriceControlEnacted { good_type: String, ceiling: Option<f32>, floor: Option<f32> },
    PriceControlViolation { good_type: String, actual_price: f32, controlled_price: f32 },
    BlackMarketSurge { good_type: String, premium_pct: f32 },

    // --- Defection Cascade ---
    DefectionOccurred { adventurer_id: u32, from_faction: usize, to_faction: usize },
    DefectionCascade { trigger_id: u32, cascade_ids: Vec<u32>, faction_id: usize },

    // --- Heist planning ---
    HeistPhaseAdvanced { heist_id: u32, new_phase: HeistPhase },
    HeistSucceeded { heist_id: u32, loot_value: f32 },
    HeistFailed { heist_id: u32, captured_ids: Vec<u32> },

    // --- Contract Negotiation ---
    NegotiationStarted { quest_id: u32, original_reward: f32 },
    NegotiationCounterOffer { quest_id: u32, new_reward: f32, round: u32 },
    NegotiationAccepted { quest_id: u32, final_reward: f32 },
    NegotiationFailed { quest_id: u32, reason: String },

    // --- Escalation Protocol ---
    EscalationIncreased { faction_id: usize, new_level: u32 },
    EliteSquadDispatched { faction_id: usize, squad_power: f32 },
    EscalationDecrease { faction_id: usize, new_level: u32 },

    // --- Dead Zones ---
    DeadZoneExpanding { region_id: usize, level: f32 },
    DeadZoneSpreading { from_region: usize, to_region: usize },
    DeadZoneRecovering { region_id: usize, new_level: f32 },

    // --- Addiction ---
    AddictionDeveloped { adventurer_id: u32, dependency_level: f32 },
    WithdrawalOnset { adventurer_id: u32, severity: f32 },
    AddictionOvercome { adventurer_id: u32 },

    // --- Party chemistry ---
    ChemistryForged { adv_id_1: u32, adv_id_2: u32, score: f32 },
    LegendaryTeamFormed { party_id: u32, mean_chemistry: f32 },
    ChemistryBroken { adv_id_1: u32, adv_id_2: u32, reason: String },

    // --- Threat Clock ---
    /// The world threat clock advanced; optionally crossed a threshold.
    ThreatClockAdvanced { power: f32, threshold_crossed: Option<f32> },
    /// A threshold was crossed, producing a world effect.
    ThreatManifested { threat_type: super::state::WorldThreat, effect: String },
    /// The threat was disrupted by guild action.
    ThreatDisrupted { power_reduction: f32, new_power: f32 },
    /// The threat clock was activated for the first time.
    WorldThreatActivated { threat_type: super::state::WorldThreat },

    // --- Bankruptcy cascade ---
    FactionDefaulted { faction_id: u32, total_debt: f32 },
    CascadeTriggered { chain_length: u32, total_losses: f32 },
    CreditFreeze { duration_ticks: u32 },

    // --- Currency debasement ---
    CurrencyDebased { faction_id: u32, new_purity: f32 },
    InflationSpike { faction_id: u32, rate: f32 },
    DebasementDetected { faction_id: u32, by_whom: String },
    DebasementExposed { faction_id: u32, reputation_impact: f32 },

    // --- Signal towers ---
    TowerDestroyed { tower_id: u32, region_id: u32 },
    TowerCompromised { tower_id: u32, by_faction: usize },
    FalseSignalDetected { tower_id: u32 },
    SignalRelayed { from_tower: u32, to_tower: u32, signal_type: super::state::SignalType },

    // --- Demonic pacts ---
    DemonicPactOffered { adventurer_id: u32, power: f32 },
    DemonicPactAccepted { adventurer_id: u32, pact_id: u32 },
    DemonicCollectorArrived { adventurer_id: u32 },
    DemonicDebtEscalated { adventurer_id: u32, debt_level: f32, effect: String },

    // --- Class system ---
    ClassGranted { adventurer_id: u32, class_name: String },
    ClassLevelUp { adventurer_id: u32, class_name: String, new_level: u32 },
    SkillGrantedByClass { adventurer_id: u32, skill_name: String, rarity: String, class_name: String },
    ClassStagnated { adventurer_id: u32, class_name: String },
}

// ---------------------------------------------------------------------------
// Action result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActionResult {
    Success(String),
    /// Action prerequisites not met (e.g. "Not enough gold").
    Failed(String),
    /// Action type not valid in current state.
    InvalidAction(String),
}

// ---------------------------------------------------------------------------
// Step deltas (observability)
// ---------------------------------------------------------------------------

/// Granular per-tick change tracking for debugging and analysis.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepDeltas {
    // --- Economy ---
    pub gold_before: f32,
    pub gold_after: f32,
    pub supplies_before: f32,
    pub supplies_after: f32,
    pub reputation_before: f32,
    pub reputation_after: f32,

    // --- Adventurer changes ---
    pub adventurer_stat_changes: Vec<AdventurerStatDelta>,

    // --- Party movements ---
    /// (party_id, old_pos, new_pos)
    pub party_position_changes: Vec<(u32, (f32, f32), (f32, f32))>,

    // --- Quest lifecycle ---
    pub quests_arrived: u32,
    pub quests_expired: u32,
    pub quests_completed: u32,
    pub quests_failed: u32,

    // --- Battles ---
    pub battles_started: u32,
    pub battles_ended: u32,

    // --- Support ---
    pub runners_sent: u32,
    pub mercenaries_hired: u32,
    pub rescues_called: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdventurerStatDelta {
    pub adventurer_id: u32,
    pub stress_delta: f32,
    pub fatigue_delta: f32,
    pub injury_delta: f32,
    pub loyalty_delta: f32,
    pub morale_delta: f32,
    /// Which system caused this change.
    pub source: String,
}

// ---------------------------------------------------------------------------
// Scout report details
// ---------------------------------------------------------------------------

/// Detailed intelligence gathered when a region is scouted past the
/// visibility threshold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoutReportDetails {
    /// Name of the region.
    pub region_name: String,
    /// Military strength of the faction controlling this region.
    pub faction_military_strength: f32,
    /// Current unrest level (0–100).
    pub unrest: f32,
    /// Regional threat level (0–100).
    pub threat_level: f32,
    /// Number of active quest opportunities in the region.
    pub quest_opportunities: usize,
    /// Champion sightings (adventurer IDs of champions in this region).
    pub champion_sightings: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Action validation
// ---------------------------------------------------------------------------

impl CampaignState {
    /// Returns all currently valid actions the player can take.
    pub fn valid_actions(&self) -> Vec<CampaignAction> {
        // Pre-game phases
        match self.phase {
            CampaignPhase::CharacterCreation => {
                // During creation, only pending choices (backstory events) are valid
                let mut actions = Vec::new();
                for choice in &self.pending_choices {
                    for (i, _) in choice.options.iter().enumerate() {
                        actions.push(CampaignAction::RespondToChoice {
                            choice_id: choice.id,
                            option_index: i,
                        });
                    }
                }
                if actions.is_empty() {
                    // No pending creation choices — transition to package selection
                    // Return starting packages as the valid actions
                    return self.available_starting_choices
                        .iter()
                        .cloned()
                        .map(|choice| CampaignAction::ChooseStartingPackage { choice })
                        .collect();
                }
                return actions;
            }
            CampaignPhase::ChoosingStartingPackage => {
                return self.available_starting_choices
                    .iter()
                    .cloned()
                    .map(|choice| CampaignAction::ChooseStartingPackage { choice })
                    .collect();
            }
            CampaignPhase::Playing => {} // fall through to normal actions
        }

        let mut actions = vec![CampaignAction::Wait, CampaignAction::Rest];

        // Pending choices — must be responded to
        for choice in &self.pending_choices {
            for (i, _option) in choice.options.iter().enumerate() {
                actions.push(CampaignAction::RespondToChoice {
                    choice_id: choice.id,
                    option_index: i,
                });
            }
        }

        // Accept/decline quests on the board
        for req in &self.request_board {
            if self.active_quests.len() < self.guild.active_quest_capacity {
                actions.push(CampaignAction::AcceptQuest {
                    request_id: req.id,
                });
            }
            actions.push(CampaignAction::DeclineQuest {
                request_id: req.id,
            });
        }

        // Assign idle adventurers to preparing quests
        let idle_adventurers: Vec<u32> = self
            .adventurers
            .iter()
            .filter(|a| a.status == AdventurerStatus::Idle)
            .map(|a| a.id)
            .collect();

        for quest in &self.active_quests {
            if quest.status == ActiveQuestStatus::Preparing {
                for &adv_id in &idle_adventurers {
                    if !quest.assigned_pool.contains(&adv_id) {
                        actions.push(CampaignAction::AssignToPool {
                            adventurer_id: adv_id,
                            quest_id: quest.id,
                        });
                    }
                }
                // Unassign from pool
                for &adv_id in &quest.assigned_pool {
                    actions.push(CampaignAction::UnassignFromPool {
                        adventurer_id: adv_id,
                        quest_id: quest.id,
                    });
                }
                // Dispatch if pool is non-empty
                if !quest.assigned_pool.is_empty() {
                    actions.push(CampaignAction::DispatchQuest { quest_id: quest.id });
                }
            }
        }

        // Purchase supplies for in-field parties
        for party in &self.parties {
            if matches!(
                party.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
            ) && self.guild.gold >= self.config.economy.supply_cost_per_unit * 10.0
            {
                actions.push(CampaignAction::PurchaseSupplies {
                    party_id: party.id,
                    amount: 10.0,
                });
            }
        }

        // Train idle adventurers
        if self.guild.gold >= self.config.economy.training_cost {
            for &adv_id in &idle_adventurers {
                for &tt in &[
                    TrainingType::Combat,
                    TrainingType::Exploration,
                    TrainingType::Leadership,
                    TrainingType::Survival,
                ] {
                    actions.push(CampaignAction::TrainAdventurer {
                        adventurer_id: adv_id,
                        training_type: tt,
                    });
                }
            }
        }

        // Equip gear
        for &adv_id in &idle_adventurers {
            for item in &self.guild.inventory {
                actions.push(CampaignAction::EquipGear {
                    adventurer_id: adv_id,
                    item_id: item.id,
                });
            }
        }

        // Send runner to in-field parties
        if self.guild.gold >= self.config.economy.runner_cost {
            for party in &self.parties {
                if matches!(
                    party.status,
                    PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
                ) {
                    actions.push(CampaignAction::SendRunner {
                        party_id: party.id,
                        payload: RunnerPayload::Supplies(20.0),
                    });
                    actions.push(CampaignAction::SendRunner {
                        party_id: party.id,
                        payload: RunnerPayload::Message,
                    });
                }
            }
        }

        // Hire mercenary
        if self.guild.gold >= self.config.economy.mercenary_cost {
            for quest in &self.active_quests {
                if matches!(
                    quest.status,
                    ActiveQuestStatus::InProgress | ActiveQuestStatus::InCombat
                ) {
                    actions.push(CampaignAction::HireMercenary { quest_id: quest.id });
                }
            }
        }

        // Call rescue
        for battle in &self.active_battles {
            if battle.status == BattleStatus::Active && battle.party_health_ratio < 0.4 {
                let can_afford = self.guild.gold >= self.config.economy.rescue_bribe_cost;
                let has_free_rescue = self
                    .npc_relationships
                    .iter()
                    .any(|r| r.rescue_available && r.relationship_score > 50.0);
                if can_afford || has_free_rescue {
                    actions.push(CampaignAction::CallRescue {
                        battle_id: battle.id,
                    });
                }
            }
        }

        // Hire scout
        if self.guild.gold >= self.config.economy.scout_cost {
            for loc in &self.overworld.locations {
                if !loc.scouted {
                    actions.push(CampaignAction::HireScout {
                        location_id: loc.id,
                    });
                }
            }
        }

        // Diplomacy — context-sensitive actions per faction
        for faction in &self.factions {
            match faction.diplomatic_stance {
                DiplomaticStance::AtWar => {
                    // Can only ceasefire or threaten enemies at war
                    actions.push(CampaignAction::DiplomaticAction {
                        faction_id: faction.id,
                        action_type: DiplomacyActionType::ProposeCeasefire,
                    });
                }
                DiplomaticStance::Hostile => {
                    actions.push(CampaignAction::DiplomaticAction {
                        faction_id: faction.id,
                        action_type: DiplomacyActionType::ImproveRelations,
                    });
                    actions.push(CampaignAction::DiplomaticAction {
                        faction_id: faction.id,
                        action_type: DiplomacyActionType::Threaten,
                    });
                }
                DiplomaticStance::Neutral => {
                    actions.push(CampaignAction::DiplomaticAction {
                        faction_id: faction.id,
                        action_type: DiplomacyActionType::ImproveRelations,
                    });
                    actions.push(CampaignAction::DiplomaticAction {
                        faction_id: faction.id,
                        action_type: DiplomacyActionType::TradeAgreement,
                    });
                }
                DiplomaticStance::Friendly => {
                    actions.push(CampaignAction::DiplomaticAction {
                        faction_id: faction.id,
                        action_type: DiplomacyActionType::TradeAgreement,
                    });
                    // Can propose coalition if relation > 60
                    if faction.relationship_to_guild > 60.0 && !faction.coalition_member {
                        actions.push(CampaignAction::ProposeCoalition {
                            faction_id: faction.id,
                        });
                    }
                }
                DiplomaticStance::Coalition => {
                    // Can request military aid from coalition members
                    if faction.military_strength > 20.0 {
                        actions.push(CampaignAction::RequestCoalitionAid {
                            faction_id: faction.id,
                        });
                    }
                }
            }
        }

        // Use active abilities
        for unlock in &self.unlocks {
            if unlock.active && !unlock.properties.is_passive && unlock.cooldown_remaining_ms == 0 {
                if self.guild.gold >= unlock.properties.resource_cost {
                    // Generate valid targets based on target_type
                    match unlock.properties.target_type {
                        TargetType::GuildSelf => {
                            actions.push(CampaignAction::UseAbility {
                                unlock_id: unlock.id,
                                target: AbilityTarget::Party(0), // guild self
                            });
                        }
                        TargetType::Adventurer => {
                            for adv in &self.adventurers {
                                if adv.status != AdventurerStatus::Dead {
                                    actions.push(CampaignAction::UseAbility {
                                        unlock_id: unlock.id,
                                        target: AbilityTarget::Adventurer(adv.id),
                                    });
                                }
                            }
                        }
                        TargetType::Party => {
                            for party in &self.parties {
                                actions.push(CampaignAction::UseAbility {
                                    unlock_id: unlock.id,
                                    target: AbilityTarget::Party(party.id),
                                });
                            }
                        }
                        TargetType::Quest => {
                            for quest in &self.active_quests {
                                actions.push(CampaignAction::UseAbility {
                                    unlock_id: unlock.id,
                                    target: AbilityTarget::Quest(quest.id),
                                });
                            }
                        }
                        TargetType::Area => {
                            for loc in &self.overworld.locations {
                                actions.push(CampaignAction::UseAbility {
                                    unlock_id: unlock.id,
                                    target: AbilityTarget::Location(loc.id),
                                });
                            }
                        }
                        TargetType::Faction => {
                            for faction in &self.factions {
                                actions.push(CampaignAction::UseAbility {
                                    unlock_id: unlock.id,
                                    target: AbilityTarget::Faction(faction.id),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Intercept traveling champion parties (Sleeping King crisis)
        // Find champion parties (parties whose sole member has rallying_to set)
        let champion_party_ids: Vec<(u32, u32)> = self
            .parties
            .iter()
            .filter(|p| p.status == PartyStatus::Traveling && p.quest_id.is_none())
            .filter_map(|p| {
                // Check if the sole member is a rallying champion
                if p.member_ids.len() == 1 {
                    let mid = p.member_ids[0];
                    if self.adventurers.iter().any(|a| {
                        a.id == mid && a.rallying_to.is_some()
                    }) {
                        return Some((p.id, mid));
                    }
                }
                None
            })
            .collect();

        if !champion_party_ids.is_empty() {
            // Any idle guild adventurer can form a party to intercept
            if !idle_adventurers.is_empty() {
                for &(_, champion_id) in &champion_party_ids {
                    // Expose one action per champion — party formation handled at dispatch
                    actions.push(CampaignAction::InterceptChampion {
                        party_id: 0, // sentinel: will form a new party
                        champion_id,
                    });
                }
            }
            // Existing guild parties can also be redirected
            for party in &self.parties {
                if party.quest_id.is_some() {
                    continue; // don't redirect quest parties
                }
                if !champion_party_ids.iter().any(|&(pid, _)| pid == party.id) {
                    for &(_, champion_id) in &champion_party_ids {
                        actions.push(CampaignAction::InterceptChampion {
                            party_id: party.id,
                            champion_id,
                        });
                    }
                }
            }
        }

        // Spend priority (always valid)
        for &sp in &[
            SpendPriority::Balanced,
            SpendPriority::SaveForEmergencies,
            SpendPriority::InvestInGrowth,
            SpendPriority::MilitaryFocus,
        ] {
            actions.push(CampaignAction::SetSpendPriority { priority: sp });
        }

        actions
    }
}
