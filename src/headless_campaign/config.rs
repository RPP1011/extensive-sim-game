//! Campaign configuration — all tunable balance parameters.
//!
//! Every numeric constant that affects game balance is exposed here.
//! The `Default` impl preserves the current tuned values (83% win / 17% defeat).
//! Load from TOML for experiments, or mutate programmatically for MAP-Elites.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Difficulty
// ---------------------------------------------------------------------------

/// Campaign difficulty level. Controls enemy threat, starting resources,
/// faction aggression, and adventurer death risk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Difficulty {
    /// Target ~90%+ win rate. Weaker enemies, more resources.
    Easy,
    /// Target ~60-70% win rate. Current defaults.
    #[default]
    Normal,
    /// Target ~30-40% win rate. Stronger enemies, fewer resources.
    Hard,
    /// Target ~5-15% win rate. Punishing enemies, scarce resources.
    Brutal,
}

impl Difficulty {
    /// All difficulty variants in order.
    pub const ALL: [Difficulty; 4] = [
        Difficulty::Easy,
        Difficulty::Normal,
        Difficulty::Hard,
        Difficulty::Brutal,
    ];
}

impl std::fmt::Display for Difficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Difficulty::Easy => write!(f, "Easy"),
            Difficulty::Normal => write!(f, "Normal"),
            Difficulty::Hard => write!(f, "Hard"),
            Difficulty::Brutal => write!(f, "Brutal"),
        }
    }
}

/// Complete campaign balance configuration.
///
/// Grouped by system. Stored in `CampaignState.config` and passed to all
/// tick systems. Changing these values changes game difficulty, pacing,
/// and strategic balance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignConfig {
    pub quest_generation: QuestGenerationConfig,
    pub combat: CombatConfig,
    pub battle: BattleConfig,
    pub adventurer_condition: AdventurerConditionConfig,
    pub adventurer_recovery: AdventurerRecoveryConfig,
    pub supply: SupplyConfig,
    pub economy: EconomyConfig,
    pub faction_ai: FactionAiConfig,
    pub recruitment: RecruitmentConfig,
    pub progression: ProgressionConfig,
    pub campaign_progress: CampaignProgressConfig,
    pub npc_relationships: NpcRelationshipsConfig,
    pub threat: ThreatConfig,
    pub quest_lifecycle: QuestLifecycleConfig,
    pub starting_state: StartingStateConfig,
    #[serde(default)]
    pub npc_economy: NpcEconomyConfig,
}

// ---------------------------------------------------------------------------
// Quest Generation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestGenerationConfig {
    /// Mean ticks between quest arrivals at reputation 50.
    pub base_arrival_interval_ticks: u64,
    /// Reputation center point for scaling.
    pub reputation_scaling_center: f32,
    /// Min/max reputation factor.
    pub reputation_factor_min: f32,
    pub reputation_factor_max: f32,
    /// Base threat before scaling.
    pub base_threat: f32,
    /// Distance-to-threat conversion rate.
    pub distance_threat_rate: f32,
    /// Global threat contribution.
    pub global_threat_rate: f32,
    /// Campaign progress threat scaling multiplier.
    pub progress_threat_scaling: f32,
    /// Random threat variance range (±).
    pub threat_variance: f32,
    /// Threat clamp.
    pub min_threat: f32,
    pub max_threat: f32,
    /// Reward scaling.
    pub gold_per_threat: f32,
    pub gold_variance: f32,
    pub rep_per_threat: f32,
    pub max_rep_reward: f32,
    /// Supply reward for gather quests.
    pub gather_supply_reward: f32,
    /// Deadline in milliseconds.
    pub quest_deadline_ms: u64,
}

impl Default for QuestGenerationConfig {
    fn default() -> Self {
        Self {
            base_arrival_interval_ticks: 20, // ~1 minute between quest arrivals at rep 50
            reputation_scaling_center: 50.0,
            reputation_factor_min: 0.5,
            reputation_factor_max: 2.0,
            base_threat: 15.0,
            distance_threat_rate: 0.5,
            global_threat_rate: 0.5,
            progress_threat_scaling: 2.0,
            threat_variance: 15.0,
            min_threat: 5.0,
            max_threat: 100.0,
            gold_per_threat: 2.0,
            gold_variance: 20.0,
            rep_per_threat: 0.1,
            max_rep_reward: 5.0,
            gather_supply_reward: 15.0,
            quest_deadline_ms: 900_000, // 300 turns * 3s = 15 minutes
        }
    }
}

// ---------------------------------------------------------------------------
// Combat Oracle
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombatConfig {
    /// Level-to-power bonus multiplier.
    pub level_power_bonus: f32,
    /// HP divisor in power calculation.
    pub hp_power_divisor: f32,
    /// Injury/fatigue divisor in condition penalty.
    pub condition_penalty_divisor: f32,
    /// Morale factor range [base, base + morale/divisor].
    pub morale_factor_base: f32,
    pub morale_factor_divisor: f32,
    /// Threat-to-enemy-power divisor.
    pub threat_enemy_divisor: f32,
    pub enemy_power_min_ratio: f32,
    /// Win probability sigmoid steepness.
    pub sigmoid_steepness: f32,
    /// Victory HP remaining range.
    pub victory_hp_floor: f32,
    pub victory_hp_scaling: f32,
    pub victory_hp_ceiling: f32,
    /// Defeat HP remaining range.
    pub defeat_hp_scaling: f32,
    pub defeat_hp_ceiling: f32,
    /// Battle duration base ticks.
    pub base_duration_ticks: u64,
    pub duration_difficulty_min: f32,
    pub duration_difficulty_max: f32,
    /// Casualty rate from win probability.
    pub casualty_rate_multiplier: f32,
}

impl Default for CombatConfig {
    fn default() -> Self {
        Self {
            level_power_bonus: 5.0,
            hp_power_divisor: 100.0,
            condition_penalty_divisor: 200.0,
            morale_factor_base: 0.8,
            morale_factor_divisor: 500.0,
            threat_enemy_divisor: 50.0,
            enemy_power_min_ratio: 0.5,
            sigmoid_steepness: 2.5,
            victory_hp_floor: 0.3,
            victory_hp_scaling: 0.5,
            victory_hp_ceiling: 0.95,
            defeat_hp_scaling: 0.1,
            defeat_hp_ceiling: 0.3,
            base_duration_ticks: 30,
            duration_difficulty_min: 0.5,
            duration_difficulty_max: 3.0,
            casualty_rate_multiplier: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Battle System
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BattleConfig {
    /// Default battle duration in campaign ticks.
    pub default_duration_ticks: u64,
    /// Party HP damage multiplier in victory.
    pub victory_party_damage: f32,
    /// Enemy HP scaling in defeat.
    pub defeat_enemy_damage: f32,
    /// Battle update event emission interval (ticks).
    pub update_interval_ticks: u64,
}

impl Default for BattleConfig {
    fn default() -> Self {
        Self {
            default_duration_ticks: 50,
            victory_party_damage: 0.3,
            defeat_enemy_damage: 0.3,
            update_interval_ticks: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Adventurer Condition
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdventurerConditionConfig {
    /// Tick interval for condition drift.
    pub drift_interval_ticks: u64,
    /// Drift rates per status: (stress, fatigue, morale, loyalty).
    pub fighting_drift: [f32; 4],
    pub on_mission_drift: [f32; 4],
    pub idle_drift: [f32; 4],
    pub injured_drift: [f32; 4],
    /// Desertion thresholds.
    pub desertion_loyalty_threshold: f32,
    pub desertion_stress_threshold: f32,
}

impl Default for AdventurerConditionConfig {
    fn default() -> Self {
        Self {
            drift_interval_ticks: 10,
            fighting_drift: [0.8, 0.6, -0.5, -0.1],
            on_mission_drift: [0.3, 0.4, -0.2, 0.0],
            idle_drift: [-0.5, -0.3, 0.3, 0.05],
            injured_drift: [-0.2, -0.5, -0.1, -0.05],
            desertion_loyalty_threshold: 10.0,
            desertion_stress_threshold: 70.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Adventurer Recovery
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdventurerRecoveryConfig {
    /// Tick interval for recovery checks.
    pub recovery_interval_ticks: u64,
    /// Recovery rates per tick.
    pub stress_recovery: f32,
    pub fatigue_recovery: f32,
    pub injury_recovery: f32,
    pub loyalty_recovery: f32,
    /// Reactivation thresholds.
    pub injury_threshold: f32,
    pub fatigue_threshold: f32,
}

impl Default for AdventurerRecoveryConfig {
    fn default() -> Self {
        Self {
            recovery_interval_ticks: 100,
            stress_recovery: 3.0,
            fatigue_recovery: 4.0,
            injury_recovery: 3.5,
            loyalty_recovery: 0.7,
            injury_threshold: 40.0,
            fatigue_threshold: 40.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Supply
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupplyConfig {
    /// Supply drain per member per second.
    pub drain_per_member_per_sec: f32,
    /// Low supply warning threshold.
    pub low_threshold: f32,
}

impl Default for SupplyConfig {
    fn default() -> Self {
        Self {
            drain_per_member_per_sec: 0.02,
            low_threshold: 20.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Economy
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EconomyConfig {
    /// Passive gold income per second.
    pub passive_gold_per_sec: f32,
    /// Action costs.
    pub runner_cost: f32,
    pub mercenary_cost: f32,
    pub scout_cost: f32,
    pub training_cost: f32,
    pub rescue_bribe_cost: f32,
    pub supply_cost_per_unit: f32,

    // --- Trade & Market ---
    /// Gold per second per point of region control for guild-owned settlements.
    #[serde(default = "default_trade_rate")]
    pub trade_income_per_control: f32,
    /// How fast market prices inflate per purchase (added to rolling history).
    #[serde(default = "default_inflation")]
    pub market_inflation_rate: f32,
    /// How fast purchase history decays per tick (multiplicative).
    #[serde(default = "default_market_decay")]
    pub market_decay_rate: f32,
    /// Maximum price multiplier from market inflation.
    #[serde(default = "default_market_max")]
    pub market_max_multiplier: f32,

    // --- Investment ---
    /// Gold drained per second into current spend priority category.
    #[serde(default = "default_invest_gold")]
    pub investment_gold_per_sec: f32,
    /// Investment level gain per gold invested (before diminishing returns).
    #[serde(default = "default_invest_return")]
    pub investment_return_rate: f32,
    /// Maximum investment level per category (diminishing returns cap).
    #[serde(default = "default_invest_max")]
    pub investment_max_level: f32,

    // --- Supply chain ---
    /// Fatigue penalty per tile of distance from base when out of supply.
    #[serde(default = "default_supply_dist")]
    pub supply_distance_penalty: f32,
    /// Fatigue added per tick to out-of-supply parties.
    #[serde(default = "default_oos_fatigue")]
    pub out_of_supply_fatigue: f32,
    /// Morale drained per tick from out-of-supply parties.
    #[serde(default = "default_oos_morale")]
    pub out_of_supply_morale: f32,

    // --- Threat reward ---
    /// Bonus gold per second per point of global threat (risk = reward).
    #[serde(default = "default_threat_bonus")]
    pub threat_reward_bonus: f32,
}

fn default_trade_rate() -> f32 { 0.02 }
fn default_inflation() -> f32 { 0.1 }
fn default_market_decay() -> f32 { 0.05 }
fn default_market_max() -> f32 { 3.0 }
fn default_invest_gold() -> f32 { 0.3 }
fn default_invest_return() -> f32 { 0.01 }
fn default_invest_max() -> f32 { 10.0 }
fn default_supply_dist() -> f32 { 0.001 }
fn default_oos_fatigue() -> f32 { 0.5 }
fn default_oos_morale() -> f32 { 0.3 }
fn default_threat_bonus() -> f32 { 0.005 }

impl Default for EconomyConfig {
    fn default() -> Self {
        Self {
            passive_gold_per_sec: 0.5,
            runner_cost: 15.0,
            mercenary_cost: 50.0,
            scout_cost: 10.0,
            training_cost: 20.0,
            rescue_bribe_cost: 80.0,
            supply_cost_per_unit: 0.5,
            trade_income_per_control: default_trade_rate(),
            market_inflation_rate: default_inflation(),
            market_decay_rate: default_market_decay(),
            market_max_multiplier: default_market_max(),
            investment_gold_per_sec: default_invest_gold(),
            investment_return_rate: default_invest_return(),
            investment_max_level: default_invest_max(),
            supply_distance_penalty: default_supply_dist(),
            out_of_supply_fatigue: default_oos_fatigue(),
            out_of_supply_morale: default_oos_morale(),
            threat_reward_bonus: default_threat_bonus(),
        }
    }
}

// ---------------------------------------------------------------------------
// Faction AI
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionAiConfig {
    /// Tick interval for faction AI decisions.
    pub decision_interval_ticks: u64,
    /// Attack power as fraction of military strength.
    pub attack_power_fraction: f32,
    /// Control threshold for territory capture.
    pub territory_capture_control: f32,
    /// Military strength buildup per AI tick (hostile).
    pub hostile_strength_gain: f32,
    /// Military strength threshold for war declaration.
    pub war_declaration_threshold: f32,
    /// Relationship penalty for declaring war.
    pub war_declaration_penalty: f32,
    /// Control gain per tick (neutral defense).
    pub neutral_control_gain: f32,
    /// Relationship gain per tick (friendly).
    pub friendly_relationship_gain: f32,
    /// Max recent actions to track.
    pub max_recent_actions: usize,
}

impl Default for FactionAiConfig {
    fn default() -> Self {
        Self {
            decision_interval_ticks: 600,
            attack_power_fraction: 0.1,
            territory_capture_control: 30.0,
            hostile_strength_gain: 3.0,
            war_declaration_threshold: 80.0,
            war_declaration_penalty: 30.0,
            neutral_control_gain: 1.0,
            friendly_relationship_gain: 2.0,
            max_recent_actions: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Recruitment
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecruitmentConfig {
    /// Tick interval for recruitment checks.
    pub interval_ticks: u64,
    /// Gold cost per recruit.
    pub recruit_cost: f32,
    /// Maximum adventurers.
    pub max_adventurers: usize,
    /// Recruitment chance range.
    pub min_recruit_chance: f32,
    pub max_recruit_chance: f32,
    /// Level range for recruits.
    pub min_level: u32,
    pub max_level: u32,
    /// Per-level stat scaling.
    pub hp_per_level: f32,
    pub attack_per_level: f32,
    pub defense_per_level: f32,
    pub ability_power_per_level: f32,
}

impl Default for RecruitmentConfig {
    fn default() -> Self {
        Self {
            interval_ticks: 3000,
            recruit_cost: 40.0,
            max_adventurers: 12,
            min_recruit_chance: 0.2,
            max_recruit_chance: 0.8,
            min_level: 1,
            max_level: 3,
            hp_per_level: 5.0,
            attack_per_level: 2.0,
            defense_per_level: 1.5,
            ability_power_per_level: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Progression
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgressionConfig {
    /// Minimum quests completed before first unlock.
    pub unlock_threshold: usize,
    /// Quests between unlock checks.
    pub unlock_interval: usize,
    /// Recent quest window for playstyle detection.
    pub recent_quest_window: usize,
    /// Quests of a type needed to trigger themed unlock.
    pub themed_unlock_count: usize,
    /// Active ability default cooldown (ms).
    pub active_ability_cooldown_ms: u64,
    /// Default unlock effect magnitude.
    pub default_magnitude: f32,
}

impl Default for ProgressionConfig {
    fn default() -> Self {
        Self {
            unlock_threshold: 3,
            unlock_interval: 5,
            recent_quest_window: 5,
            themed_unlock_count: 3,
            active_ability_cooldown_ms: 30_000,
            default_magnitude: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Campaign Progress / Victory
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignProgressConfig {
    /// Number of quest victories needed for campaign win.
    pub victory_quest_count: f32,
    /// Weight of quest progress toward campaign completion.
    pub quest_weight: f32,
    /// Weight of reputation toward campaign completion.
    pub reputation_weight: f32,
    /// Reputation cap for progress.
    pub reputation_cap: f32,
    /// Weight of territory toward campaign completion.
    pub territory_weight: f32,
    /// Every N victories, flip a hostile region.
    pub territory_flip_interval: u32,
    /// Control value after capturing territory.
    pub capture_control: f32,
    /// Unrest reduction on capture.
    pub capture_unrest_reduction: f32,
    /// Campaign progress threshold for calamity warning.
    pub calamity_warning_threshold: f32,
    /// Unrest threshold for crisis flood calamity.
    pub crisis_flood_unrest_threshold: f32,
}

impl Default for CampaignProgressConfig {
    fn default() -> Self {
        Self {
            victory_quest_count: 25.0,
            quest_weight: 0.6,
            reputation_weight: 0.25,
            reputation_cap: 70.0,
            territory_weight: 0.15,
            territory_flip_interval: 5,
            capture_control: 60.0,
            capture_unrest_reduction: 10.0,
            calamity_warning_threshold: 0.7,
            crisis_flood_unrest_threshold: 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// NPC Relationships
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpcRelationshipsConfig {
    /// Tick interval for relationship drift.
    pub drift_interval_ticks: u64,
    /// Drift rate toward neutral per tick.
    pub drift_rate: f32,
    /// Relationship threshold for free rescue.
    pub rescue_threshold: f32,
}

impl Default for NpcRelationshipsConfig {
    fn default() -> Self {
        Self {
            drift_interval_ticks: 300,
            drift_rate: 0.5,
            rescue_threshold: 50.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Threat
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThreatConfig {
    /// Tick interval for threat updates.
    pub update_interval_ticks: u64,
    /// Weight of unrest in global threat.
    pub unrest_weight: f32,
    /// Hostile faction threat multiplier.
    pub hostile_faction_multiplier: f32,
    /// Campaign progress contribution to threat.
    pub progress_threat_weight: f32,
}

impl Default for ThreatConfig {
    fn default() -> Self {
        Self {
            update_interval_ticks: 600,
            unrest_weight: 0.5,
            hostile_faction_multiplier: 10.0,
            progress_threat_weight: 20.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Quest Lifecycle / Consequences
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestLifecycleConfig {
    /// Time at location before battle triggers (ms).
    pub battle_trigger_delay_ms: u64,
    /// Non-combat quest duration multiplier (threat → ms).
    pub non_combat_duration_multiplier: f32,
    /// Victory consequences.
    pub victory_injury_scaling: f32,
    pub victory_stress_relief: f32,
    pub victory_loyalty_gain: f32,
    pub victory_morale_gain: f32,
    pub victory_base_xp: u32,
    pub victory_threat_xp_rate: f32,
    /// Level-up formula: threshold = level * level * this.
    pub level_up_xp_multiplier: u32,
    /// Per-level stat gains.
    pub level_hp_gain: f32,
    pub level_attack_gain: f32,
    pub level_defense_gain: f32,
    /// Defeat consequences.
    pub defeat_severity_divisor: f32,
    pub defeat_base_injury: f32,
    pub defeat_severity_injury: f32,
    pub defeat_stress_gain: f32,
    pub defeat_loyalty_loss: f32,
    pub defeat_morale_loss: f32,
    pub defeat_base_xp: u32,
    /// Death chance when severely injured (>90 injury).
    pub death_chance: f32,
    /// Injury threshold for incapacitation.
    pub incapacitation_threshold: f32,
    /// Abandoned quest penalties.
    pub abandon_stress: f32,
    pub abandon_morale_loss: f32,
    /// Party base travel speed (tiles/sec).
    pub party_speed: f32,
    /// Party starting supply level.
    pub party_starting_supply: f32,
    /// Party starting morale.
    pub party_starting_morale: f32,
}

impl Default for QuestLifecycleConfig {
    fn default() -> Self {
        Self {
            battle_trigger_delay_ms: 2000,
            non_combat_duration_multiplier: 500.0,
            victory_injury_scaling: 30.0,
            victory_stress_relief: 5.0,
            victory_loyalty_gain: 3.0,
            victory_morale_gain: 5.0,
            victory_base_xp: 20,
            victory_threat_xp_rate: 0.5,
            level_up_xp_multiplier: 50,
            level_hp_gain: 5.0,
            level_attack_gain: 2.0,
            level_defense_gain: 1.5,
            defeat_severity_divisor: 50.0,
            defeat_base_injury: 25.0,
            defeat_severity_injury: 15.0,
            defeat_stress_gain: 15.0,
            defeat_loyalty_loss: 5.0,
            defeat_morale_loss: 15.0,
            defeat_base_xp: 5,
            death_chance: 0.3,
            incapacitation_threshold: 65.0,
            abandon_stress: 8.0,
            abandon_morale_loss: 10.0,
            party_speed: 5.0,
            party_starting_supply: 100.0,
            party_starting_morale: 80.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Starting State
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StartingStateConfig {
    pub gold: f32,
    pub supplies: f32,
    pub reputation: f32,
    pub base_defensive_strength: f32,
    pub active_quest_capacity: usize,
    pub global_threat_level: f32,
    /// Bankrupt defeat threshold.
    pub bankrupt_gold_threshold: f32,
    /// Minimum ticks before defeat conditions can trigger.
    pub early_game_protection_ticks: u64,
}

impl Default for StartingStateConfig {
    fn default() -> Self {
        Self {
            gold: 100.0,
            supplies: 80.0,
            reputation: 25.0,
            base_defensive_strength: 10.0,
            active_quest_capacity: 3,
            global_threat_level: 20.0,
            bankrupt_gold_threshold: 10.0,
            early_game_protection_ticks: 5000, // ~4 hours of game time before territory loss = defeat
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level Default
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// NPC Economic Agent
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpcEconomyConfig {
    /// How often NPC decisions, demand, and safety are re-evaluated (ticks).
    pub decision_interval_ticks: u64,

    // --- Income ---
    /// Base gold per tick per point of stat × level_multiplier when demand > 0.
    pub service_income_per_stat_point: f32,
    /// Demand exponent: income = stat × level_mult × demand^this × base_rate.
    pub demand_income_exponent: f32,
    /// Base combat income rate (per power_rating point × threat_premium).
    pub base_combat_income_rate: f32,
    /// Fraction of quest gold that goes to participating NPCs (rest to guild).
    pub quest_reward_npc_share: f32,
    /// Guild tax rate on NPC service income.
    pub guild_tax_rate: f32,
    /// Income variance range (±this fraction). 0.2 = ±20%.
    pub income_variance: f32,

    // --- Demand ---
    /// Per-channel demand saturation rate (supply contribution divisor).
    pub demand_saturation_rate: f32,
    /// Base combat demand in any settlement (guards always needed).
    pub base_combat_demand: f32,
    /// Base medicine demand (healers always needed).
    pub base_medicine_demand: f32,
    /// Regional threat contribution to combat demand.
    pub threat_to_combat_demand: f32,
    /// Population contribution to base demand across all channels.
    pub population_to_base_demand: f32,

    // --- Expenses ---
    /// Base living cost per NPC per tick at a settlement.
    pub base_living_cost_per_tick: f32,
    /// Equipment upkeep per tick (scaled by effective_level / 10).
    pub equipment_upkeep_per_tick: f32,
    /// Medical cost multiplier when injured.
    pub injury_cost_multiplier: f32,
    /// Cost of living population scale divisor.
    pub cost_pop_scale: f32,

    // --- Power ---
    /// Resource value (gold + equipment) per effective level bonus. wealth_bonus = sqrt(resources / this).
    pub resource_level_threshold: f32,

    // --- Safety ---
    /// Threat scale divisor for min_viable_threat calculation.
    pub threat_scale: f32,
    /// Viability floor fraction: party_max_level must exceed route_threat × this.
    pub viability_floor_fraction: f32,

    // --- Travel danger ---
    /// Base injury chance per tick while traveling (ambient danger).
    pub base_travel_injury_chance: f32,
    /// Attrition injury chance when between viability floor and route threat.
    pub attrition_injury_rate: f32,
    /// Injury chance when below viability floor (capped).
    pub below_floor_injury_chance: f32,
    /// Base injury severity on a travel encounter hit.
    pub travel_injury_base: f32,
    /// Death chance per tick when injury > 90 while traveling.
    pub travel_death_chance: f32,
    /// Road safety factor: route_threat *= (1.0 - road_connectivity × this).
    pub road_safety_factor: f32,

    // --- Party formation ---
    /// Ticks to wait before abandoning party search.
    pub party_seek_patience_ticks: u32,
    /// Maximum autonomous party size.
    pub max_autonomous_party_size: usize,

    // --- Counterleveling ---
    /// Maximum adversity multiplier (from ln curve, ~2.0 in practice).
    pub max_adversity_multiplier: f32,

    // --- Death spiral floor ---
    /// Minimum combat NPCs per region before immigration pressure activates.
    pub min_combat_npcs_per_region: usize,
}

impl Default for NpcEconomyConfig {
    fn default() -> Self {
        Self {
            decision_interval_ticks: 50,

            service_income_per_stat_point: 0.1,
            demand_income_exponent: 1.5,
            base_combat_income_rate: 0.001,
            quest_reward_npc_share: 0.6,
            guild_tax_rate: 0.2,
            income_variance: 0.2,

            demand_saturation_rate: 0.02,
            base_combat_demand: 0.3,
            base_medicine_demand: 0.2,
            threat_to_combat_demand: 0.01,
            population_to_base_demand: 0.001,

            base_living_cost_per_tick: 0.05,
            equipment_upkeep_per_tick: 0.02,
            injury_cost_multiplier: 3.0,
            cost_pop_scale: 50.0,

            resource_level_threshold: 100.0,

            threat_scale: 10.0,
            viability_floor_fraction: 0.6,

            base_travel_injury_chance: 0.001,
            attrition_injury_rate: 0.05,
            below_floor_injury_chance: 0.3,
            travel_injury_base: 10.0,
            travel_death_chance: 0.05,
            road_safety_factor: 0.7,

            party_seek_patience_ticks: 200,
            max_autonomous_party_size: 6,

            max_adversity_multiplier: 2.0,
            min_combat_npcs_per_region: 2,
        }
    }
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            quest_generation: QuestGenerationConfig::default(),
            combat: CombatConfig::default(),
            battle: BattleConfig::default(),
            adventurer_condition: AdventurerConditionConfig::default(),
            adventurer_recovery: AdventurerRecoveryConfig::default(),
            supply: SupplyConfig::default(),
            economy: EconomyConfig::default(),
            faction_ai: FactionAiConfig::default(),
            recruitment: RecruitmentConfig::default(),
            progression: ProgressionConfig::default(),
            campaign_progress: CampaignProgressConfig::default(),
            npc_relationships: NpcRelationshipsConfig::default(),
            threat: ThreatConfig::default(),
            quest_lifecycle: QuestLifecycleConfig::default(),
            starting_state: StartingStateConfig::default(),
            npc_economy: NpcEconomyConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// TOML loading
// ---------------------------------------------------------------------------

impl CampaignConfig {
    /// Create a config tuned for the given difficulty level.
    ///
    /// | Difficulty | Target Win Rate | Key Changes |
    /// |---|---|---|
    /// | Easy | 90%+ | lower threat, more gold/supplies, slower faction growth |
    /// | Normal | 60-70% | current defaults |
    /// | Hard | 30-40% | higher threat, less gold, faster faction growth, more deaths |
    /// | Brutal | 5-15% | extreme threat, minimal resources, small roster, fast crises |
    pub fn with_difficulty(difficulty: Difficulty) -> Self {
        let mut cfg = Self::default();
        match difficulty {
            Difficulty::Easy => {
                cfg.quest_generation.base_threat *= 0.6;
                cfg.quest_generation.quest_deadline_ms =
                    (cfg.quest_generation.quest_deadline_ms as f64 * 1.5) as u64;
                cfg.starting_state.gold *= 1.5;
                cfg.starting_state.supplies *= 1.5;
                cfg.faction_ai.hostile_strength_gain *= 0.5;
                cfg.quest_lifecycle.death_chance *= 0.5;
                cfg.quest_lifecycle.defeat_base_injury *= 0.7;
                cfg.combat.sigmoid_steepness = 3.0; // more forgiving win curve
            }
            Difficulty::Normal => {
                // defaults are Normal
            }
            Difficulty::Hard => {
                cfg.quest_generation.base_threat *= 1.5;
                cfg.quest_generation.progress_threat_scaling *= 1.3;
                cfg.starting_state.gold *= 0.6;
                cfg.starting_state.supplies *= 0.7;
                cfg.faction_ai.hostile_strength_gain *= 1.5;
                cfg.quest_lifecycle.death_chance *= 1.5;
                cfg.quest_lifecycle.defeat_base_injury *= 1.3;
                cfg.combat.sigmoid_steepness = 2.0; // steeper = more punishing
                cfg.campaign_progress.crisis_flood_unrest_threshold *= 0.8;
            }
            Difficulty::Brutal => {
                cfg.quest_generation.base_threat *= 2.0;
                cfg.quest_generation.progress_threat_scaling *= 1.8;
                cfg.starting_state.gold *= 0.4;
                cfg.starting_state.supplies *= 0.5;
                cfg.faction_ai.hostile_strength_gain *= 2.0;
                cfg.quest_lifecycle.death_chance *= 2.0;
                cfg.quest_lifecycle.defeat_base_injury *= 1.5;
                cfg.combat.sigmoid_steepness = 1.5; // very punishing
                cfg.recruitment.max_adventurers = 8;
                cfg.recruitment.recruit_cost *= 1.5;
                cfg.recruitment.min_recruit_chance *= 0.5;
                cfg.campaign_progress.crisis_flood_unrest_threshold *= 0.6;
                cfg.starting_state.early_game_protection_ticks /= 2;
            }
        }
        cfg
    }

    /// Load from a TOML file. Missing fields use defaults.
    pub fn load_from_toml(path: &std::path::Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Save to a TOML file.
    pub fn save_to_toml(&self, path: &std::path::Path) -> Result<(), String> {
        let content =
            toml::to_string_pretty(self).map_err(|e| format!("Failed to serialize: {}", e))?;
        std::fs::write(path, content).map_err(|e| format!("Failed to write {}: {}", path.display(), e))
    }
}
