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

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpendPriority {
    Balanced,
    SaveForEmergencies,
    InvestInGrowth,
    MilitaryFocus,
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

    // --- Campaign ---
    CalamityWarning {
        description: String,
    },
    CampaignMilestone {
        description: String,
    },
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
