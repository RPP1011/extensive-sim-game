//! Self-documenting action metadata for the RL agent.
//!
//! Provides a static registry describing every action type's properties,
//! prerequisite context, predicted outcomes, synergy scoring, and action
//! space summary tokens. This lets the RL agent learn action semantics
//! from metadata rather than trial-and-error.

use serde::{Deserialize, Serialize};

use super::actions::*;
use super::heuristic_bc::action_type_name;
use super::state::*;
use super::tokens::EntityToken;

// ---------------------------------------------------------------------------
// 1. Action Metadata Registry
// ---------------------------------------------------------------------------

/// Static metadata describing an action type's properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionMeta {
    /// Short type name (matches `action_type_name()`).
    pub type_name: &'static str,
    /// Strategic category.
    pub category: &'static str,
    /// Typical gold cost (0.0 if free).
    pub typical_gold_cost: f32,
    /// Typical duration in ticks before effect is felt (0 = immediate).
    pub typical_duration_ticks: u32,
    /// Whether the action can be reversed or undone.
    pub reversible: bool,
    /// Whether the action requires an active party.
    pub requires_party: bool,
    /// Whether the action affects faction relations.
    pub affects_faction: bool,
    /// Risk level (0.0 = safe, 1.0 = very risky).
    pub risk_level: f32,
}

/// Returns metadata for a given action type name.
pub fn action_metadata(action_type: &str) -> &'static ActionMeta {
    // Strip diplomacy/spend-priority subtypes to base type
    let base_type = if action_type.starts_with("DiplomaticAction_") {
        "DiplomaticAction"
    } else if action_type.starts_with("SetSpendPriority_") {
        "SetSpendPriority"
    } else {
        action_type
    };

    match base_type {
        "Wait" => &ActionMeta {
            type_name: "Wait",
            category: "logistics",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 1,
            reversible: true,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "Rest" => &ActionMeta {
            type_name: "Rest",
            category: "logistics",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 1,
            reversible: true,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "AcceptQuest" => &ActionMeta {
            type_name: "AcceptQuest",
            category: "combat",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.2,
        },
        "DeclineQuest" => &ActionMeta {
            type_name: "DeclineQuest",
            category: "combat",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "AssignToPool" => &ActionMeta {
            type_name: "AssignToPool",
            category: "logistics",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: true,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "UnassignFromPool" => &ActionMeta {
            type_name: "UnassignFromPool",
            category: "logistics",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: true,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "DispatchQuest" => &ActionMeta {
            type_name: "DispatchQuest",
            category: "combat",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 200,
            reversible: false,
            requires_party: true,
            affects_faction: false,
            risk_level: 0.5,
        },
        "PurchaseSupplies" => &ActionMeta {
            type_name: "PurchaseSupplies",
            category: "economy",
            typical_gold_cost: SUPPLY_COST_PER_UNIT * 10.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: true,
            affects_faction: false,
            risk_level: 0.0,
        },
        "TrainAdventurer" => &ActionMeta {
            type_name: "TrainAdventurer",
            category: "social",
            typical_gold_cost: TRAINING_COST,
            typical_duration_ticks: 50,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "EquipGear" => &ActionMeta {
            type_name: "EquipGear",
            category: "logistics",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: true,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "SendRunner" => &ActionMeta {
            type_name: "SendRunner",
            category: "logistics",
            typical_gold_cost: RUNNER_COST,
            typical_duration_ticks: 30,
            reversible: false,
            requires_party: true,
            affects_faction: false,
            risk_level: 0.1,
        },
        "HireMercenary" => &ActionMeta {
            type_name: "HireMercenary",
            category: "combat",
            typical_gold_cost: MERCENARY_COST,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.2,
        },
        "CallRescue" => &ActionMeta {
            type_name: "CallRescue",
            category: "combat",
            typical_gold_cost: RESCUE_BRIBE_COST,
            typical_duration_ticks: 10,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.1,
        },
        "HireScout" => &ActionMeta {
            type_name: "HireScout",
            category: "exploration",
            typical_gold_cost: SCOUT_COST,
            typical_duration_ticks: 20,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "DiplomaticAction" => &ActionMeta {
            type_name: "DiplomaticAction",
            category: "diplomacy",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: true,
            risk_level: 0.3,
        },
        "ProposeCoalition" => &ActionMeta {
            type_name: "ProposeCoalition",
            category: "diplomacy",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: true,
            risk_level: 0.2,
        },
        "RequestCoalitionAid" => &ActionMeta {
            type_name: "RequestCoalitionAid",
            category: "diplomacy",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 10,
            reversible: false,
            requires_party: false,
            affects_faction: true,
            risk_level: 0.1,
        },
        "UseAbility" => &ActionMeta {
            type_name: "UseAbility",
            category: "combat",
            typical_gold_cost: 10.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.2,
        },
        "SetSpendPriority" => &ActionMeta {
            type_name: "SetSpendPriority",
            category: "economy",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: true,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "StartingChoice" => &ActionMeta {
            type_name: "StartingChoice",
            category: "social",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.0,
        },
        "RespondToChoice" => &ActionMeta {
            type_name: "RespondToChoice",
            category: "social",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.1,
        },
        "InterceptChampion" => &ActionMeta {
            type_name: "InterceptChampion",
            category: "combat",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 100,
            reversible: false,
            requires_party: true,
            affects_faction: false,
            risk_level: 0.7,
        },
        _ => &ActionMeta {
            type_name: "Unknown",
            category: "other",
            typical_gold_cost: 0.0,
            typical_duration_ticks: 0,
            reversible: false,
            requires_party: false,
            affects_faction: false,
            risk_level: 0.5,
        },
    }
}

/// Encode action metadata as a feature vector (8 floats).
pub fn action_meta_features(action_type: &str) -> Vec<f32> {
    let meta = action_metadata(action_type);
    let category_idx = match meta.category {
        "combat" => 0.0,
        "economy" => 1.0,
        "diplomacy" => 2.0,
        "logistics" => 3.0,
        "social" => 4.0,
        "exploration" => 5.0,
        _ => 6.0,
    };
    vec![
        category_idx / 6.0,
        (meta.typical_gold_cost / 100.0).min(1.0),
        (meta.typical_duration_ticks as f32 / 200.0).min(1.0),
        if meta.reversible { 1.0 } else { 0.0 },
        if meta.requires_party { 1.0 } else { 0.0 },
        if meta.affects_faction { 1.0 } else { 0.0 },
        meta.risk_level,
        // Redundant but useful: is this a free action?
        if meta.typical_gold_cost == 0.0 { 1.0 } else { 0.0 },
    ]
}

// ---------------------------------------------------------------------------
// 2. Action Prerequisite Encoding
// ---------------------------------------------------------------------------

/// Why this action is valid — encodes the resource/state gates that allow it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionContext {
    /// Guild has enough gold for this action.
    pub gold_available: bool,
    /// There is a party available for the action (idle or in-field).
    pub party_available: bool,
    /// Faction relation meets the threshold (for diplomacy).
    pub faction_relation_threshold: bool,
    /// Cooldown is ready (for abilities).
    pub cooldown_ready: bool,
    /// All resources sufficient (supplies, inventory, etc.).
    pub resource_sufficient: bool,
}

impl ActionContext {
    /// Encode as 5 floats (0.0/1.0).
    pub fn to_features(&self) -> Vec<f32> {
        vec![
            if self.gold_available { 1.0 } else { 0.0 },
            if self.party_available { 1.0 } else { 0.0 },
            if self.faction_relation_threshold { 1.0 } else { 0.0 },
            if self.cooldown_ready { 1.0 } else { 0.0 },
            if self.resource_sufficient { 1.0 } else { 0.0 },
        ]
    }
}

/// Compute prerequisite context for a specific action.
pub fn action_context(action: &CampaignAction, state: &CampaignState) -> ActionContext {
    let meta = action_metadata(&action_type_name(action));
    let gold = state.guild.gold;

    let gold_available = gold >= meta.typical_gold_cost;

    let has_idle = state.adventurers.iter()
        .any(|a| a.status == AdventurerStatus::Idle);
    let has_field_party = state.parties.iter()
        .any(|p| matches!(p.status, PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting));

    let party_available = match action {
        CampaignAction::DispatchQuest { .. }
        | CampaignAction::InterceptChampion { .. } => has_idle || has_field_party,
        CampaignAction::PurchaseSupplies { .. }
        | CampaignAction::SendRunner { .. } => has_field_party,
        _ => true,
    };

    let faction_relation_threshold = match action {
        CampaignAction::ProposeCoalition { faction_id } => {
            state.factions.iter()
                .find(|f| f.id == *faction_id)
                .map(|f| f.relationship_to_guild > 60.0)
                .unwrap_or(false)
        }
        CampaignAction::RequestCoalitionAid { faction_id } => {
            state.factions.iter()
                .find(|f| f.id == *faction_id)
                .map(|f| f.coalition_member && f.military_strength > 20.0)
                .unwrap_or(false)
        }
        CampaignAction::DiplomaticAction { faction_id, action_type } => {
            let faction = state.factions.iter().find(|f| f.id == *faction_id);
            match (faction, action_type) {
                (Some(f), DiplomacyActionType::TradeAgreement) => {
                    matches!(f.diplomatic_stance, DiplomaticStance::Neutral | DiplomaticStance::Friendly)
                }
                (Some(f), DiplomacyActionType::ProposeCeasefire) => {
                    f.diplomatic_stance == DiplomaticStance::AtWar
                }
                _ => true,
            }
        }
        _ => true,
    };

    let cooldown_ready = match action {
        CampaignAction::UseAbility { unlock_id, .. } => {
            state.unlocks.iter()
                .find(|u| u.id == *unlock_id)
                .map(|u| u.cooldown_remaining_ms == 0)
                .unwrap_or(false)
        }
        _ => true,
    };

    let resource_sufficient = match action {
        CampaignAction::PurchaseSupplies { amount, .. } => {
            gold >= state.config.economy.supply_cost_per_unit * amount
        }
        CampaignAction::TrainAdventurer { .. } => gold >= state.config.economy.training_cost,
        CampaignAction::HireMercenary { .. } => gold >= state.config.economy.mercenary_cost,
        CampaignAction::CallRescue { .. } => {
            gold >= state.config.economy.rescue_bribe_cost
                || state.npc_relationships.iter().any(|r| r.rescue_available && r.relationship_score > 50.0)
        }
        CampaignAction::HireScout { .. } => gold >= state.config.economy.scout_cost,
        CampaignAction::SendRunner { .. } => gold >= state.config.economy.runner_cost,
        CampaignAction::UseAbility { unlock_id, .. } => {
            state.unlocks.iter()
                .find(|u| u.id == *unlock_id)
                .map(|u| gold >= u.properties.resource_cost)
                .unwrap_or(false)
        }
        _ => true,
    };

    ActionContext {
        gold_available,
        party_available,
        faction_relation_threshold,
        cooldown_ready,
        resource_sufficient,
    }
}

// ---------------------------------------------------------------------------
// 3. Action Outcome Prediction
// ---------------------------------------------------------------------------

/// Lightweight predicted outcome for an action.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Expected gold change (negative = cost).
    pub expected_gold_change: f32,
    /// Expected reputation change.
    pub expected_reputation_change: f32,
    /// Expected risk (0-1, chance of something going wrong).
    pub expected_risk: f32,
    /// Expected duration in ticks before outcome is visible.
    pub expected_duration: u32,
}

impl ActionOutcome {
    /// Encode as 4 floats.
    pub fn to_features(&self) -> Vec<f32> {
        vec![
            (self.expected_gold_change / 100.0).clamp(-1.0, 1.0),
            (self.expected_reputation_change / 20.0).clamp(-1.0, 1.0),
            self.expected_risk,
            (self.expected_duration as f32 / 200.0).min(1.0),
        ]
    }
}

/// Predict the outcome of an action given the current state.
pub fn predict_outcome(action: &CampaignAction, state: &CampaignState) -> ActionOutcome {
    match action {
        CampaignAction::Wait => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 1,
        },
        CampaignAction::Rest => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 1,
        },
        CampaignAction::AcceptQuest { request_id } => {
            let reward = state.request_board.iter()
                .find(|r| r.id == *request_id)
                .map(|r| r.reward.gold)
                .unwrap_or(50.0);
            let threat = state.request_board.iter()
                .find(|r| r.id == *request_id)
                .map(|r| r.threat_level / 100.0)
                .unwrap_or(0.3);
            ActionOutcome {
                expected_gold_change: reward * 0.7, // average completion * reward
                expected_reputation_change: 5.0,
                expected_risk: threat,
                expected_duration: 0,
            }
        }
        CampaignAction::DeclineQuest { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: -1.0,
            expected_risk: 0.0,
            expected_duration: 0,
        },
        CampaignAction::DispatchQuest { quest_id } => {
            let quest = state.active_quests.iter().find(|q| q.id == *quest_id);
            let party_size = quest.map(|q| q.assigned_pool.len()).unwrap_or(3) as f32;
            let threat = quest
                .map(|q| q.request.threat_level / 100.0)
                .unwrap_or(0.3);
            ActionOutcome {
                expected_gold_change: 0.0, // reward comes at completion
                expected_reputation_change: 0.0,
                expected_risk: (threat - party_size * 0.1).clamp(0.0, 1.0),
                expected_duration: 200,
            }
        }
        CampaignAction::PurchaseSupplies { amount, .. } => ActionOutcome {
            expected_gold_change: -state.config.economy.supply_cost_per_unit * amount,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 0,
        },
        CampaignAction::TrainAdventurer { .. } => ActionOutcome {
            expected_gold_change: -state.config.economy.training_cost,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 50,
        },
        CampaignAction::EquipGear { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 0,
        },
        CampaignAction::SendRunner { payload, .. } => {
            let cost = match payload {
                RunnerPayload::Supplies(_) => state.config.economy.runner_cost + 10.0,
                RunnerPayload::Message => state.config.economy.runner_cost,
            };
            ActionOutcome {
                expected_gold_change: -cost,
                expected_reputation_change: 0.0,
                expected_risk: 0.1,
                expected_duration: 30,
            }
        }
        CampaignAction::HireMercenary { .. } => ActionOutcome {
            expected_gold_change: -state.config.economy.mercenary_cost,
            expected_reputation_change: 0.0,
            expected_risk: 0.2,
            expected_duration: 0,
        },
        CampaignAction::CallRescue { .. } => {
            let has_free = state.npc_relationships.iter()
                .any(|r| r.rescue_available && r.relationship_score > 50.0);
            let cost = if has_free { 0.0 } else { state.config.economy.rescue_bribe_cost };
            ActionOutcome {
                expected_gold_change: -cost,
                expected_reputation_change: 0.0,
                expected_risk: 0.1,
                expected_duration: 10,
            }
        }
        CampaignAction::HireScout { .. } => ActionOutcome {
            expected_gold_change: -state.config.economy.scout_cost,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 20,
        },
        CampaignAction::DiplomaticAction { action_type, .. } => {
            let (rep_change, risk) = match action_type {
                DiplomacyActionType::ImproveRelations => (3.0, 0.0),
                DiplomacyActionType::TradeAgreement => (2.0, 0.1),
                DiplomacyActionType::RequestAid => (0.0, 0.2),
                DiplomacyActionType::Threaten => (-5.0, 0.5),
                DiplomacyActionType::ProposeCeasefire => (5.0, 0.3),
            };
            let gold_from_trade = match action_type {
                DiplomacyActionType::TradeAgreement => 10.0,
                _ => 0.0,
            };
            ActionOutcome {
                expected_gold_change: gold_from_trade,
                expected_reputation_change: rep_change,
                expected_risk: risk,
                expected_duration: 0,
            }
        }
        CampaignAction::ProposeCoalition { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 5.0,
            expected_risk: 0.2,
            expected_duration: 0,
        },
        CampaignAction::RequestCoalitionAid { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: -2.0,
            expected_risk: 0.1,
            expected_duration: 10,
        },
        CampaignAction::UseAbility { unlock_id, .. } => {
            let cost = state.unlocks.iter()
                .find(|u| u.id == *unlock_id)
                .map(|u| u.properties.resource_cost)
                .unwrap_or(0.0);
            ActionOutcome {
                expected_gold_change: -cost,
                expected_reputation_change: 0.0,
                expected_risk: 0.2,
                expected_duration: 0,
            }
        }
        CampaignAction::SetSpendPriority { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 0,
        },
        CampaignAction::ChooseStartingPackage { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 0,
        },
        CampaignAction::RespondToChoice { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.1,
            expected_duration: 0,
        },
        CampaignAction::AssignToPool { .. } | CampaignAction::UnassignFromPool { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 0.0,
            expected_risk: 0.0,
            expected_duration: 0,
        },
        CampaignAction::InterceptChampion { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 10.0,
            expected_risk: 0.7,
            expected_duration: 100,
        },
        CampaignAction::UseClassSkill { .. } => ActionOutcome {
            expected_gold_change: 0.0,
            expected_reputation_change: 2.0,
            expected_risk: 0.1,
            expected_duration: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// 4. Action Synergy Scoring
// ---------------------------------------------------------------------------

/// Score how well a candidate action synergizes with recent actions.
/// Returns 0.0 (no synergy) to 1.0 (strong synergy).
pub fn action_synergy(recent_actions: &[String], candidate: &str) -> f32 {
    if recent_actions.is_empty() {
        return 0.0;
    }

    let mut score = 0.0f32;

    // Define synergy pairs: (recent, candidate, bonus)
    let synergies: &[(&str, &str, f32)] = &[
        // Training before quest dispatch = better prepared
        ("TrainAdventurer", "DispatchQuest", 0.4),
        ("TrainAdventurer", "AssignToPool", 0.2),
        // Scouting before dispatch = informed decisions
        ("HireScout", "DispatchQuest", 0.5),
        ("HireScout", "AcceptQuest", 0.3),
        // Equip before dispatch = stronger party
        ("EquipGear", "DispatchQuest", 0.3),
        ("EquipGear", "AssignToPool", 0.2),
        // Accept then assign then dispatch = quest pipeline
        ("AcceptQuest", "AssignToPool", 0.5),
        ("AssignToPool", "DispatchQuest", 0.6),
        // Trade agreement followed by coalition = diplomatic chain
        ("DiplomaticAction", "ProposeCoalition", 0.4),
        // Coalition followed by military aid = funded expansion
        ("ProposeCoalition", "RequestCoalitionAid", 0.5),
        // Purchase supplies for parties before sending runner
        ("PurchaseSupplies", "SendRunner", 0.3),
        // Rest then train = recover and grow
        ("Rest", "TrainAdventurer", 0.3),
        // Hire mercenary during active combat = combat support
        ("HireMercenary", "CallRescue", 0.2),
        // Set spend priority then follow through
        ("SetSpendPriority", "PurchaseSupplies", 0.2),
        ("SetSpendPriority", "TrainAdventurer", 0.2),
        ("SetSpendPriority", "HireMercenary", 0.2),
    ];

    // Check each recent action against candidate
    for (i, recent) in recent_actions.iter().rev().enumerate() {
        // Recency weight: more recent actions get higher weight
        let recency_weight = 1.0 / (i as f32 + 1.0);

        // Strip subtypes for matching
        let recent_base = if recent.starts_with("DiplomaticAction_") {
            "DiplomaticAction"
        } else if recent.starts_with("SetSpendPriority_") {
            "SetSpendPriority"
        } else {
            recent.as_str()
        };

        let candidate_base = if candidate.starts_with("DiplomaticAction_") {
            "DiplomaticAction"
        } else if candidate.starts_with("SetSpendPriority_") {
            "SetSpendPriority"
        } else {
            candidate
        };

        for &(syn_recent, syn_candidate, bonus) in synergies {
            if recent_base == syn_recent && candidate_base == syn_candidate {
                score += bonus * recency_weight;
            }
        }
    }

    score.min(1.0)
}

// ---------------------------------------------------------------------------
// 5. Action Space Summary Token
// ---------------------------------------------------------------------------

/// Compact summary of the current action space.
/// Encoded as an EntityToken (type_id = 10).
pub fn action_space_summary_token(state: &CampaignState) -> EntityToken {
    let valid = state.valid_actions();
    let total = valid.len() as f32;

    if total == 0.0 {
        return EntityToken {
            type_id: 10,
            features: vec![0.0; 14],
        };
    }

    // Count by category
    let mut combat = 0u32;
    let mut economy = 0u32;
    let mut diplomacy = 0u32;
    let mut logistics = 0u32;
    let mut social = 0u32;
    let mut exploration = 0u32;

    // Track high-value action availability
    let mut has_dispatch = false;
    let mut has_hire_merc = false;
    let mut has_scout = false;
    let mut has_coalition = false;
    let mut has_rescue = false;
    let mut has_ability = false;

    // Count distinct action types for entropy
    let mut type_set = std::collections::HashSet::new();

    for action in &valid {
        let type_name = action_type_name(action);
        let meta = action_metadata(&type_name);
        type_set.insert(type_name.clone());

        match meta.category {
            "combat" => combat += 1,
            "economy" => economy += 1,
            "diplomacy" => diplomacy += 1,
            "logistics" => logistics += 1,
            "social" => social += 1,
            "exploration" => exploration += 1,
            _ => {}
        }

        match type_name.as_str() {
            "DispatchQuest" => has_dispatch = true,
            "HireMercenary" => has_hire_merc = true,
            "HireScout" => has_scout = true,
            "ProposeCoalition" => has_coalition = true,
            "CallRescue" => has_rescue = true,
            "UseAbility" => has_ability = true,
            _ => {}
        }
    }

    // Action space entropy: how many distinct types / total actions
    let distinct_types = type_set.len() as f32;
    let entropy = distinct_types / total.max(1.0);

    EntityToken {
        type_id: 10,
        features: vec![
            // Category counts (normalized by total)
            combat as f32 / total,
            economy as f32 / total,
            diplomacy as f32 / total,
            logistics as f32 / total,
            social as f32 / total,
            exploration as f32 / total,
            // High-value action availability (binary)
            if has_dispatch { 1.0 } else { 0.0 },
            if has_hire_merc { 1.0 } else { 0.0 },
            if has_scout { 1.0 } else { 0.0 },
            if has_coalition { 1.0 } else { 0.0 },
            if has_rescue { 1.0 } else { 0.0 },
            if has_ability { 1.0 } else { 0.0 },
            // Entropy and scale
            entropy,
            (total / 50.0).min(1.0), // total action count normalized
        ],
    }
}
