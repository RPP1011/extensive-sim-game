//! Guild contracts/commissions system — every 300 ticks.
//!
//! NPCs and factions commission specific work with deadlines and penalties.
//! Contracts refresh every 1000 ticks, with 1-3 available at a time.
//! Difficulty and rewards scale with guild tier (reputation-based).
//! Max 5 active accepted contracts at once.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    lcg_f32, lcg_next, CampaignState, Contract, ContractTask,
};

/// Maximum number of simultaneously accepted contracts.
const MAX_ACTIVE_CONTRACTS: usize = 5;

/// How often to tick contracts (in ticks).
const CONTRACT_TICK_INTERVAL: u64 = 10;

/// How often to refresh available contracts (in ticks).
const CONTRACT_REFRESH_INTERVAL: u64 = 33;

/// Run the contracts system. Called every tick; gates internally on cadence.
pub fn tick_contracts(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CONTRACT_TICK_INTERVAL != 0 {
        return;
    }

    // --- Refresh available contracts ---
    if state.tick % CONTRACT_REFRESH_INTERVAL == 0 {
        generate_contracts(state, events);
    }

    // --- Check deadline enforcement on accepted contracts ---
    enforce_deadlines(state, events);

    // --- Auto-complete check ---
    check_completions(state, events);
}

/// Generate 1-3 new available (not yet accepted) contracts.
/// Clears old unaccepted contracts first.
fn generate_contracts(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Remove unaccepted contracts (keep accepted ones)
    state.contracts.retain(|c| c.accepted);

    // Determine guild tier from reputation (0-3)
    let guild_tier = (state.guild.reputation / 25.0).min(3.0).max(0.0) as u32;

    // Generate 1-3 new contracts
    let count = 1 + (lcg_next(&mut state.rng) % 3) as usize;

    for _ in 0..count {
        let contract = generate_single_contract(state, guild_tier);
        events.push(WorldEvent::ContractOffered {
            contract_id: contract.id,
            commissioner: contract.commissioner.clone(),
            reward_gold: contract.reward_gold,
        });
        state.contracts.push(contract);
    }
}

/// Generate a single contract with difficulty/reward scaled to guild tier.
fn generate_single_contract(state: &mut CampaignState, guild_tier: u32) -> Contract {
    let id = state.next_contract_id;
    state.next_contract_id += 1;

    let tier_mult = 1.0 + guild_tier as f32 * 0.5; // 1.0, 1.5, 2.0, 2.5

    // Pick a task type (6 variants)
    let task_roll = lcg_next(&mut state.rng) % 6;
    let task = match task_roll {
        0 => {
            let building_types = ["barracks", "watchtower", "trade_post", "infirmary", "training_grounds"];
            let idx = lcg_next(&mut state.rng) as usize % building_types.len();
            ContractTask::BuildStructure {
                building_type: building_types[idx].to_string(),
            }
        }
        1 => {
            let resources = ["wood", "iron", "stone", "herbs", "arcane_dust"];
            let idx = lcg_next(&mut state.rng) as usize % resources.len();
            let amount = (20.0 + lcg_f32(&mut state.rng) * 30.0) * tier_mult;
            ContractTask::DeliverResources {
                resource: resources[idx].to_string(),
                amount,
            }
        }
        2 => {
            let region_id = if state.overworld.regions.is_empty() {
                0
            } else {
                lcg_next(&mut state.rng) as usize % state.overworld.regions.len()
            };
            let duration = 500 + (lcg_next(&mut state.rng) as u64 % 1000);
            ContractTask::DefendRegion {
                region_id,
                duration,
            }
        }
        3 => {
            let count = 1 + (lcg_next(&mut state.rng) % (2 + guild_tier));
            let min_level = 1 + (lcg_next(&mut state.rng) % (1 + guild_tier));
            ContractTask::TrainAdventurers { count, min_level }
        }
        4 => {
            let region_id = if state.overworld.regions.is_empty() {
                0
            } else {
                lcg_next(&mut state.rng) as usize % state.overworld.regions.len()
            };
            ContractTask::ClearThreats { region_id }
        }
        _ => {
            let faction_id = if state.factions.is_empty() {
                0
            } else {
                lcg_next(&mut state.rng) as usize % state.factions.len()
            };
            ContractTask::EstablishTrade { faction_id }
        }
    };

    // Pick a commissioner
    let commissioner = pick_commissioner(state);

    // Pick a faction for the commission (sometimes faction-specific)
    let faction_id = if lcg_f32(&mut state.rng) < 0.5 && !state.factions.is_empty() {
        Some(lcg_next(&mut state.rng) as usize % state.factions.len())
    } else {
        None
    };

    // Scale rewards with tier
    let base_gold = 30.0 + lcg_f32(&mut state.rng) * 50.0;
    let reward_gold = base_gold * tier_mult;
    let reward_reputation = (3.0 + lcg_f32(&mut state.rng) * 5.0) * tier_mult;

    // Penalties are a fraction of rewards
    let penalty_gold = reward_gold * 0.3;
    let penalty_reputation = reward_reputation * 0.5;

    // Deadline: 67-167 turns from now (~3-8 minutes), shorter at higher tier
    let base_deadline = 167 - (guild_tier as u64 * 17);
    let deadline_variance = lcg_next(&mut state.rng) as u64 % 67;
    let deadline_tick = state.tick + base_deadline.saturating_sub(deadline_variance).max(50);

    Contract {
        id,
        commissioner,
        faction_id,
        task,
        reward_gold,
        reward_reputation,
        penalty_gold,
        penalty_reputation,
        deadline_tick,
        accepted: false,
        completed: false,
        failed: false,
    }
}

/// Pick a commissioner name from NPCs or generate a generic one.
fn pick_commissioner(state: &mut CampaignState) -> String {
    // Try to pick from existing NPC relationships
    if !state.npc_relationships.is_empty() && lcg_f32(&mut state.rng) < 0.6 {
        let idx = lcg_next(&mut state.rng) as usize % state.npc_relationships.len();
        return state.npc_relationships[idx].npc_name.clone();
    }

    // Try faction leaders
    if !state.factions.is_empty() && lcg_f32(&mut state.rng) < 0.5 {
        let idx = lcg_next(&mut state.rng) as usize % state.factions.len();
        return format!("{} representative", state.factions[idx].name);
    }

    // Generic commissioners
    let names = [
        "Guild Board",
        "Town Council",
        "Merchant Consortium",
        "Regional Governor",
        "Border Marshal",
    ];
    let idx = lcg_next(&mut state.rng) as usize % names.len();
    names[idx].to_string()
}

/// Enforce deadlines: fail any accepted, incomplete contracts past their deadline.
fn enforce_deadlines(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // Collect indices of contracts to fail
    let to_fail: Vec<usize> = state
        .contracts
        .iter()
        .enumerate()
        .filter(|(_, c)| c.accepted && !c.completed && !c.failed && tick > c.deadline_tick)
        .map(|(i, _)| i)
        .collect();

    for idx in to_fail {
        let contract = &mut state.contracts[idx];
        contract.failed = true;

        let penalty_gold = contract.penalty_gold;
        let penalty_rep = contract.penalty_reputation;
        let contract_id = contract.id;
        let faction_id = contract.faction_id;

        // Apply penalties
        state.guild.gold = (state.guild.gold - penalty_gold).max(0.0);
        state.guild.reputation = (state.guild.reputation - penalty_rep).max(0.0);

        // Faction relation penalty
        if let Some(fid) = faction_id {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == fid) {
                faction.relationship_to_guild =
                    (faction.relationship_to_guild - penalty_rep * 0.5).max(-100.0);
            }
        }

        events.push(WorldEvent::ContractFailed {
            contract_id,
            penalty_gold,
            penalty_reputation: penalty_rep,
        });
    }
}

/// Check if any accepted contracts have been completed by examining game state.
fn check_completions(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // We need to gather completion info without holding a mutable borrow on contracts.
    // First pass: identify which contracts are complete.
    let completions: Vec<(usize, bool)> = state
        .contracts
        .iter()
        .enumerate()
        .filter(|(_, c)| c.accepted && !c.completed && !c.failed)
        .map(|(i, c)| (i, is_task_complete(c, state)))
        .filter(|(_, complete)| *complete)
        .collect();

    for (idx, _) in completions {
        let contract = &mut state.contracts[idx];
        contract.completed = true;

        let reward_gold = contract.reward_gold;
        let reward_rep = contract.reward_reputation;
        let contract_id = contract.id;
        let faction_id = contract.faction_id;

        // Apply rewards
        state.guild.gold += reward_gold;
        state.guild.reputation = (state.guild.reputation + reward_rep).min(100.0);

        // Faction relation boost
        if let Some(fid) = faction_id {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == fid) {
                faction.relationship_to_guild =
                    (faction.relationship_to_guild + reward_rep * 0.5).min(100.0);
            }
        }

        events.push(WorldEvent::ContractCompleted {
            contract_id,
            reward_gold,
            reward_reputation: reward_rep,
        });
    }
}

/// Check whether a contract's task completion conditions are met.
fn is_task_complete(contract: &Contract, state: &CampaignState) -> bool {
    match &contract.task {
        ContractTask::BuildStructure { building_type } => {
            // Check if the named building has been upgraded at least once
            match building_type.as_str() {
                "barracks" => state.guild_buildings.barracks >= 1,
                "watchtower" => state.guild_buildings.watchtower >= 1,
                "trade_post" => state.guild_buildings.trade_post >= 1,
                "infirmary" => state.guild_buildings.infirmary >= 1,
                "training_grounds" => state.guild_buildings.training_grounds >= 1,
                "war_room" => state.guild_buildings.war_room >= 1,
                _ => false,
            }
        }
        ContractTask::DeliverResources { amount, .. } => {
            // Check if guild has enough supplies (proxy for resources)
            state.guild.supplies >= *amount
        }
        ContractTask::DefendRegion {
            region_id,
            duration,
        } => {
            // Region control must remain above 50 for the duration since contract acceptance
            // Simplified: check current control is high and enough time has passed
            if let Some(region) = state.overworld.regions.get(*region_id) {
                let ticks_since_accepted = state.tick.saturating_sub(
                    contract.deadline_tick.saturating_sub(5000), // approximate acceptance time
                );
                region.control > 50.0 && ticks_since_accepted >= *duration
            } else {
                false
            }
        }
        ContractTask::TrainAdventurers { count, min_level } => {
            // Check if the guild has enough adventurers at/above the required level
            let qualified = state
                .adventurers
                .iter()
                .filter(|a| a.level >= *min_level && a.status != crate::headless_campaign::state::AdventurerStatus::Dead)
                .count();
            qualified >= *count as usize
        }
        ContractTask::ClearThreats { region_id } => {
            // Region threat level must be below 20
            if let Some(region) = state.overworld.regions.get(*region_id) {
                region.threat_level < 20.0
            } else {
                false
            }
        }
        ContractTask::EstablishTrade { faction_id } => {
            // Faction relationship must be at least 30 (friendly-ish)
            if let Some(faction) = state.factions.iter().find(|f| f.id == *faction_id) {
                faction.relationship_to_guild >= 30.0
            } else {
                false
            }
        }
    }
}

/// Check if a new contract can be accepted (under the limit).
pub fn can_accept_contract(state: &CampaignState) -> bool {
    let active_count = state
        .contracts
        .iter()
        .filter(|c| c.accepted && !c.completed && !c.failed)
        .count();
    active_count < MAX_ACTIVE_CONTRACTS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        // Skip to Playing phase for testing
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;
        state.tick = 1000;
        state
    }

    #[test]
    fn generate_contracts_produces_valid_output() {
        let mut state = make_test_state();
        let mut events = Vec::new();
        generate_contracts(&mut state, &mut events);

        assert!(
            !state.contracts.is_empty(),
            "Should generate at least 1 contract"
        );
        assert!(state.contracts.len() <= 3, "Should generate at most 3 contracts");

        for contract in &state.contracts {
            assert!(!contract.accepted);
            assert!(!contract.completed);
            assert!(!contract.failed);
            assert!(contract.reward_gold > 0.0);
            assert!(contract.deadline_tick > state.tick);
        }

        // Check events
        let offered_count = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::ContractOffered { .. }))
            .count();
        assert_eq!(offered_count, state.contracts.len());
    }

    #[test]
    fn deadline_enforcement_fails_expired_contracts() {
        let mut state = make_test_state();
        state.contracts.push(Contract {
            id: 1,
            commissioner: "Test NPC".into(),
            faction_id: None,
            task: ContractTask::ClearThreats { region_id: 0 },
            reward_gold: 50.0,
            reward_reputation: 5.0,
            penalty_gold: 15.0,
            penalty_reputation: 2.5,
            deadline_tick: 500, // Already past
            accepted: true,
            completed: false,
            failed: false,
        });

        let gold_before = state.guild.gold;
        let rep_before = state.guild.reputation;
        let mut events = Vec::new();

        enforce_deadlines(&mut state, &mut events);

        assert!(state.contracts[0].failed);
        assert!(state.guild.gold < gold_before);
        assert!(state.guild.reputation < rep_before);
        assert!(events
            .iter()
            .any(|e| matches!(e, WorldEvent::ContractFailed { .. })));
    }

    #[test]
    fn max_active_contracts_enforced() {
        let mut state = make_test_state();
        for i in 0..5 {
            state.contracts.push(Contract {
                id: i,
                commissioner: "Test".into(),
                faction_id: None,
                task: ContractTask::ClearThreats { region_id: 0 },
                reward_gold: 50.0,
                reward_reputation: 5.0,
                penalty_gold: 15.0,
                penalty_reputation: 2.5,
                deadline_tick: 5000,
                accepted: true,
                completed: false,
                failed: false,
            });
        }
        assert!(!can_accept_contract(&state));
    }

    #[test]
    fn completed_contract_awards_rewards() {
        let mut state = make_test_state();
        // Add adventurers so TrainAdventurers can complete
        state.adventurers.push(crate::headless_campaign::state::Adventurer {
            id: 10,
            name: "Test Hero".into(),
            archetype: "knight".into(),
            level: 5,
            xp: 0,
            stats: Default::default(),
            equipment: Default::default(),
            traits: Vec::new(),
            status: crate::headless_campaign::state::AdventurerStatus::Idle,
            loyalty: 70.0,
            stress: 0.0,
            fatigue: 0.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 80.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),

            backstory: None,

            deeds: Vec::new(),

            hobbies: Vec::new(),

            disease_status: DiseaseStatus::Healthy,

            mood_state: MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
        });

        state.contracts.push(Contract {
            id: 1,
            commissioner: "Test NPC".into(),
            faction_id: None,
            task: ContractTask::TrainAdventurers {
                count: 1,
                min_level: 1,
            },
            reward_gold: 100.0,
            reward_reputation: 10.0,
            penalty_gold: 30.0,
            penalty_reputation: 5.0,
            deadline_tick: 5000,
            accepted: true,
            completed: false,
            failed: false,
        });

        let gold_before = state.guild.gold;
        let mut events = Vec::new();

        check_completions(&mut state, &mut events);

        assert!(state.contracts[0].completed);
        assert_eq!(state.guild.gold, gold_before + 100.0);
        assert!(events
            .iter()
            .any(|e| matches!(e, WorldEvent::ContractCompleted { .. })));
    }
}
