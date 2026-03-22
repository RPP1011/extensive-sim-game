//! Campaign step orchestrator.
//!
//! Chains all tick systems in the correct order and returns a
//! `CampaignStepResult` with events, deltas, violations, and outcome.

use super::actions::*;
use super::state::*;
use super::systems;

/// Advance the campaign by one fixed tick (100ms).
///
/// Optionally applies a player action before ticking the world.
/// Returns a detailed result with events, deltas, and any violations.
pub fn step_campaign(
    state: &mut CampaignState,
    action: Option<CampaignAction>,
) -> CampaignStepResult {
    let mut events = Vec::new();
    let mut deltas = StepDeltas::default();

    // Snapshot economy for deltas
    deltas.gold_before = state.guild.gold;
    deltas.supplies_before = state.guild.supplies;
    deltas.reputation_before = state.guild.reputation;

    // --- Apply player action (before ticking) ---
    let action_result = if let Some(action) = action {
        Some(apply_action(state, action, &mut deltas, &mut events))
    } else {
        None
    };

    // --- Advance tick ---
    state.tick += 1;
    state.elapsed_ms = state.tick * CAMPAIGN_TICK_MS as u64;

    // --- Run tick systems in order ---
    // Every-tick systems
    systems::travel::tick_travel(state, &mut deltas, &mut events);
    systems::supply::tick_supply(state, &mut deltas, &mut events);
    systems::battles::tick_battles(state, &mut deltas, &mut events);
    systems::quest_lifecycle::tick_quest_lifecycle(state, &mut deltas, &mut events);
    systems::quest_expiry::tick_quest_expiry(state, &mut deltas, &mut events);
    systems::economy::tick_economy(state, &mut deltas, &mut events);
    systems::cooldowns::tick_cooldowns(state, &mut deltas, &mut events);

    // Cadenced systems (check tick modulo internally)
    systems::quest_generation::tick_quest_generation(state, &mut deltas, &mut events);
    systems::adventurer_condition::tick_adventurer_condition(state, &mut deltas, &mut events);
    systems::adventurer_recovery::tick_adventurer_recovery(state, &mut deltas, &mut events);
    systems::npc_relationships::tick_npc_relationships(state, &mut deltas, &mut events);
    systems::faction_ai::tick_faction_ai(state, &mut deltas, &mut events);
    systems::progression::tick_progression(state, &mut deltas, &mut events);
    systems::recruitment::tick_recruitment(state, &mut deltas, &mut events);
    systems::threat::tick_threat(state, &mut deltas, &mut events);

    // Final economy snapshot
    deltas.gold_after = state.guild.gold;
    deltas.supplies_after = state.guild.supplies;
    deltas.reputation_after = state.guild.reputation;

    // --- Verify invariants ---
    let violations = systems::verify::verify_invariants(state);

    // --- Check campaign outcome ---
    let outcome = check_campaign_outcome(state);

    CampaignStepResult {
        events,
        violations,
        outcome,
        deltas,
        action_result,
        tick: state.tick,
        elapsed_ms: state.elapsed_ms,
    }
}

/// Apply a player action to the campaign state.
fn apply_action(
    state: &mut CampaignState,
    action: CampaignAction,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) -> ActionResult {
    match action {
        CampaignAction::Wait => ActionResult::Success("Observing...".into()),

        CampaignAction::AcceptQuest { request_id } => {
            if state.active_quests.len() >= state.guild.active_quest_capacity {
                return ActionResult::Failed("Quest capacity reached".into());
            }
            let idx = state.request_board.iter().position(|r| r.id == request_id);
            match idx {
                Some(i) => {
                    let request = state.request_board.remove(i);
                    state.active_quests.push(ActiveQuest {
                        id: request.id,
                        request,
                        status: ActiveQuestStatus::Preparing,
                        assigned_pool: Vec::new(),
                        dispatched_party_id: None,
                        elapsed_ms: 0,
                        events: Vec::new(),
                    });
                    ActionResult::Success(format!("Accepted quest {}", request_id))
                }
                None => ActionResult::InvalidAction(format!(
                    "Quest request {} not found",
                    request_id
                )),
            }
        }

        CampaignAction::DeclineQuest { request_id } => {
            let before = state.request_board.len();
            state.request_board.retain(|r| r.id != request_id);
            if state.request_board.len() < before {
                ActionResult::Success(format!("Declined quest {}", request_id))
            } else {
                ActionResult::InvalidAction(format!(
                    "Quest request {} not found",
                    request_id
                ))
            }
        }

        CampaignAction::AssignToPool {
            adventurer_id,
            quest_id,
        } => {
            let adv = state.adventurers.iter_mut().find(|a| a.id == adventurer_id);
            match adv {
                Some(adv) if adv.status == AdventurerStatus::Idle => {
                    if let Some(quest) = state.active_quests.iter_mut().find(|q| q.id == quest_id) {
                        if quest.status == ActiveQuestStatus::Preparing {
                            quest.assigned_pool.push(adventurer_id);
                            adv.status = AdventurerStatus::Assigned;
                            ActionResult::Success(format!(
                                "Assigned adventurer {} to quest {}",
                                adventurer_id, quest_id
                            ))
                        } else {
                            ActionResult::Failed("Quest not in Preparing status".into())
                        }
                    } else {
                        ActionResult::InvalidAction(format!("Quest {} not found", quest_id))
                    }
                }
                Some(_) => ActionResult::Failed("Adventurer not idle".into()),
                None => ActionResult::InvalidAction(format!(
                    "Adventurer {} not found",
                    adventurer_id
                )),
            }
        }

        CampaignAction::UnassignFromPool {
            adventurer_id,
            quest_id,
        } => {
            if let Some(quest) = state.active_quests.iter_mut().find(|q| q.id == quest_id) {
                quest.assigned_pool.retain(|&id| id != adventurer_id);
                if let Some(adv) = state
                    .adventurers
                    .iter_mut()
                    .find(|a| a.id == adventurer_id)
                {
                    adv.status = AdventurerStatus::Idle;
                }
                ActionResult::Success(format!(
                    "Unassigned adventurer {} from quest {}",
                    adventurer_id, quest_id
                ))
            } else {
                ActionResult::InvalidAction(format!("Quest {} not found", quest_id))
            }
        }

        CampaignAction::DispatchQuest { quest_id } => {
            if let Some(quest) = state.active_quests.iter_mut().find(|q| q.id == quest_id) {
                if quest.assigned_pool.is_empty() {
                    return ActionResult::Failed("No adventurers in pool".into());
                }
                if quest.status != ActiveQuestStatus::Preparing {
                    return ActionResult::Failed("Quest not in Preparing status".into());
                }

                // Form party from the pool
                let party_id = state.next_party_id;
                state.next_party_id += 1;

                let member_ids = quest.assigned_pool.clone();
                let target = quest.request.target_position;

                // Set adventurers to traveling
                for &mid in &member_ids {
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                        adv.status = AdventurerStatus::Traveling;
                        adv.party_id = Some(party_id);
                    }
                }

                let party = Party {
                    id: party_id,
                    member_ids: member_ids.clone(),
                    position: state.guild.base.position,
                    destination: Some(target),
                    speed: state.config.quest_lifecycle.party_speed,
                    status: PartyStatus::Traveling,
                    supply_level: state.config.quest_lifecycle.party_starting_supply,
                    morale: state.config.quest_lifecycle.party_starting_morale,
                    quest_id: Some(quest_id),
                };

                quest.status = ActiveQuestStatus::Dispatched;
                quest.dispatched_party_id = Some(party_id);

                events.push(WorldEvent::PartyFormed {
                    party_id,
                    member_ids: member_ids.clone(),
                });
                events.push(WorldEvent::QuestDispatched {
                    quest_id,
                    party_id,
                    member_count: member_ids.len(),
                });

                state.parties.push(party);

                ActionResult::Success(format!("Dispatched quest {} with party {}", quest_id, party_id))
            } else {
                ActionResult::InvalidAction(format!("Quest {} not found", quest_id))
            }
        }

        CampaignAction::PurchaseSupplies { party_id, amount } => {
            let cost = amount * state.config.economy.supply_cost_per_unit;
            if state.guild.gold < cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
                state.guild.gold -= cost;
                party.supply_level = (party.supply_level + amount).min(100.0);
                ActionResult::Success(format!(
                    "Purchased {} supplies for party {} (cost: {:.0}g)",
                    amount, party_id, cost
                ))
            } else {
                ActionResult::InvalidAction(format!("Party {} not found", party_id))
            }
        }

        CampaignAction::TrainAdventurer {
            adventurer_id,
            training_type,
        } => {
            if state.guild.gold < state.config.economy.training_cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == adventurer_id)
            {
                if adv.status != AdventurerStatus::Idle {
                    return ActionResult::Failed("Adventurer not idle".into());
                }
                state.guild.gold -= state.config.economy.training_cost;
                match training_type {
                    TrainingType::Combat => adv.stats.attack += 2.0,
                    TrainingType::Exploration => adv.stats.speed += 1.0,
                    TrainingType::Leadership => adv.resolve = (adv.resolve + 5.0).min(100.0),
                    TrainingType::Survival => adv.stats.defense += 2.0,
                }
                adv.xp += 50;
                ActionResult::Success(format!(
                    "Trained adventurer {} in {:?}",
                    adventurer_id, training_type
                ))
            } else {
                ActionResult::InvalidAction(format!(
                    "Adventurer {} not found",
                    adventurer_id
                ))
            }
        }

        CampaignAction::EquipGear {
            adventurer_id,
            item_id,
        } => {
            let item_idx = state.guild.inventory.iter().position(|i| i.id == item_id);
            match item_idx {
                Some(idx) => {
                    let item = state.guild.inventory.remove(idx);
                    if let Some(adv) = state
                        .adventurers
                        .iter_mut()
                        .find(|a| a.id == adventurer_id)
                    {
                        // Apply stat bonuses
                        adv.stats.max_hp += item.stat_bonuses.hp_bonus;
                        adv.stats.attack += item.stat_bonuses.attack_bonus;
                        adv.stats.defense += item.stat_bonuses.defense_bonus;
                        adv.stats.speed += item.stat_bonuses.speed_bonus;
                        // Store equipped item ID
                        match item.slot {
                            EquipmentSlot::Weapon => adv.equipment.weapon = Some(item.id),
                            EquipmentSlot::Offhand => adv.equipment.offhand = Some(item.id),
                            EquipmentSlot::Chest => adv.equipment.chest = Some(item.id),
                            EquipmentSlot::Boots => adv.equipment.boots = Some(item.id),
                            EquipmentSlot::Accessory => adv.equipment.accessory = Some(item.id),
                        }
                        ActionResult::Success(format!(
                            "Equipped {} on adventurer {}",
                            item.name, adventurer_id
                        ))
                    } else {
                        // Put item back
                        state.guild.inventory.push(item);
                        ActionResult::InvalidAction(format!(
                            "Adventurer {} not found",
                            adventurer_id
                        ))
                    }
                }
                None => ActionResult::InvalidAction(format!("Item {} not found", item_id)),
            }
        }

        CampaignAction::SendRunner { party_id, payload } => {
            let runner_cost = state.config.economy.runner_cost;
            if state.guild.gold < runner_cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
                state.guild.gold -= runner_cost;
                match &payload {
                    RunnerPayload::Supplies(amount) => {
                        party.supply_level = (party.supply_level + amount).min(100.0);
                    }
                    RunnerPayload::Message => {
                        party.morale = (party.morale + 10.0).min(100.0);
                    }
                }
                deltas.runners_sent += 1;
                events.push(WorldEvent::RunnerSent {
                    party_id,
                    cost: runner_cost,
                });
                ActionResult::Success(format!("Runner sent to party {}", party_id))
            } else {
                ActionResult::InvalidAction(format!("Party {} not found", party_id))
            }
        }

        CampaignAction::HireMercenary { quest_id } => {
            let merc_cost = state.config.economy.mercenary_cost;
            if state.guild.gold < merc_cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            state.guild.gold -= merc_cost;
            // Boost the battle's predicted outcome
            if let Some(battle) = state
                .active_battles
                .iter_mut()
                .find(|b| b.quest_id == quest_id)
            {
                battle.predicted_outcome = (battle.predicted_outcome + 0.4).min(1.0);
                battle.mercenary_hired = true;
            }
            deltas.mercenaries_hired += 1;
            events.push(WorldEvent::MercenaryHired {
                quest_id,
                cost: merc_cost,
            });
            ActionResult::Success(format!("Mercenary hired for quest {}", quest_id))
        }

        CampaignAction::CallRescue { battle_id } => {
            // Find a free rescue NPC or pay bribe
            let free_npc = state
                .npc_relationships
                .iter()
                .find(|r| r.rescue_available && r.relationship_score > 50.0);
            let cost = if free_npc.is_some() {
                0.0
            } else {
                state.config.economy.rescue_bribe_cost
            };
            if cost > 0.0 && state.guild.gold < cost {
                return ActionResult::Failed("Not enough gold for rescue".into());
            }
            state.guild.gold -= cost;
            if let Some(battle) = state
                .active_battles
                .iter_mut()
                .find(|b| b.id == battle_id)
            {
                battle.status = BattleStatus::Retreat;
                battle.rescue_called = true;
            }
            deltas.rescues_called += 1;
            events.push(WorldEvent::RescueCalled {
                battle_id,
                cost,
                npc_id: free_npc.map(|r| r.npc_id),
            });
            ActionResult::Success(format!("Rescue called for battle {}", battle_id))
        }

        CampaignAction::HireScout { location_id } => {
            let scout_cost = state.config.economy.scout_cost;
            if state.guild.gold < scout_cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            if let Some(loc) = state
                .overworld
                .locations
                .iter_mut()
                .find(|l| l.id == location_id)
            {
                state.guild.gold -= scout_cost;
                loc.scouted = true;
                events.push(WorldEvent::ScoutReport {
                    location_id,
                    threat_level: loc.threat_level,
                });
                ActionResult::Success(format!("Scout hired for location {}", location_id))
            } else {
                ActionResult::InvalidAction(format!("Location {} not found", location_id))
            }
        }

        CampaignAction::DiplomaticAction {
            faction_id,
            action_type,
        } => {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == faction_id) {
                let old = faction.relationship_to_guild;
                match action_type {
                    DiplomacyActionType::ImproveRelations => {
                        faction.relationship_to_guild =
                            (faction.relationship_to_guild + 5.0).min(100.0);
                    }
                    DiplomacyActionType::TradeAgreement => {
                        faction.relationship_to_guild =
                            (faction.relationship_to_guild + 3.0).min(100.0);
                        state.guild.gold += 10.0;
                    }
                    DiplomacyActionType::RequestAid => {
                        if faction.relationship_to_guild > 30.0 {
                            state.guild.supplies += 20.0;
                            faction.relationship_to_guild -= 5.0;
                        }
                    }
                    DiplomacyActionType::Threaten => {
                        faction.relationship_to_guild -= 15.0;
                        faction.diplomatic_stance = DiplomaticStance::Hostile;
                    }
                }
                events.push(WorldEvent::FactionRelationChanged {
                    faction_id,
                    old,
                    new: faction.relationship_to_guild,
                });
                ActionResult::Success(format!(
                    "{:?} with faction {}",
                    action_type, faction_id
                ))
            } else {
                ActionResult::InvalidAction(format!("Faction {} not found", faction_id))
            }
        }

        CampaignAction::UseAbility { unlock_id, target: _ } => {
            if let Some(unlock) = state.unlocks.iter_mut().find(|u| u.id == unlock_id) {
                if unlock.cooldown_remaining_ms > 0 {
                    return ActionResult::Failed("Ability on cooldown".into());
                }
                if state.guild.gold < unlock.properties.resource_cost {
                    return ActionResult::Failed("Not enough gold".into());
                }
                state.guild.gold -= unlock.properties.resource_cost;
                unlock.cooldown_remaining_ms = unlock.properties.cooldown_ms;
                // TODO: Apply ability effect based on target
                ActionResult::Success(format!("Used ability {}", unlock.name))
            } else {
                ActionResult::InvalidAction(format!("Unlock {} not found", unlock_id))
            }
        }

        CampaignAction::SetSpendPriority { priority: _ } => {
            // Store as guild state for economy system to reference
            ActionResult::Success("Spending priority updated".into())
        }
    }
}

/// Check if the campaign has ended.
fn check_campaign_outcome(state: &CampaignState) -> Option<CampaignOutcome> {
    let alive = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();

    // Defeat: no adventurers left
    if alive == 0 {
        return Some(CampaignOutcome::Defeat);
    }

    // Defeat: bankrupt with no one available
    let available = state
        .adventurers
        .iter()
        .filter(|a| matches!(a.status, AdventurerStatus::Idle | AdventurerStatus::Assigned))
        .count();
    if state.guild.gold < state.config.starting_state.bankrupt_gold_threshold && available == 0 && state.active_quests.is_empty() {
        return Some(CampaignOutcome::Defeat);
    }

    // Defeat: all territory lost (after early game)
    if state.tick > state.config.starting_state.early_game_protection_ticks && !state.overworld.regions.is_empty() {
        let guild_regions = state
            .overworld
            .regions
            .iter()
            .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
            .count();
        if guild_regions == 0 {
            return Some(CampaignOutcome::Defeat);
        }
    }

    // Victory: campaign progress >= 1.0
    if state.overworld.campaign_progress >= 1.0 {
        return Some(CampaignOutcome::Victory);
    }

    None
}
