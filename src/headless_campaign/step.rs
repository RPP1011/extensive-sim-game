//! Campaign step orchestrator.
//!
//! Chains all tick systems in the correct order and returns a
//! `CampaignStepResult` with events, deltas, violations, and outcome.

use super::actions::*;
use super::state::*;
use super::systems;

/// Advance the campaign by one turn (3 seconds of game time).
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

    // --- Handle pre-game phases ---
    if state.phase != CampaignPhase::Playing {
        if state.phase == CampaignPhase::CharacterCreation {
            // Initialize creation on first entry
            if state.player_character.is_none()
                && state.pending_choices.is_empty()
                && state.creation_event_queue.is_empty()
            {
                super::backstory::init_character_creation(state, &mut events);
            }
            // If we just resolved a creation choice, apply PC effects and queue next
            else if let Some(ActionResult::Success(_)) = &action_result {
                super::backstory::apply_creation_effects(state, &mut events);
            }
        }

        deltas.gold_after = state.guild.gold;
        deltas.supplies_after = state.guild.supplies;
        deltas.reputation_after = state.guild.reputation;
        return CampaignStepResult {
            events,
            violations: Vec::new(),
            outcome: None,
            deltas,
            action_result,
            tick: state.tick,
            elapsed_ms: state.elapsed_ms,
        };
    }

    // --- Advance tick ---
    state.tick += 1;
    state.elapsed_ms = state.tick * CAMPAIGN_TURN_SECS as u64 * 1000;

    // --- Auto-resolve expired choices ---
    resolve_expired_choices(state, &mut events);

    // --- Run tick systems in order ---
    // Every-tick systems
    systems::seasons::tick_seasons(state, &mut deltas, &mut events);
    systems::travel::tick_travel(state, &mut deltas, &mut events);
    systems::interception::tick_interception(state, &mut deltas, &mut events);
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
    systems::choices::tick_choices(state, &mut deltas, &mut events);

    // Guild buildings auto-upgrade
    systems::buildings::tick_buildings(state, &mut deltas, &mut events);

    // Random world events (every 500 ticks)
    systems::random_events::tick_random_events(state, &mut deltas, &mut events);

    // Endgame crisis tick
    systems::crisis::tick_crisis(state, &mut deltas, &mut events);

    // Bond system — decay, party bonding
    systems::bonds::tick_bonds(state, &mut deltas, &mut events);

    // Scouting / fog-of-war (every 50 ticks)
    systems::scouting::tick_scouting(state, &mut deltas, &mut events);

    // Quest hooks — check triggers against game state (every 10 turns = 30 seconds)
    if state.tick % 10 == 0 {
        super::quest_hooks::evaluate_hooks(state, &mut events);
    }
    systems::threat::tick_threat(state, &mut deltas, &mut events);
    systems::rival_guild::tick_rival_guild(state, &mut deltas, &mut events);

    // Progression triggers — detect when content should be generated
    systems::progression_triggers::tick_progression_triggers(state, &mut deltas, &mut events);

    // --- New system ticks ---
    systems::diplomacy::tick_diplomacy(state, &mut deltas, &mut events);
    systems::population::tick_population(state, &mut deltas, &mut events);
    systems::caravans::tick_caravans(state, &mut deltas, &mut events);
    systems::retirement::tick_retirement(state, &mut deltas, &mut events);
    systems::mentorship::tick_mentorship(state, &mut deltas, &mut events);
    systems::civil_war::tick_civil_wars(state, &mut deltas, &mut events);
    systems::legendary_deeds::tick_legendary_deeds(state, &mut deltas, &mut events);
    systems::rumors::tick_rumors(state, &mut deltas, &mut events);
    systems::war_exhaustion::tick_war_exhaustion(state, &mut deltas, &mut events);
    systems::guild_tiers::tick_guild_tiers(state, &mut deltas, &mut events);
    systems::espionage::tick_espionage(state, &mut deltas, &mut events);
    systems::mercenaries::tick_mercenaries(state, &mut deltas, &mut events);
    systems::black_market::tick_black_market(state, &mut deltas, &mut events);
    systems::crafting::tick_crafting(state, &mut deltas, &mut events);
    systems::migration::tick_migration(state, &mut deltas, &mut events);
    systems::festivals::tick_festivals(state, &mut deltas, &mut events);
    systems::rivalries::tick_rivalries(state, &mut deltas, &mut events);
    systems::chronicle::tick_chronicle(state, &mut deltas, &mut events);
    systems::nemesis::tick_nemesis(state, &mut deltas, &mut events);
    systems::favors::tick_favors(state, &mut deltas, &mut events);
    systems::site_prep::tick_site_prep(state, &mut deltas, &mut events);
    systems::council::tick_council(state, &mut deltas, &mut events);
    systems::disease::tick_disease(state, &mut deltas, &mut events);
    systems::prisoners::tick_prisoners(state, &mut deltas, &mut events);
    systems::propaganda::tick_propaganda(state, &mut deltas, &mut events);
    systems::monster_ecology::tick_monster_ecology(state, &mut deltas, &mut events);
    systems::visions::tick_visions(state, &mut deltas, &mut events);
    systems::hobbies::tick_hobbies(state, &mut deltas, &mut events);
    systems::loans::tick_loans(state, &mut deltas, &mut events);
    systems::victory_conditions::tick_victory_conditions(state, &mut deltas, &mut events);
    systems::dungeons::tick_dungeons(state, &mut deltas, &mut events);
    systems::npc_reputation::tick_npc_reputation(state, &mut deltas, &mut events);
    systems::weather::tick_weather(state, &mut deltas, &mut events);
    systems::artifacts::tick_artifacts(state, &mut deltas, &mut events);
    systems::difficulty_scaling::tick_difficulty_scaling(state, &mut deltas, &mut events);
    systems::culture::tick_culture(state, &mut deltas, &mut events);

    // --- Second integration systems ---
    systems::archives::tick_archives(state, &mut deltas, &mut events);
    systems::auction::tick_auction(state, &mut deltas, &mut events);
    systems::bounties::tick_bounties(state, &mut deltas, &mut events);
    systems::companions::tick_companions(state, &mut deltas, &mut events);
    systems::contracts::tick_contracts(state, &mut deltas, &mut events);
    systems::corruption::tick_corruption(state, &mut deltas, &mut events);
    systems::counter_espionage::tick_counter_espionage(state, &mut deltas, &mut events);
    systems::economic_competition::tick_economic_competition(state, &mut deltas, &mut events);
    systems::equipment_durability::tick_equipment_durability(state, &mut deltas, &mut events);
    systems::evacuation::tick_evacuations(state, &mut deltas, &mut events);
    systems::exploration::tick_exploration(state, &mut deltas, &mut events);
    systems::faction_tech::tick_faction_tech(state, &mut deltas, &mut events);
    systems::fears::tick_fears(state, &mut deltas, &mut events);
    systems::food::tick_food(state, &mut deltas, &mut events);
    systems::infrastructure::tick_infrastructure(state, &mut deltas, &mut events);
    systems::insurance::tick_insurance(state, &mut deltas, &mut events);
    systems::intel_reports::tick_intel_reports(state, &mut deltas, &mut events);
    systems::intrigue::tick_intrigue(state, &mut deltas, &mut events);
    systems::journals::tick_journals(state, &mut deltas, &mut events);
    // last_stand::check_last_stand is called from tick_battles, not per-tick
    systems::marriages::tick_marriages(state, &mut deltas, &mut events);
    systems::memorials::tick_memorials(state, &mut deltas, &mut events);
    systems::messengers::tick_messengers(state, &mut deltas, &mut events);
    systems::moods::tick_moods(state, &mut deltas, &mut events);
    systems::personal_goals::tick_personal_goals(state, &mut deltas, &mut events);
    systems::quest_chains::tick_quest_chains(state, &mut deltas, &mut events);
    systems::religion::tick_religion(state, &mut deltas, &mut events);
    systems::reputation_decay::tick_reputation_decay(state, &mut deltas, &mut events);
    systems::reputation_stories::tick_reputation_stories(state, &mut deltas, &mut events);
    systems::skill_challenges::tick_skill_challenges(state, &mut deltas, &mut events);
    systems::supply_lines::tick_supply_lines(state, &mut deltas, &mut events);
    systems::terrain_events::tick_terrain_events(state, &mut deltas, &mut events);
    systems::timed_events::tick_timed_events(state, &mut deltas, &mut events);
    systems::trade_goods::tick_trade_goods(state, &mut deltas, &mut events);
    systems::traveling_merchants::tick_traveling_merchants(state, &mut deltas, &mut events);
    systems::treasure_hunts::tick_treasure_hunts(state, &mut deltas, &mut events);
    systems::trophies::tick_trophies(state, &mut deltas, &mut events);
    systems::wanted::tick_wanted(state, &mut deltas, &mut events);

    // --- Batch 3 systems ---
    systems::alliance_blocs::tick_alliance_blocs(state, &mut deltas, &mut events);
    systems::bloodlines::tick_bloodlines(state, &mut deltas, &mut events);
    systems::grudges::tick_grudges(state, &mut deltas, &mut events);
    systems::guild_identity::tick_guild_identity(state, &mut deltas, &mut events);
    systems::guild_rooms::tick_guild_rooms(state, &mut deltas, &mut events);
    systems::leadership::tick_leadership(state, &mut deltas, &mut events);
    systems::nicknames::tick_nicknames(state, &mut deltas, &mut events);
    systems::oaths::tick_oaths(state, &mut deltas, &mut events);
    systems::romance::tick_romance(state, &mut deltas, &mut events);
    systems::seasonal_quests::tick_seasonal_quests(state, &mut deltas, &mut events);
    systems::smuggling::tick_smuggling(state, &mut deltas, &mut events);
    systems::vassalage::tick_vassalage(state, &mut deltas, &mut events);

    // --- Batch 4 systems ---
    systems::awakening::tick_awakening(state, &mut deltas, &mut events);
    systems::charter::tick_charter(state, &mut deltas, &mut events);
    systems::folk_hero::tick_folk_hero(state, &mut deltas, &mut events);
    systems::geography::tick_geography(state, &mut deltas, &mut events);
    systems::legacy_weapons::tick_legacy_weapons(state, &mut deltas, &mut events);
    systems::secrets::tick_secrets(state, &mut deltas, &mut events);

    // --- Batch 5 systems ---
    systems::commodity_futures::tick_commodity_futures(state, &mut deltas, &mut events);
    systems::coup_engine::tick_coup_engine(state, &mut deltas, &mut events);
    systems::divine_favor::tick_divine_favor(state, &mut deltas, &mut events);
    systems::wound_persistence::tick_wound_persistence(state, &mut deltas, &mut events);
    systems::plague_vectors::tick_plague_vectors(state, &mut deltas, &mut events);
    systems::price_controls::tick_price_controls(state, &mut deltas, &mut events);
    systems::defection_cascade::tick_defection_cascade(state, &mut deltas, &mut events);
    systems::heist_planning::tick_heist_planning(state, &mut deltas, &mut events);
    systems::contract_negotiation::tick_contract_negotiation(state, &mut deltas, &mut events);
    systems::escalation_protocol::tick_escalation_protocol(state, &mut deltas, &mut events);
    systems::dead_zones::tick_dead_zones(state, &mut deltas, &mut events);
    systems::addiction::tick_addiction(state, &mut deltas, &mut events);
    systems::party_chemistry::tick_party_chemistry(state, &mut deltas, &mut events);
    systems::threat_clock::tick_threat_clock(state, &mut deltas, &mut events);
    systems::bankruptcy_cascade::tick_bankruptcy_cascade(state, &mut deltas, &mut events);
    systems::signal_towers::tick_signal_towers(state, &mut deltas, &mut events);
    systems::currency_debasement::tick_currency_debasement(state, &mut deltas, &mut events);
    systems::demonic_pacts::tick_demonic_pacts(state, &mut deltas, &mut events);
    systems::class_system::tick_class_system(state, &mut deltas, &mut events);

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

        CampaignAction::Rest => {
            if state.resting {
                return ActionResult::Failed("Already resting".into());
            }
            state.resting = true;
            state.ticks_since_rest = 0;

            // Apply recovery to all idle/injured adventurers
            for adv in &mut state.adventurers {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                adv.stress = (adv.stress - 15.0).max(0.0);
                adv.fatigue = (adv.fatigue - 20.0).max(0.0);
                adv.morale = (adv.morale + 10.0).min(100.0);
                if adv.status == AdventurerStatus::Injured {
                    adv.injury = (adv.injury - 10.0).max(0.0);
                }
            }

            // Present pending progression as choice events
            let pending = std::mem::take(&mut state.pending_progression);
            for prog in &pending {
                let choice_id = state.next_event_id;
                state.next_event_id += 1;

                let choice = ChoiceEvent {
                    id: choice_id,
                    source: ChoiceSource::ProgressionUnlock,
                    prompt: prog.description.clone(),
                    options: vec![
                        ChoiceOption {
                            label: "Accept".into(),
                            description: format!("Apply: {}", prog.content.chars().take(100).collect::<String>()),
                            effects: vec![
                                ChoiceEffect::Narrative(format!("Progression applied: {}", prog.description)),
                            ],
                        },
                        ChoiceOption {
                            label: "Decline".into(),
                            description: "Pass on this opportunity.".into(),
                            effects: vec![
                                ChoiceEffect::Narrative("The opportunity fades.".into()),
                            ],
                        },
                    ],
                    default_option: 1,
                    deadline_ms: None, // No deadline — player decides during rest
                    created_at_ms: state.elapsed_ms,
                };

                events.push(WorldEvent::ChoicePresented {
                    choice_id,
                    prompt: choice.prompt.clone(),
                    num_options: 2,
                });

                state.pending_choices.push(choice);
            }

            let n = pending.len();
            state.resting = false; // Rest completes immediately for now

            events.push(WorldEvent::CampaignMilestone {
                description: format!("The guild rests. {} progression items available.", n),
            });

            ActionResult::Success(format!("Rested. {} progression choices pending.", n))
        }

        CampaignAction::AcceptQuest { request_id } => {
            if state.active_quests.len() >= state.guild.active_quest_capacity {
                return ActionResult::Failed("Quest capacity reached".into());
            }
            let idx = state.request_board.iter().position(|r| r.id == request_id);
            match idx {
                Some(i) => {
                    let request = state.request_board.remove(i);
                    let quest_id = request.id;
                    let is_combat = matches!(
                        request.quest_type,
                        QuestType::Combat | QuestType::Rescue
                    );
                    let threat = request.threat_level;

                    state.active_quests.push(ActiveQuest {
                        id: quest_id,
                        request,
                        status: ActiveQuestStatus::Preparing,
                        assigned_pool: Vec::new(),
                        dispatched_party_id: None,
                        elapsed_ms: 0,
                        events: Vec::new(),
                    });

                    // Generate quest branch choice for combat/rescue quests
                    if is_combat {
                        generate_quest_branch_choice(state, quest_id, threat, events);
                    }

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
                food_level: 100.0,
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
            let base_cost = amount * state.config.economy.supply_cost_per_unit;
            let cost = systems::economy::effective_cost(
                base_cost,
                state.guild.market_prices.supply_multiplier,
            );
            if state.guild.gold < cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
                party.supply_level = (party.supply_level + amount).min(100.0);
                // Deduct and record after releasing the party borrow
                drop(party);
                state.guild.gold = (state.guild.gold - cost).max(0.0);
                state.guild.purchase_history.supply_purchases += 1.0;
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
            let training_cost = systems::economy::effective_cost(
                state.config.economy.training_cost,
                state.guild.market_prices.training_multiplier,
            );
            if state.guild.gold < training_cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            // Check status before borrowing mutably
            let adv_status = state.adventurers.iter().find(|a| a.id == adventurer_id).map(|a| a.status);
            match adv_status {
                Some(AdventurerStatus::Idle) => {}
                Some(_) => return ActionResult::Failed("Adventurer not idle".into()),
                None => return ActionResult::InvalidAction(format!("Adventurer {} not found", adventurer_id)),
            }

            state.guild.gold = (state.guild.gold - training_cost).max(0.0);
            state.guild.purchase_history.training_purchases += 1.0;

            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == adventurer_id)
            {
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
                state.guild.gold = (state.guild.gold - runner_cost).max(0.0);
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
            let merc_cost = systems::economy::effective_cost(
                state.config.economy.mercenary_cost,
                state.guild.market_prices.mercenary_multiplier,
            );
            if state.guild.gold < merc_cost {
                return ActionResult::Failed("Not enough gold".into());
            }
            state.guild.gold = (state.guild.gold - merc_cost).max(0.0);
            state.guild.purchase_history.mercenary_purchases += 1.0;
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
            state.guild.gold = (state.guild.gold - cost).max(0.0);
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
                state.guild.gold = (state.guild.gold - scout_cost).max(0.0);
                loc.scouted = true;
                let faction_owner = loc.faction_owner;
                events.push(WorldEvent::ScoutReport {
                    location_id,
                    threat_level: loc.threat_level,
                });

                // Boost visibility of the region owned by this location's faction.
                if let Some(fid) = faction_owner {
                    if let Some(rid) = state
                        .overworld
                        .regions
                        .iter()
                        .find(|r| r.owner_faction_id == fid)
                        .map(|r| r.id)
                    {
                        systems::scouting::boost_region_visibility(
                            state,
                            rid,
                            systems::scouting::HIRE_SCOUT_BOOST,
                            events,
                        );
                    }
                }

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
                        if faction.diplomatic_stance != DiplomaticStance::AtWar {
                            faction.diplomatic_stance = DiplomaticStance::Hostile;
                        }
                    }
                    DiplomacyActionType::ProposeCeasefire => {
                        if faction.diplomatic_stance == DiplomaticStance::AtWar {
                            // Ceasefire costs gold and reputation
                            state.guild.gold = (state.guild.gold - 30.0).max(0.0);
                            faction.diplomatic_stance = DiplomaticStance::Hostile;
                            faction.at_war_with.retain(|&id| id != state.diplomacy.guild_faction_id);
                            faction.relationship_to_guild =
                                (faction.relationship_to_guild + 10.0).min(0.0); // Cap at 0
                        }
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

        CampaignAction::ProposeCoalition { faction_id } => {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == faction_id) {
                if faction.relationship_to_guild < 60.0 {
                    return ActionResult::Failed("Relationship too low for coalition".into());
                }
                if faction.coalition_member {
                    return ActionResult::Failed("Already in coalition".into());
                }
                faction.coalition_member = true;
                faction.diplomatic_stance = DiplomaticStance::Coalition;
                let old = faction.relationship_to_guild;
                faction.relationship_to_guild = (faction.relationship_to_guild + 15.0).min(100.0);
                events.push(WorldEvent::FactionRelationChanged {
                    faction_id,
                    old,
                    new: faction.relationship_to_guild,
                });
                events.push(WorldEvent::CampaignMilestone {
                    description: format!("{} joined the coalition!", faction.name),
                });
                ActionResult::Success(format!("Coalition formed with faction {}", faction_id))
            } else {
                ActionResult::InvalidAction(format!("Faction {} not found", faction_id))
            }
        }

        CampaignAction::RequestCoalitionAid { faction_id } => {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == faction_id) {
                if !faction.coalition_member {
                    return ActionResult::Failed("Not a coalition member".into());
                }
                if faction.military_strength < 20.0 {
                    return ActionResult::Failed("Coalition member too weak to help".into());
                }
                // Coalition member sends adventurers and supplies
                let aid_strength = faction.military_strength * 0.1;
                faction.military_strength -= aid_strength;
                state.guild.supplies += 15.0;
                // Boost control of lowest-control guild region
                if let Some(region) = state
                    .overworld
                    .regions
                    .iter_mut()
                    .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
                    .min_by(|a, b| a.control.partial_cmp(&b.control).unwrap_or(std::cmp::Ordering::Equal))
                {
                    region.control = (region.control + aid_strength).min(100.0);
                }
                events.push(WorldEvent::SupplyChanged {
                    amount: 15.0,
                    reason: format!("Coalition aid from faction {}", faction_id),
                });
                ActionResult::Success(format!("Coalition aid received from faction {}", faction_id))
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
                state.guild.gold = (state.guild.gold - unlock.properties.resource_cost).max(0.0);
                unlock.cooldown_remaining_ms = unlock.properties.cooldown_ms;
                // TODO: Apply ability effect based on target
                ActionResult::Success(format!("Used ability {}", unlock.name))
            } else {
                ActionResult::InvalidAction(format!("Unlock {} not found", unlock_id))
            }
        }

        CampaignAction::SetSpendPriority { priority } => {
            // Store as guild state for economy system to reference
            state.guild.spend_priority = priority;
            ActionResult::Success(format!("Spending priority set to {:?}", priority))
        }

        CampaignAction::RespondToChoice { choice_id, option_index } => {
            // Validate before removing
            let valid = state.pending_choices.iter().any(|c| {
                c.id == choice_id && option_index < c.options.len()
            });
            if !valid {
                let exists = state.pending_choices.iter().any(|c| c.id == choice_id);
                if !exists {
                    return ActionResult::InvalidAction(format!(
                        "Choice {} not found", choice_id
                    ));
                }
                return ActionResult::InvalidAction(format!(
                    "Option index {} out of range", option_index
                ));
            }

            let idx = state.pending_choices.iter().position(|c| c.id == choice_id).unwrap();
            let choice = state.pending_choices.remove(idx);
            let selected = &choice.options[option_index];
            let label = selected.label.clone();
            let effects = selected.effects.clone();

            apply_choice_effects(state, &effects, events);

            events.push(WorldEvent::ChoiceResolved {
                choice_id,
                option_index,
                label: label.clone(),
                was_default: false,
            });

            ActionResult::Success(format!("Chose: {}", label))
        }

        CampaignAction::UseClassSkill {
            adventurer_id,
            skill_name,
            target: _,
        } => {
            // Find the skill effect on the adventurer's class skills.
            let skill_data = state
                .adventurers
                .iter()
                .find(|a| a.id == adventurer_id)
                .and_then(|adv| {
                    adv.classes
                        .iter()
                        .flat_map(|c| &c.skills_granted)
                        .find(|s| s.skill_name == skill_name)
                })
                .and_then(|s| s.skill_effect.clone());

            if let Some(effect) = skill_data {
                let desc = super::skill_effects::apply_skill_effect(
                    state, &effect, adventurer_id, events,
                );
                ActionResult::Success(desc)
            } else {
                ActionResult::InvalidAction(format!(
                    "Skill '{}' not found or has no effect on adventurer {}",
                    skill_name, adventurer_id
                ))
            }
        }

        CampaignAction::InterceptChampion {
            party_id,
            champion_id,
        } => {
            // Find the champion's party
            let champ_party = state
                .parties
                .iter()
                .find(|p| {
                    p.member_ids.len() == 1
                        && p.member_ids[0] == champion_id
                        && p.status == PartyStatus::Traveling
                });
            let champ_pos = match champ_party {
                Some(p) => p.position,
                None => {
                    return ActionResult::Failed(format!(
                        "Champion {} is not traveling on the map",
                        champion_id
                    ));
                }
            };

            if party_id == 0 {
                // Form a new party from idle guild adventurers
                let idle_ids: Vec<u32> = state
                    .adventurers
                    .iter()
                    .filter(|a| {
                        a.status == AdventurerStatus::Idle
                            && a.faction_id.is_none()
                            && a.party_id.is_none()
                    })
                    .map(|a| a.id)
                    .collect();

                if idle_ids.is_empty() {
                    return ActionResult::Failed("No idle adventurers to form party".into());
                }

                // Take up to 4 idle adventurers
                let member_ids: Vec<u32> = idle_ids.into_iter().take(4).collect();
                let new_party_id = state.next_party_id;
                state.next_party_id += 1;

                for &mid in &member_ids {
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                        adv.status = AdventurerStatus::Traveling;
                        adv.party_id = Some(new_party_id);
                    }
                }

                let party = Party {
                    id: new_party_id,
                    member_ids: member_ids.clone(),
                    position: state.guild.base.position,
                    destination: Some(champ_pos),
                    speed: state.config.quest_lifecycle.party_speed,
                    status: PartyStatus::Traveling,
                    supply_level: state.config.quest_lifecycle.party_starting_supply,
                    morale: state.config.quest_lifecycle.party_starting_morale,
                    quest_id: None,
                food_level: 100.0,
                };
                state.parties.push(party);

                events.push(WorldEvent::PartyFormed {
                    party_id: new_party_id,
                    member_ids,
                });

                let champ_name = state
                    .adventurers
                    .iter()
                    .find(|a| a.id == champion_id)
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| format!("Champion {}", champion_id));

                ActionResult::Success(format!(
                    "Party dispatched to intercept {}",
                    champ_name
                ))
            } else {
                // Redirect an existing party to the champion's position
                if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
                    party.destination = Some(champ_pos);
                    party.status = PartyStatus::Traveling;

                    let champ_name = state
                        .adventurers
                        .iter()
                        .find(|a| a.id == champion_id)
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| format!("Champion {}", champion_id));

                    ActionResult::Success(format!(
                        "Party {} redirected to intercept {}",
                        party_id, champ_name
                    ))
                } else {
                    ActionResult::InvalidAction(format!("Party {} not found", party_id))
                }
            }
        }

        CampaignAction::ChooseStartingPackage { choice } => {
            if state.phase == CampaignPhase::Playing {
                return ActionResult::Failed("Campaign already initialized".into());
            }

            // Apply the starting package
            state.guild.gold += choice.gold_bonus;
            state.guild.supplies += choice.supply_bonus;
            state.guild.reputation = (state.guild.reputation + choice.reputation_bonus).clamp(0.0, 100.0);

            // Add adventurers with corrected IDs
            let base_id = state.adventurers.iter().map(|a| a.id).max().unwrap_or(0) + 1;
            for (i, mut adv) in choice.adventurers.into_iter().enumerate() {
                adv.id = base_id + i as u32;
                state.adventurers.push(adv);
            }

            // Add starting items to inventory
            for item in choice.items {
                state.guild.inventory.push(item);
            }

            state.phase = CampaignPhase::Playing;
            state.available_starting_choices.clear();

            events.push(WorldEvent::CampaignMilestone {
                description: format!("Starting package chosen: {}", choice.name),
            });

            ActionResult::Success(format!("Campaign started with: {}", choice.name))
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
        eprintln!("[DEFEAT] tick {}: all adventurers dead", state.tick);
        return Some(CampaignOutcome::Defeat);
    }

    // Defeat: bankrupt with no one available
    let available = state
        .adventurers
        .iter()
        .filter(|a| matches!(a.status, AdventurerStatus::Idle | AdventurerStatus::Assigned))
        .count();
    if state.guild.gold < state.config.starting_state.bankrupt_gold_threshold && available == 0 && state.active_quests.is_empty() {
        eprintln!("[DEFEAT] tick {}: bankrupt (gold={:.0}, available={}, quests={})",
            state.tick, state.guild.gold, available, state.active_quests.len());
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
            eprintln!("[DEFEAT] tick {}: all territory lost", state.tick);
            return Some(CampaignOutcome::Defeat);
        }
    }

    // Victory: campaign progress >= 1.0
    if state.overworld.campaign_progress >= 1.0 {
        return Some(CampaignOutcome::Victory);
    }

    None
}

/// Apply the effects of a selected choice option.
fn apply_choice_effects(
    state: &mut CampaignState,
    effects: &[ChoiceEffect],
    events: &mut Vec<WorldEvent>,
) {
    for effect in effects {
        match effect {
            ChoiceEffect::Gold(amount) => {
                state.guild.gold += amount;
                events.push(WorldEvent::GoldChanged {
                    amount: *amount,
                    reason: "Choice effect".into(),
                });
            }
            ChoiceEffect::Supplies(amount) => {
                state.guild.supplies += amount;
                events.push(WorldEvent::SupplyChanged {
                    amount: *amount,
                    reason: "Choice effect".into(),
                });
            }
            ChoiceEffect::Reputation(delta) => {
                state.guild.reputation = (state.guild.reputation + delta).clamp(0.0, 100.0);
            }
            ChoiceEffect::FactionRelation { faction_id, delta } => {
                if let Some(f) = state.factions.iter_mut().find(|f| f.id == *faction_id) {
                    let old = f.relationship_to_guild;
                    f.relationship_to_guild = (f.relationship_to_guild + delta).clamp(-100.0, 100.0);
                    events.push(WorldEvent::FactionRelationChanged {
                        faction_id: *faction_id,
                        old,
                        new: f.relationship_to_guild,
                    });
                }
            }
            ChoiceEffect::GrantUnlock(unlock) => {
                events.push(WorldEvent::ProgressionUnlocked {
                    unlock_id: unlock.id,
                    category: unlock.category,
                    name: unlock.name.clone(),
                });
                state.unlocks.push(unlock.clone());
            }
            ChoiceEffect::ModifyQuestThreat { quest_id, multiplier } => {
                if let Some(q) = state.active_quests.iter_mut().find(|q| q.id == *quest_id) {
                    q.request.threat_level *= multiplier;
                }
            }
            ChoiceEffect::ModifyQuestReward { quest_id, gold_bonus, rep_bonus } => {
                if let Some(q) = state.active_quests.iter_mut().find(|q| q.id == *quest_id) {
                    q.request.reward.gold += gold_bonus;
                    q.request.reward.reputation += rep_bonus;
                }
            }
            ChoiceEffect::AddAdventurer(adv) => {
                let mut adv = adv.clone();
                adv.id = state.adventurers.iter().map(|a| a.id).max().unwrap_or(0) + 1;
                state.adventurers.push(adv);
            }
            ChoiceEffect::ModifyAdventurerInjury { adventurer_id, delta } => {
                if let Some(a) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    a.injury = (a.injury + delta).clamp(0.0, 100.0);
                }
            }
            ChoiceEffect::ModifyAdventurerLoyalty { adventurer_id, delta } => {
                if let Some(a) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    a.loyalty = (a.loyalty + delta).clamp(0.0, 100.0);
                }
            }
            ChoiceEffect::AddItem(item) => {
                state.guild.inventory.push(item.clone());
            }
            ChoiceEffect::SetQuestStatus { quest_id, status } => {
                if let Some(q) = state.active_quests.iter_mut().find(|q| q.id == *quest_id) {
                    q.status = *status;
                }
            }
            ChoiceEffect::AttendFestival(festival_id) => {
                // Mark the festival as attended
                if let Some(f) = state.active_festivals.iter_mut().find(|f| f.id == *festival_id) {
                    f.attended = true;
                }
            }
            ChoiceEffect::Narrative(text) => {
                let id = state.next_event_id;
                state.next_event_id += 1;
                state.event_log.push(CampaignEvent {
                    id,
                    tick: state.tick,
                    description: text.clone(),
                });
            }
            ChoiceEffect::BeginEvacuation { source_region_id, destination_region_id } => {
                let evac_id = state.next_evacuation_id;
                state.next_evacuation_id += 1;
                state.evacuations.push(Evacuation {
                    id: evac_id,
                    source_region_id: *source_region_id,
                    destination_region_id: *destination_region_id,
                    evacuees: 0,
                    supplies_saved: 0.0,
                    started_tick: state.tick,
                    completed: false,
                    cost: 0.0,
                });
            }
            ChoiceEffect::NoEvacuationPenalty { region_id } => {
                // No evacuation chosen — penalty handled by evacuation system
                if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == *region_id) {
                    region.civilian_morale = (region.civilian_morale - 20.0).max(0.0);
                }
            }
        }
    }
}

/// Generate a quest branch choice when a combat/rescue quest is accepted.
fn generate_quest_branch_choice(
    state: &mut CampaignState,
    quest_id: u32,
    threat: f32,
    events: &mut Vec<WorldEvent>,
) {
    use crate::headless_campaign::choice_templates::{
        get_or_load_templates, instantiate_template, TemplateContext,
    };

    let templates = get_or_load_templates();
    let quest_templates = templates.by_trigger("quest_preparing_combat");
    if quest_templates.is_empty() {
        return;
    }

    let template_idx = (lcg_next(&mut state.rng) as usize) % quest_templates.len();
    let template = quest_templates[template_idx];

    let quest_type_str = state
        .active_quests
        .iter()
        .find(|q| q.id == quest_id)
        .map(|q| format!("{:?}", q.request.quest_type))
        .unwrap_or_else(|| "Combat".into());

    let mut ctx = TemplateContext::new();
    ctx.insert("quest_type".into(), quest_type_str);
    ctx.insert("threat".into(), format!("{:.0}", threat));
    ctx.insert("quest_id".into(), quest_id.to_string());

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = instantiate_template(template, &ctx, choice_id, state.elapsed_ms);

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}

/// Auto-resolve expired choice events by selecting the default option.
fn resolve_expired_choices(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    let now = state.elapsed_ms;
    let mut expired = Vec::new();

    for (i, choice) in state.pending_choices.iter().enumerate() {
        if let Some(deadline) = choice.deadline_ms {
            if now >= deadline {
                expired.push(i);
            }
        }
    }

    // Process in reverse to avoid index invalidation
    expired.reverse();
    for idx in expired {
        let choice = state.pending_choices.remove(idx);
        let default_idx = choice.default_option.min(choice.options.len().saturating_sub(1));
        let selected = &choice.options[default_idx];

        apply_choice_effects(state, &selected.effects, events);

        events.push(WorldEvent::ChoiceResolved {
            choice_id: choice.id,
            option_index: default_idx,
            label: selected.label.clone(),
            was_default: true,
        });
    }
}
