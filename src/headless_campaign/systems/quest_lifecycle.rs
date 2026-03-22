//! Quest lifecycle — every tick.
//!
//! Progresses quest status based on party and battle state.
//! Handles transitions: Dispatched → InProgress → InCombat → Returning → complete.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::combat_oracle::HeuristicOracle;
use crate::headless_campaign::combat_oracle::CombatOracle;
use crate::headless_campaign::state::*;

pub fn tick_quest_lifecycle(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let tick = state.tick;
    let elapsed = state.elapsed_ms;

    // Collect indices to process (avoid borrow conflicts)
    let quest_count = state.active_quests.len();
    let mut completed_indices = Vec::new();
    let mut new_battles = Vec::new();

    for i in 0..quest_count {
        let quest = &state.active_quests[i];

        match quest.status {
            ActiveQuestStatus::Dispatched => {
                // Check if party has arrived at quest location
                if let Some(pid) = quest.dispatched_party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        if party.status == PartyStatus::OnMission {
                            // Party arrived — transition to InProgress
                            let quest = &mut state.active_quests[i];
                            quest.status = ActiveQuestStatus::InProgress;
                            quest.events.push(QuestEvent {
                                tick,
                                description: "Party arrived at quest location".into(),
                            });
                        }
                    }
                }
            }
            ActiveQuestStatus::InProgress => {
                // After some time at location, combat quests trigger a battle
                if matches!(
                    quest.request.quest_type,
                    QuestType::Combat | QuestType::Rescue
                ) {
                    // Trigger battle after 20 ticks (~2s) at location
                    let time_at_location = elapsed.saturating_sub(quest.request.arrived_at_ms);
                    if time_at_location > 2000 && quest.dispatched_party_id.is_some() {
                        let battle_id = state.next_battle_id;
                        state.next_battle_id += 1;

                        let party_id = quest.dispatched_party_id.unwrap();

                        // Use combat oracle for outcome prediction
                        let oracle = HeuristicOracle;
                        let members: Vec<&Adventurer> = state
                            .adventurers
                            .iter()
                            .filter(|a| a.party_id == Some(party_id))
                            .collect();
                        let oracle_result = oracle.predict(
                            &members,
                            quest.request.threat_level,
                            quest.request.threat_level,
                        );
                        let party_health = oracle_result.expected_hp_remaining;
                        // Map win probability to predicted_outcome range (-1 to 1)
                        let predicted = oracle_result.victory_probability * 2.0 - 1.0;

                        new_battles.push((
                            i,
                            BattleState {
                                id: battle_id,
                                quest_id: quest.id,
                                party_id,
                                location: quest.request.target_position,
                                party_health_ratio: party_health,
                                enemy_health_ratio: 1.0,
                                enemy_strength: quest.request.threat_level,
                                elapsed_ticks: 0,
                                predicted_outcome: predicted,
                                status: BattleStatus::Active,
                                runner_sent: false,
                                mercenary_hired: false,
                                rescue_called: false,
                            },
                        ));
                    }
                } else {
                    // Non-combat quests complete after a duration proportional to threat
                    let duration = (quest.request.threat_level * 500.0) as u64; // ms
                    if quest.elapsed_ms > duration {
                        completed_indices.push((i, QuestResult::Victory));
                    }
                }
            }
            ActiveQuestStatus::InCombat => {
                // Check if the associated battle has resolved
                if let Some(battle) = state
                    .active_battles
                    .iter()
                    .find(|b| b.quest_id == quest.id)
                {
                    match battle.status {
                        BattleStatus::Victory => {
                            completed_indices.push((i, QuestResult::Victory));
                        }
                        BattleStatus::Defeat => {
                            completed_indices.push((i, QuestResult::Defeat));
                        }
                        BattleStatus::Retreat => {
                            completed_indices.push((i, QuestResult::Abandoned));
                        }
                        BattleStatus::Active => {} // Still fighting
                    }
                }
            }
            ActiveQuestStatus::Returning => {
                // Check if party has returned to base
                if let Some(pid) = quest.dispatched_party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        if party.status == PartyStatus::Idle {
                            completed_indices.push((i, QuestResult::Victory));
                        }
                    }
                }
            }
            _ => {}
        }

        // Accumulate elapsed time
        state.active_quests[i].elapsed_ms += CAMPAIGN_TICK_MS as u64;
    }

    // Create new battles
    for (quest_idx, battle) in new_battles {
        let quest = &mut state.active_quests[quest_idx];
        quest.status = ActiveQuestStatus::InCombat;
        quest.events.push(QuestEvent {
            tick,
            description: "Battle started!".into(),
        });

        // Set party to fighting
        if let Some(party) = state
            .parties
            .iter_mut()
            .find(|p| p.id == battle.party_id)
        {
            party.status = PartyStatus::Fighting;
        }

        // Set adventurers to fighting
        for adv in &mut state.adventurers {
            if adv.party_id == Some(battle.party_id) {
                adv.status = AdventurerStatus::Fighting;
            }
        }

        events.push(WorldEvent::BattleStarted {
            battle_id: battle.id,
            quest_id: battle.quest_id,
            party_health: battle.party_health_ratio,
            enemy_strength: battle.enemy_strength,
        });
        deltas.battles_started += 1;

        state.active_battles.push(battle);
    }

    // Complete quests (process in reverse to avoid index invalidation)
    completed_indices.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, result) in completed_indices {
        let quest = state.active_quests.remove(idx);
        let party_id = quest.dispatched_party_id.unwrap_or(0);
        let threat = quest.request.threat_level;

        // Find the associated battle (if any) for consequence application
        let battle_hp = state
            .active_battles
            .iter()
            .find(|b| b.quest_id == quest.id)
            .map(|b| b.party_health_ratio)
            .unwrap_or(1.0);

        // Apply consequences to adventurers
        let mut casualties = 0u32;
        let member_ids: Vec<u32> = state
            .adventurers
            .iter()
            .filter(|a| a.party_id == Some(party_id))
            .map(|a| a.id)
            .collect();

        for &mid in &member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                match result {
                    QuestResult::Victory => {
                        // Victory: small injuries, XP gain, stress relief
                        let damage_taken = (1.0 - battle_hp) * 30.0;
                        adv.injury = (adv.injury + damage_taken).min(100.0);
                        adv.stress = (adv.stress - 5.0).max(0.0);
                        adv.loyalty = (adv.loyalty + 3.0).min(100.0);
                        adv.morale = (adv.morale + 5.0).min(100.0);
                        adv.xp += 20 + (threat * 0.5) as u32;

                        // Level up check
                        let threshold = adv.level * adv.level * 50;
                        if adv.xp >= threshold {
                            adv.level += 1;
                            adv.stats.max_hp += 5.0;
                            adv.stats.attack += 2.0;
                            adv.stats.defense += 1.5;
                            events.push(WorldEvent::AdventurerLevelUp {
                                adventurer_id: mid,
                                new_level: adv.level,
                            });
                        }
                    }
                    QuestResult::Defeat => {
                        // Defeat: heavy injuries, possible death
                        let severity = threat / 50.0; // 0-2 range
                        adv.injury = (adv.injury + 25.0 + severity * 15.0).min(100.0);
                        adv.stress = (adv.stress + 15.0).min(100.0);
                        adv.loyalty = (adv.loyalty - 5.0).max(0.0);
                        adv.morale = (adv.morale - 15.0).max(0.0);
                        adv.xp += 5;

                        // Death check: high injury + bad luck
                        if adv.injury > 90.0 {
                            let death_roll = lcg_f32(&mut state.rng);
                            if death_roll < 0.3 { // 30% death chance when severely injured
                                adv.status = AdventurerStatus::Dead;
                                casualties += 1;
                                events.push(WorldEvent::AdventurerDied {
                                    adventurer_id: mid,
                                    cause: format!("Killed in battle (threat {})", threat as u32),
                                });
                                continue;
                            }
                        }

                        // Incapacitation
                        if adv.injury > 65.0 {
                            adv.status = AdventurerStatus::Injured;
                            events.push(WorldEvent::AdventurerInjured {
                                adventurer_id: mid,
                                injury_level: adv.injury,
                            });
                        }
                    }
                    QuestResult::Abandoned => {
                        adv.stress = (adv.stress + 8.0).min(100.0);
                        adv.morale = (adv.morale - 10.0).max(0.0);
                    }
                }

                // Return adventurer to idle (if not dead/injured)
                if adv.status == AdventurerStatus::Fighting
                    || adv.status == AdventurerStatus::OnMission
                    || adv.status == AdventurerStatus::Traveling
                {
                    adv.status = AdventurerStatus::Idle;
                    adv.party_id = None;
                }
            }
        }

        // Send party home (or disband if empty)
        if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
            // Remove dead members
            party.member_ids.retain(|id| {
                state.adventurers.iter().any(|a| a.id == *id && a.status != AdventurerStatus::Dead)
            });
            if party.member_ids.is_empty() {
                party.status = PartyStatus::Idle; // will be cleaned up
            } else {
                party.status = PartyStatus::Returning;
                party.destination = Some(state.guild.base.position);
            }
        }

        // Apply rewards on victory
        let reward = if result == QuestResult::Victory {
            quest.request.reward.clone()
        } else {
            QuestReward::default()
        };

        events.push(WorldEvent::QuestCompleted {
            quest_id: quest.id,
            result,
        });

        if result == QuestResult::Victory {
            deltas.quests_completed += 1;
        } else {
            deltas.quests_failed += 1;
        }

        state.completed_quests.push(CompletedQuest {
            id: quest.id,
            quest_type: quest.request.quest_type,
            result,
            reward_applied: reward,
            completed_at_ms: elapsed,
            party_id,
            casualties,
        });

        // Remove associated battle
        state.active_battles.retain(|b| b.quest_id != quest.id);
    }
}
