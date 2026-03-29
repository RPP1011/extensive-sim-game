//! Quest lifecycle — every tick.
//!
//! Progresses quest status based on party and battle state.
//! Handles transitions: Dispatched → InProgress → InCombat → Returning → complete.

use crate::actions::{StepDeltas, WorldEvent};
use crate::combat_oracle::HeuristicOracle;
use crate::state::*;
use super::class_system::effective_noncombat_stats;
use super::loot::process_quest_loot;

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
                    // Trigger battle after configured delay at location
                    let time_at_location = elapsed.saturating_sub(quest.request.arrived_at_ms);
                    if time_at_location > state.config.quest_lifecycle.battle_trigger_delay_ms && quest.dispatched_party_id.is_some() {
                        let battle_id = state.next_battle_id;
                        state.next_battle_id += 1;

                        let party_id = quest.dispatched_party_id.unwrap();

                        // Use combat oracle for outcome prediction.
                        // Factoring in unlock power makes generated abilities
                        // affect even the sigmoid oracle's predictions.
                        let oracle = HeuristicOracle;
                        let members: Vec<&Adventurer> = state
                            .adventurers
                            .iter()
                            .filter(|a| a.party_id == Some(party_id))
                            .collect();
                        let oracle_result = oracle.predict_with_unlocks_and_bonds(
                            &members,
                            quest.request.threat_level,
                            quest.request.threat_level,
                            &state.unlocks,
                            &state.adventurer_bonds,
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
                    // Non-combat quests complete after a duration proportional to threat.
                    // Party's relevant non-combat stats reduce effective duration.
                    let base_duration = quest.request.threat_level
                        * state.config.quest_lifecycle.non_combat_duration_multiplier;

                    // Sum relevant stat from party members based on quest type
                    let party_bonus: f32 = if let Some(pid) = quest.dispatched_party_id {
                        state.adventurers.iter()
                            .filter(|a| a.party_id == Some(pid) && a.status != AdventurerStatus::Dead)
                            .map(|a| {
                                let (dip, com, _cra, _med, sch, ste, _lea) = effective_noncombat_stats(a);
                                match quest.request.quest_type {
                                    QuestType::Diplomatic => dip,
                                    QuestType::Gather     => com,
                                    QuestType::Exploration => ste + sch * 0.5,
                                    QuestType::Escort     => dip * 0.5 + com * 0.5,
                                    _ => 0.0,
                                }
                            })
                            .sum()
                    } else {
                        0.0
                    };

                    // Each point of relevant stat reduces duration by 2% (capped at 50% reduction)
                    let speed_factor = (1.0 - party_bonus * 0.02).clamp(0.5, 1.0);
                    let duration = (base_duration * speed_factor) as u64;
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
        state.active_quests[i].elapsed_ms += CAMPAIGN_TURN_SECS as u64 * 1000;
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

        let qlcfg = &state.config.quest_lifecycle;
        for &mid in &member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                match result {
                    QuestResult::Victory => {
                        // Victory: small injuries, XP gain, stress relief
                        let damage_taken = (1.0 - battle_hp) * qlcfg.victory_injury_scaling;
                        adv.injury = (adv.injury + damage_taken).clamp(0.0, 100.0);
                        adv.stress = (adv.stress - qlcfg.victory_stress_relief).max(0.0);
                        adv.loyalty = (adv.loyalty + qlcfg.victory_loyalty_gain).min(100.0);
                        adv.morale = (adv.morale + qlcfg.victory_morale_gain).min(100.0);
                        adv.xp += qlcfg.victory_base_xp + (threat * qlcfg.victory_threat_xp_rate) as u32;

                        // Fame: scales with threat relative to level
                        // Easy quests (threat < level*2) give 0 fame
                        let fame_base = (threat - adv.level as f32 * 2.0).max(0.0);
                        let solo_bonus = if member_ids.len() == 1 { 2.0 } else { 1.0 };
                        let crisis_mult = if !state.overworld.active_crises.is_empty() {
                            1.0 + state.overworld.active_crises.len() as f32 * 0.5
                        } else { 1.0 };
                        let fame_gained = fame_base * solo_bonus * crisis_mult;
                        adv.tier_status.fame += fame_gained;
                        if adv.tier_status.fame > adv.tier_status.peak_fame {
                            adv.tier_status.peak_fame = adv.tier_status.fame;
                        }
                        adv.tier_status.quests_completed += 1;
                        adv.tier_status.party_victories += 1;

                        // Level up check
                        let threshold = adv.level * adv.level * qlcfg.level_up_xp_multiplier;
                        if adv.xp >= threshold {
                            adv.level += 1;
                            adv.stats.max_hp += qlcfg.level_hp_gain;
                            adv.stats.attack += qlcfg.level_attack_gain;
                            adv.stats.defense += qlcfg.level_defense_gain;
                            events.push(WorldEvent::AdventurerLevelUp {
                                adventurer_id: mid,
                                new_level: adv.level,
                            });
                        }

                        // --- History tag accumulation ---
                        // Quest type tags
                        match quest.request.quest_type {
                            QuestType::Combat     => { *adv.history_tags.entry("combat".into()).or_default() += 1; }
                            QuestType::Exploration => { *adv.history_tags.entry("exploration".into()).or_default() += 1; }
                            QuestType::Diplomatic  => { *adv.history_tags.entry("diplomatic".into()).or_default() += 1; }
                            QuestType::Escort      => { *adv.history_tags.entry("escort".into()).or_default() += 1; }
                            QuestType::Rescue      => { *adv.history_tags.entry("rescue".into()).or_default() += 1; }
                            QuestType::Gather      => { *adv.history_tags.entry("gather".into()).or_default() += 1; }
                        }
                        // Solo vs party
                        if member_ids.len() == 1 {
                            *adv.history_tags.entry("solo".into()).or_default() += 1;
                        } else {
                            *adv.history_tags.entry("party_combat".into()).or_default() += 1;
                        }
                        // Survived heavy damage
                        if battle_hp < 0.3 {
                            *adv.history_tags.entry("near_death".into()).or_default() += 1;
                        }
                        // High threat quest
                        if threat > 50.0 {
                            *adv.history_tags.entry("high_threat".into()).or_default() += 1;
                        }
                        // Region defense (quest had a target position near a region)
                        if quest.request.quest_type == QuestType::Combat {
                            *adv.history_tags.entry("region_defense".into()).or_default() += 1;
                        }
                        // Crisis involvement tags
                        for crisis in &state.overworld.active_crises {
                            let tag = match crisis {
                                ActiveCrisis::SleepingKing { .. } => "crisis_sleeping_king",
                                ActiveCrisis::Breach { .. } => "crisis_breach_defense",
                                ActiveCrisis::Corruption { .. } => "crisis_blight_prevention",
                                ActiveCrisis::Unifier { .. } => "crisis_unifier_resistance",
                                ActiveCrisis::Decline { .. } => "crisis_decline_survival",
                            };
                            *adv.history_tags.entry(tag.into()).or_default() += 1;
                        }
                    }
                    QuestResult::Defeat => {
                        // Fame loss on defeat
                        adv.tier_status.fame = (adv.tier_status.fame - 15.0).max(0.0);
                        adv.tier_status.quests_completed += 1;

                        // Defeat: heavy injuries, possible death
                        let severity = threat / qlcfg.defeat_severity_divisor;
                        adv.injury = (adv.injury + qlcfg.defeat_base_injury + severity * qlcfg.defeat_severity_injury).clamp(0.0, 100.0);
                        adv.stress = (adv.stress + qlcfg.defeat_stress_gain).min(100.0);
                        adv.loyalty = (adv.loyalty - qlcfg.defeat_loyalty_loss).max(0.0);
                        adv.morale = (adv.morale - qlcfg.defeat_morale_loss).max(0.0);
                        adv.xp += qlcfg.defeat_base_xp;

                        // Death check: high injury + bad luck
                        if adv.injury > 90.0 {
                            let death_roll = lcg_f32(&mut state.rng);
                            if death_roll < qlcfg.death_chance {
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
                        if adv.injury > qlcfg.incapacitation_threshold {
                            adv.status = AdventurerStatus::Injured;
                            events.push(WorldEvent::AdventurerInjured {
                                adventurer_id: mid,
                                injury_level: adv.injury,
                            });
                        }
                    }
                    QuestResult::Abandoned => {
                        adv.stress = (adv.stress + qlcfg.abandon_stress).min(100.0);
                        adv.morale = (adv.morale - qlcfg.abandon_morale_loss).max(0.0);
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

        // --- Bond system integration ---
        // Quest completion: boost bonds between all party members.
        if result == QuestResult::Victory {
            super::bonds::on_quest_completed(state, &member_ids);
        }
        // Death grief: apply stress/morale penalties to bonded adventurers.
        let dead_ids: Vec<u32> = member_ids
            .iter()
            .copied()
            .filter(|&id| {
                state.adventurers.iter().any(|a| a.id == id && a.status == AdventurerStatus::Dead)
            })
            .collect();
        for dead_id in dead_ids {
            super::bonds::on_adventurer_died(state, dead_id, events);
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

        // Distribute NPC share of gold reward to participating adventurers.
        if result == QuestResult::Victory && reward.gold > 0.0 {
            let npc_share_fraction = state.config.npc_economy.quest_reward_npc_share;
            let npc_gold = reward.gold * npc_share_fraction;
            // Compute power ratings for proportional split.
            let power_ratings: Vec<(u32, f32)> = member_ids
                .iter()
                .filter_map(|id| {
                    state.adventurers.iter().find(|a| a.id == *id).map(|a| {
                        let eff = super::npc_economy::effective_level(a);
                        (*id, super::npc_economy::power_rating(eff))
                    })
                })
                .collect();
            let total_power: f32 = power_ratings.iter().map(|(_, p)| p).sum();
            if total_power > 0.0 {
                for (adv_id, pr) in &power_ratings {
                    let share = npc_gold * (pr / total_power);
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adv_id) {
                        adv.gold += share;
                        adv.ticks_since_income = 0;
                    }
                }
            }
        }

        // Generate and equip loot on victory
        if result == QuestResult::Victory {
            process_quest_loot(
                state,
                quest.request.quest_type,
                threat,
                quest.request.reward.potential_loot,
                &member_ids,
            );
        }

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
            threat_level: quest.request.threat_level,
        });

        // Remove associated battle
        state.active_battles.retain(|b| b.quest_id != quest.id);
    }
}
