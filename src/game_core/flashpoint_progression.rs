use bevy::prelude::*;

use super::campaign_systems::ensure_faction_mission_slots;
use super::companion::*;
use super::flashpoint_helpers::*;
use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;

pub fn flashpoint_progression_system(
    run_state: Res<RunState>,
    board: Res<MissionBoard>,
    mut mission_query: Query<(&mut MissionData, &mut MissionProgress, &mut MissionTactics)>,
    mut overworld: ResMut<OverworldMap>,
    mut roster: ResMut<CampaignRoster>,
    mut flashpoints: ResMut<FlashpointState>,
    mut story: Option<ResMut<CompanionStoryState>>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 || flashpoints.chains.is_empty() {
        return;
    }

    let mut pending_events = Vec::new();
    for slot in 0..board.entities.len() {
        let Some(chain_idx) = flashpoints
            .chains
            .iter()
            .position(|c| !c.completed && c.mission_slot == slot)
        else {
            continue;
        };

        let Some(&entity) = board.entities.get(slot) else {
            continue;
        };
        let Ok((mut data, mut progress, mut tactics)) = mission_query.get_mut(entity) else {
            continue;
        };
        if progress.result == MissionResult::InProgress || !progress.outcome_recorded {
            continue;
        }

        // Build snapshot for helper functions that take &mut MissionSnapshot.
        let mut snap = MissionSnapshot::from_components(&data, &progress, &tactics);

        let mut chain = flashpoints.chains[chain_idx].clone();
        let region_name = overworld
            .regions
            .iter()
            .find(|r| r.id == chain.region_id)
            .map(|r| r.name.clone())
            .unwrap_or_else(|| format!("Region {}", chain.region_id));
        let attacker_name = overworld
            .factions
            .get(chain.attacker_faction_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", chain.attacker_faction_id));
        let defender_name = overworld
            .factions
            .get(chain.defender_faction_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", chain.defender_faction_id));

        match progress.result {
            MissionResult::Victory if chain.stage < FLASHPOINT_TOTAL_STAGES => {
                chain.stage = chain.stage.saturating_add(1);
                configure_flashpoint_stage_mission(
                    &mut snap,
                    &chain,
                    &overworld,
                    overworld.map_seed ^ (run_state.global_turn as u64).wrapping_mul(911),
                );
                let _ = apply_flashpoint_companion_hook(&mut snap, &chain, &roster);
                chain.objective = flashpoint_hook_objective_suffix(&chain, &roster)
                    .unwrap_or_else(|| "No companion hook".to_string());
                apply_flashpoint_intent(&mut snap, &chain);
                rewrite_flashpoint_mission_name(&mut snap, &chain, &overworld, Some(&roster));
                // Write snapshot back.
                progress.mission_active = snap.mission_active;
                progress.result = snap.result;
                progress.turns_remaining = snap.turns_remaining;
                progress.reactor_integrity = snap.reactor_integrity;
                progress.sabotage_progress = snap.sabotage_progress;
                progress.sabotage_goal = snap.sabotage_goal;
                progress.alert_level = snap.alert_level;
                progress.room_index = snap.room_index;
                progress.unattended_turns = snap.unattended_turns;
                progress.outcome_recorded = snap.outcome_recorded;
                tactics.tactical_mode = snap.tactical_mode;
                tactics.command_cooldown_turns = snap.command_cooldown_turns;
                data.mission_name = snap.mission_name.clone();
                data.bound_region_id = snap.bound_region_id;
                flashpoints.notice = format!(
                    "Flashpoint advanced in {}: stage {}/{}.",
                    region_name, chain.stage, FLASHPOINT_TOTAL_STAGES
                );
                pending_events.push(flashpoints.notice.clone());
                flashpoints.chains[chain_idx] = chain;
                continue;
            }
            MissionResult::Victory => {
                chain.completed = true;
                if let Some(region) = overworld
                    .regions
                    .iter_mut()
                    .find(|r| r.id == chain.region_id)
                {
                    region.owner_faction_id = chain.attacker_faction_id;
                    region.control = (region.control + 18.0).clamp(0.0, 100.0);
                    region.unrest = (region.unrest - 14.0).clamp(0.0, 100.0);
                }
                ensure_faction_mission_slots(&mut overworld);
                if let Some(attacker) = overworld.factions.get_mut(chain.attacker_faction_id) {
                    attacker.strength = (attacker.strength + 7.0).clamp(30.0, 180.0);
                    attacker.cohesion = (attacker.cohesion + 4.0).clamp(10.0, 95.0);
                }
                if let Some(defender) = overworld.factions.get_mut(chain.defender_faction_id) {
                    defender.strength = (defender.strength - 5.0).clamp(30.0, 180.0);
                    defender.cohesion = (defender.cohesion - 3.0).clamp(10.0, 95.0);
                }
                inject_recruit_for_faction(
                    &mut roster,
                    &overworld,
                    chain.attacker_faction_id,
                    overworld.map_seed ^ run_state.global_turn as u64 ^ chain.id as u64,
                );
                if let Some((hero_id, hero_name, _kind)) =
                    flashpoint_companion_hook_kind(&chain, &roster)
                {
                    if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == hero_id) {
                        hero.loyalty = (hero.loyalty + 4.0).clamp(0.0, 100.0);
                        hero.resolve = (hero.resolve + 3.0).clamp(0.0, 100.0);
                        hero.stress = (hero.stress - 4.0).clamp(0.0, 100.0);
                    }
                    pending_events.push(format!(
                        "{hero_name} gained renown from flashpoint resolution in {}.",
                        region_name
                    ));
                    if let Some(story_state) = story.as_mut() {
                        if let Some(quest) = story_state.quests.iter_mut().find(|q| {
                            q.hero_id == hero_id
                                && q.status == CompanionQuestStatus::Active
                                && q.progress < q.target
                        }) {
                            quest.progress = quest.progress.saturating_add(1).min(quest.target);
                            story_state.notice = format!(
                                "Flashpoint beat advanced companion quest: {}",
                                quest.title
                            );
                            pending_events.push(story_state.notice.clone());
                        }
                    }
                }
                if !chain.objective.is_empty() {
                    pending_events.push(format!(
                        "Hook objective resolved at {}: {}",
                        region_name, chain.objective
                    ));
                }
                flashpoints.notice = format!(
                    "Flashpoint resolved: {} seized {} from {} and opened new recruits.",
                    attacker_name, region_name, defender_name
                );
                pending_events.push(flashpoints.notice.clone());
            }
            MissionResult::Defeat => {
                chain.completed = true;
                if let Some(region) = overworld
                    .regions
                    .iter_mut()
                    .find(|r| r.id == chain.region_id)
                {
                    region.owner_faction_id = chain.defender_faction_id;
                    region.control = (region.control + 12.0).clamp(0.0, 100.0);
                    region.unrest = (region.unrest - 9.0).clamp(0.0, 100.0);
                }
                ensure_faction_mission_slots(&mut overworld);
                if let Some(defender) = overworld.factions.get_mut(chain.defender_faction_id) {
                    defender.strength = (defender.strength + 5.0).clamp(30.0, 180.0);
                    defender.cohesion = (defender.cohesion + 3.0).clamp(10.0, 95.0);
                }
                if let Some(attacker) = overworld.factions.get_mut(chain.attacker_faction_id) {
                    attacker.strength = (attacker.strength - 4.0).clamp(30.0, 180.0);
                }
                if let Some(idx) = roster
                    .recruit_pool
                    .iter()
                    .position(|r| r.origin_faction_id == chain.attacker_faction_id)
                {
                    roster.recruit_pool.remove(idx);
                }
                if let Some((hero_id, hero_name, _kind)) =
                    flashpoint_companion_hook_kind(&chain, &roster)
                {
                    if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == hero_id) {
                        hero.loyalty = (hero.loyalty - 3.0).clamp(0.0, 100.0);
                        hero.stress = (hero.stress + 6.0).clamp(0.0, 100.0);
                    }
                    pending_events.push(format!(
                        "{hero_name} took a morale hit from the failed flashpoint at {}.",
                        region_name
                    ));
                    if let Some(story_state) = story.as_mut() {
                        if let Some(quest) = story_state.quests.iter_mut().find(|q| {
                            q.hero_id == hero_id && q.status == CompanionQuestStatus::Active
                        }) {
                            if quest.progress > 0 {
                                quest.progress -= 1;
                            }
                            story_state.notice = format!(
                                "Flashpoint setback affected companion quest: {}",
                                quest.title
                            );
                            pending_events.push(story_state.notice.clone());
                        }
                    }
                }
                if !chain.objective.is_empty() {
                    pending_events.push(format!(
                        "Hook objective failed at {}: {}",
                        region_name, chain.objective
                    ));
                }
                flashpoints.notice = format!(
                    "Flashpoint collapsed in {}: {} held against {}.",
                    region_name, defender_name, attacker_name
                );
                pending_events.push(flashpoints.notice.clone());
            }
            MissionResult::InProgress => {}
        }

        flashpoints.chains[chain_idx] = chain;
    }

    flashpoints.chains.retain(|c| !c.completed);
    if flashpoints.chains.is_empty() && flashpoints.notice.is_empty() {
        flashpoints.notice = "No active flashpoints.".to_string();
    }
    if let Some(log) = event_log.as_mut() {
        for event in pending_events {
            push_campaign_event(log, run_state.global_turn, event);
        }
    }
}
