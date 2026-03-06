use bevy::prelude::*;

use super::overworld_types::FlashpointState;
use super::roster_types::*;
use super::types::*;

pub fn resolve_mission_consequences_system(
    run_state: Res<RunState>,
    mut mission_query: Query<(&MissionData, &mut MissionProgress, &mut AssignedHero)>,
    mut roster: ResMut<CampaignRoster>,
    mut ledger: ResMut<CampaignLedger>,
    mut flashpoints: Option<ResMut<FlashpointState>>,
    board: Res<MissionBoard>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 {
        return;
    }

    for (slot, &entity) in board.entities.iter().enumerate() {
        let Ok((data, mut progress, mut assigned)) = mission_query.get_mut(entity) else {
            continue;
        };
        if progress.result == MissionResult::InProgress || progress.outcome_recorded {
            continue;
        }

        let hero_id = assigned.hero_id;
        if let Some(state) = flashpoints.as_mut() {
            if let Some(chain) = state
                .chains
                .iter_mut()
                .find(|c| !c.completed && c.mission_slot == slot)
            {
                chain.companion_hook_hero_id = hero_id;
            }
        }
        let mut summary = format!("{} resolved as {:?}", data.mission_name, progress.result);

        if let Some(id) = hero_id {
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == id) {
                match progress.result {
                    MissionResult::Victory => {
                        hero.stress = (hero.stress - 6.0).max(0.0);
                        hero.fatigue = (hero.fatigue + 1.5).min(100.0);
                        hero.injury = (hero.injury - 1.0).max(0.0);
                        hero.loyalty = (hero.loyalty + 1.8).min(100.0);
                        summary = format!(
                            "{} succeeded on '{}'. morale improved.",
                            hero.name, data.mission_name
                        );
                    }
                    MissionResult::Defeat => {
                        let pressure = (progress.alert_level * 0.05)
                            + (progress.unattended_turns as f32 * 0.25);
                        hero.stress = (hero.stress + 12.0 + pressure).min(100.0);
                        hero.fatigue = (hero.fatigue + 8.0 + pressure * 0.5).min(100.0);
                        hero.injury = (hero.injury + 6.0 + pressure * 0.8).min(100.0);
                        hero.loyalty = (hero.loyalty - (3.0 + pressure * 0.4)).max(0.0);

                        if hero.injury >= 95.0 || (hero.loyalty <= 10.0 && hero.stress >= 70.0) {
                            hero.deserter = true;
                            hero.active = false;
                            summary = format!(
                                "{} deserted after '{}' collapse.",
                                hero.name, data.mission_name
                            );
                        } else if hero.injury >= 65.0 {
                            hero.active = false;
                            summary = format!(
                                "{} is sidelined with severe injuries from '{}'.",
                                hero.name, data.mission_name
                            );
                        } else {
                            summary = format!(
                                "{} survived defeat at '{}'.",
                                hero.name, data.mission_name
                            );
                        }
                    }
                    MissionResult::InProgress => {}
                }
            }
        }

        progress.outcome_recorded = true;
        assigned.hero_id = None;
        ledger.records.push(ConsequenceRecord {
            turn: run_state.global_turn,
            mission_name: data.mission_name.clone(),
            result: progress.result,
            hero_id,
            summary,
        });
        if let Some(log) = event_log.as_mut() {
            if let Some(last) = ledger.records.last() {
                push_campaign_event(log, run_state.global_turn, last.summary.clone());
            }
        }
    }
}
