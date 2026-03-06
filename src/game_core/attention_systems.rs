use bevy::prelude::*;

use super::types::*;
use super::overworld_types::*;
use super::roster_types::*;
use super::companion::*;

/// Returns the new focus index if the shift is permitted, `None` otherwise.
pub fn try_shift_focus(
    entity_count: usize,
    attention: &mut AttentionState,
    current_idx: usize,
    delta: i32,
) -> Option<usize> {
    if delta == 0 || entity_count < 2 {
        return None;
    }
    if attention.switch_cooldown_turns > 0 || attention.global_energy < attention.switch_cost {
        return None;
    }

    let len = entity_count as i32;
    let current = (current_idx as i32).clamp(0, len - 1);
    let next = (current + delta).rem_euclid(len) as usize;
    if next == current_idx {
        return None;
    }

    attention.switch_cooldown_turns = attention.switch_cooldown_max;
    attention.global_energy = (attention.global_energy - attention.switch_cost).max(0.0);
    Some(next)
}

pub fn focus_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut commands: Commands,
    board: Res<MissionBoard>,
    active_query: Query<(Entity, &MissionData), With<ActiveMission>>,
    mut attention: ResMut<AttentionState>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };

    let mut delta = 0_i32;
    if keyboard.just_pressed(KeyCode::Tab) || keyboard.just_pressed(KeyCode::BracketRight) {
        delta = 1;
    } else if keyboard.just_pressed(KeyCode::BracketLeft) {
        delta = -1;
    }
    if delta == 0 {
        return;
    }

    let Ok((active_entity, active_data)) = active_query.get_single() else {
        return;
    };
    let current_idx = board
        .entities
        .iter()
        .position(|&e| e == active_entity)
        .unwrap_or(0);

    if let Some(next_idx) =
        try_shift_focus(board.entities.len(), &mut attention, current_idx, delta)
    {
        if let Some(&next_entity) = board.entities.get(next_idx) {
            commands.entity(active_entity).remove::<ActiveMission>();
            commands.entity(next_entity).insert(ActiveMission);
            println!("Attention shifted to mission entity slot {}.", next_idx);
        }
    } else {
        println!(
            "Focus shift blocked (cooldown: {}, energy: {:.1}/{:.1}).",
            attention.switch_cooldown_turns, attention.global_energy, attention.max_energy
        );
        let _ = active_data; // suppress unused warning
    }
}

pub fn focused_attention_intervention_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&mut MissionProgress, &MissionTactics), With<ActiveMission>>,
    mut attention: ResMut<AttentionState>,
) {
    if run_state.global_turn == 0 {
        return;
    }
    let Ok((mut progress, tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    let base_gain = match tactics.tactical_mode {
        TacticalMode::Balanced => 4.8,
        TacticalMode::Aggressive => 6.0,
        TacticalMode::Defensive => 3.6,
    };
    let energy_spend = 4.0_f32.min(attention.global_energy.max(0.0));
    let leverage = (energy_spend / 4.0).clamp(0.0, 1.0);
    attention.global_energy = (attention.global_energy - energy_spend).max(0.0);

    if leverage <= 0.0 {
        return;
    }

    progress.sabotage_progress =
        (progress.sabotage_progress + base_gain * leverage).min(progress.sabotage_goal);
    let alert_relief = match tactics.tactical_mode {
        TacticalMode::Balanced => 0.9,
        TacticalMode::Aggressive => 0.4,
        TacticalMode::Defensive => 1.3,
    };
    progress.alert_level = (progress.alert_level - alert_relief * leverage).max(0.0);
}

pub fn simulate_unfocused_missions_system(
    run_state: Res<RunState>,
    mut unfocused_query: Query<(&mut MissionProgress, &MissionTactics), Without<ActiveMission>>,
    mut active_query: Query<&mut MissionProgress, With<ActiveMission>>,
) {
    if run_state.global_turn == 0 {
        return;
    }

    // Reset unattended counter on the active mission.
    for mut progress in active_query.iter_mut() {
        progress.unattended_turns = 0;
    }

    for (mut mission, tactics) in unfocused_query.iter_mut() {
        if !mission.mission_active || mission.result != MissionResult::InProgress {
            continue;
        }

        mission.unattended_turns = mission.unattended_turns.saturating_add(1);
        if mission.turns_remaining > 0 {
            mission.turns_remaining -= 1;
        }

        let base_progress = match tactics.tactical_mode {
            TacticalMode::Balanced => 6.0,
            TacticalMode::Aggressive => 8.0,
            TacticalMode::Defensive => 4.0,
        };
        let unattended_factor = 1.0 + (mission.unattended_turns as f32 * 0.08).min(0.96);
        let pressure_tax = (0.8 + mission.alert_level * 0.02) * unattended_factor;
        mission.sabotage_progress = (mission.sabotage_progress + base_progress - pressure_tax)
            .clamp(0.0, mission.sabotage_goal);
        mission.alert_level = (mission.alert_level + 1.6 * unattended_factor).min(100.0);
        mission.reactor_integrity = (mission.reactor_integrity
            - ((0.6 + mission.alert_level * 0.015) * unattended_factor))
            .max(0.0);

        if mission.sabotage_progress >= mission.sabotage_goal {
            mission.mission_active = false;
            mission.result = MissionResult::Victory;
            continue;
        }
        if mission.turns_remaining == 0 || mission.reactor_integrity <= 0.0 {
            mission.mission_active = false;
            mission.result = MissionResult::Defeat;
        }
    }
}

pub fn sync_mission_assignments_system(
    roster: Res<CampaignRoster>,
    mut mission_query: Query<(&MissionProgress, &mut AssignedHero, Has<ActiveMission>)>,
) {
    if roster.heroes.is_empty() {
        return;
    }

    let is_valid_hero = |id: u32| {
        roster
            .heroes
            .iter()
            .any(|h| h.id == id && h.active && !h.deserter)
    };
    let best_available = || {
        roster
            .heroes
            .iter()
            .filter(|h| h.active && !h.deserter)
            .max_by(|a, b| {
                let sa = (a.loyalty + a.resolve) - (a.stress + a.fatigue + a.injury);
                let sb = (b.loyalty + b.resolve) - (b.stress + b.fatigue + b.injury);
                sa.total_cmp(&sb).then(a.id.cmp(&b.id))
            })
            .map(|h| h.id)
    };
    let player_pick = roster.player_hero_id.filter(|id| is_valid_hero(*id));
    let fallback_pick = best_available();
    let non_player_pick = roster
        .heroes
        .iter()
        .filter(|h| h.active && !h.deserter)
        .filter(|h| Some(h.id) != player_pick)
        .max_by(|a, b| {
            let sa = (a.loyalty + a.resolve) - (a.stress + a.fatigue + a.injury);
            let sb = (b.loyalty + b.resolve) - (b.stress + b.fatigue + b.injury);
            sa.total_cmp(&sb).then(a.id.cmp(&b.id))
        })
        .map(|h| h.id)
        .or(fallback_pick);

    for (progress, mut assigned, is_active) in mission_query.iter_mut() {
        if progress.result != MissionResult::InProgress || !progress.mission_active {
            assigned.hero_id = None;
            continue;
        }
        if is_active {
            assigned.hero_id = player_pick.or(fallback_pick);
            continue;
        }
        let valid = assigned.hero_id.filter(|id| is_valid_hero(*id)).is_some();
        if !valid {
            assigned.hero_id = non_player_pick;
        }
    }
}

pub fn companion_mission_impact_system(
    run_state: Res<RunState>,
    roster: Res<CampaignRoster>,
    mut mission_query: Query<(&mut MissionProgress, &AssignedHero)>,
) {
    if run_state.global_turn == 0 || roster.heroes.is_empty() {
        return;
    }
    for (mut progress, assigned) in mission_query.iter_mut() {
        if !progress.mission_active || progress.result != MissionResult::InProgress {
            continue;
        }
        let Some(hero_id) = assigned.hero_id else {
            continue;
        };
        let Some(hero) = roster.heroes.iter().find(|h| h.id == hero_id) else {
            continue;
        };

        let archetype_bonus = match hero.archetype {
            PersonalityArchetype::Vanguard => 0.8,
            PersonalityArchetype::Guardian => 0.5,
            PersonalityArchetype::Tactician => 1.0,
        };
        let composure = ((hero.resolve + hero.loyalty)
            - (hero.stress + hero.fatigue + hero.injury))
            .clamp(-40.0, 60.0);
        let progress_delta = (composure * 0.03 + archetype_bonus).clamp(-1.5, 2.8);
        progress.sabotage_progress =
            (progress.sabotage_progress + progress_delta).clamp(0.0, progress.sabotage_goal);

        let alert_delta = if composure >= 0.0 {
            -(0.4 + composure * 0.01)
        } else {
            0.6 + (-composure) * 0.015
        };
        progress.alert_level = (progress.alert_level + alert_delta).clamp(0.0, 100.0);
    }
}

pub fn companion_state_drift_system(
    run_state: Res<RunState>,
    mission_query: Query<(&MissionProgress, &AssignedHero)>,
    mut roster: ResMut<CampaignRoster>,
) {
    if run_state.global_turn == 0 || roster.heroes.is_empty() {
        return;
    }

    // Build a map from hero_id -> (alert_level, is_active_in_progress).
    let hero_mission: std::collections::HashMap<u32, (f32, bool)> = mission_query
        .iter()
        .filter_map(|(progress, assigned)| {
            let hero_id = assigned.hero_id?;
            let on_active = progress.mission_active && progress.result == MissionResult::InProgress;
            Some((hero_id, (progress.alert_level, on_active)))
        })
        .collect();

    for hero in &mut roster.heroes {
        if let Some(&(alert_level, on_active)) = hero_mission.get(&hero.id) {
            if on_active {
                hero.stress = (hero.stress + 0.7 + alert_level * 0.01).min(100.0);
                hero.fatigue = (hero.fatigue + 0.6).min(100.0);
                if alert_level > 55.0 {
                    hero.loyalty = (hero.loyalty - 0.35).max(0.0);
                } else {
                    hero.loyalty = (hero.loyalty + 0.08).min(100.0);
                }
                continue;
            }
        }

        hero.stress = (hero.stress - 0.5).max(0.0);
        hero.fatigue = (hero.fatigue - 0.7).max(0.0);
        hero.loyalty = (hero.loyalty + 0.04).min(100.0);
    }
}

pub fn generate_companion_story_quests_system(
    run_state: Res<RunState>,
    mission_query: Query<(&MissionProgress, &AssignedHero)>,
    overworld: Res<OverworldMap>,
    roster: Res<CampaignRoster>,
    mut story: ResMut<CompanionStoryState>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 || roster.heroes.is_empty() || overworld.regions.is_empty() {
        return;
    }

    // Build hero_id -> pressure lookup from mission entities.
    let hero_pressure: std::collections::HashMap<u32, f32> = mission_query
        .iter()
        .filter_map(|(progress, assigned)| {
            let id = assigned.hero_id?;
            Some((
                id,
                progress.alert_level + (100.0 - progress.reactor_integrity) * 0.4,
            ))
        })
        .collect();

    let mut issued_events = Vec::new();
    for hero in roster.heroes.iter().filter(|h| h.active && !h.deserter) {
        if quest_for_hero(&story, hero.id).is_some() {
            continue;
        }
        let assigned_pressure = hero_pressure.get(&hero.id).copied().unwrap_or(0.0);
        let home_pressure = overworld
            .regions
            .iter()
            .find(|r| r.id == hero.origin_region_id)
            .map(|r| r.unrest + (100.0 - r.control) * 0.6)
            .unwrap_or(45.0);
        let trigger = hero.stress >= 42.0
            || hero.loyalty <= 55.0
            || assigned_pressure >= 58.0
            || home_pressure >= 70.0;
        if !trigger {
            continue;
        }
        let id = story.next_id;
        story.next_id = story.next_id.saturating_add(1);
        let quest = build_companion_quest(
            hero,
            run_state.global_turn,
            overworld.map_seed ^ 0xCA11_5100,
            id,
            &overworld,
        );
        story.notice = format!("Story quest issued: {}", quest.title);
        issued_events.push(format!(
            "Companion quest issued for {}: {}",
            hero.name, quest.title
        ));
        story.quests.push(quest);
    }
    if let Some(log) = event_log.as_mut() {
        for event in issued_events {
            super::roster_types::push_campaign_event(log, run_state.global_turn, event);
        }
    }
}

pub fn progress_companion_story_quests_system(
    mut roster: ResMut<CampaignRoster>,
    ledger: Res<CampaignLedger>,
    mut story: ResMut<CompanionStoryState>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if roster.heroes.is_empty() {
        return;
    }

    for hero in &roster.heroes {
        if hero.deserter {
            if let Some(quest) = quest_for_hero_mut(&mut story, hero.id) {
                quest.status = CompanionQuestStatus::Failed;
                story.notice = format!("Story quest failed: {}", quest.title);
            }
        }
    }

    if story.processed_ledger_len > ledger.records.len() {
        story.processed_ledger_len = 0;
    }
    let mut pending_notice: Option<String> = None;
    let mut pending_events = Vec::new();
    for record in ledger.records.iter().skip(story.processed_ledger_len) {
        let Some(hero_id) = record.hero_id else {
            continue;
        };
        let Some(quest) = quest_for_hero_mut(&mut story, hero_id) else {
            continue;
        };
        match record.result {
            MissionResult::Victory => {
                quest.progress = quest.progress.saturating_add(1);
            }
            MissionResult::Defeat => {
                if quest.progress > 0 {
                    quest.progress -= 1;
                } else {
                    quest.status = CompanionQuestStatus::Failed;
                    let msg = format!("Story quest failed: {}", quest.title);
                    pending_notice = Some(msg.clone());
                    pending_events.push(msg);
                }
            }
            MissionResult::InProgress => {}
        }
        if quest.status == CompanionQuestStatus::Active && quest.progress >= quest.target {
            quest.status = CompanionQuestStatus::Completed;
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == quest.hero_id) {
                hero.loyalty = (hero.loyalty + quest.reward_loyalty).clamp(0.0, 100.0);
                hero.resolve = (hero.resolve + quest.reward_resolve).clamp(0.0, 100.0);
                hero.stress = (hero.stress - 6.0).max(0.0);
            }
            let msg = format!("Story quest complete: {}", quest.title);
            pending_notice = Some(msg.clone());
            pending_events.push(msg);
        }
    }
    if let Some(msg) = pending_notice {
        story.notice = msg;
    }
    if let Some(log) = event_log.as_mut() {
        for event in pending_events {
            let turn = ledger.records.last().map(|r| r.turn).unwrap_or(0);
            super::roster_types::push_campaign_event(log, turn, event);
        }
    }
    story.processed_ledger_len = ledger.records.len();
}

pub fn companion_recovery_system(run_state: Res<RunState>, mut roster: ResMut<CampaignRoster>) {
    if run_state.global_turn == 0 || run_state.global_turn % 3 != 0 {
        return;
    }

    for hero in &mut roster.heroes {
        if hero.deserter {
            continue;
        }
        if hero.active {
            continue;
        }
        hero.stress = (hero.stress - 3.0).max(0.0);
        hero.fatigue = (hero.fatigue - 4.0).max(0.0);
        hero.injury = (hero.injury - 3.5).max(0.0);
        hero.loyalty = (hero.loyalty + 0.7).min(100.0);

        if hero.injury <= 40.0 && hero.fatigue <= 40.0 {
            hero.active = true;
        }
    }
}
