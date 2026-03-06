use bevy::prelude::*;

use super::helpers::*;
use super::super::attention_systems::*;
use super::super::companion::*;
use super::super::consequence_systems::*;
use super::super::overworld_types::*;
use super::super::roster_types::*;
use super::super::types::*;

#[test]
fn companion_state_persists_and_modifies_board() {
    let mut app = build_triage_app_with_roster();
    set_nth_progress(&mut app.world, 0, |p| {
        p.alert_level = 32.0;
        p.sabotage_progress = 25.0;
    });
    let initial = nth_progress(&mut app.world, 0);
    let initial_hero_snapshot = app.world.resource::<CampaignRoster>().heroes[0].clone();
    for _ in 0..5 {
        app.update();
    }
    let mission = nth_progress(&mut app.world, 0);
    assert!(mission.sabotage_progress > initial.sabotage_progress);
    assert!(mission.alert_level <= initial.alert_level + 8.0);
    let roster = app.world.resource::<CampaignRoster>();
    let hero = roster
        .heroes
        .iter()
        .find(|h| h.id == initial_hero_snapshot.id)
        .expect("hero must exist");
    assert!(
        hero.stress != initial_hero_snapshot.stress
            || hero.fatigue != initial_hero_snapshot.fatigue
    );
}

#[test]
fn companion_story_quest_issues_when_pressure_is_high() {
    let mut world = World::new();
    let mut roster = CampaignRoster::default();
    roster.heroes[0].stress = 78.0;
    world.insert_resource(RunState { global_turn: 4 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(OverworldMap::default());
    world.insert_resource(roster);
    world.insert_resource(CompanionStoryState::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(generate_companion_story_quests_system);
    schedule.run(&mut world);
    let story = world.resource::<CompanionStoryState>();
    assert!(!story.quests.is_empty());
    assert_eq!(story.quests[0].status, CompanionQuestStatus::Active);
}

#[test]
fn companion_story_quest_completion_rewards_hero() {
    let mut world = World::new();
    let mut roster = CampaignRoster::default();
    let hero_id = roster.heroes[0].id;
    roster.heroes[0].loyalty = 40.0;
    roster.heroes[0].resolve = 55.0;
    let base_loyalty = roster.heroes[0].loyalty;
    let base_resolve = roster.heroes[0].resolve;
    let story = CompanionStoryState {
        quests: vec![CompanionQuest {
            id: 1,
            hero_id,
            kind: CompanionQuestKind::Reckoning,
            status: CompanionQuestStatus::Active,
            title: "Test Quest".to_string(),
            objective: "Win once".to_string(),
            progress: 0,
            target: 1,
            issued_turn: 1,
            reward_loyalty: 7.0,
            reward_resolve: 5.0,
        }],
        next_id: 2,
        processed_ledger_len: 0,
        notice: String::new(),
    };
    let ledger = CampaignLedger {
        records: vec![ConsequenceRecord {
            turn: 6,
            mission_name: "Test".to_string(),
            result: MissionResult::Victory,
            hero_id: Some(hero_id),
            summary: "ok".to_string(),
        }],
    };
    world.insert_resource(roster);
    world.insert_resource(ledger);
    world.insert_resource(story);
    let mut schedule = Schedule::default();
    schedule.add_systems(progress_companion_story_quests_system);
    schedule.run(&mut world);
    let story = world.resource::<CompanionStoryState>();
    let quest = &story.quests[0];
    assert_eq!(quest.status, CompanionQuestStatus::Completed);
    let roster = world.resource::<CampaignRoster>();
    let hero = &roster.heroes[0];
    assert!(hero.loyalty > base_loyalty);
    assert!(hero.resolve > base_resolve);
}

#[test]
fn mission_outcome_records_consequence_once() {
    let mut app = build_triage_app();
    set_active_progress(&mut app.world, |p| {
        p.result = MissionResult::Defeat;
        p.mission_active = false;
        p.alert_level = 62.0;
        p.unattended_turns = 8;
    });
    {
        let entity = app.world.resource::<MissionBoard>().entities[0];
        app.world.get_mut::<AssignedHero>(entity).unwrap().hero_id = Some(1);
    }
    app.update();
    app.update();
    let ledger = app.world.resource::<CampaignLedger>();
    assert_eq!(ledger.records.len(), 1);
    let p = active_progress(&mut app.world);
    assert!(p.outcome_recorded);
}

#[test]
fn extreme_defeat_can_cause_desertion() {
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 9 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(CampaignLedger::default());
    let mut roster = CampaignRoster::default();
    let hero_id = roster.heroes[0].id;
    roster.heroes[0].loyalty = 8.0;
    roster.heroes[0].stress = 70.0;
    world.insert_resource(roster);
    let snap = MissionSnapshot {
        mission_name: "Overrun".to_string(),
        bound_region_id: Some(0),
        mission_active: false,
        result: MissionResult::Defeat,
        turns_remaining: 0,
        reactor_integrity: 2.0,
        sabotage_progress: 3.0,
        sabotage_goal: 100.0,
        alert_level: 95.0,
        room_index: 2,
        tactical_mode: TacticalMode::Aggressive,
        command_cooldown_turns: 0,
        unattended_turns: 15,
        outcome_recorded: false,
    };
    let (data, progress, tactics) = snap.into_components(0);
    let assigned = AssignedHero {
        hero_id: Some(hero_id),
    };
    let entity = world
        .spawn((data, progress, tactics, assigned, ActiveMission))
        .id();
    world.resource_mut::<MissionBoard>().entities.push(entity);
    let mut schedule = Schedule::default();
    schedule.add_systems(resolve_mission_consequences_system);
    schedule.run(&mut world);
    let roster = world.resource::<CampaignRoster>();
    let hero = &roster.heroes[0];
    assert!(hero.deserter);
    assert!(!hero.active);
}

#[test]
fn sidelined_hero_recovers_over_time() {
    let mut world = World::new();
    let mut roster = CampaignRoster::default();
    roster.heroes[0].active = false;
    roster.heroes[0].deserter = false;
    roster.heroes[0].injury = 38.0;
    roster.heroes[0].fatigue = 39.0;
    roster.heroes[0].stress = 35.0;
    world.insert_resource(roster);
    world.insert_resource(RunState { global_turn: 3 });
    let mut schedule = Schedule::default();
    schedule.add_systems(companion_recovery_system);
    schedule.run(&mut world);
    let roster = world.resource::<CampaignRoster>();
    let hero = &roster.heroes[0];
    assert!(hero.active);
}

#[test]
fn sync_assignments_prefers_player_hero_for_active_mission() {
    let mut world = World::new();
    let mut roster = CampaignRoster::default();
    let player_id = roster.heroes[1].id;
    roster.player_hero_id = Some(player_id);
    world.insert_resource(roster);
    let mut board = MissionBoard::default();
    let mut snapshots = default_mission_snapshots();
    let (d0, p0, t0) = snapshots.remove(0).into_components(0);
    let (d1, p1, t1) = snapshots.remove(0).into_components(1);
    let e0 = world
        .spawn((d0, p0, t0, AssignedHero::default(), ActiveMission))
        .id();
    let e1 = world.spawn((d1, p1, t1, AssignedHero::default())).id();
    board.entities = vec![e0, e1];
    board.next_id = 2;
    world.insert_resource(board);
    let mut schedule = Schedule::default();
    schedule.add_systems(sync_mission_assignments_system);
    schedule.run(&mut world);
    let active_assigned = world
        .get::<AssignedHero>(e0)
        .and_then(|a| a.hero_id)
        .expect("active mission assigned");
    assert_eq!(active_assigned, player_id);
}

#[test]
fn campaign_cycle_regression_snapshot() {
    let run_scenario = || {
        let mut app = build_triage_app();
        for _ in 0..14 {
            app.update();
        }
        set_nth_progress(&mut app.world, 1, |p| {
            p.mission_active = false;
            p.result = MissionResult::Defeat;
            p.alert_level = 66.0;
            p.unattended_turns = 9;
        });
        for _ in 0..2 {
            app.update();
        }
        campaign_signature(&mut app.world)
    };
    let sig_a = run_scenario();
    let sig_b = run_scenario();
    assert_eq!(sig_a, sig_b, "campaign cycle must be deterministic");
}
