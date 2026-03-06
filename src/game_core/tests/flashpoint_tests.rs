use bevy::prelude::*;

use super::helpers::*;
use super::super::companion::*;
use super::super::flashpoint_progression::*;
use super::super::flashpoint_spawn::*;
use super::super::overworld_types::*;
use super::super::roster_types::*;
use super::super::types::*;

#[test]
fn pressure_spawn_binds_each_slot_to_current_region_anchor() {
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 3 });
    world.insert_resource(OverworldMap::default());
    world.insert_resource(MissionBoard::default());
    spawn_test_missions(&mut world);
    let mut schedule = Schedule::default();
    schedule.add_systems(pressure_spawn_missions_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    let max_slot = usize::min(map.factions.len(), board_entities.len());
    for slot in 0..max_slot {
        let region = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .expect("region per slot");
        let entity = board_entities[slot];
        let data = world.get::<MissionData>(entity).unwrap();
        let progress = world.get::<MissionProgress>(entity).unwrap();
        assert_eq!(
            progress.mission_active || !data.mission_name.is_empty(),
            true
        );
        assert_eq!(data.bound_region_id, Some(region.id));
        assert!(data.mission_name.contains(&region.name));
    }
}

#[test]
fn pressure_spawn_replaces_resolved_slot_mission() {
    let mut world = World::new();
    let map = OverworldMap::default();
    world.insert_resource(RunState { global_turn: 5 });
    world.insert_resource(map);
    world.insert_resource(MissionBoard::default());
    spawn_test_missions(&mut world);
    set_nth_progress(&mut world, 1, |p| {
        p.mission_active = false;
        p.result = MissionResult::Defeat;
    });
    {
        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        world
            .get_mut::<MissionData>(entities[1])
            .unwrap()
            .mission_name = "Old Mission".to_string();
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(pressure_spawn_missions_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    let region = map
        .regions
        .iter()
        .find(|r| r.mission_slot == Some(1))
        .expect("slot region");
    let entity1 = board_entities[1];
    let data = world.get::<MissionData>(entity1).unwrap();
    let progress = world.get::<MissionProgress>(entity1).unwrap();
    assert!(progress.mission_active);
    assert_eq!(progress.result, MissionResult::InProgress);
    assert_ne!(data.mission_name, "Old Mission");
    assert_eq!(data.bound_region_id, Some(region.id));
}

#[test]
fn pressure_spawn_can_open_flashpoint_chain() {
    let mut world = World::new();
    let mut map = OverworldMap::default();
    let slot = 0usize;
    let rid = map
        .regions
        .iter()
        .find(|r| r.mission_slot == Some(slot))
        .map(|r| r.id)
        .expect("slot region");
    map.regions[rid].unrest = 96.0;
    map.regions[rid].control = 7.0;
    world.insert_resource(RunState { global_turn: 9 });
    world.insert_resource(map);
    world.insert_resource(MissionBoard::default());
    world.insert_resource(FlashpointState::default());
    spawn_test_missions(&mut world);
    let mut schedule = Schedule::default();
    schedule.add_systems(pressure_spawn_missions_system);
    schedule.run(&mut world);
    let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    let flashpoints = world.resource::<FlashpointState>();
    assert!(!flashpoints.chains.is_empty());
    assert!(flashpoints.chains.iter().any(|c| c.mission_slot == slot));
    let data = world.get::<MissionData>(board_entities[slot]).unwrap();
    assert!(data.mission_name.contains("Flashpoint 1/3"));
}

fn flashpoint_test_factions(map: &OverworldMap, slot: usize) -> (usize, usize, usize) {
    let rid = map
        .regions
        .iter()
        .find(|r| r.mission_slot == Some(slot))
        .map(|r| r.id)
        .expect("slot region");
    let defender = map.regions[rid].owner_faction_id;
    let attacker = map.regions[rid]
        .neighbors
        .iter()
        .find_map(|n| {
            let f = map.regions[*n].owner_faction_id;
            if f != defender { Some(f) } else { None }
        })
        .expect("attacker");
    (rid, defender, attacker)
}

#[test]
fn flashpoint_stage_victory_promotes_next_stage() {
    let mut world = World::new();
    let map = OverworldMap::default();
    let slot = 0usize;
    let (rid, defender, attacker) = flashpoint_test_factions(&map, slot);
    world.insert_resource(RunState { global_turn: 12 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(map);
    world.insert_resource(CampaignRoster::default());
    world.insert_resource(FlashpointState {
        chains: vec![FlashpointChain {
            id: 1, mission_slot: slot, region_id: rid,
            attacker_faction_id: attacker, defender_faction_id: defender,
            stage: 1, completed: false, companion_hook_hero_id: None,
            intent: FlashpointIntent::StealthPush, objective: String::new(),
        }],
        next_id: 2, notice: String::new(),
    });
    spawn_test_missions(&mut world);
    let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    {
        let mut data = world.get_mut::<MissionData>(entities[slot]).unwrap();
        data.mission_name = "Flashpoint 1/3 [Recon Sweep]".to_string();
        data.bound_region_id = Some(rid);
    }
    {
        let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
        progress.mission_active = false;
        progress.result = MissionResult::Victory;
        progress.outcome_recorded = true;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(flashpoint_progression_system);
    schedule.run(&mut world);
    let flashpoints = world.resource::<FlashpointState>();
    assert_eq!(flashpoints.chains[0].stage, 2);
    let progress = world.get::<MissionProgress>(entities[slot]).unwrap();
    assert_eq!(progress.result, MissionResult::InProgress);
    let data = world.get::<MissionData>(entities[slot]).unwrap();
    assert!(data.mission_name.contains("Flashpoint 2/3"));
}

#[test]
fn flashpoint_stage_hook_applies_companion_homefront_tuning() {
    let mut world = World::new();
    let map = OverworldMap::default();
    let slot = 0usize;
    let (rid, defender, attacker) = flashpoint_test_factions(&map, slot);
    let mut roster = CampaignRoster::default();
    let hero_id = roster.heroes[0].id;
    roster.heroes[0].origin_region_id = rid;
    world.insert_resource(RunState { global_turn: 13 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(map);
    world.insert_resource(roster);
    world.insert_resource(FlashpointState {
        chains: vec![FlashpointChain {
            id: 3, mission_slot: slot, region_id: rid,
            attacker_faction_id: attacker, defender_faction_id: defender,
            stage: 1, completed: false, companion_hook_hero_id: Some(hero_id),
            intent: FlashpointIntent::StealthPush, objective: String::new(),
        }],
        next_id: 4, notice: String::new(),
    });
    spawn_test_missions(&mut world);
    let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    world.get_mut::<MissionData>(entities[slot]).unwrap().bound_region_id = Some(rid);
    {
        let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
        progress.mission_active = false;
        progress.result = MissionResult::Victory;
        progress.outcome_recorded = true;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(flashpoint_progression_system);
    schedule.run(&mut world);
    let data = world.get::<MissionData>(entities[slot]).unwrap();
    assert!(data.mission_name.contains("obj="));
    assert!(!data.mission_name.contains("No companion hook"));
}

#[test]
fn decisive_flashpoint_victory_shifts_border_and_unlocks_recruit() {
    let mut world = World::new();
    let map = OverworldMap::default();
    let slot = 0usize;
    let (rid, defender, attacker) = flashpoint_test_factions(&map, slot);
    let mut roster = CampaignRoster::default();
    roster.recruit_pool.clear();
    world.insert_resource(RunState { global_turn: 18 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(map);
    world.insert_resource(roster);
    world.insert_resource(FlashpointState {
        chains: vec![FlashpointChain {
            id: 8, mission_slot: slot, region_id: rid,
            attacker_faction_id: attacker, defender_faction_id: defender,
            stage: 3, completed: false, companion_hook_hero_id: None,
            intent: FlashpointIntent::StealthPush, objective: String::new(),
        }],
        next_id: 9, notice: String::new(),
    });
    spawn_test_missions(&mut world);
    let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    world.get_mut::<MissionData>(entities[slot]).unwrap().bound_region_id = Some(rid);
    {
        let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
        progress.mission_active = false;
        progress.result = MissionResult::Victory;
        progress.outcome_recorded = true;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(flashpoint_progression_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    let roster = world.resource::<CampaignRoster>();
    let flashpoints = world.resource::<FlashpointState>();
    assert_eq!(map.regions[rid].owner_faction_id, attacker);
    assert!(roster.recruit_pool.iter().any(|r| r.origin_faction_id == attacker));
    assert!(flashpoints.chains.is_empty());
}

#[test]
fn flashpoint_victory_advances_hooked_companion_quest() {
    let mut world = World::new();
    let map = OverworldMap::default();
    let slot = 0usize;
    let (rid, defender, attacker) = flashpoint_test_factions(&map, slot);
    let mut roster = CampaignRoster::default();
    let hero_id = roster.heroes[0].id;
    roster.heroes[0].origin_region_id = rid;
    let mut story = CompanionStoryState::default();
    story.quests.push(CompanionQuest {
        id: 1, hero_id, kind: CompanionQuestKind::Homefront,
        status: CompanionQuestStatus::Active, title: "Test Quest".to_string(),
        objective: "Do a thing".to_string(), progress: 0, target: 2,
        issued_turn: 1, reward_loyalty: 1.0, reward_resolve: 1.0,
    });
    world.insert_resource(RunState { global_turn: 40 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(map);
    world.insert_resource(roster);
    world.insert_resource(story);
    world.insert_resource(FlashpointState {
        chains: vec![FlashpointChain {
            id: 44, mission_slot: slot, region_id: rid,
            attacker_faction_id: attacker, defender_faction_id: defender,
            stage: 3, completed: false, companion_hook_hero_id: Some(hero_id),
            intent: FlashpointIntent::StealthPush, objective: "hook objective".to_string(),
        }],
        next_id: 45, notice: String::new(),
    });
    spawn_test_missions(&mut world);
    let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    world.get_mut::<MissionData>(entities[slot]).unwrap().bound_region_id = Some(rid);
    {
        let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
        progress.mission_active = false;
        progress.result = MissionResult::Victory;
        progress.outcome_recorded = true;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(flashpoint_progression_system);
    schedule.run(&mut world);
    let story = world.resource::<CompanionStoryState>();
    assert_eq!(story.quests[0].progress, 1);
    assert!(story.notice.contains("Flashpoint beat advanced companion quest"));
}
