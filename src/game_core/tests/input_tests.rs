use bevy::prelude::*;

use super::helpers::*;
use super::super::attention_systems::*;
use super::super::diplomacy_systems::*;
use super::super::flashpoint_helpers::*;
use super::super::overworld_nav::*;
use super::super::overworld_types::*;
use super::super::roster_types::*;
use super::super::types::*;

// ─────────────────────────────────────────────────────────────────────────
// player_command_input_system
// ─────────────────────────────────────────────────────────────────────────

fn player_command_world() -> World {
    let mut world = World::new();
    world.init_resource::<MissionBoard>();
    spawn_test_missions(&mut world);
    set_active_progress(&mut world, |p| p.mission_active = true);
    world
}

#[test]
fn player_command_digit1_switches_to_balanced() {
    let mut world = player_command_world();
    set_active_tactics(&mut world, |t| t.tactical_mode = TacticalMode::Aggressive);
    run_player_command(&mut world, KeyCode::Digit1);
    assert_eq!(
        active_tactics(&mut world).tactical_mode,
        TacticalMode::Balanced
    );
}

#[test]
fn player_command_digit2_switches_to_aggressive() {
    let mut world = player_command_world();
    run_player_command(&mut world, KeyCode::Digit2);
    assert_eq!(
        active_tactics(&mut world).tactical_mode,
        TacticalMode::Aggressive
    );
}

#[test]
fn player_command_digit3_switches_to_defensive() {
    let mut world = player_command_world();
    run_player_command(&mut world, KeyCode::Digit3);
    assert_eq!(
        active_tactics(&mut world).tactical_mode,
        TacticalMode::Defensive
    );
}

#[test]
fn player_command_keyb_issues_breach_order() {
    let mut world = player_command_world();
    run_player_command(&mut world, KeyCode::KeyB);
    let t = active_tactics(&mut world);
    assert!(t.force_sabotage_order);
    assert_eq!(t.command_cooldown_turns, 3);
}

#[test]
fn player_command_keyr_issues_regroup_order() {
    let mut world = player_command_world();
    run_player_command(&mut world, KeyCode::KeyR);
    let t = active_tactics(&mut world);
    assert!(t.force_stabilize_order);
    assert_eq!(t.command_cooldown_turns, 3);
}

#[test]
fn player_command_keyb_blocked_by_cooldown() {
    let mut world = player_command_world();
    set_active_tactics(&mut world, |t| t.command_cooldown_turns = 1);
    run_player_command(&mut world, KeyCode::KeyB);
    assert!(!active_tactics(&mut world).force_sabotage_order);
}

#[test]
fn player_command_noop_when_mission_inactive() {
    let mut world = player_command_world();
    set_active_progress(&mut world, |p| p.mission_active = false);
    run_player_command(&mut world, KeyCode::Digit2);
    assert_eq!(
        active_tactics(&mut world).tactical_mode,
        TacticalMode::Balanced
    );
}

// ─────────────────────────────────────────────────────────────────────────
// focus_input_system
// ─────────────────────────────────────────────────────────────────────────

fn focus_app(attention: AttentionState) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.init_resource::<MissionBoard>();
    app.insert_resource(attention);
    app.add_systems(Update, focus_input_system);
    spawn_test_missions(&mut app.world);
    app
}

#[test]
fn focus_input_tab_advances_active_mission() {
    let mut app = focus_app(AttentionState::default());
    app.insert_resource(press(KeyCode::Tab));
    app.update();
    assert_eq!(board_active_idx(&mut app.world), 1);
}

#[test]
fn focus_input_bracket_right_advances_active_mission() {
    let mut app = focus_app(AttentionState::default());
    app.insert_resource(press(KeyCode::BracketRight));
    app.update();
    assert_eq!(board_active_idx(&mut app.world), 1);
}

#[test]
fn focus_input_bracket_left_retreats_active_mission() {
    let mut app = focus_app(AttentionState::default());
    let entities = app.world.resource::<MissionBoard>().entities.clone();
    app.world.entity_mut(entities[0]).remove::<ActiveMission>();
    app.world.entity_mut(entities[1]).insert(ActiveMission);
    app.insert_resource(press(KeyCode::BracketLeft));
    app.update();
    assert_eq!(board_active_idx(&mut app.world), 0);
}

#[test]
fn focus_input_blocked_by_switch_cooldown() {
    let mut app = focus_app(AttentionState {
        switch_cooldown_turns: 1,
        ..AttentionState::default()
    });
    app.insert_resource(press(KeyCode::Tab));
    app.update();
    assert_eq!(
        board_active_idx(&mut app.world),
        0,
        "cooldown should block focus shift"
    );
}

#[test]
fn focus_input_blocked_by_low_energy() {
    let mut app = focus_app(AttentionState {
        global_energy: 0.0,
        ..AttentionState::default()
    });
    app.insert_resource(press(KeyCode::Tab));
    app.update();
    assert_eq!(
        board_active_idx(&mut app.world),
        0,
        "insufficient energy should block focus shift"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// overworld_hub_input_system
// ─────────────────────────────────────────────────────────────────────────

fn overworld_hub_world() -> World {
    let mut world = World::new();
    world.init_resource::<MissionBoard>();
    world.insert_resource(OverworldMap::default());
    world.insert_resource(AttentionState::default());
    spawn_test_missions(&mut world);
    world
}

#[test]
fn overworld_hub_keyl_sets_selected_to_neighbor() {
    let mut world = overworld_hub_world();
    run_overworld_hub(&mut world, KeyCode::KeyL);
    let overworld = world.resource::<OverworldMap>();
    let current = overworld.current_region;
    assert!(
        overworld.regions[current]
            .neighbors
            .contains(&overworld.selected_region),
        "KeyL should set selected_region to a neighbor of current_region"
    );
}

#[test]
fn overworld_hub_keyj_sets_selected_to_neighbor() {
    let mut world = overworld_hub_world();
    run_overworld_hub(&mut world, KeyCode::KeyJ);
    let overworld = world.resource::<OverworldMap>();
    let current = overworld.current_region;
    assert!(
        overworld.regions[current]
            .neighbors
            .contains(&overworld.selected_region),
        "KeyJ should set selected_region to a neighbor of current_region"
    );
}

#[test]
fn overworld_hub_keyt_commits_travel_to_selected_region() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.init_resource::<MissionBoard>();
    app.add_systems(Update, overworld_hub_input_system);
    let mut overworld = OverworldMap::default();
    let current = overworld.current_region;
    let Some(&target) = overworld.regions[current].neighbors.first() else {
        return;
    };
    overworld.selected_region = target;
    overworld.travel_cooldown_turns = 0;
    app.insert_resource(overworld);
    app.insert_resource(AttentionState {
        global_energy: 9999.0,
        ..AttentionState::default()
    });
    spawn_test_missions(&mut app.world);
    app.insert_resource(press(KeyCode::KeyT));
    app.update();
    assert_eq!(
        app.world.resource::<OverworldMap>().current_region,
        target,
        "KeyT should commit travel and update current_region"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// interaction_offer_input_system
// ─────────────────────────────────────────────────────────────────────────

fn make_interaction_world() -> World {
    let mut world = World::new();
    world.init_resource::<MissionBoard>();
    world.insert_resource(AttentionState::default());
    world.insert_resource(CampaignRoster::default());
    world.insert_resource(DiplomacyState::default());
    world.insert_resource(InteractionBoard {
        offers: vec![
            InteractionOffer {
                id: 1, from_faction_id: 1, region_id: 0, mission_slot: None,
                kind: InteractionOfferKind::JointMission, summary: "Joint strike".into(),
            },
            InteractionOffer {
                id: 2, from_faction_id: 2, region_id: 1, mission_slot: None,
                kind: InteractionOfferKind::TrainingLoan, summary: "Training exchange".into(),
            },
            InteractionOffer {
                id: 3, from_faction_id: 1, region_id: 2, mission_slot: None,
                kind: InteractionOfferKind::RivalRaid, summary: "Raid support".into(),
            },
        ],
        selected: 0, notice: String::new(), next_offer_id: 4,
    });
    spawn_test_missions(&mut world);
    world
}

#[test]
fn interaction_offer_keyo_increments_selection() {
    let mut world = make_interaction_world();
    run_interaction_offer(&mut world, KeyCode::KeyO);
    assert_eq!(world.resource::<InteractionBoard>().selected, 1);
}

#[test]
fn interaction_offer_keyo_wraps_to_zero_at_end() {
    let mut world = make_interaction_world();
    world.resource_mut::<InteractionBoard>().selected = 2;
    run_interaction_offer(&mut world, KeyCode::KeyO);
    assert_eq!(world.resource::<InteractionBoard>().selected, 0);
}

#[test]
fn interaction_offer_keyu_wraps_to_last_from_zero() {
    let mut world = make_interaction_world();
    run_interaction_offer(&mut world, KeyCode::KeyU);
    assert_eq!(world.resource::<InteractionBoard>().selected, 2);
}

#[test]
fn interaction_offer_keyn_removes_selected_offer() {
    let mut world = make_interaction_world();
    world.resource_mut::<InteractionBoard>().selected = 1;
    run_interaction_offer(&mut world, KeyCode::KeyN);
    let board = world.resource::<InteractionBoard>();
    assert_eq!(board.offers.len(), 2, "rejected offer should be removed");
    assert!(
        board.selected < board.offers.len(),
        "selected should remain in bounds"
    );
}

#[test]
fn interaction_offer_empty_board_is_noop() {
    let mut world = make_interaction_world();
    world.resource_mut::<InteractionBoard>().offers.clear();
    run_interaction_offer(&mut world, KeyCode::KeyO);
    run_interaction_offer(&mut world, KeyCode::KeyN);
    let board = world.resource::<InteractionBoard>();
    assert_eq!(board.offers.len(), 0);
    assert_eq!(board.selected, 0);
}

#[test]
fn commander_intents_are_deterministic() {
    let mut world_a = World::new();
    world_a.insert_resource(OverworldMap::default());
    world_a.insert_resource(MissionBoard::default());
    world_a.insert_resource(DiplomacyState::default());
    world_a.insert_resource(CommanderState::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(generate_commander_intents_system);
    schedule.run(&mut world_a);
    let intents_a = world_a.resource::<CommanderState>().intents.clone();
    let mut world_b = World::new();
    world_b.insert_resource(OverworldMap::default());
    world_b.insert_resource(MissionBoard::default());
    world_b.insert_resource(DiplomacyState::default());
    world_b.insert_resource(CommanderState::default());
    let mut schedule_b = Schedule::default();
    schedule_b.add_systems(generate_commander_intents_system);
    schedule_b.run(&mut world_b);
    let intents_b = world_b.resource::<CommanderState>().intents.clone();
    assert_eq!(intents_a.len(), intents_b.len());
    for i in 0..intents_a.len() {
        assert_eq!(intents_a[i].faction_id, intents_b[i].faction_id);
        assert_eq!(intents_a[i].region_id, intents_b[i].region_id);
        assert_eq!(intents_a[i].mission_slot, intents_b[i].mission_slot);
        assert_eq!(intents_a[i].kind, intents_b[i].kind);
        assert_eq!(intents_a[i].urgency, intents_b[i].urgency);
    }
}

#[test]
fn interaction_offer_acceptance_changes_state() {
    let offer = InteractionOffer {
        id: 7, from_faction_id: 1, region_id: 1, mission_slot: Some(1),
        kind: InteractionOfferKind::JointMission, summary: "test".to_string(),
    };
    let mut missions = default_mission_snapshots();
    missions[1].alert_level = 44.0;
    let mut attention = AttentionState::default();
    let mut roster = CampaignRoster::default();
    let mut diplomacy = DiplomacyState::default();
    let msg = resolve_interaction_offer(
        &offer, true, &mut missions, &mut attention, &mut roster, &mut diplomacy,
    );
    assert!(msg.contains("Joint mission accepted"));
    assert!(missions[1].alert_level < 44.0);
    assert!(diplomacy.relations[0][1] > 10);
}

#[test]
fn flashpoint_intent_input_updates_stage_profile_and_telemetry() {
    let mut world = World::new();
    let map = OverworldMap::default();
    let slot = 0usize;
    let rid = map
        .regions.iter().find(|r| r.mission_slot == Some(slot)).map(|r| r.id).expect("slot region");
    let defender = map.regions[rid].owner_faction_id;
    let attacker = map.regions[rid].neighbors.iter()
        .find_map(|n| { let f = map.regions[*n].owner_faction_id; if f != defender { Some(f) } else { None } })
        .expect("attacker");
    let chain = FlashpointChain {
        id: 21, mission_slot: slot, region_id: rid,
        attacker_faction_id: attacker, defender_faction_id: defender,
        stage: 2, completed: false, companion_hook_hero_id: None,
        intent: FlashpointIntent::StealthPush, objective: String::new(),
    };
    world.insert_resource(RunState { global_turn: 33 });
    world.insert_resource(MissionBoard::default());
    world.insert_resource(map);
    world.insert_resource(FlashpointState {
        chains: vec![chain.clone()], next_id: 22, notice: String::new(),
    });
    let mut keyboard = ButtonInput::<KeyCode>::default();
    keyboard.press(KeyCode::Digit2);
    world.insert_resource(keyboard);
    spawn_test_missions(&mut world);
    let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    {
        let data_ref = world.get::<MissionData>(entities[slot]).unwrap();
        let progress_ref = world.get::<MissionProgress>(entities[slot]).unwrap();
        let tactics_ref = world.get::<MissionTactics>(entities[slot]).unwrap();
        let mut snap = MissionSnapshot::from_components(data_ref, progress_ref, tactics_ref);
        let overworld = world.resource::<OverworldMap>();
        configure_flashpoint_stage_mission(&mut snap, &chain, overworld, 77);
        rewrite_flashpoint_mission_name(&mut snap, &chain, overworld, None);
        let id = data_ref.id;
        drop(data_ref); drop(progress_ref); drop(tactics_ref); drop(overworld);
        let (new_data, new_progress, new_tactics) = snap.into_components(id);
        *world.get_mut::<MissionData>(entities[slot]).unwrap() = new_data;
        *world.get_mut::<MissionProgress>(entities[slot]).unwrap() = new_progress;
        *world.get_mut::<MissionTactics>(entities[slot]).unwrap() = new_tactics;
    }
    let before_alert = world.get::<MissionProgress>(entities[slot]).unwrap().alert_level;
    let mut schedule = Schedule::default();
    schedule.add_systems(flashpoint_intent_input_system);
    schedule.run(&mut world);
    let chain_result = &world.resource::<FlashpointState>().chains[0];
    assert_eq!(chain_result.intent, FlashpointIntent::DirectAssault);
    let data = world.get::<MissionData>(entities[slot]).unwrap();
    assert!(data.mission_name.contains("Direct Assault"));
    assert!(data.mission_name.contains("win=>"));
    let progress = world.get::<MissionProgress>(entities[slot]).unwrap();
    assert!(progress.alert_level > before_alert);
}
