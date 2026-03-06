use super::*;

#[test]
fn load_restores_region_layer_context_and_selected_party() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-region-resume-{}.json", id);
    let mut roster = game_core::CampaignRoster::default();
    let mut parties =
        game_core::bootstrap_campaign_parties(&roster, &game_core::OverworldMap::default());
    let delegated_id = parties
        .parties
        .iter()
        .find(|p| !p.is_player_controlled)
        .map(|p| p.id)
        .expect("delegated party exists");
    parties.selected_party_id = Some(delegated_id);
    if let Some(player) = parties.parties.iter().find(|p| p.id == delegated_id) {
        roster.player_hero_id = Some(player.leader_hero_id);
    }

    let mut save = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state: RunState { global_turn: 44 },
        mission_map: game_core::MissionMap::default(),
        attention_state: game_core::AttentionState::default(),
        overworld_map: game_core::OverworldMap::default(),
        commander_state: game_core::CommanderState::default(),
        diplomacy_state: game_core::DiplomacyState::default(),
        interaction_board: game_core::InteractionBoard::default(),
        campaign_roster: roster,
        campaign_parties: parties,
        campaign_ledger: game_core::CampaignLedger::default(),
        campaign_event_log: game_core::CampaignEventLog::default(),
        companion_story_state: game_core::CompanionStoryState::default(),
        flashpoint_state: game_core::FlashpointState::default(),
        character_creation: CharacterCreationState {
            selected_faction_id: Some("faction-0-test".to_string()),
            selected_faction_index: Some(0),
            selected_backstory_id: Some("scout".to_string()),
            stat_modifiers: vec!["+2 scouting".to_string()],
            recruit_bias_modifiers: vec!["rangers".to_string()],
            is_confirmed: true,
        },
        campaign_progress: CampaignProgressState::default(),
        mission_snapshots: game_core::default_mission_snapshots(),
        active_mission_id: None,
    };
    let payload = RegionTransitionPayload {
        region_id: 2,
        faction_id: "faction-0-test".to_string(),
        faction_index: 0,
        campaign_seed: save.overworld_map.map_seed,
        region_seed: derive_region_transition_seed(save.overworld_map.map_seed, 2, 0),
    };
    save.campaign_progress = CampaignProgressState {
        current_layer: CampaignLayerMarker::Region,
        current_region_id: Some(2),
        local_scene_id: None,
        intro_completed: false,
        region_payload: Some(payload),
        local_source_region_id: None,
    };
    save_campaign_data(&path, &save).expect("write save");

    let mut world = build_layer_resume_test_world();
    let msg = load_campaign_from_path_into_world(&mut world, "slot1", &path).expect("load");
    assert!(msg.contains("resumed Region layer"));
    assert_eq!(
        world.resource::<HubUiState>().screen,
        HubScreen::RegionView
    );
    assert_eq!(
        world
            .resource::<RegionLayerTransitionState>()
            .active_payload
            .as_ref()
            .map(|p| p.region_id),
        Some(2)
    );
    assert_eq!(
        world
            .resource::<game_core::CampaignParties>()
            .selected_party_id,
        Some(delegated_id)
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_restores_menu_layer_from_progress_marker() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-menu-resume-{}.json", id);
    let mut save = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state: RunState { global_turn: 8 },
        mission_map: game_core::MissionMap::default(),
        attention_state: game_core::AttentionState::default(),
        overworld_map: game_core::OverworldMap::default(),
        commander_state: game_core::CommanderState::default(),
        diplomacy_state: game_core::DiplomacyState::default(),
        interaction_board: game_core::InteractionBoard::default(),
        campaign_roster: game_core::CampaignRoster::default(),
        campaign_parties: game_core::CampaignParties::default(),
        campaign_ledger: game_core::CampaignLedger::default(),
        campaign_event_log: game_core::CampaignEventLog::default(),
        companion_story_state: game_core::CompanionStoryState::default(),
        flashpoint_state: game_core::FlashpointState::default(),
        character_creation: CharacterCreationState::default(),
        campaign_progress: CampaignProgressState::default(),
        mission_snapshots: game_core::default_mission_snapshots(),
        active_mission_id: None,
    };
    save.campaign_progress.current_layer = CampaignLayerMarker::Menu;
    save_campaign_data(&path, &save).expect("write save");

    let mut world = build_layer_resume_test_world();
    let msg = load_campaign_from_path_into_world(&mut world, "slot1", &path).expect("load");
    assert!(msg.contains("resumed Start Menu"));
    assert_eq!(
        world.resource::<HubUiState>().screen,
        HubScreen::StartMenu
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_restores_local_layer_with_intro_completion_flag() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-local-resume-{}.json", id);
    let mut save = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state: RunState { global_turn: 51 },
        mission_map: game_core::MissionMap::default(),
        attention_state: game_core::AttentionState::default(),
        overworld_map: game_core::OverworldMap::default(),
        commander_state: game_core::CommanderState::default(),
        diplomacy_state: game_core::DiplomacyState::default(),
        interaction_board: game_core::InteractionBoard::default(),
        campaign_roster: game_core::CampaignRoster::default(),
        campaign_parties: game_core::CampaignParties::default(),
        campaign_ledger: game_core::CampaignLedger::default(),
        campaign_event_log: game_core::CampaignEventLog::default(),
        companion_story_state: game_core::CompanionStoryState::default(),
        flashpoint_state: game_core::FlashpointState::default(),
        character_creation: CharacterCreationState {
            selected_faction_id: Some("faction-0-test".to_string()),
            selected_faction_index: Some(0),
            selected_backstory_id: Some("scout".to_string()),
            stat_modifiers: vec![],
            recruit_bias_modifiers: vec![],
            is_confirmed: true,
        },
        campaign_progress: CampaignProgressState::default(),
        mission_snapshots: game_core::default_mission_snapshots(),
        active_mission_id: None,
    };
    let payload = RegionTransitionPayload {
        region_id: 3,
        faction_id: "faction-0-test".to_string(),
        faction_index: 0,
        campaign_seed: save.overworld_map.map_seed,
        region_seed: derive_region_transition_seed(save.overworld_map.map_seed, 3, 0),
    };
    save.campaign_progress = CampaignProgressState {
        current_layer: CampaignLayerMarker::Local,
        current_region_id: Some(3),
        local_scene_id: Some("local-eagle-eye-intro".to_string()),
        intro_completed: true,
        region_payload: Some(payload),
        local_source_region_id: Some(3),
    };
    save_campaign_data(&path, &save).expect("write save");

    let mut world = build_layer_resume_test_world();
    let msg = load_campaign_from_path_into_world(&mut world, "slot1", &path).expect("load");
    assert!(msg.contains("resumed Local layer"));
    assert_eq!(
        world.resource::<HubUiState>().screen,
        HubScreen::LocalEagleEyeIntro
    );
    assert!(
        world
            .resource::<LocalEagleEyeIntroState>()
            .intro_completed
    );
    assert!(
        world
            .resource::<LocalEagleEyeIntroState>()
            .input_handoff_ready
    );
    assert_eq!(
        world
            .resource::<LocalEagleEyeIntroState>()
            .source_region_id,
        Some(3)
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_incompatible_newer_version_rejects_without_side_effects() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-incompatible-version-{}.json", id);
    let mut save = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION + 1,
        run_state: RunState { global_turn: 22 },
        mission_map: game_core::MissionMap::default(),
        attention_state: game_core::AttentionState::default(),
        overworld_map: game_core::OverworldMap::default(),
        commander_state: game_core::CommanderState::default(),
        diplomacy_state: game_core::DiplomacyState::default(),
        interaction_board: game_core::InteractionBoard::default(),
        campaign_roster: game_core::CampaignRoster::default(),
        campaign_parties: game_core::CampaignParties::default(),
        campaign_ledger: game_core::CampaignLedger::default(),
        campaign_event_log: game_core::CampaignEventLog::default(),
        companion_story_state: game_core::CompanionStoryState::default(),
        flashpoint_state: game_core::FlashpointState::default(),
        character_creation: CharacterCreationState::default(),
        campaign_progress: CampaignProgressState::default(),
        mission_snapshots: game_core::default_mission_snapshots(),
        active_mission_id: None,
    };
    save.campaign_progress.current_layer = CampaignLayerMarker::Region;
    save_campaign_data(&path, &save).expect("write save");

    let mut world = build_layer_resume_test_world();
    world.insert_resource(RunState { global_turn: 91 });
    {
        let mut hub = world.resource_mut::<HubUiState>();
        hub.screen = HubScreen::StartMenu;
    }
    let result = load_campaign_from_path_into_world(&mut world, "slot1", &path);
    assert!(result.is_err());
    assert!(result
        .err()
        .expect("error")
        .contains("incompatible save: version"));
    assert_eq!(world.resource::<RunState>().global_turn, 91);
    assert_eq!(
        world.resource::<HubUiState>().screen,
        HubScreen::StartMenu
    );
    let _ = std::fs::remove_file(&path);
}

