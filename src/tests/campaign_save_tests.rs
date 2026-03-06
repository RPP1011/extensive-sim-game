use super::*;

#[test]
fn campaign_save_data_json_roundtrip_preserves_state() {
    let mut run_state = RunState::default();
    run_state.global_turn = 17;
    let mut story = game_core::CompanionStoryState::default();
    story.notice = "test".to_string();
    let mut snapshots = game_core::default_mission_snapshots();
    if !snapshots.is_empty() {
        snapshots[0].alert_level = 77.0;
    }
    let creation = CharacterCreationState {
        selected_faction_id: Some("faction-x".to_string()),
        selected_faction_index: Some(1),
        selected_backstory_id: None,
        stat_modifiers: Vec::new(),
        recruit_bias_modifiers: Vec::new(),
        is_confirmed: true,
    };

    let data = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state,
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
        companion_story_state: story,
        flashpoint_state: game_core::FlashpointState::default(),
        character_creation: creation,
        campaign_progress: CampaignProgressState::default(),
        mission_snapshots: snapshots,
        active_mission_id: Some(0),
    };

    let json = serde_json::to_string(&data).expect("serialize");
    let decoded: CampaignSaveData = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded.run_state.global_turn, 17);
    assert_eq!(
        decoded.mission_snapshots.first().map(|m| m.alert_level),
        Some(77.0)
    );
    assert_eq!(decoded.companion_story_state.notice, "test");
    assert_eq!(
        decoded.character_creation.selected_faction_id.as_deref(),
        Some("faction-x")
    );
    assert_eq!(decoded.character_creation.selected_faction_index, Some(1));
}

#[test]
fn campaign_save_file_io_roundtrip_works() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-save-{}.json", id);
    let mut run_state = RunState::default();
    run_state.global_turn = 3;
    let data = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state,
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

    save_campaign_data(&path, &data).expect("save file");
    let loaded = load_campaign_data(&path).expect("load file");
    assert_eq!(loaded.run_state.global_turn, 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn campaign_save_load_roundtrip_preserves_party_order_and_target() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-party-roundtrip-{}.json", id);
    let mut run_state = RunState::default();
    run_state.global_turn = 12;
    let mut data = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state,
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
    data.campaign_parties.parties = vec![
        game_core::CampaignParty {
            id: 1,
            name: "Main Company".to_string(),
            leader_hero_id: 100,
            region_id: 0,
            supply: 90.0,
            speed: 1.0,
            is_player_controlled: true,
            order: game_core::PartyOrderKind::HoldPosition,
            order_target_region_id: None,
        },
        game_core::CampaignParty {
            id: 2,
            name: "Ranging Band".to_string(),
            leader_hero_id: 101,
            region_id: 1,
            supply: 88.0,
            speed: 0.9,
            is_player_controlled: false,
            order: game_core::PartyOrderKind::PatrolNearby,
            order_target_region_id: Some(2),
        },
    ];
    data.campaign_parties.selected_party_id = Some(1);
    data.campaign_parties.next_id = 3;
    data.campaign_parties.notice = "Delegated party now patrolling Region-B".to_string();

    save_campaign_data(&path, &data).expect("save file");
    let loaded = load_campaign_data(&path).expect("load file");
    let delegated = loaded
        .campaign_parties
        .parties
        .iter()
        .find(|p| p.id == 2)
        .expect("delegated party");
    assert_eq!(loaded.campaign_parties.selected_party_id, Some(1));
    assert_eq!(delegated.order, game_core::PartyOrderKind::PatrolNearby);
    assert_eq!(delegated.order_target_region_id, Some(2));
    assert_eq!(
        loaded.campaign_parties.notice,
        "Delegated party now patrolling Region-B"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn loading_missing_party_field_fails_and_does_not_mutate_world() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-party-missing-field-{}.json", id);
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 77 });

    let mut data = CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state: RunState { global_turn: 15 },
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
    data.campaign_parties
        .parties
        .push(game_core::CampaignParty {
            id: 1,
            name: "Ranging Band".to_string(),
            leader_hero_id: 101,
            region_id: 1,
            supply: 88.0,
            speed: 0.9,
            is_player_controlled: false,
            order: game_core::PartyOrderKind::PatrolNearby,
            order_target_region_id: Some(2),
        });
    data.campaign_parties.selected_party_id = Some(1);
    data.campaign_parties.next_id = 2;
    data.campaign_parties.notice = "delegated".to_string();

    let mut raw = serde_json::to_value(&data).expect("to value");
    raw["campaign_parties"]["parties"][0]
        .as_object_mut()
        .expect("party object")
        .remove("order");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&raw).expect("serialize invalid"),
    )
    .expect("write invalid save");

    let result = load_campaign_from_path_into_world(&mut world, "slot1", &path);
    assert!(result.is_err());
    assert!(result
        .err()
        .expect("error")
        .contains("missing campaign_parties.parties[0].order"));
    assert_eq!(world.resource::<RunState>().global_turn, 77);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn snapshot_includes_layer_marker_context_and_intro_flag() {
    let mut world = build_layer_resume_test_world();
    world.insert_resource(CharacterCreationState {
        selected_faction_id: Some("faction-0-test".to_string()),
        selected_faction_index: Some(0),
        selected_backstory_id: Some("scout".to_string()),
        stat_modifiers: vec!["+2 scouting".to_string()],
        recruit_bias_modifiers: vec!["rangers".to_string()],
        is_confirmed: true,
    });
    world.insert_resource(game_core::CampaignParties {
        parties: vec![
            game_core::CampaignParty {
                id: 1,
                name: "Main Company".to_string(),
                leader_hero_id: 10,
                region_id: 0,
                supply: 100.0,
                speed: 1.0,
                is_player_controlled: true,
                order: game_core::PartyOrderKind::HoldPosition,
                order_target_region_id: None,
            },
            game_core::CampaignParty {
                id: 2,
                name: "Scout Wing".to_string(),
                leader_hero_id: 11,
                region_id: 3,
                supply: 95.0,
                speed: 1.1,
                is_player_controlled: false,
                order: game_core::PartyOrderKind::PatrolNearby,
                order_target_region_id: Some(4),
            },
        ],
        selected_party_id: Some(2),
        next_id: 3,
        notice: "scouts in position".to_string(),
    });
    let mut hub_ui = world.resource_mut::<HubUiState>();
    hub_ui.screen = HubScreen::LocalEagleEyeIntro;
    drop(hub_ui);
    let payload = RegionTransitionPayload {
        region_id: 3,
        faction_id: "faction-0-test".to_string(),
        faction_index: 0,
        campaign_seed: world.resource::<game_core::OverworldMap>().map_seed,
        region_seed: derive_region_transition_seed(
            world.resource::<game_core::OverworldMap>().map_seed,
            3,
            0,
        ),
    };
    world.insert_resource(RegionLayerTransitionState {
        active_payload: Some(payload),
        pending_payload: None,
        pending_frames: 0,
        interaction_locked: false,
        status: "region active".to_string(),
    });
    world.insert_resource(LocalEagleEyeIntroState {
        source_region_id: Some(3),
        anchor: local_intro_anchor_for_region(3),
        phase: LocalIntroPhase::GameplayControl,
        phase_frames: 0,
        intro_completed: true,
        input_handoff_ready: true,
        status: "intro complete".to_string(),
    });

    let snapshot = snapshot_campaign_from_world(&world);
    assert_eq!(
        snapshot.campaign_progress.current_layer,
        CampaignLayerMarker::Local
    );
    assert_eq!(snapshot.campaign_progress.current_region_id, Some(3));
    assert_eq!(
        snapshot.campaign_progress.local_scene_id.as_deref(),
        Some("local-eagle-eye-intro")
    );
    assert!(snapshot.campaign_progress.intro_completed);
    assert_eq!(snapshot.campaign_parties.selected_party_id, Some(2));
    assert_eq!(
        snapshot.character_creation.selected_faction_id.as_deref(),
        Some("faction-0-test")
    );
}

