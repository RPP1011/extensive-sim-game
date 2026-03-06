use super::*;

#[test]
fn campaign_slot_path_resolves_expected_files() {
    assert_eq!(campaign_slot_path(1), CAMPAIGN_SAVE_PATH);
    assert_eq!(campaign_slot_path(2), CAMPAIGN_SAVE_SLOT_2_PATH);
    assert_eq!(campaign_slot_path(3), CAMPAIGN_SAVE_SLOT_3_PATH);
    assert_eq!(campaign_slot_path(9), CAMPAIGN_SAVE_PATH);
}

#[test]
fn save_index_upsert_replaces_same_slot() {
    let mut index = CampaignSaveIndex::default();
    let a = SaveSlotMetadata {
        slot: "slot1".to_string(),
        path: "a".to_string(),
        save_version: 2,
        compatible: true,
        global_turn: 2,
        map_seed: 11,
        saved_unix_seconds: 1,
    };
    let b = SaveSlotMetadata {
        slot: "slot1".to_string(),
        path: "b".to_string(),
        save_version: 2,
        compatible: true,
        global_turn: 5,
        map_seed: 12,
        saved_unix_seconds: 2,
    };
    upsert_slot_metadata(&mut index, 1, a);
    upsert_slot_metadata(&mut index, 1, b.clone());
    assert_eq!(index.slots.len(), 1);
    assert_eq!(index.slots[0].path, b.path);
    assert_eq!(index.slots[0].global_turn, b.global_turn);
}

#[test]
fn save_index_file_roundtrip_works() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-save-index-{}.json", id);
    let index = CampaignSaveIndex {
        slots: vec![SaveSlotMetadata {
            slot: "slot1".to_string(),
            path: "slot1.json".to_string(),
            save_version: 2,
            compatible: true,
            global_turn: 10,
            map_seed: 42,
            saved_unix_seconds: 99,
        }],
        autosave: None,
    };
    let body = serde_json::to_string_pretty(&index).expect("serialize");
    std::fs::write(&path, body).expect("write");
    let loaded_text = std::fs::read_to_string(&path).expect("read");
    let loaded: CampaignSaveIndex = serde_json::from_str(&loaded_text).expect("parse");
    assert_eq!(loaded.slots.len(), 1);
    assert_eq!(loaded.slots[0].global_turn, 10);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn slot_badge_and_preview_reflect_compatibility() {
    let meta = SaveSlotMetadata {
        slot: "slot1".to_string(),
        path: "slot1.json".to_string(),
        save_version: CURRENT_SAVE_VERSION,
        compatible: true,
        global_turn: 21,
        map_seed: 99,
        saved_unix_seconds: 123,
    };
    assert_eq!(format_slot_badge(Some(&meta)), "[OK]");
    let preview = build_save_preview(&meta);
    assert!(preview.contains("turn=21"));
    assert!(preview.contains("version="));
}

#[test]
fn panel_selected_entry_maps_selected_index() {
    let mut index = CampaignSaveIndex::default();
    upsert_slot_metadata(
        &mut index,
        2,
        SaveSlotMetadata {
            slot: "slot2".to_string(),
            path: CAMPAIGN_SAVE_SLOT_2_PATH.to_string(),
            save_version: CURRENT_SAVE_VERSION,
            compatible: true,
            global_turn: 7,
            map_seed: 2,
            saved_unix_seconds: 1,
        },
    );
    let panel = CampaignSavePanelState {
        open: true,
        selected: 1,
        pending_load_path: None,
        pending_label: None,
        preview: String::new(),
    };
    let (label, path, meta) = panel_selected_entry(&panel, &index);
    assert_eq!(label, "slot2");
    assert_eq!(path, CAMPAIGN_SAVE_SLOT_2_PATH);
    assert!(meta.is_some());
}

#[test]
fn continue_candidates_use_latest_compatible_entries() {
    let index = CampaignSaveIndex {
        slots: vec![
            SaveSlotMetadata {
                slot: "slot1".to_string(),
                path: "slot1.json".to_string(),
                save_version: CURRENT_SAVE_VERSION,
                compatible: true,
                global_turn: 5,
                map_seed: 1,
                saved_unix_seconds: 10,
            },
            SaveSlotMetadata {
                slot: "slot2".to_string(),
                path: "slot2.json".to_string(),
                save_version: CURRENT_SAVE_VERSION + 1,
                compatible: false,
                global_turn: 8,
                map_seed: 2,
                saved_unix_seconds: 30,
            },
        ],
        autosave: Some(SaveSlotMetadata {
            slot: "autosave".to_string(),
            path: "autosave.json".to_string(),
            save_version: CURRENT_SAVE_VERSION,
            compatible: true,
            global_turn: 6,
            map_seed: 3,
            saved_unix_seconds: 20,
        }),
    };

    let candidates = continue_campaign_candidates(&index);
    assert!(candidates
        .iter()
        .any(|c| c.label == "autosave" && c.path == "autosave.json"));
    assert!(candidates
        .iter()
        .any(|c| c.label == "slot1" && c.path == "slot1.json"));
    assert!(!candidates.iter().any(|c| c.path == "slot2.json"));
}

#[test]
fn continue_request_loads_latest_compatible_save_into_overworld() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let slot_path = format!("/tmp/game-campaign-continue-slot1-{}.json", id);
    let autosave_path = format!("/tmp/game-campaign-continue-autosave-{}.json", id);

    let save_data = |turn: u32| CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state: RunState { global_turn: turn },
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
    save_campaign_data(&slot_path, &save_data(7)).expect("write slot");
    save_campaign_data(&autosave_path, &save_data(19)).expect("write autosave");

    let mut world = World::new();
    world.insert_resource(RunState::default());
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(HubUiState {
        screen: HubScreen::StartMenu,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: true,
    });
    world.insert_resource(StartMenuState::default());
    world.insert_resource(CampaignSaveNotice::default());
    world.insert_resource(CampaignSaveIndexState {
        index: CampaignSaveIndex {
            slots: vec![SaveSlotMetadata {
                slot: "slot1".to_string(),
                path: slot_path.clone(),
                save_version: CURRENT_SAVE_VERSION,
                compatible: true,
                global_turn: 7,
                map_seed: 1,
                saved_unix_seconds: 10,
            }],
            autosave: Some(SaveSlotMetadata {
                slot: "autosave".to_string(),
                path: autosave_path.clone(),
                save_version: CURRENT_SAVE_VERSION,
                compatible: true,
                global_turn: 19,
                map_seed: 2,
                saved_unix_seconds: 20,
            }),
        },
    });

    hub_continue_campaign_requested_system(&mut world);

    let hub_ui = world.resource::<HubUiState>();
    assert!(hub_ui.screen == HubScreen::OverworldMap);
    assert!(!hub_ui.request_continue_campaign);
    assert_eq!(world.resource::<RunState>().global_turn, 19);
    assert!(world
        .resource::<CampaignSaveNotice>()
        .message
        .contains("Loaded autosave"));

    let _ = std::fs::remove_file(&slot_path);
    let _ = std::fs::remove_file(&autosave_path);
}

#[test]
fn continue_request_failure_keeps_start_menu_with_status() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let missing_path = format!("/tmp/game-campaign-continue-missing-{}.json", id);
    let mut world = World::new();
    world.insert_resource(HubUiState {
        screen: HubScreen::StartMenu,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: true,
    });
    world.insert_resource(StartMenuState::default());
    world.insert_resource(CampaignSaveNotice::default());
    world.insert_resource(CampaignSaveIndexState {
        index: CampaignSaveIndex {
            slots: vec![SaveSlotMetadata {
                slot: "slot1".to_string(),
                path: missing_path,
                save_version: CURRENT_SAVE_VERSION,
                compatible: true,
                global_turn: 0,
                map_seed: 0,
                saved_unix_seconds: 1,
            }],
            autosave: None,
        },
    });

    hub_continue_campaign_requested_system(&mut world);

    let hub_ui = world.resource::<HubUiState>();
    assert!(hub_ui.screen == HubScreen::StartMenu);
    assert!(!hub_ui.request_continue_campaign);
    let message = &world.resource::<CampaignSaveNotice>().message;
    assert!(message.contains("Continue failed."));
    assert!(message.contains("Attempted: slot1"));
    let start_menu = world.resource::<StartMenuState>();
    assert!(start_menu.status.contains("Continue failed."));
}

#[test]
fn migration_promotes_v1_save_to_current_version() {
    let legacy_json = serde_json::json!({
        "run_state": { "global_turn": 8 },
        "attention_state": game_core::AttentionState::default(),
        "overworld_map": game_core::OverworldMap::default(),
        "commander_state": game_core::CommanderState::default(),
        "diplomacy_state": game_core::DiplomacyState::default(),
        "interaction_board": game_core::InteractionBoard::default(),
        "campaign_roster": game_core::CampaignRoster::default(),
        "campaign_ledger": game_core::CampaignLedger::default(),
        "companion_story_state": game_core::CompanionStoryState::default()
    });
    let parsed: CampaignSaveData = serde_json::from_value(legacy_json).expect("legacy parse");
    assert_eq!(parsed.save_version, SAVE_VERSION_V1);
    let migrated = migrate_campaign_save_data(parsed).expect("migrate");
    assert_eq!(migrated.save_version, CURRENT_SAVE_VERSION);
}

#[test]
fn migration_rejects_newer_unknown_version() {
    let mut world = World::new();
    world.insert_resource(RunState::default());
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    let mut data = snapshot_campaign_from_world(&world);
    data.save_version = CURRENT_SAVE_VERSION + 1;
    assert!(migrate_campaign_save_data(data).is_err());
}

#[test]
fn migration_keeps_current_version_unchanged() {
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 9 });
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    let data = snapshot_campaign_from_world(&world);
    let migrated = migrate_campaign_save_data(data.clone()).expect("migrate");
    assert_eq!(migrated.save_version, CURRENT_SAVE_VERSION);
    assert_eq!(migrated.run_state.global_turn, data.run_state.global_turn);
}

#[test]
fn save_overwrite_creates_backup_file() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = format!("/tmp/game-campaign-save-backup-{}.json", id);
    let backup_path = format!("{}.bak", path);
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 1 });
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());

    let mut first = snapshot_campaign_from_world(&world);
    first.run_state.global_turn = 2;
    save_campaign_data(&path, &first).expect("first save");

    let mut second = snapshot_campaign_from_world(&world);
    second.run_state.global_turn = 3;
    save_campaign_data(&path, &second).expect("second save");

    assert!(std::path::Path::new(&backup_path).exists());
    let backup_loaded = load_campaign_data(&backup_path).expect("load backup");
    assert_eq!(backup_loaded.run_state.global_turn, 2);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&backup_path);
}

#[test]
fn validation_repair_clears_invalid_region_binding() {
    let mut world = World::new();
    world.insert_resource(RunState::default());
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    spawn_mission_entities_from_snapshots(
        &mut world,
        game_core::default_mission_snapshots(),
        None,
    );
    let mut data = snapshot_campaign_from_world(&world);
    // Inject an invalid region binding into the first mission
    if let Some(snap) = data.mission_snapshots.first_mut() {
        snap.bound_region_id = Some(9999);
    }
    let warnings = validate_and_repair_loaded_campaign(&mut data);
    assert!(warnings
        .iter()
        .any(|w| w.contains("invalid region binding")));
}

#[test]
fn snapshot_save_load_pipeline_keeps_current_version() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let autosave_path = format!("/tmp/game-campaign-autosave-{}.json", id);

    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 10 });
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    world.insert_resource(CampaignSaveNotice::default());
    world.insert_resource(CampaignAutosaveState {
        enabled: true,
        interval_turns: 5,
        last_autosave_turn: 0,
    });

    // local wrapper to avoid mutating global autosave path constant
    let data = snapshot_campaign_from_world(&world);
    save_campaign_data(&autosave_path, &data).expect("write autosave sample");
    let loaded =
        load_and_prepare_campaign_data(&autosave_path).expect("load autosave sample");
    assert_eq!(loaded.save_version, CURRENT_SAVE_VERSION);
    assert_eq!(loaded.run_state.global_turn, 10);
    let _ = std::fs::remove_file(&autosave_path);
}

#[test]
fn autosave_system_updates_last_turn_when_interval_met() {
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 12 });
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    world.insert_resource(CampaignSaveNotice::default());
    world.insert_resource(CampaignSaveIndexState::default());
    world.insert_resource(CampaignAutosaveState {
        enabled: true,
        interval_turns: 10,
        last_autosave_turn: 0,
    });

    campaign_autosave_system(&mut world);
    let state = world.resource::<CampaignAutosaveState>();
    assert_eq!(state.last_autosave_turn, 12);
    let _ = std::fs::remove_file(CAMPAIGN_AUTOSAVE_PATH);
}
