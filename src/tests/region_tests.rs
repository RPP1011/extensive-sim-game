use super::*;


#[test]
fn region_transition_missing_faction_payload_fails_and_stays_overworld() {
    let mut hub_ui = HubUiState {
        screen: HubScreen::OverworldMap,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut picker = RegionTargetPickerState::default();
    let camera_transition = CameraFocusTransitionState::default();
    let mut region_transition = RegionLayerTransitionState::default();
    let overworld = game_core::OverworldMap::default();
    let missing_creation = CharacterCreationState::default();

    let notice = request_enter_selected_region(
        &mut hub_ui,
        &mut picker,
        &camera_transition,
        &mut region_transition,
        &overworld,
        &missing_creation,
    );

    assert!(notice.contains("missing faction context"));
    assert_eq!(hub_ui.screen, HubScreen::OverworldMap);
    assert!(!region_transition.interaction_locked);
    assert!(region_transition.pending_payload.is_none());
}

#[test]
fn region_transition_guard_rejects_invalid_pending_payload() {
    let mut hub_ui = HubUiState {
        screen: HubScreen::OverworldMap,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let overworld = game_core::OverworldMap::default();
    let mut transition = RegionLayerTransitionState {
        active_payload: None,
        pending_payload: Some(RegionTransitionPayload {
            region_id: 0,
            faction_id: String::new(),
            faction_index: 0,
            campaign_seed: overworld.map_seed,
            region_seed: derive_region_transition_seed(overworld.map_seed, 0, 0),
        }),
        pending_frames: 0,
        interaction_locked: true,
        status: "pending".to_string(),
    };

    let status = advance_region_layer_transition(&mut hub_ui, &mut transition, &overworld)
        .expect("transition should resolve");

    assert!(status.contains("failed"));
    assert_eq!(hub_ui.screen, HubScreen::OverworldMap);
    assert!(!transition.interaction_locked);
    assert!(transition.active_payload.is_none());
}

#[test]
fn local_intro_bootstrap_from_region_d_completes_and_hands_off_input() {
    let mut hub_ui = HubUiState {
        screen: HubScreen::RegionView,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut local_intro = LocalEagleEyeIntroState::default();
    let overworld = game_core::OverworldMap::default();
    let transition = RegionLayerTransitionState {
        active_payload: Some(RegionTransitionPayload {
            region_id: 3,
            faction_id: "faction-0-test".to_string(),
            faction_index: 0,
            campaign_seed: overworld.map_seed,
            region_seed: derive_region_transition_seed(overworld.map_seed, 3, 0),
        }),
        pending_payload: None,
        pending_frames: 0,
        interaction_locked: false,
        status: "Region scene loaded".to_string(),
    };

    let status =
        bootstrap_local_eagle_eye_intro(&mut hub_ui, &mut local_intro, &transition, &overworld);
    assert!(status.contains("bootstrapped"));
    assert_eq!(hub_ui.screen, HubScreen::LocalEagleEyeIntro);
    assert_eq!(local_intro.phase, LocalIntroPhase::HiddenInside);
    assert!(local_intro.anchor.is_some());
    assert!(!local_intro.intro_completed);
    assert!(!local_intro.input_handoff_ready);
    assert!(!hub_runtime_input_enabled(
        &hub_ui,
        None,
        Some(&local_intro)
    ));

    let mut completed = false;
    for _ in 0..(local_intro::LOCAL_INTRO_HIDDEN_FRAMES + local_intro::LOCAL_INTRO_EXIT_FRAMES + 5)
    {
        if advance_local_eagle_eye_intro(&mut local_intro).is_some()
            && local_intro.intro_completed
        {
            completed = true;
            break;
        }
    }

    assert!(completed);
    assert_eq!(local_intro.phase, LocalIntroPhase::GameplayControl);
    assert!(local_intro.intro_completed);
    assert!(local_intro.input_handoff_ready);
    assert!(hub_runtime_input_enabled(&hub_ui, None, Some(&local_intro)));
}

#[test]
fn local_intro_bootstrap_aborts_safely_when_anchor_unavailable() {
    let mut hub_ui = HubUiState {
        screen: HubScreen::RegionView,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut local_intro = LocalEagleEyeIntroState::default();
    let mut overworld = game_core::OverworldMap::default();
    if overworld.regions.len() < 5 {
        let mut extra = overworld
            .regions
            .last()
            .cloned()
            .expect("at least one region exists");
        extra.id = 4;
        extra.name = "Region-E".to_string();
        extra.neighbors.clear();
        overworld.regions.push(extra);
    }
    let transition = RegionLayerTransitionState {
        active_payload: Some(RegionTransitionPayload {
            region_id: 4,
            faction_id: "faction-0-test".to_string(),
            faction_index: 0,
            campaign_seed: overworld.map_seed,
            region_seed: derive_region_transition_seed(overworld.map_seed, 4, 0),
        }),
        pending_payload: None,
        pending_frames: 0,
        interaction_locked: false,
        status: "Region scene loaded".to_string(),
    };

    let status =
        bootstrap_local_eagle_eye_intro(&mut hub_ui, &mut local_intro, &transition, &overworld);

    assert!(status.contains("aborted"));
    assert!(status.contains("unavailable"));
    assert_eq!(hub_ui.screen, HubScreen::RegionView);
    assert_eq!(local_intro.phase, LocalIntroPhase::Idle);
    assert!(local_intro.anchor.is_none());
    assert!(!local_intro.intro_completed);
    assert!(!local_intro.input_handoff_ready);
}

#[test]
fn party_panel_label_shows_selected_control_and_target_markers() {
    let overworld = game_core::OverworldMap::default();
    let party = game_core::CampaignParty {
        id: 7,
        name: "Party-B".to_string(),
        leader_hero_id: 11,
        region_id: 0,
        supply: 92.0,
        speed: 1.0,
        is_player_controlled: false,
        order: game_core::PartyOrderKind::ReinforceFront,
        order_target_region_id: Some(1),
    };

    let target_label = party_target_region_label(&party, &overworld);
    let label = party_panel_label(&party, true, "Captain Vale", "Region-A", &target_label);

    assert!(label.contains("[SELECTED]"));
    assert!(label.contains("[DELEGATED]"));
    assert!(label.contains("order=ReinforceFront"));
    assert!(label.contains("->"));
}

#[test]
fn camera_focus_transition_interpolates_and_completes() {
    let mut transition = CameraFocusTransitionState::default();
    let start = Vec3::new(0.0, 0.0, 0.0);
    let target = Vec3::new(8.0, 0.0, -3.0);

    let queued = transition.begin(start, target, 10, 4, CameraFocusTrigger::TakeCommand);
    assert_eq!(queued, CameraFocusTransitionQueueResult::Started);
    assert!(transition.is_active());

    let mut last_focus = start;
    let mut completed = false;
    for _ in 0..20 {
        if let Some(step) = transition.step(0.05) {
            last_focus = step.focus;
            if step.completed {
                completed = true;
                break;
            }
        }
    }

    assert!(completed);
    assert!(!transition.is_active());
    assert!((last_focus - target).length() <= 0.001);
}

#[test]
fn camera_focus_transition_retargets_safely_under_rapid_requests() {
    let mut transition = CameraFocusTransitionState::default();
    let first_target = Vec3::new(6.0, 0.0, 1.0);
    let second_target = Vec3::new(-5.0, 0.0, 4.0);
    let first = transition.begin(
        Vec3::ZERO,
        first_target,
        1,
        2,
        CameraFocusTrigger::TakeCommand,
    );
    assert_eq!(first, CameraFocusTransitionQueueResult::Started);
    let mid_step = transition.step(0.10).expect("step should exist");
    assert!(transition.is_active());

    let second = transition.begin(
        mid_step.focus,
        second_target,
        9,
        7,
        CameraFocusTrigger::FocusSelectedParty,
    );
    assert_eq!(second, CameraFocusTransitionQueueResult::Retargeted);
    assert_eq!(
        transition.active.as_ref().map(|t| t.target_party_id),
        Some(9)
    );
    assert_eq!(
        transition.active.as_ref().map(|t| t.target_region_id),
        Some(7)
    );

    let mut last_focus = mid_step.focus;
    let mut completed = false;
    for _ in 0..24 {
        if let Some(step) = transition.step(0.05) {
            last_focus = step.focus;
            if step.completed {
                completed = true;
                break;
            }
        }
    }

    assert!(completed);
    assert!(!transition.is_active());
    assert!((last_focus - second_target).length() <= 0.001);
}

#[test]
fn start_menu_entry_resets_menu_copy_and_ignores_runtime_notice() {
    let mut hub_ui = HubUiState {
        screen: HubScreen::OverworldMap,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut start_menu = StartMenuState {
        subtitle: "custom subtitle".to_string(),
        status: "custom status".to_string(),
        hamburger_expanded: false,
    };
    let runtime_notice = "Mission action completed in runtime.".to_string();

    enter_start_menu(&mut hub_ui, &mut start_menu);

    assert!(hub_ui.screen == HubScreen::StartMenu);
    assert_eq!(start_menu.subtitle, StartMenuState::default_subtitle());
    assert_eq!(start_menu.status, StartMenuState::default_status());
    assert_ne!(start_menu.subtitle, runtime_notice);
    assert_ne!(start_menu.status, runtime_notice);
}
