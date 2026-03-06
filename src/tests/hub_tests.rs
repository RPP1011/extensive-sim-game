use super::*;

#[test]
fn new_campaign_request_routes_to_character_creation_faction_screen() {
    let mut world = World::new();
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(HubUiState {
        screen: HubScreen::StartMenu,
        show_credits: false,
        request_quit: false,
        request_new_campaign: true,
        request_continue_campaign: false,
    });

    hub_new_campaign_requested_system(&mut world);

    let hub_ui = world.resource::<HubUiState>();
    assert!(hub_ui.screen == HubScreen::CharacterCreationFaction);
    assert!(!hub_ui.request_new_campaign);
}

#[test]
fn faction_selection_requires_choice_before_backstory() {
    let overworld = game_core::OverworldMap::default();
    let mut hub_ui = HubUiState {
        screen: HubScreen::CharacterCreationFaction,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut creation = CharacterCreationState::default();
    let mut creation_ui = CharacterCreationUiState::default();
    let mut diplomacy = game_core::DiplomacyState::default();

    let advanced = confirm_faction_selection(
        &mut hub_ui,
        &mut creation,
        &mut creation_ui,
        &mut diplomacy,
        &overworld,
    );

    assert!(!advanced);
    assert!(hub_ui.screen == HubScreen::CharacterCreationFaction);
    assert!(creation_ui.status.contains("Select a faction"));
}

#[test]
fn faction_selection_sets_identifier_and_advances_to_backstory() {
    let overworld = game_core::OverworldMap::default();
    let choice = build_faction_selection_choices(&overworld)
        .into_iter()
        .nth(1)
        .expect("at least two factions");
    let mut hub_ui = HubUiState {
        screen: HubScreen::CharacterCreationFaction,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut creation = CharacterCreationState {
        selected_faction_id: Some(choice.id.clone()),
        selected_faction_index: Some(choice.index),
        selected_backstory_id: None,
        stat_modifiers: Vec::new(),
        recruit_bias_modifiers: Vec::new(),
        is_confirmed: false,
    };
    let mut creation_ui = CharacterCreationUiState::default();
    let mut diplomacy = game_core::DiplomacyState::default();

    let advanced = confirm_faction_selection(
        &mut hub_ui,
        &mut creation,
        &mut creation_ui,
        &mut diplomacy,
        &overworld,
    );

    assert!(advanced);
    assert!(hub_ui.screen == HubScreen::CharacterCreationBackstory);
    assert_eq!(creation.selected_faction_id, Some(choice.id));
    assert!(!creation.is_confirmed);
    assert_eq!(diplomacy.player_faction_id, choice.index);
}

#[test]
fn backstory_selection_requires_choice_before_overworld() {
    let overworld = game_core::OverworldMap::default();
    let mut roster = game_core::CampaignRoster::default();
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let mut hub_ui = HubUiState {
        screen: HubScreen::CharacterCreationBackstory,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut creation = CharacterCreationState {
        selected_faction_id: Some("faction-0-test".to_string()),
        selected_faction_index: Some(0),
        selected_backstory_id: None,
        stat_modifiers: Vec::new(),
        recruit_bias_modifiers: Vec::new(),
        is_confirmed: false,
    };
    let mut creation_ui = CharacterCreationUiState::default();

    let advanced = confirm_backstory_selection(
        &mut hub_ui,
        &mut creation,
        &mut creation_ui,
        &mut roster,
        &mut parties,
        &overworld,
    );

    assert!(!advanced);
    assert!(hub_ui.screen == HubScreen::CharacterCreationBackstory);
    assert!(creation_ui.status.contains("Select a backstory archetype"));
    assert!(!creation.is_confirmed);
}

#[test]
fn backstory_selection_rejects_invalid_identifier() {
    let overworld = game_core::OverworldMap::default();
    let mut roster = game_core::CampaignRoster::default();
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let mut hub_ui = HubUiState {
        screen: HubScreen::CharacterCreationBackstory,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut creation = CharacterCreationState {
        selected_faction_id: Some("faction-0-test".to_string()),
        selected_faction_index: Some(0),
        selected_backstory_id: Some("invalid-backstory".to_string()),
        stat_modifiers: Vec::new(),
        recruit_bias_modifiers: Vec::new(),
        is_confirmed: false,
    };
    let mut creation_ui = CharacterCreationUiState::default();

    let advanced = confirm_backstory_selection(
        &mut hub_ui,
        &mut creation,
        &mut creation_ui,
        &mut roster,
        &mut parties,
        &overworld,
    );

    assert!(!advanced);
    assert_eq!(creation.selected_backstory_id, None);
    assert!(creation_ui.status.contains("no longer valid"));
    assert!(hub_ui.screen == HubScreen::CharacterCreationBackstory);
}

#[test]
fn scout_backstory_applies_modifiers_and_enters_backstory_cinematic() {
    let overworld = game_core::OverworldMap::default();
    let mut roster = game_core::CampaignRoster::default();
    roster.recruit_pool = vec![
        recruit_candidate(1, game_core::PersonalityArchetype::Guardian),
        recruit_candidate(2, game_core::PersonalityArchetype::Tactician),
        recruit_candidate(3, game_core::PersonalityArchetype::Vanguard),
    ];
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let speed_before = parties
        .parties
        .iter()
        .find(|p| p.is_player_controlled)
        .map(|p| p.speed)
        .expect("player party exists");
    let seed_before = overworld.map_seed;
    let mut hub_ui = HubUiState {
        screen: HubScreen::CharacterCreationBackstory,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    };
    let mut creation = CharacterCreationState {
        selected_faction_id: Some("faction-0-test".to_string()),
        selected_faction_index: Some(0),
        selected_backstory_id: Some("scout-pathfinder".to_string()),
        stat_modifiers: Vec::new(),
        recruit_bias_modifiers: Vec::new(),
        is_confirmed: false,
    };
    let mut creation_ui = CharacterCreationUiState::default();

    let advanced = confirm_backstory_selection(
        &mut hub_ui,
        &mut creation,
        &mut creation_ui,
        &mut roster,
        &mut parties,
        &overworld,
    );

    assert!(advanced);
    assert!(hub_ui.screen == HubScreen::BackstoryCinematic);
    assert!(creation.is_confirmed);
    assert_eq!(
        creation.selected_backstory_id,
        Some("scout-pathfinder".to_string())
    );
    assert!(creation
        .stat_modifiers
        .contains(&"Scouting: +2".to_string()));
    assert!(creation
        .recruit_bias_modifiers
        .iter()
        .any(|entry| entry.contains("Tactician and Vanguard")));
    assert!(creation_ui.status.contains("Preparing cinematic"));
    assert_eq!(overworld.map_seed, seed_before);

    let speed_after = parties
        .parties
        .iter()
        .find(|p| p.is_player_controlled)
        .map(|p| p.speed)
        .expect("player party exists");
    assert!(speed_after > speed_before);
    assert_eq!(
        roster.recruit_pool.first().map(|r| r.archetype),
        Some(game_core::PersonalityArchetype::Tactician)
    );
}

#[test]
fn take_command_transfers_control_to_selected_delegated_party() {
    let roster = game_core::CampaignRoster::default();
    let overworld = game_core::OverworldMap::default();
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let delegated_id = parties
        .parties
        .iter()
        .find(|party| !party.is_player_controlled)
        .map(|party| party.id)
        .expect("delegated party exists");
    let old_controlled_id = parties
        .parties
        .iter()
        .find(|party| party.is_player_controlled)
        .map(|party| party.id)
        .expect("controlled party exists");
    parties.selected_party_id = Some(delegated_id);

    let result = transfer_direct_command_to_selected(&mut parties).expect("handoff succeeds");

    assert_eq!(result.new_party_id, delegated_id);
    assert_eq!(result.new_party_name, "Ranging Band");
    assert_eq!(result.previous_party_name, "Main Company");
    assert!(parties
        .parties
        .iter()
        .any(|party| party.id == delegated_id && party.is_player_controlled));
    assert!(parties
        .parties
        .iter()
        .any(|party| party.id == old_controlled_id && !party.is_player_controlled));
    assert_eq!(parties.selected_party_id, Some(delegated_id));
}

#[test]
fn take_command_rejects_ineligible_selected_party_without_mutation() {
    let roster = game_core::CampaignRoster::default();
    let overworld = game_core::OverworldMap::default();
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let controlled_id = parties
        .parties
        .iter()
        .find(|party| party.is_player_controlled)
        .map(|party| party.id)
        .expect("controlled party exists");
    parties.selected_party_id = Some(controlled_id);
    let before = serde_json::to_value(&parties).expect("serialize parties");

    let result = transfer_direct_command_to_selected(&mut parties);
    assert!(result.is_err());
    let err = result.err().unwrap_or_default();
    assert!(err.contains("already directly controlled"));

    let after = serde_json::to_value(&parties).expect("serialize parties");
    assert_eq!(before, after);
}

#[test]
fn region_target_picker_enter_select_confirm_updates_party_target() {
    let roster = game_core::CampaignRoster::default();
    let overworld = game_core::OverworldMap::default();
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let delegated = parties
        .parties
        .iter()
        .find(|party| !party.is_player_controlled)
        .cloned()
        .expect("delegated party exists");
    let mut picker = RegionTargetPickerState::default();
    let target_region = delegated
        .region_id
        .checked_add(1)
        .unwrap_or(delegated.region_id)
        .min(overworld.regions.len().saturating_sub(1));

    let _ = begin_region_target_picker(&mut picker, &delegated);
    assert!(picker.is_active_for_party(delegated.id));
    assert_eq!(picker.selected_region_id(), None);

    let selected_notice = update_region_target_picker_selection(
        &mut picker,
        delegated.id,
        target_region,
        &overworld,
    )
    .expect("selection succeeds");
    assert!(selected_notice.contains("Target picker selected"));
    assert_eq!(picker.selected_region_id(), Some(target_region));

    let confirmed_notice =
        confirm_region_target_picker(&mut picker, &mut parties, delegated.id, &overworld)
            .expect("confirm succeeds");
    assert!(confirmed_notice.contains("target region set"));
    let updated_target = parties
        .parties
        .iter()
        .find(|party| party.id == delegated.id)
        .and_then(|party| party.order_target_region_id);
    assert_eq!(updated_target, Some(target_region));
    assert_eq!(picker.active_party_id(), None);
}

#[test]
fn region_target_picker_confirm_requires_selection_and_preserves_target() {
    let roster = game_core::CampaignRoster::default();
    let overworld = game_core::OverworldMap::default();
    let mut parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    let delegated_id = parties
        .parties
        .iter()
        .find(|party| !party.is_player_controlled)
        .map(|party| party.id)
        .expect("delegated party exists");
    let initial_target = Some(1usize.min(overworld.regions.len().saturating_sub(1)));
    let delegated_idx = parties
        .parties
        .iter()
        .position(|party| party.id == delegated_id)
        .expect("delegated index");
    parties.parties[delegated_idx].order_target_region_id = initial_target;
    let delegated = parties.parties[delegated_idx].clone();
    let mut picker = RegionTargetPickerState::default();
    let _ = begin_region_target_picker(&mut picker, &delegated);

    let err = confirm_region_target_picker(&mut picker, &mut parties, delegated_id, &overworld)
        .expect_err("confirm should fail without selection");
    assert!(err.contains("select a map region"));
    assert!(picker.is_active_for_party(delegated_id));
    assert_eq!(
        parties.parties[delegated_idx].order_target_region_id,
        initial_target
    );
}

#[test]
fn region_transition_payload_contract_contains_region_faction_and_seed() {
    let mut overworld = game_core::OverworldMap::default();
    let region_id = 3usize.min(overworld.regions.len().saturating_sub(1));
    overworld.selected_region = region_id;
    let creation = valid_character_creation_state();

    let payload =
        build_region_transition_payload(&overworld, &creation).expect("payload should build");

    assert_eq!(payload.region_id, region_id);
    assert_eq!(payload.faction_id, "faction-0-test");
    assert_eq!(payload.faction_index, 0);
    assert_eq!(payload.campaign_seed, overworld.map_seed);
    assert_eq!(
        payload.region_seed,
        derive_region_transition_seed(overworld.map_seed, region_id, 0)
    );
}

#[test]
fn region_transition_request_locks_and_then_enters_region_view() {
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
    let mut overworld = game_core::OverworldMap::default();
    let region_id = 2usize.min(overworld.regions.len().saturating_sub(1));
    overworld.selected_region = region_id;
    let creation = valid_character_creation_state();

    let queued_notice = request_enter_selected_region(
        &mut hub_ui,
        &mut picker,
        &camera_transition,
        &mut region_transition,
        &overworld,
        &creation,
    );
    assert!(queued_notice.contains("Transition lock active"));
    assert!(region_transition.interaction_locked);
    assert!(region_transition.pending_payload.is_some());
    assert_eq!(hub_ui.screen, HubScreen::OverworldMap);
    assert!(!hub_runtime_input_enabled(
        &hub_ui,
        Some(&region_transition),
        None
    ));

    let first_tick =
        advance_region_layer_transition(&mut hub_ui, &mut region_transition, &overworld);
    assert!(first_tick.is_none());
    assert_eq!(hub_ui.screen, HubScreen::OverworldMap);

    let second_tick =
        advance_region_layer_transition(&mut hub_ui, &mut region_transition, &overworld);
    assert!(second_tick.is_some());
    assert_eq!(hub_ui.screen, HubScreen::RegionView);
    assert!(!region_transition.interaction_locked);
    assert_eq!(
        region_transition
            .active_payload
            .as_ref()
            .map(|p| p.region_id),
        Some(region_id)
    );
}
