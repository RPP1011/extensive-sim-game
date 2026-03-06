use super::*;

#[test]
fn long_run_save_load_chain_has_no_state_drift_across_seeds() {
    let seeds = [0x11_u64, 0x22_u64, 0x1234_5678_u64];
    for seed in seeds {
        let mut world = build_campaign_test_world(seed);
        let mut schedule = build_campaign_sim_schedule();

        for turn in 1..=80_u32 {
            world.resource_mut::<RunState>().global_turn = turn;
            schedule.run(&mut world);

            let before = campaign_signature_from_world(&world);
            let repairs = canonical_roundtrip_world(&mut world);
            assert!(
                repairs.is_empty(),
                "unexpected repairs at turn {turn} seed {seed}"
            );
            let after = campaign_signature_from_world(&world);

            assert_eq!(
                before, after,
                "save/load drift at turn {} for seed {} ({} != {})",
                turn, seed, before, after
            );
        }
    }
}

#[test]
fn repeated_save_migration_roundtrip_keeps_signature_stable() {
    let mut world = build_campaign_test_world(0x00AB_CDEF);
    let repairs = canonical_roundtrip_world(&mut world);
    assert!(repairs.is_empty());
    let baseline = campaign_signature_from_world(&world);
    for _ in 0..40 {
        let repairs = canonical_roundtrip_world(&mut world);
        assert!(repairs.is_empty());
    }
    let final_sig = campaign_signature_from_world(&world);
    assert_eq!(baseline, final_sig);
}

#[test]
fn settings_visual_system_does_not_panic() {
    let mut app = App::new();
    app.insert_resource(CameraSettings {
        orbit_sensitivity: 1.2,
        zoom_sensitivity: 0.9,
        invert_orbit_y: true,
    });
    app.add_systems(Update, update_settings_menu_visual_system);

    app.world
        .spawn((OrbitSensitivitySliderFill, Style::default()));
    app.world
        .spawn((ZoomSensitivitySliderFill, Style::default()));
    app.world.spawn((
        OrbitSensitivityLabel,
        Text::from_section("o", TextStyle::default()),
    ));
    app.world.spawn((
        ZoomSensitivityLabel,
        Text::from_section("z", TextStyle::default()),
    ));
    app.world.spawn((
        InvertOrbitYLabel,
        Text::from_section("i", TextStyle::default()),
    ));

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        app.update();
    }));
    assert!(result.is_ok(), "settings visual system panicked");
}

#[test]
fn non_headless_update_smoke_does_not_panic() {
    let mut app = App::new();

    app.insert_resource(SceneViewBounds::default())
        .insert_resource(CameraSettings::default())
        .insert_resource(SettingsMenuState::default())
        .insert_resource(ManualScreenshotState::default())
        .insert_resource(CameraFocusTransitionState::default())
        .insert_resource(Time::<()>::default())
        .insert_resource(SimulationSteps(None))
        .insert_resource(ButtonInput::<MouseButton>::default())
        .insert_resource(Events::<MouseMotion>::default())
        .insert_resource(Events::<MouseWheel>::default())
        .init_resource::<RunState>()
        .init_resource::<game_core::MissionMap>()
        .init_resource::<game_core::MissionBoard>();

    app.add_systems(Startup, game_core::setup_test_scene_headless);

    app.add_systems(
        Update,
        (
            increment_global_turn,
            game_core::turn_management_system,
            game_core::auto_increase_stress,
            game_core::activate_mission_system,
            game_core::mission_map_progression_system,
            game_core::player_command_input_system,
            game_core::hero_ability_system,
            game_core::enemy_ai_system,
            game_core::combat_system,
            game_core::complete_objective_system,
            game_core::end_mission_system,
            game_core::print_game_state,
            update_mission_hud_system,
        )
            .chain(),
    );
    app.add_systems(
        Update,
        (
            settings_menu_toggle_system,
            settings_menu_slider_input_system,
            settings_menu_toggle_input_system,
            persist_camera_settings_system,
            update_settings_menu_visual_system,
            orbit_camera_controller_system,
        )
            .chain(),
    );

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        app.update();
    }));
    assert!(
        result.is_ok(),
        "non-headless one-frame update smoke panicked"
    );
}

#[test]
fn hub_assemble_expedition_stabilizes_active_missions() {
    let mut missions = game_core::default_mission_snapshots();
    let mut attention = game_core::AttentionState::default();
    let mut roster = game_core::CampaignRoster::default();

    let notice = apply_hub_action(
        HubAction::AssembleExpedition,
        &mut missions,
        &mut attention,
        &mut roster,
    );

    assert!(notice.contains("Quartermaster"));
    assert!(attention.global_energy < attention.max_energy);
    for mission in &missions {
        if mission.result == MissionResult::InProgress {
            assert!(mission.reactor_integrity >= 92.0);
            assert!(mission.alert_level <= 20.0);
        }
    }
}

#[test]
fn hub_review_recruits_targets_high_alert_mission() {
    let mut missions = game_core::default_mission_snapshots();
    missions[0].alert_level = 12.0;
    missions[1].alert_level = 45.0;
    missions[2].alert_level = 22.0;
    let mut attention = game_core::AttentionState::default();
    let mut roster = game_core::CampaignRoster::default();
    let initial_heroes = roster.heroes.len();

    let notice = apply_hub_action(
        HubAction::ReviewRecruits,
        &mut missions,
        &mut attention,
        &mut roster,
    );

    assert!(notice.contains("signed"));
    assert_eq!(
        missions[1].tactical_mode,
        game_core::TacticalMode::Defensive
    );
    assert!(missions[1].alert_level < 45.0);
    assert_eq!(roster.heroes.len(), initial_heroes + 1);
    assert!(attention.global_energy < attention.max_energy);
}

#[test]
fn hub_action_fails_when_attention_is_insufficient() {
    let mut missions = game_core::default_mission_snapshots();
    let baseline = missions[0].clone();
    let mut attention = game_core::AttentionState::default();
    let mut roster = game_core::CampaignRoster::default();
    attention.global_energy = 2.0;

    let notice = apply_hub_action(
        HubAction::DispatchRelief,
        &mut missions,
        &mut attention,
        &mut roster,
    );

    assert!(notice.contains("denied") || notice.contains("threshold"));
    assert_eq!(attention.global_energy, 2.0);
    assert_eq!(missions[0].turns_remaining, baseline.turns_remaining);
    assert_eq!(missions[0].reactor_integrity, baseline.reactor_integrity);
}

#[test]
fn hub_action_sequence_is_deterministic() {
    let actions = [
        HubAction::AssembleExpedition,
        HubAction::ReviewRecruits,
        HubAction::IntelSweep,
        HubAction::DispatchRelief,
    ];

    let mut missions_a = game_core::default_mission_snapshots();
    let mut attention_a = game_core::AttentionState::default();
    let mut roster_a = game_core::CampaignRoster::default();
    let mut notices_a = Vec::new();
    for action in actions {
        notices_a.push(apply_hub_action(
            action,
            &mut missions_a,
            &mut attention_a,
            &mut roster_a,
        ));
    }

    let mut missions_b = game_core::default_mission_snapshots();
    let mut attention_b = game_core::AttentionState::default();
    let mut roster_b = game_core::CampaignRoster::default();
    let mut notices_b = Vec::new();
    for action in actions {
        notices_b.push(apply_hub_action(
            action,
            &mut missions_b,
            &mut attention_b,
            &mut roster_b,
        ));
    }

    assert_eq!(notices_a, notices_b);
    assert_eq!(attention_a.global_energy, attention_b.global_energy);
    assert_eq!(missions_a.len(), missions_b.len());
    for idx in 0..missions_a.len() {
        assert_eq!(
            missions_a[idx].turns_remaining,
            missions_b[idx].turns_remaining
        );
        assert_eq!(missions_a[idx].alert_level, missions_b[idx].alert_level);
        assert_eq!(
            missions_a[idx].reactor_integrity,
            missions_b[idx].reactor_integrity
        );
    }
    assert_eq!(roster_a.heroes.len(), roster_b.heroes.len());
    let names_a = roster_a
        .heroes
        .iter()
        .map(|h| h.name.clone())
        .collect::<Vec<_>>();
    let names_b = roster_b
        .heroes
        .iter()
        .map(|h| h.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(names_a, names_b);
}
