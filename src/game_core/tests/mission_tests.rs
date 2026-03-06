use bevy::prelude::*;

use super::helpers::*;
use super::super::attention_systems::*;
use super::super::roster_gen::*;
use super::super::roster_types::*;
use super::super::overworld_types::*;
use super::super::types::*;

#[test]
fn mission_activates_on_turn_five() {
    let mut app = build_test_app();
    for _ in 0..5 {
        app.update();
    }
    let progress = active_progress(&mut app.world);
    assert!(progress.mission_active);
    assert_eq!(progress.result, MissionResult::InProgress);
}

#[test]
fn hero_abilities_advance_sabotage_progress() {
    let mut app = build_test_app();
    for _ in 0..10 {
        app.update();
    }
    let progress = active_progress(&mut app.world);
    assert!(progress.sabotage_progress > 0.0);
}

#[test]
fn enemy_ai_can_damage_hero() {
    let mut app = build_test_app();
    for _ in 0..12 {
        app.update();
    }
    let mut hero_query = app.world.query::<(&super::super::types::Hero, &Health)>();
    let (_, hero_health) = hero_query.single(&app.world);
    assert!(hero_health.current < hero_health.max);
}

#[test]
fn mission_can_end_in_victory() {
    let mut app = build_test_app();
    for _ in 0..40 {
        app.update();
        let p = active_progress(&mut app.world);
        if !p.mission_active && p.result == MissionResult::Victory {
            return;
        }
    }
    assert_eq!(
        active_progress(&mut app.world).result,
        MissionResult::Victory
    );
}

#[test]
fn aggressive_mode_increases_sabotage_speed() {
    let mut app = build_test_app();
    set_active_tactics(&mut app.world, |t| {
        t.tactical_mode = TacticalMode::Aggressive
    });
    for _ in 0..10 {
        app.update();
    }
    let aggressive_progress = active_progress(&mut app.world).sabotage_progress;

    let mut baseline = build_test_app();
    for _ in 0..10 {
        baseline.update();
    }
    let baseline_progress = active_progress(&mut baseline.world).sabotage_progress;
    assert!(aggressive_progress > baseline_progress);
}

#[test]
fn mission_defeats_when_timer_expires() {
    let mut app = build_test_app();
    set_active_progress(&mut app.world, |p| {
        p.mission_active = true;
        p.result = MissionResult::InProgress;
        p.turns_remaining = 1;
    });
    app.update();
    let p = active_progress(&mut app.world);
    assert!(!p.mission_active);
    assert_eq!(p.result, MissionResult::Defeat);
}

#[test]
fn map_progresses_when_threshold_is_crossed() {
    let mut app = build_test_app();
    set_active_progress(&mut app.world, |p| {
        p.mission_active = true;
        p.sabotage_progress = 20.0;
    });
    app.update();
    let p = active_progress(&mut app.world);
    assert_eq!(p.room_index, 1);
}

#[test]
fn command_cooldown_counts_down_each_turn_when_active() {
    let mut app = build_test_app();
    set_active_progress(&mut app.world, |p| p.mission_active = true);
    set_active_tactics(&mut app.world, |t| t.command_cooldown_turns = 3);
    app.update();
    let t = active_tactics(&mut app.world);
    assert_eq!(t.command_cooldown_turns, 2);
}

#[test]
fn enemy_pressure_raises_alert_and_reduces_integrity() {
    let mut app = build_test_app();
    set_active_progress(&mut app.world, |p| p.mission_active = true);
    let initial_alert = active_progress(&mut app.world).alert_level;
    let initial_integrity = active_progress(&mut app.world).reactor_integrity;
    app.update();
    app.update();
    let p = active_progress(&mut app.world);
    assert!(p.alert_level > initial_alert);
    assert!(p.reactor_integrity < initial_integrity);
}

#[test]
fn pressure_curve_is_deterministic_across_identical_runs() {
    let mut a = build_test_app();
    let mut b = build_test_app();
    set_active_progress(&mut a.world, |p| p.mission_active = true);
    set_active_progress(&mut b.world, |p| p.mission_active = true);
    for _ in 0..12 {
        a.update();
        b.update();
    }
    let pa = active_progress(&mut a.world);
    let pb = active_progress(&mut b.world);
    assert_eq!(pa.sabotage_progress, pb.sabotage_progress);
    assert_eq!(pa.alert_level, pb.alert_level);
    assert_eq!(pa.reactor_integrity, pb.reactor_integrity);
    assert_eq!(pa.turns_remaining, pb.turns_remaining);
}

#[test]
fn unfocused_missions_progress_without_focus() {
    let mut app = build_triage_app();
    let initial = nth_progress(&mut app.world, 1);
    for _ in 0..4 {
        app.update();
    }
    let p = nth_progress(&mut app.world, 1);
    assert!(p.turns_remaining < initial.turns_remaining);
    assert!(p.alert_level > initial.alert_level);
    assert!(p.reactor_integrity < initial.reactor_integrity);
    assert!(p.sabotage_progress < initial.sabotage_progress + 20.0);
    assert!(p.unattended_turns >= 4);
}

#[test]
fn try_shift_focus_spends_attention_and_sets_cooldown() {
    let mut attention = AttentionState::default();
    let result = try_shift_focus(2, &mut attention, 0, 1);
    assert_eq!(result, Some(1));
    assert_eq!(
        attention.switch_cooldown_turns,
        attention.switch_cooldown_max
    );
    assert!(attention.global_energy < attention.max_energy);
}

#[test]
fn try_shift_focus_blocks_when_attention_is_exhausted() {
    let mut attention = AttentionState::default();
    attention.global_energy = 0.0;
    let result = try_shift_focus(2, &mut attention, 0, 1);
    assert!(result.is_none());
}

#[test]
fn triage_simulation_is_deterministic_for_same_initial_state() {
    let mut a = build_triage_app();
    let mut b = build_triage_app();
    for _ in 0..8 {
        a.update();
        b.update();
    }
    let count = mission_count(&mut a.world);
    assert_eq!(count, mission_count(&mut b.world));
    assert_eq!(
        board_active_idx(&mut a.world),
        board_active_idx(&mut b.world)
    );
    for idx in 0..count {
        let pa = nth_progress(&mut a.world, idx);
        let pb = nth_progress(&mut b.world, idx);
        assert_eq!(pa.mission_active, pb.mission_active);
        assert_eq!(pa.result, pb.result);
        assert_eq!(pa.turns_remaining, pb.turns_remaining);
        assert_eq!(pa.sabotage_progress, pb.sabotage_progress);
        assert_eq!(pa.alert_level, pb.alert_level);
        assert_eq!(pa.reactor_integrity, pb.reactor_integrity);
        assert_eq!(pa.unattended_turns, pb.unattended_turns);
    }
}

#[test]
fn focused_mission_outpaces_matching_unfocused_mission() {
    let mut app = build_triage_app();
    let entities: Vec<Entity> = app.world.resource::<MissionBoard>().entities.clone();
    let focused_snap = MissionSnapshot {
        mission_name: "Focus".to_string(),
        bound_region_id: Some(0),
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining: 20,
        reactor_integrity: 95.0,
        sabotage_progress: 30.0,
        sabotage_goal: 100.0,
        alert_level: 20.0,
        room_index: 1,
        tactical_mode: TacticalMode::Balanced,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    };
    let unfocused_snap = MissionSnapshot {
        mission_name: "Unfocused".to_string(),
        bound_region_id: Some(1),
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining: 20,
        reactor_integrity: 95.0,
        sabotage_progress: 30.0,
        sabotage_goal: 100.0,
        alert_level: 20.0,
        room_index: 1,
        tactical_mode: TacticalMode::Balanced,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    };
    overwrite_mission_entity(&mut app.world, entities[0], focused_snap);
    overwrite_mission_entity(&mut app.world, entities[1], unfocused_snap);
    for _ in 0..6 {
        app.update();
    }
    let focused = nth_progress(&mut app.world, 0);
    let unfocused = nth_progress(&mut app.world, 1);
    assert!(focused.sabotage_progress > unfocused.sabotage_progress);
    assert!(focused.reactor_integrity >= unfocused.reactor_integrity);
    assert!(focused.alert_level <= unfocused.alert_level);
}

#[test]
fn sustained_focus_consumes_attention_energy() {
    let mut app = build_triage_app();
    let initial_energy = app.world.resource::<AttentionState>().global_energy;
    for _ in 0..5 {
        app.update();
    }
    let attention = app.world.resource::<AttentionState>();
    assert!(attention.global_energy < initial_energy);
}

#[test]
fn unattended_escalation_accelerates_with_time() {
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 1 });
    world.insert_resource(MissionBoard::default());

    let active_snap = MissionSnapshot {
        mission_name: "Active".to_string(),
        bound_region_id: Some(0),
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining: 30,
        reactor_integrity: 100.0,
        sabotage_progress: 0.0,
        sabotage_goal: 100.0,
        alert_level: 4.0,
        room_index: 0,
        tactical_mode: TacticalMode::Balanced,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    };
    let unfocused_snap = MissionSnapshot {
        mission_name: "Unfocused".to_string(),
        bound_region_id: Some(1),
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining: 20,
        reactor_integrity: 95.0,
        sabotage_progress: 40.0,
        sabotage_goal: 100.0,
        alert_level: 10.0,
        room_index: 0,
        tactical_mode: TacticalMode::Balanced,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    };
    let (d0, p0, t0) = active_snap.into_components(0);
    let e0 = world
        .spawn((d0, p0, t0, AssignedHero::default(), ActiveMission))
        .id();
    let (d1, p1, t1) = unfocused_snap.into_components(1);
    let e1 = world.spawn((d1, p1, t1, AssignedHero::default())).id();
    world.resource_mut::<MissionBoard>().entities = vec![e0, e1];

    let mut schedule = Schedule::default();
    schedule.add_systems(simulate_unfocused_missions_system);

    let mut previous_progress = world.get::<MissionProgress>(e1).unwrap().sabotage_progress;
    let mut previous_integrity = world.get::<MissionProgress>(e1).unwrap().reactor_integrity;
    let mut first_drop = 0.0_f32;
    let mut last_drop = 0.0_f32;

    for step in 0..6 {
        schedule.run(&mut world);
        let p = world.get::<MissionProgress>(e1).unwrap();
        let progress_drop = previous_progress - p.sabotage_progress;
        let integrity_drop = previous_integrity - p.reactor_integrity;
        if step == 0 {
            first_drop = progress_drop + integrity_drop;
        }
        last_drop = progress_drop + integrity_drop;
        previous_progress = p.sabotage_progress;
        previous_integrity = p.reactor_integrity;
    }

    assert!(last_drop > first_drop);
    let p = world.get::<MissionProgress>(e1).unwrap();
    assert!(p.unattended_turns >= 6);
}

#[test]
fn recruit_generation_is_deterministic_for_same_seed_and_id() {
    let a = generate_recruit(0x1234_5678, 11);
    let b = generate_recruit(0x1234_5678, 11);
    assert_eq!(a.codename, b.codename);
    assert_eq!(a.origin_faction_id, b.origin_faction_id);
    assert_eq!(a.origin_region_id, b.origin_region_id);
    assert_eq!(a.backstory, b.backstory);
    assert_eq!(a.archetype, b.archetype);
    assert_eq!(a.resolve, b.resolve);
    assert_eq!(a.loyalty_bias, b.loyalty_bias);
    assert_eq!(a.risk_tolerance, b.risk_tolerance);
}

#[test]
fn recruit_backstory_references_overworld_faction_and_region() {
    let map = OverworldMap::from_seed(0x0000_BEEF);
    let r = generate_recruit_for_overworld(0x1234_5678, 11, &map);
    let faction_name = map
        .factions
        .get(r.origin_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Unknown");
    let region_name = map
        .regions
        .iter()
        .find(|x| x.id == r.origin_region_id)
        .map(|x| x.name.as_str())
        .unwrap_or("Unknown");
    assert!(r.backstory.contains(faction_name));
    assert!(r.backstory.contains(region_name));
}

#[test]
fn roster_lore_sync_updates_recruit_origins_to_active_overworld() {
    let mut world = World::new();
    world.insert_resource(OverworldMap::from_seed(0x00CA_FE01));
    world.insert_resource(CampaignRoster::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(sync_roster_lore_with_overworld_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    let roster = world.resource::<CampaignRoster>();
    for recruit in &roster.recruit_pool {
        let faction_name = map
            .factions
            .get(recruit.origin_faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("Unknown");
        assert!(recruit.backstory.contains(faction_name));
    }
}

#[test]
fn signing_recruit_persists_in_roster_and_refills_pool() {
    let mut roster = CampaignRoster::default();
    let first_id = roster.recruit_pool[0].id;
    let initial_hero_count = roster.heroes.len();
    let initial_pool = roster.recruit_pool.len();
    let signed = sign_top_recruit(&mut roster).expect("expected recruit");
    assert_eq!(signed.id, first_id);
    assert_eq!(roster.heroes.len(), initial_hero_count + 1);
    assert_eq!(roster.recruit_pool.len(), initial_pool);
}
