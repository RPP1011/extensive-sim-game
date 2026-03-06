use bevy::prelude::*;

use super::super::attention_systems::*;
use super::super::companion::*;
use super::super::consequence_systems::*;
use super::super::mission_systems::*;
use super::super::overworld_systems::*;
use super::super::roster_types::*;
use super::super::types::*;

// ── helpers ──────────────────────────────────────────────────────────────

pub(super) fn active_progress(world: &mut World) -> MissionProgress {
    let mut qs = world.query_filtered::<&MissionProgress, With<ActiveMission>>();
    qs.single(world).clone()
}

pub(super) fn active_tactics(world: &mut World) -> MissionTactics {
    let mut qs = world.query_filtered::<&MissionTactics, With<ActiveMission>>();
    qs.single(world).clone()
}

pub(super) fn set_active_progress<F: FnOnce(&mut MissionProgress)>(world: &mut World, f: F) {
    let mut qs = world.query_filtered::<&mut MissionProgress, With<ActiveMission>>();
    let mut p = qs.single_mut(world);
    f(&mut p);
}

pub(super) fn set_active_tactics<F: FnOnce(&mut MissionTactics)>(world: &mut World, f: F) {
    let mut qs = world.query_filtered::<&mut MissionTactics, With<ActiveMission>>();
    let mut t = qs.single_mut(world);
    f(&mut t);
}

pub(super) fn nth_progress(world: &mut World, n: usize) -> MissionProgress {
    let entity = world.resource::<MissionBoard>().entities[n];
    world
        .get::<MissionProgress>(entity)
        .expect("MissionProgress on entity")
        .clone()
}

pub(super) fn set_nth_progress<F: FnOnce(&mut MissionProgress)>(world: &mut World, n: usize, f: F) {
    let entity = world.resource::<MissionBoard>().entities[n];
    let mut p = world
        .get_mut::<MissionProgress>(entity)
        .expect("MissionProgress");
    f(&mut p);
}

pub(super) fn mission_count(world: &mut World) -> usize {
    world.resource::<MissionBoard>().entities.len()
}

pub(super) fn board_active_idx(world: &mut World) -> usize {
    let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    entities
        .iter()
        .position(|&e| world.get::<ActiveMission>(e).is_some())
        .unwrap_or(0)
}

/// Overwrites all three components on a mission entity from a snapshot.
pub(super) fn overwrite_mission_entity(world: &mut World, entity: Entity, snap: MissionSnapshot) {
    let id = world.get::<MissionData>(entity).unwrap().id;
    let (data, progress, tactics) = snap.into_components(id);
    *world.get_mut::<MissionData>(entity).unwrap() = data;
    *world.get_mut::<MissionProgress>(entity).unwrap() = progress;
    *world.get_mut::<MissionTactics>(entity).unwrap() = tactics;
}

/// Spawns default mission entities into `world`, populating `MissionBoard.entities`.
pub(super) fn spawn_test_missions(world: &mut World) {
    let snaps = default_mission_snapshots();
    let mut entities = Vec::new();
    for (i, snap) in snaps.into_iter().enumerate() {
        let id = {
            let mut board = world.resource_mut::<MissionBoard>();
            let id = board.next_id;
            board.next_id += 1;
            id
        };
        let (data, progress, tactics) = snap.into_components(id);
        let entity = if i == 0 {
            world
                .spawn((
                    data,
                    progress,
                    tactics,
                    AssignedHero::default(),
                    ActiveMission,
                ))
                .id()
        } else {
            world
                .spawn((data, progress, tactics, AssignedHero::default()))
                .id()
        };
        entities.push(entity);
    }
    world.resource_mut::<MissionBoard>().entities = entities;
}

// ── app builders ─────────────────────────────────────────────────────────

pub(super) fn build_test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<RunState>()
        .init_resource::<MissionBoard>()
        .init_resource::<MissionMap>()
        .add_systems(
            Update,
            (
                increment_turn_for_tests,
                turn_management_system,
                activate_mission_system,
                mission_map_progression_system,
                player_command_input_system,
                hero_ability_system,
                enemy_ai_system,
                combat_system,
                complete_objective_system,
                end_mission_system,
            )
                .chain(),
        );
    // Spawn hero, enemy, objective, and mission entities directly.
    app.world.spawn((
        Hero {
            name: "Warden Lyra".to_string(),
        },
        Stress {
            value: 0.0,
            max: 100.0,
        },
        Health {
            current: 100.0,
            max: 100.0,
        },
        HeroAbilities {
            focus_fire_cooldown: 0,
            stabilize_cooldown: 0,
            sabotage_charge_cooldown: 0,
        },
    ));
    app.world.spawn((
        Enemy {
            name: "Crypt Sentinel".to_string(),
        },
        Health {
            current: 40.0,
            max: 40.0,
        },
        EnemyAI {
            base_attack_power: 6.0,
            turns_until_attack: 1,
            attack_interval: 2,
            enraged_threshold: 0.5,
        },
    ));
    app.world.spawn(MissionObjective {
        description: "Rupture the ritual anchor".to_string(),
        completed: false,
    });
    spawn_test_missions(&mut app.world);
    app
}

pub(super) fn increment_turn_for_tests(mut run_state: ResMut<RunState>) {
    run_state.global_turn += 1;
}

pub(super) fn build_triage_app() -> App {
    let mut app = App::new();
    let empty_roster = CampaignRoster {
        heroes: vec![],
        recruit_pool: vec![],
        player_hero_id: None,
        next_id: 1,
        generation_counter: 0,
    };
    app.add_plugins(MinimalPlugins)
        .init_resource::<RunState>()
        .init_resource::<MissionBoard>()
        .init_resource::<MissionMap>()
        .init_resource::<super::super::overworld_types::AttentionState>()
        .init_resource::<super::super::overworld_types::OverworldMap>()
        .insert_resource(empty_roster)
        .init_resource::<CampaignLedger>()
        .init_resource::<CompanionStoryState>()
        .add_systems(
            Update,
            (
                increment_turn_for_tests,
                super::super::setup::attention_management_system,
                super::super::setup::overworld_cooldown_system,
                overworld_sync_from_missions_system,
                turn_management_system,
                focused_attention_intervention_system,
                mission_map_progression_system,
                simulate_unfocused_missions_system,
                sync_mission_assignments_system,
                companion_mission_impact_system,
                companion_state_drift_system,
                resolve_mission_consequences_system,
                progress_companion_story_quests_system,
                generate_companion_story_quests_system,
                companion_recovery_system,
            )
                .chain(),
        );
    spawn_test_missions(&mut app.world);
    app
}

/// Like `build_triage_app` but with the default roster populated.
pub(super) fn build_triage_app_with_roster() -> App {
    let mut app = build_triage_app();
    app.world.insert_resource(CampaignRoster::default());
    app
}

// ── Input simulation helpers ─────────────────────────────────────────────

pub(super) fn press(key: KeyCode) -> ButtonInput<KeyCode> {
    let mut kb = ButtonInput::<KeyCode>::default();
    kb.press(key);
    kb
}

pub(super) fn run_player_command(world: &mut World, key: KeyCode) {
    world.insert_resource(press(key));
    let mut s = Schedule::default();
    s.add_systems(player_command_input_system);
    s.run(world);
}

pub(super) fn run_overworld_hub(world: &mut World, key: KeyCode) {
    world.insert_resource(press(key));
    let mut s = Schedule::default();
    s.add_systems(super::super::overworld_nav::overworld_hub_input_system);
    s.run(world);
}

pub(super) fn run_interaction_offer(world: &mut World, key: KeyCode) {
    world.insert_resource(press(key));
    let mut s = Schedule::default();
    s.add_systems(super::super::diplomacy_systems::interaction_offer_input_system);
    s.run(world);
}

pub(super) fn campaign_signature(world: &mut World) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mix = |h: &mut u64, v: u64| {
        *h ^= v;
        *h = h.wrapping_mul(0x1000_0000_01b3);
    };

    let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
    let active_idx = board_entities
        .iter()
        .position(|&e| world.get::<ActiveMission>(e).is_some())
        .unwrap_or(0);
    mix(&mut h, active_idx as u64);

    for &entity in &board_entities {
        if let Some(p) = world.get::<MissionProgress>(entity) {
            mix(&mut h, p.turns_remaining as u64);
            mix(&mut h, (p.sabotage_progress * 10.0) as u64);
            mix(&mut h, (p.alert_level * 10.0) as u64);
            mix(&mut h, (p.reactor_integrity * 10.0) as u64);
            mix(&mut h, p.unattended_turns as u64);
            mix(&mut h, p.outcome_recorded as u64);
            mix(&mut h, p.result as u64);
        }
    }

    let hero_data: Vec<(u32, f32, f32, f32, f32, bool, bool)> = world
        .resource::<CampaignRoster>()
        .heroes
        .iter()
        .map(|h| {
            (
                h.id, h.loyalty, h.stress, h.fatigue, h.injury, h.active, h.deserter,
            )
        })
        .collect();
    for (id, loyalty, stress, fatigue, injury, active, deserter) in hero_data {
        mix(&mut h, id as u64);
        mix(&mut h, (loyalty * 10.0) as u64);
        mix(&mut h, (stress * 10.0) as u64);
        mix(&mut h, (fatigue * 10.0) as u64);
        mix(&mut h, (injury * 10.0) as u64);
        mix(&mut h, active as u64);
        mix(&mut h, deserter as u64);
    }
    let ledger_len = world.resource::<CampaignLedger>().records.len();
    mix(&mut h, ledger_len as u64);
    h
}
