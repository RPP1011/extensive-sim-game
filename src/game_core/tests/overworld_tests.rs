use bevy::prelude::*;

use super::helpers::*;
use super::super::campaign_systems::*;
use super::super::generation::target_vassal_count;
use super::super::overworld_nav::*;
use super::super::overworld_systems::*;
use super::super::overworld_types::*;
use super::super::roster_types::*;
use super::super::types::*;

#[test]
fn bootstrap_parties_creates_player_and_delegated_party() {
    let roster = CampaignRoster::default();
    let map = OverworldMap::default();
    let parties = bootstrap_campaign_parties(&roster, &map);
    assert!(!parties.parties.is_empty());
    assert!(parties.parties.iter().any(|p| p.is_player_controlled));
    assert!(parties.parties.len() >= 2);
}

#[test]
fn delegated_party_patrol_order_moves_over_time() {
    let roster = CampaignRoster::default();
    let map = OverworldMap::default();
    let parties = bootstrap_campaign_parties(&roster, &map);
    let delegated_id = parties
        .parties
        .iter()
        .find(|p| !p.is_player_controlled)
        .map(|p| p.id)
        .expect("delegated party");
    let start_region = parties
        .parties
        .iter()
        .find(|p| p.id == delegated_id)
        .map(|p| p.region_id)
        .expect("party region");
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 6 });
    world.insert_resource(map);
    world.insert_resource(parties);
    let mut schedule = Schedule::default();
    schedule.add_systems(campaign_party_orders_system);
    schedule.run(&mut world);
    let parties = world.resource::<CampaignParties>();
    let moved_region = parties
        .parties
        .iter()
        .find(|p| p.id == delegated_id)
        .map(|p| p.region_id)
        .expect("party region");
    assert_ne!(moved_region, start_region);
}

#[test]
fn delegated_party_patrol_target_moves_toward_target_region() {
    let roster = CampaignRoster::default();
    let map = OverworldMap::default();
    let mut parties = bootstrap_campaign_parties(&roster, &map);
    let delegated_idx = parties
        .parties
        .iter()
        .position(|p| !p.is_player_controlled)
        .expect("delegated party");
    let start_region = parties.parties[delegated_idx].region_id;
    let target_region = map.regions[start_region]
        .neighbors
        .first()
        .copied()
        .expect("neighbor target exists");
    parties.parties[delegated_idx].order = PartyOrderKind::PatrolNearby;
    parties.parties[delegated_idx].order_target_region_id = Some(target_region);
    let mut world = World::new();
    world.insert_resource(RunState { global_turn: 6 });
    world.insert_resource(map);
    world.insert_resource(parties);
    let mut schedule = Schedule::default();
    schedule.add_systems(campaign_party_orders_system);
    schedule.run(&mut world);
    let parties = world.resource::<CampaignParties>();
    let moved_region = parties.parties[delegated_idx].region_id;
    assert_eq!(moved_region, target_region);
}

#[test]
fn overworld_travel_moves_focus_to_linked_region_mission() {
    let mut overworld = OverworldMap::default();
    let mut attention = AttentionState::default();
    let source = overworld.current_region;
    let target = overworld.regions[source].neighbors[0];
    overworld.selected_region = target;
    let slot = try_overworld_travel(&mut overworld, &mut attention);
    assert!(slot.is_some() || overworld.regions[target].mission_slot.is_none());
    assert_eq!(overworld.current_region, target);
    if let Some(expected_slot) = overworld.regions[target].mission_slot {
        assert_eq!(slot, Some(expected_slot));
    }
    assert!(attention.global_energy < attention.max_energy);
}

#[test]
fn overworld_travel_blocks_when_not_neighbor_or_low_energy() {
    let mut overworld = OverworldMap::default();
    let mut attention = AttentionState::default();
    let source = overworld.current_region;
    let non_neighbor = overworld
        .regions
        .iter()
        .find(|r| r.id != source && !overworld.regions[source].neighbors.contains(&r.id))
        .map(|r| r.id)
        .expect("non-neighbor region");
    overworld.selected_region = non_neighbor;
    assert!(try_overworld_travel(&mut overworld, &mut attention).is_none());
    overworld.selected_region = overworld.regions[source].neighbors[0];
    attention.global_energy = 0.0;
    assert!(try_overworld_travel(&mut overworld, &mut attention).is_none());
}

#[test]
fn overworld_sync_tracks_mission_pressure() {
    let mut world = World::new();
    world.insert_resource(MissionBoard::default());
    world.insert_resource(OverworldMap::default());
    spawn_test_missions(&mut world);
    set_nth_progress(&mut world, 1, |p| {
        p.alert_level = 70.0;
        p.reactor_integrity = 40.0;
        p.sabotage_progress = 20.0;
    });
    let mut schedule = Schedule::default();
    schedule.add_systems(overworld_sync_from_missions_system);
    schedule.run(&mut world);
    let overworld = world.resource::<OverworldMap>();
    let linked = overworld
        .regions
        .iter()
        .find(|r| r.mission_slot == Some(1))
        .expect("region");
    assert!(linked.unrest >= 20.0);
    assert!(linked.control <= 80.0);
}

#[test]
fn faction_vassal_count_scales_with_strength() {
    assert!(target_vassal_count(40.0) < target_vassal_count(120.0));
}

#[test]
fn faction_autonomy_rebalances_vassals_after_strength_shift() {
    let mut world = World::new();
    let mut map = OverworldMap::default();
    map.factions[0].strength = 150.0;
    map.factions[0].vassals.clear();
    world.insert_resource(map);
    world.insert_resource(RunState { global_turn: 2 });
    let mut schedule = Schedule::default();
    schedule.add_systems(overworld_faction_autonomy_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    assert!(map.factions[0].vassals.len() >= 7);
}

#[test]
fn overworld_default_is_deterministic_for_factions_and_vassals() {
    let a = OverworldMap::default();
    let b = OverworldMap::default();
    assert_eq!(a.map_seed, b.map_seed);
    assert_eq!(a.regions.len(), b.regions.len());
    for idx in 0..a.regions.len() {
        let ar = &a.regions[idx];
        let br = &b.regions[idx];
        assert_eq!(ar.name, br.name);
        assert_eq!(ar.neighbors, br.neighbors);
        assert_eq!(ar.owner_faction_id, br.owner_faction_id);
        assert_eq!(ar.mission_slot, br.mission_slot);
    }
    assert_eq!(a.factions.len(), b.factions.len());
    for idx in 0..a.factions.len() {
        let af = &a.factions[idx];
        let bf = &b.factions[idx];
        assert_eq!(af.name, bf.name);
        assert_eq!(af.vassals.len(), bf.vassals.len());
        let an = af
            .vassals
            .iter()
            .map(|v| v.name.clone())
            .collect::<Vec<_>>();
        let bn = bf
            .vassals
            .iter()
            .map(|v| v.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(an, bn);
    }
}

#[test]
fn factions_have_stationary_zone_managers() {
    let map = OverworldMap::default();
    for faction in &map.factions {
        let managers = faction
            .vassals
            .iter()
            .filter(|v| v.post == VassalPost::ZoneManager)
            .collect::<Vec<_>>();
        assert!(!managers.is_empty());
        for manager in managers {
            let owns_home = map
                .regions
                .iter()
                .any(|r| r.id == manager.home_region_id && r.owner_faction_id == faction.id);
            assert!(owns_home);
        }
    }
}

#[test]
fn seeded_overworld_changes_with_seed() {
    let a = OverworldMap::from_seed(11);
    let b = OverworldMap::from_seed(12);
    assert_eq!(a.regions.len(), b.regions.len());
    let owner_vec_a = a
        .regions
        .iter()
        .map(|r| r.owner_faction_id)
        .collect::<Vec<_>>();
    let owner_vec_b = b
        .regions
        .iter()
        .map(|r| r.owner_faction_id)
        .collect::<Vec<_>>();
    assert_ne!(owner_vec_a, owner_vec_b);
}

#[test]
fn seeded_overworld_has_bidirectional_neighbors_and_faction_presence() {
    let map = OverworldMap::from_seed(0x1234_5678);
    for region in &map.regions {
        assert!(!region.neighbors.is_empty());
        for neighbor in &region.neighbors {
            let other = map.regions.get(*neighbor).expect("neighbor exists");
            assert!(other.neighbors.contains(&region.id));
        }
    }
    for faction_id in 0..map.factions.len() {
        assert!(map.regions.iter().any(|r| r.owner_faction_id == faction_id));
    }
}

#[test]
fn war_goals_are_assigned_to_other_factions() {
    let mut world = World::new();
    world.insert_resource(OverworldMap::default());
    world.insert_resource(DiplomacyState::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(update_faction_war_goals_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    for faction in &map.factions {
        let goal = faction.war_goal_faction_id.expect("war goal");
        assert_ne!(goal, faction.id);
        assert!(goal < map.factions.len());
        assert!((0.0..=100.0).contains(&faction.war_focus));
    }
}

#[test]
fn border_pressure_is_deterministic() {
    let mut world_a = World::new();
    world_a.insert_resource(OverworldMap::default());
    world_a.insert_resource(RunState { global_turn: 6 });
    let mut schedule_a = Schedule::default();
    schedule_a.add_systems(overworld_ai_border_pressure_system);
    schedule_a.run(&mut world_a);
    let map_a = world_a.resource::<OverworldMap>();
    let sig_a = map_a
        .regions
        .iter()
        .map(|r| {
            format!(
                "{}:{:.1}:{:.1}:{}",
                r.id, r.unrest, r.control, r.owner_faction_id
            )
        })
        .collect::<Vec<_>>();

    let mut world_b = World::new();
    world_b.insert_resource(OverworldMap::default());
    world_b.insert_resource(RunState { global_turn: 6 });
    let mut schedule_b = Schedule::default();
    schedule_b.add_systems(overworld_ai_border_pressure_system);
    schedule_b.run(&mut world_b);
    let map_b = world_b.resource::<OverworldMap>();
    let sig_b = map_b
        .regions
        .iter()
        .map(|r| {
            format!(
                "{}:{:.1}:{:.1}:{}",
                r.id, r.unrest, r.control, r.owner_faction_id
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(sig_a, sig_b);
}

#[test]
fn border_pressure_can_shift_region_ownership_when_defender_has_depth() {
    let mut world = World::new();
    let mut map = OverworldMap::default();
    let mut chosen = None;
    for rid in 0..map.regions.len() {
        let owner = map.regions[rid].owner_faction_id;
        for neighbor in &map.regions[rid].neighbors {
            let attacker = map.regions[*neighbor].owner_faction_id;
            if attacker == owner {
                continue;
            }
            let owner_count = map
                .regions
                .iter()
                .filter(|r| r.owner_faction_id == owner)
                .count();
            if owner_count > 1 {
                chosen = Some((rid, owner, attacker));
                break;
            }
        }
        if chosen.is_some() {
            break;
        }
    }
    let (target_rid, defender, attacker) = chosen.expect("border candidate");
    map.regions[target_rid].control = 5.0;
    map.regions[target_rid].unrest = 92.0;
    map.factions[attacker].strength = 175.0;
    map.factions[attacker].cohesion = 90.0;
    map.factions[attacker].war_focus = 95.0;
    map.factions[defender].strength = 38.0;
    map.factions[defender].cohesion = 18.0;
    map.factions[defender].war_focus = 5.0;
    world.insert_resource(map);
    world.insert_resource(RunState { global_turn: 9 });
    let mut schedule = Schedule::default();
    schedule.add_systems(overworld_ai_border_pressure_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    assert_eq!(map.regions[target_rid].owner_faction_id, attacker);
}

#[test]
fn intel_update_prioritizes_current_region_and_neighbors() {
    let mut world = World::new();
    let mut map = OverworldMap::default();
    for region in &mut map.regions {
        region.intel_level = 20.0;
    }
    let current = map.current_region;
    let neighbor = map.regions[current].neighbors[0];
    let far = map
        .regions
        .iter()
        .find(|r| r.id != current && !map.regions[current].neighbors.contains(&r.id))
        .map(|r| r.id)
        .expect("far region");
    map.selected_region = far;
    world.insert_resource(map);
    world.insert_resource(RunState { global_turn: 4 });
    world.insert_resource(MissionBoard::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(overworld_intel_update_system);
    schedule.run(&mut world);
    let map = world.resource::<OverworldMap>();
    assert!(map.regions[current].intel_level > map.regions[far].intel_level);
    assert!(map.regions[neighbor].intel_level > map.regions[far].intel_level);
}
