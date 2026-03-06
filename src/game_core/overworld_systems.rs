use bevy::prelude::*;

use super::campaign_systems::ensure_faction_mission_slots;
use super::generation::*;
use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;

pub fn update_faction_war_goals_system(
    diplomacy: Res<DiplomacyState>,
    mut overworld: ResMut<OverworldMap>,
) {
    let n = overworld.factions.len();
    if n < 2 {
        return;
    }
    let mut border_counts = vec![vec![0_u32; n]; n];
    for region in &overworld.regions {
        for neighbor in &region.neighbors {
            if let Some(other) = overworld.regions.get(*neighbor) {
                let a = region.owner_faction_id;
                let b = other.owner_faction_id;
                if a < n && b < n && a != b {
                    border_counts[a][b] = border_counts[a][b].saturating_add(1);
                }
            }
        }
    }

    for faction_id in 0..n {
        let mut best_target = None;
        let mut best_score = f32::MIN;
        for target in 0..n {
            if target == faction_id {
                continue;
            }
            let relation = diplomacy
                .relations
                .get(faction_id)
                .and_then(|row| row.get(target))
                .copied()
                .unwrap_or(0);
            let hostility = (-relation) as f32;
            let border = border_counts[faction_id][target] as f32 * 6.0;
            let target_strength = overworld
                .factions
                .get(target)
                .map(|f| f.strength)
                .unwrap_or(50.0);
            let own_strength = overworld.factions[faction_id].strength;
            let strength_tension = ((target_strength - own_strength) * 0.06).abs();
            let score = hostility + border + strength_tension;
            if score > best_score {
                best_score = score;
                best_target = Some(target);
            }
        }

        let cohesion = overworld.factions[faction_id].cohesion;
        let strategic_drive = (best_score + (100.0 - cohesion) * 0.22).clamp(0.0, 100.0);
        overworld.factions[faction_id].war_goal_faction_id = best_target;
        overworld.factions[faction_id].war_focus = strategic_drive;
    }
}

pub fn overworld_ai_border_pressure_system(
    run_state: Res<RunState>,
    mut overworld: ResMut<OverworldMap>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 || overworld.regions.is_empty() || overworld.factions.len() < 2 {
        return;
    }

    let n_regions = overworld.regions.len();
    let mut control_delta = vec![0.0_f32; n_regions];
    let mut unrest_delta = vec![0.0_f32; n_regions];
    for rid in 0..n_regions {
        let owner_a = overworld.regions[rid].owner_faction_id;
        for neighbor in overworld.regions[rid].neighbors.clone() {
            if rid >= neighbor {
                continue;
            }
            let owner_b = overworld.regions[neighbor].owner_faction_id;
            if owner_a == owner_b {
                continue;
            }
            let Some(fa) = overworld.factions.get(owner_a) else {
                continue;
            };
            let Some(fb) = overworld.factions.get(owner_b) else {
                continue;
            };
            let roam_a = fa
                .vassals
                .iter()
                .filter(|v| v.post == VassalPost::Roaming)
                .map(|v| v.martial * 0.012)
                .sum::<f32>();
            let roam_b = fb
                .vassals
                .iter()
                .filter(|v| v.post == VassalPost::Roaming)
                .map(|v| v.martial * 0.012)
                .sum::<f32>();
            let power_a = fa.strength * 0.66 + fa.cohesion * 0.34 + roam_a + fa.war_focus * 0.18;
            let power_b = fb.strength * 0.66 + fb.cohesion * 0.34 + roam_b + fb.war_focus * 0.18;
            let jitter = (rand01(
                overworld.map_seed ^ run_state.global_turn as u64,
                60_000 + rid as u64 * 131 + neighbor as u64 * 17,
            ) - 0.5)
                * 1.5;
            let pressure = ((power_a - power_b) * 0.02 + jitter).clamp(-6.5, 6.5);
            if pressure > 0.0 {
                control_delta[rid] += pressure * 0.45;
                unrest_delta[rid] -= pressure * 0.30;
                control_delta[neighbor] -= pressure * 0.64;
                unrest_delta[neighbor] += pressure * 0.72;
            } else if pressure < 0.0 {
                let p = -pressure;
                control_delta[rid] -= p * 0.64;
                unrest_delta[rid] += p * 0.72;
                control_delta[neighbor] += p * 0.45;
                unrest_delta[neighbor] -= p * 0.30;
            }
        }
    }

    for rid in 0..n_regions {
        let region = &mut overworld.regions[rid];
        region.control = (region.control + control_delta[rid]).clamp(0.0, 100.0);
        region.unrest = (region.unrest + unrest_delta[rid]).clamp(0.0, 100.0);
    }

    let mut owned_counts = vec![0_u32; overworld.factions.len()];
    for region in &overworld.regions {
        if region.owner_faction_id < owned_counts.len() {
            owned_counts[region.owner_faction_id] =
                owned_counts[region.owner_faction_id].saturating_add(1);
        }
    }
    let mut changed_ownership = false;
    let mut pending_events = Vec::new();
    for rid in 0..n_regions {
        let owner = overworld.regions[rid].owner_faction_id;
        if overworld.regions[rid].control >= 22.0 || overworld.regions[rid].unrest <= 60.0 {
            continue;
        }
        if owner >= owned_counts.len() || owned_counts[owner] <= 1 {
            continue;
        }
        let mut contender = owner;
        let mut contender_score = f32::MIN;
        for neighbor in &overworld.regions[rid].neighbors {
            let maybe_owner = overworld
                .regions
                .get(*neighbor)
                .map(|r| r.owner_faction_id)
                .unwrap_or(owner);
            if maybe_owner == owner {
                continue;
            }
            let score = overworld
                .factions
                .get(maybe_owner)
                .map(|f| f.strength + f.war_focus * 0.4 + f.cohesion * 0.2)
                .unwrap_or(0.0);
            if score > contender_score {
                contender_score = score;
                contender = maybe_owner;
            }
        }
        if contender != owner {
            let owner_power = overworld
                .factions
                .get(owner)
                .map(|f| f.strength + f.cohesion * 0.2)
                .unwrap_or(0.0);
            if contender_score > owner_power + 8.0 {
                let region_name = overworld.regions[rid].name.clone();
                let old_owner = overworld
                    .factions
                    .get(owner)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", owner));
                let new_owner = overworld
                    .factions
                    .get(contender)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", contender));
                overworld.regions[rid].owner_faction_id = contender;
                overworld.regions[rid].control = 52.0;
                overworld.regions[rid].unrest = 48.0;
                if owner < owned_counts.len() {
                    owned_counts[owner] = owned_counts[owner].saturating_sub(1);
                }
                if contender < owned_counts.len() {
                    owned_counts[contender] = owned_counts[contender].saturating_add(1);
                }
                changed_ownership = true;
                pending_events.push(format!(
                    "Border shift: {} captured {} from {}.",
                    new_owner, region_name, old_owner
                ));
            }
        }
    }
    if changed_ownership {
        ensure_faction_mission_slots(&mut overworld);
    }
    if let Some(log) = event_log.as_mut() {
        for event in pending_events {
            push_campaign_event(log, run_state.global_turn, event);
        }
    }
}

pub fn overworld_intel_update_system(
    run_state: Res<RunState>,
    board: Res<MissionBoard>,
    mission_query: Query<(&MissionProgress,)>,
    mut overworld: ResMut<OverworldMap>,
) {
    if run_state.global_turn == 0 || overworld.regions.is_empty() {
        return;
    }
    let current = overworld.current_region.min(overworld.regions.len() - 1);
    let selected = overworld.selected_region.min(overworld.regions.len() - 1);

    let slot_pressure: Vec<Option<(f32, f32)>> = board
        .entities
        .iter()
        .map(|e| {
            mission_query
                .get(*e)
                .ok()
                .map(|(p,)| (p.alert_level, p.reactor_integrity))
        })
        .collect();

    for rid in 0..overworld.regions.len() {
        let region = &mut overworld.regions[rid];
        let mission_bonus = region
            .mission_slot
            .and_then(|slot| slot_pressure.get(slot))
            .and_then(|v| *v)
            .map(|(alert, integrity)| alert * 0.015 + (100.0 - integrity) * 0.01)
            .unwrap_or(0.0);
        let decay = 0.8 + region.unrest * 0.012;
        region.intel_level = (region.intel_level - decay + mission_bonus).clamp(0.0, 100.0);
        if region.owner_faction_id == 0 {
            region.intel_level = region.intel_level.max(30.0);
        }
    }

    let neighbors = overworld.regions[current].neighbors.clone();
    if let Some(region) = overworld.regions.get_mut(current) {
        region.intel_level = (region.intel_level + 16.0).clamp(0.0, 100.0);
    }
    for neighbor in neighbors {
        if let Some(region) = overworld.regions.get_mut(neighbor) {
            region.intel_level = (region.intel_level + 8.0).clamp(0.0, 100.0);
        }
    }
    if let Some(region) = overworld.regions.get_mut(selected) {
        region.intel_level = (region.intel_level + 5.0).clamp(0.0, 100.0);
    }
}

pub fn overworld_sync_from_missions_system(
    board: Res<MissionBoard>,
    mission_query: Query<(&MissionProgress,)>,
    mut overworld: ResMut<OverworldMap>,
) {
    let mut faction_bonus = vec![0.0_f32; overworld.factions.len()];
    let mut region_manager_bonus = vec![0.0_f32; overworld.regions.len()];
    for faction in &overworld.factions {
        let patrol_weight = faction
            .vassals
            .iter()
            .filter(|v| {
                v.post == VassalPost::Roaming
                    && (v.specialty == VassalSpecialty::Patrol
                        || v.specialty == VassalSpecialty::Escort)
            })
            .count() as f32;
        let roaming_quality = faction
            .vassals
            .iter()
            .filter(|v| v.post == VassalPost::Roaming)
            .map(|v| v.martial * 0.008 + v.loyalty * 0.004)
            .sum::<f32>();
        for v in faction
            .vassals
            .iter()
            .filter(|v| v.post == VassalPost::ZoneManager)
        {
            if v.home_region_id < region_manager_bonus.len() {
                region_manager_bonus[v.home_region_id] += v.martial * 0.012 + v.loyalty * 0.006;
            }
        }
        faction_bonus[faction.id] =
            patrol_weight * 0.22 + faction.cohesion * 0.01 + roaming_quality;
    }

    let slot_progress: Vec<Option<MissionProgress>> = board
        .entities
        .iter()
        .map(|e| mission_query.get(*e).ok().map(|(p,)| p.clone()))
        .collect();

    for region in &mut overworld.regions {
        let Some(slot) = region.mission_slot else {
            let bonus = faction_bonus
                .get(region.owner_faction_id)
                .copied()
                .unwrap_or(0.0);
            region.unrest = (region.unrest + 0.2 - bonus * 0.08).clamp(0.0, 100.0);
            region.control = (100.0 - region.unrest).clamp(0.0, 100.0);
            continue;
        };
        let Some(Some(mission)) = slot_progress.get(slot) else {
            continue;
        };
        let bonus = faction_bonus
            .get(region.owner_faction_id)
            .copied()
            .unwrap_or(0.0);
        let manager_bonus = region_manager_bonus.get(region.id).copied().unwrap_or(0.0);
        let pressure = (mission.alert_level + (100.0 - mission.reactor_integrity)) * 0.5;
        let progress_relief = mission.sabotage_progress * 0.08;
        let mut unrest = (pressure * 0.6 - progress_relief - bonus * 0.22 - manager_bonus * 0.2)
            .clamp(0.0, 100.0);
        if mission.result == MissionResult::Victory {
            unrest = (unrest - 12.0).max(0.0);
        } else if mission.result == MissionResult::Defeat {
            unrest = (unrest + 12.0).min(100.0);
        }
        region.unrest = unrest;
        region.control = (100.0 - unrest).clamp(0.0, 100.0);
    }
}

pub fn overworld_faction_autonomy_system(
    run_state: Res<RunState>,
    mut overworld: ResMut<OverworldMap>,
) {
    if run_state.global_turn == 0 || overworld.factions.is_empty() {
        return;
    }

    let n = overworld.factions.len();
    let mut unrest_sum = vec![0.0_f32; n];
    let mut control_sum = vec![0.0_f32; n];
    let mut counts = vec![0_u32; n];
    let mut owned_by_faction = vec![Vec::<usize>::new(); n];
    for region in &overworld.regions {
        if region.owner_faction_id < n {
            unrest_sum[region.owner_faction_id] += region.unrest;
            control_sum[region.owner_faction_id] += region.control;
            counts[region.owner_faction_id] += 1;
            owned_by_faction[region.owner_faction_id].push(region.id);
        }
    }

    let mut next_vassal_id = overworld.next_vassal_id;
    for faction in &mut overworld.factions {
        let c = counts[faction.id].max(1) as f32;
        let avg_unrest = unrest_sum[faction.id] / c;
        let avg_control = control_sum[faction.id] / c;
        faction.cohesion =
            (faction.cohesion + (avg_control - avg_unrest) * 0.015).clamp(10.0, 95.0);
        faction.strength =
            (faction.strength + (avg_control * 0.03) - (avg_unrest * 0.025)).clamp(30.0, 180.0);
        let owned = owned_by_faction[faction.id].clone();
        rebalance_faction_vassals(faction, &owned, &mut next_vassal_id);
    }
    overworld.next_vassal_id = next_vassal_id;
}
