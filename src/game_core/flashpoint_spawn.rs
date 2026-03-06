use bevy::prelude::*;

use super::flashpoint_helpers::*;
use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;

pub fn pressure_spawn_missions_system(
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
    board: Res<MissionBoard>,
    mut mission_query: Query<(&mut MissionData, &mut MissionProgress, &mut MissionTactics)>,
    roster: Option<Res<CampaignRoster>>,
    mut flashpoints: Option<ResMut<FlashpointState>>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0
        || overworld.regions.is_empty()
        || overworld.factions.is_empty()
        || board.entities.is_empty()
    {
        return;
    }

    let max_slot = usize::min(overworld.factions.len(), board.entities.len());
    for slot in 0..max_slot {
        let Some(&entity) = board.entities.get(slot) else {
            continue;
        };
        let Ok((mut data, mut progress, mut tactics)) = mission_query.get_mut(entity) else {
            continue;
        };

        let Some(region) = overworld
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
        else {
            continue;
        };
        let owner_id = region.owner_faction_id.min(overworld.factions.len() - 1);
        let owner = &overworld.factions[owner_id];
        let contested_edges = region
            .neighbors
            .iter()
            .filter(|n| {
                overworld
                    .regions
                    .get(**n)
                    .map(|r| r.owner_faction_id != owner_id)
                    .unwrap_or(false)
            })
            .count() as f32;
        let pressure =
            (region.unrest * 0.64 + (100.0 - region.control) * 0.36 + contested_edges * 10.0)
                .clamp(0.0, 100.0);

        let active_chain = flashpoints.as_ref().and_then(|state| {
            state
                .chains
                .iter()
                .find(|chain| !chain.completed && chain.mission_slot == slot)
                .cloned()
        });
        if let Some(ref chain) = active_chain {
            data.bound_region_id = Some(chain.region_id);
            if progress.result == MissionResult::InProgress && progress.mission_active {
                progress.alert_level = progress
                    .alert_level
                    .max((pressure * (0.38 + chain.stage as f32 * 0.04)).clamp(18.0, 96.0));
                progress.reactor_integrity = progress
                    .reactor_integrity
                    .min((100.0 - pressure * 0.18).clamp(18.0, 100.0));
                continue;
            }
        }
        let needs_replace = data.bound_region_id != Some(region.id)
            || progress.result != MissionResult::InProgress
            || !progress.mission_active
            || (pressure >= 74.0 && progress.alert_level <= pressure * 0.52)
            || (pressure >= 82.0 && progress.turns_remaining > 18);

        let can_start_flashpoint = active_chain.is_none()
            && pressure >= FLASHPOINT_TRIGGER_PRESSURE
            && contested_edges >= 1.0
            && needs_replace;
        if can_start_flashpoint {
            let Some(attacker_id) = pick_flashpoint_attacker(region, &overworld) else {
                continue;
            };
            if let Some(state) = flashpoints.as_mut() {
                let chain_id = state.next_id;
                state.next_id = state.next_id.saturating_add(1);
                let companion_hook_hero_id: Option<u32> = None;
                let chain = FlashpointChain {
                    id: chain_id,
                    mission_slot: slot,
                    region_id: region.id,
                    attacker_faction_id: attacker_id,
                    defender_faction_id: owner_id,
                    stage: 1,
                    completed: false,
                    companion_hook_hero_id,
                    intent: FlashpointIntent::StealthPush,
                    objective: String::new(),
                };
                let mut snap = build_pressure_mission_snapshot(
                    overworld.map_seed,
                    run_state.global_turn,
                    slot,
                    region,
                    owner,
                    pressure,
                );
                configure_flashpoint_stage_mission(
                    &mut snap,
                    &chain,
                    &overworld,
                    overworld.map_seed ^ run_state.global_turn as u64,
                );
                let mut chain = chain;
                if let Some(roster) = roster.as_ref() {
                    let _ = apply_flashpoint_companion_hook(&mut snap, &chain, roster);
                    chain.objective = flashpoint_hook_objective_suffix(&chain, roster)
                        .unwrap_or_else(|| "No companion hook".to_string());
                }
                apply_flashpoint_intent(&mut snap, &chain);
                rewrite_flashpoint_mission_name(&mut snap, &chain, &overworld, roster.as_deref());
                data.bound_region_id = snap.bound_region_id;
                data.mission_name = snap.mission_name;
                progress.mission_active = snap.mission_active;
                progress.result = snap.result;
                progress.turns_remaining = snap.turns_remaining;
                progress.reactor_integrity = snap.reactor_integrity;
                progress.sabotage_progress = snap.sabotage_progress;
                progress.sabotage_goal = snap.sabotage_goal;
                progress.alert_level = snap.alert_level;
                progress.room_index = snap.room_index;
                progress.unattended_turns = snap.unattended_turns;
                progress.outcome_recorded = snap.outcome_recorded;
                tactics.tactical_mode = snap.tactical_mode;
                tactics.command_cooldown_turns = snap.command_cooldown_turns;
                let attacker_name = overworld
                    .factions
                    .get(attacker_id)
                    .map(|f| f.name.as_str())
                    .unwrap_or("Rivals");
                state.notice = format!(
                    "Flashpoint opened in {}: {} pushes against {}.",
                    region.name, attacker_name, owner.name
                );
                state.chains.push(chain);
                if let Some(log) = event_log.as_mut() {
                    push_campaign_event(log, run_state.global_turn, state.notice.clone());
                }
                continue;
            }
        }

        if needs_replace {
            let snap = build_pressure_mission_snapshot(
                overworld.map_seed,
                run_state.global_turn,
                slot,
                region,
                owner,
                pressure,
            );
            progress.mission_active = snap.mission_active;
            progress.result = snap.result;
            progress.turns_remaining = snap.turns_remaining;
            progress.reactor_integrity = snap.reactor_integrity;
            progress.sabotage_progress = snap.sabotage_progress;
            progress.sabotage_goal = snap.sabotage_goal;
            progress.alert_level = snap.alert_level;
            progress.room_index = snap.room_index;
            progress.unattended_turns = snap.unattended_turns;
            progress.outcome_recorded = snap.outcome_recorded;
            tactics.tactical_mode = snap.tactical_mode;
            tactics.command_cooldown_turns = snap.command_cooldown_turns;
            data.bound_region_id = snap.bound_region_id;
            data.mission_name = snap.mission_name;
            continue;
        }

        data.bound_region_id = Some(region.id);
        progress.alert_level = progress.alert_level.max((pressure * 0.38).clamp(8.0, 92.0));
        progress.reactor_integrity = progress
            .reactor_integrity
            .min((100.0 - pressure * 0.2).clamp(20.0, 100.0));
    }
}
