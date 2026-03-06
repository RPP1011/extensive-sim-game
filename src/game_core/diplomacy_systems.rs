use bevy::prelude::*;

use super::flashpoint_helpers::*;
use super::overworld_types::*;
use super::roster_gen::sign_top_recruit;
use super::roster_types::*;
use super::types::*;

pub fn flashpoint_intent_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
    board: Res<MissionBoard>,
    mut mission_query: Query<(&mut MissionData, &mut MissionProgress, &mut MissionTactics)>,
    active_query: Query<Entity, With<ActiveMission>>,
    mut flashpoints: ResMut<FlashpointState>,
    roster: Option<Res<CampaignRoster>>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };
    let intent = if keyboard.just_pressed(KeyCode::Digit1) {
        Some(FlashpointIntent::StealthPush)
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        Some(FlashpointIntent::DirectAssault)
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        Some(FlashpointIntent::CivilianFirst)
    } else {
        None
    };
    let Some(intent) = intent else {
        return;
    };

    // Determine the active mission's slot index.
    let Ok(active_entity) = active_query.get_single() else {
        return;
    };
    let slot = board
        .entities
        .iter()
        .position(|&e| e == active_entity)
        .unwrap_or(0);

    let Some(chain_idx) = flashpoints
        .chains
        .iter()
        .position(|c| !c.completed && c.mission_slot == slot)
    else {
        return;
    };
    let mut chain = flashpoints.chains[chain_idx].clone();
    if chain.intent == intent {
        flashpoints.notice = format!(
            "Flashpoint intent unchanged: {}.",
            flashpoint_intent_label(intent)
        );
        return;
    }
    chain.intent = intent;

    // Apply changes via snapshot round-trip.
    if let Some(&entity) = board.entities.get(slot) {
        if let Ok((mut data, mut progress, mut tactics)) = mission_query.get_mut(entity) {
            let mut snap = MissionSnapshot::from_components(&data, &progress, &tactics);
            configure_flashpoint_stage_mission(
                &mut snap,
                &chain,
                &overworld,
                overworld.map_seed ^ (run_state.global_turn as u64).wrapping_mul(733),
            );
            if let Some(roster) = roster.as_ref() {
                let _ = apply_flashpoint_companion_hook(&mut snap, &chain, roster);
                chain.objective = flashpoint_hook_objective_suffix(&chain, roster)
                    .unwrap_or_else(|| "No companion hook".to_string());
            } else {
                chain.objective = "No companion hook".to_string();
            }
            apply_flashpoint_intent(&mut snap, &chain);
            rewrite_flashpoint_mission_name(&mut snap, &chain, &overworld, roster.as_deref());
            // Write back.
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
            data.mission_name = snap.mission_name.clone();
            data.bound_region_id = snap.bound_region_id;
        }
    }

    flashpoints.notice = format!(
        "Flashpoint intent set to {} on slot {} (keys: 1 stealth, 2 assault, 3 civilian).",
        flashpoint_intent_label(intent),
        slot + 1
    );
    if let Some(log) = event_log.as_mut() {
        push_campaign_event(log, run_state.global_turn, flashpoints.notice.clone());
    }
    flashpoints.chains[chain_idx] = chain;
}

fn relation_to_player(diplomacy: &DiplomacyState, faction_id: usize) -> i32 {
    if faction_id >= diplomacy.relations.len()
        || diplomacy.player_faction_id >= diplomacy.relations.len()
    {
        return 0;
    }
    diplomacy.relations[faction_id][diplomacy.player_faction_id]
}

fn commander_primary_region(
    faction_id: usize,
    overworld: &OverworldMap,
) -> Option<&OverworldRegion> {
    overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == faction_id)
        .max_by(|a, b| a.unrest.total_cmp(&b.unrest).then(a.id.cmp(&b.id)))
}

pub fn generate_commander_intents_system(
    overworld: Res<OverworldMap>,
    board: Res<MissionBoard>,
    mission_query: Query<(&MissionProgress,)>,
    diplomacy: Res<DiplomacyState>,
    mut commanders: ResMut<CommanderState>,
) {
    if commanders.commanders.is_empty() || overworld.regions.is_empty() {
        return;
    }

    // Build slot->pressure lookup.
    let slot_pressure: Vec<Option<f32>> = board
        .entities
        .iter()
        .map(|e| {
            mission_query
                .get(*e)
                .ok()
                .map(|(p,)| p.alert_level + (100.0 - p.reactor_integrity))
        })
        .collect();

    let mut intents = Vec::with_capacity(commanders.commanders.len());
    for commander in &commanders.commanders {
        let Some(region) = commander_primary_region(commander.faction_id, &overworld) else {
            continue;
        };
        let faction_state = overworld.factions.get(commander.faction_id);
        let relation = relation_to_player(&diplomacy, commander.faction_id);
        let mission_slot = region.mission_slot;
        let mission_pressure = mission_slot
            .and_then(|s| slot_pressure.get(s).and_then(|v| *v))
            .unwrap_or(region.unrest);
        let war_focus = faction_state.map(|f| f.war_focus).unwrap_or(0.0);
        let war_target_player = faction_state
            .and_then(|f| f.war_goal_faction_id)
            .map(|id| id == diplomacy.player_faction_id)
            .unwrap_or(false);
        let urgency =
            (mission_pressure * 0.5 + region.unrest * 0.5 + war_focus * 0.15).clamp(0.0, 100.0);

        let kind = if relation >= 20 && commander.cooperation_bias >= 0.55 && urgency >= 30.0 {
            CommanderIntentKind::JointMission
        } else if (commander.aggression >= 0.65 || war_target_player)
            && relation <= 8
            && urgency >= 26.0
        {
            CommanderIntentKind::Raid
        } else if commander.cooperation_bias >= 0.6
            && relation >= 12
            && commander.competence >= 0.68
        {
            CommanderIntentKind::TrainingExchange
        } else if relation >= 18 && commander.competence >= 0.7 {
            CommanderIntentKind::RecruitBorrow
        } else {
            CommanderIntentKind::StabilizeBorder
        };

        intents.push(CommanderIntent {
            faction_id: commander.faction_id,
            region_id: region.id,
            mission_slot,
            urgency,
            kind,
        });
    }
    commanders.intents = intents;
}

fn offer_kind_for_intent(kind: CommanderIntentKind) -> Option<InteractionOfferKind> {
    match kind {
        CommanderIntentKind::JointMission => Some(InteractionOfferKind::JointMission),
        CommanderIntentKind::Raid => Some(InteractionOfferKind::RivalRaid),
        CommanderIntentKind::TrainingExchange => Some(InteractionOfferKind::TrainingLoan),
        CommanderIntentKind::RecruitBorrow => Some(InteractionOfferKind::RecruitBorrow),
        CommanderIntentKind::StabilizeBorder => None,
    }
}

pub fn refresh_interaction_offers_system(
    commanders: Res<CommanderState>,
    overworld: Res<OverworldMap>,
    diplomacy: Res<DiplomacyState>,
    mut board: ResMut<InteractionBoard>,
) {
    let mut offers = Vec::new();
    let mut next_id = board.next_offer_id;
    for intent in &commanders.intents {
        if intent.faction_id == diplomacy.player_faction_id {
            continue;
        }
        let Some(kind) = offer_kind_for_intent(intent.kind) else {
            continue;
        };
        let relation = relation_to_player(&diplomacy, intent.faction_id);
        if matches!(
            kind,
            InteractionOfferKind::JointMission
                | InteractionOfferKind::TrainingLoan
                | InteractionOfferKind::RecruitBorrow
        ) && relation < 8
        {
            continue;
        }
        let faction_name = overworld
            .factions
            .get(intent.faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("Unknown Faction");
        let region_name = overworld
            .regions
            .iter()
            .find(|r| r.id == intent.region_id)
            .map(|r| r.name.as_str())
            .unwrap_or("Unknown Region");
        let summary = match kind {
            InteractionOfferKind::JointMission => {
                format!("{faction_name} proposes a joint strike in {region_name}.")
            }
            InteractionOfferKind::RivalRaid => {
                format!("{faction_name} is preparing a rival raid near {region_name}.")
            }
            InteractionOfferKind::TrainingLoan => {
                format!("{faction_name} offers cross-faction training in {region_name}.")
            }
            InteractionOfferKind::RecruitBorrow => {
                format!("{faction_name} proposes a recruit exchange linked to {region_name}.")
            }
        };
        offers.push(InteractionOffer {
            id: next_id,
            from_faction_id: intent.faction_id,
            region_id: intent.region_id,
            mission_slot: intent.mission_slot,
            kind,
            summary,
        });
        next_id = next_id.saturating_add(1);
    }
    board.offers = offers;
    if board.selected >= board.offers.len() {
        board.selected = board.offers.len().saturating_sub(1);
    }
    board.next_offer_id = next_id;
}

fn adjust_relation(diplomacy: &mut DiplomacyState, a: usize, b: usize, delta: i32) {
    if a >= diplomacy.relations.len() || b >= diplomacy.relations.len() {
        return;
    }
    diplomacy.relations[a][b] = (diplomacy.relations[a][b] + delta).clamp(-100, 100);
    diplomacy.relations[b][a] = (diplomacy.relations[b][a] + delta).clamp(-100, 100);
}

pub fn resolve_interaction_offer(
    offer: &InteractionOffer,
    accepted: bool,
    mission_snapshots: &mut [MissionSnapshot],
    attention: &mut AttentionState,
    roster: &mut CampaignRoster,
    diplomacy: &mut DiplomacyState,
) -> String {
    if !accepted {
        adjust_relation(
            diplomacy,
            offer.from_faction_id,
            diplomacy.player_faction_id,
            -1,
        );
        return format!(
            "Declined offer #{} from faction {}.",
            offer.id, offer.from_faction_id
        );
    }

    match offer.kind {
        InteractionOfferKind::JointMission => {
            if let Some(slot) = offer
                .mission_slot
                .and_then(|s| mission_snapshots.get_mut(s))
            {
                slot.alert_level = (slot.alert_level - 6.0).max(0.0);
                slot.turns_remaining = (slot.turns_remaining + 2).min(45);
            }
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                4,
            );
            "Joint mission accepted: pressure reduced and timeline extended.".to_string()
        }
        InteractionOfferKind::RivalRaid => {
            if let Some(slot) = offer
                .mission_slot
                .and_then(|s| mission_snapshots.get_mut(s))
            {
                slot.sabotage_progress = (slot.sabotage_progress + 5.0).min(slot.sabotage_goal);
                slot.alert_level = (slot.alert_level + 4.0).min(100.0);
            }
            attention.global_energy = (attention.global_energy - 4.0).max(0.0);
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                -6,
            );
            "Rival raid embraced: objective gain at diplomatic and alert cost.".to_string()
        }
        InteractionOfferKind::TrainingLoan => {
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.active && !h.deserter) {
                hero.stress = (hero.stress - 8.0).max(0.0);
                hero.fatigue = (hero.fatigue - 7.0).max(0.0);
                hero.resolve = (hero.resolve + 2.0).min(100.0);
            }
            attention.global_energy = (attention.global_energy - 6.0).max(0.0);
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                3,
            );
            "Training loan accepted: companion readiness improved.".to_string()
        }
        InteractionOfferKind::RecruitBorrow => {
            let signed = sign_top_recruit(roster);
            if let Some(hero) = signed {
                if let Some(last) = roster.heroes.iter_mut().find(|h| h.id == hero.id) {
                    last.loyalty = last.loyalty.min(58.0);
                }
            }
            attention.global_energy = (attention.global_energy - 7.0).max(0.0);
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                2,
            );
            "Recruit borrowing accepted: a provisional companion joins your roster.".to_string()
        }
    }
}

pub fn interaction_offer_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut board: ResMut<InteractionBoard>,
    mission_board: Res<MissionBoard>,
    mut mission_query: Query<(&MissionData, &mut MissionProgress, &mut MissionTactics)>,
    mut attention: ResMut<AttentionState>,
    mut roster: ResMut<CampaignRoster>,
    mut diplomacy: ResMut<DiplomacyState>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };
    if keyboard.just_pressed(KeyCode::KeyO) && !board.offers.is_empty() {
        board.selected = (board.selected + 1) % board.offers.len();
    }
    if keyboard.just_pressed(KeyCode::KeyU) && !board.offers.is_empty() {
        board.selected = if board.selected == 0 {
            board.offers.len() - 1
        } else {
            board.selected - 1
        };
    }
    if board.offers.is_empty() {
        return;
    }

    if keyboard.just_pressed(KeyCode::KeyY) || keyboard.just_pressed(KeyCode::KeyN) {
        let accepted = keyboard.just_pressed(KeyCode::KeyY);
        let selected = board.selected;
        let offer = board.offers[selected].clone();

        // Build snapshots for mutation.
        let mut snapshots: Vec<MissionSnapshot> = mission_board
            .entities
            .iter()
            .filter_map(|e| {
                mission_query
                    .get(*e)
                    .ok()
                    .map(|(d, p, t)| MissionSnapshot::from_components(d, &p, &t))
            })
            .collect();

        board.notice = resolve_interaction_offer(
            &offer,
            accepted,
            &mut snapshots,
            &mut attention,
            &mut roster,
            &mut diplomacy,
        );

        // Write snapshots back to components.
        for (slot, &entity) in mission_board.entities.iter().enumerate() {
            if let Some(snap) = snapshots.get(slot) {
                if let Ok((_, mut progress, mut tactics)) = mission_query.get_mut(entity) {
                    progress.alert_level = snap.alert_level;
                    progress.turns_remaining = snap.turns_remaining;
                    progress.sabotage_progress = snap.sabotage_progress;
                    progress.reactor_integrity = snap.reactor_integrity;
                    tactics.tactical_mode = snap.tactical_mode;
                    tactics.command_cooldown_turns = snap.command_cooldown_turns;
                }
            }
        }

        board.offers.remove(selected);
        if board.selected >= board.offers.len() {
            board.selected = board.offers.len().saturating_sub(1);
        }
    }
}
