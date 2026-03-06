use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::game_core::{
    CampaignEventLog, CampaignEvent, CampaignParties, CampaignRoster,
    DiplomacyState, OverworldMap, RunState,
};

use super::types::*;
use super::generation::{generate_event, faction_name, region_name};

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Runs each campaign turn and pushes a new event into the queue when
/// conditions are met.
pub fn campaign_event_generation_system(
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
    diplomacy: Res<DiplomacyState>,
    mut queue: ResMut<CampaignEventQueue>,
) {
    let turn = run_state.global_turn;

    if turn == 0 || turn <= queue.last_generated_turn + 2 {
        return;
    }

    let seed = overworld.map_seed ^ (turn as u64).wrapping_mul(0xDEAD_BEEF_CAFE_1234);

    if let Some(mut event) = generate_event(seed, turn, &overworld, &diplomacy) {
        event.id = queue.next_id;
        queue.next_id += 1;
        queue.events.push(event);
    }

    queue.last_generated_turn = turn;
}

/// Draws a small bottom-left egui notification window for the first pending
/// event in the queue.
pub fn draw_event_notification_system(
    mut contexts: EguiContexts,
    mut queue: ResMut<CampaignEventQueue>,
    mut event_log: ResMut<CampaignEventLog>,
    mut roster: ResMut<CampaignRoster>,
    mut parties: ResMut<CampaignParties>,
    mut diplomacy: ResMut<DiplomacyState>,
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
) {
    let pending_idx = queue.events.iter().position(|e| e.accepted.is_none());
    let Some(idx) = pending_idx else { return };

    let title = queue.events[idx].title.clone();
    let description = queue.events[idx].description.clone();
    let event_id = queue.events[idx].id;

    let mut accepted_action: Option<bool> = None;

    egui::Window::new(format!("Event: {}", title))
        .id(egui::Id::new(("campaign_event_notification", event_id)))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(16.0, -16.0))
        .resizable(false)
        .collapsible(false)
        .default_width(320.0)
        .show(contexts.ctx_mut(), |ui| {
            ui.label(&description);
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Accept").clicked() {
                    accepted_action = Some(true);
                }
                if ui.button("Decline").clicked() {
                    accepted_action = Some(false);
                }
            });
        });

    if let Some(accepted) = accepted_action {
        queue.events[idx].accepted = Some(accepted);

        if accepted {
            apply_event_effect(
                &queue.events[idx].kind.clone(),
                &mut event_log,
                &mut roster,
                &mut parties,
                &mut diplomacy,
                &overworld,
                run_state.global_turn,
            );
        } else {
            push_log(
                &mut event_log,
                run_state.global_turn,
                format!("Declined event '{}' (id={}).", title, event_id),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Effect application
// ---------------------------------------------------------------------------

fn push_log(log: &mut CampaignEventLog, turn: u32, summary: String) {
    log.entries.push(CampaignEvent { turn, summary });
    if log.entries.len() > log.max_entries {
        let overflow = log.entries.len() - log.max_entries;
        log.entries.drain(0..overflow);
    }
}

fn apply_event_effect(
    kind: &CampaignEventKind,
    event_log: &mut CampaignEventLog,
    roster: &mut CampaignRoster,
    parties: &mut CampaignParties,
    diplomacy: &mut DiplomacyState,
    map: &OverworldMap,
    turn: u32,
) {
    match kind {
        CampaignEventKind::MerchantOffer { supply_cost, item_description } => {
            let cost = *supply_cost as f32;
            if let Some(party) = parties.parties.iter_mut().find(|p| p.is_player_controlled) {
                party.supply = (party.supply - cost).max(0.0);
                push_log(event_log, turn, format!(
                    "Purchased '{}' from merchant for {} supply (party '{}' supply now {:.1}).",
                    item_description, supply_cost, party.name, party.supply
                ));
            } else {
                push_log(event_log, turn, format!(
                    "Accepted merchant offer for '{}' but no player party found to deduct supply.",
                    item_description
                ));
            }
        }

        CampaignEventKind::DeserterIntel { faction_id, relation_boost, risk_stress } => {
            adjust_relation(diplomacy, diplomacy.player_faction_id, *faction_id, *relation_boost);
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.active && !h.deserter) {
                hero.stress = (hero.stress + risk_stress).min(100.0);
                push_log(event_log, turn, format!(
                    "Accepted deserter intel from {} faction. Relations +{}. {} stress +{:.0}.",
                    faction_name(map, *faction_id), relation_boost, hero.name, risk_stress
                ));
            } else {
                push_log(event_log, turn, format!(
                    "Accepted deserter intel from {} faction. Relations +{}.",
                    faction_name(map, *faction_id), relation_boost
                ));
            }
        }

        CampaignEventKind::PlagueScare { region_id, stress_penalty } => {
            let affected_party_ids: Vec<u32> = parties
                .parties.iter()
                .filter(|p| p.region_id == *region_id)
                .map(|p| p.id)
                .collect();

            let mut affected_heroes: Vec<String> = Vec::new();
            for hero in roster.heroes.iter_mut().filter(|h| h.active && !h.deserter) {
                let leads_affected = affected_party_ids.iter().any(|_pid| {
                    parties.parties.iter().any(|p| p.leader_hero_id == hero.id && p.region_id == *region_id)
                });
                if leads_affected {
                    hero.stress = (hero.stress + stress_penalty).min(100.0);
                    affected_heroes.push(hero.name.clone());
                }
            }

            push_log(event_log, turn, format!(
                "Plague scare in {}. Stress +{:.0} for: {}.",
                region_name(map, *region_id), stress_penalty,
                if affected_heroes.is_empty() { "no heroes present".to_string() } else { affected_heroes.join(", ") }
            ));
        }

        CampaignEventKind::RivalPartySpotted { region_id, reward_supply } => {
            let gain = *reward_supply as f32;
            if let Some(party) = parties.parties.iter_mut().find(|p| p.is_player_controlled) {
                party.supply += gain;
                push_log(event_log, turn, format!(
                    "Confronted rival party near {}. Gained {} supply (party '{}' supply now {:.1}).",
                    region_name(map, *region_id), reward_supply, party.name, party.supply
                ));
            } else {
                push_log(event_log, turn, format!(
                    "Rival party spotted near {} but no player party to receive reward.",
                    region_name(map, *region_id)
                ));
            }
        }

        CampaignEventKind::AllyRequest { faction_id, relation_reward, supply_cost } => {
            adjust_relation(diplomacy, diplomacy.player_faction_id, *faction_id, *relation_reward);
            let cost = *supply_cost as f32;
            if let Some(party) = parties.parties.iter_mut().find(|p| p.is_player_controlled) {
                party.supply = (party.supply - cost).max(0.0);
                push_log(event_log, turn, format!(
                    "Honoured ally request from {}. Relations +{}. Lost {} supply (party '{}' supply now {:.1}).",
                    faction_name(map, *faction_id), relation_reward, supply_cost, party.name, party.supply
                ));
            } else {
                push_log(event_log, turn, format!(
                    "Honoured ally request from {}. Relations +{}.",
                    faction_name(map, *faction_id), relation_reward
                ));
            }
        }

        CampaignEventKind::FactionRumour { faction_id, description } => {
            push_log(event_log, turn, format!(
                "Faction rumour: {} {}", faction_name(map, *faction_id), description
            ));
        }

        CampaignEventKind::AbandonedCache { supply_reward } => {
            let gain = *supply_reward as f32;
            if let Some(party) = parties.parties.iter_mut().find(|p| p.is_player_controlled) {
                party.supply += gain;
                push_log(event_log, turn, format!(
                    "Claimed abandoned cache: +{} supply (party '{}' supply now {:.1}).",
                    supply_reward, party.name, party.supply
                ));
            } else {
                push_log(event_log, turn, format!(
                    "Found abandoned cache (+{} supply) but no player party.", supply_reward
                ));
            }
        }

        CampaignEventKind::StormWarning { turns_of_slow } => {
            push_log(event_log, turn, format!(
                "Storm warning acknowledged. Movement penalty for {} turns (not yet enforced).",
                turns_of_slow
            ));
        }
    }
}

/// Local helper that mirrors game_core's private `adjust_relation`.
fn adjust_relation(diplomacy: &mut DiplomacyState, a: usize, b: usize, delta: i32) {
    if a >= diplomacy.relations.len() || b >= diplomacy.relations.len() {
        return;
    }
    diplomacy.relations[a][b] = (diplomacy.relations[a][b] + delta).clamp(-100, 100);
    diplomacy.relations[b][a] = (diplomacy.relations[b][a] + delta).clamp(-100, 100);
}
