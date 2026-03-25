//! Messenger system — orders to distant parties are delayed based on distance.
//!
//! Communication takes time: orders sent to far-away parties travel at
//! `MESSENGER_SPEED` tiles per tick. Fast messengers can be hired for 2x speed.
//! Messages passing through hostile territory have a 5% chance of being lost.
//! Watchtower building reduces delivery delay by 30%.
//!
//! Cadence: every 50 ticks.

use crate::headless_campaign::actions::{CampaignAction, StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Base messenger travel speed in tiles per tick.
const MESSENGER_SPEED: f32 = 2.0;

/// Cost to hire a fast messenger (2x speed).
pub const FAST_MESSENGER_COST: f32 = 5.0;

/// Chance a messenger is lost when passing through hostile territory (0–1).
const LOST_CHANCE_HOSTILE: f32 = 0.05;

/// Watchtower delay reduction per tier (0.10 = 10% per tier, max 3 tiers = 30%).
const WATCHTOWER_DELAY_REDUCTION_PER_TIER: f32 = 0.10;

/// Cadence: runs every 50 ticks.
const MESSENGER_INTERVAL: u64 = 1;

/// Deliver pending orders whose arrival_tick has passed.
/// Also rolls for messenger loss in hostile territory.
pub fn tick_messengers(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % MESSENGER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Partition: delivered vs still pending.
    let mut delivered = Vec::new();
    let mut remaining = Vec::new();

    for order in std::mem::take(&mut state.pending_orders) {
        if state.tick >= order.arrival_tick {
            delivered.push(order);
        } else {
            remaining.push(order);
        }
    }

    state.pending_orders = remaining;

    // Execute delivered orders.
    for order in delivered {
        if order.lost {
            events.push(WorldEvent::MessengerLost {
                target: order.target_party_id,
            });
            continue;
        }

        events.push(WorldEvent::MessengerArrived {
            target: order.target_party_id,
        });

        // Execute the stored action (if any).
        if let Some(action) = order.action {
            execute_delivered_action(state, action, events);
        }
    }
}

/// Calculate delivery time in ticks for a messenger from guild base to a party.
pub fn calculate_delivery_ticks(
    state: &CampaignState,
    target_party_id: u32,
    fast: bool,
) -> Option<u64> {
    let party = state.parties.iter().find(|p| p.id == target_party_id)?;
    let base_pos = state.guild.base.position;
    let party_pos = party.position;

    let dx = party_pos.0 - base_pos.0;
    let dy = party_pos.1 - base_pos.1;
    let distance = (dx * dx + dy * dy).sqrt();

    let mut speed = MESSENGER_SPEED;
    if fast {
        speed *= 2.0;
    }

    // Watchtower reduces delay by 10% per tier (max 30%).
    let watchtower_tier = state.guild_buildings.watchtower;
    let reduction = 1.0 - (watchtower_tier as f32 * WATCHTOWER_DELAY_REDUCTION_PER_TIER);

    let ticks = ((distance / speed) * reduction).ceil() as u64;
    // Minimum 1 tick delivery.
    Some(ticks.max(1))
}

/// Check whether the route from guild base to a party passes through hostile
/// territory. Uses a simple check: if the party's region is owned by a faction
/// at war with or hostile to the guild.
pub fn route_passes_hostile_territory(state: &CampaignState, target_party_id: u32) -> bool {
    let party = match state.parties.iter().find(|p| p.id == target_party_id) {
        Some(p) => p,
        None => return false,
    };

    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Find which region the party is in by checking which region's center is
    // closest. Simplified: iterate regions and check if any hostile faction
    // controls territory near the party.
    for region in &state.overworld.regions {
        if region.owner_faction_id == guild_faction_id {
            continue;
        }
        // Check if the owning faction is hostile or at war.
        if let Some(faction) = state.factions.iter().find(|f| f.id == region.owner_faction_id) {
            if matches!(
                faction.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            ) {
                // Check if party is roughly in this region by proximity.
                // Use a simple distance threshold from party position to any
                // location in the region. Since we don't have region geometry,
                // check if the region's threat_level > 50 and the party is
                // beyond a certain distance from base (deeper = more risk).
                let base_pos = state.guild.base.position;
                let dx = party.position.0 - base_pos.0;
                let dy = party.position.1 - base_pos.1;
                let dist_sq = dx * dx + dy * dy;
                // If party is far from base (>10 tiles), hostile regions matter.
                if dist_sq > 100.0 {
                    return true;
                }
            }
        }
    }

    false
}

/// Enqueue a new pending order. Returns the order ID.
pub fn enqueue_order(
    state: &mut CampaignState,
    target_party_id: u32,
    order_description: String,
    action: Option<CampaignAction>,
    fast: bool,
    events: &mut Vec<WorldEvent>,
) -> Option<u32> {
    let delivery_ticks = calculate_delivery_ticks(state, target_party_id, fast)?;
    let arrival_tick = state.tick + delivery_ticks;

    // Check if messenger is lost in hostile territory.
    let through_hostile = route_passes_hostile_territory(state, target_party_id);
    let lost = if through_hostile {
        let roll = lcg_f32(&mut state.rng);
        roll < LOST_CHANCE_HOSTILE
    } else {
        false
    };

    let order_id = state.next_order_id;
    state.next_order_id += 1;

    state.pending_orders.push(PendingOrder {
        id: order_id,
        order: order_description,
        target_party_id,
        sent_tick: state.tick,
        arrival_tick,
        action,
        lost,
    });

    events.push(WorldEvent::MessengerSent {
        target: target_party_id,
    });

    Some(order_id)
}

/// Execute a delivered action on behalf of the messenger system.
/// This is a subset of actions that make sense for delayed delivery.
fn execute_delivered_action(
    state: &mut CampaignState,
    action: CampaignAction,
    events: &mut Vec<WorldEvent>,
) {
    match action {
        CampaignAction::DispatchQuest { quest_id } => {
            // Check quest still exists and is dispatchable.
            if let Some(quest) = state.active_quests.iter().find(|q| q.id == quest_id) {
                if quest.status == ActiveQuestStatus::Preparing && !quest.assigned_pool.is_empty() {
                    // Form party (simplified — mirrors step.rs logic).
                    let party_id = state.next_party_id;
                    state.next_party_id += 1;

                    let quest = state
                        .active_quests
                        .iter_mut()
                        .find(|q| q.id == quest_id)
                        .unwrap();
                    let member_ids = quest.assigned_pool.clone();
                    let target = quest.request.target_position;
                    quest.status = ActiveQuestStatus::InProgress;
                    quest.dispatched_party_id = Some(party_id);

                    for &mid in &member_ids {
                        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                            adv.status = AdventurerStatus::Traveling;
                            adv.party_id = Some(party_id);
                        }
                    }

                    let party = Party {
                        id: party_id,
                        member_ids: member_ids.clone(),
                        position: state.guild.base.position,
                        destination: Some(target),
                        speed: state.config.quest_lifecycle.party_speed,
                        status: PartyStatus::Traveling,
                        supply_level: state.config.quest_lifecycle.party_starting_supply,
                        morale: state.config.quest_lifecycle.party_starting_morale,
                        quest_id: Some(quest_id),
                food_level: 100.0,
                    };

                    events.push(WorldEvent::QuestDispatched {
                        quest_id,
                        party_id,
                        member_count: member_ids.len(),
                    });

                    state.parties.push(party);
                }
            }
        }
        CampaignAction::SendRunner { party_id, payload } => {
            // Deliver the runner payload to the party.
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
                match &payload {
                    super::super::actions::RunnerPayload::Supplies(amount) => {
                        party.supply_level = (party.supply_level + amount).min(100.0);
                    }
                    super::super::actions::RunnerPayload::Message => {
                        party.morale = (party.morale + 10.0).min(100.0);
                    }
                }
            }
        }
        // Other actions are no-ops if they don't apply to distant parties.
        _ => {}
    }
}
