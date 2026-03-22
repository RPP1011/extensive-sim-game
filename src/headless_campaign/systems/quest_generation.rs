//! Quest generation — Poisson arrival, checked every tick.
//!
//! New quest requests appear on the guild board based on world state.
//! Mean arrival interval scales with reputation (higher rep → more quests).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Mean ticks between quest arrivals at reputation 50.
const BASE_ARRIVAL_INTERVAL_TICKS: u64 = 150; // ~15s — faster cadence for meaningful campaigns

pub fn tick_quest_generation(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // Poisson check: probability of arrival this tick = 1 / mean_interval
    let rep_factor = (state.guild.reputation / 50.0).max(0.5).min(2.0);
    let mean_interval = (BASE_ARRIVAL_INTERVAL_TICKS as f32 / rep_factor) as u64;
    let threshold = 1.0 / mean_interval.max(1) as f32;

    let roll = lcg_f32(&mut state.rng);
    if roll >= threshold {
        return;
    }

    // Generate a quest request
    let quest_type = match (lcg_next(&mut state.rng) % 6) as u8 {
        0 => QuestType::Combat,
        1 => QuestType::Exploration,
        2 => QuestType::Diplomatic,
        3 => QuestType::Escort,
        4 => QuestType::Rescue,
        _ => QuestType::Gather,
    };

    // Pick a random location as the quest target
    let (target_pos, source_area, distance) = if state.overworld.locations.is_empty() {
        let x = lcg_f32(&mut state.rng) * 100.0 - 50.0;
        let y = lcg_f32(&mut state.rng) * 100.0 - 50.0;
        let dx = x - state.guild.base.position.0;
        let dy = y - state.guild.base.position.1;
        ((x, y), None, (dx * dx + dy * dy).sqrt())
    } else {
        let idx = (lcg_next(&mut state.rng) as usize) % state.overworld.locations.len();
        let loc = &state.overworld.locations[idx];
        let dx = loc.position.0 - state.guild.base.position.0;
        let dy = loc.position.1 - state.guild.base.position.1;
        (loc.position, Some(loc.id), (dx * dx + dy * dy).sqrt())
    };

    // Threat scales with distance, global threat, and campaign progress
    let progress_scaling = 1.0 + state.overworld.campaign_progress * 2.0; // threats grow as campaign progresses
    let base_threat = (15.0 + distance * 0.5 + state.overworld.global_threat_level * 0.5) * progress_scaling;
    let threat_level = (base_threat + lcg_f32(&mut state.rng) * 30.0 - 15.0).clamp(5.0, 100.0);

    // Rewards scale with threat
    let gold_reward = threat_level * 2.0 + lcg_f32(&mut state.rng) * 20.0;
    let rep_reward = (threat_level * 0.1).min(5.0);

    // Pick source faction
    let source_faction = if state.factions.is_empty() {
        None
    } else {
        let idx = (lcg_next(&mut state.rng) as usize) % state.factions.len();
        Some(state.factions[idx].id)
    };

    let id = state.next_quest_id;
    state.next_quest_id += 1;

    let request = QuestRequest {
        id,
        source_faction_id: source_faction,
        source_area_id: source_area,
        quest_type,
        threat_level,
        reward: QuestReward {
            gold: gold_reward,
            reputation: rep_reward,
            relation_faction_id: source_faction,
            relation_change: if source_faction.is_some() { 5.0 } else { 0.0 },
            supply_reward: if matches!(quest_type, QuestType::Gather) {
                15.0
            } else {
                0.0
            },
            potential_loot: matches!(quest_type, QuestType::Combat | QuestType::Exploration),
        },
        distance,
        target_position: target_pos,
        deadline_ms: state.elapsed_ms + 120_000, // 2 minutes to accept
        description: format!("{:?} quest (threat {:.0})", quest_type, threat_level),
        arrived_at_ms: state.elapsed_ms,
    };

    events.push(WorldEvent::QuestRequestArrived {
        request_id: id,
        quest_type,
        threat_level,
    });
    deltas.quests_arrived += 1;

    state.request_board.push(request);
}
