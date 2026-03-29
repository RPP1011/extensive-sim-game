//! Quest generation — Poisson arrival, checked every tick.
//!
//! New quest requests appear on the guild board based on world state.
//! Mean arrival interval scales with reputation (higher rep → more quests).
//! Uses the narrative grammar walker (`quest_gen`) to produce context-aware,
//! thematically coherent quests.

use crate::actions::{StepDeltas, WorldEvent};
use crate::quest_gen;
use crate::state::*;

pub fn tick_quest_generation(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.quest_generation;

    // Poisson check: probability of arrival this tick = 1 / mean_interval
    let rep_factor = (state.guild.reputation / cfg.reputation_scaling_center)
        .max(cfg.reputation_factor_min)
        .min(cfg.reputation_factor_max);
    let mean_interval = (cfg.base_arrival_interval_ticks as f32 / rep_factor) as u64;
    let threshold = 1.0 / mean_interval.max(1) as f32;

    let roll = lcg_f32(&mut state.rng);
    if roll >= threshold {
        return;
    }

    // Pick an adventurer to contextualize the quest (0 = guild-wide)
    let adv_id = if state.adventurers.is_empty() {
        0
    } else {
        let idx = (lcg_next(&mut state.rng) as usize) % state.adventurers.len();
        state.adventurers[idx].id
    };

    // Generate quest via grammar walker
    // Copy rng out to avoid simultaneous borrow of state + state.rng
    let mut rng = state.rng;
    let (mut request, choice, _narrative) =
        quest_gen::generate_quest(state, adv_id, &mut rng);
    state.rng = rng;

    // Assign quest ID
    let id = state.next_quest_id;
    state.next_quest_id += 1;
    request.id = id;

    let quest_type = request.quest_type;
    let threat_level = request.threat_level;

    events.push(WorldEvent::QuestRequestArrived {
        request_id: id,
        quest_type,
        threat_level,
    });
    deltas.quests_arrived += 1;

    state.request_board.push(request);

    // If the grammar walker generated a choice event, attach it
    if let Some(mut choice_event) = choice {
        // Fix up the quest_id in the choice source
        if let ChoiceSource::QuestBranch { quest_id } = &mut choice_event.source {
            *quest_id = id;
        }
        // Fix up any quest-referencing effects
        for opt in &mut choice_event.options {
            for eff in &mut opt.effects {
                match eff {
                    ChoiceEffect::ModifyQuestThreat { quest_id, .. }
                    | ChoiceEffect::ModifyQuestReward { quest_id, .. } => {
                        *quest_id = id;
                    }
                    _ => {}
                }
            }
        }
        state.pending_choices.push(choice_event);
    }
}
