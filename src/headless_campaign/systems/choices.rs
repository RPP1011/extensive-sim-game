//! Choice event generation — every 500 ticks (~50s).
//!
//! Generates branching decisions by instantiating templates loaded from
//! `assets/choice_templates/`. Templates are TOML files with variable
//! substitution for quest names, NPC names, threat levels, etc.

use std::collections::HashMap;

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::choice_templates::{
    instantiate_template, ChoiceTemplateRegistry, TemplateContext,
};
use crate::headless_campaign::state::*;

/// Quest branch checks run every tick to catch quests in Preparing status.
/// NPC/world event checks run at a slower cadence.
const WORLD_EVENT_INTERVAL: u64 = 10;

/// Maximum simultaneous pending choices.
const MAX_PENDING_CHOICES: usize = 3;

fn get_templates() -> &'static ChoiceTemplateRegistry {
    crate::headless_campaign::choice_templates::get_or_load_templates()
}

pub fn tick_choices(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick == 0 || state.pending_choices.len() >= MAX_PENDING_CHOICES {
        return;
    }

    let templates = get_templates();

    // Quest branches are triggered on AcceptQuest in step.rs, not here.

    // NPC and world events check at a slower cadence
    if state.tick % WORLD_EVENT_INTERVAL == 0 {
        let roll = lcg_f32(&mut state.rng);
        if roll < 0.4 {
            try_npc_encounter(state, templates, events);
        } else if roll < 0.75 {
            try_world_event(state, templates, events);
        }
    }
}

/// Try to generate a quest branch choice from templates.
fn try_quest_branch(
    state: &mut CampaignState,
    templates: &ChoiceTemplateRegistry,
    events: &mut Vec<WorldEvent>,
) {
    // Find a preparing combat/rescue quest without an existing branch choice
    let quest = state.active_quests.iter().find(|q| {
        q.status == ActiveQuestStatus::Preparing
            && matches!(q.request.quest_type, QuestType::Combat | QuestType::Rescue)
            && !state.pending_choices.iter().any(|c| {
                matches!(&c.source, ChoiceSource::QuestBranch { quest_id } if *quest_id == q.id)
            })
    });

    let quest = match quest {
        Some(q) => q,
        None => return, // No RNG consumed if no eligible quest
    };

    let quest_templates = templates.by_trigger("quest_preparing_combat");
    if quest_templates.is_empty() {
        return; // No RNG consumed if no templates
    }

    // Only consume RNG when we have both a quest and a template
    let template_idx = (lcg_next(&mut state.rng) as usize) % quest_templates.len();
    let template = quest_templates[template_idx];

    let mut ctx = TemplateContext::new();
    ctx.insert("quest_type".into(), format!("{:?}", quest.request.quest_type));
    ctx.insert("threat".into(), format!("{:.0}", quest.request.threat_level));
    ctx.insert("quest_id".into(), quest.id.to_string());

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = instantiate_template(template, &ctx, choice_id, state.elapsed_ms);

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}

/// Try to generate an NPC encounter choice from templates.
fn try_npc_encounter(
    state: &mut CampaignState,
    templates: &ChoiceTemplateRegistry,
    events: &mut Vec<WorldEvent>,
) {
    if state.npc_relationships.is_empty() {
        return;
    }

    let npc_templates = templates.by_category("npc_encounter");
    if npc_templates.is_empty() {
        return;
    }

    let npc_idx = (lcg_next(&mut state.rng) as usize) % state.npc_relationships.len();
    let npc = &state.npc_relationships[npc_idx];

    let template_idx = (lcg_next(&mut state.rng) as usize) % npc_templates.len();
    let template = npc_templates[template_idx];

    let mut ctx = TemplateContext::new();
    ctx.insert("npc_name".into(), npc.npc_name.clone());
    ctx.insert("npc_id".into(), npc.npc_id.to_string());

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = instantiate_template(template, &ctx, choice_id, state.elapsed_ms);

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}

/// Try to generate a world event choice from templates.
fn try_world_event(
    state: &mut CampaignState,
    templates: &ChoiceTemplateRegistry,
    events: &mut Vec<WorldEvent>,
) {
    let world_templates = templates.by_category("world_event");
    if world_templates.is_empty() {
        return;
    }

    let template_idx = (lcg_next(&mut state.rng) as usize) % world_templates.len();
    let template = world_templates[template_idx];

    let mut ctx = TemplateContext::new();
    // Common world event context
    let level = 2 + (lcg_next(&mut state.rng) % 3) as u32;
    let cost = 40;
    ctx.insert("level".into(), level.to_string());
    ctx.insert("cost".into(), cost.to_string());

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = instantiate_template(template, &ctx, choice_id, state.elapsed_ms);

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}
