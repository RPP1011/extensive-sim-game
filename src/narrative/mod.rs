//! LFM-Orchestrated Narrative System — trigger-based narrative generation
//! with template fallback when LFM is unavailable.
//!
//! Supports: events, dialogue, quests, post-battle content.
//! All narrative templates are data-driven via TOML and hot-reloadable.

pub mod triggers;
pub mod templates;
pub mod quest;
pub mod dialogue;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use self::dialogue::DialogueLine;
use self::quest::QuestUpdate;

// ---------------------------------------------------------------------------
// Narrative output
// ---------------------------------------------------------------------------

/// The output of a narrative trigger evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeOutput {
    pub title: String,
    pub body: String,
    #[serde(default)]
    pub dialogue: Vec<DialogueLine>,
    #[serde(default)]
    pub quest_update: Option<QuestUpdate>,
    #[serde(default)]
    pub choices: Vec<NarrativeChoice>,
}

/// A player choice in a narrative event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeChoice {
    pub label: String,
    pub consequence_tag: String,
}

// ---------------------------------------------------------------------------
// Narrative context (game state snapshot for template interpolation)
// ---------------------------------------------------------------------------

/// Snapshot of game state values used for template string interpolation.
#[derive(Debug, Clone, Default)]
pub struct NarrativeContext {
    pub values: HashMap<String, String>,
}

impl NarrativeContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.values.insert(key.into(), value.into());
        self
    }

    /// Build context from overworld state.
    pub fn from_game_state(
        faction_names: &[String],
        region_names: &[String],
        hero_names: &[String],
        current_turn: u32,
    ) -> Self {
        let mut ctx = Self::new();
        if let Some(name) = faction_names.first() {
            ctx.set("player_faction", name.as_str());
        }
        for (i, name) in faction_names.iter().enumerate() {
            ctx.set(format!("faction_{}", i), name.as_str());
        }
        for (i, name) in region_names.iter().enumerate() {
            ctx.set(format!("region_{}", i), name.as_str());
        }
        for (i, name) in hero_names.iter().enumerate() {
            ctx.set(format!("hero_{}", i), name.as_str());
        }
        ctx.set("turn", current_turn.to_string());
        ctx
    }
}

// ---------------------------------------------------------------------------
// Narrative engine (combines triggers + templates)
// ---------------------------------------------------------------------------

/// The narrative engine — evaluates triggers and generates narrative output.
#[derive(Debug, Clone, Default)]
pub struct NarrativeEngine {
    pub trigger_registry: triggers::TriggerRegistry,
    pub template_registry: templates::TemplateRegistry,
    pub quest_tracker: quest::QuestTracker,
    /// Cooldown tracking: trigger_tag → last_fired_turn
    pub cooldowns: HashMap<String, u32>,
}

impl NarrativeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluate all active triggers against the current game state.
    /// Returns narrative outputs for any matching triggers.
    pub fn evaluate(
        &mut self,
        current_turn: u32,
        context: &NarrativeContext,
        active_events: &[triggers::NarrativeTriggerKind],
    ) -> Vec<NarrativeOutput> {
        let mut outputs = Vec::new();

        for event in active_events {
            let matching = self
                .trigger_registry
                .find_matching(event);

            for trigger in matching {
                // Check cooldown
                if let Some(&last_fired) = self.cooldowns.get(&trigger.tag) {
                    if current_turn < last_fired + trigger.cooldown_turns {
                        continue;
                    }
                }

                // Generate output from template
                if let Some(template) = self.template_registry.get(&trigger.template_key) {
                    let output = templates::render_template(template, context);
                    self.cooldowns.insert(trigger.tag.clone(), current_turn);
                    outputs.push(output);
                }
            }
        }

        outputs
    }

    /// Fire post-battle narrative triggers.
    pub fn on_battle_complete(
        &mut self,
        victory: bool,
        hero_names: &[String],
        current_turn: u32,
        context: &NarrativeContext,
    ) -> Vec<NarrativeOutput> {
        let kind = if victory {
            triggers::NarrativeTriggerKind::PostBattle {
                outcome: triggers::BattleOutcome::Victory,
            }
        } else {
            triggers::NarrativeTriggerKind::PostBattle {
                outcome: triggers::BattleOutcome::Defeat,
            }
        };
        self.evaluate(current_turn, context, &[kind])
    }

    /// Fire region-enter narrative triggers.
    pub fn on_region_enter(
        &mut self,
        region_id: usize,
        current_turn: u32,
        context: &NarrativeContext,
    ) -> Vec<NarrativeOutput> {
        let kind = triggers::NarrativeTriggerKind::RegionEnter { region_id };
        self.evaluate(current_turn, context, &[kind])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use triggers::*;
    use templates::*;

    #[test]
    fn test_narrative_engine_basic() {
        let mut engine = NarrativeEngine::new();

        // Register a trigger
        engine.trigger_registry.register(NarrativeTrigger {
            tag: "victory_celebration".to_string(),
            kind: NarrativeTriggerKind::PostBattle {
                outcome: BattleOutcome::Victory,
            },
            priority: 10,
            cooldown_turns: 3,
            template_key: "victory_01".to_string(),
        });

        // Register a template
        engine.template_registry.register(NarrativeTemplate {
            key: "victory_01".to_string(),
            title: "Victory!".to_string(),
            body: "The {player_faction} celebrates their triumph on turn {turn}.".to_string(),
            dialogue: Vec::new(),
            choices: Vec::new(),
        });

        let mut ctx = NarrativeContext::new();
        ctx.set("player_faction", "Iron Vanguard");
        ctx.set("turn", "5");

        let events = vec![NarrativeTriggerKind::PostBattle {
            outcome: BattleOutcome::Victory,
        }];
        let outputs = engine.evaluate(5, &ctx, &events);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].title, "Victory!");
        assert!(outputs[0].body.contains("Iron Vanguard"));
        assert!(outputs[0].body.contains("turn 5"));
    }

    #[test]
    fn test_cooldown_prevents_spam() {
        let mut engine = NarrativeEngine::new();

        engine.trigger_registry.register(NarrativeTrigger {
            tag: "test_trigger".to_string(),
            kind: NarrativeTriggerKind::PostBattle {
                outcome: BattleOutcome::Victory,
            },
            priority: 10,
            cooldown_turns: 5,
            template_key: "test_template".to_string(),
        });

        engine.template_registry.register(NarrativeTemplate {
            key: "test_template".to_string(),
            title: "Test".to_string(),
            body: "Test body".to_string(),
            dialogue: Vec::new(),
            choices: Vec::new(),
        });

        let ctx = NarrativeContext::new();
        let events = vec![NarrativeTriggerKind::PostBattle {
            outcome: BattleOutcome::Victory,
        }];

        // First evaluation: should fire
        let outputs = engine.evaluate(1, &ctx, &events);
        assert_eq!(outputs.len(), 1);

        // Second evaluation within cooldown: should not fire
        let outputs = engine.evaluate(3, &ctx, &events);
        assert_eq!(outputs.len(), 0);

        // After cooldown: should fire again
        let outputs = engine.evaluate(7, &ctx, &events);
        assert_eq!(outputs.len(), 1);
    }
}
