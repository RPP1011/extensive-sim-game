//! Choice template loading and instantiation.
//!
//! Templates are TOML files in `assets/choice_templates/` that define
//! the structure of branching decisions. The instantiation code fills
//! in context variables ({quest_id}, {npc_name}, etc.) and converts
//! template effects to concrete `ChoiceEffect`s.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::state::*;

// ---------------------------------------------------------------------------
// Template format (deserialized from TOML)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize)]
pub struct ChoiceTemplate {
    pub category: String,
    pub trigger: String,
    pub prompt: String,
    pub deadline_secs: f32,
    pub default_option: usize,
    pub options: Vec<OptionTemplate>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OptionTemplate {
    pub label: String,
    pub description: String,
    #[serde(default)]
    pub effects: Vec<EffectTemplate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectTemplate {
    #[serde(rename = "type")]
    pub effect_type: String,
    // All fields optional — which ones are used depends on effect_type
    pub amount: Option<f32>,
    pub amount_var: Option<String>,
    pub multiplier: Option<f32>,
    pub gold_factor: Option<f32>,
    pub gold_bonus: Option<f32>,
    pub rep_bonus: Option<f32>,
    pub quest_id: Option<String>,
    pub faction_id: Option<usize>,
    pub delta: Option<f32>,
    pub text: Option<String>,
    pub name: Option<String>,
    pub archetype: Option<String>,
    pub level_var: Option<String>,
}

// ---------------------------------------------------------------------------
// Template registry
// ---------------------------------------------------------------------------

/// All loaded choice templates, keyed by filename (without extension).
#[derive(Clone, Debug, Default)]
pub struct ChoiceTemplateRegistry {
    pub templates: Vec<(String, ChoiceTemplate)>,
}

impl ChoiceTemplateRegistry {
    /// Load all templates from a directory.
    pub fn load_from_dir(dir: &std::path::Path) -> Self {
        let mut templates = Vec::new();

        if !dir.exists() {
            return Self { templates };
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Self { templates },
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str::<ChoiceTemplate>(&content) {
                    Ok(template) => templates.push((name, template)),
                    Err(e) => eprintln!("Warning: failed to parse {}: {}", path.display(), e),
                },
                Err(e) => eprintln!("Warning: failed to read {}: {}", path.display(), e),
            }
        }

        Self { templates }
    }

    /// Get templates matching a trigger type.
    pub fn by_trigger(&self, trigger: &str) -> Vec<&ChoiceTemplate> {
        self.templates
            .iter()
            .filter(|(_, t)| t.trigger == trigger)
            .map(|(_, t)| t)
            .collect()
    }

    /// Get templates matching a category.
    pub fn by_category(&self, category: &str) -> Vec<&ChoiceTemplate> {
        self.templates
            .iter()
            .filter(|(_, t)| t.category == category)
            .map(|(_, t)| t)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Template instantiation
// ---------------------------------------------------------------------------

/// Lazily loaded template registry (cached across calls).
static TEMPLATES: std::sync::OnceLock<ChoiceTemplateRegistry> = std::sync::OnceLock::new();

/// Get or load the global template registry.
pub fn get_or_load_templates() -> &'static ChoiceTemplateRegistry {
    TEMPLATES.get_or_init(|| {
        let dir = std::path::Path::new("assets/choice_templates");
        let registry = ChoiceTemplateRegistry::load_from_dir(dir);
        if !registry.templates.is_empty() {
            eprintln!(
                "Choice templates: loaded {} from {}",
                registry.templates.len(),
                dir.display(),
            );
        }
        registry
    })
}

/// Context variables for template instantiation.
pub type TemplateContext = HashMap<String, String>;

/// Instantiate a choice template with context variables.
pub fn instantiate_template(
    template: &ChoiceTemplate,
    ctx: &TemplateContext,
    choice_id: u32,
    elapsed_ms: u64,
) -> ChoiceEvent {
    let prompt = substitute(&template.prompt, ctx);
    let deadline = Some(elapsed_ms + (template.deadline_secs * 1000.0) as u64);

    let options: Vec<ChoiceOption> = template
        .options
        .iter()
        .map(|opt| {
            let label = substitute(&opt.label, ctx);
            let description = substitute(&opt.description, ctx);
            let effects = opt
                .effects
                .iter()
                .filter_map(|e| instantiate_effect(e, ctx))
                .collect();
            ChoiceOption {
                label,
                description,
                effects,
            }
        })
        .collect();

    let source = match template.category.as_str() {
        "quest_branch" => {
            let qid = ctx.get("quest_id").and_then(|s| s.parse().ok()).unwrap_or(0);
            ChoiceSource::QuestBranch { quest_id: qid }
        }
        "npc_encounter" => {
            let nid = ctx.get("npc_id").and_then(|s| s.parse().ok()).unwrap_or(0);
            ChoiceSource::NpcEncounter { npc_id: nid }
        }
        "world_event" => ChoiceSource::WorldEvent,
        _ => ChoiceSource::WorldEvent,
    };

    ChoiceEvent {
        id: choice_id,
        source,
        prompt,
        options,
        default_option: template.default_option.min(template.options.len().saturating_sub(1)),
        deadline_ms: deadline,
        created_at_ms: elapsed_ms,
    }
}

/// Substitute {variable} placeholders in a string.
fn substitute(template: &str, ctx: &TemplateContext) -> String {
    let mut result = template.to_string();
    for (key, value) in ctx {
        result = result.replace(&format!("{{{}}}", key), value);
    }
    result
}

/// Convert a template effect to a concrete ChoiceEffect.
pub fn instantiate_effect(e: &EffectTemplate, ctx: &TemplateContext) -> Option<ChoiceEffect> {
    match e.effect_type.as_str() {
        "gold" => {
            let amount = if let Some(ref var) = e.amount_var {
                let resolved = substitute(var, ctx);
                resolved.parse::<f32>().unwrap_or(0.0)
            } else {
                e.amount.unwrap_or(0.0)
            };
            Some(ChoiceEffect::Gold(amount))
        }
        "supplies" => Some(ChoiceEffect::Supplies(e.amount.unwrap_or(0.0))),
        "reputation" => Some(ChoiceEffect::Reputation(e.amount.unwrap_or(0.0))),
        "faction_relation" => Some(ChoiceEffect::FactionRelation {
            faction_id: e.faction_id.unwrap_or(0),
            delta: e.delta.unwrap_or(0.0),
        }),
        "modify_quest_threat" => {
            let quest_id = e
                .quest_id
                .as_ref()
                .map(|s| substitute(s, ctx))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            Some(ChoiceEffect::ModifyQuestThreat {
                quest_id,
                multiplier: e.multiplier.unwrap_or(1.0),
            })
        }
        "modify_quest_reward" => {
            let quest_id = e
                .quest_id
                .as_ref()
                .map(|s| substitute(s, ctx))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            // gold_factor is multiplied by threat level at instantiation
            let threat: f32 = ctx
                .get("threat")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let gold_bonus = e.gold_factor.map(|f| f * threat).or(e.gold_bonus).unwrap_or(0.0);
            let rep_bonus = e.rep_bonus.unwrap_or(0.0);
            Some(ChoiceEffect::ModifyQuestReward {
                quest_id,
                gold_bonus,
                rep_bonus,
            })
        }
        "narrative" => {
            let text = e.text.as_ref().map(|t| substitute(t, ctx)).unwrap_or_default();
            Some(ChoiceEffect::Narrative(text))
        }
        "add_adventurer" => {
            let level: u32 = e
                .level_var
                .as_ref()
                .map(|s| substitute(s, ctx))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            let archetype = e.archetype.as_deref().unwrap_or("rogue");
            let name = e.name.as_ref().map(|n| substitute(n, ctx)).unwrap_or_else(|| "Recruit".into());

            let (hp, atk, def, spd, ap) = match archetype {
                "knight" => (110.0, 12.0, 18.0, 7.0, 4.0),
                "ranger" => (75.0, 16.0, 8.0, 13.0, 7.0),
                "mage" => (55.0, 6.0, 5.0, 9.0, 22.0),
                "cleric" => (65.0, 5.0, 10.0, 8.0, 18.0),
                "rogue" => (65.0, 18.0, 6.0, 15.0, 6.0),
                _ => (70.0, 10.0, 10.0, 10.0, 10.0),
            };

            Some(ChoiceEffect::AddAdventurer(Adventurer {
                id: 0, // Reassigned at application time
                name,
                archetype: archetype.into(),
                level,
                xp: 0,
                stats: AdventurerStats {
                    max_hp: hp + level as f32 * 5.0,
                    attack: atk + level as f32 * 2.0,
                    defense: def + level as f32 * 1.5,
                    speed: spd,
                    ability_power: ap + level as f32 * 2.0,
                },
                equipment: Equipment::default(),
                traits: Vec::new(),
                status: AdventurerStatus::Idle,
                loyalty: 45.0,
                stress: 20.0,
                fatigue: 15.0,
                injury: 0.0,
                resolve: 50.0,
                morale: 55.0,
                party_id: None,
                guild_relationship: 30.0,
                leadership_role: None,
                is_player_character: false,
                faction_id: None,
                rallying_to: None,
                    tier_status: Default::default(),
                    history_tags: Default::default(),
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,

            mood_state: crate::headless_campaign::state::MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
            behavior_ledger: BehaviorLedger::default(),
            classes: Vec::new(),
            }))
        }
        _ => {
            // Unknown effect type — skip with warning
            None
        }
    }
}
