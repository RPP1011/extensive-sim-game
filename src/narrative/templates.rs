//! Template-based narrative generation with string interpolation.
//!
//! Templates use `{key}` placeholders that are substituted with values from
//! a `NarrativeContext`. Loaded from TOML and hot-reloadable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{NarrativeChoice, NarrativeContext, NarrativeOutput};
use super::dialogue::DialogueLine;

// ---------------------------------------------------------------------------
// Template definition
// ---------------------------------------------------------------------------

/// A narrative template with interpolation placeholders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeTemplate {
    pub key: String,
    pub title: String,
    pub body: String,
    #[serde(default)]
    pub dialogue: Vec<DialogueTemplate>,
    #[serde(default)]
    pub choices: Vec<NarrativeChoice>,
}

/// A dialogue line template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTemplate {
    pub speaker: String,
    pub text: String,
}

// ---------------------------------------------------------------------------
// TOML file structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeTemplatesFile {
    #[serde(default)]
    pub template: Vec<NarrativeTemplate>,
}

// ---------------------------------------------------------------------------
// Template registry
// ---------------------------------------------------------------------------

/// Registry of narrative templates, keyed by template key.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, NarrativeTemplate>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template.
    pub fn register(&mut self, template: NarrativeTemplate) {
        self.templates.insert(template.key.clone(), template);
    }

    /// Get a template by key.
    pub fn get(&self, key: &str) -> Option<&NarrativeTemplate> {
        self.templates.get(key)
    }

    /// Load templates from a TOML string, replacing existing templates.
    pub fn load_from_str(&mut self, toml_str: &str) -> Result<usize, String> {
        let file: NarrativeTemplatesFile =
            toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {}", e))?;
        self.templates.clear();
        let count = file.template.len();
        for template in file.template {
            self.templates.insert(template.key.clone(), template);
        }
        Ok(count)
    }

    /// Load from a TOML file.
    pub fn load_from_file(&mut self, path: &std::path::Path) -> Result<usize, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        self.load_from_str(&contents)
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Template rendering
// ---------------------------------------------------------------------------

/// Render a narrative template by substituting `{key}` placeholders with context values.
pub fn render_template(template: &NarrativeTemplate, context: &NarrativeContext) -> NarrativeOutput {
    NarrativeOutput {
        title: interpolate(&template.title, context),
        body: interpolate(&template.body, context),
        dialogue: template
            .dialogue
            .iter()
            .map(|d| DialogueLine {
                speaker: interpolate(&d.speaker, context),
                text: interpolate(&d.text, context),
            })
            .collect(),
        quest_update: None,
        choices: template.choices.clone(),
    }
}

/// Substitute `{key}` placeholders in a string with values from the context.
pub fn interpolate(template: &str, context: &NarrativeContext) -> String {
    let mut result = template.to_string();
    for (key, value) in &context.values {
        let placeholder = format!("{{{}}}", key);
        result = result.replace(&placeholder, value);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation() {
        let mut ctx = NarrativeContext::new();
        ctx.set("name", "Elowen");
        ctx.set("region", "Ashfall");

        let result = interpolate("Commander {name} arrives at {region}.", &ctx);
        assert_eq!(result, "Commander Elowen arrives at Ashfall.");
    }

    #[test]
    fn test_interpolation_missing_key() {
        let ctx = NarrativeContext::new();
        let result = interpolate("Hello {unknown}!", &ctx);
        assert_eq!(result, "Hello {unknown}!"); // Unresolved placeholders stay as-is
    }

    #[test]
    fn test_template_rendering() {
        let template = NarrativeTemplate {
            key: "test".to_string(),
            title: "{faction} News".to_string(),
            body: "On turn {turn}, {faction} did something.".to_string(),
            dialogue: vec![DialogueTemplate {
                speaker: "{hero_0}".to_string(),
                text: "We must press on.".to_string(),
            }],
            choices: Vec::new(),
        };

        let mut ctx = NarrativeContext::new();
        ctx.set("faction", "Iron Vanguard");
        ctx.set("turn", "12");
        ctx.set("hero_0", "Commander Elowen");

        let output = render_template(&template, &ctx);
        assert_eq!(output.title, "Iron Vanguard News");
        assert!(output.body.contains("turn 12"));
        assert_eq!(output.dialogue[0].speaker, "Commander Elowen");
    }

    #[test]
    fn test_registry_toml_loading() {
        let toml = r#"
[[template]]
key = "victory_01"
title = "Victory!"
body = "The {player_faction} has won."

[[template.dialogue]]
speaker = "{hero_0}"
text = "We did it!"

[[template]]
key = "defeat_01"
title = "Defeat"
body = "The {player_faction} has fallen."
"#;
        let mut registry = TemplateRegistry::new();
        let count = registry.load_from_str(toml).unwrap();
        assert_eq!(count, 2);
        assert!(registry.get("victory_01").is_some());
        assert!(registry.get("defeat_01").is_some());
    }
}
