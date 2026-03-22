//! Data-driven morale culture profiles loaded from TOML.
//!
//! Each culture profile defines weight multipliers for the 5 morale input
//! categories, plus cascade susceptibility, volatility, and threshold values.
//! Profiles are hot-reloadable at runtime.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// TOML schema
// ---------------------------------------------------------------------------

/// Weight multipliers for the 5 morale input categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoraleInputWeights {
    /// Weight for self-assessment (HP, recent damage).
    pub self_weight: f32,
    /// Weight for ally assessment (nearby ally HP, deaths).
    pub allies_weight: f32,
    /// Weight for threat assessment (enemy count, outnumbered ratio).
    pub threats_weight: f32,
    /// Weight for leadership assessment (leader alive/healthy).
    pub leadership_weight: f32,
    /// Weight for situation assessment (team advantage).
    pub situation_weight: f32,
}

impl Default for MoraleInputWeights {
    fn default() -> Self {
        Self {
            self_weight: 0.20,
            allies_weight: 0.20,
            threats_weight: 0.20,
            leadership_weight: 0.20,
            situation_weight: 0.20,
        }
    }
}

/// A data-driven culture profile — defined entirely in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoraleCultureProfile {
    /// Display name (e.g. "Collectivist", "Mercenary").
    pub name: String,
    /// Weight multipliers for the 5 morale input categories.
    pub input_weights: MoraleInputWeights,
    /// How much routing neighbors affect this unit's morale [0, 1].
    #[serde(default = "default_cascade")]
    pub cascade_susceptibility: f32,
    /// How fast morale swings (multiplier on delta) [0.5, 2.0].
    #[serde(default = "default_volatility")]
    pub volatility: f32,
    /// Morale value below which the unit routes [0, 0.3].
    #[serde(default = "default_routing_threshold")]
    pub routing_threshold: f32,
    /// Morale value below which the unit is wavering.
    #[serde(default = "default_wavering_threshold")]
    pub wavering_threshold: f32,
    /// Morale value above which the unit is fired up [0.7, 1.0].
    #[serde(default = "default_fired_up_threshold")]
    pub fired_up_threshold: f32,
}

fn default_cascade() -> f32 { 0.5 }
fn default_volatility() -> f32 { 1.0 }
fn default_routing_threshold() -> f32 { 0.2 }
fn default_wavering_threshold() -> f32 { 0.35 }
fn default_fired_up_threshold() -> f32 { 0.8 }

/// TOML file root structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoraleCulturesFile {
    pub culture: Vec<MoraleCultureProfile>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of culture profiles, keyed by name.
#[derive(Debug, Clone, Default)]
pub struct MoraleCultureRegistry {
    profiles: HashMap<String, MoraleCultureProfile>,
}

impl MoraleCultureRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Load profiles from a TOML file, replacing all existing profiles.
    pub fn load_from_file(&mut self, path: &Path) -> Result<usize, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        self.load_from_str(&contents)
    }

    /// Load profiles from a TOML string, replacing all existing profiles.
    pub fn load_from_str(&mut self, toml_str: &str) -> Result<usize, String> {
        let file: MoraleCulturesFile =
            toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {}", e))?;
        self.profiles.clear();
        let count = file.culture.len();
        for profile in file.culture {
            self.profiles.insert(profile.name.clone(), profile);
        }
        Ok(count)
    }

    /// Get a culture profile by name.
    pub fn get(&self, name: &str) -> Option<&MoraleCultureProfile> {
        self.profiles.get(name)
    }

    /// Get the first profile as a fallback default.
    pub fn default_profile(&self) -> Option<&MoraleCultureProfile> {
        self.profiles.values().next()
    }

    /// List all profile names.
    pub fn names(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }

    /// Number of loaded profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Insert a profile directly (useful for tests).
    pub fn insert(&mut self, profile: MoraleCultureProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }
}

// ---------------------------------------------------------------------------
// Built-in defaults (used when TOML file is missing)
// ---------------------------------------------------------------------------

/// Returns a TOML string with the 5 default culture profiles.
pub fn default_cultures_toml() -> &'static str {
    include_str!("default_cultures.toml")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOML: &str = r#"
[[culture]]
name = "TestCulture"
cascade_susceptibility = 0.5
volatility = 1.2
routing_threshold = 0.15
wavering_threshold = 0.30
fired_up_threshold = 0.85

[culture.input_weights]
self_weight = 0.30
allies_weight = 0.20
threats_weight = 0.25
leadership_weight = 0.10
situation_weight = 0.15
"#;

    #[test]
    fn test_parse_culture_toml() {
        let mut registry = MoraleCultureRegistry::new();
        let count = registry.load_from_str(TEST_TOML).unwrap();
        assert_eq!(count, 1);

        let profile = registry.get("TestCulture").unwrap();
        assert_eq!(profile.name, "TestCulture");
        assert!((profile.cascade_susceptibility - 0.5).abs() < 0.001);
        assert!((profile.input_weights.self_weight - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_registry_replace_on_reload() {
        let mut registry = MoraleCultureRegistry::new();
        registry.load_from_str(TEST_TOML).unwrap();
        assert_eq!(registry.len(), 1);

        let toml2 = r#"
[[culture]]
name = "A"
[culture.input_weights]
self_weight = 0.2
allies_weight = 0.2
threats_weight = 0.2
leadership_weight = 0.2
situation_weight = 0.2

[[culture]]
name = "B"
[culture.input_weights]
self_weight = 0.3
allies_weight = 0.3
threats_weight = 0.1
leadership_weight = 0.1
situation_weight = 0.2
"#;
        registry.load_from_str(toml2).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(registry.get("TestCulture").is_none());
        assert!(registry.get("A").is_some());
    }
}
