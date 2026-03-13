//! Ability manifest — loads abilities by name from `.ability` DSL files.
//!
//! The manifest is a TOML file mapping ability names to DSL files and tags:
//! ```toml
//! [abilities.Strike]
//! file = "abilities/tier1_instant/damage.ability"
//! tags = ["damage", "single_target", "tier1"]
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::ai::effects::defs::{AbilityDef, PassiveDef};
use crate::ai::effects::dsl;

/// A single entry in the manifest TOML.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManifestEntry {
    pub file: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Top-level manifest file structure.
#[derive(Debug, Clone, serde::Deserialize)]
struct ManifestFile {
    #[serde(default)]
    abilities: HashMap<String, ManifestEntry>,
}

/// Loaded manifest with pre-parsed abilities from DSL files.
#[derive(Debug, Clone)]
pub struct AbilityManifest {
    abilities: HashMap<String, AbilityDef>,
    passives: HashMap<String, PassiveDef>,
    tags: HashMap<String, Vec<String>>,
}

impl AbilityManifest {
    /// Load a manifest from a TOML file. `base_dir` is the directory
    /// that DSL file paths in the manifest are relative to.
    pub fn load(manifest_path: &Path, base_dir: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(manifest_path)
            .map_err(|e| format!("Failed to read manifest {}: {e}", manifest_path.display()))?;
        let file: ManifestFile = toml::from_str(&content)
            .map_err(|e| format!("Manifest parse error: {e}"))?;

        let mut abilities = HashMap::new();
        let mut passives = HashMap::new();
        let mut tags = HashMap::new();

        // Cache parsed DSL files (multiple entries may share a file)
        let mut file_cache: HashMap<String, (Vec<AbilityDef>, Vec<PassiveDef>)> = HashMap::new();

        for (name, entry) in &file.abilities {
            let parsed = file_cache.entry(entry.file.clone()).or_insert_with(|| {
                let path = base_dir.join(&entry.file);
                let dsl_str = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
                dsl::parse_abilities(&dsl_str)
                    .unwrap_or_else(|e| panic!("DSL parse error in {}: {e}", entry.file))
            });

            // Look for the name in abilities first, then passives
            if let Some(def) = parsed.0.iter().find(|a| a.name == *name) {
                abilities.insert(name.clone(), def.clone());
            } else if let Some(def) = parsed.1.iter().find(|p| p.name == *name) {
                passives.insert(name.clone(), def.clone());
            } else {
                return Err(format!("'{}' not found in '{}'", name, entry.file));
            }

            tags.insert(name.clone(), entry.tags.clone());
        }

        Ok(Self { abilities, passives, tags })
    }

    /// Get an ability definition by name.
    pub fn get_ability(&self, name: &str) -> Option<&AbilityDef> {
        self.abilities.get(name)
    }

    /// Get a passive definition by name.
    pub fn get_passive(&self, name: &str) -> Option<&PassiveDef> {
        self.passives.get(name)
    }

    /// Get tags for a named ability/passive.
    pub fn get_tags(&self, name: &str) -> Option<&[String]> {
        self.tags.get(name).map(|v| v.as_slice())
    }

    /// Find all ability names matching a tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&str> {
        self.tags
            .iter()
            .filter(|(_, t)| t.iter().any(|s| s == tag))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Number of loaded abilities.
    pub fn len(&self) -> usize {
        self.abilities.len() + self.passives.len()
    }

    /// Whether the manifest is empty.
    pub fn is_empty(&self) -> bool {
        self.abilities.is_empty() && self.passives.is_empty()
    }
}
