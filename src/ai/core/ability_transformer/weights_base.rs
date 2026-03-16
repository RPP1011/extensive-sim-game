//! Embedding registry for pre-computed ability CLS embeddings.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Embedding Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct EmbeddingRegistryJson {
    model_hash: String,
    d_model: usize,
    n_abilities: usize,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
    #[serde(default)]
    outcome_mean: Vec<f32>,
    #[serde(default)]
    outcome_std: Vec<f32>,
}

/// Pre-computed CLS embeddings for known abilities.
#[derive(Debug, Clone)]
pub struct EmbeddingRegistry {
    pub model_hash: String,
    pub d_model: usize,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
}

impl EmbeddingRegistry {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: EmbeddingRegistryJson =
            serde_json::from_str(json_str).map_err(|e| format!("Registry parse error: {e}"))?;
        assert_eq!(file.embeddings.len(), file.n_abilities, "n_abilities mismatch");
        Ok(Self { model_hash: file.model_hash, d_model: file.d_model, embeddings: file.embeddings })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Failed to read registry: {e}"))?;
        Self::from_json(&data)
    }

    pub fn get(&self, ability_name: &str) -> Option<&[f32]> {
        self.embeddings.get(ability_name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize { self.embeddings.len() }
}
