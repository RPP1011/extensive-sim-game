use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::tier::{DevicePreference, ModelTier};

/// Provider-specific configuration for the model backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderConfig {
    /// Local subprocess model (Python script, llama.cpp, etc.).
    Subprocess {
        /// Path to the model weights / checkpoint.
        model_path: PathBuf,
        /// Path to the inference script that reads JSON from stdin.
        script_path: PathBuf,
        /// Python (or other) executable to invoke.
        #[serde(default = "default_python")]
        python: String,
    },
    /// No model — all calls will return `ModelError::NoModelAvailable`.
    None,
}

fn default_python() -> String {
    "python3".to_string()
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self::None
    }
}

/// Top-level configuration for the model integration layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Performance tier (determines latency/quality expectations).
    pub tier: ModelTier,
    /// Provider-specific backend settings.
    pub provider: ProviderConfig,
    /// Optional deterministic seed (passed to the backend).
    pub seed: Option<u64>,
    /// Maximum tokens to generate per request.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Device preference for inference.
    #[serde(default)]
    pub device: DevicePreference,
}

fn default_max_tokens() -> usize {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            tier: ModelTier::Micro,
            provider: ProviderConfig::None,
            seed: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            device: DevicePreference::Auto,
        }
    }
}
