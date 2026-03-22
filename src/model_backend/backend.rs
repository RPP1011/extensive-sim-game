use std::io::Write;
use std::process::{Command, Stdio};

use super::config::{ModelConfig, ProviderConfig};
use super::tier::ModelTier;

/// Errors produced by the model integration layer.
#[derive(Debug, Clone)]
pub enum ModelError {
    /// No model backend is configured or available.
    NoModelAvailable,
    /// The backend process failed to launch or returned an error.
    BackendFailed(String),
    /// The backend returned output that could not be parsed.
    ParseError(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NoModelAvailable => write!(f, "no model backend available"),
            ModelError::BackendFailed(msg) => write!(f, "model backend failed: {msg}"),
            ModelError::ParseError(msg) => write!(f, "model output parse error: {msg}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Provider-agnostic text generation interface.
///
/// Implementations may wrap a local subprocess (Python, llama.cpp),
/// a remote API (OpenAI, Anthropic), or an in-process model (Burn).
/// New providers should implement this trait so that all consumers
/// (AOT pipeline, ASCII art generator, narrative system from Issue #14
/// Task 09) work without changes.
pub trait ModelBackend: Send + Sync {
    /// Generate text from a prompt.
    ///
    /// * `prompt` — the user/system prompt text.
    /// * `seed` — deterministic seed (backend should honour if possible).
    /// * `max_tokens` — upper bound on generated tokens.
    fn generate(&self, prompt: &str, seed: u64, max_tokens: usize) -> Result<String, ModelError>;

    /// The performance tier this backend targets.
    fn tier(&self) -> ModelTier;

    /// Whether the backend is ready to accept requests.
    fn is_available(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Subprocess backend (follows src/mission/room_gen/ml_gen.rs pattern)
// ---------------------------------------------------------------------------

/// Backend that calls an external script via stdin/stdout JSON.
///
/// Protocol:
/// - **Request** (JSON on stdin):
///   `{"prompt": "...", "seed": 42, "max_tokens": 2048, "temperature": 0.7}`
/// - **Response** (JSON on stdout):
///   `{"success": true, "text": "generated content..."}`
pub struct SubprocessBackend {
    config: ModelConfig,
}

impl SubprocessBackend {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelBackend for SubprocessBackend {
    fn generate(&self, prompt: &str, seed: u64, max_tokens: usize) -> Result<String, ModelError> {
        let (model_path, script_path, python) = match &self.config.provider {
            ProviderConfig::Subprocess {
                model_path,
                script_path,
                python,
            } => (model_path, script_path, python.as_str()),
            ProviderConfig::None => return Err(ModelError::NoModelAvailable),
        };

        let request = serde_json::json!({
            "prompt": prompt,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
        });

        let mut child = Command::new(python)
            .arg(script_path)
            .arg("--weights")
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| ModelError::BackendFailed(format!("failed to spawn: {e}")))?;

        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(request.to_string().as_bytes());
        }
        drop(child.stdin.take());

        let output = child
            .wait_with_output()
            .map_err(|e| ModelError::BackendFailed(format!("wait failed: {e}")))?;

        if !output.status.success() {
            return Err(ModelError::BackendFailed(format!(
                "exit code: {:?}",
                output.status.code()
            )));
        }

        let resp: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| ModelError::ParseError(format!("invalid JSON: {e}")))?;

        if !resp["success"].as_bool().unwrap_or(false) {
            let msg = resp["error"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string();
            return Err(ModelError::BackendFailed(msg));
        }

        resp["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| ModelError::ParseError("missing 'text' field".to_string()))
    }

    fn tier(&self) -> ModelTier {
        self.config.tier
    }

    fn is_available(&self) -> bool {
        match &self.config.provider {
            ProviderConfig::Subprocess {
                model_path,
                script_path,
                ..
            } => model_path.exists() && script_path.exists(),
            ProviderConfig::None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subprocess_backend_unavailable_when_none_provider() {
        let config = ModelConfig::default();
        let backend = SubprocessBackend::new(config);
        assert!(!backend.is_available());
        assert!(matches!(
            backend.generate("hello", 0, 100),
            Err(ModelError::NoModelAvailable)
        ));
    }
}
