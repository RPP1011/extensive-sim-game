use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crossbeam_channel::{bounded, Receiver};

use super::backend::{ModelBackend, ModelError, SubprocessBackend};
use super::config::{ModelConfig, ProviderConfig};
use super::tier::ModelTier;

/// Consumer-facing model client with lazy loading, retry logic, and async dispatch.
///
/// The `ModelClient` is the primary interface for all model consumers (AOT
/// pipeline, ASCII art generator, narrative system). It wraps a `ModelBackend`
/// implementation and provides:
///
/// - Lazy backend initialization on first call
/// - Retry with seed increment (3 attempts, matching `ml_gen.rs` pattern)
/// - Async generation via `generate_async()` returning a channel receiver
/// - Graceful degradation when no model is available
pub struct ModelClient {
    backend: Mutex<Option<Box<dyn ModelBackend>>>,
    config: ModelConfig,
    initialized: AtomicBool,
}

impl ModelClient {
    /// Create a new client. The backend is **not** loaded until the first
    /// `generate()` or `is_available()` call (lazy initialization).
    pub fn new(config: ModelConfig) -> Self {
        Self {
            backend: Mutex::new(None),
            config,
            initialized: AtomicBool::new(false),
        }
    }

    /// Create a client with a pre-built backend (useful for testing or
    /// for plugging in API-based backends).
    pub fn with_backend(config: ModelConfig, backend: Box<dyn ModelBackend>) -> Self {
        Self {
            backend: Mutex::new(Some(backend)),
            config,
            initialized: AtomicBool::new(true),
        }
    }

    /// Ensure the backend is initialized.
    fn ensure_init(&self) {
        if self.initialized.load(Ordering::Acquire) {
            return;
        }
        let mut guard = self.backend.lock().unwrap();
        if guard.is_none() {
            *guard = create_backend(&self.config);
            self.initialized.store(true, Ordering::Release);
        }
    }

    /// Generate text from a prompt with automatic retry (3 attempts, seed+1 each time).
    pub fn generate(&self, prompt: &str, seed: u64) -> Result<String, ModelError> {
        self.ensure_init();
        let guard = self.backend.lock().unwrap();
        let backend = guard.as_ref().ok_or(ModelError::NoModelAvailable)?;

        for attempt in 0..3u64 {
            match backend.generate(prompt, seed + attempt, self.config.max_tokens) {
                Ok(text) => return Ok(text),
                Err(ModelError::NoModelAvailable) => return Err(ModelError::NoModelAvailable),
                Err(_) if attempt < 2 => continue,
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    /// Asynchronously generate text in a background thread.
    ///
    /// Returns a `Receiver` that will contain exactly one result.
    /// Uses `std::thread::spawn` (no tokio) for determinism compatibility.
    pub fn generate_async(
        &self,
        prompt: String,
        seed: u64,
    ) -> Receiver<Result<String, ModelError>> {
        self.ensure_init();
        let (tx, rx) = bounded(1);

        let guard = self.backend.lock().unwrap();
        if guard.is_none() {
            let _ = tx.send(Err(ModelError::NoModelAvailable));
            return rx;
        }
        drop(guard);

        // The backend is behind a Mutex, so the thread needs to re-lock.
        // We share the whole client state via the channel pattern.
        let max_tokens = self.config.max_tokens;

        // For the async path we create a fresh subprocess backend to avoid
        // holding the mutex across the thread boundary.
        let config = self.config.clone();
        std::thread::spawn(move || {
            let backend = match create_backend(&config) {
                Some(b) => b,
                None => {
                    let _ = tx.send(Err(ModelError::NoModelAvailable));
                    return;
                }
            };

            let mut result = Err(ModelError::NoModelAvailable);
            for attempt in 0..3u64 {
                match backend.generate(&prompt, seed + attempt, max_tokens) {
                    Ok(text) => {
                        result = Ok(text);
                        break;
                    }
                    Err(ModelError::NoModelAvailable) => break,
                    Err(e) if attempt == 2 => {
                        result = Err(e);
                        break;
                    }
                    Err(_) => continue,
                }
            }
            let _ = tx.send(result);
        });

        rx
    }

    /// Whether a model backend is configured and its assets exist.
    pub fn is_available(&self) -> bool {
        self.ensure_init();
        let guard = self.backend.lock().unwrap();
        guard.as_ref().map_or(false, |b| b.is_available())
    }

    /// The tier of the configured model.
    pub fn tier(&self) -> ModelTier {
        self.config.tier
    }

    /// Reference to the underlying config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Build a backend from config. Returns `None` for `ProviderConfig::None`.
fn create_backend(config: &ModelConfig) -> Option<Box<dyn ModelBackend>> {
    match &config.provider {
        ProviderConfig::None => None,
        ProviderConfig::Subprocess { .. } => {
            Some(Box::new(SubprocessBackend::new(config.clone())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_with_no_provider_is_not_available() {
        let client = ModelClient::new(ModelConfig::default());
        assert!(!client.is_available());
    }

    #[test]
    fn client_generate_returns_no_model_when_none() {
        let client = ModelClient::new(ModelConfig::default());
        let result = client.generate("test prompt", 42);
        assert!(matches!(result, Err(ModelError::NoModelAvailable)));
    }

    #[test]
    fn client_async_returns_no_model_when_none() {
        let client = ModelClient::new(ModelConfig::default());
        let rx = client.generate_async("test prompt".to_string(), 42);
        let result = rx.recv().unwrap();
        assert!(matches!(result, Err(ModelError::NoModelAvailable)));
    }
}
