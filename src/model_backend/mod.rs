//! Provider-agnostic model integration layer.
//!
//! Defines the [`ModelBackend`] trait that any text generation model can
//! implement — subprocess-based local models (LFM, llama.cpp), remote APIs,
//! or in-process Burn models. The [`ModelClient`] provides the consumer-facing
//! facade with lazy loading, retry logic, and async dispatch.
//!
//! ## Designed for extensibility
//!
//! When Issue #14 Task 09 (LFM-Orchestrated Narrative System) lands, it
//! should implement `ModelBackend` and plug into the same `ModelClient`
//! used by the AOT pipeline (Task 12) and ASCII art generator (Task 13).

mod backend;
mod client;
mod config;
mod prompt;
mod tier;

pub use backend::{ModelBackend, ModelError, SubprocessBackend};
pub use client::ModelClient;
pub use config::{ModelConfig, ProviderConfig};
pub use prompt::{format_json_generation_prompt, format_text_prompt};
pub use tier::{detect_best_tier, detect_gpu_available, DevicePreference, ModelTier};
