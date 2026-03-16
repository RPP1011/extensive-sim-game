//! V6 model checkpoint save/load using Burn's record system.
//!
//! Provides CLI-friendly save/load that works with both NdArray (CPU)
//! and LibTorch (GPU) backends.

use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use super::actor_critic_v6::{ActorCriticV6, ActorCriticV6Config};

/// Save a V6 model to a binary checkpoint file.
///
/// The file is written using Burn's BinFileRecorder with full precision.
/// The `.bin` extension is added automatically by Burn.
pub fn save_v6<B: Backend>(
    model: &ActorCriticV6<B>,
    path: &std::path::Path,
) -> Result<(), String> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path, &recorder)
        .map_err(|e| format!("Failed to save V6 checkpoint to {}: {e}", path.display()))
}

/// Load a V6 model from a binary checkpoint file.
///
/// Creates a model from the given config, then loads weights from file.
pub fn load_v6<B: Backend>(
    config: &ActorCriticV6Config,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<ActorCriticV6<B>, String> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = config.init::<B>(device);
    model
        .load_file(path, &recorder, device)
        .map_err(|e| format!("Failed to load V6 checkpoint from {}: {e}", path.display()))
}

/// Save a V5 model to a binary checkpoint file.
pub fn save_v5<B: Backend>(
    model: &super::actor_critic::ActorCriticV5<B>,
    path: &std::path::Path,
) -> Result<(), String> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path, &recorder)
        .map_err(|e| format!("Failed to save V5 checkpoint to {}: {e}", path.display()))
}

/// Load a V5 model from a binary checkpoint file.
pub fn load_v5<B: Backend>(
    config: &super::actor_critic::ActorCriticV5Config,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<super::actor_critic::ActorCriticV5<B>, String> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = config.init::<B>(device);
    model
        .load_file(path, &recorder, device)
        .map_err(|e| format!("Failed to load V5 checkpoint from {}: {e}", path.display()))
}
