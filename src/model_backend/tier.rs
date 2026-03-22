use serde::{Deserialize, Serialize};

/// Performance tier indicating model size and expected hardware requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    /// 1-3B parameters, CPU-only, <2s latency target.
    Micro,
    /// ~7B parameters, 8GB+ GPU, <500ms latency target.
    Standard,
    /// 13B+ parameters, 16GB+ GPU, <1s latency target.
    Full,
}

impl ModelTier {
    /// Minimum VRAM in megabytes required for this tier (0 = CPU-only).
    pub fn min_vram_mb(self) -> u64 {
        match self {
            ModelTier::Micro => 0,
            ModelTier::Standard => 8 * 1024,
            ModelTier::Full => 16 * 1024,
        }
    }
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelTier::Micro => write!(f, "micro"),
            ModelTier::Standard => write!(f, "standard"),
            ModelTier::Full => write!(f, "full"),
        }
    }
}

/// Hardware preference for model execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    /// Automatically select the best available device.
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Use a specific GPU by index.
    Gpu(usize),
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::Auto
    }
}

/// Detect whether a CUDA-capable GPU is likely available.
///
/// Checks for the presence of `nvidia-smi` — the same heuristic used
/// by the existing `train_v6` xtask command.
pub fn detect_gpu_available() -> bool {
    std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Select the highest tier supported by the current hardware.
pub fn detect_best_tier() -> ModelTier {
    if detect_gpu_available() {
        // Conservative: assume standard tier unless we can query VRAM.
        ModelTier::Standard
    } else {
        ModelTier::Micro
    }
}
