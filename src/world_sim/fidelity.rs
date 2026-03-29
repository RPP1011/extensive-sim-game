use serde::{Deserialize, Serialize};

/// Simulation fidelity level, determined by location (not entity type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Fidelity {
    /// Full tactical sim: 100ms tick, per-ability combat.
    High,
    /// Settlement activity: movement, production, consumption, trade.
    Medium,
    /// Overworld travel: coarse movement, encounter detection.
    Low,
    /// Off-screen: no per-entity deltas, only aggregate stats.
    Background,
}

impl Default for Fidelity {
    fn default() -> Self {
        Fidelity::Background
    }
}
