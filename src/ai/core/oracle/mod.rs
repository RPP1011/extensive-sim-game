mod rollout;
mod squad;
mod focus;

#[allow(unused_imports)]
pub use rollout::{
    score_actions, score_actions_with_depth,
    threat_per_sec,
};
#[allow(unused_imports)]
pub(crate) use rollout::{enumerate_candidates, run_rollout, score_rollout};
#[allow(unused_imports)]
pub use squad::{SquadPlan, squad_oracle};
#[allow(unused_imports)]
pub(crate) use squad::run_squad_rollout;
#[allow(unused_imports)]
pub use focus::{FocusCandidate, search_focus_target};

use serde::{Deserialize, Serialize};

use super::{IntentAction, FIXED_TICK_MS};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Metadata about a scored action from a rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredAction {
    pub action: IntentAction,
    pub score: f64,
    pub enemy_hp_lost: i32,
    pub ally_hp_lost: i32,
    pub kills: u32,
    pub cc_ticks_applied: u32,
}

/// Full oracle result for one unit at one decision point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleResult {
    pub unit_id: u32,
    pub tick: u64,
    pub scored_actions: Vec<ScoredAction>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default rollout depth in ticks.
pub const DEFAULT_ROLLOUT_TICKS: u64 = 10;

/// Bonus score per kill (scaled by target threat).
pub(crate) const KILL_BONUS_PER_DPS: f64 = 20.0;

/// Value multiplier for CC applied (threat × duration).
pub(crate) const CC_VALUE_FACTOR: f64 = 0.1;

/// Early-exit: if a rollout's score is this much worse than the best seen
/// after the checkpoint tick, abort it.
pub(crate) const EARLY_EXIT_THRESHOLD: f64 = 50.0;

/// Tick at which early-exit comparison kicks in.
pub(crate) const EARLY_EXIT_CHECKPOINT: u64 = 20;

/// Max range factor for attack candidates — skip if target is further than
/// attack_range × this factor (the unit would just walk for most of the rollout).
pub(crate) const ATTACK_RANGE_FACTOR: f32 = 3.0;
