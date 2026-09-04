//! Belief state — per-(observer, target) fact tuple maintained by the
//! Theory-of-Mind subsystem. Stored in `SimState.cold_beliefs[observer]`
//! as a `BoundedMap<AgentId, BeliefState, 8>`.
//!
//! Hand-written (not DSL-emitted), matching `SimState` which is also
//! hand-written in `engine`.

use crate::entities::CreatureType;
use glam::Vec3;

/// Per-(observer, target) belief tuple stored in `SimState.cold_beliefs`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeliefState {
    /// Last observed world-space position of the target.
    pub last_known_pos: Vec3,
    /// Last observed current HP of the target.
    pub last_known_hp: f32,
    /// Last observed maximum HP of the target.
    pub last_known_max_hp: f32,
    /// Last observed creature type of the target.
    pub last_known_creature_type: CreatureType,
    /// Engine tick at which this belief was last updated.
    pub last_updated_tick: u32,
    /// Confidence in [0, 1]; entries below `EVICTION_THRESHOLD` are culled.
    pub confidence: f32,
}

/// Maximum number of targets a single observer tracks simultaneously.
/// Matches the `N` const on `BoundedMap<AgentId, BeliefState, BELIEFS_PER_AGENT>`.
pub const BELIEFS_PER_AGENT: usize = 8;

/// Beliefs with confidence below this threshold are evicted from the map.
pub const EVICTION_THRESHOLD: f32 = 0.05;
