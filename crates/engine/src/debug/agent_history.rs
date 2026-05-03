//! Per-agent state delta tracker.
//!
//! Captures the SoA fields each agent has at each tick. Useful for
//! "what changed for agent X between tick T and T+1" debugging.
//!
//! # Usage
//!
//! ```rust,ignore
//! use engine::debug::{DebugConfig, agent_history::{AgentHistory, Filter}};
//! use std::sync::{Arc, Mutex};
//!
//! let history = Arc::new(Mutex::new(AgentHistory::new(Filter {
//!     agents: None,      // capture all agents
//!     max_ticks: 64,     // ring-buffer: keep last 64 ticks
//! })));
//! let cfg = DebugConfig { agent_history: Some(Arc::clone(&history)), ..Default::default() };
//! // run ticks ...
//! let h = history.lock().unwrap();
//! for (tick, snap) in h.agent_trajectory(some_id) { ... }
//! ```

use crate::ids::AgentId;
use crate::state::SimState;
use engine_data::entities::CreatureType;
use std::collections::HashMap;

/// Filter controlling which agents to capture and how many ticks to retain.
#[derive(Debug, Clone)]
pub struct Filter {
    /// `None` = capture all alive agents; `Some(vec)` = only the listed ids.
    pub agents: Option<Vec<AgentId>>,
    /// Ring-buffer capacity: once `snapshots.len() > max_ticks` the oldest
    /// entry is evicted. Must be `>= 1`.
    pub max_ticks: usize,
}

impl Default for Filter {
    fn default() -> Self {
        Self { agents: None, max_ticks: 256 }
    }
}

/// Snapshot of all tracked agents at a single tick.
#[derive(Debug)]
pub struct TickSnapshot {
    pub tick: u32,
    pub per_agent: HashMap<AgentId, AgentSnapshot>,
}

/// Snapshot of one agent's observable state at one tick.
#[derive(Debug, Clone)]
pub struct AgentSnapshot {
    pub alive: bool,
    pub hp: f32,
    pub position: glam::Vec3,
    pub creature_type: Option<CreatureType>,
}

/// Ring-buffer collector for per-agent state deltas.
///
/// Wrap in `Arc<Mutex<AgentHistory>>` and store in `DebugConfig::agent_history`.
/// The tick pipeline calls `record` at `TickEnd` phase.
#[derive(Debug)]
pub struct AgentHistory {
    /// Ring buffer, front = oldest, back = newest.
    snapshots: Vec<TickSnapshot>,
    filter: Filter,
}

impl AgentHistory {
    pub fn new(filter: Filter) -> Self {
        let cap = filter.max_ticks.max(1);
        Self { snapshots: Vec::with_capacity(cap), filter }
    }

    /// Record the state of all (filtered) agents at `tick`.
    ///
    /// Called by the generated `step` function at `TickEnd`. After this call the
    /// tick counter has already been advanced, so callers pass `state.tick - 1`
    /// (the tick that just finished). The generated code handles this by
    /// capturing the tick value before the increment.
    ///
    /// This is a no-op on the critical path when `agent_history` is `None` —
    /// the caller guards the lock on `Some`.
    pub fn record(&mut self, tick: u32, state: &SimState) {
        let mut per_agent = HashMap::new();
        for agent in state.agents_alive() {
            if let Some(filter_list) = &self.filter.agents {
                if !filter_list.contains(&agent) {
                    continue;
                }
            }
            per_agent.insert(agent, AgentSnapshot {
                alive: state.agent_alive(agent),
                hp: state.agent_hp(agent).unwrap_or(0.0),
                position: state.agent_pos(agent).unwrap_or(glam::Vec3::ZERO),
                creature_type: state.agent_creature_type(agent),
            });
        }
        self.snapshots.push(TickSnapshot { tick, per_agent });
        // Enforce ring-buffer max_ticks eviction.
        if self.snapshots.len() > self.filter.max_ticks.max(1) {
            self.snapshots.remove(0);
        }
    }

    /// Return the snapshot for `tick`, if retained.
    pub fn at_tick(&self, tick: u32) -> Option<&TickSnapshot> {
        self.snapshots.iter().find(|s| s.tick == tick)
    }

    /// Iterate over all retained snapshots for `agent`, in ascending tick order.
    pub fn agent_trajectory(&self, agent: AgentId) -> impl Iterator<Item = (u32, &AgentSnapshot)> {
        self.snapshots
            .iter()
            .filter_map(move |t| t.per_agent.get(&agent).map(|s| (t.tick, s)))
    }

    /// Number of tick snapshots currently retained.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}
