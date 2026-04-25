//! Lazy-compute views with staleness tracking. A `LazyView` declares which
//! event kinds invalidate it; the engine pipeline flips its staleness marker
//! when any of those kinds lands in the event ring.

use crate::cascade::EventKindId;
use crate::event::{EventLike, EventRing};
use crate::ids::AgentId;
use crate::state::SimState;

pub trait LazyView<E: EventLike>: crate::cascade::handler::__sealed::Sealed + Send + Sync {
    /// Event kinds whose emission invalidates the cached value.
    fn invalidated_by(&self) -> &[EventKindId];

    /// Recompute the cached value from state. Clears the staleness flag.
    fn compute(&mut self, state: &SimState);

    /// True when the cached value does not reflect the current state.
    fn is_stale(&self) -> bool;

    /// Called by the tick pipeline after events are emitted. Default impl
    /// checks every event in the ring against `invalidated_by()`.
    fn invalidate_on_events(&mut self, events: &EventRing<E>) {
        let kinds = self.invalidated_by();
        if kinds.is_empty() {
            return;
        }
        for e in events.iter() {
            let k = e.kind();
            if kinds.contains(&k) {
                self.mark_stale();
                return;
            }
        }
    }

    /// Flip the staleness flag to "dirty". Overrides set the flag on their
    /// internal state; `invalidate_on_events` calls this.
    fn mark_stale(&mut self);
}

/// Per-agent "who is my nearest enemy?" view, computed on demand.
/// Invalidated by any position-changing event (`AgentMoved`, `AgentFled`) or
/// by death (a dead agent cannot be "nearest" any more).
pub struct NearestEnemyLazy {
    per_agent: Vec<Option<AgentId>>,
    stale: bool,
}

const NEAREST_ENEMY_INVALIDATED_BY: &[EventKindId] = &[
    EventKindId::AgentMoved,
    EventKindId::AgentFled,
    EventKindId::AgentDied, // dead agents can't be "nearest" any more
];

impl NearestEnemyLazy {
    pub fn new(cap: usize) -> Self {
        Self {
            per_agent: vec![None; cap],
            stale: true,
        }
    }

    pub fn value(&self, id: AgentId) -> Option<AgentId> {
        if self.stale {
            return None;
        }
        self.per_agent
            .get((id.raw() - 1) as usize)
            .copied()
            .flatten()
    }
}

// The Sealed + LazyView impls for NearestEnemyLazy use the concrete
// engine_data::Event type. engine_data is a regular dep (chronicle.rs
// retains the dep until Plan B2).
impl crate::cascade::handler::__sealed::Sealed for NearestEnemyLazy {}

impl LazyView<engine_data::events::Event> for NearestEnemyLazy {
    fn invalidated_by(&self) -> &[EventKindId] {
        NEAREST_ENEMY_INVALIDATED_BY
    }

    fn is_stale(&self) -> bool {
        self.stale
    }

    fn mark_stale(&mut self) {
        self.stale = true;
    }

    fn compute(&mut self, state: &SimState) {
        for v in &mut self.per_agent {
            *v = None;
        }
        let alive: Vec<AgentId> = state.agents_alive().collect();
        for id in &alive {
            let sp = match state.agent_pos(*id) {
                Some(p) => p,
                None => continue,
            };
            let mut best: Option<(AgentId, f32)> = None;
            for other in &alive {
                if *other == *id {
                    continue;
                }
                let op = match state.agent_pos(*other) {
                    Some(p) => p,
                    None => continue,
                };
                let d = op.distance(sp);
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((*other, d));
                }
            }
            if let Some((target, _)) = best {
                let slot = (id.raw() - 1) as usize;
                if let Some(cell) = self.per_agent.get_mut(slot) {
                    *cell = Some(target);
                }
            }
        }
        self.stale = false;
    }
}
