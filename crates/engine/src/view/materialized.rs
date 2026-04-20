use crate::event::{Event, EventRing};
use crate::ids::AgentId;

/// Trait implemented by views that want to be rebuilt/updated by folding over an
/// event log. The view owns its own storage; folds are pure-append semantics.
///
/// Implementors provide `fold_since(events, events_before)` which sees only the
/// slice of events pushed since the `events_before` cursor — the expected
/// per-tick call pattern. The legacy `fold(events)` helper folds the entire
/// retained ring and is kept for test fixtures that rebuild a view after the
/// fact (e.g. `acceptance.rs` post-run aggregation). Default implementations
/// express the equivalence: `fold` is `fold_since(events, 0)`.
pub trait MaterializedView {
    /// Fold events with push index `>= events_before`. Called from
    /// `step_full` phase 5 once per tick with the ring's `push_count()`
    /// at the top of the tick.
    fn fold_since(&mut self, events: &EventRing, events_before: usize);

    /// Fold the whole retained ring. Equivalent to `fold_since(events, 0)`.
    /// Kept for post-run aggregation (tests, trajectory export) where a
    /// per-tick fold was never wired.
    fn fold(&mut self, events: &EventRing) {
        self.fold_since(events, 0);
    }
}

/// Per-agent accumulated damage taken. Backed by a flat Vec indexed by
/// `AgentId::raw() - 1`. The vector length should match the agent capacity of
/// the SimState that produced the events.
pub struct DamageTaken {
    per_agent: Vec<f32>,
}

impl DamageTaken {
    pub fn new(cap: usize) -> Self {
        Self { per_agent: vec![0.0; cap] }
    }

    pub fn value(&self, id: AgentId) -> f32 {
        self.per_agent
            .get((id.raw() - 1) as usize)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn reset(&mut self) {
        for v in &mut self.per_agent { *v = 0.0; }
    }
}

impl MaterializedView for DamageTaken {
    fn fold_since(&mut self, events: &EventRing, events_before: usize) {
        for e in events.iter_since(events_before) {
            if let Event::AgentAttacked { target, damage, .. } = e {
                let slot = (target.raw() - 1) as usize;
                if let Some(v) = self.per_agent.get_mut(slot) {
                    *v += *damage;
                }
            }
        }
    }
}
