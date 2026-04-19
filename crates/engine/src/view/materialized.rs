use crate::event::{Event, EventRing};
use crate::ids::AgentId;

/// Trait implemented by views that want to be rebuilt/updated by folding over an
/// event log. The view owns its own storage; `fold` is pure-append semantics.
pub trait MaterializedView {
    fn fold(&mut self, events: &EventRing);
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
    fn fold(&mut self, events: &EventRing) {
        for e in events.iter() {
            if let Event::AgentAttacked { target, damage, .. } = e {
                let slot = (target.raw() - 1) as usize;
                if let Some(v) = self.per_agent.get_mut(slot) {
                    *v += *damage;
                }
            }
        }
    }
}
