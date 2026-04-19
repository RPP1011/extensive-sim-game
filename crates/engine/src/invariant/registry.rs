use super::{FailureMode, Invariant, Violation};
use crate::event::EventRing;
use crate::state::SimState;

#[derive(Clone, Debug)]
pub struct ViolationReport {
    pub violation:    Violation,
    pub failure_mode: FailureMode,
}

pub struct InvariantRegistry {
    invariants: Vec<Box<dyn Invariant>>,
}

impl InvariantRegistry {
    pub fn new() -> Self { Self { invariants: Vec::new() } }

    pub fn register(&mut self, inv: Box<dyn Invariant>) {
        self.invariants.push(inv);
    }

    pub fn check_all(&self, state: &SimState, events: &EventRing) -> Vec<ViolationReport> {
        let mut reports = Vec::new();
        for inv in &self.invariants {
            if let Some(v) = inv.check(state, events) {
                let mode = inv.failure_mode();
                if mode == FailureMode::Panic {
                    panic!("invariant violated in Panic mode: {} — {}", v.invariant, v.message);
                }
                reports.push(ViolationReport { violation: v, failure_mode: mode });
            }
        }
        reports
    }

    pub fn len(&self) -> usize { self.invariants.len() }
    pub fn is_empty(&self) -> bool { self.invariants.is_empty() }
}

impl Default for InvariantRegistry { fn default() -> Self { Self::new() } }
