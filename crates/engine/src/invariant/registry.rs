use super::{FailureMode, Invariant, Violation};
use crate::event::{EventLike, EventRing};
use crate::state::SimState;

#[derive(Clone, Debug)]
pub struct ViolationReport {
    pub violation:    Violation,
    pub failure_mode: FailureMode,
}

pub struct InvariantRegistry<E: EventLike> {
    invariants: Vec<Box<dyn Invariant<E>>>,
}

impl<E: EventLike> InvariantRegistry<E> {
    pub fn new() -> Self { Self { invariants: Vec::new() } }

    pub fn register(&mut self, inv: Box<dyn Invariant<E>>) {
        self.invariants.push(inv);
    }

    pub fn check_all(&self, state: &SimState, events: &EventRing<E>) -> Vec<ViolationReport> {
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

impl<E: EventLike> Default for InvariantRegistry<E> { fn default() -> Self { Self::new() } }
