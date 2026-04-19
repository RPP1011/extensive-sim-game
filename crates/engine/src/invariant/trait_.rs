use crate::event::EventRing;
use crate::state::SimState;

pub trait Invariant: Send + Sync {
    fn name(&self) -> &'static str;
    fn failure_mode(&self) -> FailureMode { FailureMode::Log }
    fn check(&self, state: &SimState, events: &EventRing) -> Option<Violation>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FailureMode {
    Panic,
    Log,
    Rollback { ticks: u32 },
}

#[derive(Clone, Debug)]
pub struct Violation {
    pub invariant: &'static str,
    pub tick:      u32,
    pub message:   String,
    pub payload:   Option<String>,
}
