use crate::event::{EventLike, EventRing};
use crate::state::SimState;

pub trait Invariant<E: EventLike>: Send + Sync {
    fn name(&self) -> &'static str;
    fn failure_mode(&self) -> FailureMode { FailureMode::Log }
    fn check(&self, state: &SimState, events: &EventRing<E>) -> Option<Violation>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FailureMode {
    Panic,
    Log,
}

#[derive(Clone, Debug)]
pub struct Violation {
    pub invariant: &'static str,
    pub tick:      u32,
    pub message:   String,
    pub payload:   Option<String>,
}
