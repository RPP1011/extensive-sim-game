use engine::cascade::EventKindId;
use engine::event::{Event, EventRing};
use engine::invariant::{FailureMode, Invariant, Violation};
use engine::state::SimState;

struct AlwaysFails;
impl Invariant<Event> for AlwaysFails {
    fn name(&self) -> &'static str { "always_fails" }
    fn failure_mode(&self) -> FailureMode { FailureMode::Log }
    fn check(&self, _state: &SimState, _events: &EventRing<Event>) -> Option<Violation> {
        Some(Violation {
            invariant: self.name(),
            tick: 0,
            message: "on purpose".into(),
            payload: None,
        })
    }
}

#[test]
fn trait_is_object_safe() {
    let v: Box<dyn Invariant<Event>> = Box::new(AlwaysFails);
    assert_eq!(v.name(), "always_fails");
    assert_eq!(v.failure_mode(), FailureMode::Log);
}

#[test]
fn violation_carries_tick_and_message() {
    let state = SimState::new(2, 42);
    let events = EventRing::<Event>::with_cap(8);
    let v = AlwaysFails;
    let report = v.check(&state, &events).unwrap();
    assert_eq!(report.invariant, "always_fails");
    assert_eq!(report.tick, 0);
    assert_eq!(report.message, "on purpose");
}

#[test]
fn failure_mode_variants() {
    let _ = FailureMode::Panic;
    let _ = FailureMode::Log;
    let _ = EventKindId::AgentMoved;
}
