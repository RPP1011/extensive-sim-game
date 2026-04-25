use engine::event::{Event, EventRing};
use engine::invariant::{FailureMode, Invariant, InvariantRegistry, Violation};
use engine::state::SimState;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
struct Report(Arc<Mutex<Vec<String>>>, &'static str, FailureMode, bool);
impl Invariant<Event> for Report {
    fn name(&self) -> &'static str { self.1 }
    fn failure_mode(&self) -> FailureMode { self.2 }
    fn check(&self, _s: &SimState, _e: &EventRing<Event>) -> Option<Violation> {
        if self.3 {
            Some(Violation {
                invariant: self.1, tick: 0,
                message: "fail".into(), payload: None,
            })
        } else { None }
    }
}

#[test]
fn healthy_invariants_return_no_violations() {
    let mut reg = InvariantRegistry::<Event>::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "ok", FailureMode::Log, false)));
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(8);
    let violations = reg.check_all(&state, &events);
    assert!(violations.is_empty());
}

#[test]
fn violated_log_mode_returns_violations_but_does_not_panic() {
    let mut reg = InvariantRegistry::<Event>::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "bad", FailureMode::Log, true)));
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(8);
    let violations = reg.check_all(&state, &events);
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].violation.invariant, "bad");
    assert_eq!(violations[0].failure_mode, FailureMode::Log);
}

#[test]
#[should_panic(expected = "invariant violated in Panic mode")]
fn violated_panic_mode_panics_immediately() {
    let mut reg = InvariantRegistry::<Event>::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "boom", FailureMode::Panic, true)));
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(8);
    let _ = reg.check_all(&state, &events);
}

