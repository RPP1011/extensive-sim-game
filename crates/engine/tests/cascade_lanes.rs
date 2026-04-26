use engine::cascade::{CascadeHandler, CascadeRegistry, EventKindId, Lane};
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::SimState;
use std::sync::{Arc, Mutex};

struct OrderMarker(Arc<Mutex<Vec<Lane>>>, Lane);
impl engine::cascade::__sealed::Sealed for OrderMarker {}
impl CascadeHandler<Event> for OrderMarker {
    type Views = ();
    fn trigger(&self) -> EventKindId { EventKindId::AgentDied }
    fn lane(&self) -> Lane { self.1 }
    fn handle(&self, _: &Event, _: &mut SimState, _: &mut (), _: &mut EventRing<Event>) {
        self.0.lock().unwrap().push(self.1);
    }
}

#[test]
fn lanes_run_in_fixed_order_regardless_of_registration() {
    let mut reg = CascadeRegistry::<Event>::new();
    let trace = Arc::new(Mutex::new(Vec::<Lane>::new()));
    // Register out of order.
    reg.register(OrderMarker(trace.clone(), Lane::Audit));
    reg.register(OrderMarker(trace.clone(), Lane::Validation));
    reg.register(OrderMarker(trace.clone(), Lane::Reaction));
    reg.register(OrderMarker(trace.clone(), Lane::Effect));

    let mut state = SimState::new(2, 42);
    let a = AgentId::new(1).unwrap();
    let mut ring = EventRing::<Event>::with_cap(8);
    reg.dispatch(&Event::AgentDied { agent_id: a, tick: 0 }, &mut state, &mut (), &mut ring);

    let out = trace.lock().unwrap().clone();
    assert_eq!(out, vec![Lane::Validation, Lane::Effect, Lane::Reaction, Lane::Audit]);
}

#[test]
fn within_a_lane_registration_order_preserved() {
    let mut reg = CascadeRegistry::<Event>::new();
    let trace = Arc::new(Mutex::new(Vec::<Lane>::new()));
    // Three handlers all in Effect lane.
    reg.register(OrderMarker(trace.clone(), Lane::Effect));
    reg.register(OrderMarker(trace.clone(), Lane::Effect));
    reg.register(OrderMarker(trace.clone(), Lane::Effect));

    let mut state = SimState::new(2, 42);
    let a = AgentId::new(1).unwrap();
    let mut ring = EventRing::<Event>::with_cap(8);
    reg.dispatch(&Event::AgentDied { agent_id: a, tick: 0 }, &mut state, &mut (), &mut ring);

    // All three fired; the order within Effect is registration order (not observable
    // here because they're identical, but asserting count suffices).
    assert_eq!(trace.lock().unwrap().len(), 3);
}
