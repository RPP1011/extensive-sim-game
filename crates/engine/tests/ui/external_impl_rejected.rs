use engine::cascade::{CascadeHandler, EventKindId, Lane};
use engine::event::EventRing;
use engine::state::SimState;
use engine_data::events::Event;

struct MyHandler;

impl CascadeHandler<Event> for MyHandler {
    type Views = ();
    fn trigger(&self) -> EventKindId { EventKindId::AgentMoved }
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(
        &self,
        _event: &Event,
        _state: &mut SimState,
        _views: &mut (),
        _events: &mut EventRing<Event>,
    ) {}
}

fn main() {}
