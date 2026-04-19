use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

/// Dense slot count covering all `EventKindId` ordinals — includes the 128+
/// chronicle reservation. `Vec<Vec<Box<dyn CascadeHandler>>>` indexed by
/// `[lane as usize][kind as u8 as usize]`.
const KIND_SLOTS: usize = 256;

pub struct CascadeRegistry {
    table: Vec<Vec<Vec<Box<dyn CascadeHandler>>>>,
}

impl CascadeRegistry {
    pub fn new() -> Self {
        let per_lane: Vec<Vec<Box<dyn CascadeHandler>>> =
            (0..KIND_SLOTS).map(|_| Vec::new()).collect();
        Self {
            table: (0..Lane::ALL.len()).map(|_| per_lane.iter().map(|_| Vec::new()).collect()).collect(),
        }
    }

    pub fn register<H: CascadeHandler + 'static>(&mut self, h: H) {
        let lane = h.lane() as usize;
        let kind = h.trigger() as u8 as usize;
        self.table[lane][kind].push(Box::new(h));
    }

    pub fn dispatch(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let kind = EventKindId::from_event(event) as u8 as usize;
        for lane in Lane::ALL {
            for handler in &self.table[*lane as usize][kind] {
                handler.handle(event, state, events);
            }
        }
    }
}

impl Default for CascadeRegistry {
    fn default() -> Self { Self::new() }
}
