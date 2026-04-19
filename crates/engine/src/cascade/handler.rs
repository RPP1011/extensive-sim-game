use crate::event::{Event, EventRing};
use crate::state::SimState;

/// Stable ordinal identifying an event variant. Dense so it indexes arrays
/// cheaply. Adding a variant appends; reordering is a schema-hash bump.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum EventKindId {
    AgentMoved      = 0,
    AgentAttacked   = 1,
    AgentDied       = 2,
    AgentFled       = 3,
    // Slots 4-127 reserved for replayable event variants added in later tasks.
    ChronicleEntry  = 128,
}

impl EventKindId {
    pub fn from_event(e: &Event) -> EventKindId {
        match e {
            Event::AgentMoved     { .. } => EventKindId::AgentMoved,
            Event::AgentAttacked  { .. } => EventKindId::AgentAttacked,
            Event::AgentDied      { .. } => EventKindId::AgentDied,
            Event::AgentFled      { .. } => EventKindId::AgentFled,
            Event::ChronicleEntry { .. } => EventKindId::ChronicleEntry,
        }
    }
}

/// Lane discipline — handlers within a lane run in registration order;
/// lanes run in the order listed here.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Lane {
    Validation = 0,
    Effect     = 1,
    Reaction   = 2,
    Audit      = 3,
}

impl Lane {
    pub const ALL: &'static [Lane] = &[
        Lane::Validation, Lane::Effect, Lane::Reaction, Lane::Audit,
    ];
}

pub trait CascadeHandler: Send + Sync {
    fn trigger(&self) -> EventKindId;
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing);
}
