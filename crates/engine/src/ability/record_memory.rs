//! Audit fix HIGH #4 — `RecordMemory` cascade handler.
//!
//! `step.rs::Announce` emits one `Event::RecordMemory` per recipient
//! (primary at 0.8 confidence, overhear at 0.6). Before this handler
//! existed, nothing consumed those events and the per-agent
//! `cold_memory` event ring stayed empty forever.
//!
//! This handler pushes a `MemoryEvent` into the recipient's
//! `cold_memory` slot so downstream consumers (GOAP planners, chronicles,
//! theory-of-mind) can actually find the broadcasts their agents heard.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::agent_types::MemoryEvent;
use crate::state::SimState;

pub struct RecordMemoryHandler;

impl CascadeHandler for RecordMemoryHandler {
    fn trigger(&self) -> EventKindId { EventKindId::RecordMemory }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (observer, source, fact_payload, confidence, tick) = match *event {
            Event::RecordMemory { observer, source, fact_payload, confidence, tick } =>
                (observer, source, fact_payload, confidence, tick),
            _ => return,
        };
        let confidence_q8 = (confidence.clamp(0.0, 1.0) * 255.0) as u8;
        state.push_agent_memory(observer, MemoryEvent {
            source,
            kind: 0,
            payload: fact_payload,
            confidence_q8,
            tick,
        });
    }
}
