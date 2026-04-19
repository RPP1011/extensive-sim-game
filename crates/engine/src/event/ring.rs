use super::Event;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

pub struct EventRing {
    buf: VecDeque<Event>,
    cap: usize,
}

impl EventRing {
    pub fn with_cap(cap: usize) -> Self {
        debug_assert!(cap > 0, "EventRing capacity must be nonzero");
        Self { buf: VecDeque::with_capacity(cap), cap }
    }

    pub fn push(&mut self, e: Event) {
        if self.buf.len() == self.cap { self.buf.pop_front(); }
        self.buf.push_back(e);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> { self.buf.iter() }

    pub fn len(&self) -> usize { self.buf.len() }
    pub fn is_empty(&self) -> bool { self.buf.is_empty() }

    /// Stable hash over the replayable subset. Uses explicit byte-packing
    /// (via `f32::to_bits`) so the digest is stable across Rust/glam versions
    /// and is insensitive to Debug format drift. Schema-hash-load-bearing:
    /// changing the pack format or variant-tag order requires a schema-hash bump.
    pub fn replayable_sha256(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for e in self.buf.iter().filter(|e| e.is_replayable()) {
            hash_event(&mut h, e);
        }
        h.finalize().into()
    }
}

fn hash_event(h: &mut Sha256, e: &Event) {
    match e {
        Event::AgentMoved { agent_id, from, to, tick } => {
            h.update([0u8]);                                   // variant tag
            h.update(agent_id.raw().to_le_bytes());
            for v in [from.x, from.y, from.z, to.x, to.y, to.z] {
                h.update(v.to_bits().to_le_bytes());
            }
            h.update(tick.to_le_bytes());
        }
        Event::AgentAttacked { attacker, target, damage, tick } => {
            h.update([1u8]);
            h.update(attacker.raw().to_le_bytes());
            h.update(target.raw().to_le_bytes());
            h.update(damage.to_bits().to_le_bytes());
            h.update(tick.to_le_bytes());
        }
        Event::AgentDied { agent_id, tick } => {
            h.update([2u8]);
            h.update(agent_id.raw().to_le_bytes());
            h.update(tick.to_le_bytes());
        }
        Event::ChronicleEntry { .. } => {
            // Filtered at the call site; if we reach here, the filter is broken.
            debug_assert!(false, "ChronicleEntry reached replayable hash path");
        }
    }
}
