use super::Event;
use crate::ids::EventId;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

/// An entry in the `EventRing`. The `event` is the replay payload; `id` and
/// `cause` form the sidecar metadata (never folded into `replayable_sha256`).
struct Entry {
    event: Event,
    id:    EventId,
    cause: Option<EventId>,
}

pub struct EventRing {
    entries:      VecDeque<Entry>,
    cap:          usize,
    current_tick: u32,
    next_seq:     u32,
    total_pushed: usize,
    /// Monotonic cursor: the index of the next event to dispatch. Updated by
    /// `CascadeRegistry::run_fixed_point` via `set_dispatched(..)`. Survives
    /// ring eviction because it's an index into `total_pushed`, not into
    /// `entries`.
    dispatched:   usize,
}

impl EventRing {
    pub fn with_cap(cap: usize) -> Self {
        debug_assert!(cap > 0, "EventRing capacity must be nonzero");
        Self {
            entries:      VecDeque::with_capacity(cap),
            cap,
            current_tick: 0,
            next_seq:     0,
            total_pushed: 0,
            dispatched:   0,
        }
    }

    /// Push a root-cause event. Returns the assigned `EventId`.
    pub fn push(&mut self, e: Event) -> EventId {
        self.push_impl(e, None)
    }

    /// Push an event caused by an earlier event. The `cause` pointer lives in
    /// the sidecar — it does NOT affect `replayable_sha256`.
    pub fn push_caused_by(&mut self, e: Event, cause: EventId) -> EventId {
        self.push_impl(e, Some(cause))
    }

    fn push_impl(&mut self, e: Event, cause: Option<EventId>) -> EventId {
        let tick = e.tick();
        if tick != self.current_tick {
            self.current_tick = tick;
            self.next_seq = 0;
        }
        if self.entries.len() == self.cap {
            self.entries.pop_front();
        }
        let id = EventId { tick, seq: self.next_seq };
        self.next_seq += 1;
        self.total_pushed += 1;
        self.entries.push_back(Entry { event: e, id, cause });
        id
    }

    /// Total number of pushes ever — even those evicted from the ring.
    pub fn total_pushed(&self) -> usize { self.total_pushed }

    /// The index of the next undispatched event. Used by the cascade dispatcher
    /// to resume scanning across multiple `run_fixed_point` calls.
    pub fn dispatched(&self) -> usize { self.dispatched }

    /// Advance the dispatched-cursor. Clamped to `total_pushed`.
    pub fn set_dispatched(&mut self, idx: usize) {
        self.dispatched = idx.min(self.total_pushed);
    }

    /// Look up an event by its monotonic push index. Returns None if the event
    /// has been evicted or if the index is out of range.
    pub fn get_pushed(&self, idx: usize) -> Option<Event> {
        let first = self.total_pushed.saturating_sub(self.entries.len());
        if idx < first { return None; }
        let local = idx - first;
        self.entries.get(local).map(|e| e.event)
    }

    /// Look up the parent event id for a given event id, if any.
    pub fn cause_of(&self, id: EventId) -> Option<EventId> {
        self.entries.iter().find(|e| e.id == id).and_then(|e| e.cause)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> { self.entries.iter().map(|e| &e.event) }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Stable hash over the replayable subset. Uses explicit byte-packing
    /// (via `f32::to_bits`) so the digest is stable across Rust/glam versions
    /// and is insensitive to Debug format drift. Schema-hash-load-bearing:
    /// changing the pack format or variant-tag order requires a schema-hash bump.
    ///
    /// Sidecar fields (`id`, `cause`) are INTENTIONALLY excluded from the hash
    /// so cascade fan-out annotations cannot alter replay equivalence.
    pub fn replayable_sha256(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for entry in self.entries.iter().filter(|e| e.event.is_replayable()) {
            hash_event(&mut h, &entry.event);
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
        Event::AgentFled { agent_id, from, to, tick } => {
            h.update([3u8]);                                   // variant tag
            h.update(agent_id.raw().to_le_bytes());
            for v in [from.x, from.y, from.z, to.x, to.y, to.z] {
                h.update(v.to_bits().to_le_bytes());
            }
            h.update(tick.to_le_bytes());
        }
        Event::ChronicleEntry { .. } => {
            // Filtered at the call site; if we reach here, the filter is broken.
            debug_assert!(false, "ChronicleEntry reached replayable hash path");
        }
    }
}
