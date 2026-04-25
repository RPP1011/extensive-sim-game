use super::EventLike;
use crate::ids::EventId;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

/// An entry in the `EventRing`. The `event` is the replay payload; `id` and
/// `cause` form the sidecar metadata (never folded into `replayable_sha256`).
struct Entry<E: EventLike> {
    event: E,
    id:    EventId,
    cause: Option<EventId>,
}

pub struct EventRing<E: EventLike> {
    entries:      VecDeque<Entry<E>>,
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

impl<E: EventLike> EventRing<E> {
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
    pub fn push(&mut self, e: E) -> EventId {
        self.push_impl(e, None)
    }

    /// Push an event caused by an earlier event. The `cause` pointer lives in
    /// the sidecar — it does NOT affect `replayable_sha256`.
    pub fn push_caused_by(&mut self, e: E, cause: EventId) -> EventId {
        self.push_impl(e, Some(cause))
    }

    fn push_impl(&mut self, e: E, cause: Option<EventId>) -> EventId {
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

    /// Alias for [`total_pushed`]: the monotonic push counter. Used by
    /// per-tick view folds to snapshot "where was the ring at the start
    /// of this tick" so `iter_since(start)` yields only events pushed
    /// during the current tick. Naming mirrors spec §7.1 wording.
    pub fn push_count(&self) -> usize { self.total_pushed }

    /// Iterate events pushed with monotonic index `>= start_idx`. `start_idx`
    /// is a value returned by [`push_count`] (or [`total_pushed`]) at an
    /// earlier point; the returned iterator yields events in push order.
    ///
    /// Events evicted from the ring between the snapshot and this call are
    /// silently skipped — the iterator always starts at the earliest
    /// still-retained event whose push index is `>= start_idx`. In the
    /// per-tick view-fold use case this never happens in practice: the
    /// snapshot is taken at the top of the tick and the iterator is drained
    /// before any further pushes, well before the retention cap matters.
    pub fn iter_since(&self, start_idx: usize) -> impl Iterator<Item = &E> {
        let first = self.total_pushed.saturating_sub(self.entries.len());
        let skip = start_idx.saturating_sub(first);
        self.entries.iter().skip(skip).map(|e| &e.event)
    }

    /// The index of the next undispatched event. Used by the cascade dispatcher
    /// to resume scanning across multiple `run_fixed_point` calls.
    pub fn dispatched(&self) -> usize { self.dispatched }

    /// Advance the dispatched-cursor. Clamped to `total_pushed`.
    pub fn set_dispatched(&mut self, idx: usize) {
        self.dispatched = idx.min(self.total_pushed);
    }

    /// Look up an event by its monotonic push index. Returns None if the event
    /// has been evicted or if the index is out of range.
    pub fn get_pushed(&self, idx: usize) -> Option<E> {
        let first = self.total_pushed.saturating_sub(self.entries.len());
        if idx < first { return None; }
        let local = idx - first;
        self.entries.get(local).map(|e| e.event.clone())
    }

    /// Look up the parent event id for a given event id, if any.
    pub fn cause_of(&self, id: EventId) -> Option<EventId> {
        self.entries.iter().find(|e| e.id == id).and_then(|e| e.cause)
    }

    pub fn iter(&self) -> impl Iterator<Item = &E> { self.entries.iter().map(|e| &e.event) }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    // ---- Snapshot helpers (#[doc(hidden)]) ----
    //
    // The `snapshot::format` module serialises only the ring's monotonic
    // metadata (cap + current_tick + next_seq + total_pushed + dispatched).
    // Entry contents are intentionally NOT snapshotted in v1; see the
    // `# Coverage gaps` section in `snapshot/format.rs`.

    #[doc(hidden)]
    pub fn cap_for_snapshot(&self) -> usize { self.cap }
    #[doc(hidden)]
    pub fn current_tick_for_snapshot(&self) -> u32 { self.current_tick }
    #[doc(hidden)]
    pub fn next_seq_for_snapshot(&self) -> u32 { self.next_seq }

    /// Snapshot restore entry point. Clears any existing entries and
    /// restores the monotonic cursors. The caller must create the ring
    /// with `EventRing::with_cap` at the desired capacity beforehand.
    #[doc(hidden)]
    pub fn restore_cursors_from_parts(
        &mut self,
        current_tick: u32,
        next_seq: u32,
        total_pushed: usize,
        dispatched: usize,
    ) {
        self.entries.clear();
        self.current_tick = current_tick;
        self.next_seq = next_seq;
        self.total_pushed = total_pushed;
        self.dispatched = dispatched.min(total_pushed);
    }

    /// Stable hash over the replayable subset. Uses the `EventLike::hash_replayable`
    /// method so the digest is stable across Rust versions. Schema-hash-load-bearing:
    /// changing the hash format or variant-tag order requires a schema-hash bump.
    ///
    /// Sidecar fields (`id`, `cause`) are INTENTIONALLY excluded from the hash
    /// so cascade fan-out annotations cannot alter replay equivalence.
    pub fn replayable_sha256(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for entry in self.entries.iter().filter(|e| e.event.is_replayable()) {
            entry.event.hash_replayable(&mut h);
        }
        h.finalize().into()
    }
}
