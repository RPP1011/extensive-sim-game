//! Causal tree presentation over EventRing.
//!
//! `EventRing<E>` already records `cause: Option<EventId>` per entry.
//! `CausalTree` is the read-only presentation layer that turns those flat
//! cause-pointers into a tree rooted at root-cause events (events with
//! no cause).

use crate::event::{EventLike, EventRing};
use crate::ids::EventId;
use std::collections::HashMap;

/// A read-only causal-tree view over an [`EventRing`].
///
/// Built once on construction via [`CausalTree::build`]; subsequent mutations
/// to the ring are NOT reflected. Re-build to refresh.
pub struct CausalTree<'a, E: EventLike> {
    ring:     &'a EventRing<E>,
    children: HashMap<EventId, Vec<EventId>>, // cause → [effect, ...]
    roots:    Vec<EventId>,                   // events with cause == None
}

impl<'a, E: EventLike> CausalTree<'a, E> {
    /// Build the causal index over the ring's current retained entries.
    pub fn build(ring: &'a EventRing<E>) -> Self {
        let mut children: HashMap<EventId, Vec<EventId>> = HashMap::new();
        let mut roots = Vec::new();
        for entry in ring.iter_with_meta() {
            match entry.cause {
                Some(c) => children.entry(c).or_default().push(entry.id),
                None    => roots.push(entry.id),
            }
        }
        Self { ring, children, roots }
    }

    /// Root events: events with no recorded cause.
    pub fn roots(&self) -> &[EventId] { &self.roots }

    /// Direct children of `id` in push order. Returns an empty slice if `id`
    /// has no children or has been evicted.
    pub fn children_of(&self, id: EventId) -> &[EventId] {
        self.children.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Borrow the event payload for `id`. Returns `None` if evicted.
    pub fn event(&self, id: EventId) -> Option<&E> {
        self.ring.get_by_id(id).map(|e| &e.event)
    }
}
