//! Generic slot pool with freelist reuse.
//!
//! `Pool<T>` hands out [`PoolId<T>`] handles (1-indexed, niche-optimised via
//! `NonZeroU32`). Killed slots are pushed to a freelist and popped on the next
//! `alloc()`, so agent/item/etc. ids get reused once their occupant dies.
//!
//! The `T` tag parameter is phantom — it prevents accidentally passing a
//! `PoolId<AgentTag>` where a `PoolId<ItemTag>` is expected.

pub mod bounded_map;
pub use bounded_map::BoundedMap;

#[allow(unused_imports)]
use contracts::{ensures, invariant, requires};
use std::marker::PhantomData;
use std::num::NonZeroU32;

pub struct PoolId<T> {
    raw:  NonZeroU32,
    _tag: PhantomData<fn() -> T>,
}

impl<T> PoolId<T> {
    pub fn new(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(|raw| Self { raw, _tag: PhantomData })
    }
    pub fn raw(&self) -> u32 {
        self.raw.get()
    }
    #[inline]
    pub fn slot(&self) -> usize {
        (self.raw.get() - 1) as usize
    }
}

impl<T> Clone for PoolId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for PoolId<T> {}
impl<T> PartialEq for PoolId<T> {
    fn eq(&self, o: &Self) -> bool {
        self.raw == o.raw
    }
}
impl<T> Eq for PoolId<T> {}
impl<T> std::hash::Hash for PoolId<T> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.raw.hash(h)
    }
}
impl<T> std::fmt::Debug for PoolId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PoolId({})", self.raw.get())
    }
}

pub struct Pool<T> {
    cap:       u32,
    next_raw:  u32,
    pub alive: Vec<bool>,
    freelist:  Vec<u32>,
    _tag:      PhantomData<fn() -> T>,
}

#[invariant(self.freelist.iter().all(|r| {
    let slot = (*r - 1) as usize;
    slot < self.cap as usize && !self.alive[slot]
}))]
#[invariant({
    let mut sorted = self.freelist.clone();
    sorted.sort_unstable();
    sorted.windows(2).all(|w| w[0] != w[1])
})]
#[invariant(self.next_raw <= self.cap + 1)]
impl<T> Pool<T> {
    pub fn new(cap: u32) -> Self {
        Self {
            cap,
            next_raw: 1,
            alive: vec![false; cap as usize],
            freelist: Vec::new(),
            _tag: PhantomData,
        }
    }

    #[ensures(ret.is_some() -> self.alive[(ret.as_ref().unwrap().raw() - 1) as usize])]
    #[ensures(ret.is_none() -> self.next_raw == old(self.next_raw) && self.freelist.is_empty())]
    pub fn alloc(&mut self) -> Option<PoolId<T>> {
        let raw = if let Some(r) = self.freelist.pop() {
            r
        } else if self.next_raw <= self.cap {
            let r = self.next_raw;
            self.next_raw += 1;
            r
        } else {
            return None;
        };
        self.alive[(raw - 1) as usize] = true;
        PoolId::new(raw)
    }

    #[requires((id.slot()) < self.cap as usize)]
    #[ensures(!self.alive[id.slot()])]
    pub fn kill(&mut self, id: PoolId<T>) {
        let slot = id.slot();
        if slot < self.cap as usize && self.alive[slot] {
            self.alive[slot] = false;
            self.freelist.push(id.raw());
        }
    }
    #[inline]
    pub fn is_alive(&self, id: PoolId<T>) -> bool {
        self.alive.get(id.slot()).copied().unwrap_or(false)
    }
    #[inline]
    pub fn cap(&self) -> u32 {
        self.cap
    }
    #[inline]
    pub fn slot_of(id: PoolId<T>) -> usize {
        id.slot()
    }

    /// Number of freed slots currently on the freelist. Exposed for property
    /// tests that assert `count_alive + freelist_len = next_raw - 1`.
    pub fn freelist_len(&self) -> usize { self.freelist.len() }

    /// The next allocation counter. Equals 1 + (largest raw ever handed out).
    /// Exposed for the `count_alive + freelist_len = next_raw - 1` identity.
    pub fn next_raw(&self) -> u32 { self.next_raw }

    /// Iterator over the freelist's raw values in insertion order. Exposed for
    /// property tests that assert freelist uniqueness.
    pub fn freelist_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.freelist.iter().copied()
    }
}

/// Contract-free impl block — the methods here must be callable on a
/// deliberately-corrupted pool (fault-injection tests). The struct-level
/// `#[invariant]` attributes on the primary impl above would otherwise fire
/// on entry/exit of every method, short-circuiting the corruption-detection
/// paths that tests in `invariant_pool_non_overlap.rs` depend on.
impl<T> Pool<T> {
    /// Verify pool consistency: no slot appears as both alive AND in the
    /// freelist, and the freelist contains no duplicate slots. Returns
    /// `true` when consistent. This is the predicate that
    /// `PoolNonOverlapInvariant` checks per-tick. Deliberately NOT decorated
    /// with `#[invariant]` so callers with a corrupted pool can observe the
    /// false return instead of a pre-condition panic.
    pub fn is_non_overlapping(&self) -> bool {
        // (1) No duplicate entries in the freelist.
        let mut seen = vec![false; self.cap as usize];
        for &raw in &self.freelist {
            let slot = (raw - 1) as usize;
            if slot >= self.cap as usize {
                return false;
            }
            if seen[slot] {
                return false; // duplicate
            }
            seen[slot] = true;
        }
        // (2) Any slot marked alive must NOT also be in the freelist.
        for (slot, &alive) in self.alive.iter().enumerate() {
            if alive && seen[slot] {
                return false;
            }
        }
        true
    }

    /// Test-only fault injection: force a slot to appear in the freelist
    /// without going through `kill`. Used to construct corrupted pool states
    /// for the `PoolNonOverlapInvariant` tests — proves the invariant check
    /// fires. Production code must never call this.
    #[doc(hidden)]
    pub fn force_push_freelist_for_test(&mut self, raw: u32) {
        self.freelist.push(raw);
    }

    /// Snapshot-restore entry point. Overwrites internal state with the
    /// `(next_raw, alive, freelist)` triple read back from a snapshot file.
    /// The caller is responsible for ensuring `alive.len() == self.cap` and
    /// that the (alive, freelist) pair is consistent — `is_non_overlapping`
    /// is invoked by the load path before returning to surface corruption
    /// from a tampered snapshot.
    pub fn restore_from_parts(
        &mut self,
        next_raw: u32,
        alive: Vec<bool>,
        freelist: Vec<u32>,
    ) {
        debug_assert_eq!(alive.len(), self.cap as usize, "pool restore: alive slice length mismatch");
        self.next_raw = next_raw;
        self.alive = alive;
        self.freelist = freelist;
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    struct AgentTag;

    #[test]
    fn alloc_post_is_alive() {
        let mut p: Pool<AgentTag> = Pool::new(4);
        let id = p.alloc().unwrap();
        assert!(p.is_alive(id));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pre")]
    fn kill_with_out_of_range_slot_panics_debug() {
        let mut p: Pool<AgentTag> = Pool::new(2);
        // id.slot() == 99 is out of range for cap=2.
        let bad = PoolId::<AgentTag>::new(100).unwrap();
        p.kill(bad); // contracts::requires triggers panic in debug.
    }
}
