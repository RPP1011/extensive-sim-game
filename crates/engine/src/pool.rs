//! Generic slot pool with freelist reuse.
//!
//! `Pool<T>` hands out [`PoolId<T>`] handles (1-indexed, niche-optimised via
//! `NonZeroU32`). Killed slots are pushed to a freelist and popped on the next
//! `alloc()`, so agent/item/etc. ids get reused once their occupant dies.
//!
//! The `T` tag parameter is phantom — it prevents accidentally passing a
//! `PoolId<AgentTag>` where a `PoolId<ItemTag>` is expected.

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
}
