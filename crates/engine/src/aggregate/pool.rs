use crate::pool::{Pool, PoolId};

/// Pool of non-spatial entities (quests, groups, treaties, …). Parallel to
/// the agent pool but decoupled from spatial indexing. Determinism rules
/// (freelist reuse, slot-based IDs) are the same.
pub struct AggregatePool<T> {
    pool:  Pool<T>,
    slots: Vec<Option<T>>,
}

impl<T> AggregatePool<T> {
    pub fn new(cap: u32) -> Self {
        Self {
            pool:  Pool::new(cap),
            slots: (0..cap).map(|_| None).collect(),
        }
    }

    pub fn alloc(&mut self, value: T) -> Option<PoolId<T>> {
        let id = self.pool.alloc()?;
        self.slots[id.slot()] = Some(value);
        Some(id)
    }

    pub fn kill(&mut self, id: PoolId<T>) {
        self.pool.kill(id);
        if let Some(s) = self.slots.get_mut(id.slot()) {
            *s = None;
        }
    }

    pub fn get(&self, id: PoolId<T>) -> Option<&T> {
        self.slots.get(id.slot()).and_then(|s| s.as_ref())
    }

    pub fn get_mut(&mut self, id: PoolId<T>) -> Option<&mut T> {
        self.slots.get_mut(id.slot()).and_then(|s| s.as_mut())
    }

    pub fn is_alive(&self, id: PoolId<T>) -> bool {
        self.pool.is_alive(id)
    }

    pub fn cap(&self) -> u32 {
        self.pool.cap()
    }
}
