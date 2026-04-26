//! Bounded fixed-capacity associative array.
//!
//! Linear-probe small-cap map. Used for cold-path SoA fields where
//! per-agent state may include a small dynamic set (e.g., per-target
//! beliefs, per-faction relationships). LRU eviction by an embedded
//! "last-touched tick" — caller passes the current tick on upsert.

#[derive(Clone, Debug)]
pub struct BoundedMap<K: Eq + Copy, V, const N: usize> {
    entries: Vec<(K, V)>,
}

impl<K: Eq + Copy, V, const N: usize> BoundedMap<K, V, N> {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.entries.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.entries.iter_mut().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }

    /// Insert or update. If at capacity and key is new, evicts the
    /// LRU-by-position entry (entries are kept in insertion order;
    /// callers using a "last-touched" field can shift hot entries to the
    /// back via `touch`).
    pub fn upsert(&mut self, k: K, v: V) {
        if let Some(slot) = self.entries.iter_mut().find(|(kk, _)| *kk == k) {
            slot.1 = v;
            return;
        }
        if self.entries.len() == N {
            self.entries.remove(0);  // LRU = oldest by insertion
        }
        self.entries.push((k, v));
    }

    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        self.entries.retain_mut(|(k, v)| f(k, v));
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

impl<K: Eq + Copy, V, const N: usize> Default for BoundedMap<K, V, N> {
    fn default() -> Self { Self::new() }
}
