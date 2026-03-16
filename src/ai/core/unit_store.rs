use std::ops::{Index, IndexMut};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::types::UnitState;

/// Facade over `Vec<UnitState>` with a stable API for id→index lookup.
///
/// Externally, code continues to index by `usize` (`store[idx].hp`).
/// `idx_of` uses linear scan (fastest at typical 4-16 unit counts).
#[derive(Debug, Clone)]
pub struct UnitStore {
    units: Vec<UnitState>,
}

impl UnitStore {
    pub fn new(units: Vec<UnitState>) -> Self {
        Self { units }
    }

    /// Lookup: unit id → index. Linear scan, optimal for ≤64 units.
    #[inline]
    pub fn idx_of(&self, id: u32) -> Option<usize> {
        self.units.iter().position(|u| u.id == id)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, UnitState> {
        self.units.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, UnitState> {
        self.units.iter_mut()
    }

    pub fn push(&mut self, unit: UnitState) {
        self.units.push(unit);
    }

    pub fn retain<F: FnMut(&UnitState) -> bool>(&mut self, f: F) {
        self.units.retain(f);
    }

    pub fn sort_by_key<K: Ord, F: FnMut(&UnitState) -> K>(&mut self, f: F) {
        self.units.sort_by_key(f);
    }

    pub fn extend<I: IntoIterator<Item = UnitState>>(&mut self, iter: I) {
        self.units.extend(iter);
    }

    pub fn into_inner(self) -> Vec<UnitState> {
        self.units
    }

    #[inline]
    pub fn as_slice(&self) -> &[UnitState] {
        &self.units
    }
}

// --- Index / IndexMut so `store[idx].hp` compiles unchanged ---

impl Index<usize> for UnitStore {
    type Output = UnitState;
    #[inline]
    fn index(&self, idx: usize) -> &UnitState {
        &self.units[idx]
    }
}

impl IndexMut<usize> for UnitStore {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut UnitState {
        &mut self.units[idx]
    }
}

// --- IntoIterator for `for unit in &store` / `for unit in &mut store` ---

impl<'a> IntoIterator for &'a UnitStore {
    type Item = &'a UnitState;
    type IntoIter = std::slice::Iter<'a, UnitState>;
    fn into_iter(self) -> Self::IntoIter {
        self.units.iter()
    }
}

impl<'a> IntoIterator for &'a mut UnitStore {
    type Item = &'a mut UnitState;
    type IntoIter = std::slice::IterMut<'a, UnitState>;
    fn into_iter(self) -> Self::IntoIter {
        self.units.iter_mut()
    }
}

// --- Serde: serialize as Vec<UnitState> for JSON compat ---

impl Serialize for UnitStore {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.units.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UnitStore {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let units = Vec::<UnitState>::deserialize(deserializer)?;
        Ok(UnitStore::new(units))
    }
}
