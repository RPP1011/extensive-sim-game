//! Snapshot staging — double-buffered read-back state for `snapshot()`.
//!
//! Fields land in Task 1.4.

#![cfg(feature = "gpu")]

pub struct SnapshotContext {
    // Fields land in Task 1.4.
}

impl SnapshotContext {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SnapshotContext {
    fn default() -> Self {
        Self::new()
    }
}
