//! Sync-path state on the GPU backend — kernels and buffers used
//! exclusively by `SimBackend::step()`.
//!
//! Fields move here from the flat `GpuBackend` struct in Task 1.2.
//! This skeleton exists so the `mod backend;` wiring lands first.

#![cfg(feature = "gpu")]

pub struct SyncPathContext {
    // Fields land in Task 1.2.
}

impl SyncPathContext {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SyncPathContext {
    fn default() -> Self {
        Self::new()
    }
}
