//! Resident-path (batch) state — buffers and kernels used by
//! `step_batch()` for GPU-resident execution.
//!
//! Fields land in Task 1.3.

#![cfg(feature = "gpu")]

pub struct ResidentPathContext {
    // Fields land in Task 1.3.
}

impl ResidentPathContext {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ResidentPathContext {
    fn default() -> Self {
        Self::new()
    }
}
