//! Snapshot staging — double-buffered read-back state for
//! `GpuBackend::snapshot()`.

#![cfg(feature = "gpu")]

use crate::snapshot::GpuStaging;

pub struct SnapshotContext {
    /// Phase D — double-buffered snapshot staging. `front` is the one
    /// that will be read next (filled by the previous `snapshot()`
    /// call); `back` is the one currently filling for the NEXT call.
    /// Both `None` until first `snapshot()` call lazy-inits them.
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    pub snapshot_front:               Option<GpuStaging>,
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    pub snapshot_back:                Option<GpuStaging>,

    /// Phase D — watermark tracking what portion of the main event /
    /// chronicle rings have been snapshotted. Advances monotonically.
    #[allow(dead_code)] // TODO Phase D Task D3: consumed by GpuBackend::snapshot()
    pub snapshot_event_ring_read:     u64,
    pub snapshot_chronicle_ring_read: u64,

    /// Phase D — the most recent tick recorded by `step_batch`.
    /// Exposed via snapshot for the observer.
    pub latest_recorded_tick:         u32,
}

impl SnapshotContext {
    pub fn new() -> Self {
        Self {
            snapshot_front:               None,
            snapshot_back:                None,
            snapshot_event_ring_read:     0,
            snapshot_chronicle_ring_read: 0,
            latest_recorded_tick:         0,
        }
    }
}

impl Default for SnapshotContext {
    fn default() -> Self {
        Self::new()
    }
}
