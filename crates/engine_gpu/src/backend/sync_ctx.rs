//! Sync-path state on the GPU backend.
//!
//! Post-T16 (commit `4474566c`) the hand-written kernel modules
//! (`mask`, `scoring`, `physics`, `apply_actions`, `movement`,
//! `spatial_gpu`, `alive_bitmap`, `cascade`) were retired in favor of
//! the SCHEDULE-driven dispatcher in `engine_gpu_rules`. The previous
//! sync-path orchestration (mask + scoring + cascade + apply + movement)
//! went with them; what remains here is the minimum per-tick state
//! `GpuBackend` still surfaces:
//!
//! * `view_storage` — Phase 4 fold-kernel storage. Still allocated up
//!   front; the SCHEDULE-loop's `Fold<View>` arms write into it.
//! * `backend_label` — adapter backend name captured at init.
//! * `last_phase_us` — per-phase µs surface kept for harness
//!   compatibility; populated only by `step_batch` paths that opt in.
//!
//! `ComputeBackend::step` no longer dispatches GPU kernels — see the
//! method body in `lib.rs` for the CPU-fallback rationale.

#![cfg(feature = "gpu")]

use crate::view_storage::ViewStorage;
use crate::PhaseTimings;

pub struct SyncPathContext {
    pub view_storage:  ViewStorage,
    pub backend_label: String,
    pub last_phase_us: PhaseTimings,
}

impl SyncPathContext {
    pub fn new(view_storage: ViewStorage, backend_label: String) -> Self {
        Self {
            view_storage,
            backend_label,
            last_phase_us: PhaseTimings::default(),
        }
    }
}
