//! GpuBackend composite structure. The public `GpuBackend` type
//! delegates to `sync`, `resident`, and `snapshot` sub-contexts —
//! see `docs/superpowers/specs/2026-04-22-gpu-sim-state-design.md`
//! for rationale.

#![cfg(feature = "gpu")]

pub mod resident_ctx;
pub mod snapshot_ctx;
pub mod sync_ctx;

pub use resident_ctx::ResidentPathContext;
pub use snapshot_ctx::SnapshotContext;
pub use sync_ctx::SyncPathContext;
