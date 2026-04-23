//! Shared GPU helpers used by both sync and resident kernel drivers.
//!
//! Factored out of duplicated patterns across physics.rs, mask.rs,
//! spatial_gpu.rs, etc. Each helper is a thin wrapper around a wgpu
//! idiom — kept here so the batch-path drivers in `cascade_resident`
//! and `snapshot` don't reintroduce the duplication.

pub mod readback;
