//! Observation packer — builds `[agents × feature_dim]` `f32` row-major
//! tensors for policy input (ML training / inference).
//!
//! Feature sources implement [`FeatureSource`] and compose via
//! [`ObsPacker::register`]. See `docs/engine/spec.md` §21.
//!
//! Serial-first: packing runs on the CPU against `SimState`'s host mirror.
//! A GPU kernel variant lands in Plan 7+.

pub mod packer;
pub mod sources;

pub use packer::{FeatureSource, ObsPacker};
pub use sources::{NeighborSource, PositionSource, VitalsSource};
