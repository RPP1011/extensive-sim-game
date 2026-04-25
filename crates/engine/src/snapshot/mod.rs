//! Snapshot save/load for deterministic restart.
//!
//! See `docs/engine/spec.md` §16 for the contract:
//! - Backend-agnostic: always operates on the host-mirror.
//! - Schema-hash versioned: load rejects mismatches unless a
//!   one-step migration is registered.
//! - Hand-rolled binary (little-endian); no new crate deps.

pub mod format;
pub mod migrate;

pub use format::{
    load_from_bytes, load_snapshot, save_snapshot, SnapshotError, SnapshotHeader,
    FORMAT_VERSION, HEADER_BYTES, MAGIC,
};
pub use migrate::{load_snapshot_with_migrations, MigrationFn, MigrationRegistry};
