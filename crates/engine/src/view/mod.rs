//! Materialized views — derived per-entity state that is populated by folding
//! events, not written directly by the simulation kernel. See spec §2 on view
//! compilation modes.

pub mod materialized;

pub use materialized::MaterializedView;
