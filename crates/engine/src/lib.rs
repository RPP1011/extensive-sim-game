//! World-sim engine — runtime primitives the DSL compiler targets.
//! See `docs/dsl/spec.md` for the authoritative language reference.

pub mod channel;
pub mod creature;
pub mod event;
pub mod ids;
pub mod mask;
pub mod policy;
pub mod rng;
pub mod spatial;
pub mod state;
pub mod step;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
