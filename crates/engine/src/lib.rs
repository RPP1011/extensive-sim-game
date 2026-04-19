//! World-sim engine — runtime primitives the DSL compiler targets.
//! See `docs/dsl/spec.md` for the authoritative language reference.

pub mod channel;
pub mod creature;
pub mod ids;
pub mod rng;
pub mod state;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
