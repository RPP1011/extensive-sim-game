//! Compatibility shim for the World Sim rules surface.
//!
//! The generated DSL outputs now live in `engine_data` so regenerating
//! them does not force this crate to also own the emitted source tree. This
//! crate preserves the long-lived `engine_rules::*` import surface for the
//! rest of the workspace.

pub use engine_data::*;
