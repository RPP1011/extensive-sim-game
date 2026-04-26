//! Compiler-emitted rules/data for the World Sim engine.
//!
//! This crate owns the generated DSL artefacts plus the minimal shared
//! support modules those artefacts compile against. Keeping them in a
//! dedicated crate narrows rebuild scope when `compile-dsl` rewrites the
//! emitted files.

pub mod config;
pub mod entities;
pub mod enums;
pub mod events;
pub mod id_serde;
pub mod ids;
pub mod schema;
pub mod scoring;
pub mod types;

pub use events::Event;
pub use schema::EVENT_HASH;
