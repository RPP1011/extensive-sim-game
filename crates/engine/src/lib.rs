//! World-sim engine — runtime primitives the DSL compiler targets.
//! See `docs/dsl/spec.md` for the authoritative language reference.

pub mod ability;
pub mod aggregate;
pub mod backend;
pub mod cascade;
pub mod channel;
pub mod chronicle;
pub mod creature;
pub mod engagement;
pub mod event;
/// Compiler-emitted modules (DSL → Rust). Files under `generated/`
/// are owned by `dsl_compiler`; regenerate with
/// `cargo run --bin xtask -- compile-dsl`. Do not hand-edit.
pub mod generated;
pub mod ids;
pub mod invariant;
pub mod mask;
pub mod obs;
pub mod policy;
pub mod pool;
pub mod rng;
pub mod schema_hash;
pub mod snapshot;
pub mod spatial;
pub mod state;
pub mod step;
pub mod terrain;
pub mod telemetry;
pub mod trajectory;
pub mod view;

pub use backend::{CpuBackend, SimBackend};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
