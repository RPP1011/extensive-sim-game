//! World-sim engine — runtime primitives the DSL compiler targets.
//! See `docs/dsl/spec.md` for the authoritative language reference.

pub mod ability;
pub mod aggregate;
pub mod backend;
pub mod cascade;
/// DSL IR interpreter context implementations. Gated on `interpreted-rules`
/// feature — compiles to nothing in the default build.
#[cfg(feature = "interpreted-rules")]
pub mod evaluator;
pub mod debug;
pub mod channel;
pub mod creature;
pub mod event;
pub mod ids;
pub mod invariant;
pub mod mask;
pub mod obs;
pub mod policy;
pub mod pool;
pub mod probe;
pub mod rng;
pub mod schema_hash;
/// Per-tick scratch buffers (MaskBuffer, TargetMask, actions, shuffle_idx).
/// Moved here from `step.rs` (deleted, Plan B1' Task 11) so the type
/// survives as a storage primitive. Rule-aware tick logic lives in
/// `engine_rules::step` once Task 11 lands.
pub mod scratch;
/// `CompiledSim` trait — the uniform interface per-fixture runtime crates
/// expose to the generic application layer. See module doc for the contract.
pub mod sim_trait;
/// Compile-only unimplemented!() stubs for `step`, `step_full`, etc. so the
/// many `#[ignore]`d tests that still import `engine::step::*` compile cleanly.
/// Remove this module when Task 11 lands and test imports migrate to
/// `engine_rules::step`.
pub mod step;
pub mod snapshot;
pub mod spatial;
pub mod state;
pub mod terrain;
pub mod telemetry;
pub mod trajectory;
pub mod view;

pub use backend::ComputeBackend;
pub use sim_trait::CompiledSim;
/// Re-export SimScratch from its new home so call sites that previously
/// wrote `engine::step::SimScratch` can be updated to `engine::scratch::SimScratch`.
pub use scratch::SimScratch;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
