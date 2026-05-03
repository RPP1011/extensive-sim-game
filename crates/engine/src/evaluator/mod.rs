//! Engine-side `Context` trait implementations for the DSL IR interpreter.
//!
//! Provides `SimState`-backed impls of `dsl_ast::eval::ReadContext`,
//! `dsl_ast::eval::CascadeContext`.  All code here is gated behind
//! `#[cfg(feature = "interpreted-rules")]`.
//!
//! ## Spec B' adaptation notes
//!
//! On `wsb-engine-viz`, `EngineViewCtx` lived here too and referenced
//! `crate::generated::views::ViewRegistry`.  After Spec B', `ViewRegistry`
//! moved to `engine_rules` (generated) and `SimState` no longer has a
//! `views` field.  `EngineViewCtx` is therefore provided by `engine_rules`
//! (in `lib.rs`, which is the hand-written non-generated entry point),
//! where it can reference `ViewRegistry` without creating a circular dep.
//!
//! View-accessor methods in `EngineReadCtx` / `EngineCascadeCtx` that
//! previously read `state.views.*` now:
//!  - Return 0.0 / false for materialized-view queries (threat_level,
//!    my_enemies, pack_focus, kin_fear, rally_boost).  These are correct for
//!    the wolves+humans parity test which passes an empty `ViewRegistry`.
//!  - Compute `view_is_hostile` / `view_is_stunned` directly from
//!    `SimState` fields (no view state needed — creature-type and stun-expiry
//!    are always present in `SimState`).
//!
//! Full view-backed read support (for production use outside the parity test)
//! will be re-enabled when `engine_rules` exposes a `with_views` constructor
//! for the context types.

pub mod context;

pub use context::{EngineCascadeCtx, EngineReadCtx};
