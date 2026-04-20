//! Ability subsystem — programs, registry, cast dispatch, and the unified
//! tick-start phase that decrements combat-timing fields and recomputes
//! engagement bindings.
//!
//! Scope for this commit (Combat Foundation Task 3): only `expire` is
//! populated. `id`, `program`, `registry`, `cast`, `gate` scaffolds land in
//! Task 6+ when the AbilityProgram IR is introduced.

pub mod expire;
