//! `tom_probe_runtime` build script. Collapses to the shared
//! `dsl_compiler::build_helper::emit` shim now that the helper
//! auto-detects `LowerOpts.belief_state` from the .sim's IR.
//!
//! Previously this build.rs hand-flipped `belief_state: true` on the
//! `LowerOpts` it passed to `lower_compilation_to_cg_with_opts`. The
//! 2026-05-11 belief-state auto-detect (mirrors `aoe_dispatch`) walks
//! every physics rule body for `agents.set_beliefs_<field>(...)` calls
//! and sets the flag automatically — so this fixture's build script
//! is now structurally identical to the standard sweep.
//!
//! The cycle diagnostic surfaced by `lower_compilation_to_cg_with_opts`
//! (rule order between WhatIBelieve producer + the consumers) is
//! tolerated by the helper the same way the old hand-rolled build.rs
//! tolerated it — emitted as a `cargo:warning` with the partial
//! program continuing to emit kernels.

fn main() {
    dsl_compiler::build_helper::emit("tom_probe");
}
