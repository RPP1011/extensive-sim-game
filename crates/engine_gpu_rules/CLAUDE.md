# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate is

An empty placeholder, kept alive for one reason: `crates/engine/src/schema_hash.rs` pulls in `crates/engine_gpu_rules/.schema_hash` via `include_str!` as part of the engine's schema-hash computation. Removing this crate breaks that build-time include.

`src/lib.rs` is a stub — no types, no logic. It used to hold DSL-compiler-generated WGSL/Rust kernel modules (`pub mod` / `pub use` per compiled fixture), emitted by the retired `xtask compile-dsl --cg-canonical` command. The Phase 7 wolf-sim wipe (2026-05-02) deleted the wolf-sim DSL inputs those kernels were generated from, so the generated content went with them; the crate itself stayed only for the schema-hash include.

Per-fixture GPU kernel output today is emitted per-runtime into each crate's own `OUT_DIR` (see `crates/sims/build.rs` and the legacy per-fixture crates), not into this crate. There is currently no live `compile-dsl --cg-canonical` command (xtask itself was retired in the same wipe) — if that regeneration path is ever restored, it would write back into `src/lib.rs` here.

## Commands

`cargo build -p engine_gpu_rules` / `cargo test -p engine_gpu_rules` both succeed trivially (empty crate, no tests). There's a `gpu` feature flag declared but currently unused by any code in this crate.

## Working here

Don't add hand-written code to this crate — per the repo's compiler-first ground rule (`docs/game/compiler_progress.md`), anything here would need to arrive as compiler output, and the compiler path that used to target this crate is retired. If you need this crate to do something again, the real work is restoring/rebuilding the `compile-dsl --cg-canonical` emission path, not hand-editing `lib.rs`.
