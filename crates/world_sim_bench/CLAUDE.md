# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What it does

A Criterion benchmark harness for world-sim hot loops. Currently the only bench
(`benches/movement.rs`) loads a committed bincode `WorldState` fixture and
compares `ApplyMovementSystem`'s `Scalar` vs `Simd` `Backend` on
`apply_inplace`. `src/fixtures.rs` is the fixture loader
(`fixtures::load(scale)` / `fixtures::exists(scale)`), reading
`fixtures/world_<scale>.bin` relative to the crate root
(`CARGO_MANIFEST_DIR`, not the bench runner's CWD). `src/lib.rs` just
re-exports the `fixtures` module and gates a `#[feature(portable_simd)]`
attribute behind a `nightly` cfg.

## Status: currently broken — cannot resolve

This crate depends on `game = { path = "../.." }` (root `Cargo.toml`), but the
workspace root is a **virtual manifest** (no `[package]`) — the root `game`
package it points at was deleted in the Phase 7 wolf-sim wipe (2026-05-02;
see root `Cargo.toml` comment: "src/ ... stayed gone"). `cargo check` here
fails immediately with:

```
error: failed to get `game` as a dependency of package `world_sim_bench`
Caused by: found a virtual manifest at .../Cargo.toml instead of a package manifest
```

Additionally, no `fixtures/` directory exists in this crate — the bench's own
doc comment says fixtures are "regenerated via `xtask world-sim ... --output
<path>`", but `xtask` was also retired in the same Phase 7 wipe (per root
CLAUDE.md). So even if the `game` dependency were fixed, there is currently
no supported way to (re)generate the `.bin` fixtures the bench needs — the
bench prints a `skip: no fixture for scale=2k` and no-ops for any missing
scale rather than failing.

This crate is an orphaned pre-wipe artifact left in the workspace `exclude`
list. Treat "fix `world_sim_bench`" as: (1) point `game` at whatever crate
now owns `world_sim::{state::WorldState, apply::ApplyMovementSystem,
delta::MergedDeltas, system::Backend}` (likely `engine` post-consolidation —
verify the current module layout before assuming), and (2) find or rebuild a
fixture generator, before any bench will run.

## Commands (once/if the dependency is fixed)

It is excluded from the workspace (`exclude` in root `Cargo.toml`), so it is
**not** reachable via `-p world_sim_bench` from the workspace root (`cargo
check -p world_sim_bench` from root errors with "did not match any
packages"). You must `cd crates/world_sim_bench` and run cargo from inside
the crate directory — `--manifest-path` from outside still fails the same
way, since cargo treats an excluded crate as outside any workspace for
dependency resolution.

```bash
cd crates/world_sim_bench
cargo check
cargo bench                    # runs benches/movement.rs via criterion
```

Note `rust-toolchain.toml` in this crate pins **nightly** (for
`portable_simd`), overriding whatever toolchain the rest of the workspace
uses — running `cargo` from inside this directory will auto-install/use a
nightly toolchain.

## Dependencies (Cargo.toml)

- `game` (path `../..`, workspace root) — broken, see above.
- `bincode = "1.3"` — fixture (de)serialization.
- `anyhow = "1"` — error handling in the fixture loader.
- `criterion = "0.5"` (dev-dependency, `html_reports`) — bench harness,
  `harness = false` on the `movement` bench (criterion drives its own main).

No dependency on `engine`, `sims`, `dsl_compiler`, `ability_operator`, or
`ability-vae` — despite the name, it references world-sim types only through
the now-broken `game` path dependency, not through any current workspace
crate.
