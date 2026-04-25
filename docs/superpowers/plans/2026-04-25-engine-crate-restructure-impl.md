# Engine Crate Restructure (Plan B1, SUPERSEDED) — Implementation Plan

> **Status:** SUPERSEDED 2026-04-25 by `2026-04-25-engine-crate-split-impl-v2.md` (Plan B1').
>
> Spec B (the design this plan implemented) was rewritten as Spec B' after
> brainstorming surfaced two structural errors: dep direction was inverted
> (B' has `engine ← engine_data ← engine_rules`), and `step_full` was wrongly
> kept in engine (B' emits it from the DSL compiler so LLVM can specialize).
> Tasks 1–2 of this v1 plan landed (`d4d06390`, `da008ac3`) and remain valid;
> Tasks 3+ are replaced by Plan B1'. This file is preserved for the
> decision history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the structural-impossibility scaffolding from Spec B §3 / §4 / §5 / §6 / §7: rename `engine_generated` → `engine_data`, move `crates/engine/src/generated/{mask,physics,views}` to `crates/engine_rules/src/`, seal `CascadeHandler` + the three view traits behind a private `Sealed` supertrait gated by a `GeneratedRule` marker that only `engine_rules` blanket-impls, add `build.rs` sentinels (engine_rules + engine_data require `// GENERATED` headers; engine has a primitives-only allowlist and rejects `// GENERATED` markers), add a `trybuild` compile-fail test for the seal, add an ast-grep CI rule rejecting `impl CascadeHandler` outside `engine_rules`, add `cargo run --bin xtask -- compile-dsl --check`, and extend `.githooks/pre-commit` to enforce the header rule + run `compile-dsl --check` when DSL source is staged.

**Architecture:** Pure Rust restructure + emitter-target updates + build.rs panics + thin xtask flag + pre-commit bash. No new runtime behaviour. Direction of dependencies after this plan: `engine_data` (data shapes) → `engine` (primitives) → `engine_rules` (emitted rule logic). Each generated crate enforces its own `// GENERATED` header at build time; `engine` enforces a primitives-only allowlist + rejects `// GENERATED`. The seal is belt-and-suspenders: private supertrait + ast-grep CI rule + build sentinel.

**Tech Stack:** Rust 2021, `cargo`, `git mv` for rename history, `trybuild` for compile-fail tests, `ast-grep` for CI rule, bash for pre-commit, `dsl_compiler` Rust emit code, `xtask` clap subcommand.

**Out of scope (deferred to follow-up plans):**
- **chronicle.rs migration to emitted DSL** (Spec B §8.2). Defer to **Plan B2** so this plan can land without DSL grammar extension. Until B2, `engine::chronicle` continues to live in `engine/src/chronicle.rs`; it remains hand-written but is on the `engine/build.rs` allowlist as `chronicle.rs` (an explicit, documented exception). Adding a new chronicle template_id is still a hand edit until B2.
- **engagement.rs `break_reason` constants migration to engine_data** (Spec B §8.3). Same rationale; carried by B2. Until then, `engine::engagement::break_reason` stays put; `engagement.rs` lives on the allowlist.
- **Legacy `src/` sweep + xtask move to `crates/xtask/`** (Spec B §8.4). Carried by **Plan B3**, executable in parallel with B1 in a separate worktree.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `engine_generated` crate at `crates/engine_generated/{Cargo.toml,src/lib.rs}` (data shapes; emit target #1)
  - `engine_rules` shim at `crates/engine_rules/{Cargo.toml,src/lib.rs}` (8-line `pub use engine_generated::*`)
  - `engine/src/generated/{mask,physics,views}` (35 emitted .rs files)
  - `pub trait CascadeHandler` at `crates/engine/src/cascade/handler.rs:91`
  - `pub trait MaterializedView` at `crates/engine/src/view/materialized.rs`
  - `pub trait LazyView: Send + Sync` at `crates/engine/src/view/lazy.rs`
  - `pub trait TopKView: Send + Sync` at `crates/engine/src/view/topk.rs`
  - dsl_compiler emit destinations: hard-coded defaults in `src/bin/xtask/cli/mod.rs` (`out_physics`, `out_mask`, `out_views`, `out_scoring`, `out_entity`, `out_config_rust`, `out_enum`, plus `engine_generated/src/events/`)
  - 32 cross-crate imports of `engine::generated::*` across workspace (counted via `rg "engine::generated::"`)
  - 23 files importing `engine_rules::*` (counted via `rg "use engine_rules::"`)
  - Test/demo trait impls: `crates/engine/src/view/{materialized,lazy,topk}.rs` (`DamageTaken`, `NearestEnemyLazy`, `MostHostileTopK`); `crates/engine/tests/{cascade_bounded,cascade_register_dispatch,cascade_lanes,proptest_cascade_bound}.rs` (test-only `impl CascadeHandler`)
  Search method: `rg` + direct `Read`.

- **Decision:** restructure existing crates (no new crates introduced; `engine_rules` shim is repurposed, `engine_generated` is renamed). Emitter destinations updated to point at the renamed crates; emitted code remains the source of truth for rule logic.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none (chronicle/engagement migration deferred to B2).
  - Generated outputs re-emitted: every file under `engine_rules/src/{mask,physics,views}` and `engine_data/src/{config,entities,enums,events,ids,scoring,types}` after the path rewrites land — full regeneration validates the new emit-paths and the `crate::` → `engine::` import rewrite required for relocated rule files.
  - Emit-path changes in `dsl_compiler`/xtask: physics + mask + views move from `crates/engine/src/generated/{physics,mask,views}` to `crates/engine_rules/src/{physics,mask,views}`; scoring + entity + config + enum + events + schema + ids move from `crates/engine_generated/src/...` to `crates/engine_data/src/...`; cross-references in emitted code rewrite `crate::event::*` / `crate::state::*` / `crate::ids::*` / `crate::mask::*` / `crate::cascade::*` to `engine::event::*` / `engine::state::*` / `engine::ids::*` / `engine::mask::*` / `engine::cascade::*` because the generated files are no longer inside the `engine` crate.

- **Hand-written downstream code:**
  - `crates/engine/build.rs`: NEW — primitives-only allowlist sentinel. Justification: this is the structural rule that makes Approach-2 hand-written behavior modules impossible to compile; it has no DSL representation because it constrains what *isn't* emitted.
  - `crates/engine_rules/build.rs`: NEW — every-file-must-be-generated sentinel. Same justification (structural rule about emitted vs hand-written).
  - `crates/engine_data/build.rs`: NEW — same as engine_rules (every-file-must-be-generated).
  - `crates/engine/tests/sealed_cascade_handler.rs` + `crates/engine/tests/ui/external_impl_rejected.rs`: NEW — `trybuild` compile-fail test asserting the seal works. No DSL representation; tests the seal mechanism itself.
  - `__sealed::Sealed` private supertrait + `GeneratedRule` marker: NEW in `engine/src/cascade/handler.rs` and the three view files. The blanket impl `impl<T: GeneratedRule> Sealed for T` is emitted into `engine_rules/src/lib.rs` by `dsl_compiler`.
  - `xtask compile-dsl --check`: NEW Rust subcommand body (~30 lines). Justification: the verification logic is workflow tooling, not engine behaviour.
  - `.githooks/pre-commit`: extended with two new checks (header rule, regen-on-DSL-change). Justification: the pre-commit hook is workflow tooling; this plan extends an already-landed file.
  - `.ast-grep/rules/no-cascade-handler-impl-outside-engine-rules.yml` + sibling rules for the three view traits: NEW. Justification: enforcement, not behaviour.

- **Constitution check:**
  - P1 (Compiler-First): PASS — net effect is *more* compiler-first (sealing makes the rule structurally enforced; build sentinels make hand-written rule logic structurally impossible).
  - P2 (Schema-Hash on Layout): PASS — no state-layout changes; `crates/engine/.schema_hash` baseline is unchanged after regen (verified by Task 14).
  - P3 (Cross-Backend Parity): PASS — no `step_full` semantic changes; `engine_gpu` import paths update mechanically (Task 5).
  - P4 (`EffectOp` Size Budget): N/A — no `EffectOp` changes.
  - P5 (Determinism via Keyed PCG): N/A — no RNG changes.
  - P6 (Events Are the Mutation Channel): N/A — no event-flow changes.
  - P7 (Replayability Flagged): N/A — no event flag changes.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends in a `cargo test` verification + commit.
  - P10 (No Runtime Panic): PASS — `build.rs` panics fire at *build time*, not runtime; no new `expect`/`unwrap` in runtime paths.
  - P11 (Reduction Determinism): N/A — no aggregator changes.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill). [ ] AIS reviewed post-design (after task list stabilises — tick when last task in Sequencing is complete).

---

## File Structure

```
crates/
  engine_data/            — RENAMED from engine_generated.
                            Contains: config/, entities/, enums/, events/,
                            ids.rs, schema.rs, scoring/, types.rs,
                            id_serde.rs.
                            All files emit-only (build.rs verifies header).
  engine/
    Cargo.toml            — MODIFIED: dep `engine_rules` → `engine_data`.
    build.rs              — NEW: primitives-only allowlist + reject-GENERATED sentinel.
    src/
      generated/          — DELETED (mask, physics, views moved to engine_rules).
      lib.rs              — MODIFIED: drop `pub mod generated;`.
      cascade/handler.rs  — MODIFIED: add `__sealed::Sealed` private supertrait.
      view/{materialized,lazy,topk}.rs
                          — MODIFIED: each adds `__sealed::Sealed` supertrait + cfg(test) GeneratedRule shims.
    tests/
      sealed_cascade_handler.rs    — NEW: trybuild driver.
      ui/external_impl_rejected.rs — NEW: compile-fail fixture.
      ui/external_impl_rejected.stderr — NEW: expected compiler output.
      cascade_bounded.rs / cascade_register_dispatch.rs / cascade_lanes.rs / proptest_cascade_bound.rs
                                   — MODIFIED: add `#[cfg(test)] impl engine_rules::GeneratedRule for ...` for test handlers.
  engine_rules/
    Cargo.toml            — MODIFIED: drop dep on engine_generated; add dep on engine + engine_data.
    build.rs              — NEW: every-file-must-be-generated sentinel.
    src/
      lib.rs              — REPLACED: emitted from dsl_compiler. Declares `GeneratedRule` marker + `impl<T: GeneratedRule> engine::cascade::handler::__sealed::Sealed for T` blanket. Re-exports `mask`, `physics`, `views` modules.
      mask/               — MOVED from engine/src/generated/mask/. Imports rewritten crate:: → engine::.
      physics/            — MOVED from engine/src/generated/physics/. Imports rewritten crate:: → engine::.
      views/              — MOVED from engine/src/generated/views/. Imports rewritten crate:: → engine::.
  engine_data/
    build.rs              — NEW: every-file-must-be-generated sentinel.
    src/...               — same files as old engine_generated/, headers added on regen.

dsl_compiler/
  src/emit_*.rs           — MODIFIED: any file currently emitting `use crate::*` references when target is engine_rules/* now emits `use engine::*`. lib.rs doc strings updated.
  src/lib.rs              — MODIFIED: doc-string path updates.

src/bin/xtask/
  cli/mod.rs              — MODIFIED: out-path defaults updated to new locations. Add `--check` flag to CompileDslArgs.
  compile_dsl_cmd.rs      — MODIFIED: implement `--check` mode (regen to tempdir + diff).

.githooks/
  pre-commit              — MODIFIED: add header-rule + regen-on-DSL-change checks (extends the already-landed cargo check pre-commit).

.ast-grep/
  rules/
    no-cascade-handler-impl-outside-engine-rules.yml  — NEW.
    no-materialized-view-impl-outside-engine-rules.yml — NEW.
    no-lazy-view-impl-outside-engine-rules.yml         — NEW.
    no-topk-view-impl-outside-engine-rules.yml         — NEW.

Cargo.toml (root)         — MODIFIED: workspace `members` updated (engine_generated → engine_data).
```

---

### Task 1: Rename `engine_generated` → `engine_data` (mechanical, structural)

**Files:**
- Modify: `Cargo.toml` (root)
- Move: `crates/engine_generated/` → `crates/engine_data/`
- Modify: `crates/engine_data/Cargo.toml` (package name)
- Modify: `crates/engine/Cargo.toml` (transitive dep — see Task 2; touched here only if engine has a direct path-dep on engine_generated, which it does not currently — leave for Task 2).

- [ ] **Step 1: Create branch + worktree (if not already in one).** This plan should run in `.worktrees/engine-crate-restructure` per `superpowers:using-git-worktrees`. Skip if already there.

```bash
git worktree add .worktrees/engine-crate-restructure -b engine-crate-restructure
cd .worktrees/engine-crate-restructure
cargo build --workspace
cargo test --workspace --no-run   # warm cache so post-rename failures are obviously caused by the rename
```

Expected: clean build.

- [ ] **Step 2: Rename the crate directory with `git mv`.**

```bash
git mv crates/engine_generated crates/engine_data
```

- [ ] **Step 3: Update the package name in the renamed Cargo.toml.**

Edit `crates/engine_data/Cargo.toml`:

Old:
```toml
[package]
name = "engine_generated"
```

New:
```toml
[package]
name = "engine_data"
```

- [ ] **Step 4: Update workspace members in root `Cargo.toml`.**

Old:
```toml
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_generated", "crates/engine_rules", "crates/engine_gpu", "crates/viz", "crates/dsl_compiler"]
```

New:
```toml
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_data", "crates/engine_rules", "crates/engine_gpu", "crates/viz", "crates/dsl_compiler"]
```

- [ ] **Step 5: Sed all `engine_generated` references across the workspace.**

```bash
git grep -l 'engine_generated' | xargs sed -i 's/engine_generated/engine_data/g'
```

This rewrites:
- `Cargo.toml` `path = "../engine_generated"` deps (in `crates/engine_rules/Cargo.toml`)
- `use engine_generated::*` re-exports
- doc strings + comments

- [ ] **Step 6: Build to confirm rename is clean.**

```bash
cargo build --workspace
```

Expected: SUCCESS (no behaviour changes; pure rename).

- [ ] **Step 7: Run tests.**

```bash
cargo test --workspace
```

Expected: all green. The shim `engine_rules::pub use engine_data::*` is unchanged structurally; this is a path-only rename.

- [ ] **Step 8: Commit.**

```bash
git add -A
git commit -m "refactor: rename engine_generated → engine_data (Spec B §3.1)"
```

---

### Task 2: Update `engine` to depend on `engine_data` instead of `engine_rules`

**Why:** Spec §3 direction is `engine_data → engine → engine_rules`. Currently engine depends on engine_rules (which re-exports engine_data). The engine doesn't import any rule logic from engine_rules — only DATA (`engine_rules::ids`, `engine_rules::config`, `engine_rules::events`, `engine_rules::types`, `engine_rules::scoring`, `engine_rules::entities`). Repointing those imports at `engine_data` directly inverts the dependency, which is what Spec §3 requires.

**Files:**
- Modify: `crates/engine/Cargo.toml`
- Modify: `crates/engine/src/{ids.rs, channel.rs, creature.rs, ability/id.rs, event/mod.rs, policy/{macro_kind.rs, utility.rs}, state/mod.rs, step.rs}`
- Modify: every test in `crates/engine/tests/*` that references `engine_rules::*` for data types.
- Modify: callers in other crates that pass through engine — handled in Task 5.

- [ ] **Step 1: Update `engine` Cargo.toml dependency.**

In `crates/engine/Cargo.toml`:

Old:
```toml
[dependencies]
engine_rules = { path = "../engine_rules" }
```

New:
```toml
[dependencies]
engine_data = { path = "../engine_data" }
```

- [ ] **Step 2: Sed-rewrite `engine_rules::` → `engine_data::` inside `crates/engine/`.**

```bash
git grep -l 'engine_rules' crates/engine/ | xargs sed -i 's/engine_rules::/engine_data::/g; s/engine_rules /engine_data /g'
```

Two patterns covered: `engine_rules::path::Type` and the bare `engine_rules` token (rare; only in doc/comment edge cases).

- [ ] **Step 3: Build the engine crate alone.**

```bash
cargo build -p engine
```

Expected: SUCCESS. If a test or src file referenced `engine_rules::SOMETHING_NOT_IN_engine_data`, the compiler will say so — Step 4 lists the only file group where that could happen (none today, since engine_rules is a pure re-export shim).

- [ ] **Step 4: Run engine-only tests (other crates still depend on the shim).**

```bash
cargo test -p engine
```

Expected: SUCCESS.

- [ ] **Step 5: Commit.**

```bash
git add -A
git commit -m "refactor(engine): depend directly on engine_data (Spec B §3.1)"
```

---

### Task 3: Repoint `engine_rules` shim to depend on `engine_data` (transitional)

**Why:** Other workspace crates (`engine_gpu`, root `bevy_game`/`game`, `viz`, `tactical_sim`, tests) still import via the `engine_rules::` shim. Until they're cut over (Task 5), the shim must continue to compile. Today it `pub use engine_generated::*`; after Task 1's rename, sed turned that into `pub use engine_data::*`, which already works — but the `Cargo.toml` `path = "../engine_data"` line was also rewritten by sed in Task 1, so engine_rules already points at engine_data via the renamed dep. This task verifies the transitional state.

**Files:**
- Verify: `crates/engine_rules/Cargo.toml`
- Verify: `crates/engine_rules/src/lib.rs`

- [ ] **Step 1: Verify engine_rules Cargo.toml deps.**

```bash
cat crates/engine_rules/Cargo.toml
```

Expected dep section:
```toml
[dependencies]
engine_data = { path = "../engine_data" }
```

- [ ] **Step 2: Verify engine_rules lib.rs is the transitional re-export.**

```bash
cat crates/engine_rules/src/lib.rs
```

Expected (post-sed): `pub use engine_data::*;`. Anything else means Task 1's sed missed something — fix and re-run.

- [ ] **Step 3: Workspace build.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: all green. (Engine now depends on engine_data; engine_rules shim still re-exports engine_data; downstream crates that import via `engine_rules::*` keep working through the shim.)

- [ ] **Step 4: Commit (only if any drift was fixed; otherwise skip).**

```bash
git add -A
git commit -m "chore(engine_rules): verify shim points at engine_data (transitional)"
```

---

### Task 4: Repoint emitter destinations + regenerate `engine_rules/src/{mask,physics,views}`

**Files:**
- Modify: `src/bin/xtask/cli/mod.rs` — flip `out_physics` / `out_mask` / `out_views` defaults from `crates/engine/src/generated/...` to `crates/engine_rules/src/...`.
- Modify: `crates/dsl_compiler/src/{emit_physics.rs, emit_mask.rs, emit_view.rs, emit_mask_wgsl.rs?, emit_physics_wgsl.rs?}` — change every emitted `"use crate::"` literal to `"use engine::"`. (The Wgsl emitters target shader code, not Rust; usually unaffected — verify.)
- Modify: `crates/engine_rules/Cargo.toml` — add `engine = { path = "../engine" }` to `[dependencies]` (keep `engine_data`).
- Modify: `crates/engine_rules/src/lib.rs` — explicit `pub mod {mask, physics, views};` + transitional `pub use engine_data::*;`.
- Run regen — produces fresh files at the new locations with the new imports.
- Delete: `crates/engine/src/generated/` (entire dir, including `mod.rs`).
- Modify: `crates/engine/src/lib.rs` — drop `pub mod generated;`.
- Modify: 32 hand-written callers of `engine::generated::*` (engine_gpu, tests, root src/) to use `engine_rules::*`. Hand-written code → sed is fine here.

**Discipline note:** Do NOT `git mv` files from `engine/src/generated/` to `engine_rules/src/` and then sed their contents. Those files are emitter output; modifying them directly violates P1 (Compiler-First). The right pattern is "change emitter destination + emitted content → regenerate → delete old output."

- [ ] **Step 1: Flip emit destinations in `src/bin/xtask/cli/mod.rs`.**

Edit the `default_value` strings on the three `pub out_physics`, `pub out_mask`, `pub out_views` fields:

| field | old | new |
|---|---|---|
| `out_physics` | `crates/engine/src/generated/physics` | `crates/engine_rules/src/physics` |
| `out_mask` | `crates/engine/src/generated/mask` | `crates/engine_rules/src/mask` |
| `out_views` | `crates/engine/src/generated/views` | `crates/engine_rules/src/views` |

(Other emit destinations — scoring, entity, config_rust, enum, out_rust — already point at `crates/engine_data/...` after Task 1's sed pass. Verify with `grep -E 'engine_(generated|/src/generated)' src/bin/xtask/cli/mod.rs` — should be empty after this step.)

- [ ] **Step 2: Update emitted `use crate::*` literals to `use engine::*` in dsl_compiler.**

The emitters write Rust source for files that will live under `engine_rules/src/{mask,physics,views}/`. Inside those files, `crate::*` resolves to the host crate (formerly `engine`, now `engine_rules`). Once they live in `engine_rules/`, `crate::event::*` would resolve to `engine_rules::event::*` — which doesn't exist. The fix: emit `engine::event::*` literally.

Audit:

```bash
grep -nE '"use crate::|writeln!\(.*"use crate::' crates/dsl_compiler/src/emit_physics.rs crates/dsl_compiler/src/emit_mask.rs crates/dsl_compiler/src/emit_view.rs
```

For each match, change the literal `"use crate::"` → `"use engine::"`. Also check for non-`use` references to `crate::`:

```bash
grep -nE '"\s*crate::|writeln!\(.*"crate::|format!\(.*"crate::' crates/dsl_compiler/src/emit_*.rs
```

If any match writes a `crate::path::Symbol` reference (not in a `use` statement) into the emitted code, it must also become `engine::path::Symbol`. The emitted files reference engine internals only via `crate::` paths, so any literal `crate::` in the writeout becomes `engine::`.

(WGSL emitters — `emit_mask_wgsl.rs`, `emit_physics_wgsl.rs`, `emit_scoring_wgsl.rs`, `emit_view_wgsl.rs` — emit shader source, not Rust. Their output doesn't have `use crate::` lines. Skip them unless the audit says otherwise.)

- [ ] **Step 3: Update `crates/engine_rules/Cargo.toml`.**

Old:
```toml
[dependencies]
engine_data = { path = "../engine_data" }
```

New:
```toml
[dependencies]
engine = { path = "../engine" }
engine_data = { path = "../engine_data" }
```

- [ ] **Step 4: Replace `crates/engine_rules/src/lib.rs` with the explicit module surface.**

```rust
//! engine_rules — emitted rule logic.
//!
//! This crate is fully generated by `dsl_compiler`. The build.rs sentinel
//! (Task 8) rejects any file that lacks the `// GENERATED by dsl_compiler`
//! header. Hand-edits here are forbidden by the constitution (P1).

#![allow(clippy::all)]

pub mod mask;
pub mod physics;
pub mod views;

// Transitional: data callers still import via engine_rules::*. Phased out by Task 5.
pub use engine_data::*;
```

This `lib.rs` is exempt from the build.rs `// GENERATED` rule (Task 8 puts it on the allowlist). Task 6 will further extend it with the `GeneratedRule` marker + Sealed blanket impl.

- [ ] **Step 5: Run `compile-dsl` to regenerate.**

```bash
cargo run --bin xtask -- compile-dsl
```

Expected: SUCCESS. Files appear at `crates/engine_rules/src/{mask,physics,views}/` with `// GENERATED` headers and `use engine::*` imports.

If the regen errors out (e.g. dsl_compiler refuses to write into a non-pre-existing dir, or `mod.rs` conflicts), inspect the failure. Common issues:
- Output dir doesn't exist: `mkdir -p crates/engine_rules/src/{mask,physics,views}` before regen.
- `mod.rs` already declared by `lib.rs`: that's fine — the emitter writes its own `mod.rs` files inside each subdir; `lib.rs`'s `pub mod {mask,physics,views};` reaches into them.

- [ ] **Step 6: Verify regen output looks right.**

```bash
ls crates/engine_rules/src/physics crates/engine_rules/src/mask crates/engine_rules/src/views
# Expect: same file lists that used to be under engine/src/generated/{physics,mask,views}/

head -10 crates/engine_rules/src/physics/damage.rs
# Expect: starts with `// GENERATED by dsl_compiler...` and then `use engine::event::*`, NOT `use crate::event::*`.

grep -rE '^use crate::' crates/engine_rules/src/{mask,physics,views}
# Expect: empty (every emitted file uses `engine::`, not `crate::`).
```

If any emitted file still says `use crate::`, Step 2 missed an emit-site. Fix that emitter, regen, and re-verify.

- [ ] **Step 7: Delete the old `engine/src/generated/` tree.**

```bash
git rm -r crates/engine/src/generated
```

(`git rm -r` removes the dir, all files in it, and the now-empty `mod.rs`.)

- [ ] **Step 8: Drop `pub mod generated;` from `crates/engine/src/lib.rs`.**

Hand-edit (or `sed -i '/^pub mod generated;$/d' crates/engine/src/lib.rs`) — also remove any nearby comment block solely about the generated tree.

- [ ] **Step 8a: Move the convenience registration constructors out of `engine` (rules-aware code shouldn't live in primitives).**

`engine/src/cascade/dispatch.rs` currently has:

```rust
impl CascadeRegistry {
    pub fn with_engine_builtins() -> Self { ... }       // line 54
    pub fn register_engine_builtins(&mut self) { ... }  // line 64
        // body: `crate::generated::physics::register(self);`
}
```

After Task 4's move, `crate::generated::physics::register` lives in `engine_rules::physics::register` — `engine` can't call it without a cycle. Resolution: **the constructors live where the rules live**. Emit them from `dsl_compiler` into the generated `physics/mod.rs` so they're produced as part of the regen, not hand-written.

  - Delete `with_engine_builtins` and `register_engine_builtins` from `crates/engine/src/cascade/dispatch.rs` (the impl block keeps `new`, `register`, `dispatch` etc.; only those two methods are removed).
  - Update `crates/dsl_compiler/src/emit_physics.rs` to also emit two free functions into the generated `engine_rules/src/physics/mod.rs`:

    ```rust
    /// Build a `CascadeRegistry` pre-registered with every DSL-emitted
    /// physics handler. Compiler-emitted; mirrors what was once
    /// `engine::cascade::CascadeRegistry::with_engine_builtins`.
    pub fn with_engine_builtins() -> engine::cascade::CascadeRegistry {
        let mut reg = engine::cascade::CascadeRegistry::new();
        register(&mut reg);
        reg
    }
    ```

    (The emitter already emits `pub fn register(reg: &mut engine::cascade::CascadeRegistry)`; adding `with_engine_builtins` is one extra `writeln!` block at the top of the same emit pass.)

  - Re-run regen to surface the new functions.

- [ ] **Step 9: Sed-rewrite the hand-written callers (`engine::generated::*` AND the constructor sites).**

Two patterns:

```bash
# Path 1: 32 callers of engine::generated::*
git grep -l 'engine::generated::' | xargs sed -i 's|engine::generated::|engine_rules::|g'

# Path 2: 14 callers of CascadeRegistry::with_engine_builtins() — move them to engine_rules.
git grep -l 'CascadeRegistry::with_engine_builtins\|\.register_engine_builtins(' \
  | xargs sed -i 's|CascadeRegistry::with_engine_builtins()|engine_rules::physics::with_engine_builtins()|g; s|\.register_engine_builtins()|; engine_rules::physics::register(\&mut cascade)|g'
```

The second sed is approximate — `register_engine_builtins` was rare (search before sed: `git grep 'register_engine_builtins' -- ':!crates/engine/'`); if it appears more than 1-2 times, hand-edit those callers using the pattern shown.

After this step, every caller of the old constructor uses the engine_rules-side one. Affected files (audit first to confirm): `crates/engine_gpu/tests/{async_smoke,chronicle_batch_path,batch_iter_cap_convergence,cascade_parity,chronicle_batch_probe,indirect_cascade_converges,parity_with_cpu}.rs` and any others surfaced by the grep.

Callers must `use engine_rules;` (or fully-qualify). If a caller is currently `use engine::cascade::CascadeRegistry; let r = CascadeRegistry::with_engine_builtins();`, the post-sed line becomes `let r = engine_rules::physics::with_engine_builtins();` — engine_rules must be in their `Cargo.toml` deps. Most engine_gpu tests already have `engine_rules = { path = "../engine_rules" }` (the shim path); confirm via `grep engine_rules crates/engine_gpu/Cargo.toml`.

- [ ] **Step 10: Workspace build + test.**

```bash
cargo build --workspace 2>&1 | tail -5
cargo test --workspace 2>&1 | tail -10
```

Expected: SUCCESS. Common failure modes:

- A caller still references `engine::generated` in a way Step 9 didn't catch (e.g., a `mod` declaration, or a doc-test inside a docstring): grep `git grep 'engine::generated' :^crates/dsl_compiler` and fix.
- A regenerated file references a symbol that doesn't exist on `engine::` because the emitter still wrote `engine_rules::` (or vice versa): inspect the emitter, fix, regen.

- [ ] **Step 11: Verify regen idempotence.**

```bash
cargo run --bin xtask -- compile-dsl
git diff --stat crates/engine_rules/ crates/engine_data/
```

Expected: empty diff. The output dirs are the regenerator's source of truth; running twice in a row should produce the same files.

- [ ] **Step 12: Commit.**

```bash
git add -A
git commit -m "refactor(dsl_compiler): emit physics/mask/views to engine_rules/src/; delete engine/src/generated/ (Spec B §3.2)"
```

---

### Task 5: Cut downstream crates from `engine_rules::*` data path to `engine_data::*` directly

**Why:** Now that engine_rules carries rule logic (not just a data re-export shim), keeping the `pub use engine_data::*` line in engine_rules/lib.rs is an attractive nuisance — it tempts callers to keep importing data from engine_rules and creates a path-aliasing problem. Spec §3.3 says cut `engine_rules::ids::*` (data) callers over to `engine_data::ids::*`.

**Files:**
- Modify: every caller of `engine_rules::{ids, config, types, scoring, entities, events, schema, id_serde}` outside `engine_rules/`.
- Modify: `crates/engine_rules/src/lib.rs` — remove `pub use engine_data::*;` once the last data caller is cut over.

- [ ] **Step 1: Identify all `use engine_rules::{data-path}` imports.**

```bash
git grep -E 'engine_rules::(ids|config|types|scoring|entities|events|schema|id_serde)' \
    -- ':(exclude)crates/engine_rules' \
    > /tmp/b1-task5-data-callers.txt
wc -l /tmp/b1-task5-data-callers.txt
```

- [ ] **Step 2: Sed-rewrite the listed paths to `engine_data::`.**

```bash
for sub in ids config types scoring entities events schema id_serde; do
    git grep -l "engine_rules::${sub}" \
        -- ':(exclude)crates/engine_rules' \
        | xargs sed -i "s|engine_rules::${sub}|engine_data::${sub}|g"
done
```

- [ ] **Step 3: For crates that imported via engine_rules (and don't yet have a Cargo dep on engine_data), add the dep.**

Inspect each affected crate's Cargo.toml:

```bash
for c in crates/engine_gpu crates/viz crates/tactical_sim crates/dsl_compiler; do
    grep -E '^engine_data\s*=|^engine_rules\s*=' "$c/Cargo.toml" || echo "NEEDS: $c"
done
```

For any crate marked `NEEDS:`, add `engine_data = { path = "../engine_data" }` to `[dependencies]` (or `[dev-dependencies]` if only tests use it).

- [ ] **Step 4: Workspace build.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS.

- [ ] **Step 5: Drop the transitional `pub use engine_data::*;` from engine_rules/lib.rs.**

Edit `crates/engine_rules/src/lib.rs`:

Old:
```rust
//! engine_rules — emitted rule logic.
//! ...
pub mod mask;
pub mod physics;
pub mod views;

// Backward-compat shim for the data-only re-export path. Phased out by Task 5.
pub use engine_data::*;
```

New:
```rust
//! engine_rules — emitted rule logic.
//! ...
pub mod mask;
pub mod physics;
pub mod views;
```

- [ ] **Step 6: Final workspace build to confirm no caller still relies on the shim.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS. If any failure says `unresolved import engine_rules::ids` (or similar data path), that file was missed in Step 2 — fix and re-run.

- [ ] **Step 7: Commit.**

```bash
git add -A
git commit -m "refactor: route data-path imports through engine_data; engine_rules now rule-only (Spec B §3.3)"
```

---

### Task 6: Seal `CascadeHandler` + view traits via `__sealed::Sealed` + `GeneratedRule` marker, AND emit markers from `dsl_compiler`

**Why combined:** sealing the traits and emitting the markers must land in one commit. If sealing lands first, no emitted rule satisfies `Sealed` and the build breaks. If emitter changes land first, `impl crate::GeneratedRule for X {}` references an undeclared trait. Both pieces are one logical change: "seal traits + teach emitter to emit the marker that satisfies the seal."

**Files:**
- Modify: `crates/engine/src/cascade/handler.rs` — add `pub mod __sealed { pub trait Sealed {} }`; change `pub trait CascadeHandler: Send + Sync` → `pub trait CascadeHandler: __sealed::Sealed + Send + Sync`.
- Modify: `crates/engine/src/view/{materialized,lazy,topk}.rs` — same supertrait addition for each view trait.
- Modify: `crates/engine_rules/src/lib.rs` — add `pub trait GeneratedRule {}` + `impl<T: GeneratedRule> engine::cascade::handler::__sealed::Sealed for T {}` blanket impl (and equivalents reaching into the three view module sealing private supertraits — see Step 1 substructure).
- Modify: `crates/dsl_compiler/src/emit_physics.rs` (and `emit_view.rs` if it emits trait impls) — add a sibling `writeln!(out, "impl crate::GeneratedRule for {} {{}}", handler_name)?;` after every emitted `impl CascadeHandler/MaterializedView/LazyView/TopKView` block.
- Modify: test/demo impls in `crates/engine/tests/{cascade_bounded.rs, cascade_register_dispatch.rs, cascade_lanes.rs, proptest_cascade_bound.rs}` — add `#[cfg(test)] impl engine_rules::GeneratedRule for {handler} {}`.
- Modify: in-engine demo impls (`crates/engine/src/view/{materialized,lazy,topk}.rs`'s `DamageTaken`, `NearestEnemyLazy`, `MostHostileTopK`) — add `#[cfg(test)] impl engine_rules::GeneratedRule for {Demo} {}`.
- Modify: `crates/engine/Cargo.toml` `[dev-dependencies]` — add `engine_rules = { path = "../engine_rules" }`.

- [ ] **Step 1: Add the private supertrait module to `crates/engine/src/cascade/handler.rs`.**

Insert near the top (after the existing `use` statements, before `pub enum EventKindId`):

```rust
/// Sealing private supertrait. Only `engine_rules` may implement, because
/// `engine_rules` blanket-implements `Sealed` for any type that derives
/// the `GeneratedRule` marker, which `dsl_compiler` is the only emitter of.
///
/// External code that tries `impl __sealed::Sealed for Foo` is rejected by:
///   1. Compile error at this crate boundary (the module is `pub mod` but
///      the trait inside is reachable only by the blanket impl below).
///   2. `engine_rules/build.rs` rejecting any file lacking `// GENERATED`.
///   3. The ast-grep CI rule (Task 13).
pub mod __sealed {
    pub trait Sealed {}
}
```

- [ ] **Step 2: Add the supertrait to `CascadeHandler`.**

Old:
```rust
pub trait CascadeHandler: Send + Sync {
    fn handle(&self, /* ... */);
}
```

New:
```rust
pub trait CascadeHandler: __sealed::Sealed + Send + Sync {
    fn handle(&self, /* ... */);
}
```

(Use the actual `fn handle(...)` signature already in the file; only the supertrait list changes.)

- [ ] **Step 3: Add the marker + blanket impl to `crates/engine_rules/src/lib.rs`.**

Replace the file contents with (lib.rs is exempt from the build.rs `// GENERATED` header rule via the `lib.rs` allowlist in Task 8):

```rust
//! engine_rules — emitted rule logic.
//!
//! Fully generated by `dsl_compiler`. `engine_rules/build.rs` rejects any
//! file in this crate that lacks the `// GENERATED by dsl_compiler` header
//! (lib.rs is exempt — it's the marker + blanket-impl module).
//!
//! This file declares the `GeneratedRule` marker and provides the blanket
//! impl that satisfies `engine::cascade::handler::__sealed::Sealed`. Combined
//! with the build sentinel and the ast-grep CI rule, this means only
//! compiler-emitted rule types can satisfy `CascadeHandler`'s supertrait
//! bound.

#![allow(clippy::all)]

pub mod mask;
pub mod physics;
pub mod views;

/// Marker derived (or hand-impl'd inside this crate) by every compiler-
/// emitted rule struct. `dsl_compiler` writes `impl GeneratedRule for FooHandler {}`
/// next to each emitted `impl CascadeHandler for FooHandler`.
#[doc(hidden)]
pub trait GeneratedRule {}

impl<T: GeneratedRule> engine::cascade::handler::__sealed::Sealed for T {}
```

- [ ] **Step 4: Update `dsl_compiler` emitters to emit `impl crate::GeneratedRule for X {}` next to every emitted trait impl.**

DO NOT hand-edit files under `crates/engine_rules/src/{mask,physics,views}/`. Those are emitted output. The right path is to teach the emitter, then regen.

Audit the emit-sites:

```bash
grep -nE '"impl (CascadeHandler|MaterializedView|LazyView|TopKView)' crates/dsl_compiler/src/emit_*.rs
```

For each match, locate the `writeln!(out, ...)` block that emits `impl Trait for FooHandler { ... }`. Add a sibling write-out immediately after:

```rust
writeln!(out, "impl crate::GeneratedRule for {} {{}}", handler_name)?;
```

(The exact variable name `handler_name` matches whatever the surrounding emit-block uses for the type name — `physics_decl.name`, `handler.ident`, etc.)

If the current emit pattern is `pub fn handle_foo(...)` (function-style, no struct), there's no `impl Trait for X` to seal — skip those emit-sites. The trait-impl emit-sites are the only ones that need the marker.

After updating the emitters, regenerate:

```bash
cargo run --bin xtask -- compile-dsl
```

Verify markers landed:

```bash
grep -rE 'impl crate::GeneratedRule' crates/engine_rules/src/{physics,views,mask}/ | wc -l
```

Expected: ≥ 1 per emitted struct that implements one of the four sealed traits.

If the count is 0 but the audit in Step 4 found emit-sites, the writeln addition was wrong — re-inspect and re-regen.

- [ ] **Step 5: Add `#[cfg(test)] impl GeneratedRule` shims for test handlers.**

In `crates/engine/tests/cascade_bounded.rs` (and the three sibling test files with `impl CascadeHandler for {test_handler}`), find each `impl CascadeHandler for {Test}` and add immediately above:

```rust
#[cfg(test)]
impl engine_rules::GeneratedRule for {Test} {}
```

(This admits demo/test impls without opening the seal in production. Per Spec §4.3.)

Files to edit:
- `crates/engine/tests/cascade_bounded.rs` — handlers `Amplifier`, `Once`
- `crates/engine/tests/cascade_register_dispatch.rs` — `Counting`
- `crates/engine/tests/cascade_lanes.rs` — `OrderMarker`
- `crates/engine/tests/proptest_cascade_bound.rs` — `CountingHandler`

(Exact list: each `impl CascadeHandler for X` in that directory needs a sibling `impl engine_rules::GeneratedRule for X`.)

`engine`'s `[dev-dependencies]` must include `engine_rules = { path = "../engine_rules" }` for these test imports to resolve. Add it:

```toml
[dev-dependencies]
engine_rules = { path = "../engine_rules" }
```

(Note: this creates an `engine_rules → engine → engine_rules` *dev-only* cycle. Cargo allows dev-dep cycles. If a regular-dep cycle complaint surfaces, the cycle is the wrong shape — but as a dev-dep it's clean.)

- [ ] **Step 6: Workspace build + test.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS. The seal is now active; every existing rule + test rule is admitted via `GeneratedRule`.

- [ ] **Step 7: Commit.**

```bash
git add -A
git commit -m "refactor(engine): seal CascadeHandler via __sealed::Sealed + GeneratedRule marker (Spec B §4.1)"
```

---

### Task 7: Clean up `dsl_compiler/src/lib.rs` doc comments + verify regen idempotence

**Why:** Tasks 4 + 6 already moved the emit destinations and added the `GeneratedRule` emission. This task is small mop-up: stale doc-comment paths in `dsl_compiler/src/lib.rs`, and a final idempotence check on the regenerator.

**Files:**
- Modify: `crates/dsl_compiler/src/lib.rs` — doc comments only.

- [ ] **Step 1: Audit doc-comment path references in `dsl_compiler/src/lib.rs`.**

```bash
grep -nE 'crates/engine/src/generated|crates/engine_generated' crates/dsl_compiler/src/lib.rs
```

For each match, swap to the new location:
- `crates/engine/src/generated/physics` → `crates/engine_rules/src/physics`
- `crates/engine/src/generated/mask`    → `crates/engine_rules/src/mask`
- `crates/engine/src/generated/views`   → `crates/engine_rules/src/views`
- `crates/engine_generated/...` → `crates/engine_data/...` (Task 1's sed should have caught these; verify)

Use `Edit` per match to preserve surrounding prose. Don't blanket-sed.

- [ ] **Step 2: Run regen + assert idempotence.**

```bash
cargo run --bin xtask -- compile-dsl
git diff --stat crates/engine_rules crates/engine_data
```

Expected: empty diff. If non-empty, the emitter still drifts from the committed output — investigate before continuing.

- [ ] **Step 3: Workspace build + test.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS.

- [ ] **Step 4: Commit (skip if Step 1 found nothing to change).**

```bash
git add -A
git commit -m "chore(dsl_compiler): refresh doc-comment paths to match new emit destinations"
```

---

### Task 8: `engine_rules/build.rs` + `engine_data/build.rs` — every-file-must-be-generated sentinel

**Files:**
- Create: `crates/engine_rules/build.rs`
- Create: `crates/engine_data/build.rs`
- Modify: `crates/engine_rules/Cargo.toml` — add `build = "build.rs"`.
- Modify: `crates/engine_data/Cargo.toml` — add `build = "build.rs"`.

- [ ] **Step 1: Write `crates/engine_rules/build.rs`.**

```rust
//! engine_rules build sentinel.
//!
//! Every file under `src/` (other than `lib.rs`) must start with a
//! `// GENERATED by dsl_compiler` header within the first 5 lines. This
//! makes hand-edited rule logic structurally impossible: the build fails
//! at compile time if anyone bypasses dsl_compiler.

use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src");
    walk(Path::new("src"));
}

fn walk(dir: &Path) {
    for entry in fs::read_dir(dir).expect("readable src dir") {
        let entry = entry.expect("readable entry");
        let path = entry.path();
        let ft = entry.file_type().expect("file type");
        if ft.is_dir() {
            walk(&path);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name() == Some(std::ffi::OsStr::new("lib.rs"))
            && path.parent() == Some(Path::new("src"))
        {
            continue; // top-level lib.rs is the marker + blanket-impl module.
        }
        let content = fs::read_to_string(&path).expect("readable rs file");
        let head: String = content.lines().take(5).collect::<Vec<_>>().join("\n");
        if !head.contains("// GENERATED by dsl_compiler") {
            panic!(
                "engine_rules: {} is missing the `// GENERATED by dsl_compiler` header. \
                 Hand-edited files in this crate are forbidden. Edit the .sim source \
                 in assets/sim/ and rerun `cargo run --bin xtask -- compile-dsl`.",
                path.display()
            );
        }
    }
}
```

- [ ] **Step 2: Add `build = "build.rs"` to `crates/engine_rules/Cargo.toml`.**

Old `[package]` block:
```toml
[package]
name = "engine_rules"
version = "0.1.0"
edition = "2021"
```

New:
```toml
[package]
name = "engine_rules"
version = "0.1.0"
edition = "2021"
build = "build.rs"
```

- [ ] **Step 3: Write `crates/engine_data/build.rs` (same body, swap crate name in the panic).**

```rust
//! engine_data build sentinel.
//!
//! Every file under `src/` (other than `lib.rs`) must start with a
//! `// GENERATED by dsl_compiler` header within the first 5 lines. This
//! is the data-shapes counterpart to engine_rules/build.rs.

use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src");
    walk(Path::new("src"));
}

fn walk(dir: &Path) {
    for entry in fs::read_dir(dir).expect("readable src dir") {
        let entry = entry.expect("readable entry");
        let path = entry.path();
        let ft = entry.file_type().expect("file type");
        if ft.is_dir() {
            walk(&path);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name() == Some(std::ffi::OsStr::new("lib.rs"))
            && path.parent() == Some(Path::new("src"))
        {
            continue;
        }
        let content = fs::read_to_string(&path).expect("readable rs file");
        let head: String = content.lines().take(5).collect::<Vec<_>>().join("\n");
        if !head.contains("// GENERATED by dsl_compiler") {
            panic!(
                "engine_data: {} is missing the `// GENERATED by dsl_compiler` header. \
                 Hand-edited files in this crate are forbidden. Edit the DSL source \
                 in assets/sim/ and rerun `cargo run --bin xtask -- compile-dsl`.",
                path.display()
            );
        }
    }
}
```

- [ ] **Step 4: Add `build = "build.rs"` to `crates/engine_data/Cargo.toml`.**

(Same edit as Step 2 — append `build = "build.rs"` to the `[package]` block.)

- [ ] **Step 5: Run a clean build to verify both sentinels accept the freshly-emitted code.**

```bash
cargo clean -p engine_rules -p engine_data
cargo build -p engine_rules -p engine_data
```

Expected: SUCCESS. If a file is missing the header, the build panics with a path + the fix instruction. Run `cargo run --bin xtask -- compile-dsl` to regen, then retry.

- [ ] **Step 6: Negative test — confirm the sentinel actually fails.**

Inject a hand-edited file and confirm the panic:

```bash
echo "pub fn placeholder() {}" > crates/engine_rules/src/_sentinel_test.rs
cargo build -p engine_rules 2>&1 | grep "missing the .// GENERATED" && echo "OK: sentinel fired"
rm crates/engine_rules/src/_sentinel_test.rs
cargo build -p engine_rules
```

Expected: first build panics with the helpful message; second build (after removing the file) succeeds.

- [ ] **Step 7: Workspace test pass.**

```bash
cargo test --workspace
```

Expected: SUCCESS.

- [ ] **Step 8: Commit.**

```bash
git add -A
git commit -m "feat: add build.rs sentinels enforcing // GENERATED header on engine_rules + engine_data (Spec B §5.1, §5.3)"
```

---

### Task 9: `engine/build.rs` — primitives-only allowlist + reject-`// GENERATED` sentinel

**Files:**
- Create: `crates/engine/build.rs`
- Modify: `crates/engine/Cargo.toml` — add `build = "build.rs"`.

- [ ] **Step 1: Inventory current engine top-level files + dirs to seed the allowlist.**

```bash
ls crates/engine/src/
```

Expected current set:
- top-level `.rs`: `lib.rs`, `ids.rs`, `creature.rs`, `channel.rs`, `chronicle.rs`, `engagement.rs`, `mask.rs`, `pool.rs`, `rng.rs`, `schema_hash.rs`, `spatial.rs`, `step.rs`, `terrain.rs`, `trajectory.rs`, `backend.rs`
- dirs: `ability`, `aggregate`, `cascade`, `event`, `invariant`, `obs`, `policy`, `pool`, `probe`, `snapshot`, `state`, `telemetry`, `view`

Note: `chronicle.rs` and `engagement.rs` are on the allowlist as documented exceptions (deferred to Plan B2 per the Out-of-scope section). When B2 lands, both are removed from the allowlist; their absence becomes a structural rule.

- [ ] **Step 2: Write `crates/engine/build.rs`.**

```rust
//! engine build sentinel.
//!
//! Two structural rules:
//!   1. Top-level files + directories under `src/` must be on the
//!      allowlist below. New behaviour belongs in engine_rules; new
//!      primitives require an allowlist edit, which is a constitutional
//!      governance event (per Spec B §5.2 / D11).
//!   2. No file under `src/` may carry the `// GENERATED by dsl_compiler`
//!      marker — generated code lives in engine_rules / engine_data.

use std::fs;
use std::path::Path;

const ALLOWED_TOP_LEVEL: &[&str] = &[
    "lib.rs",
    "backend.rs",
    "channel.rs",
    "chronicle.rs",      // deferred to Plan B2
    "creature.rs",
    "engagement.rs",     // deferred to Plan B2
    "ids.rs",
    "mask.rs",
    "pool.rs",
    "rng.rs",
    "schema_hash.rs",
    "spatial.rs",
    "step.rs",
    "terrain.rs",
    "trajectory.rs",
];

const ALLOWED_DIRS: &[&str] = &[
    "ability",
    "aggregate",
    "cascade",
    "event",
    "invariant",
    "obs",
    "policy",
    "pool",
    "probe",
    "snapshot",
    "state",
    "telemetry",
    "view",
];

fn main() {
    println!("cargo:rerun-if-changed=src");
    let dir = Path::new("src");
    for entry in fs::read_dir(dir).expect("readable engine/src") {
        let entry = entry.expect("readable entry");
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        let ft = entry.file_type().expect("file type");
        if ft.is_dir() {
            if !ALLOWED_DIRS.contains(&name_s.as_ref()) {
                panic!(
                    "engine/src/{}/: not in primitives allowlist. Engine contains \
                     primitives only — behaviour belongs in engine_rules. To add a \
                     new primitive subdir, edit engine/build.rs ALLOWED_DIRS and \
                     follow the §5.2 governance gate (pros/cons + 2 biased-against \
                     critic PASSes + user approval recorded as ADR).",
                    name_s
                );
            }
        } else if name_s.ends_with(".rs") {
            if !ALLOWED_TOP_LEVEL.contains(&name_s.as_ref()) {
                panic!(
                    "engine/src/{}: not in primitives allowlist. See engine/build.rs.",
                    name_s
                );
            }
        }
    }
    walk_for_pattern(dir, "// GENERATED by dsl_compiler");
}

fn walk_for_pattern(dir: &Path, pat: &str) {
    for entry in fs::read_dir(dir).expect("readable") {
        let entry = entry.expect("entry");
        let path = entry.path();
        let ft = entry.file_type().expect("ft");
        if ft.is_dir() {
            walk_for_pattern(&path, pat);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let content = fs::read_to_string(&path).expect("readable rs");
        if content.contains(pat) {
            panic!(
                "engine/src/{}: contains `// GENERATED by dsl_compiler` marker. \
                 Generated code lives in engine_rules/ or engine_data/, not engine/. \
                 If this file was hand-written, remove the marker; if it was supposed \
                 to be emitted, move it to the right crate.",
                path.strip_prefix(Path::new("src")).unwrap_or(&path).display()
            );
        }
    }
}
```

- [ ] **Step 2: Add `build = "build.rs"` to `crates/engine/Cargo.toml`.**

(Same pattern as Task 8 Step 2.)

- [ ] **Step 3: Clean build to confirm allowlist accepts current engine layout.**

```bash
cargo clean -p engine
cargo build -p engine
```

Expected: SUCCESS.

- [ ] **Step 4: Negative tests — confirm both rules fire.**

```bash
# Rule 1: unknown top-level file
echo "pub fn placeholder() {}" > crates/engine/src/_disallowed.rs
cargo build -p engine 2>&1 | grep "not in primitives allowlist" && echo "OK: allowlist fired"
rm crates/engine/src/_disallowed.rs

# Rule 2: GENERATED marker rejected
echo "// GENERATED by dsl_compiler" > crates/engine/src/_marker.rs
echo "pub fn placeholder() {}" >> crates/engine/src/_marker.rs
# To make rule 2 fire (not rule 1), temporarily add `_marker.rs` to ALLOWED_TOP_LEVEL,
# build, observe rule 2 panic, then revert allowlist + remove file.
sed -i 's|"trajectory.rs",|"trajectory.rs", "_marker.rs",|' crates/engine/build.rs
cargo build -p engine 2>&1 | grep "contains .// GENERATED.* marker" && echo "OK: marker rejected"
sed -i 's|"trajectory.rs", "_marker.rs",|"trajectory.rs",|' crates/engine/build.rs
rm crates/engine/src/_marker.rs

cargo build -p engine
```

Expected: both rules fire with their respective panic messages; clean build at the end.

- [ ] **Step 5: Workspace test pass.**

```bash
cargo test --workspace
```

Expected: SUCCESS.

- [ ] **Step 6: Commit.**

```bash
git add -A
git commit -m "feat(engine): primitives-only build.rs allowlist + reject // GENERATED markers (Spec B §5.2)"
```

---

### Task 10: `trybuild` compile-fail test for the seal

**Files:**
- Modify: `crates/engine/Cargo.toml` — add `trybuild` to `[dev-dependencies]`.
- Create: `crates/engine/tests/sealed_cascade_handler.rs`
- Create: `crates/engine/tests/ui/external_impl_rejected.rs`
- Create: `crates/engine/tests/ui/external_impl_rejected.stderr` (after first run; trybuild can populate via `TRYBUILD=overwrite`).

- [ ] **Step 1: Add `trybuild` dev-dep in `crates/engine/Cargo.toml`.**

```toml
[dev-dependencies]
# ... existing ...
trybuild = "1"
```

- [ ] **Step 2: Write the test driver `crates/engine/tests/sealed_cascade_handler.rs`.**

```rust
//! Compile-fail test: external types must NOT be able to `impl CascadeHandler`.
//!
//! If this test ever passes (i.e., the external impl compiles), the seal is
//! broken. The expected error is that `__sealed::Sealed` is not implemented
//! for the external type.

#[test]
fn external_cascade_handler_impl_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/external_impl_rejected.rs");
}
```

- [ ] **Step 3: Write the compile-fail fixture `crates/engine/tests/ui/external_impl_rejected.rs`.**

```rust
//! This fixture must FAIL to compile. The error is that __sealed::Sealed
//! isn't implemented for `MyHandler`, and `engine_rules::GeneratedRule`
//! is the only way to satisfy it (and that path is unavailable from a
//! random downstream crate).

use engine::cascade::CascadeHandler;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::SimState;

struct MyHandler;

impl CascadeHandler for MyHandler {
    fn handle(
        &self,
        _agent: AgentId,
        _event: &Event,
        _state: &mut SimState,
        _events: &mut EventRing,
    ) {}
}

fn main() {}
```

(If the actual `fn handle` signature on `CascadeHandler` differs, mirror it from `crates/engine/src/cascade/handler.rs`. The point is: a syntactically well-formed `impl CascadeHandler for MyHandler` that fails *only* because `Sealed` is unsatisfied.)

- [ ] **Step 4: Run the test once to populate the expected stderr.**

```bash
TRYBUILD=overwrite cargo test -p engine --test sealed_cascade_handler
```

This writes `crates/engine/tests/ui/external_impl_rejected.stderr` with the actual compiler diagnostic.

- [ ] **Step 5: Inspect the captured stderr.**

```bash
cat crates/engine/tests/ui/external_impl_rejected.stderr
```

Expected: contains a line referencing `the trait bound .*Sealed.* is not satisfied` (or equivalent rustc phrasing). If it instead complains about a missing function or a syntax error, the fixture is wrong — fix the fixture and re-run with `TRYBUILD=overwrite`.

- [ ] **Step 6: Run normally to confirm the test passes (i.e., stderr matches).**

```bash
cargo test -p engine --test sealed_cascade_handler
```

Expected: PASS (the fixture fails to compile with the expected error).

- [ ] **Step 7: Negative test — confirm the test catches a broken seal.**

```bash
# Temporarily relax the seal: drop __sealed::Sealed from the supertrait list.
git stash --keep-index  # save current state
# Hand-edit crates/engine/src/cascade/handler.rs: change
#   pub trait CascadeHandler: __sealed::Sealed + Send + Sync {
# to
#   pub trait CascadeHandler: Send + Sync {
# Then:
cargo test -p engine --test sealed_cascade_handler 2>&1 | grep -E "^test .* FAILED|expected an error" && echo "OK: trybuild caught broken seal"
# Restore:
git stash pop
cargo test -p engine --test sealed_cascade_handler
```

Expected: under the broken seal, trybuild reports the fixture compiles when it shouldn't, and the test fails. Restoring the seal restores green.

- [ ] **Step 8: Commit.**

```bash
git add -A
git commit -m "test(engine): trybuild compile-fail test asserting CascadeHandler seal (Spec B §4.4)"
```

---

### Task 11: Add `cargo run --bin xtask -- compile-dsl --check` subcommand

**Files:**
- Modify: `src/bin/xtask/cli/mod.rs` — add `--check` flag to `CompileDslArgs`.
- Modify: `src/bin/xtask/compile_dsl_cmd.rs` — branch on `args.check`.

- [ ] **Step 1: Add `--check` flag to the args struct.**

In `src/bin/xtask/cli/mod.rs`, find `pub struct CompileDslArgs`. Add:

```rust
    /// Regenerate to a temporary directory and diff against the working
    /// tree. Exit non-zero if any generated path differs. Used by the
    /// pre-commit hook + CI to catch staleness.
    #[arg(long, default_value_t = false)]
    pub check: bool,
```

- [ ] **Step 2: Implement `--check` in `compile_dsl_cmd.rs`.**

Add at the top of `run_compile_dsl(args: CompileDslArgs)` (or wherever the dispatch sits):

```rust
if args.check {
    return run_compile_dsl_check(&args);
}
```

Then add the function:

```rust
fn run_compile_dsl_check(args: &CompileDslArgs) -> ExitCode {
    use std::process::Command as ProcessCommand;
    let tmp = match tempfile::tempdir() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("compile-dsl --check: tempdir create failed: {e}");
            return ExitCode::from(2);
        }
    };
    let tmp_path = tmp.path();

    // Build a new args with the same input but redirected outputs into tmp.
    let mut redirected = args.clone();
    redirected.check = false;
    redirected.out_physics       = tmp_path.join("engine_rules/src/physics");
    redirected.out_mask          = tmp_path.join("engine_rules/src/mask");
    redirected.out_views         = tmp_path.join("engine_rules/src/views");
    redirected.out_scoring       = tmp_path.join("engine_data/src/scoring");
    redirected.out_entity        = tmp_path.join("engine_data/src/entities");
    redirected.out_config_rust   = tmp_path.join("engine_data/src/config");
    redirected.out_enum          = tmp_path.join("engine_data/src/enums");
    redirected.out_rust          = tmp_path.join("engine_data/src");  // events + schema + ids tree

    // Make every parent dir.
    for p in [
        &redirected.out_physics,
        &redirected.out_mask,
        &redirected.out_views,
        &redirected.out_scoring,
        &redirected.out_entity,
        &redirected.out_config_rust,
        &redirected.out_enum,
        &redirected.out_rust,
    ] {
        if let Err(e) = std::fs::create_dir_all(p) {
            eprintln!("compile-dsl --check: mkdir {} failed: {e}", p.display());
            return ExitCode::from(2);
        }
    }

    // Run the real compile-dsl into tmp.
    if !matches!(run_compile_dsl_inner(&redirected), ExitCode::SUCCESS) {
        eprintln!("compile-dsl --check: compile failed");
        return ExitCode::from(2);
    }

    // Diff each redirected dir against the corresponding live dir.
    let pairs: &[(&PathBuf, &str)] = &[
        (&redirected.out_physics,     "crates/engine_rules/src/physics"),
        (&redirected.out_mask,        "crates/engine_rules/src/mask"),
        (&redirected.out_views,       "crates/engine_rules/src/views"),
        (&redirected.out_scoring,     "crates/engine_data/src/scoring"),
        (&redirected.out_entity,      "crates/engine_data/src/entities"),
        (&redirected.out_config_rust, "crates/engine_data/src/config"),
        (&redirected.out_enum,        "crates/engine_data/src/enums"),
        // out_rust covers events/, schema.rs, ids.rs at engine_data/src/ root —
        // diff the same directory but exclude the subdirs already diffed above
        // to avoid double-reporting. (See impl note below.)
    ];
    // For the out_rust tree, diff only the top-level files (events/, schema.rs,
    // ids.rs, id_serde.rs, types.rs) — the subdir paths already covered.
    let live_root = std::path::Path::new("crates/engine_data/src");
    for entry in std::fs::read_dir(live_root).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if matches!(name_s.as_ref(), "scoring" | "entities" | "config" | "enums") {
            continue;
        }
        let live = live_root.join(&name);
        let tmp_eq = redirected.out_rust.join(&name);
        let status = ProcessCommand::new("diff").arg("-rq").arg(&tmp_eq).arg(&live).status();
        match status {
            Ok(s) if s.success() => {}
            Ok(_) => {
                eprintln!("DRIFT: {} differs from regenerated output", live.display());
                drift = true;
            }
            Err(e) => {
                eprintln!("compile-dsl --check: diff failed for {}: {e}", live.display());
                return ExitCode::from(2);
            }
        }
    }
    let mut drift = false;
    for (tmp_dir, live_dir) in pairs {
        let status = ProcessCommand::new("diff")
            .arg("-rq")
            .arg(tmp_dir)
            .arg(live_dir)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(_) => {
                eprintln!("DRIFT: {} differs from regenerated output", live_dir);
                drift = true;
            }
            Err(e) => {
                eprintln!("compile-dsl --check: diff failed for {}: {e}", live_dir);
                return ExitCode::from(2);
            }
        }
    }
    if drift {
        eprintln!(
            "compile-dsl --check: generated dirs are stale relative to DSL source. \
             Run `cargo run --bin xtask -- compile-dsl` and stage the changes."
        );
        return ExitCode::FAILURE;
    }
    eprintln!("compile-dsl --check: generated dirs match DSL source.");
    ExitCode::SUCCESS
}
```

(The function refactoring `run_compile_dsl_inner` extracts the existing body of `run_compile_dsl` so `--check` can drive it without recursing through the args branch.)

- [ ] **Step 3: Add `tempfile` to xtask dev/runtime deps if not present.**

Check root `Cargo.toml`:

```bash
grep -E '^tempfile\s*=' Cargo.toml
```

If absent, add to `[dependencies]` (xtask binary lives in the root crate):

```toml
tempfile = "3"
```

- [ ] **Step 4: Build + test the new flag.**

```bash
cargo run --bin xtask -- compile-dsl --check
```

Expected: prints "generated dirs match DSL source." and exits 0 — generated dirs were just regenerated in Task 7's commit, so they match.

- [ ] **Step 5: Negative test.**

```bash
# Mutate one generated file to inject drift.
echo "// drift" >> crates/engine_data/src/scoring/mod.rs
cargo run --bin xtask -- compile-dsl --check
echo "exit code: $?"
# Expected: exit 1, "DRIFT: crates/engine_data/src/scoring differs ..."
git checkout crates/engine_data/src/scoring/mod.rs
cargo run --bin xtask -- compile-dsl --check
# Expected: exit 0
```

- [ ] **Step 6: Commit.**

```bash
git add -A
git commit -m "feat(xtask): add compile-dsl --check (regen + diff against working tree) (Spec B §6)"
```

---

### Task 12: Extend `.githooks/pre-commit` with header-rule + regen-on-DSL-change

**Files:**
- Modify: `.githooks/pre-commit`

This extends the already-landed pre-commit hook (which currently runs `cargo check` + the dispatch-critics gate per Spec D-amendment). The new logic appends two checks: header rule, and `compile-dsl --check` when DSL source is staged.

- [ ] **Step 1: Read current state of `.githooks/pre-commit`.**

```bash
cat .githooks/pre-commit
```

- [ ] **Step 2: Append the header-rule + regen-on-DSL-change blocks.**

Add after the existing checks but before the final `exit 0` (or wherever the success exit lives). Use the `Edit` tool to insert this block immediately before the `exit 0` line:

```bash
# === Spec B header rule: files in engine_rules/ or engine_data/ require the // GENERATED header.
for f in $(git diff --cached --name-only --diff-filter=AM \
           | grep -E '^crates/(engine_rules|engine_data)/src/.*\.rs$' \
           | grep -v '/lib\.rs$'); do
    if ! head -5 "$f" | grep -q "// GENERATED by dsl_compiler"; then
        echo "ABORT: $f is in a generated crate but lacks the // GENERATED header." >&2
        echo "Edit the DSL source under assets/sim/ and rerun \`cargo run --bin xtask -- compile-dsl\`." >&2
        exit 1
    fi
done

# === Spec B inverse rule: files outside generated crates must NOT carry the marker.
for f in $(git diff --cached --name-only --diff-filter=AM \
           | grep -E '\.rs$' \
           | grep -vE '^crates/(engine_rules|engine_data)/'); do
    if grep -q "// GENERATED by dsl_compiler" "$f" 2>/dev/null; then
        echo "ABORT: $f contains // GENERATED marker but is not in a generated crate." >&2
        echo "Either move the file to engine_rules/ or engine_data/, or remove the marker." >&2
        exit 1
    fi
done

# === Spec B regen-on-DSL-change: if any DSL source is staged, regen + diff.
if git diff --cached --name-only | grep -qE '^assets/(sim|hero_templates)/'; then
    echo "DSL source changed — running \`compile-dsl --check\`..."
    if ! cargo run --bin xtask -- compile-dsl --check; then
        echo "ABORT: generated dirs are stale relative to DSL source." >&2
        echo "Run \`cargo run --bin xtask -- compile-dsl\` and stage the changes." >&2
        exit 1
    fi
fi
```

- [ ] **Step 3: Verify the hook parses cleanly.**

```bash
bash -n .githooks/pre-commit && echo OK
```

Expected: `OK`.

- [ ] **Step 4: Smoke-test the header rule (dry run via the hook script directly).**

```bash
# Stage a hand-edit to a generated file (without the marker).
cp crates/engine_rules/src/physics/heal.rs /tmp/heal.bak
sed -i '1d' crates/engine_rules/src/physics/heal.rs   # drop the GENERATED header
git add crates/engine_rules/src/physics/heal.rs
.githooks/pre-commit
echo "exit: $?"
# Expected: ABORT message + exit 1.
git restore --staged crates/engine_rules/src/physics/heal.rs
mv /tmp/heal.bak crates/engine_rules/src/physics/heal.rs
```

- [ ] **Step 5: Smoke-test the inverse rule.**

```bash
echo '// GENERATED by dsl_compiler' >> crates/engine/src/lib.rs
git add crates/engine/src/lib.rs
.githooks/pre-commit
echo "exit: $?"
# Expected: ABORT: contains // GENERATED marker but not in generated crate
git restore --staged crates/engine/src/lib.rs
git checkout crates/engine/src/lib.rs
```

- [ ] **Step 6: Smoke-test the regen-on-DSL-change rule.**

```bash
# Stage a no-op DSL edit (touch + git add) — the regen should report no drift.
touch assets/sim/physics.sim
git add assets/sim/physics.sim
.githooks/pre-commit
echo "exit: $?"
# Expected: prints "DSL source changed — running compile-dsl --check..." and exit 0
git restore --staged assets/sim/physics.sim
```

- [ ] **Step 7: Commit.**

```bash
git add .githooks/pre-commit
git commit -m "feat(githooks): pre-commit enforces // GENERATED header + DSL regen freshness (Spec B §6)"
```

---

### Task 13: ast-grep CI rules — `impl CascadeHandler` etc. only allowed in `engine_rules/`

**Files:**
- Create: `.ast-grep/rules/no-cascade-handler-impl-outside-engine-rules.yml`
- Create: `.ast-grep/rules/no-materialized-view-impl-outside-engine-rules.yml`
- Create: `.ast-grep/rules/no-lazy-view-impl-outside-engine-rules.yml`
- Create: `.ast-grep/rules/no-topk-view-impl-outside-engine-rules.yml`
- Modify: existing CI workflow (one of `.github/workflows/*.yml`) — add `ast-grep scan` step.

- [ ] **Step 1: Confirm or set up `.ast-grep/` config dir.**

```bash
ls .ast-grep/ 2>/dev/null || mkdir -p .ast-grep/rules
ls .ast-grep/rules/ 2>/dev/null
```

If `sgconfig.yml` doesn't exist at repo root, create one:

`sgconfig.yml`:
```yaml
ruleDirs:
  - .ast-grep/rules
```

- [ ] **Step 2: Write the cascade rule.**

`.ast-grep/rules/no-cascade-handler-impl-outside-engine-rules.yml`:

```yaml
id: no-cascade-handler-impl-outside-engine-rules
language: rust
rule:
  any:
    - pattern: impl CascadeHandler for $T { $$$ }
    - pattern: impl $$$::CascadeHandler for $T { $$$ }
files:
  - "**/*.rs"
not:
  any:
    - inside:
        kind: source_file
        regex: 'crates/engine_rules/src/'
    - inside:
        kind: source_file
        regex: 'crates/engine/tests/'  # test fixtures get cfg(test) GeneratedRule shim
severity: error
message: |
  `impl CascadeHandler` must live in crates/engine_rules/. Hand-written
  cascade handlers violate P1 (Compiler-First). Edit assets/sim/physics.sim
  and let dsl_compiler emit the handler. If this is a test fixture, move
  it under crates/engine/tests/ where the cfg(test) GeneratedRule shim lives.
```

- [ ] **Step 3: Write the three view rules (same shape, different trait name).**

`no-materialized-view-impl-outside-engine-rules.yml`:

```yaml
id: no-materialized-view-impl-outside-engine-rules
language: rust
rule:
  any:
    - pattern: impl MaterializedView for $T { $$$ }
    - pattern: impl $$$::MaterializedView for $T { $$$ }
files:
  - "**/*.rs"
not:
  any:
    - inside:
        kind: source_file
        regex: 'crates/engine_rules/src/'
    - inside:
        kind: source_file
        regex: 'crates/engine/src/view/materialized\.rs'  # demo impls (DamageTaken)
severity: error
message: |
  `impl MaterializedView` must live in crates/engine_rules/. Hand-written
  view impls violate P1.
```

`no-lazy-view-impl-outside-engine-rules.yml` and `no-topk-view-impl-outside-engine-rules.yml`: identical shape, swap `LazyView` and `TopKView` for the trait name and `lazy.rs`/`topk.rs` for the demo-impl exclusion path.

- [ ] **Step 4: Run ast-grep scan locally.**

```bash
# If ast-grep isn't installed:
# cargo install ast-grep
ast-grep scan
```

Expected: zero violations (the seal + the demo-impl exclusions cover the current state).

- [ ] **Step 5: Negative test — inject a violation, confirm it's caught.**

```bash
cat >> crates/engine/src/lib.rs <<EOF
struct BogusHandler;
impl crate::cascade::CascadeHandler for BogusHandler {
    fn handle(&self, _: crate::ids::AgentId, _: &crate::event::Event, _: &mut crate::state::SimState, _: &mut crate::event::EventRing) {}
}
EOF
ast-grep scan 2>&1 | grep "no-cascade-handler-impl-outside-engine-rules" && echo "OK: ast-grep caught it"
git checkout crates/engine/src/lib.rs
```

(Note: this test will also fail the engine `build.rs` allowlist + the seal — that's fine; we're checking ast-grep specifically catches the trait impl.)

- [ ] **Step 6: Add ast-grep step to CI.**

Find existing CI workflow:

```bash
ls .github/workflows/ 2>/dev/null || echo "no workflows dir"
```

If a Rust CI workflow exists (typically `.github/workflows/ci.yml` or similar), add a step:

```yaml
      - name: ast-grep scan (architectural rules)
        run: |
          curl -fsSL https://github.com/ast-grep/ast-grep/releases/latest/download/ast-grep-x86_64-unknown-linux-gnu.tar.gz | tar xz
          ./ast-grep scan
```

If no CI workflow exists yet, this step is a follow-up (note in commit message). The local enforcement still works via the pre-commit hook + build sentinels.

- [ ] **Step 7: Commit.**

```bash
git add .ast-grep/ sgconfig.yml .github/workflows/*.yml 2>/dev/null
git commit -m "feat(ci): ast-grep rules restricting CascadeHandler/View impls to engine_rules (Spec B §7.1)"
```

---

### Task 14: Stale-content + schema-hash CI guards

**Files:**
- Modify: existing CI workflow under `.github/workflows/` (or create `.github/workflows/architecture.yml` if no Rust workflow exists yet).

- [ ] **Step 1: Find current Rust CI workflow.**

```bash
ls .github/workflows/ 2>/dev/null
```

If found: pick the file that already runs `cargo test` (e.g. `ci.yml`). If absent: create `.github/workflows/architecture.yml` with a Rust toolchain setup and the steps below.

- [ ] **Step 2: Add a "regen + diff" step.**

After the existing `cargo build` / `cargo test` step:

```yaml
      - name: Regenerate DSL artefacts and verify no diff
        run: |
          cargo run --bin xtask -- compile-dsl
          if ! git diff --quiet crates/engine_rules/ crates/engine_data/; then
            echo "::error::Generated dirs are stale — DSL source was changed but generated artefacts weren't committed."
            git diff --stat crates/engine_rules/ crates/engine_data/
            exit 1
          fi
```

- [ ] **Step 3: Add a "schema-hash freshness" step.**

```yaml
      - name: Schema hash freshness
        run: cargo test -p engine --test schema_hash
```

(If `crates/engine/tests/schema_hash.rs` doesn't exist, this step is a no-op and we either find the actual schema-hash test name or defer this guard. Confirm by `ls crates/engine/tests/schema_hash*`.)

- [ ] **Step 4: Local dry-run.**

```bash
cargo run --bin xtask -- compile-dsl
git diff --quiet crates/engine_rules/ crates/engine_data/ && echo "OK: no drift"
cargo test -p engine --test schema_hash 2>/dev/null || echo "(schema_hash test not present; skip)"
```

- [ ] **Step 5: Commit.**

```bash
git add .github/workflows/
git commit -m "feat(ci): stale-content + schema-hash freshness guards (Spec B §7.2, §7.3)"
```

---

### Task 15: Final workspace-wide verification

**Files:** none (validation only).

- [ ] **Step 1: Clean build from scratch.**

```bash
cargo clean
cargo build --workspace
```

Expected: SUCCESS. Watch for any allowlist / sentinel panics — those mean Tasks 8-9 have a bug.

- [ ] **Step 2: Full test pass.**

```bash
cargo test --workspace
```

Expected: SUCCESS, including `crates/engine/tests/sealed_cascade_handler.rs` (Task 10).

- [ ] **Step 3: Confirm `compile-dsl --check` is green.**

```bash
cargo run --bin xtask -- compile-dsl --check
```

Expected: "generated dirs match DSL source."

- [ ] **Step 4: Confirm pre-commit hook runs cleanly on a no-op stage.**

```bash
git config --local core.hooksPath .githooks
git add -N .  # no-op stage
.githooks/pre-commit && echo OK
```

Expected: `OK`.

- [ ] **Step 5: Confirm the seal end-to-end.**

```bash
cargo test -p engine --test sealed_cascade_handler
```

Expected: PASS (the compile-fail fixture fails to compile with the expected `Sealed not satisfied` error).

- [ ] **Step 6: Confirm engine has no `// GENERATED` markers and engine_rules + engine_data have them on every non-lib.rs file.**

```bash
# Should be empty:
grep -rE "// GENERATED by dsl_compiler" crates/engine/src/

# Should match every non-lib.rs file in the two generated crates:
find crates/engine_rules/src crates/engine_data/src -name '*.rs' -not -name 'lib.rs' \
  | while read f; do head -5 "$f" | grep -q "// GENERATED" || echo "MISSING: $f"; done
```

Expected: empty output for both checks.

- [ ] **Step 7: Tick the AIS post-design re-evaluation checkbox in this plan file.**

In this plan's Architectural Impact Statement (top), change `[ ] AIS reviewed post-design` → `[x] AIS reviewed post-design`. Add a one-line note: "Final scope: 15 tasks landed; chronicle/engagement migration deferred to B2; legacy `src/` sweep deferred to B3."

- [ ] **Step 8: Final commit.**

```bash
git add -A
git commit -m "chore: tick AIS post-design checkbox + scope note for Plan B1"
```

---

## Sequencing summary

| Task | What | Depends on |
|---|---|---|
| 1  | Rename `engine_generated` → `engine_data` | — |
| 2  | Engine deps `engine_data` directly | 1 |
| 3  | Verify engine_rules transitional state | 1, 2 |
| 4  | Move `engine/src/generated/{mask,physics,views}` → `engine_rules/src/` | 1-3 |
| 5  | Cut downstream callers from `engine_rules::*` data path → `engine_data::*` | 4 |
| 6  | Seal `CascadeHandler` + view traits | 4, 5 |
| 7  | Update `dsl_compiler` emit paths + emit `GeneratedRule` markers | 4, 6 |
| 8  | `engine_rules/build.rs` + `engine_data/build.rs` sentinels | 7 |
| 9  | `engine/build.rs` allowlist + reject-`// GENERATED` | 8 |
| 10 | `trybuild` compile-fail test | 6 |
| 11 | xtask `compile-dsl --check` | 7 |
| 12 | Pre-commit hook extensions | 11 |
| 13 | ast-grep CI rules | 6 |
| 14 | CI stale-content + schema-hash guards | 11 |
| 15 | Final verification + AIS tick | all |

Tasks 1-9 must run in order. Tasks 10, 11, 13, 14 can interleave once their respective deps land. Task 15 is last.

Each task ends in a `cargo test` verification + commit (P9). If any task verification fails, **stop and ask** — don't push through with `--no-verify`.

## Coordination with already-landed Spec D-amendment

The dispatch-critics + pre-commit gate (commits in `.githooks/pre-commit`, `.claude/scripts/`, `.claude/skills/critic-*/SKILL.md`) are **operational** during this plan. Expect the following interactions:

- The `PreToolUse` hook fires when editing files under `crates/engine/`. It runs fast static checks (per `.claude/scripts/pre-tool-engine-edit.sh`); some checks may flag this plan's structural moves. If a check blocks legitimately, fix the underlying issue; if it false-positives, document via the existing escape hatch (the script's bypass pattern).
- The `Stop` hook (per `.claude/scripts/session-end-engine-review.sh`) fires at session end and dispatches the 6 critics. Each task's commit may trigger the dispatch. **Do not skip the verdict** — read `.claude/critic-output-*.txt` after the dispatch and address any FAIL before proceeding to the next task.
- The pre-commit gate (in `.githooks/pre-commit`) reads the same critic-output files. **Stale critic output** is gated by mtime: any engine source mtime later than the newest critic output forces a re-dispatch before commit. If you see "stale critic output, re-running dispatch-critics," that's expected behaviour and not a bug.
- The `critic-allowlist-gate` skill triggers on edits to `crates/engine/build.rs`. **Task 9 will trigger this gate**. Per Spec B §5.2 D11, this plan's pros/cons (in §5.2 of the spec) + the AIS preamble + this user-approved spec satisfy the governance gate; the critic dispatch will see the AIS + spec references and should PASS. If both biased-against critics return PASS, the commit goes through. If either returns FAIL, **stop and discuss** — don't override.
