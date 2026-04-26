# Legacy `src/` Sweep + xtask Move (Plan B3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Spec B §8.4: move `src/bin/xtask/` to `crates/xtask/`, keep `src/rendering.rs` (window code), `git rm -r` the legacy module tree under `src/{ai,world_sim,scenario,mission,game_core,content,narrative,model_backend,ascii_gen}`, drop the corresponding xtask subcommands that depend on those modules, trim `src/lib.rs`, and verify the workspace still builds and tests green.

**Architecture:** Pure deletion + relocation. The xtask CLI keeps the subcommands that don't depend on legacy `src/` code (the engine-only ones); legacy subcommands are deleted along with their dependencies. Net result: xtask becomes a small thin CLI living in `crates/xtask/`, the root `src/` directory shrinks to `rendering.rs` (or empties entirely if rendering also moves), and the workspace dependency graph contains only the engine crate tree + a few utility crates. Approximate lines removed: ~85K (src/world_sim alone is 72K).

**Tech Stack:** Rust 2021, `cargo`, `git rm`, `git mv`. No new dependencies. No DSL changes.

**Parallel-with:** This plan is independent of Plan B1 (engine crate restructure). Both can run in separate worktrees concurrently. Merge order: whichever lands first wins; the other rebases. The conflict surface is small — Plan B1 touches `crates/{engine,engine_data,engine_rules}`, this plan touches `src/` + `Cargo.toml` workspace members + creates `crates/xtask/`. Only `Cargo.toml` workspace `members` is shared, and the merge is mechanical (additive).

## Architectural Impact Statement

- **Existing primitives searched:**
  - `src/bin/xtask/` (16 subcommand modules; ~14.8K lines): `compile_dsl_cmd.rs`, `chronicle_cmd.rs`, `scenario_cmd.rs`, `roomgen_cmd.rs`, `train_v6.rs`, `map.rs`, `capture.rs`, `model_cmd.rs`, `content_gen_cmd.rs`, `ascii_gen_cmd.rs`, `champion_gen.rs`, `world_sim_cmd.rs`, `visualize_cmd.rs`, `building_ai_cmd.rs`, `oracle_cmd/`, `cli/` (clap arg structs).
  - `src/lib.rs` (33 lines): module declarations + re-exports of `ai`, `effects`, `pathing`, `squad`, `goap`, `control`, `personality`, `roles`, `utility`, `phase`, `advanced`, `student`, `tooling`, `world_sim`, `content`, `model_backend`, `ascii_gen`, `scenario`, `narrative`, `mission`, `game_core`, `rendering`.
  - `src/ai/mod.rs` (39 lines): re-export shim over `tactical_sim::{sim,effects,pathing,squad,goap,control,personality,roles,utility,phase,advanced,student,tooling}`. The actual AI implementation lives in the `crates/tactical_sim/` crate.
  - `src/world_sim/` (71,863 lines): legacy world simulation; superseded by the engine crate tree (per the constitution's compiler-first direction).
  - `src/{scenario,mission,game_core,content,narrative,model_backend,ascii_gen}` (~15,452 lines combined): legacy modules, superseded.
  - `src/rendering.rs` (window code): KEEP per user direction.
  - `crates/tactical_sim/`, `crates/combat-trainer/`, `crates/ability-vae/`, `crates/ability_operator/`, `crates/ability-vae/`: not in scope. These crates are separate decisions; whether they live or die is out-of-scope for B3. They are not in the workspace `members` list except `tactical_sim`, which means `combat-trainer` etc. compile only on demand — the deletion of `src/` doesn't affect them.

  Search method: `rg`, `find ... | wc -l`, direct `Read`.

- **Decision:** delete legacy `src/{ai,world_sim,scenario,mission,game_core,content,narrative,model_backend,ascii_gen}` along with the xtask subcommands that depend on them (`scenario_cmd.rs`, `roomgen_cmd.rs`, `model_cmd.rs`, `content_gen_cmd.rs`, `ascii_gen_cmd.rs`, `champion_gen.rs`, `world_sim_cmd.rs`, `visualize_cmd.rs`, `building_ai_cmd.rs`, `oracle_cmd/`). Move the surviving xtask subcommands (`compile_dsl_cmd.rs`, `train_v6.rs`, `map.rs`, `capture.rs` + their `cli/` arg structs) to `crates/xtask/`. Keep `src/rendering.rs` at root for now (the root `bevy_game` crate retains a binary entry-point if window code calls it; otherwise rendering can move to `crates/window/` as a follow-up).
- **Note on `chronicle_cmd.rs`:** B1' Tasks 9+10 (parallel branch `engine-crate-restructure`) deleted `engine::chronicle` and the `chronicle_cmd` xtask subcommand. B3 should NOT carry `chronicle_cmd.rs` as a survivor; if it still exists in B3's working tree (B3 branched before the engine merge), delete it as part of Task 2's cleanup. When the engine branch merges to main, both branches will have removed `src/bin/xtask/chronicle_cmd.rs` cleanly — no conflict.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none.
  - Generated outputs re-emitted: none — but the deletion of legacy `src/` removes one source of `// GENERATED` markers if any leaked there. After this plan lands, regenerating via `cargo run --bin xtask -- compile-dsl` should produce no diff in `crates/engine_rules/` or `crates/engine_data/`.

- **Hand-written downstream code:**
  - `crates/xtask/Cargo.toml`: NEW (~30 lines). Justification: standard Cargo manifest for a new crate; the surviving xtask subcommands need a home, and a dedicated crate avoids the confusion of having a `[[bin]]` target inside a directory that otherwise contains nothing.
  - `crates/xtask/src/main.rs` + `cli/mod.rs` + 5 surviving subcommand files: MOVED from `src/bin/xtask/`. Justification: relocation, not new code.
  - `src/lib.rs`: SHRUNK to declare only `pub mod rendering;` (or deleted entirely if `rendering.rs` also moves). Justification: clean-up; removing dead `pub mod` declarations after the directories are deleted.
  - Root `Cargo.toml`: MODIFIED — `[[bin]] xtask` block removed; workspace `members` adds `crates/xtask`. Justification: matches the relocation.

- **Constitution check:**
  - P1 (Compiler-First): PASS — strictly removes hand-written rule-shaped logic (legacy `src/world_sim` etc. are pre-DSL-era handwritten simulations).
  - P2 (Schema-Hash on Layout): N/A — no engine state-layout changes.
  - P3 (Cross-Backend Parity): N/A — no `step_full` semantic changes.
  - P4 (`EffectOp` Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): N/A — no RNG changes.
  - P6 (Events Are the Mutation Channel): N/A.
  - P7 (Replayability Flagged): N/A.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `cargo build --workspace` + `cargo test --workspace` + commit.
  - P10 (No Runtime Panic): PASS — net effect is *fewer* runtime paths; no new panics introduced.
  - P11 (Reduction Determinism): N/A.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill). [x] AIS reviewed post-design — final scope: ~99,893 lines deleted from src/; xtask moved to crates/xtask/ with 4 surviving subcommands (map, capture, train-v6, compile-dsl); 21 root deps + 4 features pruned; 11 root integration tests deleted; src/lib.rs now `pub mod rendering;` only.

---

## File Structure

```
crates/
  xtask/                           — NEW.
    Cargo.toml                     — package manifest with the surviving xtask deps.
    src/
      main.rs                      — MOVED from src/bin/xtask/main.rs (gutted of dead arms).
      cli/                         — MOVED from src/bin/xtask/cli/ (subset).
      compile_dsl_cmd.rs           — MOVED.
      chronicle_cmd.rs             — MOVED.
      train_v6.rs                  — MOVED.
      map.rs                       — MOVED.
      capture.rs                   — MOVED.
src/
  rendering.rs                     — KEEP.
  lib.rs                           — SHRUNK to one line: `pub mod rendering;`.
  bin/                             — DELETED (xtask moved out).
  ai/                              — DELETED (was a re-export shim over tactical_sim).
  world_sim/                       — DELETED (~72K lines).
  scenario/                        — DELETED.
  mission/                         — DELETED.
  game_core/                       — DELETED.
  content/                         — DELETED.
  narrative/                       — DELETED.
  model_backend/                   — DELETED.
  ascii_gen/                       — DELETED.
Cargo.toml (root)                  — MODIFIED:
                                     - workspace `members` adds `"crates/xtask"`.
                                     - `[[bin]] xtask` removed (xtask now its own crate).
                                     - `[dependencies]` pruned of any dep used only by deleted modules.
```

Audit confirmation: no other workspace crate (`engine`, `engine_data`, `engine_rules`, `engine_gpu`, `dsl_compiler`, `tactical_sim`, `viz`) depends on the legacy `src/` modules. They are downstream of the engine tree, not of the bootstrap `bevy_game/game` crate.

---

### Task 1: Create worktree + safety baseline

**Files:** none (validation + setup).

- [ ] **Step 1: Create branch + worktree.**

```bash
git worktree add .worktrees/legacy-src-sweep -b legacy-src-sweep
cd .worktrees/legacy-src-sweep
```

- [ ] **Step 2: Confirm clean baseline build + test.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS. If anything fails, fix or abort — we need a green baseline so post-sweep failures unambiguously identify the breakage.

If `cargo check` shows pre-existing errors in `src/bin/xtask/chronicle_cmd.rs` (per the prior dispatch-critics-hooks smoke-test note), this plan removes those errors *if* `chronicle_cmd.rs` survives the move — verify by reading the file. If the errors are in code that's about to be deleted (legacy module deps), they vanish in Tasks 3-4 and the baseline-test failure is acceptable as long as it's documented.

- [ ] **Step 3: Inventory exact survivor + casualty subcommands.**

```bash
# Survivors: depend ONLY on engine crate tree (engine, engine_data, engine_rules, dsl_compiler) + std/clap.
# Casualties: import from `game::ai::*`, `game::world_sim::*`, `game::scenario::*`,
# `game::mission::*`, `game::game_core::*`, `game::content::*`, `game::ascii_gen::*`,
# `game::model_backend::*`, `game::narrative::*`.

echo "=== survivor candidates ==="
for f in compile_dsl_cmd.rs chronicle_cmd.rs train_v6.rs map.rs capture.rs; do
    if grep -qE 'use (game|crate)::(ai|world_sim|scenario|mission|game_core|content|narrative|model_backend|ascii_gen)' "src/bin/xtask/$f"; then
        echo "DROP $f (touches legacy)"
    else
        echo "KEEP $f"
    fi
done

echo "=== casualty candidates ==="
for f in scenario_cmd.rs roomgen_cmd.rs model_cmd.rs content_gen_cmd.rs ascii_gen_cmd.rs champion_gen.rs world_sim_cmd.rs visualize_cmd.rs building_ai_cmd.rs; do
    echo "DELETE $f"
done

echo "=== oracle_cmd dir ==="
ls src/bin/xtask/oracle_cmd/
echo "(all DELETE; this dir is the RL oracle pipeline using game::ai::*)"
```

Expected output: 5 KEEPs, 0 DROPs, 9 DELETEs + the oracle_cmd directory. If any KEEP turns into DROP because of a hidden dep, fix the inventory and update the plan inline before proceeding.

- [ ] **Step 4: Commit a no-op breadcrumb so the rebase point is unambiguous.**

```bash
git commit --allow-empty -m "chore: legacy-src-sweep worktree baseline (Plan B3 Task 1)"
```

---

### Task 2: Create `crates/xtask/` skeleton + move surviving subcommands

**Files:**
- Create: `crates/xtask/Cargo.toml`
- Move: `src/bin/xtask/{main.rs,cli/,compile_dsl_cmd.rs,chronicle_cmd.rs,train_v6.rs,map.rs,capture.rs}` → `crates/xtask/src/`
- Modify: `crates/xtask/src/main.rs` — strip dispatch arms for casualty subcommands.
- Modify: `crates/xtask/src/cli/mod.rs` (and submodules) — strip clap subcommand variants for casualties.
- Modify: root `Cargo.toml` — add `crates/xtask` to workspace `members`; remove `[[bin]] xtask` block; remove deps used only by deleted xtask subcommands.

- [ ] **Step 1: Create `crates/xtask/Cargo.toml`.**

Audit which dependencies the surviving subcommands actually need:

```bash
grep -hE '^use ' src/bin/xtask/{compile_dsl_cmd.rs,chronicle_cmd.rs,train_v6.rs,map.rs,capture.rs} src/bin/xtask/cli/mod.rs src/bin/xtask/main.rs \
    | grep -vE '^use (std|crate|super|self)' \
    | sort -u
```

Expected non-std/crate imports include at least `clap`, `engine`, `dsl_compiler`, `glam`. Build the manifest accordingly:

```toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "xtask"
path = "src/main.rs"

[dependencies]
clap = { version = "4", features = ["derive"] }
engine = { path = "../engine" }
engine_data = { path = "../engine_data" }
engine_rules = { path = "../engine_rules" }
dsl_compiler = { path = "../dsl_compiler" }
glam = "0.29"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
tempfile = "3"  # required by compile-dsl --check (Plan B1 Task 11)
crossterm = "0.27"
# add others discovered in the audit above
```

Add only the deps that the audit confirms. Don't speculatively add deps "just in case."

- [ ] **Step 2: Make the directory.**

```bash
mkdir -p crates/xtask/src
```

- [ ] **Step 3: Move the survivor files using `git mv` (preserves history).**

```bash
git mv src/bin/xtask/main.rs           crates/xtask/src/main.rs
git mv src/bin/xtask/cli                crates/xtask/src/cli
git mv src/bin/xtask/compile_dsl_cmd.rs crates/xtask/src/compile_dsl_cmd.rs
git mv src/bin/xtask/chronicle_cmd.rs   crates/xtask/src/chronicle_cmd.rs
git mv src/bin/xtask/train_v6.rs        crates/xtask/src/train_v6.rs
git mv src/bin/xtask/map.rs             crates/xtask/src/map.rs
git mv src/bin/xtask/capture.rs         crates/xtask/src/capture.rs
```

- [ ] **Step 4: Strip dispatch arms for casualties from `crates/xtask/src/main.rs`.**

The current `main.rs` (read it) has arms for every subcommand. Use `Edit` to remove arms that reference deleted commands. The surviving `match args.command { ... }` should look approximately like:

```rust
match args.command {
    TaskCommand::Map(cmd) => match cmd.command {
        MapSubcommand::Voronoi(voronoi) => map::run_map_voronoi(voronoi),
    },
    TaskCommand::Capture(cmd) => match cmd.command {
        CaptureSubcommand::Windows(windows) => capture::run_capture_windows(windows),
        CaptureSubcommand::Dedupe(dedupe)   => capture::run_capture_dedupe(dedupe),
    },
    TaskCommand::TrainV6(cmd)   => train_v6::run_train_v6(cmd),
    TaskCommand::Chronicle(cmd) => chronicle_cmd::run_chronicle_cmd(cmd),
    TaskCommand::CompileDsl(cmd) => compile_dsl_cmd::run_compile_dsl(cmd),
}
```

(Match exact function names from the source — these are templates.) Drop:
- `mod scenario_cmd;`, `mod oracle_cmd;`, `mod roomgen_cmd;`, `mod model_cmd;`, `mod content_gen_cmd;`, `mod ascii_gen_cmd;`, `mod champion_gen;`, `mod world_sim_cmd;`, `mod visualize_cmd;`, `mod building_ai_cmd;` from the `mod` declarations.
- The `TaskCommand::Scenario`, `TaskCommand::Roomgen`, `TaskCommand::Model`, `TaskCommand::ContentGen`, `TaskCommand::AsciiGen`, `TaskCommand::ChampionGen`, `TaskCommand::WorldSim`, `TaskCommand::Visualize`, `TaskCommand::BuildingAi`, `TaskCommand::SynthAbilities`, `TaskCommand::Oracle*` arms from the dispatch match.

Also drop the inline `game::world_sim::ability_gen::dump_synthetic(count, seed, dsl);` call (`SynthAbilities` arm) — it imports `game::*`.

- [ ] **Step 5: Strip casualty variants from `crates/xtask/src/cli/`.**

The clap arg structs in `cli/mod.rs` (and possibly `cli/scenario.rs`, `cli/oracle.rs` etc.) define each subcommand's args. Variants for casualty subcommands must go too — otherwise clap still parses them and fails on the missing dispatch.

For each casualty subcommand:

```bash
ls crates/xtask/src/cli/
```

Delete (`git rm`) the per-command file if it exists, e.g.:

```bash
git rm crates/xtask/src/cli/scenario.rs       2>/dev/null || true
git rm crates/xtask/src/cli/oracle.rs         2>/dev/null || true
git rm crates/xtask/src/cli/roomgen.rs        2>/dev/null || true
git rm crates/xtask/src/cli/world_sim.rs      2>/dev/null || true
git rm crates/xtask/src/cli/visualize.rs      2>/dev/null || true
git rm crates/xtask/src/cli/building_ai.rs    2>/dev/null || true
git rm crates/xtask/src/cli/content_gen.rs    2>/dev/null || true
git rm crates/xtask/src/cli/ascii_gen.rs      2>/dev/null || true
git rm crates/xtask/src/cli/champion_gen.rs   2>/dev/null || true
git rm crates/xtask/src/cli/model.rs          2>/dev/null || true
```

Then in `cli/mod.rs`, remove `pub mod {scenario,oracle,roomgen,...}` declarations and remove their entries from the `enum TaskCommand` variants. Surviving variants are `Map`, `Capture`, `TrainV6`, `Chronicle`, `CompileDsl`.

- [ ] **Step 6: Update root `Cargo.toml`.**

In `Cargo.toml` at the workspace root:

Old `members`:
```toml
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_data", "crates/engine_rules", "crates/engine_gpu", "crates/viz", "crates/dsl_compiler"]
```

(Or whatever the post-B1 list is. If running B3 before B1 lands, substitute `engine_data` for `engine_data`.)

New:
```toml
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_data", "crates/engine_rules", "crates/engine_gpu", "crates/viz", "crates/dsl_compiler", "crates/xtask"]
```

Remove the `[[bin]]` block:

Old:
```toml
[[bin]]
name = "xtask"
path = "src/bin/xtask/main.rs"
```

→ delete the entire `[[bin]]` block.

Don't touch the root `[dependencies]` section yet — Task 4 prunes it after the legacy module deletion confirms which deps are dead.

- [ ] **Step 7: Build the new xtask crate.**

```bash
cargo build -p xtask
```

Expected: SUCCESS, possibly after fixing import paths inside the moved files. The most common issue is `use crate::cli::*` paths inside the moved subcommand files needing no change (they're still `crate::cli` because we moved the whole tree together) — but if `main.rs` says `mod cli;` and the moved files say `use super::cli::*;`, both should still resolve.

Common follow-up fixes:
- A moved file might `use game::SOMETHING` from a now-deleted module. If so, that subcommand should have been on the casualty list — re-classify it: either delete it from the survivor set (and update Step 4 + Step 5), or rewrite the import to use a non-legacy path.
- `train_v6.rs` only spawns Python via `Command::new`; verify it has no `game::*` imports.

- [ ] **Step 8: Run xtask --help to verify clap surface.**

```bash
cargo run --bin xtask -- --help
```

Expected: shows only the surviving subcommands.

- [ ] **Step 9: Run the engine-touching subcommands as smoke tests.**

```bash
cargo run --bin xtask -- compile-dsl --check
cargo run --bin xtask -- chronicle --help  # or any equivalent --help that doesn't require args
```

Expected: SUCCESS (or graceful "no chronicle args" message).

- [ ] **Step 10: Workspace test pass.**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS *for crates currently passing*. The root `bevy_game/game` crate may have failures because it still has `pub mod ai;` etc. in `lib.rs` referencing soon-to-be-deleted modules — those are addressed in Task 3.

Until Task 3 deletes the legacy modules, `cargo build` for the root `game` crate should still succeed (the modules are still there, just unused). If it fails, the cause is likely an old xtask dispatch arm we missed in Step 4 — re-audit.

- [ ] **Step 11: Commit.**

```bash
git add -A
git commit -m "refactor: move xtask to crates/xtask/; drop legacy-dependent subcommands (Spec B §8.4 Step 1)"
```

---

### Task 3: Delete legacy `src/{ai,world_sim,scenario,mission,game_core,content,narrative,model_backend,ascii_gen}`

**Files:**
- Delete: `src/ai/`, `src/world_sim/`, `src/scenario/`, `src/mission/`, `src/game_core/`, `src/content/`, `src/narrative/`, `src/model_backend/`, `src/ascii_gen/`
- Delete: `src/bin/` (xtask moved out in Task 2; nothing left).

- [ ] **Step 1: Delete the legacy module trees.**

```bash
git rm -r src/ai
git rm -r src/world_sim
git rm -r src/scenario
git rm -r src/mission
git rm -r src/game_core
git rm -r src/content
git rm -r src/narrative
git rm -r src/model_backend
git rm -r src/ascii_gen
```

- [ ] **Step 2: Delete the now-empty `src/bin/` directory.**

```bash
git rm -r src/bin   # was just src/bin/xtask/, all moved
```

- [ ] **Step 3: Confirm what remains in `src/`.**

```bash
find src/ -type f
```

Expected: only `src/lib.rs` and `src/rendering.rs`.

- [ ] **Step 4: Try a build to surface every reference into deleted code.**

```bash
cargo build --workspace 2>&1 | tee /tmp/b3-task3-build.log
```

Expected failures (these are the dependency-trail items the user flagged as "every reference to a deleted module is itself dead code that goes too"):
- Most are inside `src/lib.rs` (handled in Task 4).
- Possibly inside `src/rendering.rs` (if it references a deleted module). Audit with `grep`.

If a *workspace crate* (engine, engine_data, engine_rules, engine_gpu, dsl_compiler, tactical_sim, viz) fails because it referenced legacy code: that's a finding the AIS missed. **Stop and ask** — the crate either has a hidden dep on the legacy tree (re-classify it), or there's a re-export path that needs fixing. Do NOT delete from a workspace crate to make the build pass; investigate first.

- [ ] **Step 5: Audit `src/rendering.rs` for legacy refs.**

```bash
grep -E "(use|mod) (crate|self|super)::(ai|world_sim|scenario|mission|game_core|content|narrative|model_backend|ascii_gen|effects|sim|squad|goap|control|personality|roles|utility|phase|advanced|student|tooling)" src/rendering.rs
```

Expected: no matches. If any, the file referenced legacy via `crate::*` re-exports — rewrite to use `tactical_sim::*` directly (or, if the dep is structural, ask the user).

- [ ] **Step 6: Commit (build still failing is OK at this point — Task 4 trims lib.rs).**

```bash
git commit -m "chore: delete legacy src/{ai,world_sim,scenario,mission,game_core,content,narrative,model_backend,ascii_gen} (Spec B §8.4 Step 2)"
```

(Choose between commit-now-with-broken-build or commit-after-Task-4. Either is fine — but commit before Task 4 makes the deletion line obvious in `git log`. Per P9 ("Tasks Close With Verified Commit"), the verified commit comes at the end of Task 4 anyway.)

---

### Task 4: Trim `src/lib.rs` and root `Cargo.toml` `[dependencies]`

**Files:**
- Modify: `src/lib.rs`
- Modify: root `Cargo.toml`

- [ ] **Step 1: Replace `src/lib.rs` with the minimal surviving surface.**

If `rendering.rs` is the only surviving file: replace contents with a single line:

```rust
pub mod rendering;
```

(Drop the `#![allow(dead_code)]` if rendering itself doesn't trip the lint; restore if it does. Drop every `pub use` re-export of the deleted modules — they all reference dead paths now.)

- [ ] **Step 2: Build the root crate to confirm `lib.rs` is clean.**

```bash
cargo build -p game
```

Expected: SUCCESS. If `rendering.rs` references a deleted module (caught in Task 3 Step 5), the failure is here — fix in `rendering.rs` per that finding.

- [ ] **Step 3: Audit root `Cargo.toml` `[dependencies]` for orphans.**

```bash
# Show every dep listed in root Cargo.toml.
sed -n '/^\[dependencies\]/,/^\[/p' Cargo.toml | grep -E '^\w[a-zA-Z0-9_-]*\s*='
```

For each listed dep, check if `src/rendering.rs` (or any surviving root content) actually uses it:

```bash
for dep in $(sed -n '/^\[dependencies\]/,/^\[/p' Cargo.toml | grep -E '^\w' | awk -F= '{print $1}' | tr -d ' ' | grep -v '^\['); do
    if ! grep -qrE "use $dep|extern crate $dep" src/; then
        echo "ORPHAN: $dep — listed in root Cargo.toml but unused in src/."
    fi
done
```

For each ORPHAN, remove the dep line from `[dependencies]`. Common candidates: `burn`, `tch`, `winit`, `voxel_engine`, `serde_json`, `tokio`, `rand`, `glam` (verify per-dep before removing). Don't delete deps used by `[[bin]]` blocks (the xtask bin is gone now, so its deps are orphaned through the root) — but xtask's deps live in `crates/xtask/Cargo.toml` after Task 2.

Conservative approach: leave deps that *might* still be referenced by `[features]` hooks (`gpu = [...]`, `app = [...]`); aggressive approach: delete everything that isn't grepped for. Default to conservative — orphan deps cost some compile time but don't break anything.

- [ ] **Step 4: Optional — delete `[features]` blocks that reference deleted deps.**

If a feature flag like `gpu = ["dep:engine_gpu", "engine_gpu/gpu"]` only made sense for the deleted xtask subcommands (e.g., `--gpu` on `xtask chronicle`), it may still be valid (`xtask chronicle` survives in `crates/xtask/`). Check:

```bash
grep -E 'feature\s*=' crates/xtask/src/chronicle_cmd.rs
```

If chronicle uses the `gpu` feature, the feature block stays; the deps it gates may need to move to `crates/xtask/Cargo.toml`. If chronicle doesn't use it, the feature block can go.

- [ ] **Step 5: Workspace build + test.**

```bash
cargo clean
cargo build --workspace
cargo test --workspace
```

Expected: SUCCESS. If a test fails because it references a deleted module — that test was inside the deleted module tree and got `git rm`'d in Task 3, so this scenario shouldn't fire. If it does, investigate; do not paper over.

- [ ] **Step 6: Commit.**

```bash
git add -A
git commit -m "chore: trim src/lib.rs to rendering only; drop orphaned root deps (Spec B §8.4 Step 3)"
```

---

### Task 5: Final workspace verification + AIS tick

**Files:** none (validation + bookkeeping).

- [ ] **Step 1: Clean rebuild.**

```bash
cargo clean
cargo build --workspace
```

Expected: SUCCESS, no warnings about dead code in deleted modules (they're gone).

- [ ] **Step 2: Full test pass.**

```bash
cargo test --workspace
```

Expected: SUCCESS.

- [ ] **Step 3: Lines-of-code measurement (sanity check).**

```bash
echo "remaining root src/:"
find src/ -name '*.rs' -type f | xargs wc -l 2>/dev/null | tail -1
echo "crates/xtask:"
find crates/xtask -name '*.rs' -type f | xargs wc -l 2>/dev/null | tail -1
echo "git diff --stat (sweep total):"
git diff --stat HEAD~5 -- src/ crates/xtask/ Cargo.toml | tail -3
```

Expected: src/ is small (only rendering + lib.rs); crates/xtask is small (only the survivors); the diff stat shows tens of thousands of lines deleted.

- [ ] **Step 4: Run the surviving xtask subcommands as smoke tests.**

```bash
cargo run --bin xtask -- --help
cargo run --bin xtask -- compile-dsl --check
cargo run --bin xtask -- chronicle --help    # or another no-arg variant
```

Expected: clap shows only survivor subcommands; `compile-dsl --check` reports "match" (assuming Plan B1's `--check` flag has landed; if not, that's a B1 dep — note in commit message and verify when B1 merges).

- [ ] **Step 5: Confirm no dispatch-critics surprise.**

The Spec D-amendment dispatch-critics gate runs on engine-directory edits. This plan touches `src/` (deleting) and `crates/xtask/` (creating) — neither is `crates/engine*`. The pre-commit gate should be a no-op for these commits.

```bash
.githooks/pre-commit && echo "pre-commit OK"
```

Expected: OK.

If the gate fires unexpectedly (e.g., because deleting `src/world_sim` triggers something), read the message and address. Don't `--no-verify`.

- [ ] **Step 6: Tick AIS post-design checkbox.**

Edit this plan file: `[ ] AIS reviewed post-design` → `[x] AIS reviewed post-design`. Append a one-line scope note: "Final scope: ~85K lines deleted from src/; xtask shrunk to 5 surviving subcommands in crates/xtask/."

- [ ] **Step 7: Final commit.**

```bash
git add -A
git commit -m "chore: tick AIS post-design checkbox + scope note for Plan B3"
```

---

## Sequencing summary

| Task | What | Depends on |
|---|---|---|
| 1 | Worktree + baseline | — |
| 2 | Create crates/xtask/ + move survivors | 1 |
| 3 | Delete legacy src/{ai,world_sim,...} | 2 |
| 4 | Trim src/lib.rs + Cargo.toml deps | 3 |
| 5 | Final verification + AIS tick | 4 |

Strict serial order. Each task ends with the project building and (for Tasks 2, 4, 5) tests green. Task 3 commits with a possibly-broken root build because Task 4 is the cleanup that restores green.

## Coordination with parallel work

**Plan B1 (engine crate restructure):** Independent. Both can run in parallel worktrees. Conflict surface: `Cargo.toml` workspace `members` (additive merge — both add new members; order doesn't matter). Whichever lands first wins; the other rebases trivially.

**Spec D-amendment (dispatch-critics + pre-commit gate):** Already operational. Expect the `PreToolUse` hook to fire on edits inside `crates/` (engine path), but B3 doesn't edit `crates/engine*` — the hook should idle. The pre-commit gate likewise should be a no-op for B3 commits.

**Cross-plan dependency on B1's `compile-dsl --check`:** Task 5 Step 4 calls `compile-dsl --check`. If B1 hasn't landed yet, that step exits non-zero (flag doesn't exist). Treat as a follow-up: rerun the smoke once B1 merges. Document in the merge-back commit.
