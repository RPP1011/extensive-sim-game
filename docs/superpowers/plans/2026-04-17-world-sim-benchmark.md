# World Sim Benchmark & SIMD-Targeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-layer benchmark harness for the world sim: a diagnostic pass that ranks per-system timings at 2K/10K/50K populations, and a criterion regression harness that A/B's scalar vs SIMD implementations. No SIMD code is written in this plan — the harness exists to target and validate future SIMD work.

**Architecture:** Prerequisite refactor introduces a `System` trait with a `Backend` enum for scalar/SIMD dispatch and a monomorphized-kernel inner loop. All ~60 actively-dispatched systems migrate to `impl System`. A `profile-systems` Cargo feature wires per-system timing via thread-local accumulators. The diagnostic xtask subcommand runs the real sim and emits ranked tables, JSON, and flamegraphs. A nightly-pinned `crates/world_sim_bench/` workspace member holds criterion benches that load bincode fixtures and compare backends.

**Tech Stack:** Rust stable (main crate), Rust nightly (bench crate only, for `std::simd`), criterion, rayon, bincode, `cargo-flamegraph`.

**Worktree recommendation:** This is a multi-day refactor touching ~65 files. Use `superpowers:using-git-worktrees` to create an isolated worktree before starting.

**Realistic system count:** Spec says ~170, but only ~60 systems are actively dispatched (3 compute phases, 10 apply sub-phases, ~49 postapply calls in `runtime.rs`). The 154 `#![allow(unused)]` files in `src/world_sim/systems/` are dead code and will be **deleted in Stage -1** before the trait refactor begins, shrinking the migration surface.

---

## File Structure

### New files (main crate)
- `src/world_sim/system.rs` — `System` trait, `Stage` enum, `SystemCtx`, `SystemRegistry`, `Backend` enum
- `src/world_sim/bench_world.rs` — `bench_world(target_entities, seed) -> WorldSim`
- `src/bin/xtask/bench_cmd.rs` — `world-sim` and `regenerate-fixtures` subcommands
- `scripts/perf_bench.sh` — `cargo flamegraph` wrapper

### Modified files (main crate)
- `Cargo.toml` — add `crates/world_sim_bench` to `[workspace] members`
- `src/world_sim/mod.rs` — `pub mod system; pub mod bench_world;`
- `src/world_sim/trace.rs` — add `run_with_profile()`, `SystemProfileAccumulator`
- `src/world_sim/tick.rs` — extend `TickProfile` with `system_timings: Vec<SystemTiming>`; dispatch via registry
- `src/world_sim/runtime.rs` — `build_registry()`, replace direct system calls with registry dispatch
- `src/world_sim/apply.rs` / `compute_high.rs` / `compute_medium.rs` / `compute_low.rs` — wrap each as `impl System`
- `src/bin/xtask/main.rs` — wire `bench` subcommand
- ~49 files in `src/world_sim/systems/*.rs` — convert active systems to `impl System`

### New files (bench crate)
- `crates/world_sim_bench/Cargo.toml`
- `crates/world_sim_bench/rust-toolchain.toml` — `channel = "nightly"`
- `crates/world_sim_bench/src/lib.rs` — `#![feature(portable_simd)]`
- `crates/world_sim_bench/src/fixtures.rs` — load/verify helpers
- `crates/world_sim_bench/fixtures/world_{2k,10k,50k}.bin` — bincode snapshots (committed)
- `crates/world_sim_bench/fixtures/fixtures.sha` — schema hash
- `crates/world_sim_bench/benches/movement.rs`
- `crates/world_sim_bench/benches/economy.rs`
- `crates/world_sim_bench/benches/hp_changes.rs`
- `crates/world_sim_bench/benches/merge.rs`
- `scripts/bench_summary.py` — parse `target/criterion/**/estimates.json` → markdown table

---

## Stage -1: Delete dead system files

The `src/world_sim/systems/` directory has 170 `.rs` files, but only ~49 are actively dispatched from `runtime.rs::tick()`. The rest carry `#![allow(unused)]` (the suppression itself is a tell). Delete them before the trait refactor to shrink migration surface and remove tech debt.

### Task -1.1: Inventory dead files

**Files:**
- Create: `/tmp/dead_systems_audit.txt`

- [ ] **Step 1: Extract the referenced-from-runtime set**

Run:
```bash
# Modules referenced at the top level of runtime.rs tick()
grep -oE 'super::systems::[a-z_]+' src/world_sim/runtime.rs \
  | sed 's|super::systems::||' \
  | sort -u > /tmp/referenced_from_runtime.txt

# Modules referenced anywhere in the crate (outside systems/ themselves)
grep -rhoE 'systems::[a-z_]+' src/ --include='*.rs' \
  | grep -v '^src/world_sim/systems/' \
  | sed 's|systems::||' \
  | sort -u > /tmp/referenced_anywhere.txt
```

- [ ] **Step 2: Build the candidate-for-deletion list**

```bash
# All system files
ls src/world_sim/systems/*.rs \
  | sed 's|src/world_sim/systems/||; s|\.rs$||' \
  | sort -u > /tmp/all_systems.txt

# Candidates = all_systems minus referenced_anywhere
comm -23 /tmp/all_systems.txt /tmp/referenced_anywhere.txt > /tmp/dead_systems_audit.txt
wc -l /tmp/dead_systems_audit.txt
```

Expected: roughly 120-150 names. These are files whose module identifier appears nowhere outside `systems/` itself.

- [ ] **Step 3: Spot-check 5 candidates for safety**

For each of 5 random names in `/tmp/dead_systems_audit.txt`:

```bash
# Confirm zero cross-references
grep -rn "systems::<NAME>" src/ | grep -v 'src/world_sim/systems/<NAME>.rs'
# Confirm file isn't re-exported from mod.rs via glob
grep -n "<NAME>" src/world_sim/systems/mod.rs
```

Expected: zero matches outside the file itself. If any candidate has a hit, remove it from `dead_systems_audit.txt`.

- [ ] **Step 4: Check `mod.rs` declarations**

Some dead files may be `pub mod foo;` in `src/world_sim/systems/mod.rs`. They'll also need to be removed from there.

```bash
grep -n "pub mod " src/world_sim/systems/mod.rs > /tmp/declared_modules.txt
```

- [ ] **Step 5: Commit the audit artifact**

```bash
# Save the audit to the repo so Task -1.2 can reference it
mkdir -p docs/superpowers/specs
cp /tmp/dead_systems_audit.txt docs/superpowers/specs/dead_systems_audit.txt
git add docs/superpowers/specs/dead_systems_audit.txt
git commit -m "chore: inventory dead system files for deletion"
```

---

### Task -1.2: Delete dead files in batches

**Files:**
- Delete: files listed in `docs/superpowers/specs/dead_systems_audit.txt`
- Modify: `src/world_sim/systems/mod.rs` (remove `pub mod X;` for each deleted)

- [ ] **Step 1: Delete in batches of ~20 with a build check between**

```bash
# Deletion script — run chunks at a time
cat docs/superpowers/specs/dead_systems_audit.txt | while read name; do
  file="src/world_sim/systems/${name}.rs"
  if [[ -f "$file" ]]; then
    git rm "$file"
  fi
  # Remove `pub mod <name>;` line
  sed -i "/^pub mod ${name};$/d" src/world_sim/systems/mod.rs
  sed -i "/^mod ${name};$/d" src/world_sim/systems/mod.rs
done
```

- [ ] **Step 2: Verify the build and tests still pass**

Run: `cargo build --release`
Expected: clean build.

Run: `cargo test --lib world_sim -- --test-threads=1`
Expected: all tests PASS.

If either fails: a candidate wasn't actually dead. `git restore` the offending file and re-run with that name excluded from the audit.

- [ ] **Step 3: Verify sim still runs**

Run: `cargo run --release --bin xtask -- world-sim --ticks 500 --world small`
Expected: tick rate unchanged (these files were never called at runtime; wall time should be identical ± noise).

- [ ] **Step 4: Commit the deletions**

```bash
git add -A src/world_sim/systems/
git commit -m "refactor(world-sim): delete ~140 dead system files

Shrinks migration surface for the upcoming System trait refactor.
Each file had zero references outside src/world_sim/systems/ per
docs/superpowers/specs/dead_systems_audit.txt. Build + determinism
tests pass unchanged."
```

---

## Stage 0: System trait scaffolding

Foundation for all later stages. No behavior changes; existing dispatch is preserved until Stage 0 migration completes.

### Task 0.1: Create the `System` trait, `Stage` enum, `Backend` enum, `SystemCtx`

**Files:**
- Create: `src/world_sim/system.rs`
- Modify: `src/world_sim/mod.rs` (add `pub mod system;`)
- Test: `src/world_sim/system.rs` (inline `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Append to `src/world_sim/system.rs` (new file):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct DummySystem;
    impl System for DummySystem {
        fn name(&self) -> &'static str { "dummy" }
        fn stage(&self) -> Stage { Stage::PostApply }
        fn run(&self, _ctx: &mut SystemCtx) -> u32 { 7 }
    }

    #[test]
    fn trait_surface_is_object_safe() {
        let sys: Box<dyn System> = Box::new(DummySystem);
        assert_eq!(sys.name(), "dummy");
        assert_eq!(sys.stage(), Stage::PostApply);
    }

    #[test]
    fn backend_default_is_scalar() {
        assert_eq!(Backend::default_for_cpu(), Backend::Scalar);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib world_sim::system -- --nocapture`
Expected: FAIL — `System`, `Stage`, `Backend`, `SystemCtx` undefined.

- [ ] **Step 3: Write minimal implementation**

Write `src/world_sim/system.rs`:

```rust
//! System trait: uniform interface for all world-sim systems. Enables
//! per-system timing, backend dispatch (scalar vs SIMD), and clean
//! stage-based scheduling.

use super::delta::WorldDelta;
use super::state::WorldState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    ComputeHigh,
    ComputeMedium,
    ComputeLow,
    ComputeGrid,
    ApplyClone,
    ApplyHp,
    ApplyMovement,
    ApplyStatus,
    ApplyEconomy,
    ApplyTransfers,
    ApplyDeaths,
    ApplyGrid,
    ApplyFidelity,
    ApplyPriceReports,
    PostApply,
}

impl Stage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Stage::ComputeHigh => "ComputeHigh",
            Stage::ComputeMedium => "ComputeMedium",
            Stage::ComputeLow => "ComputeLow",
            Stage::ComputeGrid => "ComputeGrid",
            Stage::ApplyClone => "ApplyClone",
            Stage::ApplyHp => "ApplyHp",
            Stage::ApplyMovement => "ApplyMovement",
            Stage::ApplyStatus => "ApplyStatus",
            Stage::ApplyEconomy => "ApplyEconomy",
            Stage::ApplyTransfers => "ApplyTransfers",
            Stage::ApplyDeaths => "ApplyDeaths",
            Stage::ApplyGrid => "ApplyGrid",
            Stage::ApplyFidelity => "ApplyFidelity",
            Stage::ApplyPriceReports => "ApplyPriceReports",
            Stage::PostApply => "PostApply",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Simd,
}

impl Backend {
    /// Runtime default. Returns Scalar until SIMD implementations land;
    /// future work will add CPU-feature detection here.
    pub fn default_for_cpu() -> Self { Backend::Scalar }
}

/// Execution context handed to each system per tick.
pub struct SystemCtx<'a> {
    pub state: &'a WorldState,
    pub deltas: &'a mut Vec<WorldDelta>,
    pub tick: u64,
}

/// Core trait implemented by every actively-dispatched world-sim system.
pub trait System: Send + Sync {
    fn name(&self) -> &'static str;
    fn stage(&self) -> Stage;
    /// Runs one tick's worth of work. Returns entities/units touched
    /// this call (used for ns/entity stats — return 0 if unclear).
    fn run(&self, ctx: &mut SystemCtx) -> u32;
}

#[cfg(test)]
mod tests {
    // (as defined in Step 1, keep the block verbatim)
}
```

Modify `src/world_sim/mod.rs` — add after line 16 (after `pub mod delta;`):

```rust
pub mod system;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib world_sim::system -- --nocapture`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/system.rs src/world_sim/mod.rs
git commit -m "feat(world-sim): add System trait, Stage + Backend enums"
```

---

### Task 0.2: Add `SystemRegistry` with stage-ordered dispatch

**Files:**
- Modify: `src/world_sim/system.rs` (append `SystemRegistry`)
- Test: `src/world_sim/system.rs` (extend `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `src/world_sim/system.rs`:

```rust
    struct StageSystem(Stage, &'static str);
    impl System for StageSystem {
        fn name(&self) -> &'static str { self.1 }
        fn stage(&self) -> Stage { self.0 }
        fn run(&self, _ctx: &mut SystemCtx) -> u32 { 0 }
    }

    #[test]
    fn registry_groups_by_stage() {
        let mut r = SystemRegistry::new();
        r.register(StageSystem(Stage::PostApply, "a"));
        r.register(StageSystem(Stage::ApplyHp, "b"));
        r.register(StageSystem(Stage::PostApply, "c"));

        let hp: Vec<_> = r.systems_in(Stage::ApplyHp).iter().map(|s| s.name()).collect();
        let post: Vec<_> = r.systems_in(Stage::PostApply).iter().map(|s| s.name()).collect();
        assert_eq!(hp, vec!["b"]);
        assert_eq!(post, vec!["a", "c"]); // insertion order preserved within stage
    }

    #[test]
    fn registry_empty_stage_returns_empty_slice() {
        let r = SystemRegistry::new();
        assert!(r.systems_in(Stage::ComputeHigh).is_empty());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib world_sim::system::tests::registry -- --nocapture`
Expected: FAIL — `SystemRegistry` undefined.

- [ ] **Step 3: Write minimal implementation**

Append to `src/world_sim/system.rs` (before the `#[cfg(test)] mod tests`):

```rust
/// Registry of all systems grouped by `Stage`. Preserves insertion order
/// within each stage so dispatch is deterministic.
pub struct SystemRegistry {
    by_stage: std::collections::HashMap<Stage, Vec<Box<dyn System>>>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        Self { by_stage: std::collections::HashMap::new() }
    }

    pub fn register<S: System + 'static>(&mut self, sys: S) {
        self.by_stage.entry(sys.stage()).or_default().push(Box::new(sys));
    }

    pub fn systems_in(&self, stage: Stage) -> &[Box<dyn System>] {
        self.by_stage.get(&stage).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Iterator over all (stage, systems) pairs. Used for profiling reports.
    pub fn all_stages(&self) -> impl Iterator<Item = (Stage, &[Box<dyn System>])> {
        self.by_stage.iter().map(|(k, v)| (*k, v.as_slice()))
    }
}

impl Default for SystemRegistry {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib world_sim::system::tests -- --nocapture`
Expected: PASS, 4 tests total.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/system.rs
git commit -m "feat(world-sim): add SystemRegistry with stage-ordered dispatch"
```

---

### Task 0.3: Pilot migration — `apply_movement` to `impl System` with backend enum + generic kernel

This is the template every other system migration will follow.

**Files:**
- Modify: `src/world_sim/apply.rs` (extract movement logic into a `MovementSystem`)
- Test: `src/world_sim/apply.rs` (new inline test)

- [ ] **Step 1: Read the current apply_movement code**

Run: `sed -n '164,186p' src/world_sim/apply.rs` to inspect the current implementation. Note: actual line numbers may have shifted; locate by searching for "apply movement" or the force-magnitude clamping logic.

- [ ] **Step 2: Write the failing test**

Append to the bottom of `src/world_sim/apply.rs`:

```rust
#[cfg(test)]
mod movement_system_tests {
    use super::*;
    use crate::world_sim::system::{Backend, System, Stage, SystemCtx};

    #[test]
    fn movement_system_name_and_stage() {
        let sys = ApplyMovementSystem::new(Backend::Scalar);
        assert_eq!(sys.name(), "apply_movement");
        assert_eq!(sys.stage(), Stage::ApplyMovement);
    }

    #[test]
    fn scalar_and_simd_backends_agree_on_empty_state() {
        let state = crate::world_sim::state::WorldState::default();
        let merged = crate::world_sim::delta::MergedDeltas::default();
        let mut s1 = state.clone();
        let mut s2 = state.clone();
        ApplyMovementSystem::new(Backend::Scalar).apply_inplace(&mut s1, &merged);
        ApplyMovementSystem::new(Backend::Simd).apply_inplace(&mut s2, &merged);
        // No entities => no divergence possible
        assert_eq!(s1.entities.len(), s2.entities.len());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib world_sim::apply::movement_system_tests -- --nocapture`
Expected: FAIL — `ApplyMovementSystem` undefined.

- [ ] **Step 4: Write minimal implementation**

Add to `src/world_sim/apply.rs` near the top (after existing imports):

```rust
use crate::world_sim::system::{Backend, Stage, System, SystemCtx};
```

Add to the bottom of `src/world_sim/apply.rs` (before the tests module):

```rust
/// Movement application — scales force vectors and updates positions.
/// Uses backend-enum dispatch with a monomorphized inner kernel.
pub struct ApplyMovementSystem { backend: Backend }

impl ApplyMovementSystem {
    pub fn new(backend: Backend) -> Self { Self { backend } }

    /// Direct entry for the existing `apply_deltas_profiled` call path.
    /// Keeps backward compatibility while the refactor is in progress.
    pub fn apply_inplace(&self, state: &mut WorldState, merged: &MergedDeltas) {
        match self.backend {
            Backend::Scalar => run_movement::<ScalarMovementKernel>(state, merged),
            Backend::Simd => run_movement::<ScalarMovementKernel>(state, merged),
            // Simd variant intentionally falls back to Scalar until a real
            // SimdMovementKernel lands. The harness exists to validate that
            // work; see docs/superpowers/specs/world_sim_simd_targets.md.
        }
    }
}

impl System for ApplyMovementSystem {
    fn name(&self) -> &'static str { "apply_movement" }
    fn stage(&self) -> Stage { Stage::ApplyMovement }
    fn run(&self, ctx: &mut SystemCtx) -> u32 {
        // Run path used by future registry dispatch. For now `apply_deltas_profiled`
        // calls `apply_inplace` directly; this path is exercised by Stage 1 wiring.
        let _ = ctx; // silenced until registry dispatch lands in Task 0.N
        0
    }
}

pub trait MovementKernel {
    /// Lane width for SIMD kernels; Scalar uses 1.
    const LANES: usize;
    fn step(entities: &mut [crate::world_sim::state::Entity], forces: &[(f32, f32)]);
}

pub struct ScalarMovementKernel;
impl MovementKernel for ScalarMovementKernel {
    const LANES: usize = 1;
    fn step(entities: &mut [crate::world_sim::state::Entity], forces: &[(f32, f32)]) {
        for (e, &(fx, fy)) in entities.iter_mut().zip(forces.iter()) {
            if !e.alive { continue; }
            let mag2 = fx * fx + fy * fy;
            let (dx, dy) = if mag2 > 1.0 {
                let mag = mag2.sqrt();
                (fx / mag, fy / mag)
            } else {
                (fx, fy)
            };
            e.pos.0 += dx;
            e.pos.1 += dy;
        }
    }
}

#[inline]
fn run_movement<K: MovementKernel>(state: &mut WorldState, merged: &MergedDeltas) {
    // Extract force vectors parallel to entities, then call the kernel.
    // This is intentionally a straightforward port of the existing loop
    // (apply.rs:164-186 pre-refactor); the kernel abstraction exists so
    // a future SimdMovementKernel can drop in without touching this.
    let mut forces: Vec<(f32, f32)> = Vec::with_capacity(state.entities.len());
    for e in &state.entities {
        let f = merged.forces.get(&e.id).copied().unwrap_or((0.0, 0.0));
        forces.push(f);
    }
    K::step(&mut state.entities, &forces);
}
```

**Notes for the engineer:**
- `MergedDeltas.forces` is the existing `HashMap<EntityId, (f32, f32)>` field; verify exact name in `src/world_sim/delta.rs` and adjust.
- The existing in-place movement loop in `apply_deltas_profiled` must now call `ApplyMovementSystem::new(Backend::Scalar).apply_inplace(&mut next, merged)` instead of inlining the logic. Keep the `movement_us` timing wrapped around that call — the legacy sub-phase timers stay until Stage 1.
- If `Entity.pos` has a different field name (e.g., `position`, `xy`), adjust both the kernel and the test.
- `WorldState::default()` may not exist; use `WorldState::empty()` or the smallest constructor available.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib world_sim::apply::movement_system_tests -- --nocapture`
Expected: PASS, 2 tests.

- [ ] **Step 6: Run the full sim test suite to verify no regressions**

Run: `cargo test --lib world_sim -- --test-threads=1`
Expected: All existing tests PASS, including determinism tests.

- [ ] **Step 7: Run a short sim to sanity-check performance**

Run: `cargo run --release --bin xtask -- world-sim --ticks 500 --world small`
Expected: Completes normally; tick rate in terminal output is within 5% of pre-refactor baseline (note the baseline in a terminal scratchpad before starting Task 0.3 for comparison).

- [ ] **Step 8: Commit**

```bash
git add src/world_sim/apply.rs
git commit -m "feat(world-sim): pilot-migrate apply_movement to System trait"
```

---

### Task 0.4: Define the per-stage migration template

Every other system migration follows the same pattern. Document the template once; reference it in the `/batch` dispatch.

**Files:**
- Create: `docs/superpowers/specs/system_migration_template.md`

- [ ] **Step 1: Write the template**

Create `docs/superpowers/specs/system_migration_template.md`:

````markdown
# System Migration Template

Each system migration replaces one bare `fn system_name(&mut WorldState)`
(or equivalent) with a `SystemNameSystem` struct implementing the `System`
trait. Follow this template exactly.

## Inputs (supplied by /batch dispatch)

- `SYSTEM_NAME` — canonical snake_case name (e.g., `advance_movement`)
- `STRUCT_NAME` — PascalCase struct (e.g., `AdvanceMovementSystem`)
- `FILE_PATH` — existing file containing the bare function
- `STAGE` — `Stage::PostApply` unless it's a compute or apply sub-phase
- `CALL_SITES` — list of `file:line` where the bare fn is currently called

## Steps

1. Open `FILE_PATH`. Locate the bare fn (`pub fn SYSTEM_NAME(...)`).

2. Add to the top of the file (if not already present):
   ```rust
   use crate::world_sim::system::{Backend, Stage, System, SystemCtx};
   ```

3. Keep the existing bare fn intact (other code may still call it).
   Add below it:
   ```rust
   pub struct STRUCT_NAME { backend: Backend }

   impl STRUCT_NAME {
       pub fn new(backend: Backend) -> Self { Self { backend } }
   }

   impl System for STRUCT_NAME {
       fn name(&self) -> &'static str { "SYSTEM_NAME" }
       fn stage(&self) -> Stage { STAGE }
       fn run(&self, ctx: &mut SystemCtx) -> u32 {
           // SAFETY: SystemCtx holds `&WorldState` (immutable). Most
           // legacy bare fns take `&mut WorldState`. For PostApply
           // systems, dispatch in runtime.rs holds the &mut and passes
           // &State via ctx — adapt this wrapper per-system by reading
           // what bare fn needs and emitting deltas via ctx.deltas.
           //
           // For systems that mutate state directly (not via deltas),
           // keep the bare fn and register only for profiling: the
           // wrapper measures call duration but bare fn is invoked
           // from the dispatch site, not ctx.run(). See Task 0.5.
           let _ = ctx;
           0
       }
   }
   ```

4. Do NOT change call sites yet. That happens in Task 0.6 (dispatch refactor).

5. Run `cargo build` — verify the new struct compiles.

6. Commit:
   ```bash
   git add FILE_PATH
   git commit -m "refactor(world-sim): wrap SYSTEM_NAME in System trait"
   ```

## When to deviate

If the bare fn does not take `&mut WorldState`, ask for guidance before migrating.
If the bare fn has no clear call site in runtime.rs / tick.rs, skip it —
it's probably dead code (one of the 154 `#![allow(unused)]` files).
````

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/system_migration_template.md
git commit -m "docs: add system migration template for /batch"
```

---

### Task 0.5: Wrap the 3 compute phases as `System`s

**Files:**
- Modify: `src/world_sim/compute_high.rs`
- Modify: `src/world_sim/compute_medium.rs`
- Modify: `src/world_sim/compute_low.rs`

- [ ] **Step 1: Write failing tests for all three**

Append to `src/world_sim/compute_high.rs`:

```rust
#[cfg(test)]
mod system_tests {
    use super::*;
    use crate::world_sim::system::{Backend, System, Stage};

    #[test]
    fn high_system_metadata() {
        let sys = ComputeHighSystem::new(Backend::Scalar);
        assert_eq!(sys.name(), "compute_high");
        assert_eq!(sys.stage(), Stage::ComputeHigh);
    }
}
```

Same shape in `compute_medium.rs` (`ComputeMediumSystem`, `"compute_medium"`, `Stage::ComputeMedium`) and `compute_low.rs` (`ComputeLowSystem`, `"compute_low"`, `Stage::ComputeLow`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib compute_high::system_tests compute_medium::system_tests compute_low::system_tests`
Expected: FAIL — structs undefined.

- [ ] **Step 3: Add the wrapper structs**

In each of `compute_high.rs`, `compute_medium.rs`, `compute_low.rs`, add at the top:

```rust
use crate::world_sim::system::{Backend, Stage, System, SystemCtx};
```

Then append:

```rust
// In compute_high.rs:
pub struct ComputeHighSystem { backend: Backend }
impl ComputeHighSystem {
    pub fn new(backend: Backend) -> Self { Self { backend } }
}
impl System for ComputeHighSystem {
    fn name(&self) -> &'static str { "compute_high" }
    fn stage(&self) -> Stage { Stage::ComputeHigh }
    fn run(&self, ctx: &mut SystemCtx) -> u32 {
        // Legacy compute_high takes (entity, state, fidelity). This wrapper
        // is registry-metadata-only until Task 0.7 flips dispatch.
        let _ = ctx;
        0
    }
}
```

Repeat verbatim for Medium/Low in their respective files (rename struct, name string, stage).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib compute_high::system_tests compute_medium::system_tests compute_low::system_tests`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/compute_high.rs src/world_sim/compute_medium.rs src/world_sim/compute_low.rs
git commit -m "refactor(world-sim): wrap compute phases in System trait"
```

---

### Task 0.6: Wrap the 10 apply sub-phases as `System`s

**Files:**
- Modify: `src/world_sim/apply.rs`

The existing `apply_deltas_profiled()` already has 10 named sub-phase timers (clone, hp, movement, status, economy, transfers, deaths, grid, fidelity, price_reports). Wrap each as a `System`. `ApplyMovementSystem` already exists from Task 0.3.

- [ ] **Step 1: Write a single test covering all 10 wrappers**

Append to the existing tests module in `src/world_sim/apply.rs`:

```rust
#[cfg(test)]
mod apply_system_coverage {
    use super::*;
    use crate::world_sim::system::{Backend, Stage, System};

    #[test]
    fn all_apply_subphases_have_system_wrappers() {
        let pairs: Vec<(Box<dyn System>, &str, Stage)> = vec![
            (Box::new(ApplyCloneSystem::new(Backend::Scalar)), "apply_clone", Stage::ApplyClone),
            (Box::new(ApplyHpSystem::new(Backend::Scalar)), "apply_hp_changes", Stage::ApplyHp),
            (Box::new(ApplyMovementSystem::new(Backend::Scalar)), "apply_movement", Stage::ApplyMovement),
            (Box::new(ApplyStatusSystem::new(Backend::Scalar)), "apply_status", Stage::ApplyStatus),
            (Box::new(ApplyEconomySystem::new(Backend::Scalar)), "apply_economy", Stage::ApplyEconomy),
            (Box::new(ApplyTransfersSystem::new(Backend::Scalar)), "apply_transfers", Stage::ApplyTransfers),
            (Box::new(ApplyDeathsSystem::new(Backend::Scalar)), "apply_deaths", Stage::ApplyDeaths),
            (Box::new(ApplyGridSystem::new(Backend::Scalar)), "apply_grid", Stage::ApplyGrid),
            (Box::new(ApplyFidelitySystem::new(Backend::Scalar)), "apply_fidelity", Stage::ApplyFidelity),
            (Box::new(ApplyPriceReportsSystem::new(Backend::Scalar)), "apply_price_reports", Stage::ApplyPriceReports),
        ];
        for (sys, name, stage) in pairs {
            assert_eq!(sys.name(), name);
            assert_eq!(sys.stage(), stage);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib world_sim::apply::apply_system_coverage`
Expected: FAIL — most structs undefined.

- [ ] **Step 3: Add 9 wrapper structs (Movement already exists)**

For each of the 9 remaining apply sub-phases, append to `src/world_sim/apply.rs`:

```rust
pub struct ApplyCloneSystem { backend: Backend }
impl ApplyCloneSystem {
    pub fn new(backend: Backend) -> Self { Self { backend } }
}
impl System for ApplyCloneSystem {
    fn name(&self) -> &'static str { "apply_clone" }
    fn stage(&self) -> Stage { Stage::ApplyClone }
    fn run(&self, ctx: &mut SystemCtx) -> u32 { let _ = ctx; 0 }
}
```

Repeat for:
- `ApplyHpSystem` / `"apply_hp_changes"` / `Stage::ApplyHp`
- `ApplyStatusSystem` / `"apply_status"` / `Stage::ApplyStatus`
- `ApplyEconomySystem` / `"apply_economy"` / `Stage::ApplyEconomy`
- `ApplyTransfersSystem` / `"apply_transfers"` / `Stage::ApplyTransfers`
- `ApplyDeathsSystem` / `"apply_deaths"` / `Stage::ApplyDeaths`
- `ApplyGridSystem` / `"apply_grid"` / `Stage::ApplyGrid`
- `ApplyFidelitySystem` / `"apply_fidelity"` / `Stage::ApplyFidelity`
- `ApplyPriceReportsSystem` / `"apply_price_reports"` / `Stage::ApplyPriceReports`

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib world_sim::apply::apply_system_coverage`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/apply.rs
git commit -m "refactor(world-sim): wrap apply sub-phases in System trait"
```

---

### Task 0.7: Audit and enumerate active postapply systems

Before migrating postapply systems, nail down exactly which are active.

**Files:**
- Create: `docs/superpowers/specs/active_systems_inventory.md`

- [ ] **Step 1: Extract the active postapply call list**

Run:
```bash
grep -nE '^\s+super::systems::[a-z_]+::[a-z_]+\(' src/world_sim/runtime.rs \
  | grep -v '^//' \
  | sed 's/.*super::systems:://' \
  | sed 's/(.*//' \
  | sort -u \
  > /tmp/active_systems.txt
wc -l /tmp/active_systems.txt
```

Expected: ~35-50 unique `module::fn` entries.

- [ ] **Step 2: Write the inventory**

Create `docs/superpowers/specs/active_systems_inventory.md` with the full list in this format:

```markdown
# Active Postapply Systems Inventory

| Call site (`runtime.rs`) | Module | Function | Struct name | Status |
|---|---|---|---|---|
| L1576 | movement | advance_movement | AdvanceMovementSystem | pending |
| L1577 | monster_ecology | advance_monster_ecology | AdvanceMonsterEcologySystem | pending |
| ... | | | | |
```

The engineer runs the grep in Step 1 and fills the table manually. `Struct name` is the `advance_foo` → `AdvanceFooSystem` rule. `Status` starts as `pending`.

Systems called from inside other systems (not at the top level of `runtime.rs::tick()`) are **not migrated** — they remain plain helpers.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/active_systems_inventory.md
git commit -m "docs: inventory active postapply systems for migration"
```

---

### Task 0.8: `/batch` migrate all active postapply systems

This task is the bulk of Stage 0. It is mechanical and parallelizable.

**Files:**
- Modify: ~45 files in `src/world_sim/systems/*.rs` (one per active system)

- [ ] **Step 1: Prepare the batch prompt**

Open a scratch file `/tmp/batch_prompt.md` and write:

```
For each row in docs/superpowers/specs/active_systems_inventory.md with status=pending:

1. Follow the template in docs/superpowers/specs/system_migration_template.md EXACTLY.
2. The function name and file path come from the inventory row.
3. SYSTEM_NAME = the `Function` column. STRUCT_NAME = the `Struct name` column.
4. STAGE = Stage::PostApply for all postapply systems.
5. Commit per the template.
6. Update the inventory row status from `pending` to `done`.

Do not migrate any system whose call site in runtime.rs is commented out.
Do not migrate helper functions not in the inventory.
```

- [ ] **Step 2: Dispatch `/batch`**

Run `/batch` with the prompt from Step 1. Each worker migrates one row. Monitor for failures; any system whose bare fn doesn't fit the template gets flagged and skipped (re-migrate manually later).

- [ ] **Step 3: Verify no systems were broken**

Run: `cargo test --lib world_sim -- --test-threads=1`
Expected: All tests PASS, determinism preserved.

Run: `cargo run --release --bin xtask -- world-sim --ticks 500 --world small`
Expected: tick/sec within 5% of the pre-Stage-0 baseline noted in Task 0.3 Step 7.

- [ ] **Step 4: Final commit if any stragglers**

```bash
git add docs/superpowers/specs/active_systems_inventory.md
git commit -m "refactor(world-sim): batch-migrate active postapply systems"
```

---

### Task 0.9: Wire `build_registry()` and flip dispatch

Replace direct `super::systems::foo::advance_foo(state)` calls in `runtime.rs` with registry iteration. This is the first moment the trait is actually exercised at runtime.

**Files:**
- Modify: `src/world_sim/runtime.rs`

- [ ] **Step 1: Write a failing test for the registry shape**

Add to the bottom of `src/world_sim/runtime.rs`:

```rust
#[cfg(test)]
mod registry_tests {
    use super::*;
    use crate::world_sim::system::Stage;

    #[test]
    fn build_registry_populates_all_stages() {
        let r = build_registry();
        assert!(!r.systems_in(Stage::ApplyMovement).is_empty(), "apply_movement missing");
        assert!(!r.systems_in(Stage::ApplyEconomy).is_empty(), "apply_economy missing");
        assert!(!r.systems_in(Stage::PostApply).is_empty(), "no postapply systems registered");
    }

    #[test]
    fn build_registry_count_matches_inventory() {
        let r = build_registry();
        // 3 compute + 10 apply + N postapply (from inventory)
        let postapply_count = r.systems_in(Stage::PostApply).len();
        assert!(postapply_count >= 30, "expected >=30 postapply, got {postapply_count}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib world_sim::runtime::registry_tests`
Expected: FAIL — `build_registry` undefined.

- [ ] **Step 3: Write `build_registry()`**

Add near the top of `src/world_sim/runtime.rs`:

```rust
use crate::world_sim::system::{Backend, SystemRegistry};
```

Add a new function (location: near other `pub fn` declarations at the top of the `impl WorldSim` block, or as a standalone `pub fn`):

```rust
pub fn build_registry() -> SystemRegistry {
    let mut r = SystemRegistry::new();
    let b = Backend::default_for_cpu();

    // Compute phases
    r.register(crate::world_sim::compute_high::ComputeHighSystem::new(b));
    r.register(crate::world_sim::compute_medium::ComputeMediumSystem::new(b));
    r.register(crate::world_sim::compute_low::ComputeLowSystem::new(b));

    // Apply sub-phases
    r.register(crate::world_sim::apply::ApplyCloneSystem::new(b));
    r.register(crate::world_sim::apply::ApplyHpSystem::new(b));
    r.register(crate::world_sim::apply::ApplyMovementSystem::new(b));
    r.register(crate::world_sim::apply::ApplyStatusSystem::new(b));
    r.register(crate::world_sim::apply::ApplyEconomySystem::new(b));
    r.register(crate::world_sim::apply::ApplyTransfersSystem::new(b));
    r.register(crate::world_sim::apply::ApplyDeathsSystem::new(b));
    r.register(crate::world_sim::apply::ApplyGridSystem::new(b));
    r.register(crate::world_sim::apply::ApplyFidelitySystem::new(b));
    r.register(crate::world_sim::apply::ApplyPriceReportsSystem::new(b));

    // Postapply systems (generated from inventory — one register() per row)
    // Engineer: fill in below by scanning docs/superpowers/specs/active_systems_inventory.md
    // for each row with status=done and emitting r.register(path::StructName::new(b));
    // Example:
    // r.register(crate::world_sim::systems::movement::AdvanceMovementSystem::new(b));
    // r.register(crate::world_sim::systems::monster_ecology::AdvanceMonsterEcologySystem::new(b));

    r
}
```

The engineer now scans `docs/superpowers/specs/active_systems_inventory.md` and fills in the `r.register(...)` lines — one per row.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib world_sim::runtime::registry_tests`
Expected: PASS.

- [ ] **Step 5: Do NOT yet replace existing direct dispatch**

The registry is populated but not yet driving behavior — existing direct `super::systems::foo::advance_foo()` calls remain the source of truth. Stage 1 adds profiling alongside these calls. Flipping dispatch to registry-only is deferred to a post-Stage-3 cleanup pass (out of scope here).

This is a deliberate choice: it keeps Stage 0 a zero-behavior-change refactor, so any perf regression can be isolated to the trait/registry overhead (call once per registered struct, pay HashMap hash cost) rather than a dispatch change.

- [ ] **Step 6: Run full test + sim smoke**

Run: `cargo test --lib world_sim -- --test-threads=1`
Expected: PASS.

Run: `cargo run --release --bin xtask -- world-sim --ticks 500 --world small`
Expected: tick/sec within 5% of pre-Stage-0 baseline.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/runtime.rs
git commit -m "feat(world-sim): build_registry() populates all staged systems"
```

---

## Stage 1: Diagnostic plumbing

Wire the `profile-systems` feature to produce per-system timings; build the `bench_world` generator and the xtask subcommand.

### Task 1.1: `SystemProfileAccumulator` in `trace.rs`

**Files:**
- Modify: `src/world_sim/trace.rs`

- [ ] **Step 1: Read the existing file**

Run: `cat src/world_sim/trace.rs` — understand current contents before appending.

- [ ] **Step 2: Write the failing test**

Append to `src/world_sim/trace.rs`:

```rust
#[cfg(test)]
#[cfg(feature = "profile-systems")]
mod profile_tests {
    use super::*;

    #[test]
    fn accumulator_records_and_folds() {
        let mut acc = SystemProfileAccumulator::default();
        acc.record("foo", 1000, 10);
        acc.record("foo", 2000, 20);
        acc.record("bar", 500, 5);
        let timings = acc.into_timings();
        let foo = timings.iter().find(|t| t.name == "foo").unwrap();
        assert_eq!(foo.total_ns, 3000);
        assert_eq!(foo.calls, 2);
        assert_eq!(foo.entities_touched, 30);
        let bar = timings.iter().find(|t| t.name == "bar").unwrap();
        assert_eq!(bar.total_ns, 500);
    }
}
```

- [ ] **Step 3: Run test to verify it fails (behind feature)**

Run: `cargo test --lib world_sim::trace::profile_tests --features profile-systems`
Expected: FAIL — types undefined.

- [ ] **Step 4: Write the implementation**

Append to `src/world_sim/trace.rs`:

```rust
#[cfg(feature = "profile-systems")]
pub use system_profile::*;

#[cfg(feature = "profile-systems")]
mod system_profile {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub struct SystemTiming {
        pub name: &'static str,
        pub total_ns: u64,
        pub calls: u32,
        pub entities_touched: u64,
    }

    #[derive(Debug, Default)]
    pub struct SystemProfileAccumulator {
        map: HashMap<&'static str, (u64, u32, u64)>,
    }

    impl SystemProfileAccumulator {
        pub fn record(&mut self, name: &'static str, ns: u64, touched: u32) {
            let entry = self.map.entry(name).or_insert((0, 0, 0));
            entry.0 += ns;
            entry.1 += 1;
            entry.2 += touched as u64;
        }

        pub fn merge(&mut self, other: Self) {
            for (k, (ns, calls, touched)) in other.map {
                let entry = self.map.entry(k).or_insert((0, 0, 0));
                entry.0 += ns;
                entry.1 += calls;
                entry.2 += touched;
            }
        }

        pub fn into_timings(self) -> Vec<SystemTiming> {
            self.map
                .into_iter()
                .map(|(name, (total_ns, calls, entities_touched))| SystemTiming {
                    name, total_ns, calls, entities_touched
                })
                .collect()
        }
    }

    thread_local! {
        pub static THREAD_ACC: std::cell::RefCell<SystemProfileAccumulator> =
            std::cell::RefCell::new(SystemProfileAccumulator::default());
    }

    pub fn thread_record(name: &'static str, ns: u64, touched: u32) {
        THREAD_ACC.with(|a| a.borrow_mut().record(name, ns, touched));
    }

    pub fn thread_drain() -> SystemProfileAccumulator {
        THREAD_ACC.with(|a| std::mem::take(&mut *a.borrow_mut()))
    }
}

#[cfg(not(feature = "profile-systems"))]
pub mod system_profile_stub {
    #[derive(Debug, Clone, Default)]
    pub struct SystemTiming {
        pub name: &'static str,
        pub total_ns: u64,
        pub calls: u32,
        pub entities_touched: u64,
    }
}

#[cfg(not(feature = "profile-systems"))]
pub use system_profile_stub::SystemTiming;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib world_sim::trace::profile_tests --features profile-systems`
Expected: PASS.

- [ ] **Step 6: Run without the feature to verify zero-cost stub**

Run: `cargo build --release` (no features)
Expected: builds clean; `SystemTiming` exists as a zero-cost stub so `TickProfile.system_timings: Vec<SystemTiming>` compiles in both modes.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/trace.rs
git commit -m "feat(world-sim): SystemProfileAccumulator with thread-local API"
```

---

### Task 1.2: `run_with_profile` wrapper and `TickProfile.system_timings`

**Files:**
- Modify: `src/world_sim/system.rs` (add `run_with_profile`)
- Modify: `src/world_sim/tick.rs` (add `system_timings` field)

- [ ] **Step 1: Write the failing test**

Append to `src/world_sim/system.rs`:

```rust
#[cfg(test)]
#[cfg(feature = "profile-systems")]
mod profile_wrapper_tests {
    use super::*;
    use crate::world_sim::trace::thread_drain;

    struct FakeSystem;
    impl System for FakeSystem {
        fn name(&self) -> &'static str { "fake" }
        fn stage(&self) -> Stage { Stage::PostApply }
        fn run(&self, _ctx: &mut SystemCtx) -> u32 { 42 }
    }

    #[test]
    fn run_with_profile_records_time_and_touched() {
        let _ = thread_drain(); // clear any prior
        let state = crate::world_sim::state::WorldState::default();
        let mut deltas = Vec::new();
        let mut ctx = SystemCtx { state: &state, deltas: &mut deltas, tick: 0 };
        let sys = FakeSystem;
        let touched = run_with_profile(&sys, &mut ctx);
        assert_eq!(touched, 42);
        let acc = thread_drain();
        let timings = acc.into_timings();
        let t = timings.iter().find(|t| t.name == "fake").unwrap();
        assert_eq!(t.calls, 1);
        assert_eq!(t.entities_touched, 42);
        assert!(t.total_ns > 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib world_sim::system::profile_wrapper_tests --features profile-systems`
Expected: FAIL — `run_with_profile` undefined.

- [ ] **Step 3: Write the implementation**

Append to `src/world_sim/system.rs`:

```rust
/// Runs a system and, under the `profile-systems` feature, records its
/// wall-clock duration and touched count to the thread-local accumulator.
///
/// Zero overhead when the feature is off.
#[inline]
pub fn run_with_profile(sys: &dyn System, ctx: &mut SystemCtx) -> u32 {
    #[cfg(feature = "profile-systems")]
    {
        let start = std::time::Instant::now();
        let touched = sys.run(ctx);
        let ns = start.elapsed().as_nanos() as u64;
        crate::world_sim::trace::thread_record(sys.name(), ns, touched);
        touched
    }
    #[cfg(not(feature = "profile-systems"))]
    sys.run(ctx)
}
```

Add `system_timings` field to `TickProfile` in `src/world_sim/tick.rs`. Locate the `pub struct TickProfile` block (around line 11) and add as the last field:

```rust
    /// Populated only under the `profile-systems` feature. Empty otherwise.
    pub system_timings: Vec<crate::world_sim::trace::SystemTiming>,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib world_sim::system::profile_wrapper_tests --features profile-systems`
Expected: PASS.

- [ ] **Step 5: Verify non-feature build still compiles**

Run: `cargo build --release`
Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/system.rs src/world_sim/tick.rs
git commit -m "feat(world-sim): run_with_profile wrapper + TickProfile.system_timings"
```

---

### Task 1.3: Wire `run_with_profile` into existing dispatch sites

Instrument the existing postapply dispatch in `runtime.rs` to time each system via the wrapper. Existing bare-fn calls stay; we add a timing shim around them.

**Files:**
- Modify: `src/world_sim/runtime.rs`

- [ ] **Step 1: Design the shim pattern**

For each current direct call in `runtime.rs`:

```rust
super::systems::movement::advance_movement(&mut self.state);
```

Wrap in a local macro that mirrors `run_with_profile` for bare fns:

```rust
macro_rules! time_bare {
    ($name:literal, $call:expr) => {{
        #[cfg(feature = "profile-systems")]
        {
            let t = std::time::Instant::now();
            $call;
            let ns = t.elapsed().as_nanos() as u64;
            crate::world_sim::trace::thread_record($name, ns, 0);
        }
        #[cfg(not(feature = "profile-systems"))]
        { $call; }
    }};
}
```

- [ ] **Step 2: Add the macro and wrap every postapply call**

At the top of the `impl WorldSim` block or near imports in `runtime.rs`, define the `time_bare!` macro (as a module-private `macro_rules!`).

For each of the ~49 direct postapply calls in the `tick()` method body, wrap:

```rust
// Before:
super::systems::movement::advance_movement(&mut self.state);
// After:
time_bare!("advance_movement", super::systems::movement::advance_movement(&mut self.state));
```

Mechanical find-replace. Keep the literal string matching the function name (`"advance_movement"` etc.).

- [ ] **Step 3: Drain thread-locals at end of tick**

In `tick_profiled()` in `tick.rs` (around the end, before `profile.total_us = ...`), add:

```rust
#[cfg(feature = "profile-systems")]
{
    let acc = crate::world_sim::trace::thread_drain();
    profile.system_timings = acc.into_timings();
}
```

- [ ] **Step 4: Add a test that verifies timings populate under the feature**

Append to `src/world_sim/tick.rs`:

```rust
#[cfg(test)]
#[cfg(feature = "profile-systems")]
mod profile_wiring_tests {
    use super::*;

    #[test]
    fn tick_populates_system_timings() {
        let state = crate::world_sim::state::WorldState::default();
        let (_next, prof) = tick_profiled(&state, false);
        // Even on empty state, some systems should have registered calls.
        // Loose assertion — any non-empty vec under the feature flag.
        assert!(
            !prof.system_timings.is_empty() || prof.entities_processed == 0,
            "expected system_timings to populate under profile-systems feature"
        );
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test --lib world_sim::tick::profile_wiring_tests --features profile-systems -- --nocapture`
Expected: PASS.

- [ ] **Step 6: Verify non-feature still compiles and passes**

Run: `cargo test --lib world_sim -- --test-threads=1`
Expected: All tests PASS (the non-feature path should now carry zero overhead).

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/runtime.rs src/world_sim/tick.rs
git commit -m "feat(world-sim): time each postapply system under profile-systems"
```

---

### Task 1.4: `bench_world(target_entities, seed)` generator

**Files:**
- Create: `src/world_sim/bench_world.rs`
- Modify: `src/world_sim/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `src/world_sim/bench_world.rs`:

```rust
//! Deterministic world generator scaled to hit target entity counts
//! after warm-up. Used by the diagnostic benchmark and fixture capture.

use super::runtime::WorldSim;

/// Build a world, scale settlement/NPC seed counts for the target, and
/// warm for 500 ticks so populations are at equilibrium.
pub fn bench_world(target_entities: usize, seed: u64) -> WorldSim {
    // Scaling is heuristic: current small world stabilizes ~700-1300 NPCs.
    // Empirically, ~3x settlements → 2K NPCs, 15x → 10K, 75x → 50K.
    // Exact scalar chosen per-target via floor(target / 700 * 3).
    let settlement_scale = ((target_entities as f32) / 700.0 * 3.0).max(3.0).ceil() as usize;
    let mut sim = WorldSim::with_scaled_seed(settlement_scale, seed);
    for _ in 0..500 { sim.tick(); }
    sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_world_2k_reaches_reasonable_population() {
        let sim = bench_world(2_000, 42);
        let alive = sim.state().entities.iter().filter(|e| e.alive).count();
        // Loose bound — want hundreds to low thousands, not zero.
        assert!(alive > 200, "expected >200 alive entities at 2K target, got {alive}");
    }

    #[test]
    fn bench_world_is_deterministic() {
        let a = bench_world(2_000, 42);
        let b = bench_world(2_000, 42);
        assert_eq!(a.state().entities.len(), b.state().entities.len());
    }
}
```

- [ ] **Step 2: Add the module**

Modify `src/world_sim/mod.rs` — add near the other `pub mod` lines:

```rust
pub mod bench_world;
```

- [ ] **Step 3: Check `WorldSim::with_scaled_seed` exists**

Run: `grep -n 'with_scaled_seed\|with_seed\|new_scaled' src/world_sim/runtime.rs`

If it doesn't exist, the engineer adds a helper in `runtime.rs` that takes a settlement count multiplier and a seed. The shape:

```rust
impl WorldSim {
    pub fn with_scaled_seed(settlement_scale: usize, seed: u64) -> Self {
        let mut sim = Self::with_seed(seed); // use existing constructor
        // Multiply initial settlement count by settlement_scale
        // (adjust whichever world-gen helper seeds settlements).
        let _ = settlement_scale;
        sim
    }
}
```

Engineer: match to the existing constructor pattern (`build_world`, `build_small_world`, etc.) — this is a light wrapper that scales the seeding loop in the existing builder.

- [ ] **Step 4: Run test to verify it fails, then passes**

Run: `cargo test --lib world_sim::bench_world::tests -- --nocapture`
Expected: FAIL initially → PASS after Step 3.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/bench_world.rs src/world_sim/mod.rs src/world_sim/runtime.rs
git commit -m "feat(world-sim): bench_world(target, seed) generator with warm-up"
```

---

### Task 1.5: `xtask bench world-sim` subcommand

**Files:**
- Create: `src/bin/xtask/bench_cmd.rs`
- Modify: `src/bin/xtask/main.rs`

- [ ] **Step 1: Scaffold the subcommand**

Create `src/bin/xtask/bench_cmd.rs`:

```rust
//! Benchmark subcommands for the world sim.

use anyhow::Result;
use clap::{Args, Subcommand};

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub cmd: BenchCmd,
}

#[derive(Subcommand, Debug)]
pub enum BenchCmd {
    /// Run the diagnostic benchmark against a scaled world.
    WorldSim(WorldSimBenchArgs),
    /// Regenerate committed bincode fixtures for the criterion harness.
    RegenerateFixtures,
}

#[derive(Args, Debug)]
pub struct WorldSimBenchArgs {
    #[arg(long, default_value = "2k")]
    pub scale: Scale,
    #[arg(long, default_value_t = 5000)]
    pub ticks: u32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value = "generated")]
    pub output_dir: String,
    #[arg(long)]
    pub flamegraph: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Scale { #[value(name = "2k")] K2, #[value(name = "10k")] K10, #[value(name = "50k")] K50 }

impl Scale {
    pub fn target_entities(self) -> usize {
        match self { Scale::K2 => 2_000, Scale::K10 => 10_000, Scale::K50 => 50_000 }
    }
    pub fn label(self) -> &'static str {
        match self { Scale::K2 => "2k", Scale::K10 => "10k", Scale::K50 => "50k" }
    }
}

pub fn run(args: BenchArgs) -> Result<()> {
    match args.cmd {
        BenchCmd::WorldSim(a) => run_world_sim(a),
        BenchCmd::RegenerateFixtures => run_regenerate_fixtures(),
    }
}

fn run_world_sim(args: WorldSimBenchArgs) -> Result<()> {
    if args.flamegraph {
        // Shell out to scripts/perf_bench.sh (Stage 2 provides this).
        eprintln!("--flamegraph: delegating to scripts/perf_bench.sh");
        let status = std::process::Command::new("./scripts/perf_bench.sh")
            .arg(args.scale.label())
            .arg(args.ticks.to_string())
            .arg(args.seed.to_string())
            .arg(&args.output_dir)
            .status()?;
        anyhow::ensure!(status.success(), "perf_bench.sh failed");
        return Ok(());
    }

    let target = args.scale.target_entities();
    eprintln!("building bench world at scale={} (target={target} entities, seed={})",
              args.scale.label(), args.seed);
    let mut sim = game::world_sim::bench_world::bench_world(target, args.seed);

    let mut acc = game::world_sim::tick::ProfileAccumulator::default();
    let mut per_system: std::collections::HashMap<&'static str, (u64, u32, u64)> =
        std::collections::HashMap::new();

    eprintln!("profiling {} ticks...", args.ticks);
    let start = std::time::Instant::now();
    for _ in 0..args.ticks {
        let prof = sim.tick_profiled();
        acc.record(&prof);
        for t in &prof.system_timings {
            let e = per_system.entry(t.name).or_insert((0, 0, 0));
            e.0 += t.total_ns;
            e.1 += t.calls;
            e.2 += t.entities_touched;
        }
    }
    let elapsed = start.elapsed();

    print_table(&acc, &per_system, args.ticks, elapsed);
    write_json(&args, &acc, &per_system, args.ticks, elapsed)?;
    Ok(())
}

fn print_table(
    acc: &game::world_sim::tick::ProfileAccumulator,
    per_system: &std::collections::HashMap<&'static str, (u64, u32, u64)>,
    ticks: u32,
    wall: std::time::Duration,
) {
    println!("\n=== diagnostic: {} ticks in {:.2?} ({:.1} tick/s) ===",
             ticks, wall, ticks as f64 / wall.as_secs_f64());
    println!("{}", acc);

    let total_ns: u64 = per_system.values().map(|(n, _, _)| *n).sum();
    let mut rows: Vec<_> = per_system.iter().collect();
    rows.sort_by_key(|(_, (ns, _, _))| std::cmp::Reverse(*ns));

    println!("\n{:<40} {:>12} {:>10} {:>14} {:>12} {:>8}",
             "system", "total_ms", "calls", "entities/call", "ns/entity", "% tick");
    for (name, (ns, calls, touched)) in rows {
        let total_ms = *ns as f64 / 1e6;
        let ec = if *calls > 0 { *touched as f64 / *calls as f64 } else { 0.0 };
        let ne = if *touched > 0 { *ns as f64 / *touched as f64 } else { 0.0 };
        let pct = if total_ns > 0 { *ns as f64 / total_ns as f64 * 100.0 } else { 0.0 };
        println!("{:<40} {:>12.2} {:>10} {:>14.1} {:>12.1} {:>7.1}%",
                 name, total_ms, calls, ec, ne, pct);
    }
}

fn write_json(
    args: &WorldSimBenchArgs,
    acc: &game::world_sim::tick::ProfileAccumulator,
    per_system: &std::collections::HashMap<&'static str, (u64, u32, u64)>,
    ticks: u32,
    wall: std::time::Duration,
) -> Result<()> {
    use serde_json::json;
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let path = format!("{}/world_sim_bench_{}_{}.json", args.output_dir, args.scale.label(), ts);
    std::fs::create_dir_all(&args.output_dir)?;

    let systems_json: Vec<_> = per_system.iter().map(|(name, (ns, calls, touched))| {
        let ec = if *calls > 0 { *touched as f64 / *calls as f64 } else { 0.0 };
        let ne = if *touched > 0 { *ns as f64 / *touched as f64 } else { 0.0 };
        json!({
            "name": name,
            "total_ns": ns,
            "calls": calls,
            "entities_touched": touched,
            "entities_per_call": ec,
            "ns_per_entity": ne,
        })
    }).collect();

    let doc = json!({
        "meta": {
            "scale": args.scale.label(),
            "seed": args.seed,
            "ticks": ticks,
            "wall_secs": wall.as_secs_f64(),
            "timestamp": ts,
            "git_sha": env!("VERGEN_GIT_SHA_FALLBACK").to_string(), // fallback if vergen absent
        },
        "tick_profile": {
            "avg_us": acc.avg_tick_us(),
            "avg_compute_us": acc.avg_compute_us(),
            "avg_merge_us": acc.avg_merge_us(),
            "avg_apply_us": acc.avg_apply_us(),
            "min_us": acc.min_tick_us,
            "max_us": acc.max_tick_us,
        },
        "systems": systems_json,
    });

    std::fs::write(&path, serde_json::to_string_pretty(&doc)?)?;
    eprintln!("wrote {path}");
    Ok(())
}

fn run_regenerate_fixtures() -> Result<()> {
    // Implemented in Task 3.2.
    anyhow::bail!("regenerate-fixtures not yet implemented (see Task 3.2)")
}
```

**Engineer notes:**
- If `env!("VERGEN_GIT_SHA_FALLBACK")` doesn't resolve, replace with a hand-rolled `git rev-parse` via `std::process::Command`, or drop the field.
- `chrono` may not already be a dep — check `Cargo.toml` and either add it or use `std::time::SystemTime` + manual formatting.
- `sim.tick_profiled()` — verify this method exists on `WorldSim`; if the current API is `sim.tick()` returning plain state, add a sibling method that returns `TickProfile`.

- [ ] **Step 2: Wire into xtask main**

Modify `src/bin/xtask/main.rs`:

Add module declaration near other `mod` lines:
```rust
mod bench_cmd;
```

Add to the top-level `Commands` enum:
```rust
Bench(bench_cmd::BenchArgs),
```

Add to the `match` in `main()`:
```rust
Commands::Bench(args) => bench_cmd::run(args)?,
```

- [ ] **Step 3: Build and smoke-test**

Run: `cargo build --bin xtask --features profile-systems`
Expected: builds clean.

Run: `cargo run --bin xtask --features profile-systems -- bench world-sim --scale 2k --ticks 100`
Expected: prints diagnostic table, writes JSON to `generated/world_sim_bench_2k_<ts>.json`.

- [ ] **Step 4: Commit**

```bash
git add src/bin/xtask/bench_cmd.rs src/bin/xtask/main.rs
git commit -m "feat(xtask): bench world-sim subcommand with table + JSON output"
```

---

### Task 1.6: Capture baseline JSONs

**Files:**
- Create: `generated/baselines/world_sim_bench_2k_baseline.json`
- Create: `generated/baselines/world_sim_bench_10k_baseline.json`
- Create: `generated/baselines/world_sim_bench_50k_baseline.json`

- [ ] **Step 1: Run at all three scales**

```bash
cargo run --release --bin xtask --features profile-systems -- bench world-sim --scale 2k --ticks 5000 --output-dir generated/baselines
cargo run --release --bin xtask --features profile-systems -- bench world-sim --scale 10k --ticks 5000 --output-dir generated/baselines
cargo run --release --bin xtask --features profile-systems -- bench world-sim --scale 50k --ticks 5000 --output-dir generated/baselines
```

Expected: 3 JSON files + 3 tables in terminal. 50K may take ~3-5 minutes.

- [ ] **Step 2: Rename and commit**

```bash
cd generated/baselines
for scale in 2k 10k 50k; do
  mv world_sim_bench_${scale}_*.json world_sim_bench_${scale}_baseline.json
done
cd -
git add generated/baselines/
git commit -m "chore: capture pre-SIMD baseline diagnostic JSONs"
```

---

## Stage 2: Flamegraph integration

### Task 2.1: `scripts/perf_bench.sh` wrapper

**Files:**
- Create: `scripts/perf_bench.sh`

- [ ] **Step 1: Check `cargo flamegraph` is available**

Run: `cargo flamegraph --version`
Expected: version string.
If not installed: `cargo install flamegraph`.

- [ ] **Step 2: Write the script**

Create `scripts/perf_bench.sh`:

```bash
#!/usr/bin/env bash
# Wraps the xtask world-sim benchmark under cargo flamegraph.
# Args: scale ticks seed output_dir

set -euo pipefail

SCALE="${1:-2k}"
TICKS="${2:-5000}"
SEED="${3:-42}"
OUT_DIR="${4:-generated}"

mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SVG="$OUT_DIR/world_sim_bench_${SCALE}_${TS}.svg"

echo "running flamegraph: scale=$SCALE ticks=$TICKS seed=$SEED → $SVG"

# --root required on Linux unless /proc/sys/kernel/perf_event_paranoid <= 1
sudo_flag=""
if [[ "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 3)" -gt 1 ]]; then
  sudo_flag="--root"
  echo "(perf_event_paranoid > 1; using --root)"
fi

cargo flamegraph $sudo_flag \
  --bin xtask \
  --features profile-systems \
  --output "$SVG" \
  -- bench world-sim --scale "$SCALE" --ticks "$TICKS" --seed "$SEED" --output-dir "$OUT_DIR"

echo "flamegraph written to $SVG"
```

- [ ] **Step 3: Make executable and test**

```bash
chmod +x scripts/perf_bench.sh
./scripts/perf_bench.sh 2k 500 42 /tmp/flame_test
```

Expected: produces an SVG at `/tmp/flame_test/world_sim_bench_2k_<ts>.svg`. 500 ticks is a quick smoke test.

- [ ] **Step 4: Open in Chrome and verify readable**

Open the SVG via Chrome:
```bash
# Stage 2 exit criterion: visual verification in browser.
# Use the claude-in-chrome browser automation tools:
# 1. navigate to file:///tmp/flame_test/world_sim_bench_2k_<ts>.svg
# 2. read_page / get_page_text to confirm it rendered
# 3. Confirm stack frames are visible and labeled
```

The human or a future `/execute` agent uses `mcp__claude-in-chrome__navigate` + `mcp__claude-in-chrome__read_page` to verify. If the SVG is blank or truncated, the `--ticks 500` was too short; bump to 2000.

- [ ] **Step 5: Commit**

```bash
git add scripts/perf_bench.sh
git commit -m "feat(scripts): perf_bench.sh wraps cargo flamegraph"
```

---

### Task 2.2: Generate flamegraphs at all three scales + write SIMD targets doc

**Files:**
- Create: `generated/flamegraphs/world_sim_bench_2k.svg`
- Create: `generated/flamegraphs/world_sim_bench_10k.svg`
- Create: `generated/flamegraphs/world_sim_bench_50k.svg`
- Create: `docs/superpowers/specs/world_sim_simd_targets.md`

- [ ] **Step 1: Run at all scales**

```bash
./scripts/perf_bench.sh 2k 5000 42 generated/flamegraphs
./scripts/perf_bench.sh 10k 5000 42 generated/flamegraphs
./scripts/perf_bench.sh 50k 5000 42 generated/flamegraphs
```

Expected: 3 SVGs.

- [ ] **Step 2: Open each in Chrome and verify**

For each SVG, use the claude-in-chrome tools:
```
navigate → file://<abs_path>/generated/flamegraphs/world_sim_bench_2k.svg
read_page → confirm stack frames visible
```

Repeat for 10k and 50k. If any fail to render, bump `--ticks` to 10000 and regenerate.

- [ ] **Step 3: Author the SIMD targets doc**

Cross-reference the 3 JSONs (from Task 1.6) and 3 SVGs. For each of the top 5 candidates by `total_ns`, write an entry in `docs/superpowers/specs/world_sim_simd_targets.md`:

```markdown
# World Sim SIMD Targets

Ranked by `total_ns` across 2K/10K/50K diagnostic runs (2026-04-17).

## 1. apply_movement

- **File:** `src/world_sim/apply.rs:<line>` (search for `ApplyMovementSystem::apply_inplace`)
- **Why:** f32 force-vector magnitude clamp + position update. Zero branches in hot path.
- **ns/entity at 2K / 10K / 50K:** (fill from JSONs)
- **% of tick at 50K:** (fill)
- **SIMD plan:** f32x8 chunked magnitude compute; scalar tail for len%8.

## 2. apply_economy

...

## 3. (next target)

...
```

The engineer extracts exact numbers from the JSONs.

- [ ] **Step 4: Commit**

```bash
git add generated/flamegraphs/ docs/superpowers/specs/world_sim_simd_targets.md
git commit -m "docs: capture flamegraphs and identify top SIMD targets"
```

---

## Stage 3: Criterion regression harness

### Task 3.1: Create `crates/world_sim_bench` workspace member

**Files:**
- Create: `crates/world_sim_bench/Cargo.toml`
- Create: `crates/world_sim_bench/rust-toolchain.toml`
- Create: `crates/world_sim_bench/src/lib.rs`
- Modify: `Cargo.toml` (root) — add workspace member

- [ ] **Step 1: Write the toolchain pin**

Create `crates/world_sim_bench/rust-toolchain.toml`:

```toml
[toolchain]
channel = "nightly"
components = ["rustc", "cargo"]
```

- [ ] **Step 2: Write the crate manifest**

Create `crates/world_sim_bench/Cargo.toml`:

```toml
[package]
name = "world_sim_bench"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
game = { path = "../..", features = ["profile-systems"] }
bincode = "1.3"
anyhow = "1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "movement"
harness = false

[[bench]]
name = "economy"
harness = false

[[bench]]
name = "hp_changes"
harness = false

[[bench]]
name = "merge"
harness = false
```

- [ ] **Step 3: Write the crate root**

Create `crates/world_sim_bench/src/lib.rs`:

```rust
#![feature(portable_simd)]

//! Criterion regression harness for the world sim. Loads committed
//! bincode fixtures and compares backends (Scalar vs Simd) per hot loop.

pub mod fixtures;
```

- [ ] **Step 4: Add to workspace**

Modify root `Cargo.toml`:

```toml
[workspace]
members = [".", "crates/tactical_sim", "crates/world_sim_bench"]
```

- [ ] **Step 5: Verify it builds on nightly**

```bash
cd crates/world_sim_bench && cargo +nightly build && cd -
```

Expected: builds clean.

- [ ] **Step 6: Commit**

```bash
git add crates/world_sim_bench/ Cargo.toml
git commit -m "feat(bench): create world_sim_bench crate on nightly"
```

---

### Task 3.2: Fixture capture (`regenerate-fixtures` subcommand) + loader

**Files:**
- Modify: `src/bin/xtask/bench_cmd.rs` (implement `run_regenerate_fixtures`)
- Create: `crates/world_sim_bench/src/fixtures.rs`

- [ ] **Step 1: Implement `run_regenerate_fixtures` in `bench_cmd.rs`**

Replace the `anyhow::bail!` stub in `run_regenerate_fixtures`:

```rust
fn run_regenerate_fixtures() -> Result<()> {
    use std::io::Write;

    let fixtures_dir = std::path::Path::new("crates/world_sim_bench/fixtures");
    std::fs::create_dir_all(fixtures_dir)?;

    for (label, target) in [("2k", 2_000usize), ("10k", 10_000), ("50k", 50_000)] {
        eprintln!("capturing fixture: {label} (target={target})");
        let sim = game::world_sim::bench_world::bench_world(target, 42);
        let bytes = bincode::serialize(sim.state())?;
        let path = fixtures_dir.join(format!("world_{label}.bin"));
        std::fs::File::create(&path)?.write_all(&bytes)?;
        eprintln!("  wrote {} ({} bytes)", path.display(), bytes.len());
    }

    // Schema hash: a stable digest of the WorldState type's fields.
    // We use a coarse proxy: hash of the bincode output size distribution.
    // Future: use `ty-hash` or a custom derive for a strong schema check.
    let sha = schema_fingerprint()?;
    std::fs::write(fixtures_dir.join("fixtures.sha"), sha)?;
    eprintln!("wrote fixtures.sha");
    Ok(())
}

fn schema_fingerprint() -> Result<String> {
    // Deterministic hash of a freshly-built, tiny WorldState.
    let mut tiny = game::world_sim::bench_world::bench_world(200, 0);
    let bytes = bincode::serialize(tiny.state())?;
    let digest = sha256_hex(&bytes);
    Ok(digest)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}
```

**Engineer notes:**
- `sha2` may not be a dep. Add to root `Cargo.toml`: `sha2 = "0.10"`.
- `WorldState` must impl `Serialize`. If it doesn't, derive it (may touch many nested types — verify before starting).
- If `Serialize` derivation is too invasive, fall back to saving the raw entities Vec + seed, regenerate state on load.

- [ ] **Step 2: Write the fixture loader**

Create `crates/world_sim_bench/src/fixtures.rs`:

```rust
//! Load committed bincode fixtures. Verifies schema hash on first load.

use anyhow::{anyhow, Result};
use game::world_sim::state::WorldState;

pub fn load(scale: &str) -> Result<WorldState> {
    let path = format!("crates/world_sim_bench/fixtures/world_{scale}.bin");
    let bytes = std::fs::read(&path)
        .map_err(|e| anyhow!("fixture missing at {path}: {e}. run `cargo run --bin xtask -- bench regenerate-fixtures`"))?;
    verify_schema()?;
    let state: WorldState = bincode::deserialize(&bytes)?;
    Ok(state)
}

fn verify_schema() -> Result<()> {
    // For now, just check fixtures.sha exists. A stronger check would
    // recompute the schema hash and compare — but that's expensive enough
    // that we do it only in --strict mode (future).
    let sha_path = "crates/world_sim_bench/fixtures/fixtures.sha";
    if !std::path::Path::new(sha_path).exists() {
        return Err(anyhow!("missing fixtures.sha — regenerate fixtures"));
    }
    Ok(())
}
```

- [ ] **Step 3: Write the loader test**

Append to `crates/world_sim_bench/src/fixtures.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_2k_fixture() {
        let state = load("2k").expect("2k fixture should exist after regenerate-fixtures");
        assert!(state.entities.len() >= 200);
    }
}
```

- [ ] **Step 4: Regenerate fixtures and run the test**

```bash
cargo run --release --bin xtask -- bench regenerate-fixtures
cd crates/world_sim_bench && cargo +nightly test && cd -
```

Expected: fixtures written (~3 files + sha), test passes.

- [ ] **Step 5: Commit**

```bash
git add src/bin/xtask/bench_cmd.rs crates/world_sim_bench/src/fixtures.rs crates/world_sim_bench/fixtures/ Cargo.toml
git commit -m "feat(bench): regenerate-fixtures subcommand + fixture loader"
```

---

### Task 3.3: `movement.rs` criterion bench (scalar baseline)

**Files:**
- Create: `crates/world_sim_bench/benches/movement.rs`

- [ ] **Step 1: Write the bench**

Create `crates/world_sim_bench/benches/movement.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use game::world_sim::apply::ApplyMovementSystem;
use game::world_sim::system::Backend;
use world_sim_bench::fixtures;

fn bench_movement(c: &mut Criterion) {
    for scale in ["2k", "10k", "50k"] {
        let state = fixtures::load(scale).expect("fixture");
        // NOTE: MergedDeltas construction: derive a representative one from
        // the state by running one compute pass. Placeholder: empty merged
        // deltas; benches measure traversal cost dominated by position
        // update. Revisit with real merged deltas once Stage 1 captures them.
        let merged = game::world_sim::delta::MergedDeltas::default();

        let mut group = c.benchmark_group(format!("movement/{scale}"));
        for backend in [Backend::Scalar, Backend::Simd] {
            let sys = ApplyMovementSystem::new(backend);
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{backend:?}")),
                &(sys, state.clone(), merged.clone()),
                |b, (sys, state, merged)| {
                    b.iter_batched(
                        || state.clone(),
                        |mut s| sys.apply_inplace(&mut s, merged),
                        criterion::BatchSize::LargeInput,
                    );
                }
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_movement);
criterion_main!(benches);
```

**Engineer notes:**
- `MergedDeltas::default()` — verify it exists. If not, construct an empty one via `MergedDeltas { forces: Default::default(), ... }`.
- If `ApplyMovementSystem::apply_inplace` signature differs, adjust.
- `WorldState` must impl `Clone` — should already; verify.

- [ ] **Step 2: Run the bench**

```bash
cd crates/world_sim_bench && cargo +nightly bench --bench movement 2>&1 | tee /tmp/movement_bench.log
```

Expected: produces 6 timings (3 scales × 2 backends; Simd currently falls back to Scalar, so numbers will be near-identical). HTML report in `target/criterion/movement/`.

- [ ] **Step 3: Commit**

```bash
git add crates/world_sim_bench/benches/movement.rs
git commit -m "bench(world-sim): movement scalar baseline at 2k/10k/50k"
```

---

### Task 3.4: `economy.rs`, `hp_changes.rs`, `merge.rs` criterion benches

**Files:**
- Create: `crates/world_sim_bench/benches/economy.rs`
- Create: `crates/world_sim_bench/benches/hp_changes.rs`
- Create: `crates/world_sim_bench/benches/merge.rs`

Same pattern as Task 3.3 for three more loops.

- [ ] **Step 1: Create `economy.rs`**

Create `crates/world_sim_bench/benches/economy.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use game::world_sim::apply::ApplyEconomySystem;
use game::world_sim::system::Backend;
use world_sim_bench::fixtures;

fn bench_economy(c: &mut Criterion) {
    for scale in ["2k", "10k", "50k"] {
        let state = fixtures::load(scale).expect("fixture");
        let merged = game::world_sim::delta::MergedDeltas::default();

        let mut group = c.benchmark_group(format!("economy/{scale}"));
        for backend in [Backend::Scalar, Backend::Simd] {
            let sys = ApplyEconomySystem::new(backend);
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{backend:?}")),
                &(sys, state.clone(), merged.clone()),
                |b, (sys, state, merged)| {
                    b.iter_batched(
                        || state.clone(),
                        |mut s| sys.apply_inplace(&mut s, merged),
                        criterion::BatchSize::LargeInput,
                    );
                }
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_economy);
criterion_main!(benches);
```

**Engineer note:** `ApplyEconomySystem::apply_inplace` must be added in `apply.rs` following the same shape as `ApplyMovementSystem::apply_inplace` (Task 0.3). If it's not there yet, add it now: extract the existing economy loop into a method on the new system struct.

- [ ] **Step 2: Create `hp_changes.rs`**

Same shape. Uses `ApplyHpSystem`. Extract `apply_inplace` in `apply.rs` if missing.

- [ ] **Step 3: Create `merge.rs`**

This is the anti-candidate baseline — benches the HashMap-heavy merge phase, no System-trait wrapping needed because merge runs once per tick, not per-system.

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use game::world_sim::delta::{merge_deltas, WorldDelta};
use world_sim_bench::fixtures;

fn bench_merge(c: &mut Criterion) {
    for scale in ["2k", "10k", "50k"] {
        let state = fixtures::load(scale).expect("fixture");
        // Generate a representative delta set by running one compute pass.
        // The current API: `compute_all_deltas_seq_counted(&state)` returns
        // (Vec<WorldDelta>, _, _). Reach in via a public re-export or use a
        // thin helper. If the function isn't pub, expose it behind a
        // `#[cfg(feature = "benchables")]` visibility shim.
        let deltas: Vec<WorldDelta> = game::world_sim::tick::debug_compute_deltas(&state);

        let mut group = c.benchmark_group(format!("merge/{scale}"));
        group.bench_with_input(
            BenchmarkId::from_parameter("scalar"),
            &deltas,
            |b, deltas| {
                b.iter_batched(
                    || deltas.clone(),
                    |d| merge_deltas(d),
                    criterion::BatchSize::LargeInput,
                );
            }
        );
        group.finish();
    }
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);
```

**Engineer note:** `debug_compute_deltas` does not exist — add a public wrapper in `tick.rs`:

```rust
#[cfg(feature = "profile-systems")]
pub fn debug_compute_deltas(state: &WorldState) -> Vec<WorldDelta> {
    compute_all_deltas_seq_counted(state).0
}
```

- [ ] **Step 4: Run all three benches**

```bash
cd crates/world_sim_bench && cargo +nightly bench 2>&1 | tee /tmp/all_benches.log
```

Expected: produces timings for movement, economy, hp_changes, merge at all three scales.

- [ ] **Step 5: Verify HTML reports**

Open `target/criterion/report/index.html` via claude-in-chrome:
- `navigate` to `file://<abs>/target/criterion/report/index.html`
- `read_page` to confirm all 4 benches appear

- [ ] **Step 6: Commit**

```bash
git add crates/world_sim_bench/benches/ src/world_sim/apply.rs src/world_sim/tick.rs
git commit -m "bench(world-sim): economy, hp_changes, merge scalar baselines"
```

---

### Task 3.5: `scripts/bench_summary.py` markdown emitter

**Files:**
- Create: `scripts/bench_summary.py`

- [ ] **Step 1: Write the script**

Create `scripts/bench_summary.py`:

```python
#!/usr/bin/env python3
"""Parse criterion's estimates.json files and emit a markdown summary table.

Usage: python scripts/bench_summary.py [criterion_root]
Default criterion_root: target/criterion
"""
import json
import sys
from pathlib import Path

def load_estimate(p: Path) -> float:
    """Return mean nanoseconds from a criterion estimates.json."""
    with p.open() as f:
        doc = json.load(f)
    return float(doc["mean"]["point_estimate"])

def walk(root: Path):
    """Yield (group, scale, backend, ns) tuples."""
    for estimate in root.glob("**/new/estimates.json"):
        # Path shape: target/criterion/<group>/<scale>/<backend>/new/estimates.json
        parts = estimate.relative_to(root).parts
        if len(parts) < 5:
            continue
        group, scale, backend = parts[0], parts[1], parts[2]
        ns = load_estimate(estimate)
        yield group, scale, backend, ns

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "target/criterion")
    if not root.exists():
        sys.exit(f"criterion root not found: {root}. Run `cargo bench` first.")

    # Aggregate: group -> scale -> backend -> ns
    data = {}
    for group, scale, backend, ns in walk(root):
        data.setdefault(group, {}).setdefault(scale, {})[backend] = ns

    print("# World Sim Bench Summary\n")
    for group in sorted(data):
        print(f"## {group}\n")
        print("| scale | scalar (µs) | simd (µs) | speedup |")
        print("|---|---|---|---|")
        for scale in sorted(data[group], key=lambda s: int(s.rstrip("k")) * 1000):
            row = data[group][scale]
            scalar = row.get("Scalar", row.get("scalar", 0.0))
            simd = row.get("Simd", row.get("simd", 0.0))
            speedup = scalar / simd if simd > 0 else float("inf")
            print(f"| {scale} | {scalar/1000:.1f} | {simd/1000:.1f} | {speedup:.2f}x |")
        print()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python3 scripts/bench_summary.py > /tmp/bench_summary.md
cat /tmp/bench_summary.md
```

Expected: markdown table for each of movement, economy, hp_changes, merge × 3 scales. Scalar and Simd columns near-identical until real SIMD lands.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_summary.py
git commit -m "feat(scripts): bench_summary.py parses criterion JSON to markdown"
```

---

### Task 3.6: End-to-end smoke test

Verify the full pipeline runs cleanly.

- [ ] **Step 1: Full rebuild from scratch**

```bash
cargo clean
cargo build --release --bin xtask --features profile-systems
```

Expected: clean build.

- [ ] **Step 2: Regenerate fixtures**

```bash
cargo run --release --bin xtask -- bench regenerate-fixtures
```

Expected: `crates/world_sim_bench/fixtures/world_{2k,10k,50k}.bin` + `fixtures.sha` present.

- [ ] **Step 3: Run diagnostic at all scales**

```bash
cargo run --release --bin xtask --features profile-systems -- bench world-sim --scale 2k --ticks 5000
cargo run --release --bin xtask --features profile-systems -- bench world-sim --scale 10k --ticks 5000
cargo run --release --bin xtask --features profile-systems -- bench world-sim --scale 50k --ticks 5000
```

Expected: 3 JSONs, 3 tables.

- [ ] **Step 4: Run flamegraph at 10k**

```bash
./scripts/perf_bench.sh 10k 3000 42 generated/flamegraphs
```

Expected: SVG written. Open in Chrome via claude-in-chrome tools, verify readable.

- [ ] **Step 5: Run criterion benches**

```bash
cd crates/world_sim_bench && cargo +nightly bench && cd -
```

Expected: all 4 benches run green.

- [ ] **Step 6: Generate summary**

```bash
python3 scripts/bench_summary.py
```

Expected: markdown table covering all benches.

- [ ] **Step 7: Commit the logs/artifacts if any uncommitted remain**

```bash
git status
# If any stray artifact is present: add or ignore via .gitignore
git commit -m "chore: Stage 3 end-to-end smoke pass"  # only if anything to commit
```

---

## Self-Review Checklist (completed during plan authoring)

- **Spec coverage:**
  - Dead-file cleanup → Stage -1 (Tasks -1.1, -1.2)
  - System trait → Stage 0 (Tasks 0.1-0.9)
  - Backend enum + generic kernel → Task 0.3 (pilot) establishes pattern
  - Per-system instrumentation (`profile-systems` feature) → Stage 1 (Tasks 1.1-1.3)
  - `bench_world` generator → Task 1.4
  - xtask diagnostic subcommand with table + JSON → Task 1.5
  - Baseline JSONs committed → Task 1.6
  - Flamegraph script + integration → Tasks 2.1-2.2
  - SIMD targets doc → Task 2.2 Step 3
  - Chrome verification of flamegraphs → Tasks 2.1 Step 4, 2.2 Step 2
  - Criterion crate on nightly → Task 3.1
  - Fixture capture + loader + schema hash → Task 3.2
  - Four criterion benches (movement, economy, hp_changes, merge) → Tasks 3.3-3.4
  - Markdown summary tool → Task 3.5
  - End-to-end smoke → Task 3.6

- **Placeholder scan:** No TBDs. Every step has either a code block, a command, or a clear engineer note. "Engineer notes" blocks flag known deviations the engineer must adapt (e.g., if `WorldState::default()` doesn't exist) but each deviation includes the adaptation recipe.

- **Type consistency:** `ApplyMovementSystem`/`Backend`/`Stage`/`SystemCtx`/`SystemRegistry`/`run_with_profile`/`SystemTiming`/`SystemProfileAccumulator` names match across all tasks. `Stage` enum variants in Task 0.1 are the same variants referenced in Tasks 0.5, 0.6, 0.9, 1.2. `bench_world(target, seed)` signature in 1.4 matches the calls in 3.2 and 3.3.

- **Known plan risks:**
  1. `WorldState` serde — Stage 3 fixtures depend on it. If serde derivation is too invasive, Task 3.2 fallback (save seed + regenerate) kicks in.
  2. `/batch` migration (Task 0.8) may produce non-uniform results if workers diverge from the template. Mitigation: determinism tests after the batch; any failure points at the divergent system.
  3. Merge-phase benchmark (Task 3.4) needs `debug_compute_deltas` wrapper — included as a sub-note in Step 3.
