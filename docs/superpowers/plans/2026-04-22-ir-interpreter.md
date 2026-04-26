# DSL IR Interpreter (all rule classes) — Implementation Plan

> **Migration note (2026-04-25, post-Spec-B'):** Authored on `wsb-engine-viz`
> pre-B'. Re-derivation rather than cherry-pick (decision (B), 2026-04-25).
> Key adaptations needed in §"File Structure":
> - `crates/engine/src/evaluator/` triggers engine/build.rs allowlist gate
>   (Spec B' D11). Place under `engine/src/policy/evaluator.rs` to avoid,
>   or accept the gate with critic dispatch.
> - `crates/engine/src/step.rs` is gone (step body emitted into
>   `engine_rules/src/step.rs` per Plan B1' Task 11). Dispatch branches
>   under `interpreted-rules` feature live in emitted code via
>   `dsl_compiler::emit_step` feature-gated emit-passes. Larger refactor
>   than original plan anticipated.
> - `crates/engine/src/mask.rs` is storage-only post-B'; `mark_*_allowed`
>   moved to emitted `engine_rules/src/mask_fill.rs`.
> - `Context` impls thread `EventRing<E>` + `CascadeRegistry<E, V>`
>   (generic primitives per Spec B' D13).
> - `wolves_and_humans_parity` baseline is post-B'.
>
> The `wsb-engine-viz` branch has a working implementation (~5K LoC) that
> serves as a design reference, NOT a cherry-pick target. Per (B) decision,
> the agent re-derives the same end-state on post-B' main. Companion:
> `plans/2026-04-22-dsl-ast-extraction.md` (P1a — must land first).

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an interpreted evaluation path for every rule-class IR type in `dsl_ast` (`MaskIR`, `ScoringIR`, `PhysicsIR`, `ViewIR`), exposed as methods on the IR types using the visitor pattern. Gate it behind a `cargo` feature `interpreted-rules` on `engine`. Prove byte-for-byte parity against the compiled path using the existing `wolves_and_humans` baseline, one rule class at a time.

**Architecture:** The interpreter lives inside `dsl_ast` as `impl` blocks on IR types (`MaskIR::eval`, `ScoringIR::eval`, `PhysicsIR::apply`, `ViewIR::fold`). A small per-class `Context` trait in `dsl_ast` abstracts state access; `engine` provides the impls against `SimState`. At each rule-dispatch call site in the engine tick loop, a `#[cfg(feature = "interpreted-rules")]` branch routes to the interpreter; the default (compiled) path is untouched. Production binaries never enable the feature.

**Tech Stack:** Rust 2021, `dsl_ast` (landed in P1a), `engine`. No new crate deps. No new workspace members.

**Prerequisites:** P1a (`dsl_ast` extraction) landed — `dsl_ast::ir::{MaskIR, ScoringIR, PhysicsIR, ViewIR}` are reachable. The `wolves_and_humans_parity.rs` test + committed baseline exist in `crates/engine/tests/`. Engine is on HEAD `bbb0ace0` or later.

---

## Scope

### In scope

- Method-on-IR interpreters for the four rule-evaluating classes in `dsl_ast`.
- One `Context` trait per class (or a unified trait — implementer decides based on the survey in Task 1).
- `impl Context for SimState` (or equivalents) in `engine`.
- `interpreted-rules` cargo feature on `engine`.
- Four `#[cfg(feature = "interpreted-rules")]` dispatch branches in the engine tick loop, one per rule class.
- Parity gate: wolves+humans fixture must produce byte-identical `replayable_sha256` under `--features interpreted-rules` at each cutover stage.
- CI job: add `--features interpreted-rules` to the test matrix.

### Out of scope

- Removing the compiled path. The compiled path stays; production runtime uses it.
- Dev daemon / file-watch / control protocol (spec §4.3). Separate later plan.
- LSP / viz changes (spec §4.6–§4.7). Separate later plans.
- Non-`SimState` `Context` impls (e.g., a no-op context for testing). Not needed for the parity gate.
- Any IR type beyond the four rule-evaluating classes. `EntityIR`, `EnumIR`, `ConfigIR`, `EventIR` are pure data — no `eval` method required.
- Performance tuning of the interpreter. Correctness first; if wolves+humans runs too slowly to fit CI, flag it in the last task.

### Non-goals

- **No runtime mode toggle.** A single binary has exactly one path; the feature flag is a build-time choice.
- **No trait object dispatch at the engine tick-loop call sites.** Use `#[cfg]` branches; keep the hot path monomorphic.

---

## File Structure

### New files

```
crates/dsl_ast/src/eval/
├── mod.rs              (Context trait(s); re-exports)
├── mask.rs             (impl MaskIR { fn eval(...) })
├── scoring.rs          (impl ScoringIR { fn eval(...) })
├── physics.rs          (impl PhysicsIR { fn apply(...) })
└── view.rs             (impl ViewIR { fn fold(...) })

crates/engine/src/evaluator/
├── mod.rs              (pub use; feature-gated)
└── context.rs          (impl *Context for SimState)
```

Interpreter lives in `dsl_ast` so it's reusable independently of `engine`. `engine` owns the `SimState`-backed `Context` impls and the dispatch plumbing.

### Modified files

- `crates/dsl_ast/src/lib.rs` — `pub mod eval;`.
- `crates/dsl_ast/Cargo.toml` — no new deps expected.
- `crates/engine/Cargo.toml` — add `[features] interpreted-rules = []` (+ any other gating the feature needs).
- `crates/engine/src/lib.rs` — `pub mod evaluator;` under `#[cfg(feature = "interpreted-rules")]` or always-on. Implementer decides.
- `crates/engine/src/step.rs` / `crates/engine/src/mask.rs` / similar — feature-gated dispatch branches at each rule-class call site.
- `crates/engine/tests/wolves_and_humans_parity.rs` — add an `#[cfg(feature = "interpreted-rules")]` sibling test reusing the same baseline.
- CI config (`.github/workflows/*.yml` or equivalent) — add a job / step running `cargo test --features interpreted-rules -p engine`.

### Deleted files

None.

---

## Acceptance criteria (plan-level)

1. **`cargo check --workspace` clean** (with and without `--features interpreted-rules`).
2. **`cargo check -p engine --features interpreted-rules` clean.**
3. **`cargo test -p engine` green** (default — compiled path).
4. **`cargo test -p engine --features interpreted-rules` green** — including the `wolves_and_humans_parity` test, which must match the committed baseline byte-for-byte.
5. **Each of `MaskIR::eval`, `ScoringIR::eval`, `PhysicsIR::apply`, `ViewIR::fold` has unit tests against hand-crafted IR values** — tests live in the respective `dsl_ast/src/eval/*.rs` files and run under `cargo test -p dsl_ast`.
6. **`dsl_ast::eval::{MaskContext, ScoringContext, CascadeContext, ViewContext}` (or a unified `RuleContext`) is a stable public API** — no panics-on-default, no unimplemented methods.
7. **The compiled path (`generated/mask/*.rs` etc.) is unchanged.** No drift in files under `crates/engine/src/generated/` or `crates/engine_generated/src/`.
8. **CI enforces the interpreted-rules parity gate** — a new CI job / matrix entry fails the PR if the hash diverges.

If any of (1)–(8) fails, the plan is not done.

---

## Tasks

### Task 1: Primitive survey

**Goal:** Inventory exactly which IR variants, stdlib functions, and field reads the wolves+humans fixture exercises in each of the four rule classes. This bounds the interpreter's required coverage for P1b.

- [ ] Read `assets/sim/{masks,scoring,physics,views}.sim` and `assets/sim/entities.sim`.
- [ ] For each of `MaskIR`, `ScoringIR`, `PhysicsIR`, `ViewIR`, list the IR variants present in the wolves+humans rules. Example: does the mask DSL use `BinOp::And`, `UnOp::Not`, `ExprKind::FieldRead`, stdlib `distance`, etc.?
- [ ] Capture the inventory as a short comment block at the top of each `eval/<class>.rs` file (created empty in this task) — or a single `docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md` if you prefer. This is the checklist for Tasks 3–6.

**Acceptance:** Inventory committed. Subsequent tasks use this as the "what must be supported" ceiling. Anything outside this inventory is explicitly P1c+ scope.

---

### Task 2: Define `Context` trait(s) in `dsl_ast`

**Goal:** Design the minimal state-access surface the interpreter needs.

- [ ] Create `crates/dsl_ast/src/eval/mod.rs`.
- [ ] Decide: one unified `RuleContext` trait with all methods, or four class-specific traits (`MaskContext`, `ScoringContext`, `CascadeContext`, `ViewContext`)? The survey in Task 1 informs this — if the four classes read distinct state, split. If they overlap heavily, unify.
- [ ] Trait methods cover only what the survey says the interpreter needs. No aspirational methods.
- [ ] `pub mod eval;` in `dsl_ast/src/lib.rs`.

**Acceptance:** `cargo check -p dsl_ast` passes. The trait(s) compile with no impls yet — `engine` adds impls in Task 7.

**Commit:** `feat(dsl_ast): define Context trait(s) for IR interpreter`

---

### Task 3: `MaskIR::eval`

**Goal:** Implement the mask predicate interpreter.

- [ ] Create `crates/dsl_ast/src/eval/mask.rs`.
- [ ] `impl MaskIR { pub fn eval<C: MaskContext>(&self, ctx: &C, agent: AgentId, tick: u32) -> bool }`. Walk the IR via `match` over variants. Covers exactly the primitives from Task 1's survey; anything outside = `unimplemented!()` with a clear panic message pointing at the survey doc.
- [ ] Unit tests in the same file. Construct hand-crafted `MaskIR` values, run `eval` against a mock `MaskContext`, assert expected `bool`.

**Acceptance:** `cargo test -p dsl_ast eval::mask` green. Module-level doc comment lists the supported primitive set, matching the Task 1 survey.

**Commit:** `feat(dsl_ast): MaskIR::eval interpreter`

---

### Task 4: `ScoringIR::eval`

**Goal:** Implement the scoring-expression interpreter.

- [ ] Create `crates/dsl_ast/src/eval/scoring.rs`.
- [ ] `impl ScoringIR { pub fn eval<C: ScoringContext>(&self, ctx: &C, agent: AgentId, target: TargetId, tick: u32) -> f32 }` (exact signature depends on the survey — may need tiebreak RNG from context).
- [ ] Per-entry eval for `ScoringEntryIR` where applicable.
- [ ] Unit tests.

**Acceptance:** `cargo test -p dsl_ast eval::scoring` green. Tiebreak RNG semantics match the engine's scoring RNG discipline (hash(world_seed, agent_id, tick, "scoring") per engine spec §9).

**Commit:** `feat(dsl_ast): ScoringIR::eval interpreter`

---

### Task 5: `PhysicsIR::apply`

**Goal:** Implement the cascade-handler interpreter.

- [ ] Create `crates/dsl_ast/src/eval/physics.rs`.
- [ ] `impl PhysicsIR { pub fn apply<C: CascadeContext>(&self, event: &Event, ctx: &mut C) }` — side-effecting via `ctx.emit(...)` or equivalent. The exact event type comes from `dsl_ast::ir` or is abstracted behind the Context trait.
- [ ] Also implement for `PhysicsHandlerIR` if the dispatch is two-level in the current IR.
- [ ] Unit tests with a mock `CascadeContext` that captures emitted events.

**Acceptance:** `cargo test -p dsl_ast eval::physics` green. The unit tests demonstrate event-fan-out matches an expected emission list.

**Commit:** `feat(dsl_ast): PhysicsIR::apply interpreter`

---

### Task 6: `ViewIR::fold`

**Goal:** Implement the materialized-view fold interpreter.

- [ ] Create `crates/dsl_ast/src/eval/view.rs`.
- [ ] `impl ViewIR { pub fn fold<C: ViewContext>(&self, event: &Event, ctx: &mut C) }`. Also `ViewBodyIR` if needed.
- [ ] Unit tests with a mock `ViewContext` recording fold mutations.

**Acceptance:** `cargo test -p dsl_ast eval::view` green. Fold behavior matches a simple recorded-expectation on hand-crafted view IRs.

**Commit:** `feat(dsl_ast): ViewIR::fold interpreter`

---

### Task 7: `impl Context for SimState`

**Goal:** Provide the engine-side Context implementations so the interpreter can read/write real engine state.

- [ ] Create `crates/engine/src/evaluator/context.rs`.
- [ ] `impl MaskContext for SimState { ... }` (or a narrower type if `SimState` is too broad — e.g. a view that also carries `SimScratch`, `EventRing`). Use composition: `struct MaskCtx<'a> { state: &'a SimState, scratch: &'a SimScratch }` etc. Keep the surface minimum-viable.
- [ ] Same for the other three Context traits.
- [ ] Under `#[cfg(feature = "interpreted-rules")]` so unused Context glue doesn't pollute the default-build symbol table.

**Acceptance:** `cargo check -p engine --features interpreted-rules` passes.

**Commit:** `feat(engine): Context impls against SimState under interpreted-rules feature`

---

### Task 8: Mask dispatch branch + parity #1

**Goal:** Cut mask evaluation over to the interpreter (under the feature flag) and verify wolves+humans hash.

- [ ] Add `[features] interpreted-rules = []` to `crates/engine/Cargo.toml`.
- [ ] Locate the single mask-dispatch call site in the engine's tick loop (probably in `step.rs` or `mask.rs`). Wrap it in a `#[cfg(feature = "interpreted-rules")]` / `#[cfg(not(...))]` pair: interpreted path calls `mask.eval(&ctx, agent, tick)` walking the `Compilation`; compiled path is today's call.
- [ ] The engine needs the `Compilation` in scope. Decide how it enters: either a new ctor param `SimState::new(..., compilation: &'static Compilation)` under the feature flag, or load once at engine init from `assets/sim/*.sim` via `dsl_ast::compile`. Simpler = ctor param; caller supplies it.
- [ ] Extend `wolves_and_humans_parity.rs` with a `#[cfg(feature = "interpreted-rules")]` sibling test that runs the same scenario and asserts the committed baseline hash.
- [ ] Run: `cargo test -p engine --features interpreted-rules wolves_and_humans_parity`. Must match the baseline.

**Acceptance:** Both the default-path and interpreted-path variants of the parity test pass. Hash matches byte-for-byte.

**Commit:** `feat(engine): route mask dispatch through interpreter under feature flag (parity #1)`

---

### Task 9: Scoring dispatch branch + parity #2

- [ ] Same pattern as Task 8, for the scoring dispatch site. Feature-gated.
- [ ] Parity test still matches baseline.

**Acceptance:** `cargo test -p engine --features interpreted-rules` green; hash matches.

**Commit:** `feat(engine): route scoring dispatch through interpreter (parity #2)`

---

### Task 10: Cascade dispatch branch + parity #3

- [ ] Feature-gated cutover of cascade-handler dispatch to `PhysicsIR::apply`.
- [ ] Parity test still matches baseline.

**Acceptance:** `cargo test -p engine --features interpreted-rules` green; hash matches.

**Commit:** `feat(engine): route cascade dispatch through interpreter (parity #3)`

---

### Task 11: View dispatch branch + parity #4

- [ ] Feature-gated cutover of materialized-view fold dispatch to `ViewIR::fold`.
- [ ] Parity test still matches baseline.

**Acceptance:** `cargo test -p engine --features interpreted-rules` green; hash matches. At this point all four rule classes are under interpreted dispatch when the feature is on, and the full wolves+humans sim produces the same bytes as the compiled path.

**Commit:** `feat(engine): route view-fold dispatch through interpreter (parity #4)`

---

### Task 12: CI + docs

**Goal:** Lock the parity gate in CI and document the interpreter surface.

- [ ] Add a CI job (or matrix entry) that runs `cargo test -p engine --features interpreted-rules`. The job fails the PR if the parity hash diverges. Place this alongside existing engine test jobs.
- [ ] Write a short `crates/dsl_ast/src/eval/README.md` (or module-level `mod.rs` doc-comment) describing: the Context traits, the `eval`/`apply`/`fold` methods, the primitive coverage (from Task 1's survey), and the explicit non-goal of covering primitives beyond wolves+humans in this plan.
- [ ] Benchmark note: if `cargo test -p engine --features interpreted-rules wolves_and_humans_parity` takes longer than, say, 30 seconds in release mode, flag it in the PR description so the owner can decide whether to tune before merging. Not a blocking criterion — correctness first.

**Acceptance:** CI is green on both default and `interpreted-rules` configs. Docs exist. Interpreter-path runtime is logged in the PR.

**Commit:** `ci+docs: interpreted-rules parity gate + eval module README`

---

## Notes for the implementer

- **Survey first, code second.** Task 1 exists to avoid the trap of writing `unimplemented!()` for variants you didn't realize the fixture uses. Finish the inventory before Task 3.
- **Each class is independent.** If one class's interpreter hits a snag, the others still land as separate commits. Don't couple them.
- **Don't touch the compiled path.** If you find yourself editing `crates/engine/src/generated/` or `crates/engine_generated/src/`, stop — you've left this plan's scope.
- **Feature-flag discipline.** The default build MUST be byte-identical to pre-plan HEAD. Any behavior change outside the `#[cfg(feature = "interpreted-rules")]` branches is a regression.
- **Baseline ownership.** `wolves_and_humans_baseline.txt` is the canonical ground truth. Never regenerate it casually. If parity diverges, the fix is in the interpreter, not the baseline.
