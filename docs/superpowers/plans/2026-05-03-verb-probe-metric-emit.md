# Verb / Probe / Metric Emit — Close Three Remaining Silent-Drop Decl Kinds

> **Status: DRAFT (2026-05-03).** Written as the follow-up to the same-day partial close of the §2.8 `invariant`
> emit (`crates/dsl_compiler/src/cg/emit/invariants.rs`), which landed the simplest of the four silent-drop
> decl kinds flagged in the 2026-04-26 spec audit (`docs/spec/dsl.md` §2.6 callout). This plan covers the three
> remaining: `verb` (§2.6), `probe` (§2.9), `metric` (§2.11). All three parse cleanly and resolve to typed IR
> (`VerbIR` / `ProbeIR` / `MetricIR` in `dsl_ast::ir`), but **`EmittedArtifacts` has no entries for any of them
> and no emit pass exists** — a fixture that *declares* one appears to be exercising the surface but the compiler
> emits nothing. Today's invariant close used a host-side check pattern; verb/probe/metric each need a different
> emit shape (cascade-composition, test-harness driver, telemetry side-channel respectively) and so warrant
> separate work tracks.

**Goal:** every `verb`, `probe`, `metric` declaration in the shipped fixtures (`predator_prey_min.sim`,
`crowd_navigation.sim`, `particle_collision.sim`, `swarm_event_storm.sim`) compiles to an artifact a downstream
runtime / harness consumes — no decl kind silently produces zero output. The invariant precedent is the model:
when the predicate fits a supported emit shape it produces a real check; when it doesn't, it surfaces a `// SKIP
<reason>` comment so authors see the gap at build time rather than at runtime.

**Architecture:** Three independent slices, each landable in isolation. They are deliberately separate plans-in-
one because the consumer surfaces are different and a single grand emit pass would conflate concerns.

1. **Slice A — `verb` (composition sugar).** A `verb` is `mask_predicate + cascade_handler + scoring_entry`
   bundled under one name. The spec (§2.6) is explicit that the compiler should expand into those three
   pre-existing primitives. Emit shape: extend `cg::lower::driver::lower_compilation_to_cg` to inject one
   `MaskIR` entry, one `PhysicsHandlerIR` (cascade), one `ScoringEntryIR` per `VerbIR`. **No new emit file** —
   the existing mask / physics / scoring emit paths inherit the verb's expansion. The "is it firing" gate is
   that the emitted mask kernel for the bundled action gains a new bit position; the cascade kernel gains a new
   handler; scoring rows include the new entry.

2. **Slice B — `probe` (test-harness driver).** A `probe` is a named scenario + per-tick assertion bundle.
   Emit shape: synthesise a `probes.rs` artifact that exposes `pub fn run_<name>(state: &mut <RuntimeState>) ->
   ProbeOutcome` for each `ProbeIR`. The function reads the probe's `scenario` / `seed` / `seeds` / `ticks`
   metadata as a constant-folded test header, drives the supplied runtime through `ticks` ticks via the existing
   `CompiledSim` trait, and after each tick (or once at the end, depending on the assert kind) checks the
   `IrAssertExpr::{Count, Pr, Mean}` predicates against event-ring readback. This needs a `RuntimeState`-shaped
   trait in `engine` (not yet defined) — the smallest extension is a `ProbeRunnable` trait the per-fixture
   runtimes implement. Stress-test each probe against a fixture that *should* fail (reduce `ticks` below the
   threshold; assert the probe returns `Failed`).

3. **Slice C — `metric` (telemetry side-channel).** A `metric` is `metric <name> = <expr>` with optional
   `window` / `emit_every` / `alert when` modifiers (`MetricIR`). Per-tick the runtime evaluates `<expr>`
   against the current state and records the value into a typed sink. Emit shape: synthesise a `metrics.rs`
   artifact with `pub struct MetricsSink { <one-field-per-metric>: <Histogram | Gauge | Counter> }` and `pub fn
   record_tick(&mut self, state: &<state>) -> ()`. The expression evaluator is the same scalar-comparator
   subset used by the invariant close, extended with `histogram(expr)` / `gauge(expr)` / `counter(expr)`
   namespace forms (the spec audit flagged these belong on a `MetricKind` enum, not as stdlib calls — see
   `docs/spec/dsl.md:1009`). Slice C owns lifting them.

**Tech Stack:** Rust workspace; `dsl_compiler` (lower + emit), per-fixture runtimes (consumer wiring). No GPU
work — verb cascades emit through the existing physics/mask/scoring kernel pipelines (Slice A); probes and
metrics are host-side (Slices B + C) like the invariant close. Verification by re-running the fixture apps and
asserting each new artifact exposes its consumer surface.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `synthesize_invariants` at `crates/dsl_compiler/src/cg/emit/invariants.rs:67` — the precedent host-side
    emit pass landed today. Demonstrates the per-decl artifact pattern: one `Compilation` field → one
    `EmittedArtifacts.rust_files` entry.
  - `emit_cg_program_with_invariants` at `crates/dsl_compiler/src/cg/emit/invariants.rs:111` — the wrapper that
    keeps `emit_cg_program` un-changed and injects the `invariants.rs` artifact. Slices B + C extend the same
    wrapper with `probes.rs` / `metrics.rs` injection.
  - `VerbIR` at `crates/dsl_ast/src/ir.rs:861` (`name`, `params`, `action`, `when`, `emits`, `scoring`) —
    Slice A consumer.
  - `ProbeIR` at `crates/dsl_ast/src/ir.rs:890` (`scenario`, `seed`, `seeds`, `ticks`, `tolerance`, `asserts`)
    + `IrAssertExpr` enum at `crates/dsl_ast/src/ir.rs:903` — Slice B consumer.
  - `MetricIR` at `crates/dsl_ast/src/ir.rs:931` (`value`, `window`, `emit_every`, `conditioned_on`,
    `alert_when`) — Slice C consumer. The spec audit at `docs/spec/dsl.md:1009` flags `histogram` / `gauge` /
    `counter` need to lift from "stdlib calls" to a `MetricKind` enum on the IR; that lift may need to land
    before Slice C's emit pass.
  - `MaskIR` / `PhysicsIR` / `ScoringIR` at `crates/dsl_ast/src/ir.rs:649,580,683` — Slice A's expansion
    targets. Verb expansion mutates these arenas at lower time, not at emit time, so the existing emit
    pipelines pick up the new entries automatically.
  - `predator_prey_min.sim` (verb examples deferred), `crowd_navigation.sim` (verb + probe + metric),
    `particle_collision.sim` (probe + metric), `swarm_event_storm.sim` (metric only) — fixture workload for
    each slice.
  - `engine::CompiledSim` trait at `crates/engine/src/sim_trait.rs` — the per-fixture state contract Slice B
    extends with `ProbeRunnable`.
  - `engine::invariant::registry::InvariantRegistry` at `crates/engine/src/invariant/registry.rs:11` — exists,
    not yet wired to compiler-emitted invariants (today's close emits per-fixture host fns rather than
    registering trait impls; Slice B's probe shape may want the same trait-impl direction depending on
    determinism / replayability requirements).
  - Spec audit callout to update on each slice landing: `docs/spec/dsl.md` §2.6 (verb), §2.9 (probe), §2.11
    (metric) — same pattern as today's §2.8 invariant downgrade from "❌ silent-drop" to "⚠️ partial close".

  Search method: `rg -n` on `VerbIR|ProbeIR|MetricIR|EmittedArtifacts|synthesize_invariants`, direct `Read` on
  `dsl_ast/src/ir.rs`, `dsl_compiler/src/cg/emit/{invariants,program}.rs`, the four fixture `.sim` files.

- **Decision:** **Extend** the same emit pattern today's invariant close established. Each slice adds one new
  emit module under `crates/dsl_compiler/src/cg/emit/` (`verbs.rs` / `probes.rs` / `metrics.rs`), each owning
  its own predicate-shape classifier with `// SKIP <reason>` for unsupported shapes. Each extends
  `emit_cg_program_with_invariants` (already the orchestrator wrapper) to also inject its artifact. **Slice A
  is structurally different**: it doesn't add a new artifact file — it expands the `Compilation` arenas
  (`masks` / `physics` / `scoring`) at lower time so the existing emit paths produce the verb's bundled output.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE for the slices themselves. Verification needs at least one *intentionally-failing*
    fixture per slice (B + C) to prove the artifact reports the failure — those test fixtures live next to the
    test files, not in `assets/sim/`, to keep production fixtures clean.
  - Generated outputs re-emitted: every `<fixture>_runtime/build.rs`-driven emit re-runs. Slice A produces
    new mask / physics / scoring kernels per verb (mechanical extension of existing emit pipelines). Slices B
    + C produce one new `<probes,metrics>.rs` per fixture that declares one — wrapped by the same `for sibling
    in [...]` loop the runtime build scripts already use for `schedule` / `dispatch` / `invariants`.

- **Hand-written downstream code:** Per-fixture runtime crates each gain one method per slice they consume:
  `state.run_probe_<name>()` (B), `state.metrics_sink()` (C). Verb cascades (A) flow through existing kernel
  pipelines; no new runtime hooks. Justification: B and C produce host-side outputs the runtime must surface —
  the alternative is a globally-shared probe/metric runner crate, which couples every fixture to every other
  fixture's runtime types. Per-fixture exposure stays consistent with the today-landed
  `state.check_invariants()` pattern.

- **Constitution check:**
  - P1 (Compiler-First): PASS — every per-decl semantics lives in `dsl_compiler/src/cg/emit/<kind>.rs`. The
    per-fixture runtime methods are 3-line wrappers that call into the generated module; no rule logic
    duplicates outside the compiler.
  - P2 (Schema-Hash on Layout): N/A — no SimState SoA changes for any slice. Verb expansion adds rows to the
    rules table (which `rules_hash` already covers per spec §3.3 line 567).
  - P3 (Cross-Backend Parity): PASS for A (mask / cascade / scoring already cross-backend); host-only for B + C
    (probes and metrics are by definition test-harness / telemetry — not part of the per-tick deterministic
    closure).
  - P4 (`EffectOp` Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): PASS — A inherits cascade determinism from existing physics emit; B + C
    are pure scalar evaluations over already-deterministic state readback.
  - P6 (Events Are the Mutation Channel): PASS — verb cascades emit through the standard event ring; probes and
    metrics are read-only over event-ring + view-storage.
  - P7 (Replayability Flagged): N/A — no new event decls added.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — each slice closes with the gate test below + a
    `closes_commit` SHA.
  - P10 (No Runtime Panic): PASS — every emit pass returns `Result<String, EmitError>` and skips with a
    typed reason on unsupported shapes; the invariant close is the precedent.
  - P11 (Reduction Determinism): N/A for B + C (no reductions). PASS for A — verb cascades emit through the
    same per-pair fold path the existing cascades use, which already routes through the sort-then-fold
    reduction pattern.

- **Runtime gate:**
  - Slice A: `verb_emit_expands_mask_and_cascade` at `crates/dsl_compiler/tests/verb_emit.rs` — assert the
    `physics::Pray` cascade kernel from a synthetic fixture appears in `EmittedArtifacts.kernel_index` and
    fires when its precondition holds. Plus: `predator_prey_runtime` smoke run still produces its expected
    per-Wolf kill-count observable.
  - Slice B: `probe_fails_on_intentionally_short_run` at
    `crates/dsl_compiler/tests/probe_emit_runtime.rs` — drive a probe through 50 ticks when its assertion
    requires 500; assert `ProbeOutcome::Failed` with a typed payload. Plus: every existing fixture's probe
    declarations compile cleanly and report `Passed` at the documented tick budgets.
  - Slice C: `metric_records_per_tick_value` at `crates/dsl_compiler/tests/metric_emit.rs` — assert the
    histogram / gauge / counter sinks accumulate the right values across 100 ticks of a synthetic fixture.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task
  list stabilises — pending slice-A scope sign-off).

## Slice ordering

1. Slice C (metric) is the closest analogue to today's invariant close — same host-side scalar-evaluator
   shape, same per-fixture artifact pattern. Likely lands in 1-2 sessions.
2. Slice B (probe) needs the `ProbeRunnable` trait + harness wiring; estimate 2-3 sessions.
3. Slice A (verb) is the largest because it touches lower-time arena expansion; estimate 3-5 sessions and
   needs cycle-detection re-validation (verb cascades that write the same field as an existing physics rule
   must not introduce a new SCC). Land last.

Each slice is an independent plan once it leaves draft; the slice headings here are a roadmap, not the final
plan files.

## Open questions

- **Slice A:** does verb expansion happen during `lower_compilation_to_cg` (cleanest — same site that builds
  the rest of the CG arenas) or during `resolve` (which would mean `VerbIR` becomes a transient IR kind that
  doesn't reach `Compilation` at all)? Lower-time expansion lets the resolver stay shape-preserving.
- **Slice B:** the spec (§2.9) hints probes might need cross-seed comparison (`forall (run_a, run_b) in [...]`
  in `particle_collision.sim`'s `deterministic_collision_stream`). Single-seed probes are a strict subset and
  are fine for slice B's first cut; multi-seed needs per-fixture parallel sim instantiation, deferred to a
  slice B.2.
- **Slice C:** the `MetricKind` lift from stdlib-call to enum (audit at `dsl.md:1009`) — does that land as a
  prerequisite (parse + resolve change) or alongside slice C (parser stays, emitter does the recognition)?
  Decision: emit-side recognition is simpler; the parser change is a separate plan.

## Out of scope

- Schema-hash bump for verb-expanded mask / cascade / scoring rows. Spec §3.3 line 567 says the `rules_hash`
  already covers the post-expansion mask + cascade + scoring set, so verb adds don't bump the hash; this is
  consistent with the spec and needs no change in this plan.
- Trait-based invariant registration (`engine::invariant::registry`). Today's invariant close emits per-fixture
  host fns; whether to migrate to trait-impl registration is a separate refactor.
- Probe scenario file format (`scenarios/<name>.toml`). Slice B uses whatever exists; if no scenario file
  loader exists yet, slice B's harness either inlines the scenario or spawns the runtime in its default-init
  shape. Real scenario loading is a Phase 9 concern.
