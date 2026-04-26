# DSL Authoring Engine — Design

> **Migration note (2026-04-25, post-Spec-B'):** Authored on `wsb-engine-viz`
> pre-Spec-B'. Now: engine has zero rule-aware code; `engine/src/step.rs`
> deleted (body emitted into `engine_rules`); sealed `CascadeHandler<E, V>`
> + view traits; build sentinels; primitives-only allowlist on `engine/`.
> Companion plans (P1a `dsl-ast-extraction`, P1b `ir-interpreter`) need
> rewriting against post-B' file layout: dispatch branches under
> `interpreted-rules` feature live in **emitted** code (`engine_rules`,
> via feature-gated emit-passes in `dsl_compiler`), not hand-edited
> engine source. Per (B) decision 2026-04-25, the implementations get
> re-derived; this spec stays as a design reference. References to
> `crates/engine_generated` should now read `crates/engine_data` (renamed
> in Spec B' Task 1).

> Spec: AI-native authoring surface that exposes the world-sim DSL,
> interpreter-mode runtime, and visualization layer as a unified
> "engine" product. Targets a tight edit → run → inspect loop for an
> agent driver (primary) and a human observer (secondary).
>
> Branch: `world-sim-bench` (this doc assumes the `crates/engine`,
> `crates/dsl_compiler`, `crates/viz`, `crates/engine_rules`,
> `crates/engine_generated` layout introduced on that branch).
> Cross-refs: `docs/dsl/spec.md`, `docs/compiler/spec.md`,
> `docs/engine/spec.md`, `docs/project_toc.md`.

---

## 1. Problem

The DSL, the engine, and the viz each work in isolation but don't
compose into a usable authoring loop.

- **Compilation delay is minutes.** Changes to mask predicates,
  scoring expressions, cascade handlers, or system-level rules all
  require a full `cargo` rebuild of emitted Rust in
  `engine_rules`/`engine_generated`. Iteration is slow enough that
  nothing converges.
- **Impact of a rule change is opaque.** Runs emit gigabytes of trace.
  Even with deterministic replay, comparing "before" and "after" is
  eyeballing terminal output. No semantic diff, no per-agent ghost
  path, no "this change caused 40% more sieges."
- **There is no AI-native driver seat.** The primary human is an AI
  agent. There is no stable, typed control surface over which an
  agent can drive the loop: compile, run, diff, probe, follow an
  agent, read a causal trace. Today every piece has to be scraped out
  of CLI stdout or raw trace bytes.

The target is **a stable authoring surface** that treats the agent as
the first-class client, exposes simulation data as structured queries,
and collapses the edit-to-result latency for the A–E rule classes
(defined below) from minutes to seconds.

---

## 2. Scope

### 2.1 In scope (Phase 1)

- `dsl_ast` crate: extract parser + AST types shared by compiler and
  interpreter.
- `InterpretedEvaluator` inside `crates/engine`: AST walker for A–E
  rule classes against the same SoA as `StaticEvaluator`. Schema
  layout stays compiled.
- `engine_dev` daemon: file-watcher, trace archive, control protocol
  over Unix-domain socket + stdio JSON-lines.
- `engine_diff` crate: pure `(TraceA, TraceB) → Diff` reduction.
- `engine_narrator` crate: templated prose generation for run
  summaries, agent stories, and encounter narratives.
- LSP server for the DSL with inlay hints, diagnostics, navigation,
  completion, code actions.
- Introspection protocol verbs: `schema.*`, `rules.*`, `lint.*`,
  `scaffold.*`.
- Minimum-viable viz: scene render, timeline scrub, play/pause, click
  agent → state panel, compare mode, chronicle panel. "Usable and
  legible," not polished.
- Contract test suite per §9.

### 2.2 Phase 2

- Narrator LLM rewrite layer (non-deterministic, cached by
  `(encounter_id, style, model_version)`).
- `diff.agent_stories` verb.
- Cascade DAG viewer (E2a): read-only rendering of parsed cascade
  topology with runtime overlays (fire counts per node,
  "fired-this-tick" highlight).
- Scoring / rule-dependency graph viewer (E2b).
- Viz polish: dockable panels, saved layouts, keyboard-driven
  command palette.

### 2.3 Phase 3

- Interactive graph authoring with DSL round-trip (E2c).
- Windows daemon support.
- Micro-crate FFI hot-reload as an alternative to the interpreter, if
  perf demands native-code reload in dev.

### 2.4 Non-goals (will not build)

- **Mid-run hot-swap.** Rule reloads happen only at run boundaries.
  Mid-run swap tangles determinism for no gain; cancel-and-rerun is
  one verb away.
- **Schema hot-reload.** Entity fields, event variants, and action
  kinds still require `cargo` rebuild. The `schema_hash` guard from
  `docs/dsl/spec.md` §4 stays authoritative.
- **Visual rule editor / graph authoring.** Text DSL is the source of
  truth. Graph viewers (E2a, E2b) are read-only. E2c is a Phase 3
  question, not a promise.
- **Bytecode VM.** If the interpreter's perf ceiling becomes a
  bottleneck, the answer is FFI-linked micro-crates, not a VM.
- **In-memory `rule.replace` verb.** Rule authoring goes through
  normal file edits on disk. The daemon file-watches; the OS + LRU
  cap handle lifecycle. Avoids OOM risk from accumulated in-memory
  rule versions.
- **Privileged protocol clients.** Viz, agent, and a human terminal
  all speak the same protocol. No back-channel APIs.
- **GPU parity for the interpreter.** Interpreter runs on
  `SerialBackend` only. Cross-evaluator parity is CPU-vs-CPU.
  GPU backend keeps its own separate parity story (Serial-static vs
  GPU-static), untouched by this work.

### 2.5 Rule classes (terminology)

Reload discipline is per rule class, not per file. The A–E
classification below determines whether an edit triggers interpreter
reload (cheap) or a cargo rebuild (expensive).

| Class | Content | Reload path |
|---|---|---|
| A | Scoring weights / utility tables | Interpreter reload |
| B | Mask predicates | Interpreter reload |
| C | Cascade handlers (event → emit) | Interpreter reload |
| D | Scoring formula structure | Interpreter reload |
| E | System-level tick logic, new views, new verbs | Interpreter reload |
| F | Schema: entity fields, event variants, action kinds | Full cargo rebuild |

---

## 3. Architecture

Two orthogonal additions to the existing three-crate picture
(`engine`, `dsl_compiler`, `viz`):

```
                  ┌────────────────────────────────────────────┐
                  │  Agent / CLI / Viz (all protocol clients)  │
                  └─────────────┬──────────────────────────────┘
                                │ JSON-lines over UDS / stdio
                  ┌─────────────▼──────────────────────────────┐
                  │   engine_dev  (daemon)                     │
                  │   • file-watch DSL tree                    │
                  │   • control protocol                       │
                  │   • trace archive (LRU 10 GB)              │
                  │   • drives engine instance                 │
                  └─────────────┬──────────────────────────────┘
                                │
         ┌──────────────────────┼────────────────────────┐
         │                      │                        │
  ┌──────▼─────┐         ┌──────▼─────┐          ┌───────▼───────┐
  │ dsl_ast    │◀────────│  engine    │          │ engine_diff   │
  │ parse+AST  │         │  Static+   │          │ engine_narrator│
  │ spans      │         │  Interp    │          │ (pure fns over │
  │            │         │  Evaluator │          │  trace)        │
  └─────▲──────┘         └────────────┘          └───────────────┘
        │
  ┌─────┴──────┐                                 ┌───────────────┐
  │ dsl_compiler│                                 │ lsp_server    │
  │ (emit Rust) │                                 │ (sidecar)     │
  └────────────┘                                 └───────────────┘
```

**3.1. Second rule-evaluation path inside the engine.**
`RuleEvaluator` trait with two impls:

- `StaticEvaluator` — wraps today's `engine_rules` / `engine_generated`
  dispatch. Full native Rust. Unchanged from current behavior.
- `InterpretedEvaluator` — walks AST nodes cached from `dsl_ast`.
  Operates on the same SoA, event ring, spatial index, RNG, pools,
  invariants, telemetry. Only the rule-body dispatch differs.

Engine picks at init via config (`mode: static | interpreted`). A
single run is either all-static or all-interpreted — not mixed. The
default mode is a daemon config setting; `probe.*` verbs may force
interpreted mode for re-run replay regardless of that default
(§6.4). Cross-evaluator parity (same DSL, same seed, byte-identical
`replayable_sha256`) is the load-bearing test.

**3.2. Daemon owns the loop.**
`engine_dev` is an independent crate, not a feature of `engine`. It
owns:

- The DSL source tree (file-watch via `notify`).
- The engine instance (run in-process; one active at a time for
  Phase 1).
- The trace archive (on-disk, LRU, keyed by
  `(seed, dsl_hash, schema_hash)`).
- The control protocol (Unix-domain socket + stdio JSON-lines).

Keeping this out of `engine` keeps the deterministic core free of
`notify`, IPC, and archive code.

**3.3. Viz is a client.**
`crates/viz` connects to `engine_dev` over the same socket as an
agent or CLI. No privileged API. Viz adds rendering and camera
navigation; no more.

**3.4. LSP sidecar.**
An LSP server process, separate from the daemon, reads the same
DSL tree + connects to `engine_dev` for run-stats-derived inlay
hints. Standalone so it's usable from VS Code, neovim, Emacs, and an
agent without needing the daemon to be running for pure editing.

---

## 4. Components

### 4.1. `dsl_ast` (new crate)

Parsing + AST types, extracted from `dsl_compiler`. Both the compiler
(lowering to Rust) and the engine (interpretation) depend on it.

- Parser errors carry structured `{file, line, col, len, kind,
  hint}` spans. No string-only errors.
- Source-position preservation for LSP round-trip (rename, extract).
- Pretty-printer for canonicalization (used by DSL hash computation
  in §6.5).

### 4.2. Interpreter in `crates/engine`

New module `engine/src/evaluator/` with `RuleEvaluator` trait,
`StaticEvaluator` impl (wraps existing dispatch), and
`InterpretedEvaluator` impl (AST walker).

Scope of interpretation covers rule classes A–E:

- **Mask predicates (B):** boolean expressions over agent + aggregate
  state. Evaluated per candidate action per agent.
- **Scoring expressions (A, D):** utility-backend scoring tables +
  formulas. Deterministic tiebreak via hashed RNG.
- **Cascade bodies (C):** reactive handlers `on_event(E) { emit X; ...
  }`. Schema-hash-guarded against event-variant drift.
- **System bodies (E):** per-phase tick functions registered with the
  pipeline.

Interpreter is **instrumentable** in a way the static path is not.
Every rule evaluation can be tagged with `(rule_id, agent_id, tick,
inputs_snapshot, result)` and dropped into a probe ring buffer. This
is what makes `probe.event` and `probe.agent` tractable: the
interpreter *is* the probe. Instrumentation is opt-in per run to
keep the non-probe path fast.

### 4.3. `engine_dev` (new crate)

Daemon responsibilities:

- File-watch DSL source tree; re-parse on save with debounce.
- Classify change: schema (F) vs. A–E.
- Manage engine instance; load DSL AST on next run.
- Trace archive: on-disk ring, LRU-capped (default 10 GB, config
  override).
- Control protocol server (§5).
- Emit pub/sub events on the socket: `run_started`, `run_completed`,
  `rule_reloaded`, `schema_changed`, `parse_error`, `probe_ready`.

Dependencies: `notify`, `serde_json`, `tokio` or `smol` for async
socket handling, plus `engine`, `dsl_ast`, `engine_diff`,
`engine_narrator`.

### 4.4. `engine_diff` (new crate)

Pure function: `(TraceA, TraceB) → Diff`. No rendering. Produces:

- `divergence_tick: u64` — first disagreement (cheap via
  `replayable_sha256` prefix comparison).
- `per_agent: HashMap<AgentId, DivergencePoint>` — tick + old/new
  event at the first per-agent divergence.
- `event_kind_delta: HashMap<EventKind, i64>` — count deltas.
- `chronicle_delta: { added: [], removed: [], changed: [] }`.
- `aggregate_metrics_delta: { ... }` — named scalar deltas (alive
  agents, open quests, settlement counts, ongoing wars, etc.).

Paginated via cursors on any field that can grow.

### 4.5. `engine_narrator` (new crate)

Trace → `Narrative` reduction. Phase 1 is template-only (deterministic
pure function of the trace). Three levels of output:

- `RunSummary` — whole-run reduction, target ~500 tokens.
- `AgentStory` — ordered `AgentEpisode`s with participants, actions,
  belief/need deltas, rendered prose.
- `EncounterNarrative` — per-encounter narrative. An encounter is a
  contiguous-tick run where ≥ 2 agents are spatially co-located and
  ≥ 1 inter-agent event fires.

Detail levels per call: `terse | paragraph | beat`. Determinism
holds because the renderer is pure function of the trace.

LLM rewrite is Phase 2, keyed by `(encounter_id, style,
model_version)` when it lands.

### 4.6. `lsp_server` (new crate)

LSP server over the DSL. Sidecar process. Depends on `dsl_ast` and
optionally connects to `engine_dev` over the protocol for runtime
data.

- Syntax highlighting via a TextMate / tree-sitter grammar shipped
  alongside.
- **Diagnostics:** parse errors, `schema_hash` mismatches, unused
  rules, unreachable cascade branches, type mismatches, event-kind
  typos, scoring-table key typos.
- **Navigation:** go-to-definition, find-references (rule refs,
  event producers/consumers, field reads/writes).
- **Hover:** rule signature, field docs, event schema.
- **Completion:** rule names, event kinds, field names,
  scoring-table keys.
- **Code actions:** rename symbol, extract rule, new-rule scaffold,
  "create handler for this event."
- **Inlay hints from last run:** when `engine_dev` is reachable, each
  rule body is annotated with `// fired N times`, `// avg X
  invocations per tick`, `// effect size Y`. This is the
  authoring-to-impact feedback loop.

### 4.7. `viz` (extended)

Existing winit + `voxel_engine` scene stays. Additions in Phase 1:

- Protocol client (same socket as agent).
- Subscribe to `run_completed`; fetch `Narrative` + `Diff` + trace.
- Compare mode: two traces loaded side-by-side; timeline scrub
  drives both; overlays per §5.3d.
- Click-agent-see-state panel; pulls from `probe.agent`.
- Chronicle / narrative side panel; pulls from
  `chronicle.summarize`.
- Console: filterable event log.

No hierarchy / project-browser panels, no dockable layout in Phase 1.

---

## 5. Control protocol

Unix-domain socket (primary) + stdio JSON-lines (for convenience
when driving from a single process). NDJSON framing. All responses
include `dsl_hash` and `schema_hash` so the caller knows what
version produced them.

### 5.1. Lifecycle verbs

| Verb | Semantics |
|---|---|
| `run {seed, ticks, mode: "interpreted" \| "static"}` | Compile if needed; execute; archive; return `run_id`. Blocks until complete (or streams progress events). |
| `run.cancel {run_id}` | Abort a running run. Partial trace archived with `status: canceled`. |
| `runs.list {filter?}` | Metadata only: `(run_id, seed, dsl_hash, schema_hash, tick_count, status, archived_at)`. Paginated. |
| `runs.get_trace {run_id, range?}` | Stream raw trace bytes for a run. |

### 5.2. Introspection verbs

| Verb | Semantics |
|---|---|
| `schema.entities` | Return entity catalog (field lists, types, hot/cold tags). |
| `schema.events` | Return event catalog (variants, fields). |
| `schema.actions` | Return micro + macro action vocabulary. |
| `schema.rules` | Return rule catalog grouped by class (A–E). |
| `rules.definition {rule_id}` | DSL source span + raw text for a rule. |
| `rules.references {rule_id}` | All sites in the DSL tree that reference this rule. |
| `rules.last_run_stats {rule_id, run_id?}` | Fire counts, input distributions, effect-size summary. Feeds LSP inlay hints. |
| `lint.run {paths?}` | Static analysis pass. Returns diagnostics in the same shape as LSP emits them. |
| `scaffold.cascade_handler {event_kind}` | Emit DSL source template for a new cascade handler. |
| `scaffold.rule {class, name}` | Emit DSL source template for a new rule of the given class. |

### 5.3. Analysis verbs

| Verb | Semantics |
|---|---|
| `diff.get {run_a, run_b}` | Full `Diff` object. Paginated for large categories. |
| `probe.agent {run, agent, tick}` | Scoring table, mask results, believed knowledge, decision trace for that tick. |
| `probe.event {run, event_id}` | Causal trace: rule that fired it, inputs, upstream events. Auto-replays from nearest snapshot with interpreter instrumentation on. |
| `state.at {run, tick, scope?}` | Reconstructed state snapshot at tick. Uses N=500 snapshot cadence + replay. |
| `events.query {run, predicate}` | Filter + paginate over trace events. |
| `chronicle.summarize {run, range?, filter?}` | `RunSummary` via `engine_narrator`. |
| `agent.follow {run, agent, from?, to?, detail}` | Streams `AgentEpisode`s. |
| `agent.timeline {run, agent}` | Episode headers only. |
| `agent.state_history {run, agent, fields[]}` | Time-series of named fields (hp, emotions[], relationships). |
| `encounter.list {run, filter?}` | Paginated encounter headers. |
| `encounter.describe {run, encounter_id, style?, include_causal?}` | Full narrative, optionally inlined with causal annotations. |

### 5.4. Pub/sub events

All with `(dsl_hash, schema_hash, timestamp)`:

- `run_started {run_id, seed, ticks_planned}`
- `run_completed {run_id, tick_count, status}`
- `run_progress {run_id, tick}` (throttled)
- `rule_reloaded {files_changed[], rule_ids_affected[]}`
- `schema_changed {diff, cargo_rebuild_required: true}`
- `parse_error {errors[]}`
- `probe_ready {run_id}` (after instrumentation-enabled replay)

### 5.5. DSL hash and schema hash

- `schema_hash`: combined hash per `docs/dsl/spec.md` §4. Derived
  from entity layouts, event taxonomy, rules/masks/verbs, scoring
  declarations. Bumps on any F change.
- `dsl_hash`: covers A–E rule source. Computed over the canonical
  pretty-printed form from `dsl_ast` §4.1 so reformatting and
  comments do not change it. Bumps on any A–E change.
- Both stamped into every trace, every `Diff`, every `Narrative`,
  every protocol response. If a trace is requested with a mismatched
  `schema_hash`, the request fails per `docs/compiler/spec.md` §2.

---

## 6. Data flows

### 6.1. Standard dev iteration (hot path)

```
agent edits DSL file
  └─▶ engine_dev file-watcher → re-parse delta (debounced)
        ├─ schema changed → parse_error event { cargo_rebuild_required: true } ; halt
        └─ A–E only     → mark dsl_hash dirty

agent calls run { seed, ticks, mode: "interpreted" }
  └─▶ engine_dev: compile-if-needed (interpreter: no cargo) → run → archive
        → emit run_completed { run_id, dsl_hash, schema_hash }

agent calls diff.get { run_prev, run_current } → Diff
agent calls chronicle.summarize { run_current } → Narrative
agent calls probe.agent | agent.follow | encounter.describe  (as needed)
```

### 6.2. Schema change (cold path)

```
agent edits entity / event / action schema
  └─▶ file-watcher → schema_hash changed
        └─ engine_dev emits schema_changed { diff, cargo_rebuild_required: true }
           (refuses new runs until rebuilt; keeps serving archive queries)

agent runs cargo build externally (or invokes compile-dsl xtask)
  └─▶ engine_dev SIGHUP or protocol `reload` verb → pick up new binary → resume
```

### 6.3. Viz follow-along

```
viz connects to UDS → subscribes to run_completed, rule_reloaded
  └─▶ engine_dev pushes events
        └─ viz fetches Narrative + Diff + trace → renders
```

### 6.4. Causal probe

```
agent asks probe.event { run, event_id }
  └─▶ engine_dev:
        1. find nearest snapshot ≤ event.tick
        2. replay to event.tick in interpreted mode with instrumentation on
        3. filter instrumentation ring for event_id
        4. return CausalTrace { rule_id, inputs_snapshot, upstream_events[] }
```

The interpreter instrumentation is opt-in per run (off by default
for throughput). Probe requests against a run archived without
instrumentation trigger a re-run from the relevant snapshot with
instrumentation on; the probe re-run is a satellite keyed by
`(source_run_id, tick_range)` and does not replace the original
archived trace. Subsequent probes in the same range reuse the
satellite.

### 6.5. Trace archive

- On-disk. Each run has a unique `run_id` (UUID); the daemon also
  maintains a lookup index keyed by `(seed, dsl_hash, schema_hash)`
  so identical re-runs can be short-circuited at the daemon's
  discretion. Determinism guarantees the output would match; the
  index is an optimization, not a correctness requirement.
- Trace ring-buffer per the engine spec, plus snapshots every N=500
  ticks (engine spec §19).
- LRU cap: 10 GB default, config override. Eviction by `archived_at`
  ascending.
- `state.at` replays from nearest snapshot; no eager per-tick caching.
- Eviction of a run referenced by an in-flight request returns a
  clean `{ code: "archive_evicted", run_id }` error.

---

## 7. Error handling

Every failure round-trips through the protocol as structured data.
Five error surfaces:

### 7.1. DSL parse / lower errors

Shape: `{ kind, span: {file, line, col, len}, message, hint? }`.
Returned on `parse_error` events and in `errors[]` of any verb that
parses. Daemon holds last-good AST for live queries and refuses
`run` until fixed, with `{ code: "parse_dirty", errors[] }`.

### 7.2. Schema-hash mismatch

Shape: `{ code: "schema_changed", diff: {added[], removed[],
reordered[]}, cargo_rebuild_required: true }`. Daemon keeps serving
queries against archived runs; only `run` is refused.

### 7.3. Interpreter runtime errors

Shape: `{ code, rule_id, agent_id, tick, inputs_snapshot, ... }`.
Codes: arithmetic overflow, divide by zero, missing field read,
type-tag mismatch, cascade depth exceeded, recursion bound hit.
Default disposition: **fatal-for-run** (aborts, archives with
`status: failed`). Configurable per rule class to `rule-skipped`
(logged to `run.anomalies[]`, rule is skipped for that eval).

### 7.4. Protocol / daemon errors

Shape: `{ code, message, retry_after_ms? }`. Codes: `unknown_verb`,
`malformed_args`, `unknown_run`, `archive_evicted`,
`resource_exhausted`, `parse_dirty`. Closed enum so agents can
pattern-match.

### 7.5. Compile-path errors (static mode + schema rebuilds)

Shape: `{ code: "compile_failed", tool: "cargo" | "shaderc", log:
string, parsed_diagnostics?: [...] }`. Raw log always included;
structured parse best-effort.

---

## 8. Determinism contract

- `InterpretedEvaluator` + `SerialBackend` must produce
  byte-identical `replayable_sha256` against `StaticEvaluator` +
  `SerialBackend` for the fixture corpus (§9.1).
- Interpreter uses the same RNG stream and unit-processing order
  discipline as the static path. No extra RNG consumption from
  instrumentation.
- Narrator template output is a pure function of the trace; same
  inputs yield byte-identical bytes.
- `engine_diff` is a pure function; same inputs yield byte-identical
  `Diff`.
- GPU backend is out of scope; its own parity story (Serial-static ↔
  GPU-static) stays untouched and aspirational.

---

## 9. Testing

### 9.1. Cross-evaluator parity (load-bearing)

Fixture corpus of 5–10 small DSL programs, each stressing one rule
class (A–E). For every program: run N ticks with both evaluators on
`SerialBackend`, assert byte-identical `replayable_sha256`. Runs on
every PR; failure blocks merge. Corpus expands with new rule
primitives.

### 9.2. Protocol contract tests

Every verb in §5 has a fixture pack:

- One success case with a golden JSON response.
- One case per error code it can emit.

Goldens checked in. Diff on change is a reviewable signal that the
agent-facing surface shifted.

### 9.3. Interpreter unit tests

One test file per rule class in `engine/src/evaluator/`. Covers:

- Happy path against synthetic SoA.
- Each runtime error (§7.3).
- Instrumentation: known rule + inputs → assert probe record
  contents.

### 9.4. Diff / narrator / probe integration

Synthetic two-run fixtures with hand-crafted divergences. Assert:

- `engine_diff` flags expected divergence tick + event deltas.
- `engine_narrator` produces stable prose against golden templates.
- `probe.event` returns a causal trace whose `contributing_inputs`
  match the fields the rule actually reads.

### 9.5. Archive / lifecycle

- LRU eviction at size threshold; oldest by `archived_at` goes first.
- `state.at` on evicted run returns clean error, no panic.
- File-watcher debounce: save-in-flight does not trigger mid-write
  re-parse.
- Long-running daemon memory ceiling snapshot in CI.

### 9.6. LSP contract tests

- Diagnostics fixtures: each diagnostic kind has a bad-source
  fixture + expected diagnostic output.
- Navigation: go-to-definition and find-references on a canonical
  DSL program.
- Code-action fixtures: apply the action, diff against the golden.

### 9.7. CI matrix

- Debug + release (matches engine determinism policy).
- Interpreted-only, static-only, both.
- Linux x86_64. Windows deferred to Phase 3.

---

## 10. Build sequence

Phase 1 subtasks, in recommended build order:

1. Extract `dsl_ast` from `dsl_compiler` (no behavior change; both
   crates compile against it).
2. `InterpretedEvaluator` skeleton; plug into engine pipeline behind
   a config flag; parity test on minimal fixture.
3. Expand interpreter to cover rule classes A–E; fixture corpus to
   5–10 programs; parity green.
4. `engine_dev` daemon skeleton: `run`, `runs.list`, trace archive,
   file-watch, UDS + stdio transport.
5. `engine_diff` crate; wire into `diff.get`.
6. `engine_narrator` crate (templates only); wire into
   `chronicle.summarize`, `agent.follow`, `encounter.describe`.
7. Interpreter instrumentation + `probe.*` verbs.
8. Introspection verbs: `schema.*`, `rules.*`, `lint.*`, `scaffold.*`.
9. `lsp_server` crate: diagnostics + navigation + completion + code
   actions. Inlay hints wired to `rules.last_run_stats`.
10. Minimum-viable viz compare mode + inspector panel + chronicle
    panel.
11. Contract test suite (§9.2, §9.6) green.
12. Phase 1 ships.

Phase 2 and 3 items sequence separately; not committed by this spec.

---

## 11. Open items

None currently. Any surprises from the implementation plan that
contradict this spec should bounce back here for amendment, not be
silently absorbed.
