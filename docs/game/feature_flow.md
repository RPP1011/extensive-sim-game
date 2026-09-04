# Feature flow

How to add a new game feature. The path is compiler-first: every feature enters the codebase as
DSL source (`.sim`) compiled through `dsl_ast` + `dsl_compiler` into Rust + WGSL. There is no
hand-write-then-compile loop for the compiled kinds; where the compiler can't emit a shape yet,
extend the compiler.

This file describes the flow as it exists today, after the Phase 7 "wolf-sim wipe" (2026-05-02)
retired the `xtask` umbrella binary and the `crates/engine_rules` crate. Neither exists any more.
If you're reading an old commit, a stale doc, or `docs/game/compiler_progress.md` (also stale on
this point — it still describes `xtask compile-dsl` and `engine_rules`), do not trust it over this
file or the source of truth it's grounded in: `crates/sims/CLAUDE.md`, `crates/dsl_compiler/CLAUDE.md`,
`crates/dsl_ast/CLAUDE.md`, `crates/engine/CLAUDE.md`, `crates/engine_data/CLAUDE.md`.

## Two scenarios

**(A) The compiler already supports the declaration kinds your feature needs.** Most features
land this way once the compiler matures. Write DSL source, build, test.

**(B) The compiler does not yet support one of the declaration kinds you need.** Extend the
compiler first (grammar in `dsl_ast`, lowering + emission in `dsl_compiler`'s `cg::*` pipeline),
then write the DSL source. This is where the compiler milestones from `compiler_progress.md` get
advanced (bearing in mind that doc's own emission-pipeline details are stale — trust the code).

Both paths end the same way: a commit that lands DSL source + a fixture pin test asserting
behavior. Unlike the old flow, the emitted Rust/WGSL is **not** part of that commit — it isn't
checked in at all (see "Where compiled output lands" below).

## Scenario A — feature in supported kinds

```
1. Write the DSL source
   └─ assets/sim/<feature>.sim

2. Register the fixture
   └─ add "<feature>" as a new |-joined arm to the `matches!` allowlist
      in crates/sims/build.rs

3. Build
   └─ cargo build -p sims
      build.rs calls dsl_compiler::build_helper::emit_namespaced("<feature>"),
      which parses → resolves (dsl_ast) → lowers → schedules → emits (dsl_compiler's
      cg::* pipeline), writing generated Rust + WGSL into OUT_DIR (build-time only,
      never checked in). This produces sims::<feature>::GeneratedRuntime.

4. Write a fixture (pin) test
   └─ crates/sims/tests/<feature>_pin.rs
      seeded GeneratedRuntime → step N ticks → assert exact/hardcoded values
      (host-side SoA columns and/or @materialized view storage)

5. cargo test -p sims <feature>_pin
6. Commit (DSL source + pin test — NOT the generated output, which isn't checked in)
```

Forgetting step 2 is the single most common mistake here: the crate still builds, but
`sims::<feature>` doesn't exist and the test in step 4 fails to compile with an unresolved-module
error that gives no hint the fixture was silently skipped by the allowlist gate. See
`crates/sims/CLAUDE.md` for the full allowlist mechanics, pin-test conventions, and pitfalls
(stack size for deep dispatch chains, GPU-adapter-unavailable skip behavior, etc.).

Two crates (`crates/tom_probe_runtime`, `crates/viewer_runtime`) still follow the older
one-crate-per-fixture pattern predating the `sims` mega-crate consolidation — each compiles its
own `.sim` via its own `build.rs`/`OUT_DIR`. New fixtures go into `sims`; don't add a new
per-fixture crate.

The generated output is not committed and not reviewed directly. Reviewers read the `.sim` source
and the pin test's asserted values; if the emitted output looks wrong, the fix is in
`dsl_compiler`'s lowering/emission code (`crates/dsl_compiler/src/cg/lower/*`,
`crates/dsl_compiler/src/cg/emit/*`), never in a patched `OUT_DIR` file — it's silently overwritten
on the next build anyway.

## Scenario B — compiler extension required

```
0. Identify the missing declaration kind (or the missing shape within an
   existing kind — e.g. @materialized views if only @lazy is supported).

1. Extend the frontend, if the grammar itself is missing
   └─ crates/dsl_ast/src/
      - parser.rs / ast.rs: recognize the new syntax, add an ast::Decl variant
      - resolve.rs / ir.rs: typed IR shape for the new decl kind

2. Extend the backend
   └─ crates/dsl_compiler/src/cg/
      - lower/<kind>.rs: IR → Compute-Graph IR (one module per declaration kind
        already exists for mask, view, physics, scoring, spatial, terrain, etc.)
      - emit/kernel.rs, emit/wgsl_body.rs, emit/program.rs: CG IR → WGSL + Rust

3. Add a test for the new lowering/emission shape
   └─ crates/dsl_compiler/tests/<kind>_*.rs
      No golden/snapshot files are committed (ahash drift caveat, see that
      crate's CLAUDE.md) — assert `rust_src.contains("...")` / `wgsl.contains("...")`
      substrings, or run emitted WGSL through naga to confirm it parses.

4. Confirm milestone row in docs/game/compiler_progress.md flips to 🚧
   (treat that doc's specific command/path claims as unreliable — only trust
   its milestone status column, not its "how it works" prose)

5. Proceed through scenario A steps above for the actual feature.

6. On commit: milestone row flips to ✅ IF this feature's fixture tests
   pass AND the legacy code it replaces is deleted in the same commit.
```

Extending the compiler is slower than writing Rust directly. That is the cost of the
compiler-first discipline. The return: every declaration kind that lands is permanent, and the
emission shape is proven by being used.

## Legacy replacement

Some features still exist as hand-written Rust — e.g. `crates/engine`'s `ability/*.rs` and
whatever remains of `policy/utility.rs`. Per `crates/engine/CLAUDE.md`, `creature.rs` and `step.rs`
are no longer live hand-written logic: `creature.rs` is now a doc-comment-only stub (its vocabulary
moved to `engine_data::entities::CreatureType`), and `step.rs`/`backend.rs`/`probe/mod.rs` are
`unimplemented!()` compile-only stubs left over from Plan B1' Task 11 — they are not a legacy
system you migrate away from, they're dead code pending cleanup. Don't assume every file that used
to hold hand-written game logic still does; check the crate's own CLAUDE.md before treating
something as "the legacy handler to delete."

Where legacy hand-written game logic genuinely does still run, the rule holds: when a compiler
milestone covers it, the SAME commit that lands compiler support must delete the legacy handler.
No parallel paths. Fixture tests that currently exercise legacy code will now exercise
compiler-emitted code and must still pass.

If a feature spans multiple declaration kinds (an event + a physics rule + a mask + a scoring
entry), all of them move together. Partial migration is not a valid state — a mask predicate won't
compile against an event type that no longer has an emitter.

## Where compiler output actually lands today

There is no single checked-in destination for "compiler output" any more — that was the
`engine_rules` model, and it's gone. What happens to each declaration kind's output depends on
whether it's part of a per-fixture compile or part of the small shared vocabulary crate:

| Declaration kind | Where it actually lives today | Notes |
|---|---|---|
| `event` (per-fixture, in a `.sim` a specific fixture owns) | `OUT_DIR` via `crates/sims` (or a legacy `crates/*_runtime` crate), build-time only, never checked in | Lowered by `dsl_compiler::cg::lower::event_binding` as part of that fixture's compile. |
| `event` (the shared core vocabulary — `Event`, `EventKindId`, structs like `AgentAte`/`AgentDied`) | `crates/engine_data/src/events/` | Hand-maintained today, NOT live-regenerated. Files carry a `// GENERATED by dsl_compiler` header that is a **misleading historical artifact** — see "The `// GENERATED` header" below. `engine` consumes this via `crates/engine/src/event/event_like_impl.rs`'s `impl EventLike for engine_data::events::Event` (the one carve-out `engine`'s build.rs allows for a `// GENERATED`-headed file). |
| `physics` | `OUT_DIR` via `crates/sims` (or a legacy runtime crate), build-time only, never checked in. **No `crates/engine/src/generated/physics/` directory exists.** | Lowered by `cg::lower::physics`, emitted as WGSL kernels + Rust dispatch/binding glue by `cg::emit::*`. `crates/engine`'s `cascade::{CascadeRegistry, CascadeHandler, Lane}` supply only the primitives the emitted code targets; `engine` itself never holds generated physics handlers (its own build.rs forbids `// GENERATED` files outside one exempted path). |
| `mask` | Same as `physics`: `OUT_DIR` via a per-fixture compile, never checked in. No dedicated checked-in location. | Lowered by `cg::lower::mask`; `engine::mask::MaskBuffer` is the primitive it targets. |
| `scoring` | `crates/engine_data/src/scoring/` (`SCORING_TABLE`, `ScoringEntry`/`ModifierRow`/`PredicateDescriptor`) | Hand-maintained POD data today — **not** a "dedicated generated-only crate split out of `engine_rules`" (that used to be this table's story; it no longer applies now that `engine_rules` doesn't exist). Same `// GENERATED` provenance caveat as `event` above: the header is present but stale/misleading. |
| `config` | `crates/engine_data/src/config/` (`CombatConfig`, `MovementConfig`, `NeedsConfig`, `CommunicationConfig`, `BeliefConfig`, aggregated into `Config`) | Same hand-maintained-despite-header situation as `event`/`scoring`. |
| `entity` (the shared `CreatureType`/`Capabilities` vocabulary) | `crates/engine_data/src/entities/` | Same situation again: hand-maintained, `// GENERATED`-headed, no live `.sim` source (`assets/sim/entities.sim` cited by the header does not exist in the repo). Fixed to 4 variants (Human/Wolf/Deer/Dragon) — see `crates/engine/CLAUDE.md`'s "Wolf-sim coupling" section. |
| `view` (`@materialized`/`@lazy`/topk) | `OUT_DIR` via a per-fixture compile, never checked in | `engine::view::{MaterializedView, LazyView, TopKView}` are the trait primitives a fixture's generated view implements. |

If you need the precise current mapping for a kind not in this table (`verb`, `invariant`, `probe`,
`metric`, `spatial_query`, `terrain`, `table`, `belief`, ...), check `crates/dsl_compiler/CLAUDE.md`'s
"Internal architecture" section (lists every `cg/lower/*.rs` module) before assuming it follows one
of the two patterns above.

## The `// GENERATED` header: two different, non-interchangeable stories

Do not conflate these:

1. **Live, build-time generation.** `crates/sims/build.rs` (and the two remaining legacy
   `crates/*_runtime` build scripts) really do call `dsl_compiler::build_helper::emit_namespaced`
   at every build, compiling `assets/sim/*.sim` fresh into `OUT_DIR`. This output is never checked
   in; there is nothing to hand-edit because the next build overwrites it unconditionally. If it
   looks wrong, fix `dsl_compiler`'s lowering/emission, not the file.

2. **A frozen historical header on hand-maintained files.** `crates/engine_data/src/{events,scoring,config,entities}/**` carry a `// GENERATED by dsl_compiler` header (plus a second line noting `xtask compile-dsl` used to regenerate them) but are, per `crates/engine_data/CLAUDE.md`'s "Provenance" section, **100% hand-maintained today with no live regeneration path** — the `.sim` files the headers cite (`entities.sim`, `enums.sim`, `events.sim`, `config.sim`) do not exist anywhere in the repo. The header survives only because `crates/engine_data/build.rs` (a sentinel, not a generator) and `.githooks/pre-commit` both hard-require its presence on any file outside a small hand-written allowlist. Editing these files by hand is the *normal* way to change them now — just keep the header string intact or the pre-commit hook blocks the commit.

When writing or reviewing code that touches `engine_data`, treat every `// GENERATED`-headed file
there as story (2): hand-edit it directly, same as any other Rust file, and don't go looking for a
`.sim` source or a regeneration command that will bring it back in sync — none exists.

## Shape contract for compiler-emitted code

Every emitted Rust/WGSL file (story (1) above) starts with a header of the form:

```
// GENERATED by dsl_compiler — do not edit by hand.
```

(exact wording varies slightly per emitter — `crates/dsl_compiler/CLAUDE.md` notes
`schema_hash.rs`'s variant as an example; copy whichever existing header string is closest rather
than inventing a new one if you add an emitter). Reviewers can't skim this output directly since
it isn't checked in, but the same discipline applies when reading it locally out of `target/**/out/`:
if the emitted code has helper functions, macros, or trait abstractions the DSL declaration didn't
ask for, that's a signal to push back on the emitter, not to hand-patch the output. The compiler's
output should be verbose and mechanical — that's what makes it auditable when you do go look at it.

## Forbidden shortcuts

- **Do not hand-author files under `OUT_DIR`.** They're regenerated unconditionally on every build
  of `crates/sims` (or a legacy `crates/*_runtime` crate); nothing you write there survives. If you
  need different output, change the `.sim` source or extend `dsl_compiler`.
- **Do not assume a `// GENERATED`-headed file in `crates/engine_data` has a live source to edit
  instead.** For those files (see above), hand-editing the `.rs` file directly *is* correct — but
  keep the header string intact, since the pre-commit hook requires it.
- **Do not add game logic to `crates/engine`.** The engine contains primitives (cascade runtime,
  SoA, spatial, event ring, RNG, schema hashing, GPU platform layer). Its own `build.rs` enforces
  this structurally: a closed allowlist of top-level `src/` files/dirs, and a hard ban on any
  `// GENERATED by dsl_compiler` marker under `src/` except one narrow carve-out
  (`event/event_like_impl.rs`). Adding `ATTACK_DAMAGE = 12.0` to anything under `crates/engine` is
  a regression even if "temporary."
- **Do not skip the fixture (pin) test.** Every feature lands with a seeded scenario and
  hardcoded/asserted expected values. Without it, there's no safety net for the DSL-owned behavior.
- **Do not mix compiler-extension and feature-add in one commit.** Land the compiler extension
  first (with its own lowering/emission test), then use it for the actual feature. Keeps review
  reviewable.

## Schema-hash implications

Adding an event, entity field, scoring row, or physics rule can bump the relevant schema sub-hash.
`crates/engine::schema_hash` computes a SHA-256 fingerprint over a hand-maintained description of
every layout-relevant type; `crates/engine/.schema_hash` pins the baseline, and
`crates/engine/tests/schema_hash.rs` fails CI on drift. `crates/dsl_compiler/src/schema_hash.rs`
separately hashes the byte content of every `// GENERATED` file to drive that regeneration check.
Because the engine-side description is hand-written, it can silently under- or over-cover reality
(see `crates/engine/CLAUDE.md`'s note that nothing currently proves e.g. `ItemId` size flows into
the hash) — when you change a layout-relevant type, update both the literal and the baseline file
in the same commit, and don't assume a passing schema-hash test proves full coverage.
