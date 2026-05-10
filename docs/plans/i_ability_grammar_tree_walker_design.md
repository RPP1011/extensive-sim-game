# Plan I — Ability-Grammar Tree Walker + Valid-Grammar Fuzzer

> Status: design. Implementation lands after Plan H.

## Goal

A tree-walker that exhaustively traverses the `.ability` AST grammar,
generates synthetic `.ability` files exercising every valid
production, and asserts the full pipeline (parse → resolve → lower →
emit → naga validate) accepts each one. Catches regressions where a
grammar variant compiles but produces invalid output, OR where a new
parser feature adds a production but the lowering / emit doesn't
support it.

The corpus we have today (`dataset/abilities/lol_heroes/*.ability` —
172 files) covers historical `.ability` shapes but doesn't
systematically exercise the grammar's ALL valid combinations. The
tree-walker fills that gap.

## Architectural Impact Statement

* **P1 (Compiler-First):** PASS. The walker is a host-side test
  harness; no runtime engine code paths are added.
* **P2/P3/P5/P10:** N/A — the walker doesn't touch sim semantics.
* **Coverage P:** every reachable AST production is exercised at
  least once. Surfaces gaps where the grammar accepts a shape the
  lowering rejects (a `LowerError::Unsupported*` variant).

## Tree-walker design

The grammar is defined by:
* `crates/dsl_ast/src/ast.rs::AbilityDecl` (top-level production).
* `crates/dsl_ast/src/ast.rs::AbilityHeader` (10+ header variants:
  Target, Range, Cooldown, Cast, Hint, Cost, Charges, Recharge,
  Toggle, Recast, RecastWindow).
* `crates/dsl_ast/src/ast.rs::EffectStmt` (29 verb dispatch arms +
  modifier slots: tags, area, scaling, when, chance, stacking,
  lifetime, nested).
* `crates/dsl_ast/src/ast.rs::AbilityProgramStep` (Cast / Effects).
* `crates/dsl_ast/src/ast.rs::CastSpec` + `InterruptSet` shapes.

The walker lives in `crates/dsl_ast/tests/grammar_tree_walker.rs`
(or a separate `grammar-fuzzer` crate if it grows large). Its
algorithm:

```rust
fn walk_grammar() -> Vec<String> {
    let mut authored: Vec<String> = Vec::new();
    for header_combo in iterate_header_combos() {        // ~40 shapes
        for body in iterate_bodies() {                    // 29 verbs * modifier slots
            for program_shape in [None, Some(cast_only), Some(cast_effect),
                                   Some(cast_effect_cast_effect)] {
                authored.push(synthesize_ability_text(header_combo, body, program_shape));
            }
        }
    }
    authored
}
```

The synthesis loop is bounded: not Cartesian over every modifier
combination (would explode to millions), but rather a **per-axis
sweep** — for each axis (header / verb / modifier), iterate its
variants while holding others at default. Total: ~500-1000
synthesized files (manageable).

Each authored file goes through:

```rust
let parsed = parse_ability_file(&authored).expect("parse");
let lowered = lower_ability_decl(&parsed.abilities[0]).expect("lower");
// emit + naga validate happen via existing apply_ability_smoke harness
```

Failures = the test panics with the failing file printed. Author can
copy-paste into a permanent regression test.

## Hot-reload variant (deferred)

The optional second mode: rather than authoring files in memory,
write them to disk, point a runtime crate at the file, exercise
hot-reload (re-parse + re-emit on change). This requires:

1. A runtime crate that exposes a `reload_ability(path: &Path)`
   method.
2. File-system watching (e.g. `notify` crate) that triggers reload.
3. The runtime's AbilityRegistry can swap an ability's program
   in-place — needs a new `update(id: AbilityId, program:
   AbilityProgram)` API on the registry.

Hot-reload is plumbing-heavy (file watching + registry mutability)
and has marginal value for the grammar-coverage goal — the
in-memory walker is sufficient. **Defer hot-reload to Plan I-step-2.**

## Implementation slices

1. **I1 — In-memory walker.** Iterate header combos × verb arms ×
   program shapes. For each, synthesize text and run through
   parse + lower + emit. Test in `dsl_ast/tests/`.
2. **I2 — Coverage report.** The walker emits a coverage matrix:
   "production X exercised by N synthesized files". Surfaces gaps
   where a production isn't reachable from the synthesis loop.
3. **I3 — Hot-reload (deferred).** Per the disclaimer above.

I1 ~3-4 hours; I2 ~1 hour; I3 deferred.

## Why this matters

Right now, the `.ability` grammar is implicitly defined by the
parser + 172 LoL corpus files. New grammar additions (Plan G's
`cast { } effect { }` blocks; recast/recast_window; scaling
modifiers) get one ad-hoc test each, but no systematic coverage of
ALL the valid combinations. A regression that breaks `cooldown:
5s @ resolve` (a Plan G CooldownPhase shape) might slip through if
no LoL corpus file uses that exact combo.

The tree-walker IS the systematic coverage — every valid grammar
shape gets exercised exactly once per test run. Failures localize
to a single synthesized file the author can copy into a permanent
regression.
