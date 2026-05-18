# Parameterised Rules — Design

**Status:** Design — awaiting user review before plan write-up.
**Date:** 2026-05-17
**Companion plan (TBD):** `docs/superpowers/plans/2026-05-1x-parameterised-rules-*.md`.
**Related shipped specs:**
`docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md` —
this design depends on multi-file imports so stdlib rules can be
shared across fixtures.

## Summary

Extend the existing `physics <Name> @phase(...) { body }` rule decl
to accept parameters: `physics chase(target: EntityKind, aggro: f32,
speed: f32) @phase(per_agent) { ... }`. Fixtures use the rule by
writing `physics HunterChase = chase(target: Wolf, aggro: 15.0,
speed: 1.5);` — each application monomorphises the parameterised rule
with concrete arg values and produces one emit-ready concrete rule.
Parameters are limited to scalars (`f32`, `i32`, `u32`, `bool`) and
`EntityKind` references in v1. No closures, no captures, no
higher-order parameters.

## Motivation

The DSL today has no parameterisation. Every `physics` decl is a
single concrete rule with inline literal values:

```text
physics WolfHunt @phase(per_agent) {
  on Tick {} {
    let aggro = 8.0;
    // ... hard-coded body for wolves hunting sheep
  }
}

physics HunterStalk @phase(per_agent) {
  on Tick {} {
    let aggro = 15.0;
    // ... duplicated body for hunters stalking wolves
  }
}
```

The two rules implement the same chase algorithm with different
constants. Today the only options are (a) copy-paste the body, (b)
write one rule with branching on `agents.kind(self)` (couples the
rule to specific entity kinds). Neither composes well across the
~80 fixtures the workspace already has.

The shipped multi-file imports system gives us cross-fixture
content sharing, but you can't share a single named definition
without committing every importer to its exact constants. Parameters
close that gap.

## Out of scope (this spec)

- **First-class functions / function values.** Function-typed
  parameters (`predicate: fn(Agent) -> bool`), rule composition
  combinators (`sequential`, `gated`), closure captures. The
  parameterised-rule model handles ~all current shared-rule needs;
  FCF would be significant compiler work (defunctionalisation,
  scoping rules, function-type checks) for marginal short-term
  benefit. If concrete combinator use cases surface later, design
  FCF then with that evidence in hand.
- **Entity tags / groupings.** A rule's `target` is a single
  `EntityKind`, not a group. Polymorphic dispatch over groups is a
  follow-up.
- **Default values for parameters.** Every application supplies
  every parameter explicitly. Default-value support can be added
  later without grammar churn.
- **Positional args.** Application uses by-name args only.
  Positional support is a future ergonomic addition.
- **Enum-typed parameters.** v1 covers scalars + `EntityKind`. Enum
  params can be approximated by `u32` until a follow-up extends the
  type set.

## Design

### Architecture & integration

- The DSL grows two new top-level decl shapes (both reusing the
  existing `physics` keyword):
  - **Parameterised rule**: `physics <name>(<params>) @phase(<phase>)
    { body }` — has parameters, has a body, does NOT directly
    produce a kernel.
  - **Rule application**: `physics <NewName> = <RuleName>(<args>);` —
    no params, no body, just a name binding to the result of
    applying a parameterised rule. Produces exactly one emitted
    kernel.
- Existing zero-param `physics <Name> @phase(<phase>) { body }`
  decls remain unchanged — no migration cost.
- A new monomorphisation pass between merge and rule-lowering walks
  applications, substitutes args into the parameterised rule's
  body, and produces concrete rule IR. The existing rule emitter,
  schedule builder, and parity tests see only concrete rules — no
  awareness of the template/application distinction.
- Source-span attribution is preserved through substitution: each
  substituted node carries both its in-template span AND the apply
  site's span so errors point at both ends of the type contract.

| Form | AST shape | Emits a kernel? |
|---|---|---|
| Parameterised rule | params present, body present, no `=` | No |
| Rule application | no body, has `=` | Yes (substituted from parameterised rule) |
| Concrete rule (existing) | no params, body present, no `=` | Yes (unchanged) |

### Syntax & grammar

```text
// Parameterised rule decl
physics chase(target: EntityKind, aggro: f32, speed: f32)
              @phase(per_agent) {
  on Tick {} {
    // Param refs in body are ordinary identifiers; resolver tags
    // them as "param refs" that get substituted at apply time.
    let dx = nearest_of_kind(target).x - agents.pos(self).x;
    let dy = nearest_of_kind(target).y - agents.pos(self).y;
    if (dx * dx + dy * dy < aggro * aggro) {
      // Apply movement at `speed` toward the target...
    }
  }
}

// Applications (by-name args only)
physics HunterChase = chase(target: Wolf,  aggro: 15.0, speed: 1.5);
physics WolfChase   = chase(target: Sheep, aggro:  8.0, speed: 1.0);

// Existing zero-param form unchanged
physics MoveBoid @phase(per_agent) {
  on Tick {} { /* ... */ }
}
```

**Grammar rules:**

- Param list `(<ident>: <Type>, ...)` after the name. `<Type>` is
  one of `f32`, `i32`, `u32`, `bool`, `EntityKind`.
- Each param's `<ident>` is unique within the rule's param list.
- Param refs in the body are ordinary identifiers; they look like
  locals but the resolver tags them as `ParamRef(name)` nodes that
  the monomorphisation pass substitutes.
- Top-level decls remain order-independent. A parameterised rule
  and its applications may appear in any order across files.
- Application names share the existing `physics` namespace — they
  go through the same collision pass as concrete rules.

### Monomorphisation & lowering

New pass `lower_param_rules` between `parse_with_imports` and the
existing rule-lowering stages.

1. **Catalog parameterised rules.** Collect every `physics <name>(<params>)
   { body }` into a side table keyed by `<name>`. These produce no
   downstream IR.
2. **Walk applications.** For each `physics <NewName> = <RuleName>(<args>);`:
   - Look up `<RuleName>` in the catalog. Missing →
     `LowerError::UnknownParameterisedRule { name, site }`.
   - Type-check args against params (count, names, types). Mismatch →
     `LowerError::ApplicationTypeMismatch { rule, param, expected, actual }`
     or `LowerError::ApplicationParamMismatch { rule, missing, extra, duplicates }`.
   - **Substitute** param references in the body with arg values.
     Substitution is purely AST-level: a `ParamRef("aggro")` node
     becomes `Literal(15.0)`. `EntityKind` params become
     `EntityKindLit(<id>)`. The result is a complete concrete-rule
     AST identical in shape to a hand-written zero-param decl.
   - Tag the substituted body with `<NewName>` and the
     parameterised rule's `<phase>`.
   - Push the resulting concrete rule into the existing rule-decl
     list as if the user had written it inline.
3. **Hand the rewritten Program to the existing lowering pipeline.**

**Why AST-level substitution is sufficient:**

- The DSL already emits constants inline in WGSL. After substitution,
  `aggro * aggro` reads as `15.0 * 15.0` in WGSL source; the shader
  compiler folds it to a constant. No runtime cost, no indirect
  dispatch.
- Substitution preserves source spans by attribution: substituted
  nodes carry both their original-in-template span AND the apply
  site span. Emit-time errors point at the right line in both files.

**Determinism (P5/P11):**

- Application order in source = monomorphisation order. Each
  application produces exactly one concrete rule with a
  deterministic name.
- No new RNG paths; param values are compile-time constants after
  substitution.
- The kernel scheduler iterates the resulting concrete rule list in
  source order, as today.

**Interaction with multi-file imports:**

- All three forms (parameterised rule decl, application, existing
  concrete rule) live in the same `physics` kind for collision-table
  purposes in v1. Single shared namespace.
- Two `physics`-keyword decls with the same name → `DuplicateDefinition`
  per the existing multi-file collision pass, regardless of form.
- This means: a parameterised rule and a concrete rule with the
  same name collide; an application and a concrete rule with the
  same name collide; two parameterised rules with the same name
  collide. Splitting the namespace (e.g. so a stdlib parameterised
  `chase` doesn't collide with a fixture's concrete `chase`) is a
  future ergonomic improvement; v1 keeps the single namespace for
  simplicity.

### Type checking & validation

A new validation pass between merge and monomorphisation.

**Per parameterised rule:**

- Every param name is a valid identifier.
- Param name uniqueness within the param list.
- Param types are in the v1 set (`f32`, `i32`, `u32`, `bool`,
  `EntityKind`). Anything else → `LowerError::UnsupportedParamType
  { rule, param, ty }`.
- Body references to param names resolve cleanly via the existing
  resolver (param refs piggyback on the local-identifier resolution
  path).

**Per application:**

- Referenced parameterised rule exists in the merged Program.
- Every required param is provided exactly once (no missing, no
  duplicate, no extras).
- **v1 args must be literals.** Scalar args are numeric / boolean
  literals (`15.0`, `true`, `42`); `EntityKind` args are bare
  identifiers naming a top-level `entity` decl. Non-literal expressions
  (`aggro: x + 1`, `target: choose_target()`) are rejected with
  `LowerError::NonLiteralArg { rule, param, site }`. Restricting to
  literals keeps the monomorphisation pass purely structural and
  defers the "compile-time expression evaluation" question to a
  future spec.
- Each arg's type matches the corresponding param's declared type.
  Numeric literal widening/narrowing uses the existing literal-coercion
  rules (e.g. `aggro: 15` coerces to `f32`).
- `EntityKind` args resolve to a known top-level `entity` decl.
  Unknown → `LowerError::UnknownEntityKind { rule, param, name, site }`.

**Application name uniqueness:**

- Handled by the existing multi-file collision pass (kind `physics`).
  No new logic needed.

**Error messages:**

- Every error carries both source spans:
  > `error: type mismatch in application 'HunterChase'`
  > `→ chase declared 'aggro: f32' at stdlib/rules/chase.sim:3:18`
  > `→ apply passed 'true' at fixture.sim:12:35`

## Testing strategy

| Test | Location | What it pins |
|---|---|---|
| AST: ParamDecl + ParamRef nodes | `crates/dsl_ast/tests/param_rule_node.rs` | New AST types construct + field-access cleanly. |
| Parse parameterised rule decl | `crates/dsl_compiler/tests/param_rule_parse_decl.rs` | `physics chase(target: EntityKind, aggro: f32) @phase(per_agent) { ... }` parses, params land in AST. |
| Parse application | `crates/dsl_compiler/tests/param_rule_parse_apply.rs` | `physics HunterChase = chase(target: Wolf, aggro: 15.0);` parses, args land by-name. |
| Reject unsupported param type | same file | `physics foo(x: SomeWeirdType) ...` → parse or validate error. |
| Reject duplicate param name in decl | same file | `physics foo(a: f32, a: f32) ...` → parse error. |
| Validate unknown parameterised rule | `crates/dsl_compiler/tests/param_rule_validate_unknown.rs` | `physics X = nonexistent(a: 1);` → `LowerError::UnknownParameterisedRule`. |
| Validate missing arg | same file | `chase(target: Wolf)` (aggro missing) → `ApplicationParamMismatch.missing`. |
| Validate extra arg | same file | `chase(... bogus: 0)` → `ApplicationParamMismatch.extra`. |
| Validate type mismatch | same file | `chase(target: Wolf, aggro: true, speed: 1.0)` → `ApplicationTypeMismatch`. |
| Validate unknown EntityKind | same file | `chase(target: NotAnEntity, ...)` → `UnknownEntityKind`. |
| Monomorphisation produces concrete rule | `crates/dsl_compiler/tests/param_rule_mono.rs` | One application → one concrete rule; param refs in body replaced with literal values. |
| Multiple applications → distinct concrete rules | same file | Two applications produce two independent concrete rules with different bodies. |
| Cross-file via import | `crates/dsl_compiler/tests/param_rule_cross_file.rs` | `import std/rules/chase.sim;` then `physics MyChase = chase(...)` in the importing file works end-to-end. |
| Collision: two parameterised rules same name | same file | Two `physics chase(...) { ... }` in different files → `DuplicateDefinition`. |
| Collision: parameterised rule vs concrete rule same name | same file | `physics chase(...) {...}` and `physics chase {...}` collision. |
| Smoke fixture | `assets/sim/param_rule_smoke.sim` + `crates/sims/tests/param_rule_smoke.rs` | A real fixture imports `stdlib/rules/chase.sim`, applies it twice with different args, the megacrate exposes both as distinct emitted modules. The runtime-gate test confirms both kernels exist and both appear in the schedule. |
| WGSL constant-folding sanity | manual / docs | Emitted WGSL for `HunterChase` contains literal `15.0` (not a variable reference). Local-dev visual check; not CI. |

**Stdlib seed content (separate commits, not part of v1 plan exit criteria):**

Once the spec ships, `stdlib/rules/` can grow `chase.sim`, `wander.sim`,
`flock.sim`, `regen.sim` — one parameterised rule per file. Populating
the stdlib is content work, not language work.

**No new parity test (P3).** Monomorphisation happens before backend
selection. Both backends see the same concrete-rule list. Existing
`parity_*.rs` tests cover behaviour.

**No schema-hash bump (P2).** No `SimState` SoA fields change.

## Constitution touchpoints (for plan AIS)

- **P1 (Compiler-First):** PASS — parameterised rules and applications
  flow through the parser + compiler; the monomorphisation pass is
  compiler work, not hand-written rule logic. No new `crates/engine/src/handlers/`
  code.
- **P2 (Schema-Hash):** N/A — no `SimState` SoA fields change.
- **P3 (Cross-Backend Parity):** PASS — monomorphisation happens
  before backend selection. Existing parity coverage suffices.
- **P4 (`EffectOp` Size):** N/A — no new event variants.
- **P5 (Determinism via Keyed PCG):** PASS — no RNG entered.
  Application order is AST source order; substitution is purely
  AST-level transformation.
- **P6 (Events Are the Mutation Channel):** N/A — no state mutation.
- **P7 (Replayability Flagged):** N/A — no new events.
- **P8 (AIS Required):** the implementation plan will carry the
  full AIS template; this design summarises the touchpoints.
- **P10 (No Runtime Panic):** PASS — parse / validate / lower errors
  are `Result`s, surfaced as build-time compile errors. No new
  runtime panic paths.
- **P11 (Reduction Determinism):** N/A — no reductions.

## Known risks

- **Span attribution complexity.** Substituting nodes from a template
  into a synthetic concrete rule requires careful source-span
  threading. Errors that point at "line 12 of fixture.sim" when the
  bug is actually in the template's body confuse users. The
  validation pass should normalise span reporting so errors point
  at *both* the template decl site and the application site.
- **Application-name collisions with concrete rules.** v1 enforces
  uniqueness across both kinds in the `physics` namespace. If
  stdlib ever ships a parameterised rule with a name a fixture
  already uses for a concrete rule, the fixture must rename. This is
  by design but worth noting for stdlib naming conventions.
- **Numeric literal coercion at apply sites.** The existing
  literal-coercion rules permit some surprising conversions (e.g.
  `15` (i32) → `f32`). Applications inherit these rules; users
  may write `aggro: 15` and not realise the `f32` coercion happened.
  Coercion errors must clearly say "i32 literal 15 coerced to f32"
  in the diagnostic.
- **EntityKind references across imports.** A parameterised rule in
  `stdlib/rules/chase.sim` doesn't know about the importing
  fixture's entity decls. The `target: EntityKind` parameter is
  resolved at apply time, against the *merged* Program's entity
  set. The validation pass must run after multi-file merge.
