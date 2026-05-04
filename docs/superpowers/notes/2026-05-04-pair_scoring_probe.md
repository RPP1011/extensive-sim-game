# Pair-scoring probe — discovery report (2026-05-04)

This is the report from the smallest end-to-end probe of **PAIR-FIELD
SCORING** — the "scoring rule that picks both an action AND a target"
pattern foundational to RPG-style decision making. Per the spec audit
(early session):

> Pair-field scoring (`target.*` in scoring rows, spec §8.3) — scoring
> fixtures only use `self.*` fields.

Today every shipped scoring row reads `self.<field>` only; NO fixture
or test exercises `target.<field>` predicates in scoring expressions.

The probe attempts the canonical "every Medic picks the most-injured
OTHER Medic to heal" pattern via:

```
verb Heal(self, target: Agent) =
  action HealAction
  when (target != self)
  emit Healed { healer: self, target: target, amount: config.heal.amount }
  score (1000.0 - agents.cooldown_next_ready_tick(target))
```

The score uses `cooldown_next_ready_tick` as a proxy "injury indicator"
(reusing the SoA proven by `cooldown_probe`) so per-slot variation is
analytical: with `ready_at[N] = N * 10`, slot 0 has the lowest
cooldown → highest inverted score → every healer picks slot 0.

## Outcome (closure update — 2026-05-04 PM)

**OUTCOME (a) FULL FIRE — pair-field scoring ships end-to-end.**

```
pair_scoring_probe_app: received readback — min=0.000 mean=437.500 max=3500.000 sum=3500.000
pair_scoring_probe_app: nonzero slots: 1/8 (fraction = 12.5%)
pair_scoring_probe_app: expected (FULL FIRE): received[0]=3500, received[1..]=0  → expected sum = 3500
pair_scoring_probe_app: preview received[0..8] = [3500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pair_scoring_probe_app: OUTCOME = (a) FULL FIRE — pair-field scoring picked the lowest-cooldown target every tick.
```

The four originally-anticipated gaps (Gaps #1-#4) all closed in
sequence. Two STILL-OPEN gaps surfaced after them and closed
together in this commit:

- **Gap A (compiler)** — `cg::lower::expr::lower_namespace_call`'s
  `(NamespaceId::Agents, field) if args.len() == 1` arm wrapped the
  arg expression as `AgentRef::Target(<expr_id>)` unconditionally.
  When the arg lowered to `CgExpr::PerPairCandidateId`, the emit
  side rendered `agent_<field>[target_expr_<N>]` — relying on a
  stmt-prefix `let target_expr_<N>: u32 = <wgsl>;` hoist that the
  scoring kernel emit (`cg::emit::kernel::lower_scoring_argmax_body`)
  doesn't run. Result: invalid WGSL with an undefined identifier.
  Fix: structural fold — when `target_id` resolves to
  `CgExpr::PerPairCandidateId` / `CgExpr::AgentSelfId`, collapse to
  `AgentRef::PerPairCandidate` / `AgentRef::Self_` directly so the
  emit side renders `agent_<field>[per_pair_candidate]` /
  `agent_<field>[agent_id]` with no hoist needed.

- **Gap B (compiler)** — `cg::lower::scoring::lower_standard_row`
  set `ctx.target_local = true` for `Positional([("target", _,
  AgentId)])` heads but did NOT shadow the binder's `LocalRef` in
  `ctx.local_ids`. The verb expander's earlier chronicle-physics
  lowering (`synthesize_cascade_physics`) registers the verb's
  `target` LocalRef → some `LocalId` for the chronicle's
  `on ActionSelected { target: <verb_target_local> }` event-pattern
  binder. When scoring lowering then ran the score body, bare
  `target` hit `lower_bare_local`'s Step 1 (let-bound local) FIRST
  and resolved to `ReadLocal { local: <chronicle's local id> }` —
  bypassing Step 2-3's `target_local`-aware
  `PerPairCandidateId` arm. The downstream per-pair detector
  (`expr_references_per_pair_candidate`) then saw no candidate-side
  reference and the scoring kernel emitted with no inner candidate
  loop. Fix: temporarily `remove(&binder_local_ref)` from
  `ctx.local_ids` before lowering the row body, restore after.

The runtime (`crates/pair_scoring_probe_runtime/src/lib.rs`) was
also rewritten to dispatch the full per-tick chain (mask + scoring
+ chronicle + fold) instead of the reduced (b) NO-FIRE shape.

Original gap chain (preserved for context — all closed):

```
[BEFORE FIX]
pair_scoring_probe_app: received readback — min=0.000 mean=0.000 max=0.000 sum=0.000
pair_scoring_probe_app: nonzero slots: 0/8 (fraction = 0.0%)
pair_scoring_probe_app: preview received[0..8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pair_scoring_probe_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0
```

The fixture parses + resolves cleanly; lowering produces TWO typed
diagnostics that drop the mask + scoring kernels from the partial CG
program; the chronicle + view-fold still emit and dispatch but never
see an `ActionSelected` event because no scoring kernel produced one.
(All gaps closed by 2026-05-04 PM — see closure update above.)

## Gap chain

Four candidate gap layers were anticipated; the probe surfaces TWO at
compile time and confirms a third would block on the next layer down.
The fourth is hypothesised but cannot be exercised until the first
three close.

### Gap #1 (BLOCKING — HIGH severity)

**`mask#0 head shape \`positional\` requires a \`from\` clause —
parametric heads without an explicit dispatch source are not yet
routable (Task 2.6)`**

- File: `crates/dsl_compiler/src/cg/lower/mask.rs:130-138`
- Trigger: verb expansion synthesises a mask with head shape
  `IrActionHeadShape::Positional([("target", _, AgentId)])`
  (`crates/dsl_compiler/src/cg/lower/verb_expand.rs:557-582`,
  `build_mask_head`). The verb syntax has NO `from` clause to thread
  (`parse_verb_decl` at `crates/dsl_ast/src/parser.rs:1187-1219` —
  only `action`, `when`, `emit`, `score` clauses exist; mask `from`
  parsing lives in `mask_decl` at line 1076).
- Effect: the mask op is dropped from the partial CG program. Without
  the per-pair candidate enumeration the mask would set up, NO
  scoring kernel can know which (actor, candidate) pairs to score.

This is the structural root cause. Fixing requires either:

  (a) Verb syntax grows `from <candidate_source>` (e.g.,
      `verb Heal(self, target: Agent) from query.allies(self, 10.0) = ...`),
      mirroring mask declaration syntax, OR

  (b) The compiler synthesises a default candidate source for verbs
      with target-typed positional binders (e.g., "all alive Medics"),
      then routes through a new `PerPairSource` variant
      (`crates/dsl_compiler/src/cg/dispatch.rs`).

Either route lands inside Task 2.6 territory.

### Gap #2 (PARALLEL — MEDIUM severity)

**`lowering: binary \`Sub\` at <span> has mismatched operands — lhs is
f32, rhs is u32`**

- File: `crates/dsl_compiler/src/cg/lower/expr.rs` (binary-op type
  checker — exact line per `BinaryOperandTyMismatch` constructor)
- Trigger: the score expression is `1000.0 - agents.cooldown_next_
  ready_tick(target)`. The literal lowers as `f32`, the SoA read
  lowers as `u32`. The arithmetic doesn't auto-promote.
- Effect: even if Gap #1 were closed, the score expression would
  still fail to lower. The scoring entry would drop, falling back to
  the same NO-FIRE observable.

This is independent of Gap #1 — it surfaces in the same lower pass
because the compiler doesn't bail at the first error. Fix requires
either:

  (a) Caller-side cast (e.g., `f32(target.cooldown_next_ready_tick)` —
      requires the cast surface to lower), OR

  (b) Implicit u32→f32 widening for arithmetic operators (would need
      a typed coercion pass in `lower_binary_op`).

### Gap #3 (HIDDEN — surfaces when Gap #1 + Gap #2 close)

**Verb-injected scoring entry hardcodes `IrActionHeadShape::None`**

- File: `crates/dsl_compiler/src/cg/lower/verb_expand.rs:298-307`
- Code:

  ```rust
  if let Some(score_expr) = &verb.scoring {
      let entry = ScoringEntryIR {
          head: IrActionHead {
              name: synthetic_name.clone(),
              shape: IrActionHeadShape::None,   // <-- HARDCODED
              span: verb.span,
          },
          expr: score_expr.clone(),
          span: verb.span,
      };
      ...
  }
  ```

- Effect: even if Gaps #1 + #2 close, the score expression would
  lower in a context where `ctx.target_local = false` (because
  `lower_standard_row` only flips it when the head shape is
  `Positional` —
  `crates/dsl_compiler/src/cg/lower/scoring.rs:307-345`). Then
  `target.<field>` reads would fail with `UnsupportedLocalBinding`
  (the `Local("target")` arm in `lower_bare_local` at
  `crates/dsl_compiler/src/cg/lower/expr.rs:1010-1029` only resolves
  `target` when `target_local == true`).
- Fix: `verb_expand.rs::expand_one_verb` must mirror its mask-side
  `build_mask_head` logic when synthesising the scoring entry — drop
  `self`, thread the rest as `IrActionHeadShape::Positional(rest)`.

The infrastructure for this is already in place (the
`positional_target_binder` arm at scoring.rs:307-345 is fully wired
and tested — see `lower_scoring_positional_target_binder_resolves_to_per_pair_candidate`
test). The only missing piece is the verb-expand site.

### Gap #4 (HYPOTHESISED — surfaces when Gaps #1-3 close)

**ScoringArgmax dispatch shape is per-actor (1D), not per-(actor,
candidate) (2D)**

- Files: `crates/dsl_compiler/src/cg/op.rs` (`ScoringArgmax` op),
  `crates/dsl_compiler/src/cg/emit_scoring/` (kernel emit), and
  the kernel WGSL emitter (search for `cs_scoring`).
- Trigger: today the scoring kernel dispatches `agent_count` threads,
  each computing one row's utility per agent. A pair-field scoring
  row needs `agent_count × candidate_count` evaluations — every
  (self, target) pair — followed by an argmax across candidates per
  actor. The kernel structure would need either:
    - A 2D dispatch `(agent_count, candidate_count)` with per-actor
      reduction, OR
    - A 1D dispatch `(agent_count)` with an inner loop over candidate
      slots inside the kernel body (similar to the spatial-query
      mask kernels — see `physics_WanderAndTrade` in trade_market).
- Effect: if Gaps #1-3 close but #4 doesn't, the per-pair scoring row
  would compile but the kernel would compute the wrong thing — likely
  `score(self, NO_TARGET)` for every actor, with all candidates
  collapsing to a single `target = self` evaluation, producing one
  ActionSelected per actor with `target = NO_TARGET`. The probe's
  observable would shift from "all zeros" to "received[N] = TICKS *
  amount" for whatever NO_TARGET resolves to (typically slot 0 or a
  sentinel).
- Severity: BIGGEST of the four — touches both the CG-IR op shape
  (needs a `target: Some(...)` slot populated by lowering) AND the
  kernel emitter (needs per-pair iteration).

This gap is purely hypothesised — it cannot be exercised until Gap #1
opens the upstream path. The probe is an early-warning marker for the
shape of the eventual solution.

## Files added (5 / +566 LOC under budget)

```
assets/sim/pair_scoring_probe.sim                           110 LOC  (fixture)
crates/pair_scoring_probe_runtime/Cargo.toml                 25 LOC
crates/pair_scoring_probe_runtime/build.rs                  111 LOC  (build script)
crates/pair_scoring_probe_runtime/src/lib.rs                300 LOC  (runtime)
crates/sim_app/src/pair_scoring_probe_app.rs                132 LOC  (harness)
docs/superpowers/notes/2026-05-04-pair_scoring_probe.md     (this file)
```

Plus +119 LOC negative-compile-gate test in
`crates/dsl_compiler/tests/stress_fixtures_compile.rs`.

## Files touched

```
Cargo.toml                                          (+1 member: pair_scoring_probe_runtime)
crates/sim_app/Cargo.toml                           (+5 dep, +4 bin lines)
crates/dsl_compiler/tests/stress_fixtures_compile.rs (+~119 LOC negative test)
```

## Negative compile-gate test

`pair_scoring_probe_negative_compile_gate` in
`stress_fixtures_compile.rs` PINS the gap chain. Asserts:

1. Parse + resolve succeed (`comp.verbs.len() == 1`, params include
   `target` named slot).
2. `lower_compilation_to_cg` returns `Err` with TWO diagnostics:
   - Gap #1: contains `head shape \`positional\`` AND `requires a
     \`from\` clause`.
   - Gap #2: contains ``binary `Sub` `` AND `lhs is f32, rhs is u32`.
3. Partial-emit set INCLUDES `physics_verb_chronicle_Heal` +
   `fold_received` (the chronicle + fold survived because they don't
   reference target.<field> in any context).
4. Partial-emit set EXCLUDES any `mask*` kernel and the `scoring`
   kernel.

When a future commit closes Gap #1, the test FAILS — the panic
message guides the maintainer to flip the assertion to test the
positive FULL-FIRE shape.

## Why this probe is valuable even at outcome (b)

The pair-field scoring pattern is foundational to every RPG-shaped
decision: "heal the most injured ally", "attack the lowest-HP enemy",
"assist the closest squadmate", "buff the highest-DPS teammate". Every
one of those is a `verb V(self, target: T) = ... score f(self, target)`
shape. Without it, the engine's verb cascade is structurally
single-target-per-actor — every action's target is `NO_TARGET` (per
the existing `verb_fire_probe` and `abilities_probe` outputs).

Pinning the gap chain as a negative test means the next agent who
attempts spec §8.3 inherits the discovery — they don't have to
re-derive the four-layer chain from scratch. The fix is roughly:

  1. **Cheapest fix**: only Gap #3 (5 lines in `verb_expand.rs`).
     Insufficient on its own — Gap #1 still blocks.
  2. **Cheapest USEFUL fix**: Gap #1 via synthetic candidate-source
     ("all alive entities of T type") + Gap #3. Gives positive
     OUTCOME (a) for `target != self` predicates. Gap #4 then
     surfaces (kernel computes wrong utility).
  3. **Real fix**: Gap #1 + Gap #3 + Gap #4. Multi-week effort —
     touches AST surface (verb `from`), CG (PerPairSource variant),
     scheduler (kernel selection by dispatch shape), emitter (per-pair
     argmax kernel). Gap #2 (typed coercion) is orthogonal but
     surfaces here too.

## Test count delta

- `cargo test --workspace`: +1 test
  (`pair_scoring_probe_negative_compile_gate`)
- All existing tests: no regression
- 16 existing fixture apps: unchanged

## Acceptance gates

- [x] `cargo build -p pair_scoring_probe_runtime` — clean (with
  `cargo:warning=` lower diags as documented)
- [x] `cargo test --workspace` — no regression; +1 negative test
- [x] Compile-gate test in `stress_fixtures_compile.rs` — pins
  failure mode
- [x] Discovery doc lands at this path
- [x] All 16 existing fixture apps unchanged (no edits to other
  runtimes / fixtures)
