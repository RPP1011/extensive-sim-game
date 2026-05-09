# `dsl.md` Drift Audit (2026-05-08)

> Audit of `docs/spec/dsl.md` against `crates/dsl_ast/src/` (parser/AST/resolver) and
> `crates/dsl_compiler/src/` (lowering, code-gen). Read-only observations; the only
> spec edit landing alongside this note is the §1.2 declaration list correction +
> the top-of-file callout pointer to this audit.
>
> **Scope clarification.** `dsl.md` documents the `.sim` DSL (entity / event /
> view / physics / mask / `verb`-decl / scoring / probe / metric). It does **not**
> own the `.ability` DSL effect-verb vocabulary — `damage` / `heal` / `stealth` /
> `disguise` / `plant_belief` / `observe` / `scry` / `reveal` / `decoy` /
> `erase_belief` / `travel_to` / `cast_recipe` / `wear_tool` etc. are `.ability`
> EffectOps (lowered by `crates/dsl_compiler/src/ability_lower.rs`) and live in
> `docs/spec/ability.md` + `docs/spec/ability_dsl_unified.md`. Any "missing
> effect verbs in §15" gap belongs in the ability-spec audit, not here. dsl.md
> has no §15 today; the file ends at §13 + Appendix A.
>
> The `EffectOp` variants `Propose` (42) and `Announce` (43) are present in
> `crates/engine/src/ability/program.rs` (Lift C engine surface) but the
> `.ability` parser does **not** lower them yet — no `"propose"` / `"announce"`
> arm in `ability_lower.rs::lower_effect_stmt`. That gap is also `ability.md`'s
> contract, not `dsl.md`'s.

---

## Summary

| Category | Count |
|---|---|
| `[CRITICAL]` (silently miscompiles) | **0** |
| `[MISSING]` (spec describes feature with no impl) | **0** |
| `[UNDOCUMENTED]` (impl has feature spec doesn't describe) | **8** |
| `[STALE]` (spec describes deprecated/removed surface) | **1** |

No critical drifts. The bulk of drift is `[UNDOCUMENTED]` — recent
ToM / multi-tick / lexer surfaces landed without spec backfill.

---

## `[UNDOCUMENTED]` items

### U1. Missing top-level declaration kinds — `event_tag`, `enum`, `query`, `config`

- **§ in spec**: §1.2 "Declaration kinds" — enumerates 11 kinds.
- **What spec says**: lists `entity`, `event`, `view`, `physics`, `mask`,
  `verb`, `scoring`, `invariant`, `probe`, `metric`, `spatial_query`.
- **What code does**: `crates/dsl_ast/src/parser.rs::decl` (line 183) dispatches on
  **15** keywords — adds `event_tag`, `enum`, `query`, `config` to the spec's 11.
  `Decl` enum (`crates/dsl_ast/src/ast.rs:45`) has 15 variants matching that
  parser table.
- **Surface evidence**: `assets/sim/auction_market.sim` and 4 other shipped
  fixtures use top-level `config <name> { ... }` blocks. `.sim` `query` decls
  parse as `Decl::Query` (the `view` section §2.3 mentions them inline but
  there's no top-level §2.x entry). `event_tag` and top-level `enum` parse
  but have zero shipped uses (`grep -rn "^event_tag\|^enum " assets/sim/` →
  empty).
- **Fix landing in this PR**: §1.2 is corrected to enumerate all 15 kinds and
  flags `event_tag` / `enum` / `query` / `config` as undocumented at §2 detail
  level. Per-kind §2.x sections are NOT added in this slice (additive-only).

### U2. `apply_ability <expr> [by <c>] [target <t>]` statement

- **§ in spec**: §6 "Runtime semantics" / §9.5 "Lowering passes" — neither
  documents this statement form.
- **What spec says**: nothing. The cascade-rule example body in §2.4 uses only
  `emit <Event> { ... }` form.
- **What code does**: `parser.rs::parse_apply_ability_stmt` (line 1999) parses
  `apply_ability <ability_expr> [by <caster_expr>] [target <target_expr>]`
  inside any `Stmt`-position context. `Stmt::ApplyAbility` (`ast.rs:707`) is the
  AST variant. The WGSL emitter expands it into a per-effect-slot dispatch loop
  reading from the `PackedAbilityRegistry` SoA (#125 / #132 / slice δ #161 /
  slice ε). Used by `assets/sim/duel_25v25.sim`, `boss_fight.sim`,
  `apply_ability_chronicle_consumer.sim`, `apply_ability_verb_smoke.sim`,
  `apply_ability_verb_chronicle_consumer.sim`.
- **Recommended action**: add a `### 2.4.1 apply_ability statement` subsection
  under physics, or a new statement-grammar §2.13. Deferred to a focused
  documentation slice — not in scope for this audit-only PR.

### U3. `beliefs(observer).observe(target) with { ... }` statement (ToM mutation)

- **§ in spec**: not present.
- **What spec says**: §7.2 mentions a roadmap-stub `theory_of_mind` namespace
  with `believes_knows` / `can_deceive` / `is_surprised_by` view methods, but
  no statement form for mutating belief cells.
- **What code does**: `parser.rs::parse_belief_observe_stmt` (line 2099) parses
  `beliefs(<ident>).observe(<ident>) with { field: expr, ... }`.
  `Stmt::BeliefObserve` (`ast.rs:686`) is the AST variant. Mutates
  `SimState::cold_beliefs` for the (observer, target) pair (Plan ToM Task 4).
  Used by `assets/sim/tom_probe.sim`.
- **Recommended action**: document under a new ToM subsection. Deferred.

### U4. `beliefs(observer).about(target).<field>` / `.confidence(target)` / `.<view>(_)` expressions

- **§ in spec**: not present.
- **What spec says**: nothing. `theory_of_mind` namespace at §7.2 is the closest,
  but the `beliefs(...)` expression surface has different shape (callee-receiver,
  not `theory_of_mind.<method>(...)`).
- **What code does**: `parser.rs::parse_belief_expr` (line 2204) parses three
  read-form tails:
  - `.about(target).<field>` → `ExprKind::BeliefsAccessor`
  - `.confidence(target)` → `ExprKind::BeliefsConfidence`
  - `.<view_name>(_)` → `ExprKind::BeliefsView`
- **Recommended action**: document under the same ToM subsection as U3.
  Deferred.

### U5. Statement language not documented as a grammar

- **§ in spec**: §2.3 fold body operator set + §2.4 `for` example are the only
  statement-grammar mentions.
- **What spec says**: nothing systematic. Examples use `let x = ...`, `for x in
  iter where pred { ... }`, `if cond { ... } else { ... }`, `match scrut { pat
  => body, ... }`, `emit Event { ... }`, `self += expr`, but the grammar isn't
  enumerated.
- **What code does**: `parser.rs::parse_stmt` (line 1856) supports 9 statement
  shapes: `Let`, `Emit`, `ApplyAbility`, `For` (with optional `where`), `If`
  (with optional `else`), `Match`, `BeliefObserve`, `SelfUpdate` (six ops:
  `+=` / `-=` / `*=` / `/=` / `|=` / `=`), and bare `Expr`.
- **Recommended action**: a single statement-grammar §2.13 would cover U2, U3,
  U5 in one edit. Deferred.

### U6. `query` top-level decl — listed neither in §1.2 nor as its own §2.x

- **§ in spec**: §1.2 omits `query`. §2.3 (`view`) shows a `@spatial query
  nearby_agents(...)` example but the spec explicitly notes (§2.3 audit
  callout) that `Decl::Query(QueryDecl)` is silently dropped after parsing.
- **What code does**: `parser.rs::query_decl` (line 779) parses top-level
  `query <name>(...) { ... }` as `Decl::Query`. The 2026-04-26 audit callout
  at §2.3 already pins this — but it's framed as a `view`-related note rather
  than an entry in the §1.2 enumeration.
- **Fix landing in this PR**: §1.2 now lists `query` alongside the other
  parsed-but-silently-dropped decl kinds with a forward-pointer to the §2.3
  callout.

### U7. Hex literals (`0x...`) and integer suffixes (`u`/`i`/`u8..u64`/`i8..i64`)

- **§ in spec**: no lexical-grammar section. (Compare with `ability.md` §2,
  which does have one.)
- **What spec says**: nothing about literal forms — readers must infer from
  examples.
- **What code does**: `crates/dsl_ast/src/tokens.rs::consume_int_suffix` (line
  91) consumes a Rust-style suffix; the lexer also accepts `0x` / `0X`-prefixed
  hex literals with `_` digit separators (commit `fa8e5f0c`, 2026-05-08). Both
  ergonomic features motivated by the recently-added `decoy <subject_idx>
  <fake_pos>` `.ability` verb whose packed `(i8,i8,i8,u8)` coordinate is more
  legible in hex.
- **Recommended action**: a small lexical-grammar §1.3 (mirroring `ability.md`
  §2) would cover this in a focused follow-up. Deferred.

### U8. `event_tag` decl — parsed but never used in shipped sims

- **§ in spec**: not present.
- **What code does**: `parser.rs::event_tag_decl` (line 582) parses
  `event_tag <Name> { <field>: <type>, ... }` as a compile-time field-shape
  contract that subsequent `event` decls can claim membership in (`ast.rs:202`).
  Resolver consumes it (`resolve.rs:786`). Zero `event_tag` declarations exist
  in `assets/sim/*.sim` today — the surface is reserved for future use.
- **Recommended action**: document in a §2.2 subsection or note as
  parser-only-no-uses. Deferred — non-blocking.

---

## `[STALE]` items

### S1. `assets/hero_templates/` reference in §1.2 area is fine; but `hero_templates`-style legacy references elsewhere should be checked

- **§ in spec**: `dsl.md` itself does NOT reference `assets/hero_templates/` —
  good. The retired hero-template layer (per `CLAUDE.md`) doesn't bleed into
  `dsl.md`.
- **Note for completeness**: `docs/spec/ability.md §1.3` still says
  *"Hero TOML files gain `abilities_file = "<name>.ability"`"* and *"`.ability`
  files in `dataset/hero_templates/` and `assets/hero_templates/` are loaded
  by the TacticalSim parser"* — those are stale per `CLAUDE.md` ("the
  `assets/hero_templates/` hero-template layer was retired"). Out of scope for
  this audit (ability.md is a separate audit task).

### S2. §2.6 `verb` callout numbering refers to "Slices A + cascade-followup"

- **§ in spec**: §2.6 closing callout (lines 297–324) refers to "Slices A +
  cascade-followup" landing 2026-05-03. The cascade-followup language IS
  current — verified `crates/dsl_compiler/src/cg/lower/verb_expand.rs` exists
  and contains the documented `synthesize_cascade_physics` doc-comment.
- **What's stale**: nothing functionally — the callout reflects current state.
  Skip.

(Net: 1 stale item — S1's pointer to a different spec file. Nothing inside
`dsl.md` is currently stale at the language-surface level.)

---

## Items that LOOK like drift but aren't

These were checked and confirmed correctly documented or out of scope:

- **`disguise` / `plant_belief` / `observe` / `scry` / `reveal` / `decoy` /
  `erase_belief` / `travel_to` / `cast_recipe` / `wear_tool` / `propose` /
  `announce`** — all are `.ability` DSL effect verbs (variants 32–43 in
  `EffectOp`). Owned by `ability.md`. The two-stack reality at the top of
  `ability.md` already pre-warns readers; full coverage of these 12 verbs is
  the next task on the audit DAG.
- **`is_duration_bearing_verb`** (`ability_lower.rs:2030`) — set of `.ability`
  verbs that consume `for <duration>` modifier. Currently:
  `stun, slow, root, silence, fear, taunt, lifesteal, damage_modify, buff,
  stealth, charm, grounded, suppress, reflect, disguise, travel_to`. This
  belongs in `ability.md`, not `dsl.md`.
- **`@symmetric_pair_topk(K)` / `@per_entity_ring(K)`** view annotations —
  documented at §2.3 (lines 187–192). Verified against
  `dsl_ast/src/resolve.rs::lower_view_kind`.
- **§2.3 / §2.4 / §2.6 / §2.8 / §2.9 / §2.11 audit callouts** — pre-existing
  2026-04-26 callouts; status unchanged this slice.
- **`agents.stun_remaining_ticks` naming drift** — already pinned at §7.2's
  `agents` namespace audit callout (2026-04-26). No change.

---

## Recommended follow-up tasks

(Out of scope for this docs-only audit slice; tracked here so the next
documentation pass can pick them up.)

1. **Statement-grammar §2.13** — covers U2 (`apply_ability`), U3
   (`beliefs(...).observe`), U5 (full statement enumeration).
2. **Lexical-grammar §1.3** — covers U7 (hex + suffixes); align with
   `ability.md` §2's table format.
3. **`query` decl §2.13** — covers U6; could fold into §2.3 instead.
4. **`event_tag` decl** — covers U8; one paragraph under §2.2 suffices.
5. **ToM expression surface** — covers U4 (`beliefs(...).about` / `.confidence`
   / `.<view>`); fits under the same statement-grammar §2.13 or a new §2.14
   "ToM access expressions" paired with the statement form.
