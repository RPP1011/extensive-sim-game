# DSL-as-Engine Scorecard

> Consolidated assessment of how the World Sim DSL held up as a *general
> game engine*, derived from the `vampire_survivors` fixture exercise
> (2026-05-24). The fixture was built as a coverage probe: a real-time
> swarm survival game is deliberately outside the DSL's original tactical-
> RPG envelope, so every place the build snagged is a signal about the
> compiler/engine, not about the game.
>
> This doc is the single authoritative gap ledger. The per-fixture design
> ledgers (`2026-05-24-vampire-survivors-design.md` §8,
> `-viewer-design.md`) and the `project_vampire_survivors_fixture.md`
> memory feed into it; from here on, update **this** file as gaps close.

## Verdict

The DSL composed a working real-time game — homing swarm AI, two weapons,
arena bounds, wave spawning, a 3D viewer — almost entirely from primitives
that already existed (cross-agent indexed reads, `@phase(per_agent)` rules,
spatial queries as an *optimization*, normalized-direction movement,
config bands). That is a genuinely strong result: the rules-as-data surface
is more general than the tactical-RPG framing suggested.

But "proper engine" surfaced a **class of silent failure modes** the
tactical fixtures never hit. The meta-lesson dominates everything below:

> **Compile-gate ≠ execution.** Every real defect in this exercise compiled
> green and surfaced only at GPU runtime (or not at all — see G1). The
> schema-hash test, the build.rs lower/emit pass, and `cargo check` give
> *false confidence*. A clean compile says the kernel is well-formed WGSL,
> not that it does what the `.sim` says.

The corollary is a process rule, now encoded in the AIS template's
"Runtime gate" field: **every engine change needs a test that runs the
changed path and asserts an observable post-condition** (tick advance,
event count, byte equality), not just that it compiles.

## Gap ledger

Severity: **S1** silent-wrong (compiles green, runs wrong, no diagnostic) —
the worst, because nothing tells you. **S2** runtime-only failure (compiles
green, panics/misbehaves at GPU runtime). **S3** missing capability (had to
hand-roll outside the DSL). **S4** ergonomic footgun (works, but easy to get
wrong).

Status: **OPEN** / **CLOSING** / **CLOSED** / **PLANNED**.

| # | Gap | Sev | Status | Evidence / locus |
|---|-----|-----|--------|------------------|
| G1 | **Fold-handler `where`-guard silently dropped.** A `where` on a `@materialized` / belief fold handler is parsed and stored on the AST, but never carried into `FoldHandlerIR` and never emitted. The guard is 100% ignored. | S1 | CLOSING | **Verified 2026-05-24 with WGSL evidence** (below). |
| G2 | **`@phase(event)` + `self.*` → undefined `agent_id`.** A rule phased on an event that references `self` emits WGSL referencing an `agent_id` local the PerEvent preamble never binds → naga panic at first `step()`. | S2 | OPEN | Foundation rules (BoltFire/NovaFire) hit this; re-phased to `@phase(per_agent)` as a workaround. Locus: `cg/emit/kernel.rs` PerEvent preamble. Needs re-confirm + typed error. |
| G3 | **`top_k` spatial query under-fill → slot-0 sentinel.** An under-filled `spatial.closest_enemy(self)` yields slot 0 (the AgentId-absent sentinel) as a "result"; the loop body then treats slot 0 as a live agent. Player bolted *itself* (hp 100→94→−230) at tick 13. | S1/S2 | OPEN | Worked around with an in-rule re-check guard (`target.alive && band`). Locus: spatial top_k emit; pad entries are not marked invalid. |
| G4 | **No global / singleton agent read.** Spatial queries are the *only* cross-agent read primitive. "Every enemy homes on the player" needed a per-agent `engaged_with` AgentId pre-seeded to a fixed slot + `agents.pos(self.engaged_with)`. There is no `the_player.pos` / singleton-row read. | S3 | OPEN | Cross-indexed read works (target_chaser precedent) but is a workaround for a missing primitive. |
| G5 | **Summon → GPU slot allocation unbuilt.** `EffectOp::Summon` emits an `EffectSummonApplied` chronicle, but nothing allocates a GPU agent slot from it. `apply_summon_event_to_state` targets the legacy CPU `SimState`, not the compiled GPU runtime. Had to hand-roll `summon_alloc::drain_summons` (host-side ring decode + dead-slot claim + buffer writes). | S3 | OPEN | `crates/sims/src/summon_alloc.rs` is ~300 lines of hand-written host glue per the design. Generic engine support would emit this. |
| G6 | **AgentId is `NonZeroU32`; slot 0 is the absent sentinel.** Seeding the player at slot 0 makes it unreferenceable by `engaged_with`; an under-filled query returns 0 and reads garbage. Costs a wasted slot and is an easy off-by-one. | S4 | OPEN | Mitigated by convention (`PLAYER_SLOT=1`, `ENEMY_POOL_START=2`). Interacts with G3. |
| G7 | **Config is baked; no per-tick global broadcast.** `config` blocks bake to WGSL literals at build time; the `sim_cfg` binding is 16 zero bytes. No runtime override path for per-runtime parameterization. `@runtime` per-kernel fields exist but there is no global per-tick uniform. | S3 | OPEN | See `project_config_driven_sims_gap.md`. |
| G8 | **No CPU/Serial path for compiled fixtures.** `crates/sims` fixtures are GPU-only. P3 (cross-backend parity) is aspirational here — there is no SerialBackend reference to diff against, so G1/G3-class silent-wrong bugs have no second opinion. | S3 | OPEN | Structural; the interpreter (`dsl_ast::eval`) covers only wolves+humans primitives. |
| G9 | **Misc primitive gaps.** `floor` lowers on the compiled path but `eval_numeric_builtin` (interp) lacks an arm; `else if` chains hit a parser gap (`parser.rs:4742`); `vec3` `.x`/`.y`/`.z` field access unsupported. | S3/S4 | OPEN | Encountered incidentally; each is a small isolated add. |

## G1 evidence (the flagship)

The DSL surface (`dsl_stress_coverage.sim:79`, `crowd_navigation.sim:123-135`):

```
on Damaged { target: t, amount: a } where t == target && a > config.probe.thresh { self += 1.0 }
```

- AST `FoldHandler` **carries** `where_clause: Option<Expr>` (`ast.rs:747`).
- Parser `parse_fold_handler` **parses and stores** it (`parser.rs:2067-2090`).
- Resolver builds `FoldHandlerIR { pattern, body, span }` — **drops `where_clause`** at `resolve.rs:1771` (views) and `resolve.rs:2170` (belief propagation). `FoldHandlerIR` (`ir.rs:1112`) has **no field** to hold it. (Contrast: `PhysicsHandlerIR` *does* carry `where_clause`, resolved at `resolve.rs:1587-1595`; social-merge handlers carry it at `resolve.rs:2205-2234`.)
- No `dsl_compiler` code consumes a view-fold `where_clause` (`rg where_clause crates/dsl_compiler` → physics + monomorphization only).

Emitted `fold_big_hits.wgsl` for the clause above:

```wgsl
if (event_ring[_ei * 11u + 0u] == 0u) {            // event kind == Damaged
    let local_0: u32 = event_ring[_ei * 11u + 3u]; // t  (event.target)
    let local_1: f32 = bitcast<f32>(event_ring[_ei * 11u + 4u]); // a (event.amount) — BOUND BUT NEVER USED
    if (local_0 == observer_slot) {                // keying: from PATTERN field-name match, NOT the where
        accum = accum + (1.0);                      // unconditional — `a > thresh` is GONE
    }
}
```

The kernel is **byte-identical with and without the `where`-clause** (confirmed by compiling both). So:

- The **keying** (`if (local_0 == observer_slot)`) is derived purely from pattern field-name matching (`target` field ↔ `target` key param). The `t == target` term of the where is redundant.
- The **genuine guard** (`a > config.probe.thresh`) is silently discarded — `local_1` is bound and dead.

**Production impact:** `crowd_navigation.sim`'s `stuck_ticks` has two `on Tick {}` handlers distinguished *only* by `where w.last_progress < thresh` vs `>= thresh`. With the guard dropped, both fold unconditionally every tick (`self += 1` then `self = 0`) → the view is pinned near 0 and stuck-detection never fires. It compiles green; no test catches it. This is the meta-lesson made flesh.

Probe / TDD anchor: `crates/dsl_compiler/tests/fold_where_guard_emit.rs`.

## Close-out plan (the "close" half of the goal)

Priority order = (silent-wrong first) × (tractable first):

1. **G1 — honor the fold `where`-guard.** Carry `where_clause` into `FoldHandlerIR`; resolve it in the inner (binder + param) scope; lower it to a guard expression; wrap the handler body emit in `if (guard) { … }` at the op level (parallel to the existing keying guard at `wgsl_body.rs:2359`); add the interpreter arm (`dsl_ast/src/eval/view.rs`) for parity. Pure keying-equality terms (`binder == key_param`) are already handled by the emitter, so the guard wrapper is additive, not a replacement. Gate: `fold_where_guard_emit.rs` (emit) + a behavioral pin that a guarded fold filters.
2. **G2 — `@phase(event)` + `self.*` → typed compile error.** Lowest-risk silent→loud conversion: reject `self.*` in an event-phased rule at resolve/lower with a diagnostic pointing at `@phase(per_agent)`. Gate: a compile test asserting the typed error.
3. **G3 — `top_k` under-fill.** Mark padded query entries invalid (or emit a result count and bound the loop), so the loop body never sees slot 0. Gate: emit test + a behavioral pin (no self-target when no enemy in range).
4. **G4 — global/singleton agent read.** Larger; design a `the_<entity>` singleton-row read primitive. Spec separately before implementing.
5. **G5 — generic Summon→GPU allocation.** Larger; fold the `drain_summons` logic into an emitted engine pass keyed on `EffectSummonApplied`. Spec separately.
6. **G6/G7/G9** — smaller follow-ups, batched.
7. **G8** — structural; out of scope for a single pass (revisit if a Serial path lands).

## Architectural Impact Statement

- **Existing primitives searched:**
  - `FoldHandlerIR` at `crates/dsl_ast/src/ir.rs:1112`
  - `PhysicsHandlerIR.where_clause` (precedent) at `crates/dsl_ast/src/resolve.rs:1587`
  - fold-body keying guard emit at `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:2359`
  - PerEvent preamble at `crates/dsl_compiler/src/cg/emit/kernel.rs`
  Search method: `rg` + direct `Read` + WGSL emit probes.

- **Decision:** extend existing primitives (carry an already-parsed AST field through resolve→IR→lower→emit) rather than add new surface. G1/G2/G3 are bug-class closures, not features.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none (the `.sim` surface is unchanged; `where` on fold handlers already parses).
  - Code edited: `crates/dsl_ast/src/{ir,resolve,eval/view}.rs`, `crates/dsl_compiler/src/cg/{op,lower/view,emit/wgsl_body}.rs`.
  - Generated outputs re-emitted: per-runtime `fold_*.wgsl` regenerated by each `build.rs`; `crates/engine/.schema_hash` if the IR layout hash shifts (routine, automatic).

- **Hand-written downstream code:** NONE. All behavior stays in the emitter/interpreter; no hand-written rule logic in engine handler paths.

- **Constitution check:**
  - P1 (Compiler-First): PASS — fix lives in the compiler/interpreter; no `impl Rule` outside generated.
  - P2 (Schema-Hash): N/A→PASS — if `FoldHandlerIR` gaining a field shifts the layout hash, regen is automatic plumbing.
  - P3 (Cross-Backend Parity): PASS — G1 implements both the compiled emit and the interpreter arm; wolves+humans parity baseline re-run.
  - P4 (`EffectOp` budget): N/A.
  - P5 (Keyed PCG): N/A — no new RNG.
  - P6 (Events Are the Mutation Channel): PASS — folds still consume events; the guard only filters.
  - P7 (Replayability Flagged): N/A.
  - P8 (AIS Required): PASS — this section.
  - P9 (Verified Commit): PASS — each close lands with its gate test.
  - P10 (No Runtime Panic): PASS — G2 turns a runtime panic into a compile error (strictly improves P10).
  - P11 (Reduction Determinism): PASS — the guard wraps the existing keyed accumulate; it does not change reduction order.

- **Runtime gate:** the whole point of this doc. Each close lands with a test that *runs* the path:
  - G1: `fold_where_guard_emit.rs` (the guard appears in WGSL) + a behavioral pin that a guarded fold filters events.
  - G2: a compile test asserting the typed error (the path that previously panicked at runtime now fails at compile).
  - G3: an emit test + a behavioral pin (no self-target with no enemy in range).

- **Re-evaluation:** [x] AIS reviewed at design phase.  [ ] post-design.
