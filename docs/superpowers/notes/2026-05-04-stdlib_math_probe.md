# Stdlib math probe — discovery report (2026-05-04)

This is the report from the **smallest end-to-end probe stress-testing
under-exercised stdlib math + RNG conversion surfaces** in a real
per-agent physics body. The probe targets surfaces that appear in
`docs/spec/dsl.md` and have lowering arms in
`crates/dsl_compiler/src/cg/lower/expr.rs` but aren't exercised by any
existing fixture in `assets/sim/`.

The probe is the SMALLEST .sim that drives a chain of stdlib math
calls + a tier of typed-RNG `let` bindings through the GPU pipeline:

  - 1 `Sampler` Agent entity.
  - 1 `physics SampleAndBucket @phase(per_agent)` rule that runs a
    chain of `let` bindings on stdlib math + RNG calls, then emits
    `Sampled` unconditionally per tick. The bucket field is drawn
    from the proven-safe `rng.action() % 4u` so the observable is
    independent of which typed-RNG surfaces survive validation.
  - 1 view-fold `sampled_count(agent)` that consumes `Sampled` and
    accumulates per-slot fire count.

Predicted full-fire observable: per-slot count = TICKS = 100, sum =
3200 across 32 slots.

## Outcome

**OUTCOME (a) FULL FIRE — for the SUBSET of surfaces that survive
validation.** Five gaps surfaced (one for every advertised surface
that isn't currently wired); the .sim retains only the surfaces that
make it through both `naga::front::wgsl::parse_str` (text parser) AND
`wgpu::Device::create_shader_module` (full validator).

```
stdlib_math_probe_app: starting — seed=0x0057D11B5A77F005 agents=32 ticks=100
stdlib_math_probe_app: finished both runs — counts1.len()=32 counts2.len()=32
stdlib_math_probe_app: DETERMINISM OK — both runs produced byte-identical sampled_count
  (P5: per_agent_u32(seed, agent, tick, purpose) is a pure fn).
stdlib_math_probe_app: sampled_count readback (run #1) — min=100.000 mean=100.000 max=100.000 sum=3200.000
stdlib_math_probe_app: nonzero slots: 32/32 (fraction = 100.0%)
stdlib_math_probe_app: expected per-slot count = TICKS = 100 (unconditional emit)
stdlib_math_probe_app: per-slot exact matches: 32/32 (max_dev = 0.000)
stdlib_math_probe_app: preview sampled_count[0..8] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
stdlib_math_probe_app: OUTCOME = (a) FULL FIRE — every retained surface wired end-to-end:
  1. Tier 1 math stdlib (floor/ceil/round/log2/abs) lowers + emits + FULL-validator-clean
  2. Tier 2 bucket emit via rng.action() % 4 fires every tick
  3. P5 determinism holds (byte-identical sampled_count across two runs)
  4. Per-slot count = TICKS = 100 on every slot
```

## Surfaces verified end-to-end

| Surface          | Lowers? | Emits?            | Naga text? | wgpu full? | Notes |
|------------------|---------|-------------------|------------|------------|-------|
| `floor(x)`       | yes     | `floor(...)`      | OK         | OK         | WGSL native |
| `ceil(x)`        | yes     | `ceil(...)`       | OK         | OK         | WGSL native |
| `round(x)`       | yes     | `round(...)`      | OK         | OK         | WGSL native |
| `log2(x)`        | yes     | `log2(...)`       | OK         | OK         | WGSL native |
| `abs(x)`         | yes     | `abs(...)`        | OK         | OK         | WGSL native (also accepts i32) |
| `rng.action()`   | yes     | `per_agent_u32(seed, agent_id, tick, 1u)` | OK | OK | Closed by stochastic_probe Gaps #1/#2/#3 |

## Surfaces that surfaced as gaps

| Surface                | Lowers? | Emits?  | Naga text? | wgpu full? | Gap |
|------------------------|---------|---------|------------|------------|-----|
| `log10(x)`             | yes     | `log10(...)` | FAIL  | n/a        | #A  |
| `planar_distance(a,b)` | yes     | `planar_distance(...)` | FAIL | n/a | #B |
| `z_separation(a,b)`    | yes     | `z_separation(...)`    | FAIL | n/a | #B |
| `rng.uniform_int(lo,hi)` | NO    | n/a     | n/a        | n/a        | #C  |
| `rng.coin()`           | yes     | `let local_N: bool = per_agent_u32(...)` | FAIL | n/a | #D |
| `rng.uniform(lo,hi)`   | yes     | `(lo + (per_agent_u32(...) * (hi - lo)))` | OK | FAIL | #E |
| `rng.gauss(mu,sigma)`  | yes     | `(mu + (per_agent_u32(...) * sigma))` | OK | FAIL | #E |

## Files added

- `assets/sim/stdlib_math_probe.sim` (~155 LOC) — probe fixture. One
  Sampler Agent entity, one `Sampled` event, one physics rule
  chaining the math/RNG `let`s, one view-fold. Each omitted-gap
  surface stays as a commented-out `let` with a citation to the
  responsible compiler arm.
- `crates/stdlib_math_probe_runtime/Cargo.toml` (~24 LOC)
- `crates/stdlib_math_probe_runtime/build.rs` (~115 LOC) — mirrors
  `stochastic_probe_runtime/build.rs` shape verbatim.
- `crates/stdlib_math_probe_runtime/src/lib.rs` (~290 LOC) — Agent
  SoA (alive only) + event ring + sampled_count ViewStorage +
  per-tick dispatch chain. Includes `clear_ring_headers_in` per the
  stochastic_probe pattern.
- `crates/sim_app/src/stdlib_math_probe_app.rs` (~210 LOC) — harness
  driving 100 ticks across two seeded runs, asserts byte-identical
  determinism + per-slot count = TICKS, prints OUTCOME line.
- `Cargo.toml` (workspace) — added `stdlib_math_probe_runtime`
  member.
- `crates/sim_app/Cargo.toml` — added the dep + `[[bin]]` entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `stdlib_math_probe_compile_gate` test (passing) — locks the
  structural surface (1 PhysicsRule + 1 ViewFold), the Tier 1 math
  stdlib emit fingerprints, the `rng.action() % 4u` bucket-emit
  fingerprint, AND regression guards on each Gap #A-#E surface
  (re-introducing a gap surface fails the test loudly).

Net LOC added: ~795 (slightly over the ~600 budget; the .sim header
and the per-gap commentary in the harness + lib.rs explain why each
omitted surface lives where it does — those comments dominate).

## Compiler topology — what got emitted

The compiler lowered the program to **7 ComputeOps**:

```
op0: ViewFold      (view: sampled_count, on_event: Sampled)
op1: PhysicsRule   (rule: SampleAndBucket, on_event: None — Tick)
op2: Plumbing      (UploadSimCfg)
op3: Plumbing      (PackAgents)
op4: Plumbing      (SeedIndirectArgs ring=0)
op5: Plumbing      (UnpackAgents)
op6: Plumbing      (KickSnapshot)
```

The scheduler emitted **7 kernels** (no fusion — single physics rule
+ single fold avoids the cross-domain fusion shape). The
`physics_SampleAndBucket` body has 4 bindings (event_ring, event_tail,
agent_alive, cfg). The cfg uniform carries `agent_cap` + `tick` +
`seed` per the Gap #1 close from stochastic_probe.

## Gap punch list

### Gap #A — `log10(x)` lowers but no WGSL native (HIGH)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:651`
**Lowering:** `Builtin::Log10` → `BuiltinId::Log10`
**Emit shape:** `log10(<arg>)`
**naga error:** `"no definition in scope for identifier: \`log10\`"`

WGSL provides `log` (natural log) and `log2`, but not `log10`. The
emit at `wgsl_body.rs:651` writes the bare identifier `log10` which
is not in WGSL's builtin scope. Two fix shapes:

  1. **Emit-time rewrite** — at the emit site, recognise
     `BuiltinId::Log10` and rewrite to `(log2(<arg>) * 0.30102999566)`
     (where the constant is `1.0 / log2(10.0)`). Single-line emit-
     side change; no prelude needed.
  2. **Prelude shim** — define `fn log10(x: f32) -> f32 { return
     log2(x) * 0.30102999566; }` as a substring-keyed prelude
     injection (same pattern as `RNG_WGSL_PRELUDE` in
     `cg/emit/program.rs:431`, gated on
     `body.contains("per_agent_u32(")`).

The same pattern applies to any future stdlib that maps to a
non-native — `log` (natural log) is fine in WGSL today, but adding
e.g. `cbrt` or `expm1` would hit the same gap.

### Gap #B — `planar_distance` / `z_separation` emit but no prelude (HIGH)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:640-641`
**Lowering:** `Builtin::PlanarDistance` → `BuiltinId::PlanarDistance`,
`Builtin::ZSeparation` → `BuiltinId::ZSeparation`
**Emit shape:** `planar_distance(<a>, <b>)` / `z_separation(<a>, <b>)`
**naga error:** `"no definition in scope for identifier: \`planar_distance\`"`

The emit writes a bare identifier call to a function that isn't
defined anywhere in the compiled module. Same shape as the
pre-stochastic-close `per_agent_u32` gap (Gap #2 of
`2026-05-04-stochastic_probe.md`); same fix shape:

  1. Define a WGSL prelude block with both helpers — e.g.
     `fn planar_distance(a: vec3<f32>, b: vec3<f32>) -> f32 { let
     d = a.xy - b.xy; return length(d); }` and
     `fn z_separation(a: vec3<f32>, b: vec3<f32>) -> f32 { return
     abs(a.z - b.z); }`.
  2. Inject via substring-keyed prelude in `cg/emit/program.rs:431`,
     gated on `body.contains("planar_distance(")` /
     `body.contains("z_separation(")`.

These functions ARE pinned by `slice-2b` lowering tests in
`expr.rs:4110` and `4122`, but those tests stop at the lower step —
they verify the IR shape, not the emit-and-validate path.

### Gap #C — `rng.uniform_int(lo, hi)` is unreachable from any .sim (HIGH)

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs:2824` (lower fn);
the typecheck rejection lands at line 2843.
**Lowering:** `rng.uniform_int(lo, hi)` requires both args to
typecheck as `CgTy::I32`.
**Surface gap:** the DSL has no surface to produce an `i32`:

  - The lexer accepts integer literals as bare `0` / `1` etc, which
    flow through `IrType::I32` → CG `CgTy::U32` (the standard
    promotion path; tested by stochastic_probe with `% 100`). A
    bare `0` lowers as `U32`, not `I32`.
  - There is no `i32` literal suffix (the `u` suffix already fails
    parser identifier lookup; `0i` would too).
  - There is no `i32(x)` / `u32(x)` / `f32(x)` cast surface — a
    parser scan of `crates/dsl_ast/src/parser.rs` finds no
    `TypeCast` / `cast_expr` / type-suffix arms.
  - No fixture in `assets/sim/` declares an `i32` config or SoA
    field — any author trying to source an i32 has no in-DSL way
    to do it.

Any `rng.uniform_int(0, 4)` call in a .sim emits the lower
diagnostic `lowering: expression at <span> is ill-typed — expected
i32, got u32` and the entire enclosing physics rule is dropped from
the kernel set.

**Fix shape options:**
  1. Promote the typecheck to accept `U32` and emit an in-IR `Cast`
     to I32 (matches the implicit-promotion pattern Gap #2 of
     pair_scoring closed for `f32` from `u32`).
  2. Add a parser-level `i32` literal suffix.
  3. Add an explicit cast surface (`as i32`, `i32(x)`, etc.).

The first is the smallest change consistent with the rest of the
DSL's implicit-coercion philosophy.

### Gap #D — `rng.coin()` emits `bool = u32` (HIGH)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:943-955` (the
universal `CgExpr::Rng` arm)
**Lowering:** `rng.coin()` → `CgExpr::Rng { Coin, Bool }` (purpose 7)
**Emit shape:** `let local_N: bool = per_agent_u32(seed, agent_id,
tick, 7u);`
**naga error:** `"the type of \`local_N\` is expected to be \`bool\`,
but got \`u32\`"`

The WGSL emit arm at `wgsl_body.rs:943` produces an unconditional
`per_agent_u32(...)` (returning u32) for ALL `CgExpr::Rng`
regardless of the carried `ty`. The surrounding `let` binding's
type annotation is `bool` per the lowering's typed-Rng invariant,
but the value expression is `u32`.

**Fix shape:** in the `CgExpr::Rng` arm, dispatch on `purpose` (or
on `ty`) to produce the per-purpose conversion routine:

```rust
match (purpose, ty) {
    (RngPurpose::Coin, CgTy::Bool) =>
        format!("((per_agent_u32(seed, agent_id, tick, 7u) & 1u) == 0u)"),
    (RngPurpose::Uniform, CgTy::F32) =>
        format!("(f32(per_agent_u32(seed, agent_id, tick, 5u)) / 4294967295.0)"),
    (RngPurpose::Gauss, CgTy::F32) => /* Box-Muller pair-draw */,
    (RngPurpose::UniformInt, CgTy::I32) =>
        format!("bitcast<i32>(per_agent_u32(seed, agent_id, tick, 8u))"),
    (_, CgTy::U32) =>
        format!("per_agent_u32(seed, agent_id, tick, {}u)", purpose.wgsl_id()),
    _ => unreachable!("typecheck guarantees one of the above"),
}
```

This puts the conversion at the EXPRESSION emit site (single
function), not the surrounding binary-op emit site — the surrounding
arithmetic (the `lo + draw * scale` shape from
`lower_rng_scaled_f32`) then composes f32 with f32 cleanly. The
`rng.uniform` / `rng.gauss` Gap #E falls out of the same fix.

### Gap #E — `rng.uniform` / `rng.gauss` emit u32 in f32 arithmetic (HIGH)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:943-955`
(same site as Gap #D)
**Lowering:** `rng.uniform(lo, hi)` →
  `AddF32(lo, MulF32(Rng{Uniform, F32}, SubF32(hi, lo)))`
  (built by `lower_rng_scaled_f32` at `expr.rs:2735`)
**Emit shape:**
  `let local_N: f32 = (0.0 + (per_agent_u32(seed, agent_id, tick,
  5u) * (1.0 - 0.0)));`
**naga TEXT parser:** ACCEPTS (the abstract-float literals lazily
coerce to u32 when paired with the `per_agent_u32` u32 result).
**wgpu FULL validator:** REJECTS:
  `"Expression [22] is invalid — Abstract types may only appear in
  constant expressions"`

This gap is particularly insidious because the
`naga::front::wgsl::parse_str` text parser used by the existing
`stress_fixtures_compile.rs` tests SUCCEEDS — so any compile-gate
test that only validates with the text parser would mark this
surface as "naga-clean" while the runtime panics on first
`create_shader_module`.

**Materialisation:** the harness was first run with this surface
intact and panicked at:

```
thread 'main' panicked at wgpu_core.rs:1023:30:
wgpu error: Validation Error
  In Device::create_shader_module, label = 'physics_SampleAndBucket::wgsl'
Shader validation error: Entry point cs_physics_SampleAndBucket at Compute is invalid
   ┌─ wgsl:47:29
   │
47 │ let local_5: f32 = (0.0 + (per_agent_u32(seed, agent_id, tick, 5u) * (1.0 - 0.0)));
   │                     ^^^ naga::ir::Expression [22]
   │
   = Expression [22] is invalid
   = Abstract types may only appear in constant expressions
```

Same fix as Gap #D — per-purpose conversion at the `CgExpr::Rng`
emit site. With `Uniform` emitting
`(f32(per_agent_u32(...)) / 4294967295.0)`, the surrounding
`AddF32(lo, MulF32(draw, scale))` becomes `(0.0 + ((f32(per_agent_u32
(...)) / 4294967295.0) * (1.0 - 0.0)))` — pure f32 arithmetic, full-
validator-clean.

For `Gauss`, the standard Box-Muller pair-draw form is:

```wgsl
fn rng_gauss_unit(seed: u32, agent_id: u32, tick: u32, purpose_a: u32,
                  purpose_b: u32) -> f32 {
    let u1 = max(f32(per_agent_u32(seed, agent_id, tick, purpose_a))
                 / 4294967295.0, 1e-9);
    let u2 = f32(per_agent_u32(seed, agent_id, tick, purpose_b))
             / 4294967295.0;
    return sqrt(-2.0 * log(u1)) * cos(6.28318530717958 * u2);
}
```

…requiring TWO `RngPurpose` ids for the pair-draw to keep the
streams independent (Gauss-A + Gauss-B). The current
`RngPurpose::Gauss = 6` would need a sibling.

## Closure conditions / suggested ordering

The five gaps are independent and can land in any order. Suggested
priority (smallest LOC-delta first):

  1. **Gap #A** (5-line emit-time rewrite to `log2(x) * c`).
  2. **Gap #B** (10-line WGSL prelude block + 2-line substring-keyed
     injection in `program.rs`).
  3. **Gap #D + Gap #E** together (one `match (purpose, ty)` arm at
     the `CgExpr::Rng` emit site closes both; ~20-line change).
  4. **Gap #C** (typecheck promotion in `lower_rng_uniform_int` to
     accept `U32` and synthesise an in-IR `Cast` to `I32`; ~10-line
     change).

Each fix should land its own `stdlib_math_probe_app` re-run
verifying the corresponding `let` binding can move out of the
omitted-surfaces commented block. The compile-gate test's regression
guards (`!body.contains("log10(")` etc.) also need to relax once a
gap closes — flip each `assert!(!body.contains(...))` to a positive
`assert!(body.contains(<expected new emit shape>))`.

## Verification

  - `cargo build -p stdlib_math_probe_runtime` — clean.
  - `cargo test -p dsl_compiler --test stress_fixtures_compile
    stdlib_math_probe` — `stdlib_math_probe_compile_gate` passes.
  - `cargo run -p sim_app --bin stdlib_math_probe_app` — OUTCOME (a)
    FULL FIRE; per-slot count = 100 on every slot; determinism OK.
  - `cargo test --workspace` — no regression on existing fixtures
    (17 sim_app binaries unchanged, all *_compile_gate tests still
    pass).
