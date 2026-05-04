# Stochastic probe — discovery report (2026-05-04)

This is the report from the **smallest end-to-end probe of the
`rng.*` namespace** in a real per-agent physics body. No prior
fixture in the workspace exercises `rng.*` from a `.sim` file; this
probe is the first.

The probe is the SMALLEST .sim that drives the per-agent
deterministic RNG primitive through the GPU pipeline:

  - 1 `Trigger` Agent entity.
  - 1 `physics MaybeFire @phase(per_agent)` rule that draws
    `rng.action()` and gates an `emit Activated` on
    `(draw % 100) < 30` (a 30% per-tick activation gate).
  - 1 view-fold `activations(agent)` that consumes `Activated` and
    accumulates per-slot fire counts.

The runtime initialises every agent's `alive` slot to 1 and runs
1000 ticks. The analytical observable is per-slot count converging
to T × p = 300 ± ~5%.

## Outcome

**OUTCOME (b) WGSL VALIDATION FAILED.** The probe surfaces three
high-severity gaps in the WGSL emit pipeline that together prevent
ANY `rng.*` call from running on GPU today. The DSL → CG lowering
arm fires correctly (rng.action() → CgExpr::Rng{Action}); the WGSL
emit at `wgsl_body.rs:937-947` writes a literal call shape that
references three undeclared symbols.

```
stochastic_probe_app: starting — seed=0x570CA571CDEC0DE5 agents=32 ticks=1000 prob=0.30

thread 'main' panicked at wgpu_core.rs: wgpu error: Validation Error
  In Device::create_shader_module, label = 'physics_MaybeFire::wgsl'
Shader parsing error: expected expression, found "\""
   ┌─ wgsl:22:64
   │
22 │     let local_0: u32 = per_agent_u32(seed, agent_id, tick, "action");
   │                                                            ^ expected expression

stochastic_probe_app: OUTCOME = (b) WGSL VALIDATION FAILED
```

The harness wraps the GPU path in `catch_unwind` so the panic
becomes a clean OUTCOME (b) report instead of a raw process abort.

## Gap punch list

The first naga error stops at the string-literal `"action"` (Gap
#3); the kernel has TWO MORE undeclared symbols (Gaps #1 + #2)
which would surface in sequence after each prior gap closes. The
gap punch list is ordered by emit-order in the broken kernel body:

### Gap #1 — `seed` not bound by kernel preamble (HIGH)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:944`
**Body line:** `per_agent_u32(seed, agent_id, tick, "action")`

The kernel preamble at
`crates/dsl_compiler/src/cg/emit/kernel.rs:3385`
(`thread_indexing_preamble` for `DispatchShape::PerAgent`) binds:

```wgsl
let agent_id = gid.x;
if (agent_id >= cfg.agent_cap) { return; }
let tick = cfg.tick;
```

`seed` is referenced but never bound. There is no `cfg.seed` field
on any `PhysicsRuleCfg` (the cfg structs at
`generated.rs:125` are `agent_cap` + `tick` + padding only). There
is no `seed` global const in the emitted shader module either. P5
requires the seed to be a per-tick host-supplied parameter that is
constant across the dispatch — i.e. a uniform-bound value the
runtime can update each tick.

**One-line characterisation:** the WGSL emitter assumes `seed: u32`
is in scope; it isn't, and nothing in the cfg-uniform synthesis
path adds it.

### Gap #2 — `fn per_agent_u32` not emitted (HIGH)

**File:** none — there is no WGSL prelude / shim for the RNG primitive.

The host Rust function lives at `crates/engine/src/rng.rs:50`:

```rust
pub fn per_agent_u32(world_seed: u64, agent_id: AgentId, tick: u64, purpose: &[u8]) -> u32 {
    per_agent_u64(world_seed, agent_id, tick, purpose) as u32
}
```

The body uses `ahash::RandomState::with_seeds` — that algorithm
doesn't translate to WGSL (no ahash; no `&[u8]` parameter type).
A WGSL-side hash that satisfies P5's per-stream determinism
requirement (the canonical PCG-XSH-RR over the 4-tuple, mixed via
e.g. wyhash or xxhash3-32) needs to be emitted as a kernel-prelude
function alongside any kernel that touches `CgExpr::Rng`.

**One-line characterisation:** the WGSL caller exists; the WGSL
callee doesn't.

### Gap #3 — `"action"` is a WGSL string literal (HIGH)

**File:** `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:944`
**Body line:** `per_agent_u32(seed, agent_id, tick, "action")`

WGSL has no string type. The purpose tag must encode as a u32
constant — either:

  - a small enum (`action=1u`, `sample=2u`, `shuffle=3u`,
    `conception=4u`), assigned at emit time from `RngPurpose`; or
  - a fixed pre-computed hash of the purpose bytes (e.g. the FNV-1a
    32-bit hash of `b"action"`), so the WGSL primitive's fold mixes
    the same purpose-derived value the host-side `per_agent_u32`
    produces.

The CPU-side `per_agent_u32` hashes `purpose: &[u8]` directly via
ahash. To keep CPU↔GPU streams identical (a P5 requirement
documented at `docs/spec/dsl.md:782`), either both sides must use a
common hash over the byte slice, or the host-side primitive must
also accept a `purpose_id: u32` and the engine must canonicalise
the mapping.

**One-line characterisation:** WGSL doesn't have strings; the
emitter passes one anyway.

### Gap #4 — Spec drift on `rng.*` surface (LOW)

**Spec:** `docs/spec/dsl.md:922-925`
**Lowering:** `crates/dsl_compiler/src/cg/lower/expr.rs:2532-2544`

The spec table advertises four typed surfaces:

| spec entry | spec type |
|---|---|
| `rng.uniform(lo, hi)` | `(f32, f32) -> f32` |
| `rng.gauss(mu, sigma)` | `(f32, f32) -> f32` |
| `rng.coin()` | `() -> bool` |
| `rng.uniform_int(lo, hi)` | `(i32, i32) -> i32` |

The AST resolver registers all four
(`crates/dsl_ast/src/resolve.rs:254-257`); the CG lowering pass
recognises NONE of them, only the four nullary u32 purposes used by
the engine's per-agent stream derivation: `action`, `sample`,
`shuffle`, `conception`. A negative-test pin lives at
`crates/dsl_compiler/src/cg/lower/expr.rs:4029` confirming
`rng.uniform()` lowers to `UnsupportedNamespaceCall`.

**One-line characterisation:** spec advertises higher-level RNG
surfaces; today only the lowest-level u32 stream is wired.

## Files added

- `assets/sim/stochastic_probe.sim` (~110 LOC) — probe fixture.
  One Trigger Agent, one Activated event, one physics rule, one
  view-fold.
- `crates/stochastic_probe_runtime/Cargo.toml` (~22 LOC)
- `crates/stochastic_probe_runtime/build.rs` (~110 LOC) — mirrors
  `cooldown_probe_runtime/build.rs` shape verbatim.
- `crates/stochastic_probe_runtime/src/lib.rs` (~290 LOC) — Agent
  SoA (alive only) + event ring + activations ViewStorage +
  per-tick dispatch chain. Uses `EventRing::note_emits` /
  `tail_value()` cascade pattern from commit `16905527`.
- `crates/sim_app/src/stochastic_probe_app.rs` (~245 LOC) — harness
  driving 1000 ticks × 2 runs (P5 determinism check), reads back
  `activations`, asserts T × p = 300 per slot ±5%, prints OUTCOME
  line. Wraps the GPU path in `catch_unwind` to convert WGSL
  validation panics into clean OUTCOME (b) gap-list reports.
- `Cargo.toml` (workspace) — added `stochastic_probe_runtime` member.
- `crates/sim_app/Cargo.toml` — added the dep + `[[bin]]` entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `stochastic_probe_compile_gate` test (passing) — pins the
  structural surface (1 PhysicsRule + 1 ViewFold), the
  `per_agent_u32(`/`"action"` body shape, AND the negative naga
  validation result on `physics_MaybeFire` (locks the broken
  shape; flips when any of Gaps #1-#3 close).

Net LOC added: ~775 (above the ~600 budget; the .sim + lib.rs +
sim_app explanatory comments + the four-gap punch list dominate.
Strip the comments and it's ~340 LOC of executable code.)

## Compiler topology — what got emitted

The compiler lowered the program to **7 ComputeOps** (mirroring
cooldown_probe exactly):

```
op0: ViewFold      (view: activations,  on_event: Activated)
op1: PhysicsRule   (rule: MaybeFire,    on_event: None — Tick)
op2: Plumbing      (UploadSimCfg)
op3: Plumbing      (PackAgents)
op4: Plumbing      (SeedIndirectArgs ring=0)
op5: Plumbing      (UnpackAgents)
op6: Plumbing      (KickSnapshot)
```

The scheduler emitted **7 kernels** (no fusion — single physics
rule + single fold avoids the cross-domain fusion shape that bit
abilities_probe). Of those, **6 are naga-clean and 1 is
naga-broken** (the rng-touching `physics_MaybeFire`).

```
warning: stochastic_probe_runtime: emit-stats 7 kernels, schedule has 7 stages
warning:   fold_activations:  1515 B, 7 bindings    naga-clean
warning:   physics_MaybeFire: 1378 B, 4 bindings    naga-FAILS (Gap #1/#2/#3)
warning:   upload_sim_cfg:     666 B, 2 bindings    naga-clean
warning:   pack_agents:       3686 B, 41 bindings   naga-clean
warning:   seed_indirect_0:   1396 B, 4 bindings    naga-clean
warning:   unpack_agents:     3876 B, 41 bindings   naga-clean
warning:   kick_snapshot:      656 B, 2 bindings    naga-clean
```

## Determinism check (deferred)

P5 requires the per-agent stream to be a pure function of (seed,
agent_id, tick, purpose). The harness has the test wired (two
independent `StochasticProbeState::new(SEED, 32)` constructions, each
runs 1000 ticks, then byte-compares the activations buffers). The
test is **deferred** — it can't run until Gaps #1-#3 close because
the GPU path panics in `create_shader_module`. Once the WGSL emit
binds `seed` + emits `per_agent_u32` + encodes purpose as a u32, the
determinism check should pass byte-identically.

## Confirmation: no other fixtures touched

```
$ git status --short
?? assets/sim/stochastic_probe.sim
?? crates/stochastic_probe_runtime/
?? crates/sim_app/src/stochastic_probe_app.rs
?? docs/superpowers/notes/2026-05-04-stochastic_probe.md
 M Cargo.toml                                            (+1 member)
 M crates/sim_app/Cargo.toml                             (+1 dep + 1 bin)
 M crates/dsl_compiler/tests/stress_fixtures_compile.rs  (+1 test)
```

All 15 prior fixture apps in `crates/sim_app/src/` are unchanged.
No `crates/dsl_compiler/` source files outside the tests directory
were modified.

## Next steps (not in this slice)

A follow-up that closes Gaps #1-#3 would lift the rng surface from
"discovery only" to "production-grade", unblocking every future
`.sim` that wants per-agent stochasticity (currently zero
fixtures). The smallest sequence is:

  1. Add `seed: u64` to `engine_data::SimCfg`. Plumb a `seed: u32`
     field into every per-rule `Cfg` struct that mentions
     `DataHandle::Rng` in its `reads` set. Update the kernel
     preamble to bind `let seed = cfg.seed;`.
  2. Add a `wgsl_rng_prelude(purpose_id_consts: &[(u32, &str)])`
     helper that emits `fn per_agent_u32(seed: u32, agent_id: u32,
     tick: u32, purpose: u32) -> u32 { … pcg-xsh-rr … }` plus the
     purpose-id consts. Inject into any kernel module whose body
     uses the function.
  3. In `wgsl_body.rs:944`, swap the string literal for the matching
     `purpose_<name>` const reference.
  4. Update the host-side `per_agent_u32` to accept `purpose_id: u32`
     instead of `purpose: &[u8]`, with a small canonical mapping.
  5. Re-run this probe — observable should match T × p = 300 ±5%
     per slot AND determinism check should pass.

The probe + compile-gate stay; once the WGSL is naga-clean, swap
the `physics_naga.is_err()` assertion in
`stress_fixtures_compile.rs::stochastic_probe_compile_gate` to
`is_ok()` and add positive assertions on the new RNG lowering shape.
