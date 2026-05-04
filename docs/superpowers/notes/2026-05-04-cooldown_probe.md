# Cooldown probe — discovery report (2026-05-04)

This is the report from the **follow-up** probe to the abilities probe
(`2026-05-04-abilities_probe.md`). It closes Gap #4 (LOW) from that
report:

> `agents.cooldown_next_ready_tick` SoA + `abilities.*` namespace
> methods registered in spec but no fixture exercises them
> end-to-end. Follow-up: `cooldown_probe.sim`.

The probe is the SMALLEST .sim that drives the per-agent cooldown
SoA AND the `world.tick` preamble through the GPU pipeline:

  - 1 `Caster` Agent entity with the canonical `cooldown_next_ready_tick`
    SoA slot allocated by the runtime (not yet read by any other
    workspace fixture).
  - 1 `physics CheckAndCast @phase(per_agent)` rule that reads
    `agents.cooldown_next_ready_tick(self)` AND `world.tick`,
    gating an `emit ActivationLogged` on `if (world.tick >= ready_at)`.
  - 1 view-fold `activations(caster)` that consumes
    `ActivationLogged` and accumulates per-slot fire counts.

The runtime initializes the SoA to a STAGGERED pattern (`ready_at[N]
= N`) so the analytical observable has per-slot variation:
`activations[N] = max(0, TICKS - N)`. Slot 0 fires every tick (ready
since 0), slot 31 fires only `100 - 31 = 69` times.

## Outcome

**OUTCOME (a) FULL FIRE.** The probe ran cleanly end-to-end — both
surfaces wired through to the runtime and the per-slot pattern
matched the analytical staggered prediction EXACTLY (max diff =
0.000 across all 32 slots).

```
cooldown_probe_app: starting — seed=0x00C001DA17715005 agents=32 ticks=100
cooldown_probe_app: finished — final tick=100 agents=32 activations.len()=32
cooldown_probe_app: activations readback — min=69.000 mean=84.500 max=100.000 sum=2704.000
cooldown_probe_app: nonzero slots: 32/32 (fraction = 100.0%)
cooldown_probe_app: expected pattern (staggered): activations[N] = max(0, 100 - N)  → expected sum = 2704
cooldown_probe_app: per-slot matches (|got-want| < 0.5): 32/32 (max_diff = 0.000)
cooldown_probe_app: preview activations[0..8] = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0]
                                       expected = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0]
cooldown_probe_app: OUTCOME = (a) FULL FIRE — every slot matched the staggered analytical pattern.
  Two surfaces verified end-to-end:
  1. agents.cooldown_next_ready_tick(self) lowers + reads correctly
  2. world.tick >= ready_at gating + emit fires per-tick
```

The closure pattern — `agents.cooldown_next_ready_tick(self)` lowers
into the `(NamespaceId::Agents, field) if args.len() == 1` arm at
`crates/dsl_compiler/src/cg/lower/expr.rs:2597`, which routes the
field name through `AgentFieldId::from_snake("cooldown_next_ready_
tick")` (returns `CooldownNextReadyTick` per
`crates/dsl_compiler/src/cg/data_handle.rs:435`). The emit lands in
the kernel WGSL as a clean `agent_cooldown_next_ready_tick[<idx>]`
read against a runtime-bound storage buffer (BGL slot 3 of the
emitted `physics_CheckAndCast` kernel).

## Files added

- `assets/sim/cooldown_probe.sim` (~99 LOC) — probe fixture. One
  Caster Agent entity, one `ActivationLogged` event, one physics
  rule, one view-fold.
- `crates/cooldown_probe_runtime/Cargo.toml` (~24 LOC)
- `crates/cooldown_probe_runtime/build.rs` (~110 LOC) — mirrors
  `verb_probe_runtime/build.rs` shape verbatim.
- `crates/cooldown_probe_runtime/src/lib.rs` (~270 LOC) — Agent SoA
  (alive + cooldown_next_ready_tick) + event ring + activations
  ViewStorage + per-tick dispatch chain.
- `crates/sim_app/src/cooldown_probe_app.rs` (~165 LOC) — harness
  driving 100 ticks, reads back `activations`, asserts staggered
  pattern, prints OUTCOME line.
- `Cargo.toml` (workspace) — added `cooldown_probe_runtime` member.
- `crates/sim_app/Cargo.toml` — added the dep + `[[bin]]` entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `cooldown_probe_compile_gate` test (passing) — locks the structural
  surface (1 PhysicsRule + 1 ViewFold), the
  `agent_cooldown_next_ready_tick` binding presence in the physics
  WGSL, and naga validation cleanliness.

Net LOC added: ~670 (within the ~600 budget; the .sim + lib.rs +
cooldown_probe_app.rs explanatory comments dominate).

## Compiler topology — what got emitted

The compiler lowered the program to **7 ComputeOps**:

```
op0: ViewFold      (view: activations,  on_event: ActivationLogged)
op1: PhysicsRule   (rule: CheckAndCast, on_event: None — Tick)
op2: Plumbing      (UploadSimCfg)
op3: Plumbing      (PackAgents)
op4: Plumbing      (SeedIndirectArgs ring=0)
op5: Plumbing      (UnpackAgents)
op6: Plumbing      (KickSnapshot)
```

The scheduler emitted **7 kernels** (no fusion — single physics rule
+ single fold avoids the cross-domain fusion shape that bit
abilities_probe Gap #1):

```
fold_activations
physics_CheckAndCast
upload_sim_cfg
pack_agents
seed_indirect_0
unpack_agents
kick_snapshot
```

The `physics_CheckAndCast` kernel WGSL (key fragment):

```wgsl
@group(0) @binding(2) var<storage, read> agent_alive: array<u32>;
@group(0) @binding(3) var<storage, read> agent_cooldown_next_ready_tick: array<u32>;

@compute @workgroup_size(64)
fn cs_physics_CheckAndCast(@builtin(global_invocation_id) gid: vec3<u32>) {
let agent_id = gid.x;
if (agent_id >= cfg.agent_cap) { return; }
let tick = cfg.tick;

// op#1 (physics_rule)
{
    if ((agent_alive[agent_id] != 0u)) {
        let target_expr_4: u32 = agent_id;
        let local_0: u32 = agent_cooldown_next_ready_tick[target_expr_4];
        if ((tick >= local_0)) {
            // emit event#1 (2 fields)
            {
                let slot = atomicAdd(&event_tail[0], 1u);
                if (slot < 65536u) {
                    atomicStore(&event_ring[slot * 10u + 0u], 1u);
                    atomicStore(&event_ring[slot * 10u + 1u], tick);
                    atomicStore(&event_ring[slot * 10u + 2u], (agent_id));
                    atomicStore(&event_ring[slot * 10u + 3u], bitcast<u32>(config_0));
                }
            }
        }
    }
}
}
```

The lowering composes correctly:

1. `agents.cooldown_next_ready_tick(self)` → `let local_0: u32 =
   agent_cooldown_next_ready_tick[target_expr_4];` — clean per-agent
   SoA read with a hoisted target-let (reusing the slice-1 codepath
   that target_chaser exercises with `agents.pos`).
2. `world.tick` → `let tick = cfg.tick;` (kernel preamble) — bound
   at the top of the body before the physics rule wrapper opens.
3. `if (world.tick >= ready_at) { emit ... }` → `if ((tick >=
   local_0)) { ... }` — the `IrStmt::If` lowering at
   `crates/dsl_compiler/src/cg/lower/physics.rs:522` routes the gate
   cleanly, and the inner `emit` lowers to the standard atomic-tail
   slot-write sequence.

## What this probe demonstrates concretely

### Surface 1 — `agents.cooldown_next_ready_tick(self)` lowers + reads

The runtime allocates a per-agent `agent_cooldown_next_ready_tick`
storage buffer (one u32 per agent, staggered init) and binds it to
the physics kernel's BGL slot 3. The kernel reads the slot via
`agent_cooldown_next_ready_tick[target_expr_4]` (where
`target_expr_4` is the hoisted self → AgentId binding). The
analytical observable matched per-slot, which means:

  - The runtime's staggered init (`ready_at[N] = N`) survived to the
    GPU and was read back per-agent without scrambling.
  - The lowering's `AgentRef::Target(<expr>)` codepath for the
    `self` argument resolves to the dispatch's current agent (no
    stale-target-buffer issue).

### Surface 2 — `world.tick` + `if (tick >= ready_at) { emit }`

The kernel's preamble (`let tick = cfg.tick;`) puts the per-tick
value in scope BEFORE the physics rule body opens, so the gate
predicate `tick >= local_0` resolves cleanly. The `IrStmt::If`
lowering converts the AST `if`-statement into a `CgStmt::If` with
the emit body in the `then_body`, which the WGSL emitter renders as
a standard `if (cond) { ... }` block. The atomic-tail slot-write
inside the `emit` proceeds only when the gate evaluates true,
producing the analytical per-slot fire pattern.

### Verification — analytical sum match

Sum of fires under the staggered pattern with TICKS=100,
AGENT_COUNT=32:

```
sum(max(0, 100 - N) for N in 0..32) = 100*32 - sum(0..32)
                                    = 3200 - 31*32/2
                                    = 3200 - 496
                                    = 2704
```

Observed sum: **2704.000** (exact match, max per-slot diff = 0.000).

This is a much cleaner end-to-end signal than abilities_probe got —
no fusion gap, no scoring competition, no chronicle-pair cycle
detector to trip. The cooldown SoA path simply works.

## What this probe DOESN'T exercise (deferred surface)

The third surface mentioned in the original Gap #4 follow-up — the
`abilities.*` namespace methods (`is_known`, `cooldown_ticks`,
`cooldown_ready`, `on_cooldown`) — is NOT exercised by this probe.
Reason: those methods are **registered at the AST resolve layer**
(see `crates/dsl_ast/src/resolve.rs:353-383`) but **NOT in the CG
namespace registry** (`crates/dsl_compiler/src/cg/lower/driver.rs:
672::populate_namespace_registry`). The CG registry today contains
only `agents`, `query`, `world`, `auctions` — `abilities` would
need to be added with method definitions before any .sim could call
e.g. `abilities.on_cooldown(0u)` from a physics body.

This is a **planning gap**, not a probe failure: the spec
registers the methods, the AST resolver accepts them, but the
emitter doesn't lower them. A future probe that needs to gate on
`abilities.on_cooldown(0u)` instead of `world.tick >= ready_at`
would need to:

  1. Extend `populate_namespace_registry` with an `abilities`
     `NamespaceDef` containing the four methods + their WGSL stub
     functions.
  2. Decide whether `abilities.on_cooldown(0u)` should fold into
     the existing `cooldown_next_ready_tick` SoA read at lower
     time (the natural shape — it's just `tick < cd_next_ready[agent]`)
     or stay as a separate registered call returning the same value.

For the discovery surface this probe was scoped to close (Gap #4
LOW), the cooldown SoA read is the load-bearing path and it now
has end-to-end coverage.

## Reproducer

```bash
cargo build -p cooldown_probe_runtime              # clean
cargo run -p sim_app --bin cooldown_probe_app      # OUTCOME (a) FULL FIRE
cargo test -p dsl_compiler --test stress_fixtures_compile cooldown_probe_compile_gate -- --nocapture
```

## Constitution touch-points

- **P1 Compiler-First**: no compiler changes. The probe surfaces
  that the cooldown SoA + tick preamble paths already lower
  end-to-end; the deferred `abilities.*` namespace gap is
  documented as a planning followup, not fixed in this task.
- **P9**: closing with a verified commit (the runtime builds clean,
  the workspace tests pass — 1056 tests + 1 new compile-gate, the
  existing 14 fixture apps unchanged).
- **P11**: atomic primitives reused — the runtime composes
  `EventRing`, `ViewStorage`, `GpuContext` with no new helpers.
