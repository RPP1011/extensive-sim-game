# Quest probe — discovery report (2026-05-04)

This is the smallest end-to-end probe targeting **`entity X : Quest`**
AND the **`quests.*` namespace** — both surfaces are documented in
the spec but never exercised by any prior fixture in the workspace.

The probe shape mirrors prior probes (cooldown_probe, tom_probe):
1 producer physics rule + 1 view-fold + a `:Item` entity declaration
playing the role of a Quest analog (since `:Quest` parse-fails today).

## Outcome

**OUTCOME: NO FIRE / GAP CONFIRMED on three surfaces.** None of the
three originally-targeted surfaces wires through to the runtime
today. The probe surfaces them as a triplet of typed compiler gaps.
The fall-back live shape (the `Item`-rooted analog + a u32-result
view fed by `+= 1u`) runs end-to-end on the GPU but exposes a
**fourth, separate gap**: `+= 1u` on a u32-result view silently
lowers to `atomicOr`, leaving every per-slot value at `1u` instead
of accumulating to `100u`.

```
quest_probe_app: starting — seed=0x09E571715EE5DADA agents=32 ticks=100
quest_probe_app: finished — final tick=100 agents=32 progress.len()=32
quest_probe_app: progress readback — min=1 mean=1.000 max=1 sum=32
quest_probe_app: nonzero slots: 32/32 (fraction = 100.0%)
quest_probe_app: predicted shapes — (a) FULL FIRE: progress[N] = 100 for every N (atomicAdd-style); (b) GAP: progress[N] = 1 for every N (atomicOr idempotent)
quest_probe_app: ones_slots = 32/32 (would be 100% under GAP); ticks_slots = 0/32 (would be 100% under FULL FIRE)
quest_probe_app: preview progress[0..8] = [1, 1, 1, 1, 1, 1, 1, 1]
quest_probe_app: OUTCOME = (b) GAP CONFIRMED — every slot stuck at 1u.
```

## Files added

- `assets/sim/quest_probe.sim` (~140 LOC) — probe fixture. One live
  Adventurer Agent entity, one `Mission : Item` Quest analog, one
  ProgressTick event, one `physics ProgressAndComplete` rule, one
  `progress(agent: Agent) -> u32` view. Two commented-out lines
  (the `entity Mission : Quest` declaration and the `quests.is_
  active(0u)` call) document the directly-targeted gaps without
  blocking the rest of the program from compiling.
- `crates/quest_probe_runtime/Cargo.toml` (~24 LOC)
- `crates/quest_probe_runtime/build.rs` (~110 LOC) — mirrors
  `cooldown_probe_runtime/build.rs` shape verbatim.
- `crates/quest_probe_runtime/src/lib.rs` (~280 LOC) — Agent SoA
  (`alive` only) + event ring + raw u32 `progress_primary`
  storage + per-tick dispatch chain (mirrors `tom_probe_runtime`'s
  raw-u32 staging shape — `engine::ViewStorage::readback` returns
  `&[f32]`, which would mis-cast u32 atomic bits to f32).
- `crates/sim_app/src/quest_probe_app.rs` (~145 LOC) — harness
  driving 100 ticks, reads back `progress`, classifies the
  observed pattern (FULL FIRE vs GAP CONFIRMED vs PARTIAL).
- `Cargo.toml` (workspace) — added `crates/quest_probe_runtime`
  member.
- `crates/sim_app/Cargo.toml` — added the dep + `[[bin]]` entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `quest_probe_compile_gate` test (passing) — locks the structural
  surface (1 PhysicsRule + 1 ViewFold), the `Mission : Item`
  catalog entry, the `atomicOr` LIVE GAP (asserts the WGSL
  contains `atomicOr`, locking that any future fix forcing
  `atomicAdd` semantics on u32 views surfaces here as a test
  failure), and naga validation cleanliness.

Net LOC added: ~700 (matches budget; the .sim + lib.rs +
quest_probe_app.rs explanatory comments dominate).

## Compiler topology — what got emitted

The compiler lowered the program to **7 ComputeOps** (single
producer + single fold, no fusion shape — same compositional shape
as cooldown_probe):

```
op0: ViewFold      (view: progress, on_event: ProgressTick)
op1: PhysicsRule   (rule: ProgressAndComplete, on_event: None — Tick)
op2: Plumbing      (UploadSimCfg)
op3: Plumbing      (PackAgents)
op4: Plumbing      (SeedIndirectArgs ring=0)
op5: Plumbing      (UnpackAgents)
op6: Plumbing      (KickSnapshot)
```

The scheduler emitted **7 kernels**:

```
fold_progress
physics_ProgressAndComplete
upload_sim_cfg
pack_agents
seed_indirect_0
unpack_agents
kick_snapshot
```

The `fold_progress` kernel WGSL (key fragment):

```wgsl
@group(0) @binding(2) var<storage, read_write> view_storage_primary: array<atomic<u32>>;
...
{
    let _idx = local_0;
    atomicOr(&view_storage_primary[_idx], (1u));
}
```

The `+= 1` self-update on a u32 view emits `atomicOr(&storage[_idx],
(1u))` — identical to the WGSL the same view would produce under
`|= 1u`. Operator semantics are silently dropped at the result-type
branch in `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:1326-1338`.

## Three originally-targeted gaps

### Gap A (HIGH) — `EntityRoot::Quest` not in parser accept set

The spec table at `docs/spec/dsl.md:653-663` lists `Quest` alongside
`Agent`, `Item`, `Group`, `Auction`, `Invite`, `Document`,
`ChronicleEntry` as a typed entity root. The parser at
`crates/dsl_ast/src/parser.rs:373-385` only accepts `Agent`, `Item`,
`Group` and rejects everything else with:

```
"expected `Agent`, `Item`, or `Group`; got `Quest`"
```

(verified by uncommenting the `entity Mission : Quest { reward:
f32 }` line in the fixture; the test fails at parse with span
`4122..4127`).

The IR-level enum `dsl_ast::ast::EntityRoot` at
`crates/dsl_ast/src/ast.rs:147-151` only has the three variants
(no `Quest`, `Auction`, `Invite`, `Document`, `ChronicleEntry`).

**Fix shape**: extend the enum + parser accept set + the `populate_
entity_field_catalog` driver match (`crates/dsl_compiler/src/cg/lower/
driver.rs:843-847`) to route `EntityRoot::Quest` to a new
`catalog.quests` map (or fold it into a generic per-root hashmap).
Today the catalog has separate `items` + `groups` fields; the same
shape extends to quests.

### Gap B (HIGH) — `quests.*` namespace registered with ZERO methods

The resolver routes `quests` through `NamespaceId::Quests`
(`crates/dsl_ast/src/resolve.rs:117`) but `populate_namespace_
registry` (`crates/dsl_compiler/src/cg/lower/driver.rs:672-805`)
registers methods only for `agents`, `query`, `world`, `auctions`.
Any `quests.<method>(...)` call falls through to the catch-all at
`crates/dsl_compiler/src/cg/lower/expr.rs:2714` with:

```
LoweringError::UnsupportedNamespaceCall {
  namespace: Quests,
  method: "<name>",
  ...
}
```

(verified by uncommenting `let _active = quests.is_active(0);` in
the physics body; the lower-time diagnostic surfaces:
`namespace call 'quests.is_active()' at 5856..5875 has no
expression-level lowering`).

**Fix shape**: register a `quests` `NamespaceDef` in `populate_
namespace_registry` mirroring the `auctions` B1 stubs (lines 745-802):
each method's WGSL stub is a placeholder no-op (e.g. `is_active`
returns `false`, `eligible_for` returns `false`, `count` returns
`0u`). The spec at `docs/spec/dsl.md:1072` lists three methods on
the singular `quest` namespace (`can_accept`, `is_target`,
`party_near_destination`); the plural `quests` namespace's method
table isn't enumerated in the spec but the natural set is the
collection-shaped accessors (`count`, `is_active`, `for_target`).

### Gap C (LOW — but the fixture's LIVE gap) — `+= 1u` on u32 view → atomicOr

The view-body lowering at `crates/dsl_compiler/src/cg/lower/view.rs:
564` accepts both `+=` and `|=` operators on any view (the operator
gate doesn't discriminate by view result type). The WGSL emitter
at `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:1326-1338` then
branches on the view's `result` type only — NOT the operator:

```rust
if matches!(view_result_ty, Some(crate::cg::expr::CgTy::U32)) {
    return Ok(format!(
        "{{\n\
         \x20   let _idx = {idx_expr};\n\
         \x20   atomicOr(&{storage}[_idx], ({rhs}));\n\
         }}"
    ));
}
// ... f32 CAS+add fallthrough
```

So `self += 1u` on a `u32`-result view emits `atomicOr(&storage[idx],
(1u))` — bitwise OR, NOT addition. With rhs = `1u` constant, the
op is idempotent: every emit ORs `1u` into the slot, leaving the
per-slot value at `1u` regardless of fire count.

The runtime confirms this end-to-end: 32 Adventurers × 100 ticks =
3200 events folded; observed `progress` = 32 × 1 = 32 (instead of
the operator-intent 32 × 100 = 3200).

**Fix shape (one of two)**:
  - **(a)** Add a separate fold semantic — when the operator is
    `+=` on a `u32`-result view, emit `atomicAdd(&storage[idx],
    rhs)`. WGSL's `atomicAdd<u32>` is commutative + associative
    (P11-trivial — same guarantee as `atomicOr`), so no CAS retry
    is needed. This preserves the operator's add semantics.
  - **(b)** Reject `+=` at view-body lower time when the view
    result is `u32` AND the rhs isn't bit-shaped (force designers
    to write `|=` for the bit-OR case). This surfaces the gap as a
    typed `LoweringError::UnsupportedFoldOperator` rather than
    silent miscompile.

Today the only u32 view shipped is tom_probe's `beliefs` (which
uses `|=` correctly). This fixture is the first to write `+=` on
a u32 view, which is why the gap stayed buried.

## What this fixture DOES exercise (live wins)

Even with three of the originally-targeted surfaces gapped, the
probe is a small but novel coverage win:

- **`Mission : Item` declaration as Quest analog** — the entity
  reaches the `entity_field_catalog.items` map with `entity_name
  == "Mission"` and a single `reward: f32` field. The catalog
  entry is present (asserted in the compile-gate test) but no
  kernel reads it (no `items.reward(...)` call site). Confirms
  the Item-root catalog path tolerates a "quest-shaped" field
  declaration (single primitive field, no nested struct).
- **`progress(agent: Agent) -> u32` view with `+= 1u` self-update**
  — first fixture to combine a u32-result view with the `+=`
  operator. tom_probe uses `|=` and bartering's `trade_count` uses
  f32 + `+= 1.0`. This combination triggers Gap C above.
- **End-to-end u32 view readback** in a single-key (no pair_map)
  shape — tom_probe is the only other u32 view but it uses
  pair_map storage, so the runtime's raw u32 staging-buffer dance
  picks up a new shape (smaller storage = `agent_count` slots).

## Reproducer

```bash
cargo build -p quest_probe_runtime              # clean
cargo run -p sim_app --bin quest_probe_app      # OUTCOME (b) GAP CONFIRMED
cargo test -p dsl_compiler --test stress_fixtures_compile quest_probe_compile_gate -- --nocapture
```

To re-verify Gaps A + B as the same probe:

1. Uncomment `entity Mission : Quest { reward: f32 }` (line 82) →
   parse-fail at `quest_probe.sim:82:18` with the `Quest` reject
   message.
2. Insert `let _active = quests.is_active(0);` at the top of the
   physics body → lower-time `UnsupportedNamespaceCall` diagnostic.

Both reproductions are deterministic (parse + resolve are pure);
restoring the file restores the live shape.

## Constitution touch-points

- **P1 Compiler-First**: no compiler changes in this slice. The
  probe surfaces three structural gaps and locks the LIVE GAP
  (Gap C) with a compile-gate `atomicOr` assertion so a future
  fix surfaces here.
- **P9**: closing with a verified commit — the runtime builds
  clean, the workspace tests pass (28 stress-fixture tests,
  +1 from baseline 27), the existing 18 fixture apps unchanged.
- **P11**: no new atomic primitives. The runtime composes
  `EventRing` + raw u32 staging (mirroring tom_probe) with no
  new helpers; the gap is in the COMPILER's choice of which
  atomic primitive to emit, not in the runtime's primitives.
