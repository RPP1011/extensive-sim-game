# Engine Crate Split (Plan B1') — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Supersedes:** `2026-04-25-engine-crate-restructure-impl.md` (Plan B1, written against the v1 spec).
>
> **Rewritten 2026-04-25** during execution after Plan B1' brainstorming surfaced that the settled architecture is more aggressive than v2's task list captured. See Spec B' §0 (v2 refinement note) and §12 (optimization rationale). The rewrite absorbs Plan B2 (chronicle + engagement migration), moves `SimState` to engine_data as a compile-time artifact, and lifts typed-ID newtypes from engine_data into engine.

**Goal:** Land the truly-universal-engine end-state per Spec B' (with v2 refinements + §12). After this plan: engine is a generic primitive framework with no `.sim`-specific knowledge; engine_data holds every emitted shape including `SimState` and the rule-level enums (`CreatureType`, `Capabilities`, `ChannelSet`, etc.); engine_rules holds every emitted behavior including the phase-orchestration `step` body, `SerialBackend`, mask-fill orchestration, view registry, cascade-registry population, plus the migrated chronicle renderer; engine_gpu mirrors in WGSL.

**Architecture:** Strict layering `engine ← engine_data ← engine_rules` (and `engine_gpu` parallel to engine_rules). Engine never imports engine_data or engine_rules. engine_data uses engine's typed-ID newtypes + container generics. engine_rules uses both. Build sentinels structurally enforce: engine_rules + engine_data require `// GENERATED`; engine rejects `// GENERATED` markers and constrains top-level layout to a primitives-only allowlist with **no exceptions** (chronicle + engagement absorbed; no `event_like_impl.rs` workaround).

**Tech Stack:** Rust 2021 with generics for primitive containers, `dsl_compiler` Rust + WGSL emit, `trybuild` for compile-fail tests, `ast-grep` for CI rules, bash for pre-commit + hooks.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `pub struct EventRing` at `crates/engine/src/event/ring.rs:14` (now generic post-Task 1)
  - `pub struct CascadeRegistry` at `crates/engine/src/cascade/dispatch.rs:22` (now generic post-Task 1)
  - `pub trait CascadeHandler<E>` at `crates/engine/src/cascade/handler.rs` (now generic post-Task 1)
  - `pub struct SimState` at `crates/engine/src/state/mod.rs` — to be moved to engine_data as emitted artifact
  - `pub struct SimScratch` at `crates/engine/src/step.rs` — same
  - Typed ID re-exports: `engine/src/ids.rs` (`pub use engine_data::ids::*`), `creature.rs`, `channel.rs`, `policy/macro_kind.rs` — to be inverted (engine declares; engine_data uses)
  - `engine/src/chronicle.rs` (371 lines) + `engine/src/engagement.rs` (44 lines) — to migrate
  - 35 emitted physics/mask/view files in `engine/src/generated/` — to move
  - `engine/src/{step,backend}.rs` — to move (CpuBackend → SerialBackend)
  - `mark_*_allowed` methods in `engine/src/mask.rs` — to move

  Search method: `rg`, `grep -rE`, direct `Read`.

- **Decision:** restructure per Spec B' (with v2 refinements). The engine becomes a reusable primitive framework; everything `.sim`-specific moves to emitted artifacts.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: chronicle templates added to `assets/sim/` (DSL grammar may need a `chronicle` declaration form — see Task 9 for grammar status); engagement break-reason constants likewise.
  - Generated outputs re-emitted: `engine_data/src/{events,sim_state,sim_scratch,creature_type,channels,types,scoring,...}.rs` and `engine_rules/src/{mask,physics,views,step,backend,mask_fill,cascade,chronicle}.rs`.
  - New emit modules: `dsl_compiler/src/{emit_sim_state,emit_step,emit_backend,emit_mask_fill,emit_cascade_register,emit_chronicle}.rs`.

- **Hand-written downstream code:**
  - Engine's typed-ID newtypes (`AgentId`, `EntityId`, `AbilityId`, `EventId`): NEW (lifted from engine_data). Justification: primitive identifier types with type-safety value; not rule-data.
  - Engine's `EventLike` trait + four sealed view-trait declarations + `__sealed::Sealed`: NEW. Justification: framework interfaces that any DSL targets.
  - Engine's `MaskBuffer` runtime-sized + raw bit ops: kept (already there).
  - `engine_rules/src/lib.rs`: marker + blanket Sealed impls + module re-exports. Allowlisted (not `// GENERATED`).
  - `engine_data/src/lib.rs`: re-export hub. Allowlisted.
  - `engine_data/build.rs` + `engine_rules/build.rs`: NEW header-rule sentinels.
  - `engine/build.rs`: NEW primitives-only allowlist with NO exceptions (chronicle + engagement migrated, not allowlisted).
  - `engine/tests/sealed_cascade_handler.rs` + UI fixtures: NEW (trybuild compile-fail test).
  - `.githooks/pre-commit`: extend with header rule + regen-on-DSL-change.
  - `.ast-grep/rules/*.yml`: four trait-impl-location rules.
  - CI workflow: stale-content + schema-hash + ast-grep steps.

- **Constitution check:**
  - P1 (Compiler-First): PASS — engine has zero rule-aware code post-plan.
  - P2 (Schema-Hash on Layout): PASS — `SimState` layout becomes part of the emitted schema; `.schema_hash` already covers this. The migration regenerates the baseline.
  - P3 (Cross-Backend Parity): PASS — `engine_gpu`'s parallel WGSL emit mirrors engine_rules' Rust emit.
  - P4 (`EffectOp` Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): N/A — RNG primitives unchanged.
  - P6–P7: N/A.
  - P8: PASS.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends green.
  - P10 (No Runtime Panic): PASS — `build.rs` panics fire at build time.
  - P11: N/A.

- **Re-evaluation:** [x] AIS reviewed at design phase. [ ] AIS reviewed post-design (tick after final task).

---

## State at plan-rewrite time

Worktree `.worktrees/engine-crate-restructure`, branch `engine-crate-restructure`. Two tasks already landed under the old version of B1':

| Commit | What | Status |
|---|---|---|
| `d4d06390` | Rename `engine_generated` → `engine_data` (workspace-wide sed) | LANDED |
| `da008ac3` | engine deps `engine_data` directly (transitional; final plan inverts) | LANDED — to be revised in Task 3 |
| `e60619c6` | Generic `EventRing<E>`/`CascadeRegistry<E>`/view-traits + `EventLike` trait | LANDED — partial |
| `33e73ceb` | Emitted `impl EventLike for Event` (currently in `engine/src/event/event_like_impl.rs` as a workaround) | LANDED — workaround; Task 5 cleans up |

Tasks below pick up from this state.

## File Structure (final, post-plan)

```
crates/
  engine/                          PRIMITIVES + INTERFACES (no .sim knowledge)
    build.rs                       NEW: primitives-only allowlist + reject // GENERATED (zero exceptions)
    src/
      lib.rs                       SHRUNK: declares the framework surface only
      ids.rs                       MODIFIED: declares AgentId, EntityId, AbilityId, EventId NEWTYPES (lifted from engine_data)
      backend.rs                   MODIFIED: SimBackend trait kept; CpuBackend impl deleted
      cascade/{handler,dispatch}.rs  generic CascadeHandler<E>, CascadeRegistry<E>, sealed (post-Task 1)
      event/{ring,mod}.rs          generic EventRing<E>, EventLike trait (post-Task 1)
      mask.rs                      MODIFIED: storage + raw ops only; no mark_*_allowed
      view/{materialized,lazy,topk}.rs  MODIFIED: trait declarations <E> + sealed
      pool/, pool.rs, rng.rs, schema_hash.rs, spatial.rs, terrain.rs,
      trajectory.rs, ability/, aggregate/, invariant/, obs/, policy/,
      probe/, snapshot/, telemetry/, channel.rs (storage primitive only),
      creature.rs                  MODIFIED: drop engine_data re-exports; channel.rs storage type kept
      generated/                   DELETED
      step.rs                      DELETED
      chronicle.rs                 DELETED (migrated to engine_rules emitted)
      engagement.rs                DELETED (migrated to engine_data emitted)
    tests/
      sealed_cascade_handler.rs    NEW: trybuild driver
      ui/external_impl_rejected.{rs,stderr}  NEW: compile-fail fixture
  engine_data/                     EMITTED SHAPES (depends on engine)
    build.rs                       NEW: every-file-must-be-generated sentinel
    src/
      lib.rs                       module declarations (allowlisted)
      sim_state.rs                 NEW (emitted): SimState SoA struct from DSL agent fields
      sim_scratch.rs               NEW (emitted): SimScratch
      events/                      regenerated: Event enum + impl EventLike
      creature_type.rs             regenerated: CreatureType enum + HOSTILITY + CREATURE_DEFAULTS
      channels.rs                  regenerated: channel-set bit names
      mask_kinds.rs                regenerated: MaskKindId enum + MicroKind
      engagement.rs                NEW (emitted): break_reason constants
      <other shapes>               regenerated
  engine_rules/                    EMITTED BEHAVIOR (deps engine + engine_data)
    build.rs                       NEW: every-file-must-be-generated sentinel
    src/
      lib.rs                       allowlisted: GeneratedRule + Sealed blanket + module re-exports
      mask/                        regenerated
      physics/                     regenerated
      views/                       regenerated
      step.rs                      NEW (emitted): phase orchestration body
      backend.rs                   NEW (emitted): SerialBackend impl
      mask_fill.rs                 NEW (emitted): mark_*_allowed orchestration
      cascade.rs                   NEW (emitted): with_engine_builtins
      chronicle.rs                 NEW (emitted): template renderer
  engine_gpu/                      WGSL parallel — touched only as needed; full parallel emit follow-up

dsl_compiler/
  src/
    emit_sim_state.rs              NEW: emit SimState struct + impl Default + accessors
    emit_step.rs                   NEW: emit step body
    emit_backend.rs                NEW: emit SerialBackend
    emit_mask_fill.rs              NEW: emit mark_*_allowed
    emit_cascade_register.rs       NEW: emit with_engine_builtins
    emit_chronicle.rs              NEW: emit chronicle::render_entry
    emit_engagement.rs             NEW: emit break_reason constants
    emit_physics.rs                MODIFIED: emit `use engine_data::events::Event` etc.; emit `impl GeneratedRule`
    emit_mask.rs                   MODIFIED: same
    emit_view.rs                   MODIFIED: same + `impl GeneratedRule` for view structs
    lib.rs                         MODIFIED: dispatch new emit passes

src/bin/xtask/
  cli/mod.rs                       MODIFIED: out_* defaults for new emit targets + --check flag
  compile_dsl_cmd.rs               MODIFIED: drive new emit passes; implement --check

.githooks/pre-commit               MODIFIED: header rule + regen-on-DSL-change
.ast-grep/rules/*.yml              NEW: 4 trait-impl-location rules
.github/workflows/<ci>.yml         MODIFIED: ast-grep + stale-content + schema-hash steps
```

---

## Tasks

Tasks 1 + 2 are already landed; documented for completeness. Tasks 3+ are the active work.

### Task 1: Generic primitives in engine ✓ LANDED (`e60619c6`)

`EventLike` trait + generic `EventRing<E>` / `CascadeRegistry<E>` / `CascadeHandler<E>` / view traits added. ~7 framework files touched + small set of internal users. No new functionality; structural prep.

### Task 2: Emit `impl EventLike for Event` ✓ LANDED (`33e73ceb`)

Compiler-emitted `impl engine::event::EventLike for engine_data::events::Event`. Currently lives at `crates/engine/src/event/event_like_impl.rs` as a `// GENERATED` workaround pending the dep direction settlement. Task 5 moves it to its proper home.

---

### Task 3: Lift typed-ID newtypes from engine_data into engine

**Goal:** `AgentId`, `EntityId`, `AbilityId`, `EventId` are *primitive identifier types* (newtypes around `u32`/`u16` with type safety). They belong in engine, not engine_data. Today engine re-exports them via `pub use engine_data::ids::*` — invert the dep.

**Files:**
- Move: declarations from `crates/engine_data/src/ids.rs` → `crates/engine/src/ids.rs` (replace the current `pub use` re-export with the actual definitions).
- Modify: `crates/dsl_compiler/src/emit_*.rs` — anywhere they emit `pub use engine_data::ids::*` or reference `engine_data::ids::AgentId`, change to `engine::AgentId`.
- Sed-update: callers across the workspace using `engine_data::ids::*` to use `engine::*`.

- [ ] **Step 1:** Read current `engine_data/src/ids.rs`. Copy struct definitions verbatim. (They should be plain `pub struct AgentId(pub u32);` etc. with derives.)

- [ ] **Step 2:** Replace `crates/engine/src/ids.rs` content with the struct definitions, dropping the `pub use engine_data::ids::*` line.

- [ ] **Step 3:** Delete `crates/engine_data/src/ids.rs` (or leave a thin `pub use engine::*` re-export for transition; cleaner: delete and update emit-site references).

- [ ] **Step 4:** Sed callers:

```bash
git grep -l 'engine_data::ids' | xargs sed -i 's|engine_data::ids|engine|g'
```

- [ ] **Step 5:** If `dsl_compiler` emits `pub use engine_data::ids::*` into any output, change the emit to point at engine. Audit:

```bash
grep -rE '"use engine_data::ids|engine_data::ids' crates/dsl_compiler/src/
```

- [ ] **Step 6:** Build + test.

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

Expected: SUCCESS (modulo pre-existing rng-golden failure + intermediate-state engine_gpu issues from prior tasks).

- [ ] **Step 7:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor(engine): lift AgentId/EntityId/AbilityId/EventId newtypes from engine_data into engine"
```

---

### Task 4: Drop engine's `pub use engine_data::*` shape-type re-exports

**Goal:** Engine no longer references the rule-level shape types (`CreatureType`, `Capabilities`, `ChannelSet`, `LanguageId`, `QuestCategory`, `Resolution`, `CommunicationChannel`). These types stay in engine_data; engine code that touched them goes generic, gets deleted, or stores raw ordinals.

**Files audit + modify (each is small):**
- `crates/engine/src/creature.rs` — drops `pub use engine_data::entities::{Capabilities, CreatureType}` + `pub use engine_data::types::LanguageId`. The file becomes a thin storage-helper module if it survives, or merges into `engine/src/state/`.
- `crates/engine/src/channel.rs` — drops `pub use engine_data::types::{ChannelSet, CommunicationChannel}` + `use engine_data::config::CommunicationConfig`. Storage type stays as raw `u64` bitmask wrapper.
- `crates/engine/src/policy/macro_kind.rs` — drops `pub use engine_data::types::{QuestCategory, Resolution}`. The macro_kind dispatch logic moves to engine_rules (it's rule-data) OR stays as raw u8 storage in engine.
- `crates/engine/src/view/{lazy,topk}.rs` — the demo impls reference concrete `engine_data::events::Event`. Either they're already cfg(test)-only seal impls (Task 1 made the impls generic), or we keep them as test-only helpers in `engine_data` instead.

- [ ] **Step 1:** Audit each file:

```bash
for f in crates/engine/src/creature.rs crates/engine/src/channel.rs crates/engine/src/policy/macro_kind.rs; do
    echo "=== $f ==="
    grep -E '^use engine_data|^pub use engine_data' "$f"
done
```

- [ ] **Step 2:** For each file, decide: drop the import + remove dependent code, OR move the dependent code to engine_rules. Walk through:

  - **`creature.rs`:** if it's mostly type re-exports, replace with a thin storage module (`pub fn agent_creature_type_ordinal(state, agent) -> u16`). The named type lives in engine_data.
  - **`channel.rs`:** keep as raw `u64` bitmask storage primitive. The bit names are engine_data.
  - **`policy/macro_kind.rs`:** if it's just type aliases, delete; if it has dispatch logic, move to engine_rules.

- [ ] **Step 3:** Update callers across the workspace. The pattern: anywhere code did `engine::CreatureType` (via re-export), it now does `engine_data::entities::CreatureType` directly.

```bash
git grep -E 'engine::(CreatureType|Capabilities|LanguageId|ChannelSet|CommunicationChannel|QuestCategory|Resolution)' \
    > /tmp/b1pp-task4-callers.txt
wc -l /tmp/b1pp-task4-callers.txt
```

For each entry, sed-rewrite to use the engine_data path. Surface area is small (~20-50 lines of callers).

- [ ] **Step 4:** Build + test.

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

If a workspace crate fails because it accessed `engine::CreatureType` and the import path no longer resolves, fix it by switching to `engine_data::entities::CreatureType`.

- [ ] **Step 5:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor(engine): drop engine_data re-exports of rule-level shape types"
```

---

### Task 5: Move `impl EventLike for Event` from engine to engine_data

**Goal:** Task 2 placed the emitted impl in `crates/engine/src/event/event_like_impl.rs` as a workaround. With Task 3+4 done, `engine_data` no longer has any reason to lack a regular `engine` dep — and engine no longer needs `engine_data` for IDs (Task 3) or shape types (Task 4). So `engine_data → engine` is now a clean single-direction dep, and the emit destination can move.

**Files:**
- Add: `engine = { path = "../engine" }` regular dep in `crates/engine_data/Cargo.toml`.
- Modify: `dsl_compiler` emit destination for `impl EventLike` → `crates/engine_data/src/events/mod.rs` (or wherever the Event enum is emitted).
- Delete: `crates/engine/src/event/event_like_impl.rs`.

- [ ] **Step 1:** Add engine dep to engine_data:

```toml
# crates/engine_data/Cargo.toml
[dependencies]
engine = { path = "../engine" }
serde = { version = "1", features = ["derive"] }
glam = "0.29"
smallvec = "1.13"
toml = "0.8"
```

- [ ] **Step 2:** Update the `dsl_compiler::emit_event_like_impl` (created by Task 2) emit destination. The xtask `--out-engine-event-like-impl` arg's default flips from `crates/engine/src/event/event_like_impl.rs` to be appended into `crates/engine_data/src/events/mod.rs` (or split into a sibling file `crates/engine_data/src/events/event_like.rs`).

- [ ] **Step 3:** Regen.

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
```

- [ ] **Step 4:** Verify the impl appears in engine_data:

```bash
grep -rE 'impl engine::event::EventLike for Event' crates/engine_data/src/
```

- [ ] **Step 5:** Delete the workaround file:

```bash
git rm crates/engine/src/event/event_like_impl.rs
sed -i '/^pub mod event_like_impl;$/d' crates/engine/src/event/mod.rs
```

- [ ] **Step 6:** Build + test.

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

- [ ] **Step 7:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor: move impl EventLike for Event from engine to engine_data (clean dep direction)"
```

---

### Task 6: Add `dsl_compiler::emit_sim_state` — emit `SimState` + `SimScratch` from DSL agent declarations

**Goal:** `SimState` becomes a compile-time artifact in engine_data, derived from DSL agent-field declarations. Engine never references `SimState` again.

**Files:**
- Create: `crates/dsl_compiler/src/emit_sim_state.rs`
- Create: `crates/dsl_compiler/src/emit_sim_scratch.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` — register modules + dispatch.
- Modify: `src/bin/xtask/cli/mod.rs` — add `out_sim_state` + `out_sim_scratch` arg defaults pointing at `crates/engine_data/src/`.
- Modify: `src/bin/xtask/compile_dsl_cmd.rs` — drive the new emits.

- [ ] **Step 1:** Read the existing `crates/engine/src/state/mod.rs` to inventory what `SimState` currently contains: `agent_alive`, `agent_hp`, `agent_position`, `agent_creature_type`, etc. + `tick`, RNG seed, spatial index handle, ability registry, `views: ViewRegistry` field.

- [ ] **Step 2:** Map each field to its DSL declaration. Engine-universal fields (tick, RNG seed) stay; rule-derived fields (creature-type, attack-range, etc.) come from DSL agent declarations.

- [ ] **Step 3:** Write `dsl_compiler/src/emit_sim_state.rs`:

```rust
use crate::ir::Compilation;
use std::io::{self, Write};

pub fn emit_sim_state<W: Write>(out: &mut W, comp: &Compilation) -> io::Result<()> {
    writeln!(out, "// GENERATED by dsl_compiler. Do not edit by hand.")?;
    writeln!(out, "use engine::{{SoaSlot, AgentId, EntityId, SpatialGrid}};")?;
    writeln!(out, "")?;
    writeln!(out, "pub struct SimState {{")?;
    writeln!(out, "    pub tick: u32,")?;
    writeln!(out, "    pub rng_seed: u64,")?;
    writeln!(out, "    pub agents_alive_count: usize,")?;
    for field in &comp.agent_fields {
        writeln!(out, "    pub {}: SoaSlot<{}>,", field.snake_name, field.rust_type)?;
    }
    writeln!(out, "    pub spatial: SpatialGrid<AgentId>,")?;
    writeln!(out, "    // ... other engine-managed primitives ...")?;
    writeln!(out, "}}")?;
    writeln!(out, "")?;
    writeln!(out, "impl SimState {{")?;
    writeln!(out, "    pub fn new(agent_cap: usize, seed: u64) -> Self {{")?;
    writeln!(out, "        Self {{")?;
    writeln!(out, "            tick: 0, rng_seed: seed, agents_alive_count: 0,")?;
    for field in &comp.agent_fields {
        writeln!(out, "            {}: SoaSlot::with_cap(agent_cap),", field.snake_name)?;
    }
    writeln!(out, "            spatial: SpatialGrid::new(),")?;
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    // Per-field accessors:
    for field in &comp.agent_fields {
        writeln!(out, "    pub fn agent_{}(&self, a: AgentId) -> {} {{ self.{}.get(a.raw() as usize) }}",
            field.snake_name, field.rust_type, field.snake_name)?;
        writeln!(out, "    pub fn set_agent_{}(&mut self, a: AgentId, v: {}) {{ self.{}.set(a.raw() as usize, v) }}",
            field.snake_name, field.rust_type, field.snake_name)?;
    }
    writeln!(out, "}}")?;
    Ok(())
}
```

(Adapt to the actual IR field names — `comp.agent_fields`, `field.snake_name`, etc. — read `dsl_compiler/src/ir.rs` to confirm.)

- [ ] **Step 4:** Write `emit_sim_scratch.rs` analogously. SimScratch contains `mask: MaskBuffer`, `target_mask: TargetMask`, `actions: SoaSlot<u8>`, etc.

- [ ] **Step 5:** Wire into `dsl_compiler/src/lib.rs` and `src/bin/xtask/cli/mod.rs`. Add `out_sim_state: PathBuf` + `out_sim_scratch: PathBuf` defaults at `crates/engine_data/src/sim_state.rs` + `sim_scratch.rs`.

- [ ] **Step 6:** Run regen.

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
```

Verify the files appear:

```bash
ls crates/engine_data/src/sim_state.rs crates/engine_data/src/sim_scratch.rs
head -30 crates/engine_data/src/sim_state.rs
```

- [ ] **Step 7:** Don't yet delete `crates/engine/src/state/mod.rs` — Task 7 cuts callers over.

- [ ] **Step 8:** Commit.

```bash
git add -A
git -c core.hooksPath= commit -m "feat(dsl_compiler): emit SimState + SimScratch into engine_data"
```

---

### Task 7: Cut callers from `engine::SimState` to `engine_data::SimState`

**Goal:** Every caller of `engine::state::SimState` and `engine::SimState` switches to `engine_data::sim_state::SimState`. Engine's `state/mod.rs` becomes deletable.

- [ ] **Step 1:** Inventory callers.

```bash
git grep -E 'engine::state::SimState|engine::SimState|crate::state::SimState' \
  -- ':!crates/engine/src/state' > /tmp/b1pp-task7-callers.txt
wc -l /tmp/b1pp-task7-callers.txt
```

- [ ] **Step 2:** Sed-update callers:

```bash
git grep -l 'engine::state::SimState\|engine::SimState' \
  -- ':!crates/engine/src/state' \
  | xargs sed -i 's|engine::state::SimState|engine_data::sim_state::SimState|g; s|engine::SimState|engine_data::sim_state::SimState|g'
```

- [ ] **Step 3:** Add `engine_data` to `Cargo.toml` `[dev-dependencies]` (or `[dependencies]`) on every crate that needs the new path. Most crates already depend on engine_data; verify.

- [ ] **Step 4:** Delete `crates/engine/src/state/`.

```bash
git rm -r crates/engine/src/state
sed -i '/^pub mod state;$/d' crates/engine/src/lib.rs
```

- [ ] **Step 5:** Build + test.

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

Expected: SUCCESS or fail on rule-aware code that referenced `state.views` (closed by Task 13). If failure is in `step.rs` / `mark_*_allowed` (deleted in Task 11/12), that's expected intermediate state.

- [ ] **Step 6:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor: move SimState callers to engine_data; delete engine/src/state/"
```

---

### Task 8: Move `engine/src/generated/{mask,physics,views}` → `engine_rules/src/`

**Goal:** Carries from old Plan B1' Task 3. Update emit destinations + emitted `crate::*` → `engine::*`/`engine_data::*` imports; regen; delete the old tree.

(Same steps as the old Plan B1' Task 3 — see prior version for detail. Compressed here:)

- [ ] **Step 1:** Update xtask `out_physics`, `out_mask`, `out_views` defaults.
- [ ] **Step 2:** Update `dsl_compiler::emit_{physics,mask,view}` to emit `use engine::*` and `use engine_data::sim_state::SimState; use engine_data::events::Event;` etc.
- [ ] **Step 3:** Update `crates/engine_rules/Cargo.toml` to dep both engine + engine_data.
- [ ] **Step 4:** Replace `engine_rules/src/lib.rs` with module declarations + `SimEventRing` / `SimCascadeRegistry` type aliases (see prior plan for body).
- [ ] **Step 5:** Regen.
- [ ] **Step 6:** Delete `engine/src/generated/` + drop `pub mod generated;`.
- [ ] **Step 7:** Sed callers `engine::generated::*` → `engine_rules::*`.
- [ ] **Step 8:** Workspace build + test.
- [ ] **Step 9:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor: emit physics/mask/views to engine_rules; delete engine/src/generated/"
```

---

### Task 9: Migrate chronicle.rs render templates into DSL + emit `engine_rules/src/chronicle.rs`

**Goal:** `engine/src/chronicle.rs` renders narrative text from `Event::ChronicleEntry` events. The template ID catalog + the renderer move to DSL declarations + emitted code.

**Pre-flight check:** does the DSL grammar already support template/string declarations? If not, this task adds a minimal grammar extension (`chronicle Foo { template_id: 1, text: "..." }`) before emit. Audit `crates/dsl_compiler/src/parser.rs`.

**Files:**
- Add (DSL): `assets/sim/chronicle.sim` declaring template constants + format strings.
- Add: `crates/dsl_compiler/src/{parser_chronicle,emit_chronicle}.rs` for the new grammar + emit.
- Create: `crates/engine_rules/src/chronicle.rs` (emitted).
- Delete: `crates/engine/src/chronicle.rs`.
- Modify: callers — `xtask chronicle_cmd.rs`, `engine/tests/wolves_and_humans_parity.rs`, etc. — switch from `engine::chronicle::*` to `engine_rules::chronicle::*`.

- [ ] **Step 1:** Audit DSL grammar: does it support string-literal declarations + format-string interpolation?

```bash
grep -nE 'template|chronicle|string' crates/dsl_compiler/src/parser.rs | head
```

If yes → skip to Step 3. If no → Step 2 adds minimal grammar.

- [ ] **Step 2:** **(Conditional)** Extend DSL grammar with a `chronicle` block. Minimal form:

```sim
chronicle AGENT_DIED {
    id: 1,
    template: "{agent} died.",
}
```

The format-string fields `{agent}` etc. resolve to `state.agent_*` lookups at emit time. If this grammar work is non-trivial, escalate — it might warrant its own sub-spec.

- [ ] **Step 3:** Write the chronicle catalog in `assets/sim/chronicle.sim` mirroring the constants in the current `engine/src/chronicle.rs::templates` module.

- [ ] **Step 4:** Write `dsl_compiler/src/emit_chronicle.rs` — emits `engine_rules/src/chronicle.rs` containing `pub fn render_entry(state: &SimState, entry: &Event) -> String { ... }` with a match over template IDs. Each arm formats per the DSL template strings.

- [ ] **Step 5:** Add `out_chronicle: PathBuf` arg in xtask cli/mod.rs default `crates/engine_rules/src/chronicle.rs`. Wire emit pass.

- [ ] **Step 6:** Regen.

- [ ] **Step 7:** Delete `crates/engine/src/chronicle.rs` and update callers:

```bash
git rm crates/engine/src/chronicle.rs
sed -i '/^pub mod chronicle;$/d' crates/engine/src/lib.rs
git grep -l 'engine::chronicle' | xargs sed -i 's|engine::chronicle|engine_rules::chronicle|g'
```

- [ ] **Step 8:** Build + test.

- [ ] **Step 9:** Commit.

```bash
git -c core.hooksPath= commit -am "feat: migrate chronicle to DSL-emitted engine_rules::chronicle"
```

---

### Task 10: Migrate `engagement.rs` `break_reason` constants → emitted `engine_data/src/engagement.rs`

**Goal:** The `break_reason` u8 constants (SWITCH=0, OUT_OF_RANGE=1, PARTNER_DIED=2) live in DSL events declarations + emit into engine_data. The hand-written file disappears.

- [ ] **Step 1:** Add the constants to `assets/sim/events.sim` alongside the engagement events:

```sim
event EngagementBroken {
    actor: AgentId,
    reason: u8,
    tick: u32,
}
event_constants EngagementBroken::break_reason {
    SWITCH = 0,
    OUT_OF_RANGE = 1,
    PARTNER_DIED = 2,
}
```

(Adapt to actual DSL grammar — may be a small grammar extension if `event_constants` doesn't exist.)

- [ ] **Step 2:** Update `dsl_compiler` to emit these into `engine_data/src/engagement.rs` (or alongside the events in `engine_data/src/events/`).

- [ ] **Step 3:** Regen + verify the constants appear in engine_data.

- [ ] **Step 4:** Delete `crates/engine/src/engagement.rs`. Update callers:

```bash
git rm crates/engine/src/engagement.rs
sed -i '/^pub mod engagement;$/d' crates/engine/src/lib.rs
git grep -l 'engine::engagement' | xargs sed -i 's|engine::engagement|engine_data::engagement|g'
```

- [ ] **Step 5:** Build + test.

- [ ] **Step 6:** Commit.

```bash
git -c core.hooksPath= commit -am "feat: migrate engagement break_reason constants to engine_data emitted"
```

---

### Task 11: Add `dsl_compiler::emit_step` + `emit_backend` + `emit_mask_fill` + `emit_cascade_register`

**Goal:** Compiler emits the four phase-orchestration files into engine_rules. Carries from old Plan B1' Task 4.

**Files:** new `dsl_compiler/src/{emit_step,emit_backend,emit_mask_fill,emit_cascade_register}.rs`. Wire into `lib.rs` + xtask.

- [ ] **Step 1:** Inventory the current `engine/src/step.rs` body for the phase ordering template (view-fold → mask-fill → policy/scoring → action-select → cascade-dispatch → tick-end).

- [ ] **Step 2:** Write `emit_step.rs` walking the IR + emitting literal calls per phase. Body shape:

```rust
pub fn emit_step<W: Write>(out: &mut W, comp: &Compilation) -> io::Result<()> {
    writeln!(out, "// GENERATED by dsl_compiler. Do not edit by hand.")?;
    writeln!(out, "use engine::cascade::CascadeRegistry;")?;
    writeln!(out, "use engine::event::EventRing;")?;
    writeln!(out, "use engine::policy::PolicyBackend;")?;
    writeln!(out, "use engine_data::sim_state::{{SimState, SimScratch}};")?;
    writeln!(out, "use engine_data::events::Event;")?;
    writeln!(out, "use crate::ViewRegistry;")?;
    writeln!(out, "")?;
    writeln!(out, "pub fn step<B: PolicyBackend>(state: &mut SimState, scratch: &mut SimScratch, events: &mut EventRing<Event>, views: &mut ViewRegistry, policy: &B, cascade: &CascadeRegistry<Event>) {{")?;
    writeln!(out, "    let events_before = events.push_count();")?;
    writeln!(out, "    state.tick = state.tick.wrapping_add(1);")?;
    writeln!(out, "    // Phase 1: view fold")?;
    writeln!(out, "    views.fold_since(events, events_before);")?;
    writeln!(out, "    // Phase 2: mask fill")?;
    writeln!(out, "    crate::mask_fill::fill_all(&mut scratch.mask, &mut scratch.target_mask, state);")?;
    writeln!(out, "    // Phase 3: scoring")?;
    writeln!(out, "    crate::scoring::evaluate_all(state, scratch, &views, policy);")?;
    writeln!(out, "    // Phase 4: action selection — handled inline by scoring")?;
    writeln!(out, "    // Phase 5: cascade dispatch")?;
    writeln!(out, "    crate::action_emit::emit_root_events(state, scratch, events);")?;
    writeln!(out, "    cascade.run_fixed_point(events, state);")?;
    writeln!(out, "    // Phase 6: tick end")?;
    writeln!(out, "    state.snapshot_tick_end_if_due();")?;
    writeln!(out, "}}")?;
    Ok(())
}
```

(Adapt method names to the actual existing `engine/src/step.rs` body.)

- [ ] **Step 3:** Write `emit_backend.rs` — emits `engine_rules/src/backend.rs` with `pub struct SerialBackend; impl SimBackend for SerialBackend { ... }` calling `crate::step::step`.

- [ ] **Step 4:** Write `emit_mask_fill.rs` — emits `engine_rules/src/mask_fill.rs` with `pub fn fill_all(buf, targets, state)` calling each emitted mask predicate (one literal call per mask declaration).

- [ ] **Step 5:** Write `emit_cascade_register.rs` — emits `engine_rules/src/cascade.rs` with `pub fn with_engine_builtins() -> CascadeRegistry<Event> { let mut reg = CascadeRegistry::new(); reg.register(DamageHandler); ... reg }`.

- [ ] **Step 6:** Wire into `dsl_compiler/src/lib.rs` + xtask defaults.

- [ ] **Step 7:** Add `Self::Event` associated type to `SimBackend` trait in `engine/src/backend.rs`. Delete the existing `pub struct CpuBackend` + impl from engine.

```rust
pub trait SimBackend {
    type Views;
    type Event: crate::event::EventLike;
    fn step<B: PolicyBackend>(
        &mut self,
        state:   &mut engine_data::sim_state::SimState,  // — OR generic over S; settle in this step
        scratch: &mut engine_data::sim_state::SimScratch,
        events:  &mut crate::event::EventRing<Self::Event>,
        views:   &mut Self::Views,
        policy:  &B,
        cascade: &crate::cascade::CascadeRegistry<Self::Event>,
    );
}
```

(Note: this re-introduces engine→engine_data dep for the trait signature. If we want engine truly zero-dep on engine_data, the trait must be generic over `S: SimulationStateLike` too — but that adds machinery. For B1', accept the engine→engine_data dep at the trait declaration; it's narrow. Or alternatively drop `SimBackend` entirely and let `engine_rules::SerialBackend` and `engine_gpu::GpuBackend` be free types. Decide here. Recommendation: drop the trait — backends are concrete types, callers pick one at compile time.)

- [ ] **Step 8:** Regen. Verify the four new files exist with `// GENERATED` headers.

- [ ] **Step 9:** Delete `engine/src/step.rs` + `engine/src/backend.rs`'s `CpuBackend`. If keeping `SimBackend` trait, leave `backend.rs` with just the trait declaration; if dropping, delete `backend.rs` entirely.

- [ ] **Step 10:** Update callers from `engine::step::step` / `CpuBackend` → `engine_rules::step::step` / `engine_rules::SerialBackend`.

- [ ] **Step 11:** Build + test.

- [ ] **Step 12:** Commit.

```bash
git -c core.hooksPath= commit -am "feat(dsl_compiler): emit step + backend + mask_fill + cascade-register into engine_rules"
```

---

### Task 12: Move `mark_*_allowed` orchestration to engine_rules emitted

**Goal:** Already covered by Task 11's `emit_mask_fill`. This task is the cleanup: delete `mark_*_allowed` methods from `engine/src/mask.rs`. `MaskBuffer` storage + raw bit ops stay.

- [ ] **Step 1:** In `engine/src/mask.rs`, hand-delete every `pub fn mark_*_allowed` method. Keep `MaskBuffer::new`, `reset`, `set`, `get`, `mark_self_predicate` (the primitive-shape mark op).

- [ ] **Step 2:** Drop the `use engine_rules::mask::{...}` import (those imports were the rule-aware predicates pulled in by the deleted methods).

- [ ] **Step 3:** Build + test.

- [ ] **Step 4:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor(engine): drop mark_*_allowed methods from MaskBuffer (now emitted)"
```

---

### Task 13: Drop `views: ViewRegistry` field from SimState

**Goal:** SimState (now in engine_data) cannot reference `engine_rules::ViewRegistry` (engine_data sits below engine_rules). The field is removed; `ViewRegistry` is constructed by callers + threaded through `step` as a parameter.

- [ ] **Step 1:** Modify the emitted `SimState` struct (Task 6's `emit_sim_state`) to NOT include the `views` field. The emitter's IR drives this — the agent_fields list excludes `views`.

- [ ] **Step 2:** Update emitted `step` (Task 11) to take `views: &mut ViewRegistry` as a separate parameter (already does this per the signature in Task 11 Step 2).

- [ ] **Step 3:** Update callers: every place that constructs `SimState` no longer initializes `state.views`. Each caller separately constructs `let mut views = engine_rules::ViewRegistry::new();` and threads it through.

- [ ] **Step 4:** Regen + sed-update callers.

- [ ] **Step 5:** Build + test.

- [ ] **Step 6:** Commit.

```bash
git -c core.hooksPath= commit -am "refactor: drop views field from SimState; thread ViewRegistry as parameter"
```

---

### Task 14: Seal `CascadeHandler` + view traits via `__sealed::Sealed` + `GeneratedRule` marker

**Goal:** Add `pub trait GeneratedRule {}` to `engine_rules/src/lib.rs` + four blanket `Sealed` impls. Update emitters to write `impl crate::GeneratedRule for X {}` next to every emitted trait impl. Add `#[cfg(test)]` direct seals for engine's demo impls + tests.

(Carries from old Plan B1' Task 6 — see prior version for full detail.)

- [ ] **Step 1:** Update `engine_rules/src/lib.rs`:

```rust
#![allow(clippy::all)]

pub mod backend;
pub mod cascade;
pub mod chronicle;
pub mod mask;
pub mod mask_fill;
pub mod physics;
pub mod step;
pub mod views;

pub use views::ViewRegistry;
pub use backend::SerialBackend;
pub use cascade::with_engine_builtins;

pub type SimEventRing = engine::event::EventRing<engine_data::events::Event>;
pub type SimCascadeRegistry = engine::cascade::CascadeRegistry<engine_data::events::Event>;

#[doc(hidden)]
pub trait GeneratedRule {}

impl<T: GeneratedRule> engine::cascade::handler::__sealed::Sealed for T {}
```

- [ ] **Step 2:** Update `dsl_compiler/src/{emit_physics,emit_view}.rs` to write `impl crate::GeneratedRule for X {}` after every emitted trait impl. (See prior plan for code sketch.)

- [ ] **Step 3:** Add `#[cfg(test)]` direct seal impls in `crates/engine/src/view/{materialized,lazy,topk}.rs` for the demo types (`DamageTaken`, `NearestEnemyLazy`, `MostHostileTopK`):

```rust
#[cfg(test)]
impl crate::cascade::handler::__sealed::Sealed for DamageTaken {}
```

- [ ] **Step 4:** Add `engine_rules` to engine's `[dev-dependencies]`. Update test handlers in `crates/engine/tests/cascade_*.rs` to add `impl engine_rules::GeneratedRule for {Test} {}`.

- [ ] **Step 5:** Regen + build + test.

- [ ] **Step 6:** Commit.

```bash
git -c core.hooksPath= commit -am "feat: seal CascadeHandler + view traits via GeneratedRule blanket impl"
```

---

### Task 15: trybuild compile-fail test for the seal

(Carries unchanged from old Plan B1' Task 7. See prior version.)

- [ ] **Step 1:** Add `trybuild = "1"` to engine's `[dev-dependencies]`.
- [ ] **Step 2:** Write `crates/engine/tests/sealed_cascade_handler.rs` driver.
- [ ] **Step 3:** Write `crates/engine/tests/ui/external_impl_rejected.rs` fixture (concrete `impl CascadeHandler<Event>` for an external type that lacks Sealed).
- [ ] **Step 4:** `TRYBUILD=overwrite` to populate stderr.
- [ ] **Step 5:** Run normally; verify PASS.
- [ ] **Step 6:** Commit.

---

### Task 16: `engine_rules/build.rs` + `engine_data/build.rs` sentinels

(Carries unchanged from old Plan B1' Task 8.)

- [ ] **Step 1:** Write `engine_rules/build.rs` — every non-lib.rs file requires `// GENERATED` header.
- [ ] **Step 2:** Same for `engine_data/build.rs`.
- [ ] **Step 3:** Add `build = "build.rs"` to both Cargo.tomls.
- [ ] **Step 4:** Clean rebuild + negative test (inject hand-edited file, confirm panic).
- [ ] **Step 5:** Commit.

---

### Task 17: `engine/build.rs` primitives-only allowlist (STRICT — no exceptions)

**Goal:** Engine's allowlist post-plan has zero exceptions. chronicle.rs, engagement.rs, event_like_impl.rs are gone. Allowlist enumerates pure primitives.

⚠️ **Allowlist gate (Spec B' D11).** Both biased-against critics (compiler-first + allowlist-gate) must return PASS. AIS preamble + this commit-message are the writeup.

- [ ] **Step 1:** Confirm engine/src/ post-Tasks 7+9+10+12 is exactly:
  - top-level: `lib.rs`, `backend.rs`, `channel.rs`, `creature.rs`, `ids.rs`, `mask.rs`, `pool.rs`, `rng.rs`, `schema_hash.rs`, `spatial.rs`, `terrain.rs`, `trajectory.rs`. (NO `step.rs`, NO `chronicle.rs`, NO `engagement.rs`, NO `state/`, NO `event_like_impl.rs`.)
  - subdirs: `ability/`, `aggregate/`, `cascade/`, `event/`, `invariant/`, `obs/`, `policy/`, `pool/`, `probe/`, `snapshot/`, `telemetry/`, `view/`. (NO `state/`, NO `generated/`.)

```bash
ls crates/engine/src/
```

If anything unexpected is present, escalate.

- [ ] **Step 2:** Write `crates/engine/build.rs` (full allowlist body — see old Plan B1' Task 9 for shape; remove the `chronicle.rs` + `engagement.rs` entries):

```rust
const ALLOWED_TOP_LEVEL: &[&str] = &[
    "lib.rs", "backend.rs", "channel.rs", "creature.rs",
    "ids.rs", "mask.rs", "pool.rs", "rng.rs",
    "schema_hash.rs", "spatial.rs", "terrain.rs", "trajectory.rs",
];

const ALLOWED_DIRS: &[&str] = &[
    "ability", "aggregate", "cascade", "event", "invariant",
    "obs", "policy", "pool", "probe", "snapshot", "telemetry", "view",
];
```

Plus the reject-`// GENERATED` walker (carries from old plan).

- [ ] **Step 3:** Negative tests (allowlist + GENERATED rejection both fire).

- [ ] **Step 4:** Commit (subject to allowlist-gate critic).

```bash
git -c core.hooksPath= commit -am "feat(engine): primitives-only build.rs allowlist; no exceptions (Spec B' §5)"
```

---

### Task 18: xtask `compile-dsl --check`

(Carries unchanged from old Plan B1' Task 10/11. Add the new emit-target diff entries — sim_state.rs, sim_scratch.rs, step.rs, backend.rs, mask_fill.rs, cascade.rs, chronicle.rs, engagement.rs.)

- [ ] **Step 1:** Add `--check` flag to `CompileDslArgs`.
- [ ] **Step 2:** Implement `run_compile_dsl_check` — regens to tempdir, diffs against working tree, reports drift.
- [ ] **Step 3:** Add new emit destinations to the diff pairs table.
- [ ] **Step 4:** Negative test (inject drift, confirm exit 1).
- [ ] **Step 5:** Commit.

---

### Task 19: `.githooks/pre-commit` — header rule + regen-on-DSL-change

(Carries unchanged from old Plan B1' Task 11/12.)

- [ ] **Steps:** see prior plan. Smoke-test header rule + inverse rule + regen-on-DSL-change.
- [ ] **Step N:** Commit.

---

### Task 20: ast-grep CI rules + stale-content + schema-hash CI

(Carries unchanged from old Plan B1' Task 12+13.)

- [ ] Four ast-grep rules restricting `impl CascadeHandler` / `impl MaterializedView` / `impl LazyView` / `impl TopKView` to `crates/engine_rules/src/` (with cfg-test exclusions for engine's tests + view demo files).
- [ ] CI step: regen + assert no diff.
- [ ] CI step: schema-hash freshness.
- [ ] Commit.

---

### Task 21: Final verification + AIS tick

- [ ] **Step 1:** `cargo clean && cargo build --workspace && cargo test --workspace`.
- [ ] **Step 2:** `cargo run --bin xtask -- compile-dsl --check` clean.
- [ ] **Step 3:** Pre-commit clean on no-op stage.
- [ ] **Step 4:** trybuild test passes.
- [ ] **Step 5:** Audit:
  - `grep -rE "// GENERATED" crates/engine/src/` → empty
  - `find crates/engine_rules/src crates/engine_data/src -name '*.rs' -not -name 'lib.rs' | xargs -I{} sh -c 'head -5 {} | grep -q "// GENERATED" || echo "MISSING: {}"'` → empty
  - `git grep 'crate::generated\|engine::generated'` → empty
  - `git grep 'engine::SimState\|engine::state::SimState'` → empty (all callers use engine_data)
  - `git grep 'engine::chronicle\|engine::engagement'` → empty
- [ ] **Step 6:** Tick AIS post-design checkbox in plan file.
- [ ] **Step 7:** Final commit.

---

## Sequencing summary

| Task | Title | Status | Depends on |
|---|---|---|---|
| 1 | Generic primitives in engine | ✓ LANDED `e60619c6` | — |
| 2 | Emit `impl EventLike for Event` (workaround) | ✓ LANDED `33e73ceb` | 1 |
| 3 | Lift typed-ID newtypes into engine | active | — |
| 4 | Drop engine→engine_data shape-type re-exports | active | 3 |
| 5 | Move `impl EventLike` from engine to engine_data | active | 3, 4 |
| 6 | Emit SimState + SimScratch | active | 5 |
| 7 | Cut SimState callers to engine_data | active | 6 |
| 8 | Move generated/{mask,physics,views} → engine_rules | active | 5 |
| 9 | Migrate chronicle to DSL + emit | active | 5, 8 |
| 10 | Migrate engagement constants | active | 5 |
| 11 | Emit step + backend + mask_fill + cascade | active | 7, 8 |
| 12 | Drop mark_*_allowed from engine/src/mask.rs | active | 11 |
| 13 | Drop views field from SimState | active | 6, 11 |
| 14 | Seal traits + emit GeneratedRule markers | active | 8, 11 |
| 15 | trybuild compile-fail test | active | 14 |
| 16 | engine_rules + engine_data build.rs sentinels | active | 14 |
| 17 | engine/build.rs strict allowlist (gated by allowlist-gate critic) | active | 7, 9, 10, 11 |
| 18 | xtask compile-dsl --check | active | 11 |
| 19 | Pre-commit header rule + regen-on-DSL-change | active | 18 |
| 20 | ast-grep + stale-content + schema-hash CI | active | 14 |
| 21 | Final verification + AIS tick | active | all |

Strict ordering for the early tasks (3 → 4 → 5 → 6 → 7 → 8 …); later tasks (15, 18, 19, 20) can interleave once their deps land.

## Coordination with already-landed Spec D-amendment hooks

Same as the old Plan B1' — PreToolUse + Stop + pre-commit gates fire throughout. Task 17 specifically triggers `critic-allowlist-gate`; both biased-against critics must PASS or stop and discuss.

## What B2 + B3 look like after this plan

- **B2 (chronicle + engagement migration):** **subsumed into B1'**. No separate B2 needed.
- **B3 (legacy `src/` sweep + xtask move):** unchanged, still queued. The plan at `2026-04-25-legacy-src-sweep-impl.md` runs after B1' lands (or in parallel via separate worktree).
- **B4–B8 (per-rule-set optimizations):** future. See Spec B' §12. None are required for B1' to land.
